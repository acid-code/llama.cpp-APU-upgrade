// SPDX-License-Identifier: MIT
//
// ggml-hip-ssd.cpp - V1 Basic Zero-Copy GTT Memory Arena for AMD Ryzen APUs
// Allocates 4096-byte page-aligned coherent GTT memory via hipHostMallocMapped
// and resolves GPU virtual alias pointers via hipHostGetDevicePointer.

#ifndef GGML_USE_HIP
#define GGML_USE_HIP
#endif

#include "ggml-hip-ssd.h"

#if defined(__has_include)
#  if __has_include(<hip/hip_runtime.h>)
#    include <hip/hip_runtime.h>
#    include <hip/hip_runtime_api.h>
#    define GGML_HIP_SSD_HAS_HIP_HEADERS 1
#  endif
#endif

#ifndef GGML_HIP_SSD_HAS_HIP_HEADERS
using hipError_t = int;
constexpr hipError_t hipSuccess = 0;
constexpr unsigned int hipHostMallocDefault  = 0x0;
constexpr unsigned int hipHostMallocPortable = 0x1;
constexpr unsigned int hipHostMallocMapped   = 0x2;
constexpr unsigned int hipHostMallocCoherent = 0x40000000;
struct hipDeviceProp_t {
    int integrated;
};
extern "C" {
hipError_t hipSetDevice(int device);
hipError_t hipGetDeviceProperties(hipDeviceProp_t * prop, int device);
hipError_t hipHostMalloc(void ** ptr, size_t size, unsigned int flags);
hipError_t hipHostFree(void * ptr);
hipError_t hipHostGetDevicePointer(void ** dev_ptr, void * host_ptr, unsigned int flags);
hipError_t hipGetLastError(void);
}
#endif

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <unordered_map>

namespace ggml_hip_ssd {

struct allocation_entry {
    void * raw_host     = nullptr;
    void * aligned_host = nullptr;
    void * aligned_dev  = nullptr;
    size_t user_bytes   = 0;
    size_t raw_bytes    = 0;
};

struct gtt_zero_copy_arena::impl {
    mutable std::mutex mutex;
    int                device_id        = 0;
    bool               initialized      = false;
    bool               unified_apu      = false;
    size_t             allocated_bytes  = 0;
    size_t             peak_bytes       = 0;
    uint64_t           zero_copy_allocs = 0;
    mutable std::atomic<uint64_t> zero_copy_hits{0};
    std::unordered_map<const void *, allocation_entry> table;

    ~impl() noexcept {
        shutdown();
    }

    bool init(int dev) noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        if (initialized) {
            return unified_apu;
        }
        device_id = dev >= 0 ? dev : 0;
        if (hipSetDevice(device_id) != hipSuccess) {
            (void) hipGetLastError();
            return false;
        }
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, device_id) == hipSuccess) {
            unified_apu = (prop.integrated != 0);
        } else {
            (void) hipGetLastError();
            unified_apu = true;
        }
        const char * force_env = std::getenv("GGML_HIP_SSD_FORCE_GTT");
        if (force_env && *force_env && std::atoi(force_env) != 0) {
            unified_apu = true;
        }
        initialized = true;
        std::fprintf(stderr,
            "[ggml-hip-ssd-v1] status=initialized device=%d unified_apu=%d page_align=%zu\n",
            device_id, int(unified_apu), kGttPageAlignment);
        return unified_apu;
    }

    void * alloc_zero_copy(size_t bytes) noexcept {
        if (bytes == 0 || bytes > SIZE_MAX - kGttPageAlignment) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            (void) hipSetDevice(device_id);
            unified_apu = true;
            initialized = true;
        }

        const size_t raw_bytes = bytes + kGttPageAlignment;
        const unsigned int flags = unified_apu
            ? (hipHostMallocMapped | hipHostMallocCoherent | hipHostMallocPortable)
            : hipHostMallocDefault;

        void * raw_host = nullptr;
        if (hipHostMalloc(&raw_host, raw_bytes, flags) != hipSuccess || !raw_host) {
            (void) hipGetLastError();
            return nullptr;
        }

        void * raw_dev = raw_host;
        if (unified_apu) {
            if (hipHostGetDevicePointer(&raw_dev, raw_host, 0) != hipSuccess || !raw_dev) {
                (void) hipGetLastError();
                raw_dev = raw_host;
            }
        }

        const uintptr_t addr = reinterpret_cast<uintptr_t>(raw_host);
        const size_t pad = (kGttPageAlignment - (addr % kGttPageAlignment)) % kGttPageAlignment;
        char * aligned_host = static_cast<char *>(raw_host) + pad;
        char * aligned_dev  = static_cast<char *>(raw_dev) + pad;

        allocation_entry entry{};
        entry.raw_host     = raw_host;
        entry.aligned_host = aligned_host;
        entry.aligned_dev  = aligned_dev;
        entry.user_bytes   = bytes;
        entry.raw_bytes    = raw_bytes;

        try {
            table[aligned_host] = entry;
        } catch (...) {
            (void) hipHostFree(raw_host);
            return nullptr;
        }

        allocated_bytes += bytes;
        peak_bytes = std::max(peak_bytes, allocated_bytes);
        ++zero_copy_allocs;
        return aligned_host;
    }

    bool free_zero_copy(void * host_ptr) noexcept {
        if (!host_ptr) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex);
        auto it = table.find(host_ptr);
        if (it == table.end()) {
            return false;
        }
        void * raw_host = it->second.raw_host;
        const size_t bytes = it->second.user_bytes;
        table.erase(it);
        if (allocated_bytes >= bytes) {
            allocated_bytes -= bytes;
        } else {
            allocated_bytes = 0;
        }
        if (raw_host) {
            (void) hipHostFree(raw_host);
        }
        return true;
    }

    void * resolve_device_ptr(void * host_ptr) const noexcept {
        if (!host_ptr) {
            return nullptr;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto it = table.find(host_ptr);
            if (it != table.end()) {
                zero_copy_hits.fetch_add(1, std::memory_order_relaxed);
                return it->second.aligned_dev;
            }
        }
        if (unified_apu) {
            void * dev_ptr = nullptr;
            if (hipHostGetDevicePointer(&dev_ptr, host_ptr, 0) == hipSuccess && dev_ptr) {
                zero_copy_hits.fetch_add(1, std::memory_order_relaxed);
                return dev_ptr;
            }
            (void) hipGetLastError();
        }
        return host_ptr;
    }

    void shutdown() noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized && table.empty()) {
            return;
        }
        // Note: Do not call hipHostFree inside the static destructor at process exit(),
        // as libhsa-runtime64 may already be tearing down KFD MES queues. All active ggml
        // host buffers are freed via free_zero_copy() during llama_free() before exit().
        table.clear();
        std::fprintf(stderr,
            "[ggml-hip-ssd-v1] status=shutdown allocs=%llu hits=%llu peak_mib=%.2f\n",
            static_cast<unsigned long long>(zero_copy_allocs),
            static_cast<unsigned long long>(zero_copy_hits.load(std::memory_order_relaxed)),
            double(peak_gtt_bytes_v1()) / (1024.0 * 1024.0));
        allocated_bytes = 0;
        initialized = false;
    }
    size_t peak_gtt_bytes_v1() const noexcept { return peak_bytes; }
};

gtt_zero_copy_arena::gtt_zero_copy_arena() noexcept
    : impl_(new (std::nothrow) impl()) {}

gtt_zero_copy_arena::~gtt_zero_copy_arena() noexcept {
    delete impl_;
    impl_ = nullptr;
}

// cppcheck-suppress unusedFunction
gtt_zero_copy_arena & gtt_zero_copy_arena::instance() noexcept {
    static gtt_zero_copy_arena inst;
    return inst;
}

// cppcheck-suppress unusedFunction
bool gtt_zero_copy_arena::init(int device_id) noexcept {
    return impl_ ? impl_->init(device_id) : false;
}

// cppcheck-suppress unusedFunction
void gtt_zero_copy_arena::shutdown() noexcept {
    if (impl_) {
        impl_->shutdown();
    }
}

// cppcheck-suppress unusedFunction
void * gtt_zero_copy_arena::alloc_zero_copy(size_t bytes) noexcept {
    return impl_ ? impl_->alloc_zero_copy(bytes) : nullptr;
}

// cppcheck-suppress unusedFunction
bool gtt_zero_copy_arena::free_zero_copy(void * host_ptr) noexcept {
    return impl_ ? impl_->free_zero_copy(host_ptr) : false;
}

void * gtt_zero_copy_arena::resolve_device_ptr(void * host_ptr) const noexcept {
    return impl_ ? impl_->resolve_device_ptr(host_ptr) : host_ptr;
}

// cppcheck-suppress unusedFunction
const void * gtt_zero_copy_arena::resolve_device_ptr(const void * host_ptr) const noexcept {
    return resolve_device_ptr(const_cast<void *>(host_ptr));
}

// cppcheck-suppress unusedFunction
bool gtt_zero_copy_arena::is_unified_apu() const noexcept {
    if (!impl_) {
        return false;
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->unified_apu;
}

// cppcheck-suppress unusedFunction
gtt_arena_stats gtt_zero_copy_arena::snapshot() const noexcept {
    gtt_arena_stats out{};
    if (!impl_) {
        return out;
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    out.unified_apu      = impl_->unified_apu;
    out.allocated_bytes  = impl_->allocated_bytes;
    out.peak_bytes       = impl_->peak_bytes;
    out.zero_copy_allocs = impl_->zero_copy_allocs;
    out.zero_copy_hits   = impl_->zero_copy_hits.load(std::memory_order_relaxed);
    return out;
}

} // namespace ggml_hip_ssd
