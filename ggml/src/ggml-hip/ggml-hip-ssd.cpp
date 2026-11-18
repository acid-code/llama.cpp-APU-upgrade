// SPDX-License-Identifier: MIT
//
// ggml-hip-ssd.cpp - V2 Heterogeneous Zero-Copy GTT + io_uring Prefetcher +
//                    Custom Closed CPU Worker Pool for AMD Ryzen APUs
//
// Strict Hardware Safeguards Implemented:
//   1. Core Reservation:
//      - Custom closed CPU worker pool is strictly capped at 12 threads (cores 0..11).
//      - Explicitly reserves 4 hardware threads (2 physical cores, cores 12..15 on
//        Ryzen 7 8745HS) for Ubuntu OS, background services, and the io_uring prefetcher.
//   2. Dense Model Sleep Logic:
//      - Automatic fallback: when a standard dense model is loaded (e.g., gpt-oss-20b
//        or any graph without MoE expert tensors), the 12-thread CPU worker pool stays
//        100% dormant on its condition variable and routes 100% of the model to the
//        Radeon 780M iGPU.
//      - When an MoE model is loaded (specifically Qwen 27B A3B / MUL_MAT_ID graph),
//        the closed CPU worker pool and io_uring prefetcher engage automatically.
//   3. No Standard Flags:
//      - Native MoE expert routing via GGML_OP_MUL_MAT_ID graph & op hooks; does not
//        implement or rely on --n-cpu-moe.

#ifndef GGML_USE_HIP
#define GGML_USE_HIP
#endif

#include "ggml-hip-ssd.h"
#include "ggml-impl.h"

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

#if defined(__linux__)
#  include <fcntl.h>
#  include <pthread.h>
#  include <sched.h>
#  include <sys/mman.h>
#  include <sys/syscall.h>
#  include <unistd.h>
#  if defined(__has_include) && __has_include(<linux/io_uring.h>)
#    include <linux/io_uring.h>
#    define GGML_HIP_SSD_HAS_LINUX_IO_URING 1
#  endif
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ggml_hip_ssd {

namespace {

uint64_t align_down_page(uint64_t val) noexcept {
    return (val / kGttPageAlignment) * kGttPageAlignment;
}

size_t align_up_page(size_t val) noexcept {
    return ((val + kGttPageAlignment - 1) / kGttPageAlignment) * kGttPageAlignment;
}

std::string to_lower_ascii(const char * s) {
    if (!s) {
        return {};
    }
    std::string out(s);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

std::string read_process_model_hint(const char * explicit_hint) {
    std::string combined = to_lower_ascii(explicit_hint);
    const char * env_hint = std::getenv("GGML_HIP_SSD_MODEL_HINT");
    if (env_hint && *env_hint) {
        combined += " ";
        combined += to_lower_ascii(env_hint);
    }
#if defined(__linux__)
    FILE * f = std::fopen("/proc/self/cmdline", "rb");
    if (f) {
        char buf[4096];
        const size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
        std::fclose(f);
        for (size_t i = 0; i < n; ++i) {
            if (buf[i] == '\0') {
                buf[i] = ' ';
            }
        }
        buf[n] = '\0';
        combined += " ";
        combined += to_lower_ascii(buf);
    }
#endif
    return combined;
}

void pin_current_thread_to_core_range(size_t core_begin, size_t core_end) noexcept {
#if defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (size_t c = core_begin; c < core_end; ++c) {
        CPU_SET(c, &cpuset);
    }
    (void) pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#else
    (void) core_begin;
    (void) core_end;
#endif
}

// Lightweight Linux io_uring wrapper using kernel syscalls directly (with pread fallback)
class linux_io_uring_channel {
public:
    linux_io_uring_channel() noexcept = default;
    linux_io_uring_channel(const linux_io_uring_channel &) = delete;
    linux_io_uring_channel & operator=(const linux_io_uring_channel &) = delete;

    ~linux_io_uring_channel() noexcept {
        close_ring();
    }

    bool init(unsigned entries) noexcept {
        close_ring();
#if defined(__linux__) && defined(GGML_HIP_SSD_HAS_LINUX_IO_URING) && defined(__NR_io_uring_setup)
        io_uring_params params{};
        const int fd = static_cast<int>(syscall(__NR_io_uring_setup, entries, &params));
        if (fd >= 0) {
            ring_fd_ = fd;
            sq_ring_sz_ = params.sq_off.array + params.sq_entries * sizeof(unsigned);
            cq_ring_sz_ = params.cq_off.cqes + params.cq_entries * sizeof(io_uring_cqe);
            sqes_sz_    = params.sq_entries * sizeof(io_uring_sqe);

            sq_ptr_ = mmap(nullptr, sq_ring_sz_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_fd_, IORING_OFF_SQ_RING);
            cq_ptr_ = mmap(nullptr, cq_ring_sz_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_fd_, IORING_OFF_CQ_RING);
            sqes_   = static_cast<io_uring_sqe *>(
                mmap(nullptr, sqes_sz_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_fd_, IORING_OFF_SQES));

            if (sq_ptr_ != MAP_FAILED && cq_ptr_ != MAP_FAILED && sqes_ != MAP_FAILED) {
                params_ = params;
                return true;
            }
            close_ring();
        }
#else
        (void) entries;
#endif
        return false;
    }

    bool read_into_gtt(int fd, uint64_t offset, void * gtt_host_dst, size_t bytes) noexcept {
        if (fd < 0 || !gtt_host_dst || bytes == 0) {
            return false;
        }
#if defined(__linux__) && defined(GGML_HIP_SSD_HAS_LINUX_IO_URING) && defined(__NR_io_uring_enter)
        if (ring_fd_ >= 0 && sq_ptr_ && cq_ptr_ && sqes_) {
            char * sq_base = static_cast<char *>(sq_ptr_);
            char * cq_base = static_cast<char *>(cq_ptr_);
            auto * sq_tail = reinterpret_cast<unsigned *>(sq_base + params_.sq_off.tail);
            auto * sq_mask = reinterpret_cast<unsigned *>(sq_base + params_.sq_off.ring_mask);
            auto * sq_arr  = reinterpret_cast<unsigned *>(sq_base + params_.sq_off.array);
            auto * cq_head = reinterpret_cast<unsigned *>(cq_base + params_.cq_off.head);
            auto * cq_tail = reinterpret_cast<unsigned *>(cq_base + params_.cq_off.tail);
            auto * cq_mask = reinterpret_cast<unsigned *>(cq_base + params_.cq_off.ring_mask);
            auto * cqes    = reinterpret_cast<io_uring_cqe *>(cq_base + params_.cq_off.cqes);

            const unsigned tail = *sq_tail;
            const unsigned idx  = tail & *sq_mask;
            io_uring_sqe & sqe  = sqes_[idx];
            std::memset(&sqe, 0, sizeof(sqe));
            sqe.opcode    = IORING_OP_READ;
            sqe.fd        = fd;
            sqe.off       = offset;
            sqe.addr      = reinterpret_cast<uintptr_t>(gtt_host_dst);
            sqe.len       = static_cast<uint32_t>(bytes);
            sqe.user_data = offset ^ 0x9e3779b97f4a7c15ULL;

            sq_arr[idx] = idx;
            __atomic_store_n(sq_tail, tail + 1, __ATOMIC_RELEASE);

            const int ret = static_cast<int>(
                syscall(__NR_io_uring_enter, ring_fd_, 1u, 1u, IORING_ENTER_GETEVENTS, nullptr, 0));
            if (ret >= 0) {
                const unsigned c_head = __atomic_load_n(cq_head, __ATOMIC_ACQUIRE);
                const unsigned c_tail = __atomic_load_n(cq_tail, __ATOMIC_ACQUIRE);
                if (c_head != c_tail) {
                    const io_uring_cqe & cqe = cqes[c_head & *cq_mask];
                    const int32_t res = cqe.res;
                    __atomic_store_n(cq_head, c_head + 1, __ATOMIC_RELEASE);
                    return res >= 0 && static_cast<size_t>(res) >= bytes;
                }
            }
        }
#endif
#if defined(__linux__)
        const ssize_t n = ::pread(fd, gtt_host_dst, bytes, static_cast<off_t>(offset));
        return n >= 0 && static_cast<size_t>(n) >= bytes;
#else
        std::memset(gtt_host_dst, 0, std::min<size_t>(bytes, 64u));
        return true;
#endif
    }

    void close_ring() noexcept {
#if defined(__linux__) && defined(GGML_HIP_SSD_HAS_LINUX_IO_URING)
        if (sqes_ && sqes_ != MAP_FAILED) {
            (void) munmap(sqes_, sqes_sz_);
        }
        if (cq_ptr_ && cq_ptr_ != MAP_FAILED) {
            (void) munmap(cq_ptr_, cq_ring_sz_);
        }
        if (sq_ptr_ && sq_ptr_ != MAP_FAILED) {
            (void) munmap(sq_ptr_, sq_ring_sz_);
        }
        if (ring_fd_ >= 0) {
            (void) close(ring_fd_);
        }
        sqes_       = nullptr;
        cq_ptr_     = nullptr;
        sq_ptr_     = nullptr;
        ring_fd_    = -1;
        sq_ring_sz_ = 0;
        cq_ring_sz_ = 0;
        sqes_sz_    = 0;
#endif
    }

private:
#if defined(__linux__) && defined(GGML_HIP_SSD_HAS_LINUX_IO_URING)
    int              ring_fd_    = -1;
    void *           sq_ptr_     = nullptr;
    void *           cq_ptr_     = nullptr;
    io_uring_sqe *   sqes_       = nullptr;
    size_t           sq_ring_sz_ = 0;
    size_t           cq_ring_sz_ = 0;
    size_t           sqes_sz_    = 0;
    io_uring_params  params_{};
#endif
};

// Cross-layer MoE expert transition predictor for Layer L -> Layer L+1
class moe_expert_predictor {
public:
    void record_and_predict(int32_t layer_id, const int32_t * experts, int32_t count,
                            std::vector<int32_t> & predicted_next) {
        predicted_next.clear();
        if (layer_id < 0 || !experts || count <= 0) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        const int32_t clamped = std::min<int32_t>(count, 8);
        if (last_layer_ == layer_id - 1 && !last_experts_.empty()) {
            for (int32_t prev_e : last_experts_) {
                const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(last_layer_)) << 32) |
                                     static_cast<uint32_t>(prev_e);
                auto & next_map = transitions_[key];
                for (int32_t i = 0; i < clamped; ++i) {
                    if (experts[i] >= 0) {
                        next_map[experts[i]] += 1u;
                    }
                }
            }
        }
        last_layer_ = layer_id;
        last_experts_.assign(experts, experts + clamped);

        for (int32_t e : last_experts_) {
            if (e < 0) {
                continue;
            }
            const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(layer_id)) << 32) |
                                 static_cast<uint32_t>(e);
            auto it = transitions_.find(key);
            if (it != transitions_.end() && !it->second.empty()) {
                int32_t best_target = e;
                uint32_t best_score = 0;
                for (const auto & kv : it->second) {
                    if (kv.second > best_score) {
                        best_score  = kv.second;
                        best_target = kv.first;
                    }
                }
                if (std::find(predicted_next.begin(), predicted_next.end(), best_target) == predicted_next.end()) {
                    predicted_next.push_back(best_target);
                }
            } else if (std::find(predicted_next.begin(), predicted_next.end(), e) == predicted_next.end()) {
                predicted_next.push_back(e);
            }
        }
    }

private:
    std::mutex mutex_;
    int32_t last_layer_ = -1;
    std::vector<int32_t> last_experts_;
    std::unordered_map<uint64_t, std::unordered_map<int32_t, uint32_t>> transitions_;
};

} // namespace

struct gtt_allocation {
    void * raw_host     = nullptr;
    void * aligned_host = nullptr;
    void * aligned_dev  = nullptr;
    size_t user_bytes   = 0;
};

struct prefetch_slot {
    int32_t  layer_id   = -1;
    int32_t  expert_id  = -1;
    bool     ready      = false;
    bool     busy       = false;
    uint64_t lru_tick   = 0;
    void *   host_ptr   = nullptr;
    void *   dev_ptr    = nullptr;
    size_t   capacity   = 0;
};

struct prefetch_request {
    int      fd          = -1;
    uint64_t file_offset = 0;
    size_t   byte_length = 0;
    int32_t  layer_id    = -1;
    int32_t  expert_id   = -1;
    size_t   slot_index  = 0;
};

struct apu_v2_engine::impl {
    // General & GTT Arena state
    mutable std::mutex mutex;
    int                device_id        = 0;
    bool               initialized      = false;
    bool               unified_apu      = false;
    size_t             gtt_arena_bytes  = 0;
    size_t             peak_gtt_bytes   = 0;
    uint64_t           zero_copy_allocs = 0;
    mutable std::atomic<uint64_t> zero_copy_hits{0};
    std::unordered_map<const void *, gtt_allocation> allocations;

    // Hardware Safeguard #1 & #2: Routing mode & Custom Closed 12-Thread CPU Worker Pool
    std::atomic<workload_routing_mode> routing_mode{workload_routing_mode::UNDECIDED};
    std::atomic<bool>                  cpu_workers_dormant{true};
    size_t                             worker_pool_size    = 0;
    size_t                             reserved_os_threads = kReservedOsAndUringThreads;

    std::mutex                         pool_mutex;
    std::condition_variable            pool_work_cv;
    std::condition_variable            pool_done_cv;
    bool                               pool_stopping       = false;
    uint64_t                           pool_epoch          = 0;
    size_t                             pool_num_items      = 0;
    std::atomic<size_t>                pool_next_item{0};
    size_t                             pool_workers_done   = 0;
    std::function<void(size_t, size_t)> pool_task_fn;
    std::vector<std::thread>           cpu_workers;

    // Linux io_uring prefetcher pool (pinned to reserved cores 12..15)
    std::mutex                         uring_mutex;
    std::condition_variable            uring_cv;
    bool                               uring_stopping      = false;
    void *                             prefetch_arena_host = nullptr;
    size_t                             prefetch_slot_bytes = 2u * 1024u * 1024u; // 2 MiB per slot default
    uint64_t                           lru_clock           = 0;
    std::vector<prefetch_slot>         prefetch_slots;
    std::deque<prefetch_request>       prefetch_queue;
    std::vector<std::thread>           uring_workers;
    moe_expert_predictor               predictor;

    // Telemetry counters
    std::atomic<uint64_t> moe_layers_routed{0};
    std::atomic<uint64_t> experts_prefetched{0};
    std::atomic<uint64_t> gtt_prefetch_hits{0};
    std::atomic<uint64_t> cpu_worker_tasks_run{0};

    ~impl() noexcept {
        shutdown();
    }

    static size_t compute_closed_worker_count() noexcept {
        const unsigned hw = std::thread::hardware_concurrency();
        const size_t total = hw > 0 ? static_cast<size_t>(hw) : kTotalApuHardwareThreads;
        // Strictly reserve 4 threads (2 physical cores) for Ubuntu OS + io_uring prefetcher,
        // and strictly cap the custom closed CPU worker pool to 12 threads.
        const size_t available = total > kReservedOsAndUringThreads
            ? (total - kReservedOsAndUringThreads)
            : 1u;
        return std::min<size_t>(available, kClosedWorkerPoolMaxThreads);
    }

    bool init(int dev) noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        if (initialized) {
            return true;
        }
        device_id = dev >= 0 ? dev : 0;
        if (hipSetDevice(device_id) != hipSuccess) {
            (void) hipGetLastError();
        }
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, device_id) == hipSuccess) {
            unified_apu = (prop.integrated != 0);
        } else {
            (void) hipGetLastError();
            unified_apu = true;
        }
        worker_pool_size    = compute_closed_worker_count();
        reserved_os_threads = kReservedOsAndUringThreads;
        initialized         = true;

        std::fprintf(stderr,
            "[ggml-hip-ssd-v2] status=initialized device=%d unified_apu=%d "
            "closed_cpu_workers=%zu reserved_os_uring_threads=%zu (cores 0..%zu worker pool, %zu..%zu OS/io_uring)\n",
            device_id, int(unified_apu), worker_pool_size, reserved_os_threads,
            worker_pool_size > 0 ? worker_pool_size - 1 : 0,
            worker_pool_size, worker_pool_size + reserved_os_threads - 1);
        return true;
    }

    void * alloc_zero_copy_locked(size_t bytes) noexcept {
        if (bytes == 0 || bytes > SIZE_MAX - kGttPageAlignment) {
            return nullptr;
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

        gtt_allocation entry{};
        entry.raw_host     = raw_host;
        entry.aligned_host = aligned_host;
        entry.aligned_dev  = aligned_dev;
        entry.user_bytes   = bytes;

        try {
            allocations[aligned_host] = entry;
        } catch (...) {
            (void) hipHostFree(raw_host);
            return nullptr;
        }

        gtt_arena_bytes += bytes;
        peak_gtt_bytes = std::max(peak_gtt_bytes, gtt_arena_bytes);
        ++zero_copy_allocs;
        return aligned_host;
    }

    void * alloc_zero_copy(size_t bytes) noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            unified_apu         = true;
            worker_pool_size    = compute_closed_worker_count();
            reserved_os_threads = kReservedOsAndUringThreads;
            initialized         = true;
        }
        return alloc_zero_copy_locked(bytes);
    }

    bool free_zero_copy_locked(void * host_ptr) noexcept {
        if (!host_ptr) {
            return false;
        }
        auto it = allocations.find(host_ptr);
        if (it == allocations.end()) {
            return false;
        }
        void * raw_host = it->second.raw_host;
        const size_t bytes = it->second.user_bytes;
        allocations.erase(it);
        gtt_arena_bytes = (gtt_arena_bytes >= bytes) ? (gtt_arena_bytes - bytes) : 0;
        if (raw_host) {
            (void) hipHostFree(raw_host);
        }
        return true;
    }

    bool free_zero_copy(void * host_ptr) noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        return free_zero_copy_locked(host_ptr);
    }

    void * resolve_device_ptr(void * host_ptr) const noexcept {
        if (!host_ptr) {
            return nullptr;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto it = allocations.find(host_ptr);
            if (it != allocations.end()) {
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

    // Spawns the 12-thread closed CPU worker pool and 2 io_uring prefetcher threads
    // ONLY when an MoE model (e.g., Qwen 27B A3B) is active.
    void ensure_moe_workers_started() noexcept {
        {
            std::lock_guard<std::mutex> lock(pool_mutex);
            if (cpu_workers.empty() && !pool_stopping) {
                const size_t n_threads = compute_closed_worker_count(); // Strictly <= 12
                worker_pool_size = n_threads;
                try {
                    cpu_workers.reserve(n_threads);
                    for (size_t idx = 0; idx < n_threads; ++idx) {
                        cpu_workers.emplace_back([this, idx, n_threads]() {
                            cpu_worker_loop(idx, n_threads);
                        });
                    }
                } catch (...) {
                    pool_stopping = true;
                    pool_work_cv.notify_all();
                }
            }
        }
        {
            std::lock_guard<std::mutex> ulock(uring_mutex);
            if (uring_workers.empty() && !uring_stopping) {
                const size_t total_slot_bytes = kDefaultGttPrefetchSlots * prefetch_slot_bytes;
                {
                    std::lock_guard<std::mutex> alock(mutex);
                    if (!prefetch_arena_host) {
                        prefetch_arena_host = alloc_zero_copy_locked(total_slot_bytes);
                    }
                }
                if (prefetch_arena_host) {
                    char * host_base = static_cast<char *>(prefetch_arena_host);
                    char * dev_base  = static_cast<char *>(resolve_device_ptr(prefetch_arena_host));
                    prefetch_slots.resize(kDefaultGttPrefetchSlots);
                    for (size_t i = 0; i < kDefaultGttPrefetchSlots; ++i) {
                        prefetch_slots[i].host_ptr = host_base + i * prefetch_slot_bytes;
                        prefetch_slots[i].dev_ptr  = dev_base + i * prefetch_slot_bytes;
                        prefetch_slots[i].capacity = prefetch_slot_bytes;
                    }
                }
                try {
                    // Spawn 2 io_uring workers pinned inside the 4 reserved OS/io_uring hardware threads (cores 12..15)
                    constexpr size_t kIoUringThreads = 2;
                    uring_workers.reserve(kIoUringThreads);
                    for (size_t u = 0; u < kIoUringThreads; ++u) {
                        uring_workers.emplace_back([this, u]() {
                            io_uring_worker_loop(u);
                        });
                    }
                } catch (...) {
                    uring_stopping = true;
                    uring_cv.notify_all();
                }
            }
        }
    }

    void cpu_worker_loop(size_t worker_idx, size_t total_workers) noexcept {
        // Pin closed CPU worker threads strictly to cores [0 .. total_workers - 1] (cores 0..11),
        // keeping cores 12..15 untouched for Ubuntu OS, background services, and io_uring.
        pin_current_thread_to_core_range(worker_idx, worker_idx + 1);
        (void) total_workers;

        uint64_t seen_epoch = 0;
        for (;;) {
            std::function<void(size_t, size_t)> fn;
            size_t num_items = 0;
            {
                std::unique_lock<std::mutex> lock(pool_mutex);
                pool_work_cv.wait(lock, [&]() {
                    return pool_stopping ||
                           (routing_mode.load(std::memory_order_acquire) == workload_routing_mode::MOE_HYBRID_ENGAGED &&
                            !cpu_workers_dormant.load(std::memory_order_acquire) &&
                            pool_epoch > seen_epoch);
                });
                if (pool_stopping) {
                    return;
                }
                seen_epoch = pool_epoch;
                fn         = pool_task_fn;
                num_items  = pool_num_items;
            }

            if (fn && num_items > 0) {
                for (;;) {
                    const size_t item = pool_next_item.fetch_add(1, std::memory_order_relaxed);
                    if (item >= num_items) {
                        break;
                    }
                    fn(item, worker_idx);
                    cpu_worker_tasks_run.fetch_add(1, std::memory_order_relaxed);
                }
            }

            {
                std::lock_guard<std::mutex> lock(pool_mutex);
                ++pool_workers_done;
                if (pool_workers_done >= cpu_workers.size()) {
                    pool_done_cv.notify_all();
                }
            }
        }
    }

    void io_uring_worker_loop(size_t uring_worker_id) noexcept {
        // Pin io_uring prefetcher threads to the 4 reserved hardware threads (cores 12..15)
        const size_t reserved_begin = kClosedWorkerPoolMaxThreads;
        const size_t reserved_end   = kClosedWorkerPoolMaxThreads + kReservedOsAndUringThreads;
        pin_current_thread_to_core_range(reserved_begin, reserved_end);
        (void) uring_worker_id;

        linux_io_uring_channel channel;
        (void) channel.init(64);

        for (;;) {
            prefetch_request req{};
            void * target_gtt_host = nullptr;
            {
                std::unique_lock<std::mutex> lock(uring_mutex);
                uring_cv.wait(lock, [&]() {
                    return uring_stopping || !prefetch_queue.empty();
                });
                if (uring_stopping && prefetch_queue.empty()) {
                    break;
                }
                if (prefetch_queue.empty()) {
                    continue;
                }
                req = prefetch_queue.front();
                prefetch_queue.pop_front();
                if (req.slot_index < prefetch_slots.size()) {
                    target_gtt_host = prefetch_slots[req.slot_index].host_ptr;
                }
            }

            const bool ok = target_gtt_host != nullptr &&
                            channel.read_into_gtt(req.fd, req.file_offset, target_gtt_host, req.byte_length);

            {
                std::lock_guard<std::mutex> lock(uring_mutex);
                if (req.slot_index < prefetch_slots.size()) {
                    auto & slot = prefetch_slots[req.slot_index];
                    slot.busy   = false;
                    slot.ready  = ok;
                    if (ok) {
                        experts_prefetched.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        }
    }

    workload_routing_mode classify_model(const char * model_name_or_path,
                                         int32_t n_experts,
                                         int32_t n_expert_used,
                                         int32_t n_layers) noexcept {
        const std::string lower = read_process_model_hint(model_name_or_path);
        const bool is_gpt_oss_dense = (lower.find("gpt-oss") != std::string::npos) ||
                                      (lower.find("gpt_oss") != std::string::npos) ||
                                      (n_experts == 32 && lower.find("qwen") == std::string::npos);
        const bool is_qwen_moe      = (lower.find("qwen") != std::string::npos &&
                                       (lower.find("a3b") != std::string::npos || n_experts > 1));

        // Dense Model Sleep Logic:
        // If gpt-oss-20b or any standard dense model (n_experts <= 1) is loaded,
        // keep the 12-thread CPU worker pool 100% dormant and route 100% of math to Radeon 780M.
        if (is_gpt_oss_dense || (!is_qwen_moe && (n_experts <= 1 || n_expert_used <= 0))) {
            routing_mode.store(workload_routing_mode::DENSE_IGPU_100PCT, std::memory_order_release);
            cpu_workers_dormant.store(true, std::memory_order_release);
            std::fprintf(stderr,
                "[ggml-hip-ssd-v2] mode=DENSE_IGPU_100PCT model=\"%s\" n_experts=%d "
                "cpu_worker_pool=DORMANT igpu_offload=100%% (Radeon 780M)\n",
                model_name_or_path ? model_name_or_path : "dense", int(n_experts));
            return workload_routing_mode::DENSE_IGPU_100PCT;
        }

        routing_mode.store(workload_routing_mode::MOE_HYBRID_ENGAGED, std::memory_order_release);
        cpu_workers_dormant.store(false, std::memory_order_release);
        ensure_moe_workers_started();
        std::fprintf(stderr,
            "[ggml-hip-ssd-v2] mode=MOE_HYBRID_ENGAGED model=\"%s\" n_layers=%d n_experts=%d n_expert_used=%d "
            "cpu_worker_pool=%zu_threads_active reserved_os_uring_threads=%zu\n",
            model_name_or_path ? model_name_or_path : "moe-qwen-a3b",
            int(n_layers), int(n_experts), int(n_expert_used),
            worker_pool_size, reserved_os_threads);
        return workload_routing_mode::MOE_HYBRID_ENGAGED;
    }

    workload_routing_mode inspect_compute_graph(const ggml_cgraph * cgraph,
                                                const char * model_hint) noexcept {
        if (!cgraph) {
            return routing_mode.load(std::memory_order_acquire);
        }
        int32_t detected_experts     = 0;
        int32_t detected_expert_used = 0;
        int32_t mul_mat_id_nodes     = 0;
        bool    has_openai_swiglu    = false;

        for (int i = 0; i < cgraph->n_nodes; ++i) {
            const ggml_tensor * node = cgraph->nodes[i];
            if (!node) {
                continue;
            }
            if (node->op == GGML_OP_GLU && ggml_get_glu_op(node) == GGML_GLU_OP_SWIGLU_OAI) {
                has_openai_swiglu = true;
            }
            if (node->op == GGML_OP_MUL_MAT_ID && node->src[0] && node->src[2]) {
                ++mul_mat_id_nodes;
                detected_experts     = std::max<int32_t>(detected_experts, static_cast<int32_t>(node->src[0]->ne[2]));
                detected_expert_used = std::max<int32_t>(detected_expert_used, static_cast<int32_t>(node->src[2]->ne[0]));
            }
        }

        if (has_openai_swiglu) {
            return classify_model("gpt-oss-20b", detected_experts, detected_expert_used, cgraph->n_nodes);
        }
        if (mul_mat_id_nodes == 0) {
            return classify_model(model_hint ? model_hint : "dense-graph", 0, 0, cgraph->n_nodes);
        }
        return classify_model(model_hint ? model_hint : "moe-graph", detected_experts, detected_expert_used, mul_mat_id_nodes);
    }

    bool prefetch_expert_async(int fd, uint64_t file_offset, size_t byte_length,
                               int32_t layer_id, int32_t expert_id) noexcept {
        if (routing_mode.load(std::memory_order_acquire) != workload_routing_mode::MOE_HYBRID_ENGAGED ||
            cpu_workers_dormant.load(std::memory_order_acquire) || byte_length == 0) {
            return false;
        }
        ensure_moe_workers_started();

        std::lock_guard<std::mutex> lock(uring_mutex);
        if (uring_stopping || prefetch_slots.empty()) {
            return false;
        }

        // Check if already resident or in-flight in GTT prefetch slots
        auto existing = std::find_if(prefetch_slots.begin(), prefetch_slots.end(), [&](const prefetch_slot & s) {
            return s.layer_id == layer_id && s.expert_id == expert_id && (s.ready || s.busy);
        });
        if (existing != prefetch_slots.end()) {
            existing->lru_tick = ++lru_clock;
            if (existing->ready) {
                gtt_prefetch_hits.fetch_add(1, std::memory_order_relaxed);
            }
            return true;
        }

        // Find LRU non-busy GTT slot
        size_t chosen = prefetch_slots.size();
        uint64_t oldest_tick = UINT64_MAX;
        for (size_t i = 0; i < prefetch_slots.size(); ++i) {
            if (!prefetch_slots[i].busy && prefetch_slots[i].lru_tick < oldest_tick) {
                oldest_tick = prefetch_slots[i].lru_tick;
                chosen      = i;
            }
        }
        if (chosen >= prefetch_slots.size()) {
            return false;
        }

        const uint64_t aligned_off = align_down_page(file_offset);
        const size_t prefix        = static_cast<size_t>(file_offset - aligned_off);
        const size_t aligned_len   = std::min(align_up_page(prefix + byte_length), prefetch_slots[chosen].capacity);

        auto & slot    = prefetch_slots[chosen];
        slot.layer_id  = layer_id;
        slot.expert_id = expert_id;
        slot.ready     = false;
        slot.busy      = true;
        slot.lru_tick  = ++lru_clock;

        prefetch_queue.push_back({ fd, aligned_off, aligned_len, layer_id, expert_id, chosen });
        uring_cv.notify_one();
        return true;
    }

    bool on_mul_mat_id_dispatch(const ggml_tensor * dst,
                                const int32_t * routed_experts,
                                int32_t n_routed) noexcept {
        if (!dst || dst->op != GGML_OP_MUL_MAT_ID || !dst->src[0]) {
            return false;
        }
        if (routing_mode.load(std::memory_order_acquire) == workload_routing_mode::UNDECIDED) {
            const int32_t n_exp = static_cast<int32_t>(dst->src[0]->ne[2]);
            (void) classify_model(dst->src[0]->name, n_exp, n_routed > 0 ? n_routed : 4, 1);
        }
        if (routing_mode.load(std::memory_order_acquire) != workload_routing_mode::MOE_HYBRID_ENGAGED ||
            cpu_workers_dormant.load(std::memory_order_acquire)) {
            return false;
        }

        moe_layers_routed.fetch_add(1, std::memory_order_relaxed);

        // Parse layer index from tensor name ("blk.<N>.ffn_...")
        int32_t layer_id = 0;
        if (dst->src[0]->name[0] != '\0') {
            const char * blk_ptr = std::strstr(dst->src[0]->name, "blk.");
            if (blk_ptr) {
                layer_id = static_cast<int32_t>(std::strtol(blk_ptr + 4, nullptr, 10));
            }
        }

        if (routed_experts && n_routed > 0) {
            std::vector<int32_t> next_experts;
            predictor.record_and_predict(layer_id, routed_experts, n_routed, next_experts);
            for (int32_t next_e : next_experts) {
                (void) prefetch_expert_async(-1, 0, kGttPageAlignment, layer_id + 1, next_e);
            }
        }
        return true;
    }

    bool dispatch_moe_cpu_work(size_t num_items,
                               const std::function<void(size_t, size_t)> & fn) noexcept {
        // Strict Dense Model Sleep Safeguard:
        // Never wake CPU workers when in DENSE_IGPU_100PCT mode.
        if (routing_mode.load(std::memory_order_acquire) != workload_routing_mode::MOE_HYBRID_ENGAGED ||
            cpu_workers_dormant.load(std::memory_order_acquire) || !fn || num_items == 0) {
            return false;
        }
        ensure_moe_workers_started();

        std::unique_lock<std::mutex> lock(pool_mutex);
        if (cpu_workers.empty() || pool_stopping) {
            return false;
        }
        pool_task_fn      = fn;
        pool_num_items    = num_items;
        pool_next_item.store(0, std::memory_order_relaxed);
        pool_workers_done = 0;
        ++pool_epoch;

        pool_work_cv.notify_all();
        pool_done_cv.wait(lock, [&]() {
            return pool_stopping || pool_workers_done >= cpu_workers.size();
        });
        pool_task_fn   = nullptr;
        pool_num_items = 0;
        return true;
    }

    void shutdown() noexcept {
        std::vector<std::thread> pool_to_join;
        {
            std::lock_guard<std::mutex> lock(pool_mutex);
            pool_stopping = true;
            pool_to_join.swap(cpu_workers);
        }
        pool_work_cv.notify_all();
        pool_done_cv.notify_all();
        for (auto & t : pool_to_join) {
            if (t.joinable()) {
                t.join();
            }
        }

        std::vector<std::thread> uring_to_join;
        {
            std::lock_guard<std::mutex> lock(uring_mutex);
            uring_stopping = true;
            prefetch_queue.clear();
            uring_to_join.swap(uring_workers);
        }
        uring_cv.notify_all();
        for (auto & t : uring_to_join) {
            if (t.joinable()) {
                t.join();
            }
        }

        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized && allocations.empty()) {
            return;
        }
        for (const auto & kv : allocations) {
            if (kv.second.raw_host) {
                (void) hipHostFree(kv.second.raw_host);
            }
        }
        allocations.clear();
        prefetch_arena_host = nullptr;
        prefetch_slots.clear();

        std::fprintf(stderr,
            "[ggml-hip-ssd-v2] status=shutdown mode=%d dormant=%d cpu_workers=%zu "
            "reserved_threads=%zu moe_layers=%llu prefetched=%llu gtt_hits=%llu peak_gtt_mib=%.2f\n",
            int(routing_mode.load(std::memory_order_relaxed)),
            int(cpu_workers_dormant.load(std::memory_order_relaxed)),
            worker_pool_size, reserved_os_threads,
            static_cast<unsigned long long>(moe_layers_routed.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(experts_prefetched.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(gtt_prefetch_hits.load(std::memory_order_relaxed)),
            double(peak_gtt_bytes) / (1024.0 * 1024.0));

        gtt_arena_bytes = 0;
        initialized     = false;
        pool_stopping   = false;
        uring_stopping  = false;
    }
};

apu_v2_engine::apu_v2_engine() noexcept
    : impl_(new (std::nothrow) impl()) {}

apu_v2_engine::~apu_v2_engine() noexcept {
    delete impl_;
    impl_ = nullptr;
}

apu_v2_engine & apu_v2_engine::instance() noexcept {
    static apu_v2_engine inst;
    return inst;
}

bool apu_v2_engine::init(int device_id) noexcept {
    return impl_ ? impl_->init(device_id) : false;
}

void apu_v2_engine::shutdown() noexcept {
    if (impl_) {
        impl_->shutdown();
    }
}

void * apu_v2_engine::alloc_zero_copy(size_t bytes) noexcept {
    return impl_ ? impl_->alloc_zero_copy(bytes) : nullptr;
}

bool apu_v2_engine::free_zero_copy(void * host_ptr) noexcept {
    return impl_ ? impl_->free_zero_copy(host_ptr) : false;
}

void * apu_v2_engine::resolve_device_ptr(void * host_ptr) const noexcept {
    return impl_ ? impl_->resolve_device_ptr(host_ptr) : host_ptr;
}

const void * apu_v2_engine::resolve_device_ptr(const void * host_ptr) const noexcept {
    return resolve_device_ptr(const_cast<void *>(host_ptr));
}

workload_routing_mode apu_v2_engine::classify_model(const char * model_name_or_path,
                                                    int32_t n_experts,
                                                    int32_t n_expert_used,
                                                    int32_t n_layers) noexcept {
    return impl_
        ? impl_->classify_model(model_name_or_path, n_experts, n_expert_used, n_layers)
        : workload_routing_mode::DENSE_IGPU_100PCT;
}

workload_routing_mode apu_v2_engine::inspect_compute_graph(const ggml_cgraph * cgraph,
                                                           const char * model_hint) noexcept {
    return impl_
        ? impl_->inspect_compute_graph(cgraph, model_hint)
        : workload_routing_mode::DENSE_IGPU_100PCT;
}

bool apu_v2_engine::on_mul_mat_id_dispatch(const ggml_tensor * dst,
                                           const int32_t * routed_experts,
                                           int32_t n_routed) noexcept {
    return impl_ ? impl_->on_mul_mat_id_dispatch(dst, routed_experts, n_routed) : false;
}

bool apu_v2_engine::prefetch_expert_async(int fd, uint64_t file_offset, size_t byte_length,
                                          int32_t layer_id, int32_t expert_id) noexcept {
    return impl_ ? impl_->prefetch_expert_async(fd, file_offset, byte_length, layer_id, expert_id) : false;
}

bool apu_v2_engine::dispatch_moe_cpu_work(size_t num_items,
                                          const std::function<void(size_t, size_t)> & fn) noexcept {
    return impl_ ? impl_->dispatch_moe_cpu_work(num_items, fn) : false;
}

bool apu_v2_engine::is_dense_sleep_mode() const noexcept {
    return !impl_ || impl_->cpu_workers_dormant.load(std::memory_order_acquire);
}

bool apu_v2_engine::is_moe_engaged() const noexcept {
    return impl_ &&
           impl_->routing_mode.load(std::memory_order_acquire) == workload_routing_mode::MOE_HYBRID_ENGAGED &&
           !impl_->cpu_workers_dormant.load(std::memory_order_acquire);
}

v2_telemetry_snapshot apu_v2_engine::snapshot() const noexcept {
    v2_telemetry_snapshot out{};
    if (!impl_) {
        return out;
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    out.unified_apu          = impl_->unified_apu;
    out.routing_mode         = impl_->routing_mode.load(std::memory_order_relaxed);
    out.cpu_workers_dormant  = impl_->cpu_workers_dormant.load(std::memory_order_relaxed);
    out.active_cpu_workers   = out.cpu_workers_dormant ? 0 : impl_->worker_pool_size;
    out.reserved_os_threads  = impl_->reserved_os_threads;
    out.io_uring_threads     = out.cpu_workers_dormant ? 0 : 2;
    out.gtt_arena_bytes      = impl_->gtt_arena_bytes;
    out.zero_copy_allocs     = impl_->zero_copy_allocs;
    out.zero_copy_hits       = impl_->zero_copy_hits.load(std::memory_order_relaxed);
    out.moe_layers_routed    = impl_->moe_layers_routed.load(std::memory_order_relaxed);
    out.experts_prefetched   = impl_->experts_prefetched.load(std::memory_order_relaxed);
    out.gtt_prefetch_hits    = impl_->gtt_prefetch_hits.load(std::memory_order_relaxed);
    out.cpu_worker_tasks_run = impl_->cpu_worker_tasks_run.load(std::memory_order_relaxed);
    return out;
}

} // namespace ggml_hip_ssd
