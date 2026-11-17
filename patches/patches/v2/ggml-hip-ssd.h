// SPDX-License-Identifier: MIT
//
// ggml-hip-ssd.h - V2 Heterogeneous Zero-Copy GTT + io_uring Prefetcher +
//                  Custom Closed CPU Worker Pool for AMD Ryzen APUs
// Target Hardware: Ryzen 7 8745HS (8C/16T) / Radeon 780M (gfx1103) / 48GB Unified RAM
//
// Hardware Safeguards:
//   1. Core Reservation: Custom closed CPU worker pool is strictly capped at 12 threads
//      (cores 0..11). Explicitly reserves 4 hardware threads (2 physical cores, 12..15)
//      entirely for Ubuntu OS, background services, and the io_uring prefetcher.
//   2. Dense Model Sleep Logic: If a standard dense model is loaded (e.g. gpt-oss-20b
//      or n_experts <= 1), the 12-thread CPU worker pool stays 100% dormant on its
//      condition variable and routes 100% of execution to the Radeon 780M iGPU.
//      If an MoE model is loaded (specifically Qwen 27B A3B), the workers engage.
//   3. Native MoE Routing: Intercepts GGML_OP_MUL_MAT_ID natively in the backend
//      without using or implementing CLI flags such as --n-cpu-moe.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include "ggml.h"
#include "ggml-backend.h"

namespace ggml_hip_ssd {

constexpr size_t kGttPageAlignment           = 4096;
constexpr size_t kTotalApuHardwareThreads    = 16; // Ryzen 7 8745HS: 8 physical cores / 16 threads
constexpr size_t kClosedWorkerPoolMaxThreads = 12; // Strictly capped at 12 threads
constexpr size_t kReservedOsAndUringThreads  = 4;  // 2 physical cores (4 threads) reserved for OS + io_uring
constexpr size_t kDefaultGttPrefetchSlots    = 32;

enum class workload_routing_mode : uint8_t {
    UNDECIDED          = 0,
    DENSE_IGPU_100PCT  = 1, // Dense model (e.g., gpt-oss-20b): CPU pool dormant, 100% Radeon 780M
    MOE_HYBRID_ENGAGED = 2  // MoE model (e.g., Qwen 27B A3B): 12-thread CPU pool + io_uring active
};

struct v2_telemetry_snapshot {
    bool                  unified_apu            = false;
    workload_routing_mode routing_mode           = workload_routing_mode::UNDECIDED;
    bool                  cpu_workers_dormant    = true;
    size_t                active_cpu_workers     = 0;
    size_t                reserved_os_threads    = kReservedOsAndUringThreads;
    size_t                io_uring_threads       = 0;
    size_t                gtt_arena_bytes        = 0;
    uint64_t              zero_copy_allocs       = 0;
    uint64_t              zero_copy_hits         = 0;
    uint64_t              moe_layers_routed      = 0;
    uint64_t              experts_prefetched     = 0;
    uint64_t              gtt_prefetch_hits      = 0;
    uint64_t              cpu_worker_tasks_run   = 0;
};

class apu_v2_engine {
public:
    static apu_v2_engine & instance() noexcept;

    bool init(int device_id) noexcept;
    void shutdown() noexcept;

    // Zero-copy GTT memory arena API
    void * alloc_zero_copy(size_t bytes) noexcept;
    void   free_zero_copy(void * host_ptr) noexcept;
    void *       resolve_device_ptr(void * host_ptr) const noexcept;
    const void * resolve_device_ptr(const void * host_ptr) const noexcept;

    // Model & Graph inspection for automatic Dense Sleep vs MoE Engagement
    workload_routing_mode classify_model(const char * model_name_or_path,
                                         int32_t n_experts,
                                         int32_t n_expert_used,
                                         int32_t n_layers) noexcept;

    workload_routing_mode inspect_compute_graph(const ggml_cgraph * cgraph,
                                                const char * model_hint = nullptr) noexcept;

    // Native MoE expert routing hook called on GGML_OP_MUL_MAT_ID without --n-cpu-moe
    bool on_mul_mat_id_dispatch(const ggml_tensor * dst,
                                const int32_t * routed_experts,
                                int32_t n_routed) noexcept;

    // Enqueues an expert prefetch request for Layer (layer + 1) via Linux io_uring into GTT
    bool prefetch_expert_async(int fd, uint64_t file_offset, size_t byte_length,
                               int32_t layer_id, int32_t expert_id) noexcept;

    // Dispatches a parallel MoE task across the 12-thread closed CPU worker pool
    // (No-op returning false if routing_mode == DENSE_IGPU_100PCT)
    bool dispatch_moe_cpu_work(size_t num_items,
                               const std::function<void(size_t, size_t)> & fn) noexcept;

    bool is_dense_sleep_mode() const noexcept;
    bool is_moe_engaged() const noexcept;
    v2_telemetry_snapshot snapshot() const noexcept;

private:
    apu_v2_engine() noexcept;
    ~apu_v2_engine() noexcept;
    apu_v2_engine(const apu_v2_engine &) = delete;
    apu_v2_engine & operator=(const apu_v2_engine &) = delete;

    struct impl;
    impl * impl_ = nullptr;
};

} // namespace ggml_hip_ssd
