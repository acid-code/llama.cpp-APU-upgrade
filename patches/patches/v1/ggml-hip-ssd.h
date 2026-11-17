// SPDX-License-Identifier: MIT
//
// ggml-hip-ssd.h - V1 Basic Zero-Copy GTT Memory Arena for AMD Ryzen APUs
// Target: Ryzen 7 8745HS / Radeon 780M (gfx1103) / 48GB Unified DDR5 RAM

#pragma once

#include <cstddef>
#include <cstdint>
#include "ggml.h"
#include "ggml-backend.h"

namespace ggml_hip_ssd {

constexpr size_t kGttPageAlignment = 4096;

struct gtt_arena_stats {
    bool     unified_apu       = false;
    size_t   allocated_bytes   = 0;
    size_t   peak_bytes        = 0;
    uint64_t zero_copy_allocs  = 0;
    uint64_t zero_copy_hits    = 0;
};

class gtt_zero_copy_arena {
public:
    static gtt_zero_copy_arena & instance() noexcept;

    bool init(int device_id) noexcept;
    void shutdown() noexcept;

    void * alloc_zero_copy(size_t bytes) noexcept;
    void   free_zero_copy(void * host_ptr) noexcept;

    void *       resolve_device_ptr(void * host_ptr) const noexcept;
    const void * resolve_device_ptr(const void * host_ptr) const noexcept;

    bool is_unified_apu() const noexcept;
    gtt_arena_stats snapshot() const noexcept;

private:
    gtt_zero_copy_arena() noexcept;
    ~gtt_zero_copy_arena() noexcept;
    gtt_zero_copy_arena(const gtt_zero_copy_arena &) = delete;
    gtt_zero_copy_arena & operator=(const gtt_zero_copy_arena &) = delete;

    struct impl;
    impl * impl_ = nullptr;
};

} // namespace ggml_hip_ssd
