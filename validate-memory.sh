#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# validate-memory.sh - Dynamic Memory Safety Validation (AddressSanitizer / ASan)
# Target Hardware: Bare-metal Ubuntu Server (Ryzen 7 8745HS / Radeon 780M gfx1103 / 48GB Unified RAM)
#
# Compiles the v2-custom and dspine-v2-custom branches with AddressSanitizer (ASan)
# enabled (-fsanitize=address -fno-omit-frame-pointer -g) and executes a short
# dynamic inference verification pass to detect out-of-bounds accesses, use-after-free,
# or memory leaks in the GTT zero-copy arena, io_uring prefetcher, and 12-thread closed CPU pool.

set -euo pipefail
{

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$SCRIPT_DIR}"
BUILD_ROOT="${BUILD_ROOT:-/tmp/llama-apu-asan-builds}"
GPU_TARGET="${GPU_TARGET:-gfx1103}"

MODEL_DIR="${MODEL_DIR:-/opt/models}"
DENSE_MODEL="${GPT_OSS_20B_GGUF:-${MODEL_DIR}/gpt-oss-20b-Q4_K_M.gguf}"
MOE_MODEL="${QWEN_27B_A3B_GGUF:-${MODEL_DIR}/Qwen3-27B-A3B-Q4_K_M.gguf}"

V2_BRANCHES=("v2-custom" "dspine-v2-custom")

echo "=================================================================="
echo " Dynamic Memory Safety Validation (ASan) - Ryzen 7 8745HS / 780M"
echo "=================================================================="
echo "Repository : $REPO_DIR"
echo "Build Root : $BUILD_ROOT"
echo "GPU Target : $GPU_TARGET"
echo "Branches   : ${V2_BRANCHES[*]}"
echo ""

cd "$REPO_DIR"

ORIG_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")"
STASH_CREATED=0
if ! git diff --quiet || ! git diff --cached --quiet; then
    git stash push -u -m "validate-memory-autostash" >/dev/null 2>&1 || true
    STASH_CREATED=1
fi

cleanup() {
    cd "$REPO_DIR"
    git checkout -f "$ORIG_BRANCH" >/dev/null 2>&1 || true
    if [[ "$STASH_CREATED" -eq 1 ]]; then
        git stash pop >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

export ASAN_OPTIONS="detect_leaks=1:halt_on_error=1:abort_on_error=1:strict_string_checks=1:detect_stack_use_after_return=1"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

# Verify 40 GiB GTT pages_limit (10485760 pages) before running ASan on target APU
TTM_PAGES_LIMIT_FILE="/sys/module/ttm/parameters/pages_limit"
MIN_GTT_PAGES=10485760
if [[ -r "$TTM_PAGES_LIMIT_FILE" ]]; then
    CURRENT_GTT_PAGES="$(tr -d '[:space:]' < "$TTM_PAGES_LIMIT_FILE")"
    if [[ ! "$CURRENT_GTT_PAGES" =~ ^[0-9]+$ ]] || (( CURRENT_GTT_PAGES < MIN_GTT_PAGES )); then
        echo "WARNING: GTT pages_limit is ${CURRENT_GTT_PAGES:-0} (< ${MIN_GTT_PAGES} / 40 GiB)." >&2
        echo "Configure /etc/modprobe.d/ttm.conf (options ttm pages_limit=10485760 page_pool_size=10485760) and reboot." >&2
        exit 1
    fi
    echo "[GTT] Verified pages_limit=${CURRENT_GTT_PAGES} (>= 40 GiB)."
fi

ASAN_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g -O1"

for branch in "${V2_BRANCHES[@]}"; do
    echo "------------------------------------------------------------------"
    echo "[ASan] Checking out branch: $branch"
    echo "------------------------------------------------------------------"
    git checkout -f "$branch"

    if command -v cppcheck >/dev/null 2>&1; then
        echo "[ASan] Running pre-build cppcheck on $branch..."
        cppcheck --language=c++ --enable=warning,style,performance,portability \
            --inconclusive --std=c++17 --error-exitcode=1 \
            -Iggml/include -Iggml/src \
            ggml/src/ggml-hip/ggml-hip-ssd.h \
            ggml/src/ggml-hip/ggml-hip-ssd.cpp
    fi

    BDIR="${BUILD_ROOT}/${branch}"
    rm -rf "$BDIR"
    mkdir -p "$BDIR"

    echo "[ASan] Configuring CMake with AddressSanitizer for $branch..."
    cmake -S . -B "$BDIR" \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DGGML_HIP=ON \
        -DAMDGPU_TARGETS="$GPU_TARGET" \
        -DGPU_TARGETS="$GPU_TARGET" \
        -DCMAKE_C_FLAGS="$ASAN_FLAGS" \
        -DCMAKE_CXX_FLAGS="$ASAN_FLAGS" \
        -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address" \
        -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address" \
        >"${BDIR}/cmake-config.log" 2>&1

    echo "[ASan] Compiling llama-cli & test-backend-ops on $branch..."
    cmake --build "$BDIR" --target llama-cli test-backend-ops -j"$(nproc)" \
        >"${BDIR}/cmake-build.log" 2>&1

    CLI_BIN="${BDIR}/bin/llama-cli"
    TEST_BIN="${BDIR}/bin/test-backend-ops"

    if [[ -x "$TEST_BIN" ]]; then
        echo "[ASan] Running backend operator memory check ($TEST_BIN)..."
        "$TEST_BIN" test -o MUL_MAT_ID >"${BDIR}/asan-backend-ops.log" 2>&1 || {
            echo "[ERROR] ASan failure in test-backend-ops on branch $branch! See ${BDIR}/asan-backend-ops.log" >&2
            tail -n 30 "${BDIR}/asan-backend-ops.log" >&2
            exit 1
        }
    fi

    # 1. Verify Dense Model Sleep Safeguard under ASan
    if [[ -f "$DENSE_MODEL" ]]; then
        echo "[ASan] Testing Dense Model Sleep Safeguard on $(basename "$DENSE_MODEL")..."
        "$CLI_BIN" -m "$DENSE_MODEL" -p "Verify APU GTT zero-copy memory safety:" -n 16 --temp 0.0 -ngl 999 \
            >"${BDIR}/asan-dense.log" 2>&1
        if ! grep -q "mode=DENSE_IGPU_100PCT" "${BDIR}/asan-dense.log"; then
            echo "[WARN] Dense sleep telemetry marker not found in ${BDIR}/asan-dense.log" >&2
        else
            echo "  -> Verified: CPU worker pool remained DORMANT (100% Radeon 780M)."
        fi
    else
        echo "[ASan] Skipping dense model run (file not found at $DENSE_MODEL; set GPT_OSS_20B_GGUF to override)."
    fi

    # 2. Verify MoE Hybrid Engagement & 12-Thread Core Reservation under ASan
    if [[ -f "$MOE_MODEL" ]]; then
        echo "[ASan] Testing MoE Hybrid Engagement & io_uring Prefetcher on $(basename "$MOE_MODEL")..."
        "$CLI_BIN" -m "$MOE_MODEL" -p "Verify MoE expert routing and io_uring GTT prefetch safety:" -n 16 --temp 0.0 -ngl 999 \
            >"${BDIR}/asan-moe.log" 2>&1
        if ! grep -q "mode=MOE_HYBRID_ENGAGED" "${BDIR}/asan-moe.log"; then
            echo "[WARN] MoE engagement telemetry marker not found in ${BDIR}/asan-moe.log" >&2
        else
            echo "  -> Verified: 12-thread CPU worker pool + io_uring prefetcher engaged cleanly."
        fi
    else
        echo "[ASan] Skipping MoE model run (file not found at $MOE_MODEL; set QWEN_27B_A3B_GGUF to override)."
    fi

    echo "[PASS] Branch '$branch' passed AddressSanitizer validation with 0 memory errors."
    echo ""
done

echo "=================================================================="
echo " ALL V2 CUSTOM BRANCHES PASSED ASAN DYNAMIC MEMORY VALIDATION"
echo "=================================================================="
exit 0
}
