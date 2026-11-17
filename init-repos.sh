#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# init-repos.sh - Initialize and populate the 5 architectural branches in llama.cpp-APU-upgrade:
#   1. main             : Baseline upstream llama.cpp
#   2. v1-custom        : Injects V1 ggml-hip-ssd (basic zero-copy GTT memory arena)
#   3. v2-custom        : Injects V2 ggml-hip-ssd (zero-copy + io_uring prefetcher + 12-thread closed CPU worker pool)
#   4. dspine-baseline  : Clean pull from the dspine branch of llama.cpp
#   5. dspine-v2-custom : Merges dspine-baseline with v2-custom logic
#
# Target Hardware: Bare-metal Ubuntu Server (AMD Ryzen 7 8745HS / Radeon 780M gfx1103 / 48GB Unified RAM)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$SCRIPT_DIR}"
UPSTREAM_REPO="${UPSTREAM_REPO:-$SCRIPT_DIR/../llama.cpp}"

echo "=== Phase 1: Initializing llama.cpp-APU-upgrade Git Repository & 5 Branches ==="
echo "Target Repo   : $REPO_DIR"
echo "Source Repo   : $UPSTREAM_REPO"

cd "$REPO_DIR"

if [[ ! -d ".git" ]]; then
    git init
fi

git config user.name "${GIT_AUTHOR_NAME:-APU Benchmark Suite}"
git config user.email "${GIT_AUTHOR_EMAIL:-apu-bench@local}"

# Save our patches directory and scripts to a temporary staging directory before switching branches
PATCH_STAGE="$(mktemp -d)"
trap 'rm -rf "$PATCH_STAGE"' EXIT
cp -r "$REPO_DIR/patches" "$PATCH_STAGE/patches"
for s in init-repos.sh validate-memory.sh bench-inferssd.sh; do
    [[ -f "$REPO_DIR/$s" ]] && cp "$REPO_DIR/$s" "$PATCH_STAGE/$s"
done

# Discard any unstaged working tree modifications before switching branches
git reset --hard HEAD >/dev/null 2>&1 || true
git clean -fd >/dev/null 2>&1 || true

# 1. Fetch dspine and its parent (clean upstream baseline) from local llama.cpp clone
echo "[1/5] Fetching upstream baseline (main) and dspine-baseline from $UPSTREAM_REPO..."
git fetch "$UPSTREAM_REPO" "refs/remotes/origin/dspine:refs/heads/dspine-baseline" --force

# Set 'main' to the exact upstream llama.cpp commit right before the dspine commit (dspine-baseline~1)
# so main and dspine-baseline share an identical upstream base.
MAIN_COMMIT="$(git rev-parse dspine-baseline~1)"
git branch -f main "$MAIN_COMMIT"
git checkout -f main
echo "  -> Branch 'main' set to upstream commit: $(git rev-parse --short main)"
echo "  -> Branch 'dspine-baseline' set to commit: $(git rev-parse --short dspine-baseline)"

PY_BIN="$(command -v python3 2>/dev/null || command -v python 2>/dev/null)"

# Helper function to wire ggml-hip-ssd.cpp into ggml/src/ggml-hip/CMakeLists.txt and ggml-cuda.cu
inject_hip_ssd_cmake_and_hooks() {
    local variant="$1" # "v1" or "v2"
    local cmake_file="ggml/src/ggml-hip/CMakeLists.txt"

    if ! grep -q "ggml-hip-ssd.cpp" "$cmake_file"; then
        "$PY_BIN" - "$cmake_file" "$variant" <<'PY'
import pathlib, sys

cmake_path = pathlib.Path(sys.argv[1])
variant = sys.argv[2]
text = cmake_path.read_text(encoding="utf-8")

needle = "ggml_add_backend_library(ggml-hip"
injection = (
    f"# Custom {variant.upper()} ggml-hip-ssd backend for AMD Ryzen 7 8745HS / Radeon 780M APU\n"
    "list(APPEND GGML_HEADERS_ROCM \"ggml-hip-ssd.h\")\n"
    "list(APPEND GGML_SOURCES_ROCM \"ggml-hip-ssd.cpp\")\n"
    f"add_compile_definitions(GGML_HIP_SSD_{variant.upper()}=1)\n\n"
)
if needle in text:
    text = text.replace(needle, injection + needle, 1)
else:
    text += "\n" + injection
cmake_path.write_text(text, encoding="utf-8")
PY
    fi

    # Hook ggml-hip-ssd zero-copy GTT arena and V2 native MoE routing into ggml-cuda.cu when GGML_USE_HIP is active
    local cuda_file="ggml/src/ggml-cuda/ggml-cuda.cu"
    if ! grep -q "ggml-hip-ssd.h" "$cuda_file"; then
        "$PY_BIN" - "$cuda_file" "$variant" <<'PY'
import pathlib, sys

cuda_path = pathlib.Path(sys.argv[1])
variant = sys.argv[2]
text = cuda_path.read_text(encoding="utf-8")

include_hook = (
    '#include "ggml.h"\n'
    '#if defined(GGML_USE_HIP)\n'
    '#include "../ggml-hip/ggml-hip-ssd.h"\n'
    '#endif\n'
)
text = text.replace('#include "ggml.h"\n', include_hook, 1)

if variant == "v1":
    host_hook = (
        "static void * ggml_cuda_host_malloc(size_t size) {\n"
        "#if defined(GGML_USE_HIP) && defined(GGML_HIP_SSD_V1)\n"
        "    if (void * gtt_ptr = ggml_hip_ssd::gtt_zero_copy_arena::instance().alloc_zero_copy(size)) {\n"
        "        return gtt_ptr;\n"
        "    }\n"
        "#endif\n"
    )
    free_hook = (
        "static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {\n"
        "#if defined(GGML_USE_HIP) && defined(GGML_HIP_SSD_V1)\n"
        "    if (ggml_hip_ssd::gtt_zero_copy_arena::instance().free_zero_copy(buffer->context)) {\n"
        "        return;\n"
        "    }\n"
        "#endif\n"
    )
else:
    host_hook = (
        "static void * ggml_cuda_host_malloc(size_t size) {\n"
        "#if defined(GGML_USE_HIP) && defined(GGML_HIP_SSD_V2)\n"
        "    if (void * gtt_ptr = ggml_hip_ssd::apu_v2_engine::instance().alloc_zero_copy(size)) {\n"
        "        return gtt_ptr;\n"
        "    }\n"
        "#endif\n"
    )
    free_hook = (
        "static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {\n"
        "#if defined(GGML_USE_HIP) && defined(GGML_HIP_SSD_V2)\n"
        "    if (ggml_hip_ssd::apu_v2_engine::instance().free_zero_copy(buffer->context)) {\n"
        "        return;\n"
        "    }\n"
        "#endif\n"
    )
    mul_mat_id_hook = (
        "static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {\n"
        "#if defined(GGML_USE_HIP) && defined(GGML_HIP_SSD_V2)\n"
        "    (void) ggml_hip_ssd::apu_v2_engine::instance().on_mul_mat_id_dispatch(dst, nullptr, 0);\n"
        "#endif\n"
    )
    graph_compute_hook = (
        "static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {\n"
        "#if defined(GGML_USE_HIP) && defined(GGML_HIP_SSD_V2)\n"
        "    if (ggml_hip_ssd::apu_v2_engine::instance().snapshot().routing_mode == ggml_hip_ssd::workload_routing_mode::UNDECIDED) {\n"
        "        (void) ggml_hip_ssd::apu_v2_engine::instance().inspect_compute_graph(cgraph, nullptr);\n"
        "    }\n"
        "#endif\n"
    )
    text = text.replace(
        "static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {\n",
        mul_mat_id_hook,
        1,
    )
    text = text.replace(
        "static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {\n",
        graph_compute_hook,
        1,
    )

text = text.replace("static void * ggml_cuda_host_malloc(size_t size) {\n", host_hook, 1)
text = text.replace(
    "static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {\n",
    free_hook,
    1,
)
cuda_path.write_text(text, encoding="utf-8")
PY
    fi
}

# 2. Create v1-custom branch from main
echo "[2/5] Creating 'v1-custom' branch (V1 basic zero-copy GTT memory arena)..."
git checkout -B v1-custom main
cp "$PATCH_STAGE/patches/v1/ggml-hip-ssd.h"   ggml/src/ggml-hip/ggml-hip-ssd.h
cp "$PATCH_STAGE/patches/v1/ggml-hip-ssd.cpp" ggml/src/ggml-hip/ggml-hip-ssd.cpp
cp -r "$PATCH_STAGE/patches" "$REPO_DIR/patches"
for s in init-repos.sh validate-memory.sh bench-inferssd.sh; do
    [[ -f "$PATCH_STAGE/$s" ]] && cp "$PATCH_STAGE/$s" "$REPO_DIR/$s" && chmod +x "$REPO_DIR/$s" && git add "$s"
done
inject_hip_ssd_cmake_and_hooks "v1"
git add patches ggml/src/ggml-hip/ggml-hip-ssd.h ggml/src/ggml-hip/ggml-hip-ssd.cpp \
        ggml/src/ggml-hip/CMakeLists.txt ggml/src/ggml-cuda/ggml-cuda.cu
git commit -m "ggml-hip : add V1 zero-copy GTT memory arena backend (ggml-hip-ssd)"

# 3. Create v2-custom branch from main
echo "[3/5] Creating 'v2-custom' branch (V2 zero-copy GTT + io_uring prefetcher + 12-thread closed CPU pool)..."
git checkout -B v2-custom main
cp "$PATCH_STAGE/patches/v2/ggml-hip-ssd.h"   ggml/src/ggml-hip/ggml-hip-ssd.h
cp "$PATCH_STAGE/patches/v2/ggml-hip-ssd.cpp" ggml/src/ggml-hip/ggml-hip-ssd.cpp
cp -r "$PATCH_STAGE/patches" "$REPO_DIR/patches"
for s in init-repos.sh validate-memory.sh bench-inferssd.sh; do
    [[ -f "$PATCH_STAGE/$s" ]] && cp "$PATCH_STAGE/$s" "$REPO_DIR/$s" && chmod +x "$REPO_DIR/$s" && git add "$s"
done
inject_hip_ssd_cmake_and_hooks "v2"
git add patches ggml/src/ggml-hip/ggml-hip-ssd.h ggml/src/ggml-hip/ggml-hip-ssd.cpp \
        ggml/src/ggml-hip/CMakeLists.txt ggml/src/ggml-cuda/ggml-cuda.cu
git commit -m "ggml-hip : add V2 zero-copy GTT + io_uring prefetcher + 12-thread closed CPU worker pool"

# 4. Verify dspine-baseline branch
echo "[4/5] Verifying 'dspine-baseline' branch..."
git checkout -f dspine-baseline

# 5. Create dspine-v2-custom by merging dspine-baseline with v2-custom
echo "[5/5] Creating 'dspine-v2-custom' branch (merging dspine-baseline with v2-custom)..."
git checkout -B dspine-v2-custom dspine-baseline
git merge --no-edit v2-custom -m "Merge v2-custom (ggml-hip-ssd V2 APU backend) into dspine-baseline"

echo ""
echo "=== All 5 Branches Successfully Initialized in $REPO_DIR ==="
git branch -v
git status -s
