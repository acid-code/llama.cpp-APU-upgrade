#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# bench-inferssd.sh - 5-Branch Architectural Benchmark Suite for Bare-Metal Ubuntu Server
# Target Hardware: AMD Ryzen 7 8745HS (8C/16T) / Radeon 780M (gfx1103) / 48GB Unified DDR5 RAM
#
# Workflow:
#   1. Read-Only GTT Memory Check (/sys/module/ttm/parameters/pages_limit >= 10485760 [40 GiB]).
#   2. Model Array Matrix targeting exact GGUF paths for gpt-oss-20b (dense/100% iGPU fallback)
#      and the full Qwen 27B A3B MoE model (native MoE routing without --n-cpu-moe).
#   3. Compile & Execute Matrix across all 5 branches:
#        - main
#        - v1-custom
#        - v2-custom
#        - dspine-baseline
#        - dspine-v2-custom
#   4. Drafting Logic:
#        - dspine-* & exp-* branches: sweep unclamped DFlash2 draft sizes 4, 5, 6, 7 and 7@p0.55.
#        - main, v1-custom, v2-custom: standard non-drafting inference.
#   5. Memory Safety Guard: verifies MemAvailable >= 28 GiB and GTT used <= 2 GiB before every run.
#   5. Logging: Outputs benchmark_results.csv:
#        Model Name,Branch Name,Block Size,Tokens-per-Second,Draft Acceptance Rate
#   6. Verdict: Parses benchmark_results.csv and prints final "VERDICT:" with winning configuration
#      and performance margin over the main baseline for each respective model.

set -euo pipefail
{

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$SCRIPT_DIR}"
CSV_OUT="${CSV_OUT:-${REPO_DIR}/benchmark_results.csv}"
BUILD_ROOT="${BUILD_ROOT:-/tmp/llama-apu-bench-builds}"
GPU_TARGET="${GPU_TARGET:-gfx1100}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
GEN_TOKENS="${GEN_TOKENS:-256}"

# ==============================================================================
# 1. GTT Memory Check (Read-Only): Require >= 10485760 pages (40 GiB)
# ==============================================================================
TTM_PAGES_LIMIT_FILE="/sys/module/ttm/parameters/pages_limit"
MIN_GTT_PAGES=10485760

if [[ ! -r "$TTM_PAGES_LIMIT_FILE" ]]; then
    echo "WARNING: Cannot read $TTM_PAGES_LIMIT_FILE." >&2
    echo "Please configure /etc/modprobe.d/ttm.conf with:" >&2
    echo "  options ttm pages_limit=10485760 page_pool_size=10485760" >&2
    echo "Then run 'sudo update-initramfs -u' and reboot." >&2
    exit 1
fi

CURRENT_GTT_PAGES="$(tr -d '[:space:]' < "$TTM_PAGES_LIMIT_FILE")"
if [[ ! "$CURRENT_GTT_PAGES" =~ ^[0-9]+$ ]] || (( CURRENT_GTT_PAGES < MIN_GTT_PAGES )); then
    CURRENT_GIB="$(awk -v p="${CURRENT_GTT_PAGES:-0}" 'BEGIN {printf "%.2f", (p * 4096.0) / (1024.0 * 1024.0 * 1024.0)}')"
    echo "WARNING: GTT pages_limit is ${CURRENT_GTT_PAGES:-0} (${CURRENT_GIB} GiB), which is less than ${MIN_GTT_PAGES} (40 GiB)." >&2
    echo "Please configure /etc/modprobe.d/ttm.conf:" >&2
    echo "  echo 'options ttm pages_limit=10485760 page_pool_size=10485760' | sudo tee /etc/modprobe.d/ttm.conf" >&2
    echo "  sudo update-initramfs -u && sudo reboot" >&2
    exit 1
fi

GTT_GIB="$(awk -v p="$CURRENT_GTT_PAGES" 'BEGIN {printf "%.2f", (p * 4096.0) / (1024.0 * 1024.0 * 1024.0)}')"
echo "=== GTT Memory Check Passed: pages_limit=${CURRENT_GTT_PAGES} (${GTT_GIB} GiB >= 40.00 GiB) ==="

# ==============================================================================
# 2. Model Array Matrix (Exact Paths for gpt-oss-20b and Full Qwen 27B A3B GGUF)
# ==============================================================================
MODEL_DIR="${MODEL_DIR:-${REPO_DIR}/models}"
GPT_OSS_20B_PATH="${GPT_OSS_20B_GGUF:-${MODEL_DIR}/gpt-oss-20b-Q4_K_M.gguf}"
QWEN_27B_A3B_PATH="${QWEN_27B_A3B_GGUF:-${MODEL_DIR}/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"

# Optional DSpine/DFlash draft sidecar/model paths per target model
GPT_OSS_DSPINE_DRAFT="${GPT_OSS_DSPINE_DRAFT_GGUF:-${MODEL_DIR}/gpt-oss-20b-dspine-draft-Q8_0.gguf}"
QWEN_27B_DSPINE_DRAFT="${QWEN_27B_DSPINE_DRAFT_GGUF:-${MODEL_DIR}/qwen36-35b-a3b-dflash-Q8_0.gguf}"

MODELS=(
    "$GPT_OSS_20B_PATH"
    "$QWEN_27B_A3B_PATH"
)

BRANCHES=(
    "main"
    "v1-custom"
    "v2-custom"
    "dspine-baseline"
    "dspine-v2-custom"
    "exp-coalesced-dflash"
)

# DFlash2 model has dflash.block_size=8 (max draft tokens = block_size - 1 = 7).
# Values >= 8 are silently clamped to 7; sweep real unclamped sizes plus p_min=0.55 adaptive truncation.
DRAFT_BLOCK_SIZES=(4 5 6 7 "7@p0.55")

cd "$REPO_DIR"
ORIG_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")"
STASH_CREATED=0
if ! git diff --quiet || ! git diff --cached --quiet; then
    git stash push -u -m "bench-inferssd-autostash" >/dev/null 2>&1 || true
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

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

if [[ -e /dev/kfd && ! -w /dev/kfd ]]; then
    echo "ERROR: Current user ($(id -un)) does not have write access to /dev/kfd." >&2
    echo "Run: sudo usermod -aG render,video $(id -un) && sudo chmod 666 /dev/kfd /dev/dri/render* /dev/dri/card*" >&2
    exit 1
fi

# Initialize CSV header
echo "Model Name,Branch Name,Block Size,Tokens-per-Second,Draft Acceptance Rate" > "$CSV_OUT"

SYNTHETIC_PROMPT="$(python3 -c "print('Explain unified memory Graphics Translation Table zero-copy DMA and Mixture-of-Experts routing on AMD Ryzen APUs in detail. ' * ((${PROMPT_TOKENS} // 16) + 1))")"

resolve_draft_model() {
    local model_path="$1"
    local base_lower
    base_lower="$(basename "$model_path" | tr '[:upper:]' '[:lower:]')"
    if [[ "$base_lower" == *"gpt-oss"* || "$base_lower" == *"gpt_oss"* ]]; then
        echo "$GPT_OSS_DSPINE_DRAFT"
    else
        echo "$QWEN_27B_DSPINE_DRAFT"
    fi
}

check_gpu_memory_guard() {
    local min_avail_kb=$((28 * 1024 * 1024))
    local max_gtt_bytes=$((2 * 1024 * 1024 * 1024))
    local avail_kb
    avail_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    local gtt_used=0
    for f in /sys/class/drm/card*/device/mem_info_gtt_used; do
        if [[ -r "$f" ]]; then
            local v
            v="$(tr -d '[:space:]' < "$f")"
            if [[ "$v" =~ ^[0-9]+$ ]] && (( v > gtt_used )); then
                gtt_used="$v"
            fi
        fi
    done

    if (( avail_kb < min_avail_kb || gtt_used > max_gtt_bytes )); then
        local avail_gib gtt_gib
        avail_gib="$(awk -v k="$avail_kb" 'BEGIN {printf "%.2f", k / (1024.0 * 1024.0)}')"
        gtt_gib="$(awk -v b="$gtt_used" 'BEGIN {printf "%.2f", b / (1024.0 * 1024.0 * 1024.0)}')"
        echo "SAFETY ABORT: MemAvailable=${avail_gib} GiB (need >= 28 GiB), GTT used=${gtt_gib} GiB (need <= 2 GiB)." >&2
        echo "Another process (e.g. ollama or llama-server) holds GPU memory. Stop it to avoid a TTM OOM freeze." >&2
        exit 1
    fi
}

run_inference_and_parse() {
    local bin_dir="$1"
    local model_path="$2"
    local branch="$3"
    local block_spec="$4"
    local log_file="$5"

    local cli_bin="${bin_dir}/llama-cli"
    local spec_bin="${bin_dir}/llama-speculative"

    local block_size="${block_spec%%@*}"
    local p_min="0.00"
    if [[ "$block_spec" == *"@p"* ]]; then
        p_min="${block_spec#*@p}"
    fi

    # Note: Do NOT pass --n-cpu-moe; custom V2 backend handles MoE routing natively
    local attempt=1
    while (( attempt <= 2 )); do
        check_gpu_memory_guard
        if [[ "$branch" == dspine-* || "$branch" == exp-* ]]; then
            local draft_path
            draft_path="$(resolve_draft_model "$model_path")"
            local draft_args=()
            if [[ -f "$draft_path" ]]; then
                draft_args=(-md "$draft_path" -ngld 999 --spec-draft-n-max "$block_size" --spec-draft-n-min 1 --spec-draft-p-min "$p_min" -lv 5)
            else
                draft_args=(--spec-type ngram-simple --spec-draft-n-max "$block_size" --spec-draft-n-min 1 -lv 5)
            fi

            if [[ -x "$spec_bin" && -f "$draft_path" ]]; then
                timeout 180 "$spec_bin" -m "$model_path" "${draft_args[@]}" \
                    -fit off -lm none -fa on -ub 64 -b 512 -t 12 \
                    -p "$SYNTHETIC_PROMPT" -n "$GEN_TOKENS" --temp 0.0 --seed 42 -ngl 999 -c 4096 \
                    >"$log_file" 2>&1 || true
            else
                timeout 180 "$cli_bin" -m "$model_path" "${draft_args[@]}" \
                    -fit off -lm none -fa on -ub 64 -b 512 -t 12 \
                    -p "$SYNTHETIC_PROMPT" -n "$GEN_TOKENS" --temp 0.0 --seed 42 -ngl 999 -c 4096 --single-turn \
                    >"$log_file" 2>&1 || true
            fi
        else
            timeout 180 "$cli_bin" -m "$model_path" \
                -fit off -lm none -fa on -ub 64 -b 512 -t 12 \
                -p "$SYNTHETIC_PROMPT" -n "$GEN_TOKENS" --temp 0.0 --seed 42 -ngl 999 -c 4096 --single-turn \
                >"$log_file" 2>&1 || true
        fi

        if ! grep -q "ROCm error" "$log_file" 2>/dev/null; then
            break
        fi
        sleep 6
        (( ++attempt ))
    done

    python3 - "$log_file" "$branch" <<'PY'
import re, sys

log_path, branch = sys.argv[1], sys.argv[2]
try:
    text = open(log_path, "r", encoding="utf-8", errors="replace").read()
except OSError:
    text = ""

# Parse decode tokens-per-second from llama.cpp timing output
tps = 0.0
patterns = [
    r"eval time\s*=\s*[\d.]+\s*ms\s*/\s*\d+\s*(?:runs|tokens)\s*\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second\)",
    r"predicted_per_second[\"']?\s*[:=]\s*([\d.]+)",
    r"Generation:\s*([\d.]+)\s*t/s",
]
for pat in patterns:
    matches = re.findall(pat, text, flags=re.IGNORECASE)
    if matches:
        tps = float(matches[-1])
        break

# Parse draft acceptance rate for dspine/exp branches
acc_rate = "N/A"
if branch.startswith("dspine-") or branch.startswith("exp-"):
    json_drafts = re.findall(r'"draft_n"\s*:\s*(\d+)\s*,\s*"draft_n_accepted"\s*:\s*(\d+)', text)
    spec_stats = re.findall(r"#gen tokens\s*=\s*(\d+),\s*#acc tokens\s*=\s*(\d+)", text)
    acc_matches = re.findall(r"(?:accept(?:ance|ed)?(?:\s+rate)?|f_acc)\s*[:=]\s*([\d.]+)\s*%?", text, flags=re.IGNORECASE)
    if json_drafts and int(json_drafts[-1][0]) > 0:
        d_n, d_acc = int(json_drafts[-1][0]), int(json_drafts[-1][1])
        acc_rate = f"{(100.0 * d_acc / d_n):.2f}%"
    elif spec_stats and int(spec_stats[-1][0]) > 0:
        gen_t, acc_t = int(spec_stats[-1][0]), int(spec_stats[-1][1])
        acc_rate = f"{(100.0 * acc_t / gen_t):.2f}%"
    elif acc_matches:
        val = float(acc_matches[-1])
        if val <= 1.0 and "%" not in text[text.find(acc_matches[-1]):text.find(acc_matches[-1]) + 12]:
            val *= 100.0
        acc_rate = f"{val:.2f}%"
    else:
        draft_pair = re.search(r"n_accept\s*=\s*(\d+)\s*/\s*n_draft\s*=\s*(\d+)", text)
        if draft_pair and int(draft_pair.group(2)) > 0:
            pct = 100.0 * float(draft_pair.group(1)) / float(draft_pair.group(2))
            acc_rate = f"{pct:.2f}%"
        else:
            acc_rate = "0.00%"

print(f"{tps:.2f},{acc_rate}")
PY
}

# ==============================================================================
# 3 & 4. Compile & Execute Matrix Across All 5 Branches and Models
# ==============================================================================
for branch in "${BRANCHES[@]}"; do
    echo "=================================================================="
    echo "[Branch] Checking out and compiling: $branch"
    echo "=================================================================="
    git checkout -f "$branch"

    BDIR="${BUILD_ROOT}/${branch}"
    BIN_DIR="${BDIR}/bin"
    mkdir -p "${BDIR}/tools/ui"
    if [[ -d "/tmp/llama-apu-asan-builds/v2-custom/tools/ui/dist" && ! -f "${BDIR}/tools/ui/dist/workbox-b3c04f83.js" ]]; then
        rm -rf "${BDIR}/tools/ui/dist"
        cp -a "/tmp/llama-apu-asan-builds/v2-custom/tools/ui/dist" "${BDIR}/tools/ui/dist"
    fi

    if [[ -x "${BIN_DIR}/llama-cli" ]]; then
        echo "  -> Reusing existing Release build in ${BIN_DIR}/llama-cli"
    else
        cmake -S . -B "$BDIR" \
            -DCMAKE_BUILD_TYPE=Release \
            -DGGML_HIP=ON \
            -DCMAKE_HIP_ARCHITECTURES="$GPU_TARGET" \
            -DAMDGPU_TARGETS="$GPU_TARGET" \
            -DGPU_TARGETS="$GPU_TARGET" \
            >"${BDIR}/cmake-config.log" 2>&1

        cmake --build "$BDIR" --target llama-cli -j"$(nproc)" \
            2>&1 | tee "${BDIR}/cmake-build.log"
    fi

    for model_path in "${MODELS[@]}"; do
        model_name="$(basename "$model_path" .gguf)"
        echo "  -> [Model] $model_name ($model_path)"

        if [[ ! -f "$model_path" ]]; then
            echo "     [WARN] Model file not found at $model_path; recording 0.00 tok/s placeholder." >&2
            if [[ "$branch" == dspine-* || "$branch" == exp-* ]]; then
                for bsize in "${DRAFT_BLOCK_SIZES[@]}"; do
                    echo "${model_name},${branch},${bsize},0.00,0.00%" >> "$CSV_OUT"
                done
            else
                echo "${model_name},${branch},N/A,0.00,N/A" >> "$CSV_OUT"
            fi
            continue
        fi

        if [[ "$branch" == dspine-* || "$branch" == exp-* ]]; then
            # Drafting Logic: sweep real unclamped DFlash2 draft sizes 4, 5, 6, 7 and 7@p0.55
            for bsize in "${DRAFT_BLOCK_SIZES[@]}"; do
                sleep 4
                echo "     [Run] branch=${branch} model=${model_name} draft_block_size=${bsize}..."
                log_file="${BDIR}/run-${model_name}-b${bsize}.log"
                metrics="$(run_inference_and_parse "$BIN_DIR" "$model_path" "$branch" "$bsize" "$log_file")"
                tps="${metrics%%,*}"
                acc="${metrics#*,}"
                echo "           => ${tps} tok/s (Draft Acceptance: ${acc})"
                echo "${model_name},${branch},${bsize},${tps},${acc}" >> "$CSV_OUT"
            done
        else
            # Standard non-drafting inference
            sleep 4
            echo "     [Run] branch=${branch} model=${model_name} (non-drafting)..."
            log_file="${BDIR}/run-${model_name}-standard.log"
            metrics="$(run_inference_and_parse "$BIN_DIR" "$model_path" "$branch" "N/A" "$log_file")"
            tps="${metrics%%,*}"
            acc="${metrics#*,}"
            echo "           => ${tps} tok/s"
            echo "${model_name},${branch},N/A,${tps},${acc}" >> "$CSV_OUT"
        fi
    done
done

# ==============================================================================
# 5 & 6. Parse CSV and Print Final VERDICT
# ==============================================================================
echo ""
echo "=================================================================="
echo " Benchmark Results Saved to: $CSV_OUT"
echo "=================================================================="
cat "$CSV_OUT"
echo ""

python3 - "$CSV_OUT" <<'PY'
import csv, sys
from collections import defaultdict

csv_path = sys.argv[1]
by_model = defaultdict(list)

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            tps = float(row["Tokens-per-Second"])
        except (ValueError, KeyError):
            tps = 0.0
        by_model[row["Model Name"]].append({
            "branch": row["Branch Name"],
            "block_size": row["Block Size"],
            "tps": tps,
            "acc": row["Draft Acceptance Rate"],
        })

print("==================================================================")
for model_name, rows in by_model.items():
    baseline_rows = [r for r in rows if r["branch"] == "main"]
    baseline_tps = baseline_rows[0]["tps"] if baseline_rows else 0.0
    winner = max(rows, key=lambda r: r["tps"])

    if baseline_tps > 0.0:
        margin_pct = ((winner["tps"] - baseline_tps) / baseline_tps) * 100.0
        speedup_x = winner["tps"] / baseline_tps
        margin_str = f"{margin_pct:+.2f}% ({speedup_x:.2f}x vs main baseline @ {baseline_tps:.2f} tok/s)"
    else:
        margin_str = f"N/A (main baseline was {baseline_tps:.2f} tok/s)"

    block_desc = f", Block Size={winner['block_size']}, Acceptance={winner['acc']}" if winner["block_size"] != "N/A" else ""
    print(
        f"VERDICT [{model_name}]: Winner is '{winner['branch']}'{block_desc} "
        f"at {winner['tps']:.2f} tok/s -- Margin: {margin_str}"
    )
print("==================================================================")
PY
exit 0
}
