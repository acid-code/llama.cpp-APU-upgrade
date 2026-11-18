# llama.cpp APU Upgrade & DFlash2 Speculative Decoding Suite

**Target Hardware:** Bare-Metal Ubuntu Server (`mini-pc` - GT105)
- **APU:** AMD Ryzen 7 8745HS (8 Cores / 16 Threads, Zen 4 with AVX-512 VNNI & BF16)
- **iGPU:** AMD Radeon 780M (`gfx1103`, 12 RDNA3 Compute Units, ROCm target `gfx1100` + `HSA_OVERRIDE_GFX_VERSION=11.0.0`)
- **Memory:** 48 GB Unified Dual-Channel DDR5 (`1 GiB` BIOS UMA Frame Buffer + `45 GiB` Linux System RAM / `40.00 GiB` Unified TTM GTT Pool)
- **Storage:** 1 TB PCIe Gen4 NVMe SSD (`/dev/nvme0n1p2`, ~842 GiB free)

---

## 1. Executive Summary & Recommended `local-agent` Setup

For daily 24/7 operation of `local-agent` (accessed via Tailscale from laptop/phone), **run `Qwen3.6-35B-A3B-UD-Q4_K_M` paired with its 545 MB `DFlash2` draft sidecar (`qwen36-35b-a3b-dflash-Q8_0.gguf`) on branch `dspine-v2-custom`**, alongside a lightweight 1.5B-3B router/embedding model.

### Why `Qwen3.6-35B-A3B` + `DFlash2` Wins Everything:
1. **Faster AND Smarter than `gpt-oss-20b`:**
   - Without the DFlash2 sidecar, `Qwen3.6-35B-A3B` decodes at **18.80 tok/s** (slower than `gpt-oss-20b` at **24.70 tok/s**).
   - With the **545 MB DFlash2 sidecar** (`--spec-draft-n-max 12 --spec-draft-n-min 1`), `Qwen3.6-35B-A3B` jumps to **27.8 - 28.9 tok/s** sustained (and **32.3 - 36.8 tok/s** on 128-token reasoning/coding bursts, with **43.7% to 81.3% draft acceptance**).
   - You get 35B MoE intelligence at **1.5x - 2.0x faster generation** and **158 - 165 tok/s prompt prefill**.
2. **Plenty of Unified RAM Headroom (45 GiB Usable RAM):**
   - `Qwen3.6-35B-A3B` (`20.6 GiB`) + `DFlash2 Sidecar` (`0.55 GiB`) + `16K-32K KV Cache` (`~1.5 GiB`) = **~22.6 GiB**.
   - Adding a fast **1.5B-3B Router / Embedding model** (`~1.5 - 2.0 GiB`) + Ubuntu / Docker / Tailscale (`~1.5 GiB`) uses only **~25.6 GiB out of 45 GiB**, leaving **~19.4 GiB of free RAM**.

---

## 2. Models & File Paths on `mini-pc`

All model files reside in `/home/acid/projects/llama.cpp-APU-upgrade/models/`:

| Role | Model Name / Hugging Face Repo | Exact File Path on `mini-pc` | Disk Size | GTT / RAM Footprint |
| :--- | :--- | :--- | :---: | :---: |
| **Primary MoE LLM (35B, 3B Active)** | `unsloth/Qwen3.6-35B-A3B-GGUF` (`UD-Q4_K_M`) | `/home/acid/projects/llama.cpp-APU-upgrade/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | `20.2 GiB` | `~21.1 GiB` |
| **DFlash2 Neural Draft Sidecar** | `aminya/Qwen3.6-35B-A3B-DFlash2-GGUF` (`Q8_0`, arch: `dflash`) | `/home/acid/projects/llama.cpp-APU-upgrade/models/qwen36-35b-a3b-dflash-Q8_0.gguf` | `545 MiB` | `~550 MiB` |
| **Secondary / Comparison LLM (20B)** | `gpt-oss-20b-Q4_K_M` (`openai-moe`, 32 experts) | `/home/acid/projects/llama.cpp-APU-upgrade/models/gpt-oss-20b-Q4_K_M.gguf` | `11.3 GiB` | `~12.2 GiB` |
| **Fast Agent Router / Classifier (Optional)** | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` or `Qwen3-1.7B-Q8_0.gguf` | `/home/acid/projects/llama.cpp-APU-upgrade/models/qwen-router-Q8_0.gguf` | `~1.6 GiB` | `~1.8 GiB` |

### How to Re-Download the DFlash2 Sidecar (If Needed):
> **Important:** Do **not** use the older July 2026 `ridd1er1/Qwen3.6-35B-A3B-DFlash-GGUF` file (it uses the deprecated `"dflash-draft"` architecture string and fails to load). Always use `aminya/Qwen3.6-35B-A3B-DFlash2-GGUF`, which has `"architecture": "dflash"`:
```bash
curl -L --fail --progress-bar \
  -o /home/acid/projects/llama.cpp-APU-upgrade/models/qwen36-35b-a3b-dflash-Q8_0.gguf \
  https://huggingface.co/aminya/Qwen3.6-35B-A3B-DFlash2-GGUF/resolve/main/Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf
```

---

## 3. The 5 Architectural Branches

| Branch | Description | When to Use |
| :--- | :--- | :--- |
| **`main`** | Clean upstream `llama.cpp` baseline (`a691b2345`). | Reference baseline only. |
| **`v1-custom`** | Injects V1 `ggml-hip-ssd` (4096-byte page-aligned zero-copy GTT memory arena via `hipHostMallocMapped`). | Testing basic zero-copy GTT host allocations without speculative decoding. |
| **`v2-custom`** | Injects V2 `ggml-hip-ssd` (zero-copy GTT arena + Linux `io_uring` async prefetcher + closed 12-thread CPU worker pool + Dense Model Sleep logic). | Non-drafting MoE or dense inference with automatic core reservation (`12` compute threads on cores `0..11`, `4` threads on cores `12..15` reserved for OS/Tailscale/`io_uring`). |
| **`dspine-baseline`** | Clean pull of upstream `origin/dspine` (`d493f841b` - DFlash/DSpine block-diffusion speculative decoding). | Baseline DFlash2 speculative decoding without custom V2 APU hooks. |
| **`dspine-v2-custom`** | **Recommended Production Branch.** Merges `dspine-baseline` + `v2-custom` (`ggml-hip-ssd` V2). | **All `local-agent` production workloads** (`Qwen3.6-35B-A3B` + DFlash2 sidecar, `gpt-oss-20b`, and oversized >45 GB MoE SSD streaming). |

---

## 4. Complete 5-Branch Benchmark Results (`benchmark_results.csv`)

Measured on `mini-pc` (512-token technical prompt prefill + 256-token deterministic generation, `--temp 0.0 --seed 42 -fit off -lm none -fa on -ub 64 -b 512 -t 12`):

| Model Name | Branch Name | Draft Block Size | Decode (`tok/s`) | Prompt Prefill (`tok/s`) | Draft Acceptance Rate | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `gpt-oss-20b-Q4_K_M` | `main` | N/A | **24.70** | ~195.0 | N/A | 100% Radeon 780M |
| `gpt-oss-20b-Q4_K_M` | `v1-custom` | N/A | **24.70** | ~195.0 | N/A | Zero-copy GTT host arena |
| `gpt-oss-20b-Q4_K_M` | `v2-custom` | N/A | **24.60** | ~195.0 | N/A | Verified `DENSE_IGPU_100PCT` (CPU pool dormant) |
| `gpt-oss-20b-Q4_K_M` | `dspine-baseline` | 8 / 12 / 16 / 20 | **24.41 / 24.40 / 24.41 / 24.44** | ~195.0 | `12.05%` | `ngram-simple` self-speculation |
| `gpt-oss-20b-Q4_K_M` | `dspine-v2-custom` | 8 / 12 / 16 / 20 | **24.43 / 24.19 / 24.45 / 24.42** | ~195.0 | `12.05%` | `ngram-simple` + V2 Dense Sleep |
| `Qwen3.6-35B-A3B-UD-Q4_K_M` | `main` | N/A | **18.80** | ~165.0 | N/A | Baseline without sidecar |
| `Qwen3.6-35B-A3B-UD-Q4_K_M` | `v1-custom` | N/A | **18.30** | ~165.0 | N/A | V1 GTT without sidecar |
| `Qwen3.6-35B-A3B-UD-Q4_K_M` | `v2-custom` | N/A | **18.10** | ~165.0 | N/A | V2 GTT without sidecar |
| `Qwen3.6-35B-A3B-UD-Q4_K_M` | `dspine-baseline` | **8** | **27.77** *(peak 28.37)* | **165.2** | **`43.71%`** | DFlash2 sidecar (`+47.7%` vs `main`) |
| `Qwen3.6-35B-A3B-UD-Q4_K_M` | `dspine-baseline` | **12** | **27.78** | **165.2** | **`43.71%`** | DFlash2 sidecar (`+47.8%` vs `main`) |
| `Qwen3.6-35B-A3B-UD-Q4_K_M` | `dspine-baseline` | **16** | **27.37** *(36.6 @ 128 tok)* | **165.2** | **`43.71%`** *(81.3% @ 32 tok)* | DFlash2 sidecar |
| `Qwen3.6-35B-A3B-UD-Q4_K_M` | `dspine-baseline` | **20** | **27.49** | **165.2** | **`43.71%`** | DFlash2 sidecar |
| `Qwen3.6-35B-A3B-UD-Q4_K_M` | `dspine-v2-custom` | **8** | **27.48** *(peak 28.88)* | **165.2** | **`43.71%`** | DFlash2 + V2 (`+53.6%` peak vs `main`) |
| `Qwen3.6-35B-A3B-UD-Q4_K_M` | `dspine-v2-custom` | **12** | **27.60** | **165.2** | **`43.71%`** | **Recommended Sweet Spot** |
| `Qwen3.6-35B-A3B-UD-Q4_K_M` | `dspine-v2-custom` | **16** | **27.35** *(36.8 @ 128 tok)* | **165.2** | **`43.71%`** *(81.3% @ 32 tok)* | Highest burst speed on reasoning/code |
| `Qwen3.6-35B-A3B-UD-Q4_K_M` | `dspine-v2-custom` | **20** | **27.50** | **165.2** | **`43.71%`** | DFlash2 + V2 |

### Architectural Comparison: 100% iGPU vs Hybrid CPU+iGPU on Models That Fit in GTT
We also tested splitting 16 out of 40 MoE layers of `Qwen3.6-35B-A3B` onto the Zen 4 AVX-512 CPU (`7,956 MiB` in V2 zero-copy GTT host memory) while keeping Attention + 24 MoE layers + DFlash2 on the Radeon 780M:
- **100% Radeon 780M (`-ngl 999 -ngld 999 -lm none -fa on -ub 64`)**: **165.2 tok/s** Prompt | **36.6 tok/s** Decode (128-token test)
- **Hybrid Split (16 MoE layers on Zen 4 CPU + 24 on 780M iGPU)**: **41.0 tok/s** Prompt | **15.5 tok/s** Decode
- **Conclusion:** Because the Zen 4 CPU and Radeon 780M share the same dual-channel DDR5 memory controller (~75 GB/s), splitting layers onto the CPU when the model fits inside the 40 GiB GTT pool adds 32 CPU<->GPU `hipStreamSynchronize` barriers per token without adding memory bandwidth. **Always keep 100% of layers in `ROCm0` (`-ngl 999 -ngld 999`) for models <= 38 GiB**, and reserve CPU/GTT hybrid streaming for oversized MoE models (>45 GiB) that exceed physical RAM.

### Oversized Out-of-Core SSD Offloading Tests (>43 GiB Models)
We benchmarked all three oversized MoE models on `mini-pc` (`45 GiB` RAM + PCIe Gen4 NVMe SSD) using `dspine-v2-custom` (`-lm mmap -fit off -ub 64 -b 64 -t 12 -ngl 0`, `HIP_VISIBLE_DEVICES=-1`):

| Model | Params (Total / Active) | Disk Size | Key Flags | Prompt (`tok/s`) | Decode (`tok/s`) | Peak RSS | SSD Major Page Faults |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **`GLM-4.5-Air-UD-IQ3_XXS`** | 106B / 12B | `47.9 GiB` | `-c 2048` | **13.1** | **3.6** | `43.59 GiB` | `850,627` |
| **`Llama-4-Scout-Q3_K_S`** | 109B / 17B | `43.5 GiB` | `-c 4096 -t 16` | **12.5** | **1.8** | `25.86 GiB` | `426,327` |
| **`Qwen3-235B-A22B-UD-Q2_K_XL`** | 235B / 22B | `82.7 GiB` | `--no-repack -c 2048` | **1.6** | **0.7** | `22.76 GiB` | `519,245` |

- **Key Findings:**
  1. **`--no-repack` is mandatory for >45 GiB models on Zen 4:** Without `--no-repack`, `llama.cpp`'s `CPU_REPACK` tries to `malloc` an extra 83 GiB repack buffer in RAM and fails to load `Qwen3-235B-A22B`.
  2. **Why Hybrid `--n-cpu-moe 999` failed on `ROCm0`:** Splitting Attention on `ROCm0` and MoE experts on `CPU` bounces activations between `ROCm0` and `ROCm_Host` (`userptr`) 45-94 times per token, triggering Linux KFD `svm_range_restore_work` page-table thrashing and `MES` GPU resets on `gfx1103`.
  3. **Takeaway for `local-agent`:** All 3 oversized models run without crashing, **but** at `0.7 - 3.6 tok/s` decode and `1.6 - 13.1 tok/s` prefill (vs `27.8 - 36.8 tok/s` decode / `165.2 tok/s` prefill on `Qwen3.6-35B-A3B` + `DFlash2`). Reserve `GLM-4.5-Air` (`3.6 tok/s`) or `Qwen3-235B` (`0.7 tok/s`) strictly for rare overnight/offline deep-reasoning jobs—use **`Qwen3.6-35B-A3B` + `DFlash2`** for all interactive `local-agent` work.

---

## 5. Production Launch Commands & Required Flags (`local-agent`)

### A. Primary Brain: `Qwen3.6-35B-A3B` + `DFlash2` Sidecar (Port `8080`)
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0

/home/acid/projects/llama.cpp-APU-upgrade/build-dspine-v2/bin/llama-server \
  -m /home/acid/projects/llama.cpp-APU-upgrade/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  -md /home/acid/projects/llama.cpp-APU-upgrade/models/qwen36-35b-a3b-dflash-Q8_0.gguf \
  -ngl 999 -ngld 999 \
  --spec-draft-n-max 12 --spec-draft-n-min 1 \
  -fit off -lm none -fa on -ub 64 -b 512 -t 12 -c 16384 \
  --host 0.0.0.0 --port 8080
```

### B. Optional Fast Router / Sub-Agent: Small 1.5B-3B Model (Port `8081`, Running Concurrently)
If `local-agent` runs a small router/intent-classifier alongside `Qwen3.6-35B-A3B`:
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0

/home/acid/projects/llama.cpp-APU-upgrade/build-dspine-v2/bin/llama-server \
  -m /home/acid/projects/llama.cpp-APU-upgrade/models/qwen-router-Q8_0.gguf \
  -ngl 999 \
  -fit off -lm none -fa on -ub 128 -b 512 -t 4 -c 8192 \
  --host 0.0.0.0 --port 8081
```

### C. Alternative Model: `gpt-oss-20b-Q4_K_M` (Port `8080`)
If switching the main server to `gpt-oss-20b-Q4_K_M`:
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0

/home/acid/projects/llama.cpp-APU-upgrade/build-dspine-v2/bin/llama-server \
  -m /home/acid/projects/llama.cpp-APU-upgrade/models/gpt-oss-20b-Q4_K_M.gguf \
  -ngl 999 \
  -fit off -lm none -fa on -ub 128 -b 512 -t 12 -c 16384 \
  --host 0.0.0.0 --port 8080
```

---

## 6. Why Every Flag in the Launch Command Is Mandatory on Radeon 780M

| Flag | What It Does & Why It Is Required on `gfx1103` |
| :--- | :--- |
| **`-lm none`** (`--load-mode none`) | **Prevents Double-Allocation in RAM.** Default `mmap` caches the 21 GiB GGUF in Linux's filesystem page cache AND allocates 21 GiB in TTM GTT (`42 GiB` total), causing Linux swap thrashing and `amdgpu_amdkfd_restore_userptr_worker` hangs. `-lm none` loads tensors once directly into GTT (`~21.8 GiB` total). |
| **`-fit off`** | **Disables Startup Dummy Context Probe.** By default, `llama.cpp` allocates and destroys a temporary 21 GiB GPU context at boot to probe free memory before allocating the real context. Turning `-fit off` stops back-to-back 21 GiB KFD queue teardowns. |
| **`-fa on`** | **Enables RDNA3 FlashAttention.** Dramatically reduces KV-cache memory size and speeds up both prompt prefill (`165 tok/s`) and DFlash2 parallel token verification. |
| **`-ub 64`** | **Caps Physical Micro-Batch Size at 64.** In `Qwen3.6-35B-A3B` (256 experts, 8 active/token), the default `-ub 512` launches a single `MUL_MAT_ID` GPU kernel with `512 * 8 = 4,096` non-contiguous expert gathers across 500 MB per layer, tripping the Radeon 780M (12 CUs) MES queue watchdog (`GPU reset`). `-ub 64` keeps every kernel fast and 100% stable on long agent prompts. |
| **`--spec-draft-n-max 12 --spec-draft-n-min 1`** | **DFlash2 Draft Block Size.** Drafts 12 tokens at once with the 545 MB sidecar and verifies them in a single forward pass of the 35B MoE model (`28 - 36.8 tok/s`). Use `12` for steady long outputs or `16` for short/medium reasoning bursts. |
| **`-t 12`** | **Core Reservation Safeguard.** Pairs with `dspine-v2-custom`'s closed 12-thread pool (`cores 0..11`), keeping 4 hardware threads (`cores 12..15`) dedicated to Ubuntu OS, Tailscale, Docker, and `local-agent`. |

---

## 7. Bare-Metal BIOS, GRUB & ROCm Configuration Reference

If you ever reinstall Ubuntu or migrate to a new APU server, apply these exact system settings:

### 1. BIOS Settings (AMI BIOS -> `Advanced` -> `AMD CBS` -> `NBIO Common Options` -> `GFX Configuration`)
- **iGPU Configuration:** `UMA_SPECIFIED`
- **UMA Frame Buffer Size:** `1G` (leaves **45 GiB** of physical RAM visible to Linux instead of carving out 12 GiB as static VRAM)

### 2. GRUB Kernel Parameters (`/etc/default/grub`)
Allocates a **40 GiB Unified GTT Pool** (`10485760` pages * 4096 bytes) and disables the 1 GiB `crashkernel` reservation:
```bash
sudo sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT=.*/GRUB_CMDLINE_LINUX_DEFAULT="ttm.pages_limit=10485760 ttm.page_pool_size=10485760 amdgpu.gttsize=40960 crashkernel=no"/' /etc/default/grub
if [ -f /etc/default/grub.d/kdump-tools.cfg ]; then
    sudo sed -i 's/crashkernel=[^"]*/crashkernel=no/g' /etc/default/grub.d/kdump-tools.cfg
fi
sudo update-grub
```

Verify after reboot:
```bash
free -h                                              # Should show Mem: 45Gi total, ~44Gi available
cat /sys/module/ttm/parameters/pages_limit           # Should print 10485760 (40 GiB)
cat /sys/class/drm/card*/device/mem_info_vram_total  # Should print 1073741824 (1 GiB)
cat /sys/class/drm/card*/device/mem_info_gtt_total   # Should print 42949672960 (40 GiB)
```

### 3. `/dev/kfd` Permissions & ROCm Build Flags
Ensure your user is in the `render` and `video` groups:
```bash
sudo usermod -aG render,video "$USER"
```
When compiling `llama.cpp` with ROCm on Ubuntu where `rocBLAS` provides `gfx1100` Tensile libraries:
```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DAMDGPU_TARGETS=gfx1100 \
  -DGPU_TARGETS=gfx1100
cmake --build build --target llama-cli llama-server -j"$(nproc)"
```

---

## 8. Repository Scripts

- **[`init-repos.sh`](init-repos.sh)**: Rebuilds the 5-branch architecture (`main`, `v1-custom`, `v2-custom`, `dspine-baseline`, `dspine-v2-custom`) from `patches/v1` and `patches/v2`.
- **[`validate-memory.sh`](validate-memory.sh)**: Compiles `v2-custom` and `dspine-v2-custom` with AddressSanitizer (`-fsanitize=address -fno-omit-frame-pointer -g -O1`) and runs `cppcheck`, `test-backend-ops -o MUL_MAT_ID`, and 16-token ASan smoke tests on both `gpt-oss-20b` (verifying `DENSE_IGPU_100PCT` CPU pool sleep) and `Qwen3.6-35B-A3B` (verifying `MOE_HYBRID_ENGAGED`).
- **[`bench-inferssd.sh`](bench-inferssd.sh)**: Automated 5-branch benchmark suite that verifies the 40 GiB GTT limit, compiles/reuses Release builds, sweeps draft block sizes `8, 12, 16, 20`, outputs [`benchmark_results.csv`](benchmark_results.csv), and prints the final `VERDICT`.

---
---

# Upstream `llama.cpp` Overview

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)
