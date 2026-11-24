# llama.cpp APU Upgrade & DFlash2 Speculative Decoding Suite

**Target Hardware:** Bare-Metal Ubuntu Server (`mini-pc` - GT105)
- **APU:** AMD Ryzen 7 8745HS (8 Cores / 16 Threads, Zen 4 with AVX-512 VNNI & BF16)
- **iGPU:** AMD Radeon 780M (`gfx1103`, 12 RDNA3 Compute Units, ROCm target `gfx1100` + `HSA_OVERRIDE_GFX_VERSION=11.0.0`)
- **Memory:** 48 GB Unified Dual-Channel DDR5 (`1 GiB` BIOS UMA Frame Buffer + `45 GiB` Linux System RAM / `40.00 GiB` Unified TTM GTT Pool)
- **Storage:** 1 TB PCIe Gen4 NVMe SSD (`/dev/nvme0n1p2`, ~842 GiB free)

---

## 1. Executive Summary & Recommended `local-agent` Setup

For daily 24/7 operation of `local-agent` (accessed via Tailscale from laptop/phone), **run `Qwen3.6-35B-A3B-UD-Q4_K_M` paired with its 545 MB `DFlash2` draft sidecar (`qwen36-35b-a3b-dflash-Q8_0.gguf`) on branch `exp-coalesced-dflash`** (built on `dspine-v2-custom` with the multi-size `gf_res_prev[10]` HIP graph cache and `--spec-draft-p-min 0.55`), alongside a lightweight 1.5B-3B router/embedding model.

### Why `Qwen3.6-35B-A3B` + `DFlash2` Wins Everything:
1. **Faster AND Smarter than `gpt-oss-20b`:**
   - Without the DFlash2 sidecar, `Qwen3.6-35B-A3B` decodes at **19.26 tok/s** (5-prompt average, slower than `gpt-oss-20b` at **24.70 tok/s**).
   - With the **545 MB DFlash2 sidecar** (`--spec-draft-n-max 7 --spec-draft-n-min 1 --spec-draft-p-min 0.55`) + multi-size HIP graph caching (`gf_res_prev[10]`), `Qwen3.6-35B-A3B` jumps to **37.28 tok/s** 5-prompt average (**+93.6%** over non-speculative, **+12.3%** over the earlier `p_min=0.0` baseline at `33.20 tok/s`), peaking at **49.2 tok/s** on structured reasoning bursts with **71.1% average draft acceptance** (100% lossless).
   - You get 35B MoE intelligence at **1.9x faster generation** and **105 - 196 tok/s prompt prefill**.
2. **Plenty of Unified RAM Headroom (45 GiB Usable RAM):**
   - `Qwen3.6-35B-A3B` (`20.6 GiB`) + `DFlash2 Sidecar` (`0.55 GiB`) + `16K-32K KV Cache` (`~1.5 GiB`) = **~22.6 GiB**.
   - Adding a fast **1.5B-3B Router / Embedding model** (`~1.5 - 2.0 GiB`) + Ubuntu / Docker / Tailscale (`~1.5 GiB`) uses only **~25.6 GiB out of 45 GiB**, leaving **~19.4 GiB of free RAM**.

---

## 2. Models & File Paths on `mini-pc`

All model files reside in `/home/acid/projects/llama.cpp-APU-upgrade/models/`:

| Role | Model Name / Hugging Face Repo | Exact File Path on `mini-pc` | Disk Size | GTT / RAM Footprint |
| :--- | :--- | :--- | :---: | :---: |
| **Primary MoE LLM (35B, 3B Active)** | `unsloth/Qwen3.6-35B-A3B-GGUF` (`UD-Q4_K_M`) | `/home/acid/projects/llama.cpp-APU-upgrade/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | `20.2 GiB` | `~21.1 GiB` |
| **Secondary Qwen3.6 Fallback (35B, 3B Active)** | `HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive` (`Q4_K_M` + `mmproj-f16`) | `/home/acid/projects/llama.cpp-APU-upgrade/models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf` | `19.7 GiB` + `858 MiB` | `~22.5 GiB` *(32K ctx)* |
| **DFlash2 Neural Draft Sidecar** | `aminya/Qwen3.6-35B-A3B-DFlash2-GGUF` (`Q8_0`, arch: `dflash`) | `/home/acid/projects/llama.cpp-APU-upgrade/models/qwen36-35b-a3b-dflash-Q8_0.gguf` | `545 MiB` | `~550 MiB` |
| **Flint Coding MoE (39.8B, 2.5B Active)** | `mradermacher/Flint-40B-A2.5B-abliterated-i1-GGUF` (`i1-IQ4_XS`) | `/home/acid/projects/llama.cpp-APU-upgrade/models/flint-abliterated/mrad-i1-IQ4_XS.gguf` | `19.9 GiB` | `~21.3 GiB` *(32K ctx)* |
| **Flint DFlash2 Neural Draft Sidecar** | `z-lab/Flint-DFlash2` (`Q8_0`, arch: `dflash`) | `/home/acid/projects/llama.cpp-APU-upgrade/models/Flint-DFlash2-Q8_0.gguf` | `414 MiB` | `~450 MiB` |
| **Inkling 2 Mini Mainline (26.4B)** | `unsloth/Inkling-2-Mini-GGUF` (`UD-Q4_K_XL`) | `/home/acid/projects/llama.cpp-APU-upgrade/models/Inkling-2-Mini-UD-Q4_K_XL.gguf` | `17 GiB` | `~18 GiB` |
| **Inkling 2 Mini DSpark Drafter** | `unsloth/Inkling-2-Mini-GGUF` (`Q8_0`, arch: `dflash`) | `/home/acid/projects/llama.cpp-APU-upgrade/models/dspark-Inkling-2-Mini-Q8_0.gguf` | `384 MiB` | `~400 MiB` |
| **Inkling 2 Mini Uncensored (26.4B)** | `AtomicChat/Inkling-2-Mini-Uncensored-GGUF` (`AD-Q4_K`) | `/home/acid/projects/llama.cpp-APU-upgrade/models/Inkling-2-Mini-Uncensored-AD-Q4_K.gguf` | `16 GiB` | `~17 GiB` |
| **Secondary / Comparison LLM (20B)** | `gpt-oss-20b-Q4_K_M` (`openai-moe`, 32 experts) | `/home/acid/projects/llama.cpp-APU-upgrade/models/gpt-oss-20b-Q4_K_M.gguf` | `11.3 GiB` | `~12.2 GiB` |
| **Fast Agent Router / Classifier (Optional)** | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` or `Qwen3-1.7B-Q8_0.gguf` | `/home/acid/projects/llama.cpp-APU-upgrade/models/qwen-router-Q8_0.gguf` | `~1.6 GiB` | `~1.8 GiB` |

### How to Re-Download the DFlash2 Sidecar (If Needed):
> **Important:** Do **not** use the older July 2026 `ridd1er1/Qwen3.6-35B-A3B-DFlash-GGUF` file (it uses the deprecated `"dflash-draft"` architecture string and fails to load). Always use `aminya/Qwen3.6-35B-A3B-DFlash2-GGUF`, which has `"architecture": "dflash"`:
```bash
curl -L --fail --progress-bar \
  -o /home/acid/projects/llama.cpp-APU-upgrade/models/qwen36-35b-a3b-dflash-Q8_0.gguf \
  https://huggingface.co/aminya/Qwen3.6-35B-A3B-DFlash2-GGUF/resolve/main/Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf
```

### Inkling 2 Mini Comparison Benchmark

Use [`bench-inkling-dspark.py`](bench-inkling-dspark.py) to compare Qwen3.6-35B-A3B, Inkling 2 Mini, and the uncensored Inkling 2 Mini model. The quick pass is 128 output tokens over three prompts; the full pass is 256 output tokens over five prompts:

```bash
python3 bench-inkling-dspark.py quick
python3 bench-inkling-dspark.py full
```

The benchmark preserves the successful APU setup: full iGPU offload, native F16 KV, `-ub 64`, `-b 512`, `-t 12`, no approximate-routing hooks, and the same 128-token or 256-token generation targets used in the earlier DFlash2 comparisons. Qwen uses DFlash2 with `p_min=0.55` and tests `n_max=5/7`. The Inkling DSpark sidecar has `dflash.block_size=8` and `dflash.has_confidence_head=False`, so both Inkling variants use `p_min=0.00`; the benchmark starts with the documented `n_max=3` setting and ablates `n_max=5/7/8`.

The root GGUF layout uses the looped `inkling2`/`inkling2_loop` implementation and requires looped-layer plus per-expert NVFP4-scale support. The supported DSpark path is Unsloth's llama.cpp fork/prebuilt build. Set `LLAMA_SERVER_BIN` to that build's `llama-server` before running:

```bash
QWEN_SERVER_BIN=/path/to/llama.cpp/build-dspine-v2/bin/llama-server \
INKLING_SERVER_BIN=/path/to/unsloth-llama.cpp/build/bin/llama-server \
LD_LIBRARY_PATH=/path/to/unsloth-llama.cpp/build/lib \
  python3 bench-inkling-dspark.py quick
```

For mainline llama.cpp, use PR 32704 and the `Mainline_Support` GGUF variant. PR 32218 adds DSpark, but the official mainline-support guidance says that it predates and is not tested with the `inkling2_loop` layout. The benchmark therefore uses DSpark only with an Unsloth build and the root GGUF layout.

---

### 2026-11-22 uncensored Inkling DSpark validation

A separate validation build was created outside this repository at `/home/acid/projects/llama-inkling-main`. It combines upstream PR 32704 and PR 32218 with `merge --squash` on a non-committing `inkling-dspark-validation` branch. The validation binary is `/home/acid/projects/llama-inkling-main/build/bin/llama-server`.

AtomicChat's uncensored GGUF loaded with this build and the official uncensored F16 mmproj. The server logged `set_token_state: state size mismatch, got 2304 expected 6912`, so these DSpark measurements need a production fix before they are declared lossless or reliable. The uncensored model is now known to load and to produce DSpark acceptance, but the state mismatch remains a correctness risk.

Measured with 128 generated tokens, coding and reasoning prompts, `-ngl 999 -ngld 999 -ub 64 -b 512 -t 12 -c 4096`, `--spec-draft-p-min 0.00`, and no other draft threshold:

| Uncensored Inkling mode | Coding tok/s | Reasoning tok/s | Coding acceptance | Reasoning acceptance |
| :--- | ---: | ---: | ---: | ---: |
| No drafter, fresh mainline b12628 | `8.71` | `8.69` | N/A | N/A |
| DSpark n1 p0.00 | `11.06` | `10.79` | `65.8%` | `61.5%` |
| DSpark n2 p0.00 | `10.07` | `9.96` | `49.6%` | `48.8%` |
| DSpark n3 p0.00 | `8.78` | `8.57` | `38.9%` | `37.1%` |

Best result: `n1` improved the uncensored fallback from about `8.69-8.71` tok/s to `10.79-11.06` tok/s in this two-prompt smoke test.

**2026-11-23 fix & unified mainline validation (`bench-unified-models.py`):**
- **Root cause of `2304 / 6912` fixed:** `common_batch_set_zero_state()` in `/home/acid/projects/llama-inkling-main/common/common.cpp` (startup warmup & `seq_rm` probe only) hardcoded `llama_model_n_embd_out(model)` (`2304`) instead of querying `n_embd_state` (`3 * 2304 = 6912` for `dspark_context_rank`). Exposed `llama_n_embd_state(ctx)` and updated `common_batch_set_zero_state()`, eliminating all `state size mismatch` errors (`0` across all 9 runs).
- **Unified mainline build:** Ported `gf_res_prev[10]`, RDNA3 `gfx1103` host-sync on `ev_layer_inp_read` + `ev_h_nextn_read` / prefill ubatch flush, and opt-in `llama-approx-routing.{h,cpp}` onto `/home/acid/projects/llama-inkling-main` (`v1.0.0-dev` / `b12629` + PR 32704 + PR 32218).
- **Unified 3-prompt benchmark (`bench_unified_quick.csv`, 128 tok, `-ngl 999 -fit off -lm none -fa on -ub 64 -b 512 -t 12 -c 4096 -np 1`, `0` GPU resets, `0/0` graph reallocs, `0` state mismatches):**

| Unified Mainline `b12629` Config | Avg PP (`tok/s`) | Avg Decode (`tok/s`) | DFlash/DSpark Acc | Mean Len | GTT (`GiB`) | Notes |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **`Qwen3.6-35B-A3B Q4_K_M` + `DFlash2 n7 p0.55`** | `114.5` | **`38.98`** | `76.2%` | `4.28` | `21.5` | **#1 daily driver** (`40.33 tok/s` reasoning, `0/0` reallocs) |
| **`Flint 39.8B-A2.5B UD-Q4_K_XL` (no draft)** | `111.0` | `22.45` | N/A | `1.00` | `23.4` | Fastest raw MoE (2.16x Inkling Q4 raw, +17% over Qwen3.6 raw) |
| `Flint 39.8B-A2.5B UD-Q4_K_XL` + `Flint-DFlash2 n7 p0.00` | `99.3` | `21.88` | `26.8%` | `2.77` | `23.9` | Naive `p_min=0` slower than raw on DDR5 (`-2.5%`) |
| **`Flint 39.8B-A2.5B UD-Q4_K_XL` + `Flint-DFlash2 n7 p0.60`** | `109.7` | **`33.51`** | `62.8%` | `2.87` | `23.9` | **+49.3% vs raw (`36.20 tok/s` coding)**, `0/0` reallocs via `gf_res_prev[10]` |
| `Inkling-2-Mini Q4_K_XL` (mainline, no draft) | `44.8` | `10.38` | N/A | `1.00` | `16.6` | +7.1% over Unsloth `b12588` (`9.69`) |
| `Inkling-2-Mini Q4_K_XL` + `DSpark n1` | `45.2` | **`11.23`** | `65.7%` | `1.65` | `17.0` | **+8.2%**, `0` state mismatch |
| `Inkling-2-Mini Q4_K_XL` + `DSpark n2` | `43.2` | `9.82` | `46.6%` | `1.91` | `17.0` | Slower on DDR5 (`193` vs `145 ms/step`) |
| `Inkling-2-Mini Uncensored AD-Q4_K` (no draft) | `58.3` | `9.30` | N/A | `1.00` | `15.5` | Baseline uncensored fallback |
| **`Inkling-2-Mini Uncensored AD-Q4_K` + `DSpark n1`** | `59.7` | **`11.16`** | `61.7%` | `1.60` | `15.9` | **+20.0%**, `0` state mismatch, `2/3` exact SHA match |
| `Inkling-2-Mini Q3_K_XL` (mainline, no draft) | `47.0` | `11.62` | N/A | `1.00` | `13.0` | +11.9% over Q4 raw; **`reasoning_effort="low"` cut wall-clock 3.25x (`45.5 s -> 14.0 s`)** |
| **`Inkling-2-Mini Q3_K_XL` + `DSpark n1`** | `47.0` | **`12.58`** | `64.5%` | `1.63` | `13.4` | **+21.2% over Q4 raw** (`13.18 tok/s` reasoning) |

> **Long-Context Regression Warning (2026-11-24 — see Section 4.2.1):** While `b12629` (`v1.0.0-dev`) passed the 4K-context `bench_unified_quick.csv` sweep above, subsequent long-context testing uncovered a `b12217 -> b12629` regression in `LLAMA_DFLASH_DEV=1` on `gfx1103` that causes all-`NaN` logits at `>=12K` tokens on `Qwen3.6-35B-A3B-UD-Q4_K_M` + `DFlash2` and ~11–14% slower long-context prefill than `b12217`. Keep `:8080` (`Qwen3.6-35B-A3B-UD-Q4_K_M`) on **`b12217` (`build-dspine-v2`, `19e225579`)**.

**2026-11-23 Abliterated Flint (`dima6312/Flint-40B-A2.5B-abliterated`) + `Flint-DFlash2` Quant & Sweet-Spot Sweep (`bench-flint-abliterated.py` / `bench_flint_abliterated.csv`):**
- Both `dima6312` (plain GGUF) and `mradermacher` (`i1` imatrix GGUF) quantize `dima6312/Flint-40B-A2.5B-abliterated` (`MITRE ATT&CK` refusal rate `15% -> 0%`). Tested across 24 configs (`3` prompts x `128` tok, `-c 4096 -ub 64 -np 1`, `0` GPU resets, `0/0` graph reallocs), plus qualitative coherence + non-refusal verification:

| Abliterated Flint Quant | Disk (`GiB`) | GTT (`GiB`) | Raw Decode | Best `Flint-DFlash2` Config | Best Decode | Acc% | Mean Len | Quality Check (`[sec_ctf]` + `[code]`) |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: | :--- |
| `dima-Q4_K_M` (plain) | `22.5` | `23.2` | `25.53` | `n7 p0.65` (`n7 p0.60`: `33.63`) | `33.65` | `68.1%` | `2.31` | Slight clipped CoT (`37.45` code / `29.12` ctf) |
| `mrad-i1-Q4_K_M` (imatrix) | `22.5` | `23.2` | `31.68` | `n6 p0.60` (`n7 p0.60`: `32.89`) | `34.29` | `64.5%` | `2.43` | Slower than `i1-IQ4_XS`, higher GTT |
| **`mrad-i1-IQ4_XS` (imatrix) - WINNER** | **`19.9`** | **`20.6`** | **`33.54`** | **`n6 p0.65`** (`n5 p0.60`: `37.02`) | **`38.10`** | **`68.8%`** | **`2.44`** | **Clean fluent English CoT (`40.49` code / `32.95` ctf; `45.15` binary-search smoke test at 32K ctx)** |
| `mrad-i1-Q3_K_M` (imatrix) | `17.8` | `18.5` | `36.07` | `n7 p0.65` (`n7 p0.70`: `44.72`) | `45.89` | `75.7%` | `3.23` | **Degraded CoT grammar** (`"We need answer user..."`) |

- **2026-11-23 `Qwen3.6-35B-A3B` (`HauhauCS` `Q4_K_M`) + `DFlash2` Sweep (`bench-qwen36-uncensored.py` / `bench_qwen36_uncensored.csv`):**
  - Replaces Inkling and Flint on `:8081` purely on APU throughput and draft efficiency: **`46.68 tok/s`** average decode with `qwen36-35b-a3b-dflash-Q8_0.gguf` (**4.2x faster than Inkling-2-Mini + DSpark at `11.16 tok/s`**, `0/0` graph reallocs, 100% lossless match vs raw, `22.5 GiB` GTT at `-c 32768`):

| `Qwen3.6-35B-A3B` (`Q4_K_M`) Config | Avg PP (`tok/s`) | Avg Decode (`tok/s`) | vs Raw | DFlash2 Acc | Mean Len | GTT (`GiB`) | Notes |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `nodraft` (raw baseline) | `131.9` | `25.08` | `+0.0%` | `0.0%` | `1.00` | `19.85` | `0/0` reallocs |
| **`dflash2_n7_p55` - WINNER** | **`130.9`** | **`46.68`** | **`+86.1%`** | **`82.2%`** | **`5.69`** | **`22.24`** | **4.2x faster than Inkling (`11.16`), peak `49.54 tok/s`, `0/0` reallocs** |
| `dflash2_n7_p60` | `132.3` | `46.42` | `+85.1%` | `83.5%` | `6.21` | `22.11` | Virtually tied with `p0.55` |
| `dflash2_n5_p55` | `133.4` | `44.24` | `+76.4%` | `86.9%` | `5.02` | `21.97` | Higher acc%, slightly shorter burst |
| `dflash2_n7_p50` | `126.7` | `43.69` | `+74.2%` | `81.3%` | `6.29` | `25.87` | Lower threshold hurts speed slightly |

> **Note (`b12217` vs `b12629`, updated 2026-11-24):** The `46.68 tok/s` sweep (`bench_qwen36_uncensored.csv`) originally ran on `b12629` at `-c 4096`. Head-to-head testing in **Section 4.2.1** now confirms `b12217` (`build-dspine-v2`) with `mmproj` + `DFlash2` wins across the board on `Qwen3.6-35B-A3B-Uncensored` too: **`46.75` vs `43.04 tok/s`** at 4K (`+8.6%`), **`+27–36%` faster prefill** at `18.5K–25.6K`, and `26.2%` faster total wall time at `25.6K` (`136.4 s` vs `184.9 s`), with working `mmproj` vision and zero `NaN`s.

Secondary fallback command (`:8081`, `Qwen3.6-35B-A3B` `Q4_K_M` + `DFlash2`):

```bash
LD_LIBRARY_PATH=/opt/llama-server/uncensored-bin \
HSA_OVERRIDE_GFX_VERSION=11.0.0 \
HIP_VISIBLE_DEVICES=0 \
/opt/llama-server/uncensored-bin/llama-server \
  -m /opt/llama-server/models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf \
  --mmproj /opt/llama-server/models/mmproj-Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf \
  -md /opt/llama-server/models/qwen36-35b-a3b-dflash-Q8_0.gguf \
  --spec-draft-n-max 7 --spec-draft-n-min 1 --spec-draft-p-min 0.55 \
  -ngl 999 -ngld 999 -fit off -lm none -fa on -ub 64 -b 512 -t 12 -c 32768 -np 1 \
  -a qwen3.6-35b-a3b-uncensored --jinja --host 0.0.0.0 --port 8081
```

Legacy Inkling fallback command:

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 \
HIP_VISIBLE_DEVICES=0 \
/home/acid/projects/llama-inkling-main/build/bin/llama-server \
  -m /home/acid/projects/llama.cpp-APU-upgrade/models/Inkling-2-Mini-Uncensored-AD-Q4_K.gguf \
  --mmproj /home/acid/projects/llama.cpp-APU-upgrade/models/mmproj-Inkling-2-Mini-Uncensored-F16.gguf \
  -ngl 999 -fit off -lm none -fa on -ub 64 -b 512 -t 12 -c 32768 \
  --jinja --host 0.0.0.0 --port 8081
```

Experimental uncensored DSpark command:

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 \
HIP_VISIBLE_DEVICES=0 \
/home/acid/projects/llama-inkling-main/build/bin/llama-server \
  -m /home/acid/projects/llama.cpp-APU-upgrade/models/Inkling-2-Mini-Uncensored-AD-Q4_K.gguf \
  --mmproj /home/acid/projects/llama.cpp-APU-upgrade/models/mmproj-Inkling-2-Mini-Uncensored-F16.gguf \
  -md /home/acid/projects/llama.cpp-APU-upgrade/models/dspark-Inkling-2-Mini-Q8_0.gguf \
  --spec-type draft-dspark --spec-draft-n-max 1 --spec-draft-n-min 1 --spec-draft-p-min 0.00 \
  -ngl 999 -ngld 999 -fit off -lm none -fa on -ub 64 -b 512 -t 12 -c 32768 \
  --jinja --host 0.0.0.0 --port 8081
```

Copy-ready prompt for local-agent/Copilot:

> Add Inkling 2 Mini Uncensored as a local-agent fallback. Keep Qwen3.6-35B-A3B with DFlash2 as the primary server. Use the separate `/home/acid/projects/llama-inkling-main/build/bin/llama-server` binary because it contains native Inkling loop support and the DSpark validation merge. Run the baked uncensored GGUF at `/home/acid/projects/llama.cpp-APU-upgrade/models/Inkling-2-Mini-Uncensored-AD-Q4_K.gguf` with `/home/acid/projects/llama.cpp-APU-upgrade/models/mmproj-Inkling-2-Mini-Uncensored-F16.gguf` on port 8081. Start without a drafter for a known-compatible production fallback. Use DSpark only experimentally with `--spec-type draft-dspark --spec-draft-n-max 1 --spec-draft-n-min 1 --spec-draft-p-min 0.00`; do not enable it until the `2304/6912` state-size mismatch is fixed and verified lossless. Preserve the existing Qwen server on port 8080 with `-ngl 999 -fit off -lm none -fa on -ub 64 -b 512 -t 12 -c 32768 -np 1 --spec-draft-n-max 7 --spec-draft-n-min 1 --spec-draft-p-min 0.55`.

---

## 3. The 5 Architectural Branches

| Branch | Description | When to Use |
| :--- | :--- | :--- |
| **`main`** | Clean upstream `llama.cpp` baseline (`a691b2345`). | Reference baseline only. |
| **`v1-custom`** | Injects V1 `ggml-hip-ssd` (4096-byte page-aligned zero-copy GTT memory arena via `hipHostMallocMapped`). | Testing basic zero-copy GTT host allocations without speculative decoding. |
| **`v2-custom`** | Injects V2 `ggml-hip-ssd` (zero-copy GTT arena + Linux `io_uring` async prefetcher + closed 12-thread CPU worker pool + Dense Model Sleep logic). | Non-drafting MoE or dense inference with automatic core reservation (`12` compute threads on cores `0..11`, `4` threads on cores `12..15` reserved for OS/Tailscale/`io_uring`). |
| **`dspine-baseline`** | Clean pull of upstream `origin/dspine` (`d493f841b` - DFlash/DSpine block-diffusion speculative decoding). | Baseline DFlash2 speculative decoding without custom V2 APU hooks. |
| **`dspine-v2-custom`** | Merges `dspine-baseline` + `v2-custom` (`ggml-hip-ssd` V2). | Baseline DFlash2 + V2 APU backend without multi-size HIP graph cache. |
| **`exp-coalesced-dflash`** | **Recommended Production Branch (Part B).** Builds on `dspine-v2-custom` with multi-size `gf_res_prev[10]` HIP graph cache (eliminates adaptive draft-length graph thrashing), host-sync DFlash prefill fix, opt-in `LLAMA_MOE_STATS=1` unique-expert telemetry, and experimental `--spec-draft-exp-*` coalescing flags. | **All `local-agent` production workloads** (`Qwen3.6-35B-A3B` + DFlash2 with `--spec-draft-p-min 0.55`, **37.28 tok/s** 5-prompt avg / **49.2 tok/s** reasoning bursts / **25.0-30.9 tok/s** at 4K-18K context). |
| **`exp-approx-routing`** | **Oversized MoE SSD Branch (Part A).** Builds on `exp-coalesced-dflash` with opt-in `LLAMA_APPROX_ROUTING` (`src/llama-approx-routing.{h,cpp}`): mean-centered router cosine-similarity substitution (`lru`, `mincore`, `hotset`, `trace`) and tail-expert `drop` (`id=-1`). Zero overhead when unset. | **Oversized >45 GiB MoE models paging from NVMe SSD** (`GLM-4.5-Air-UD-IQ3_XXS`: **-80.0% real SSD major page faults**, **+20.5% decode** `4.4 -> 5.3 tok/s`, `+0.83%` PPL). Leave unset for in-RAM `Qwen3.6-35B-A3B`. |

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

> **Note on earlier `8 / 12 / 16 / 20` rows:** `qwen36-35b-a3b-dflash-Q8_0.gguf` has `dflash.block_size = 8`, so `common/speculative.cpp` silently clamped all values >= 8 down to `n_max = 7` (`block_size - 1`), which is why all four rows produced identical `43.71%` acceptance. See Section 4.1 below for the clean unclamped 5-prompt `llama-server` benchmark suite.

### 4.1 Clean 5-Prompt `llama-server` Ablation (`exp-coalesced-dflash`, `bench-coalesced-dflash.py`)
Measured across 5 diverse prompts (`coding`, `tech_512`, `reasoning`, `agent_json`, `python_algo`; 256 tokens each, `--temp 0.0 --seed 42`, warm `llama-server` with GTT/RAM collision guard):

| Configuration (`Qwen3.6-35B-A3B` + `DFlash2`) | Avg Decode (`tok/s`) | `ms/step` | Avg Draft | Acceptance | Mean Len | Peak (`reasoning`) | vs Base `n7` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `no-draft` (100% Radeon 780M) | `19.26` | `51.72` | `0.00` | `0.0%` | `1.00` | `19.44` | `-42.0%` |
| `base n7 p=0.00` *(old README config)* | `33.20` | `126.21` | `6.72` | `47.8%` | `4.20` | `45.49` | baseline |
| `base n5 p=0.00` *(fixed `n_max=5`)* | `35.77` | `105.56` | `4.86` | `57.5%` | `3.79` | `46.24` | `+7.7%` |
| `p_min=0.55` *(before `gf_res_prev[10]`)* | `34.88` | `111.88` | `4.09` | `71.1%` | `3.94` | `45.77` | `+5.1%` |
| **`p_min=0.55` + `gf_res_prev[10]`** | **`37.28`** | **`104.69`** | **`4.09`** | **`71.1%`** | **`3.94`** | **`49.18`** | **`+12.3%`** |
| **`p_min=0.60` + `gf_res_prev[10]`** | **`37.06`** | **`102.54`** | **`3.65`** | **`77.1%`** | **`3.85`** | **`49.20`** | **`+11.6%`** |
| `p=0.45 + exp-penalty=0.20` (Layer-0 router) | `36.33` | `106.18` | `4.07` | `70.4%` | `3.89` | `47.79` | `+9.4%` |
| `exp-cost=0.25` (uniform lookahead control) | `36.29` | `107.84` | `4.11` | `71.0%` | `3.95` | `46.14` | `+9.3%` |
| `exp-cost=0.28` (4-layer online+router hybrid) | `35.20` | `113.64` | `4.70` | `63.7%` | `4.03` | `47.71` | `+6.0%` |

- **Key Findings:**
  1. **Multi-size `gf_res_prev[10]` HIP graph cache (`src/llama-context.cpp`):** Upstream only kept 2 arenas (`n_outputs == 0` vs `> 0`), so adaptive truncation (`p_min`) thrashed the `ggml-cuda` HIP graph cache across batch sizes `1..8` (+7 to 15 ms/step). Giving batch sizes `1..8` dedicated arenas raised `p_min=0.55` from `34.88` to **`37.28 tok/s`** (+12.3% over `base n7`, 100% lossless).
  2. **Expert-coalescing proxy negative result (in-RAM 256-expert MoE):** `LLAMA_MOE_STATS=1` confirmed unique experts/layer scale from `13.7` (batch=2) to `32.4` (batch=8), and both proxies separated lower- vs higher-novelty tokens, but per-token expert novelty varied by only ~5-11% across 256 experts while DFlash2 confidence `p_top` dominated acceptance value; sync/lookup overhead made pure `--spec-draft-p-min 0.55` + `gf_res_prev[10]` the clear winner.
  3. **GTT/Ollama OOM collision warning:** Never run a second 21 GiB `llama-server` or `ollama` GPU model concurrently with `Qwen3.6-35B-A3B` (2x ~22 GiB pinned TTM GTT exceeds 45 GiB RAM and causes an unrecoverable kernel OOM stall). Always verify `mem_info_gtt_used < 2 GiB` and `MemAvailable >= 28 GiB` before launching benchmarks.

### 4.2 Long-Context Coding Agent Benchmark (4K / 8K / 16K, Native F16 KV + `amdgpu.cwsr_enable=0`)
Measured on `Qwen3.6-35B-A3B` + `DFlash2` (`-c 32768 -np 1 -fa on -ub 64 -b 512 --spec-draft-n-max 7 --spec-draft-n-min 1 --spec-draft-p-min 0.55`, native F16 KV, `cache_prompt=true`, 256 output tokens/turn, 0 GPU resets, 0/0 graph reallocs, 468 graphs reused):

| Case (Real C++ Code Excerpt) | Total In | Cached | Eval In | Prefill (`tok/s`) | TTFT (`s`) | Decode (`tok/s`) | DFlash Acc | Mean Len |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `4K_code T1` (cold) | `4,684` | `0` | `4,684` | `194.5` | `24.08` | **`29.38`** | `68.6%` | `3.24` |
| `4K_code T2` (prefix hit) | `4,816` | `4,680` | `136` | `127.3` | **`1.07`** | **`30.87`** | `66.4%` | `3.46` |
| `8K_code T1` (cold) | `9,348` | `23` | `9,325` | `186.3` | `50.07` | **`30.21`** | `70.5%` | `3.66` |
| `8K_code T2` (prefix hit) | `9,474` | `9,344` | `130` | `126.3` | **`1.03`** | **`27.78`** | `70.2%` | `3.24` |
| `16K_code T1` (cold) | `18,641` | `23` | `18,618` | `175.9` | `105.82` | **`25.63`** | `65.0%` | `3.28` |
| `16K_code T2` (prefix hit) | `18,763` | `18,637` | `126` | `114.1` | **`1.10`** | **`24.95`** | `65.4%` | `3.12` |

- **Long-Context Findings for `local-agent`:**
  1. **Do NOT use `-ctk q8_0 -ctv q8_0` on `Qwen3.6-35B-A3B` + `DFlash2`:** Only 10/40 layers use attention (Gated DeltaNet), so native F16 KV at `-c 32768` uses only `~640 MiB`. Enabling `q8_0` KV forces `ggml-cuda/fattn.cu` `BEST_FATTN_KERNEL_MMA_F16` (`DKQ=256, gqa=8, n_tokens >= 3`) to dequantize K/V into temporary F16 buffers on every step, dropping 18K-22K decode from **`25.0-25.6 tok/s`** down to `~19.2 tok/s`.
  2. **Prefix caching (`cache_prompt=true` + `-np 1`) cuts 18.7K TTFT by 96x (`105.8 s -> 1.10 s`):** Keep system prompt and repo context prefixes stable across turns and set client HTTP timeout >= `300 s` for cold 16K-24K Turn-1 prefills.
  3. **`amdgpu.cwsr_enable=0` in GRUB is mandatory on `gfx1103`:** Eliminated recurring `MES failed to respond to msg=REMOVE_QUEUE` / `Failed to evict queue 3` GPU resets when `ctx_tgt` (queue 2) and `ctx_dft` (queue 3) run long multi-turn sessions.

### 4.2.1 Long-Context `b12217` vs `b12629` Regression & `DFlash2` Isolation Benchmark (2026-11-24)

Tested on cold `llama-server` instances (Request #1 unless noted, greedy `temp=0`, `cache_prompt=false`, `logprobs=true`, `-ngl 999 -ngld 999 -fit off -lm none -fa on -ub 64 -b 512 -t 12 -c 32768 -np 1 --spec-draft-n-max 7 --spec-draft-n-min 1 --spec-draft-p-min 0.55`):

| Binary | Model | `-md` DFlash2 | `LLAMA_DFLASH_DEV` | HIP Graphs | Prompt | Result |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `b12629` (`llama-inkling-main`, `1fb8a2696`) | `UD-Q4_K_M` | ON | `1` (default) | ON | `14,319` tok (synthetic 560-para) | **PASS** (`106.1 s`, coherent) |
| `b12629` | `UD-Q4_K_M` | ON | `1` (default) | ON | `12,732` tok (real README slice, Req #1) | **`NaN` (`////////////`, `78.2 s`; `76.9 s` as Req #2)** |
| `b12629` | `UD-Q4_K_M` | ON | `1` (default) | OFF (`GGML_CUDA_DISABLE_GRAPHS=1`) | `12,732` tok (Req #1) | **`NaN` (`////////////`, `77.6 s`)** |
| `b12629` | `UD-Q4_K_M` | **OFF** | — | ON | `12,732` tok (Req #1) | **PASS** (`77.5 s`, coherent) |
| `b12629` | `UD-Q4_K_M` | ON | **`0`** (host copy) | ON | `12,732` tok (Req #1) | **PASS** (`80.1 s`, coherent) |
| **`b12217` (`build-dspine-v2`, `19e225579`)** | **`UD-Q4_K_M`** | **ON** | **`1`** (default) | **ON** | **`12,732` tok (Req #1)** | **PASS (`68.9 s`, coherent — 14.0% faster than `b12629` `DEV=0`, 11.1% faster than `b12629` no-draft)** |
| **`b12217` (`build-dspine-v2`, `19e225579`)** | **`UD-Q4_K_M`** | **ON** | **`1`** (default) | **ON** | **Full `18,354` tok README (Req #1)** | **PASS (`103.3 s`, coherent)** |
| `b12629` | `Uncensored-HauhauCS-Q4_K_M` | ON | `1` (default) | ON | `12,732` tok (Req #1) | **PASS** (`72.8 s`, coherent; *see head-to-head below for `4K / 18.5K / 25.6K`*) |

**`Qwen3.6-35B-A3B-Uncensored` (`HauhauCS Q4_K_M` + `mmproj-f16` + `DFlash2 n7 p0.55`) Head-to-Head (`b12217` vs `b12629`, 2026-11-24):**

| Test Case (Cold Server, `temp=0`, `cache_prompt=false`, `0` Reallocs) | `b12217` (`build-dspine-v2`) | `b12629` (`llama-inkling-main`) | `b12217` Delta |
| :--- | :--- | :--- | :--- |
| **4K 3-Prompt Avg** (`code`, `reasoning`, `sec_ctf`, `mean len 5.52/5.55`) | **`46.75 tok/s`** dec · `121.46 tok/s` PP · `82.25%` acc · `22.06 GiB` | `43.04 tok/s` dec · `121.68 tok/s` PP · `82.25%` acc · `22.16 GiB` | **`+8.6%` decode** (`NaN=False`) |
| **Vision `mmproj-f16`** (`127` tok image + text, `64` gen) | **`3.23 s`** wall · `117.90 tok/s` PP · `29.79 tok/s` dec (`NaN=False`) | `3.29 s` wall · `110.21 tok/s` PP · `30.14 tok/s` dec (`NaN=False`) | **`+7.0%` PP**, `mmproj` verified |
| **Cold `18,492` tok (`~18.5K`, `-c 32768`)** | **`95.25 s`** wall · **`202.86 tok/s`** PP · **`31.29 tok/s`** dec · `77.24%` acc · `26.40 GiB` | `119.95 s` wall · `159.79 tok/s` PP · `30.63 tok/s` dec · `79.03%` acc · `26.33 GiB` | **`+27.0%` PP, `-20.6%` wall** (`NaN=False`) |
| **Cold `25,595` tok (`~25.6K`, `-c 32768`)** | **`136.39 s`** wall · **`194.28 tok/s`** PP · **`27.64 tok/s`** dec · `69.23%` acc · `26.41 GiB` | `184.90 s` wall · `142.49 tok/s` PP · `24.37 tok/s` dec · `66.42%` acc · `26.48 GiB` | **`+36.3%` PP, `+13.4%` dec, `-26.2%` wall** (`NaN=False`) |

- **Root Cause:**
  - `b12217 -> b12629` regression in `LLAMA_DFLASH_DEV=1` device-buffer feature extraction (`build_layer_inp_dev()` + `ggml_set_rows` in `src/llama-graph.cpp` / `ggml/src/ggml-cuda/set-rows.cu`) on RDNA3 HIP (`gfx1103`).
  - Produces all-`NaN` logits (`"logprob": null` on every token), `0.00%` DFlash2 acceptance, and deterministic repetition of token `14` (`"/"`) or `18` (`"3"`). Not an OOM and not a sampling bug.

- **Guidance for Future Model & Build Decisions:**
  1. **Binary selection:** Keep `Qwen3.6-35B-A3B-UD-Q4_K_M` + `DFlash2` on **`b12217` (`build-dspine-v2`)**. Reserve `b12629` for architectures/features `b12217` lacks (e.g., `inkling2_loop` / `DSpark`); if running `b12629` + `DFlash2` on long contexts and `NaN`s appear, set `LLAMA_DFLASH_DEV=0` as a workaround.
  2. **Mandatory pre-deployment validation gate:** Never promote a new `llama.cpp` build or GGUF based solely on 4K-context benchmarks or synthetic repetitive prompts. Always run a cold Request #1 realistic dense prompt (`>= 12K–18K` tokens, `cache_prompt=false`, `logprobs=true`) with `-md` active and assert `logprob != null` (no `NaN`s), coherent output, and non-regressed prefill wall time.
  3. **Uncensored Qwen3.6 verdict (`2026-11-24`):** Although `b12629` did not hit `NaN` on `Uncensored-HauhauCS-Q4_K_M` up to `25.6K` tokens, **`b12217` is strictly faster at every context length** (`+8.6%` 4K decode, `+27–36%` long-context prefill, `-26.2%` wall time at `25.6K`) and supports `mmproj` (`qwen3vl_merger`) cleanly, making `b12217` the better binary for `:8081` as well.

### 4.3 Part A: Similarity-Bounded Cold-Expert Substitution (`exp-approx-routing`)
Opt-in MoE routing hook (`src/llama-approx-routing.{h,cpp}`, controlled by `LLAMA_APPROX_ROUTING`, zero overhead when unset) that computes per-layer mean-centered router cosine similarity from `blk.<L>.ffn_gate_inp.weight` and redirects cold low-weight tail experts (`w_rel < tau`, `protect_top=1`) to their highest-similarity distinct resident neighbor (`sim >= delta`).

- **Phase-0 Router Geometry & Routing Skew:**
  - `Qwen3.6-35B-A3B` (`256 x 2048` softmax, 40 MoE layers): `36-64%` shared mean-vector energy; centered top-1 median cos sim **`0.385`** (`70.3% > 0.30`, max `0.976`). Routing skew on WikiText-2: top 50% cover **`93.45%`**, top 75% cover **`98.76%`**.
  - `GLM-4.5-Air-UD-IQ3_XXS` (`128 x 4096` sigmoid, 46 MoE layers): `19-27%` mean-vector energy; centered top-1 median cos sim **`0.099`** (~4x more orthogonal, requires `delta=0.05`). Routing skew: top 50% cover **`87.76%`**, top 75% cover **`98.08%`**.

- **Simulated RAM Cap on `Qwen3.6-35B-A3B` (16-chunk WikiText-2, 8,192 tokens, `ROCm0`):**

| Config (`Qwen3.6-35B-A3B`, `cap=50%` = `128/256` experts) | `tau / delta` | Cold Misses Eliminated | WikiText-2 PPL | `dPPL` vs Exact |
| :--- | :---: | :---: | :---: | :---: |
| Baseline (exact) | - | `0%` | **`7.0978 +/- 0.278`** | `0.000` |
| `lru + sim` centered (`cap=50%`) | `0.12 / 0.25` | **`59.9%`** | **`7.0963`** | **`-0.002` (`-0.02%`)** |
| `lru + sim` centered (`cap=50%`, aggressive) | `0.15 / 0.15` | **`92.1%`** | **`7.1258`** | **`+0.028` (`+0.39%`)** |
| `lru + sim` centered (`cap=75%` = `192/256`) | `0.12 / 0.25` | **`77.8%`** | **`7.0789`** | **`-0.019` (`-0.27%`)** |
| *Ablation:* `lru + random_resident` (`cap=50%`) | `0.12 / 0.25` | `60.1%` | `7.0510` | `-0.047` |

- **Real NVMe SSD Paging on `GLM-4.5-Air-UD-IQ3_XXS` (`47.9 GiB` on `45 GiB` RAM, `-ngl 0 -lm mmap --no-repack`):**

| Config (`GLM-4.5-Air-UD-IQ3_XXS`) | Decode (`-n 64`) | 3-Chunk PPL Time | **Real SSD Major Faults** | Fault Reduction | WikiText-2 PPL | `dPPL` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (exact)** | `4.4 tok/s` | `114 s` | **`836,571`** | `0%` | **`2.9751 +/- 0.216`** | `0.000` |
| **`lru` (`75%` = `96/128`) + `sim` (`tau=0.15, delta=0.05`)** | **`5.3 tok/s` (`+20.5%`)** | **`103 s`** | **`167,718`** | **`-80.0%` (5.0x fewer)** | **`2.9999`** | **`+0.025` (`+0.83%`)** |
| `lru` (`75%`) + `drop` (`tau=0.15`, CPU `id=-1`) | `5.2 tok/s` (`+18.2%`) | `107 s` | `194,525` | `-76.7%` (4.3x fewer) | `3.0424` | `+0.067` (`+2.26%`) |
| `mincore` + `sim` / `drop` (`tau=0.15`) | `5.0 - 5.1 tok/s` | - | - | `74.4% - 81.4%` cold cut | - | - |
| *Ablation:* `lru` (`75%`) + `random_resident` | - | `106 s` | `790,347` | `-5.5%` | `3.0192` | `+0.044` (`+1.48%`) |
| *Stress test:* `hotset` (`75%`, `tau=0.25, protect_top=0`) + `drop` | `5.4 tok/s` | `96 s` | `148,670` | `-82.2%` | `3.5345` | `+0.559` (`+18.8%`) |

- **Usage for oversized >45 GiB MoE models (`GLM-4.5-Air-UD-IQ3_XXS`):**
```bash
HIP_VISIBLE_DEVICES=-1 \
LLAMA_APPROX_ROUTING=lru LLAMA_APPROX_SUB_POLICY=sim LLAMA_APPROX_CENTERED=1 \
LLAMA_APPROX_CAP_PCT=75 LLAMA_APPROX_TAU=0.15 LLAMA_APPROX_DELTA=0.05 \
./build-dspine-v2/bin/llama-cli \
  -m models/GLM-4.5-Air-UD-IQ3_XXS/GLM-4.5-Air-UD-IQ3_XXS-00001-of-00002.gguf \
  -lm mmap -fit off --no-repack -ub 64 -b 64 -t 12 -ngl 0 -c 2048
```
> **Note:** Leave `LLAMA_APPROX_ROUTING` unset for in-RAM `Qwen3.6-35B-A3B` (`20.6 GiB` fits 100% in GTT with 0 SSD faults, and enabling a CPU custom op splits HIP graph capture).

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
  --spec-draft-n-max 7 --spec-draft-n-min 1 --spec-draft-p-min 0.55 \
  -fit off -lm none -fa on -ub 64 -b 512 -t 12 -c 32768 -np 1 --jinja \
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
| **`--spec-draft-n-max 7 --spec-draft-n-min 1 --spec-draft-p-min 0.55`** | **DFlash2 Adaptive Draft Block.** `qwen36-35b-a3b-dflash-Q8_0.gguf` has `dflash.block_size = 8` (max `7` draft tokens). `--spec-draft-p-min 0.55` truncates low-confidence tail tokens before Radeon 780M verification, and `exp-coalesced-dflash`'s `gf_res_prev[10]` HIP graph cache keeps batch sizes `1..8` fast (**37.3 tok/s** avg, **49.2 tok/s** reasoning bursts, 100% lossless). |
| **`-t 12`** | **Core Reservation Safeguard.** Pairs with `dspine-v2-custom`'s closed 12-thread pool (`cores 0..11`), keeping 4 hardware threads (`cores 12..15`) dedicated to Ubuntu OS, Tailscale, Docker, and `local-agent`. |

---

## 7. Bare-Metal BIOS, GRUB & ROCm Configuration Reference

If you ever reinstall Ubuntu or migrate to a new APU server, apply these exact system settings:

### 1. BIOS Settings (AMI BIOS -> `Advanced` -> `AMD CBS` -> `NBIO Common Options` -> `GFX Configuration`)
- **iGPU Configuration:** `UMA_SPECIFIED`
- **UMA Frame Buffer Size:** `1G` (leaves **45 GiB** of physical RAM visible to Linux instead of carving out 12 GiB as static VRAM)

### 2. GRUB Kernel Parameters (`/etc/default/grub`)
Allocates a **40 GiB Unified GTT Pool** (`10485760` pages * 4096 bytes), disables CWSR preemption (`amdgpu.cwsr_enable=0`, required on `gfx1103` to prevent `MES REMOVE_QUEUE` / `Failed to evict queue 3` GPU resets under DFlash2 dual-stream workloads), and disables the 1 GiB `crashkernel` reservation:
```bash
sudo sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT=.*/GRUB_CMDLINE_LINUX_DEFAULT="ttm.pages_limit=10485760 ttm.page_pool_size=10485760 amdgpu.gttsize=40960 amdgpu.cwsr_enable=0 crashkernel=no"/' /etc/default/grub
if [ -f /etc/default/grub.d/kdump-tools.cfg ]; then
    sudo sed -i 's/crashkernel=[^"]*/crashkernel=no/g' /etc/default/grub.d/kdump-tools.cfg
fi
sudo update-grub
```

Verify after reboot:
```bash
free -h                                              # Should show Mem: 45Gi total, ~44Gi available
cat /sys/module/ttm/parameters/pages_limit           # Should print 10485760 (40 GiB)
cat /sys/module/amdgpu/parameters/cwsr_enable        # Should print 0
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
- **[`bench-inferssd.sh`](bench-inferssd.sh)**: Automated multi-branch benchmark suite that verifies the 40 GiB GTT limit and enforces a pre-run GTT/RAM collision safety guard (`MemAvailable >= 28 GiB`, `GTT <= 2 GiB`), sweeps real unclamped DFlash2 draft sizes `4, 5, 6, 7, 7@p0.55`, outputs [`benchmark_results.csv`](benchmark_results.csv), and prints the final `VERDICT`.
- **[`bench-coalesced-dflash.py`](bench-coalesced-dflash.py)**: Memory-guarded warm `llama-server` 5-prompt benchmark & ablation suite for Part B (`pass1`, `pass_hipcache`, `pass_router`, `telemetry` with `LLAMA_MOE_STATS=1`).
- **[`bench-inkling-dspark.py`](bench-inkling-dspark.py)**: Cross-model benchmark for Qwen3.6-35B-A3B and the Inkling 2 Mini mainline and uncensored GGUFs. It supports separate Qwen and Inkling server binaries.

### Inkling 2 Mini benchmark changes

Added the Inkling 2 Mini model, DSpark drafter, and uncensored GGUF paths in this repository on 2026-11-22.

The quick benchmark used the custom `build-dspine-v2` binary for Qwen and Unsloth's `b12588-mix-c0c3de8` gfx110X prebuilt for Inkling. Qwen DFlash2 used `n_max=5/7` and `p_min=0.55`; Inkling DSpark used `n_max=3/5/7/8` and `p_min=0.00`.

Results:

| Configuration | Average decode |
| :--- | ---: |
| Qwen3.6 no drafter | `18.44 tok/s` |
| Qwen3.6 DFlash2 n5 p0.55 | `39.32 tok/s` |
| Qwen3.6 DFlash2 n7 p0.55 | `40.81 tok/s` |
| Inkling 2 Mini no drafter | `9.69 tok/s` |
| Inkling 2 Mini DSpark n3 p0.00 | `8.42 tok/s` |
| Inkling 2 Mini DSpark n5 p0.00 | `7.70 tok/s` |
| Inkling 2 Mini DSpark n7 p0.00 | `6.94 tok/s` |
| Inkling 2 Mini DSpark n8 p0.00 | `6.96 tok/s` |
| Inkling 2 Mini Uncensored no drafter | `8.69-8.71 tok/s` |

Inkling DSpark did not improve this APU's 128-token benchmark. Acceptance fell from `44.7%` at n3 to `24.1%` at n7/n8, while larger verification blocks were slower.

The uncensored GGUF uses `inkling2_loop` and was rejected by the b12588 Unsloth prebuilt. It needs a fresh mainline llama.cpp build with native `inkling2_loop` support. Using root-layout Qwen DFlash2 as an Inkling drafter crashed during DFlash graph construction and is unsupported.

Planned follow-up tests:

- Build fresh mainline llama.cpp outside this repository for native `inkling2_loop` support.
- Completed: fresh mainline b12628 loads the uncensored `inkling2_loop` GGUF and F16 mmproj at 8.69-8.71 tok/s with no drafter.
- Blocked: the published DSpark drafter has `markov_w1.weight` shape 256 x 200058, while the uncensored target expects a 201024-token vocabulary. DSpark rejects the uncensored target as incompatible.
- Required: an uncensored-compatible DSpark export, or a vocab-remapped drafter, is needed before n1/n2/n3 can be measured safely.
- Run AtomicChat's uncensored GGUF with the matching F16 mmproj and no drafter.
- Test DSpark on the uncensored and mainline models with `n_max=1/2/3` and `p_min=0.55`.
- Test `UD-IQ3_XXS` or `UD-Q3_K_XL` to measure the lower-bandwidth quantization tradeoff.
- Test the variable-loop 2-pass checkpoint if it becomes available.
- Use MoE telemetry to verify expert union growth instead of relying only on the DDR5 bandwidth explanation.

The Qwen local-agent server is restored with the custom `build-dspine-v2` binary and DFlash2. Verify it with `curl http://127.0.0.1:8080/health`.

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
