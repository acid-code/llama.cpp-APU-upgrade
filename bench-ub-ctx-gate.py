#!/usr/bin/env python3
"""
Section 4.2.1 cold validation gate benchmark for b12217 (build-dspine-v2):
  Part 1: -ub 64 vs 96 vs 128 on cold 18.5K and 25.6K dense README prompts
          (cache_prompt=false, logprobs=true, temp=0, assert no NaN & 0 GPU resets).
  Part 2: Context scaling test (-c 32768 / 65536 / 131072 / 262144) measuring
          exact GTT footprint, MemAvailable, load time, 18.5K cold prefill + decode,
          and verifying no NaNs or MES resets with native F16 KV + DFlash2.
"""
import csv
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

BIN = "/opt/llama-server/bin/llama-server"
LIB = "/opt/llama-server/bin"
MODEL = "/home/acid/projects/llama.cpp-APU-upgrade/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
DRAFT = "/home/acid/projects/llama.cpp-APU-upgrade/models/qwen36-35b-a3b-dflash-Q8_0.gguf"
README_PATH = "/home/acid/projects/llama.cpp-APU-upgrade/README.md"
PORT = 8097
CSV_OUT = "/home/acid/projects/llama.cpp-APU-upgrade/bench_ub_ctx_gate.csv"


def gtt_gib():
    for d in ("/sys/class/drm/card0/device", "/sys/class/drm/card1/device"):
        p = os.path.join(d, "mem_info_gtt_used")
        if os.path.exists(p):
            return int(open(p).read().strip()) / (1024 ** 3)
    return 0.0


def mem_avail_gib():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable:"):
            return int(ln.split()[1]) / (1024 ** 2)
    return 0.0


def gpu_reset_count():
    try:
        out = subprocess.check_output(
            ["journalctl", "-k", "--since", "10 minutes ago", "--no-pager"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return sum(1 for ln in out.splitlines() if "GPU reset" in ln or "MES failed" in ln or "Failed to evict" in ln)
    except Exception:
        return -1


def unload_ollama():
    try:
        ps = json.loads(urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=3).read())
        for m in ps.get("models", []):
            req = urllib.request.Request(
                "http://127.0.0.1:11434/api/generate",
                data=json.dumps({"model": m["name"], "keep_alive": 0}).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10).read()
        time.sleep(1.5)
    except Exception:
        pass


def build_dense_prompt(target_chars):
    txt = open(README_PATH, "r", encoding="utf-8", errors="ignore").read()
    # Repeat with distinct section headers so prefix/content stays realistic and dense
    chunks = []
    idx = 1
    while sum(len(c) for c in chunks) < target_chars:
        chunks.append(f"\n\n--- ARCHIVE SECTION {idx} ---\n" + txt)
        idx += 1
    body = "".join(chunks)[:target_chars]
    return (
        "Read the following technical documentation carefully and summarize the 3 most important "
        "hardware findings and recommended flags for Radeon 780M in concise bullet points.\n\n"
        + body
    )


def run_case(label, ub, batch, ctx, prompt_text, max_tokens=96):
    unload_ollama()
    g0 = gtt_gib()
    m0 = mem_avail_gib()
    if g0 > 4.0 or m0 < 26.0:
        raise RuntimeError(f"Unsafe GPU memory before {label}: GTT={g0:.2f} GiB, MemAvail={m0:.2f} GiB")

    resets_before = gpu_reset_count()
    cmd = [
        BIN, "-m", MODEL,
        "-md", DRAFT, "-ngld", "999",
        "--spec-draft-n-max", "7", "--spec-draft-n-min", "1", "--spec-draft-p-min", "0.55",
        "-ngl", "999", "-fit", "off", "-lm", "none", "-fa", "on",
        "-ub", str(ub), "-b", str(batch), "-t", "12", "-c", str(ctx), "-np", "1",
        "--jinja", "--host", "127.0.0.1", "--port", str(PORT),
    ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LIB
    env["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
    env["HIP_VISIBLE_DEVICES"] = "0"

    log_path = f"/tmp/ub_ctx_{label}.log"
    log_f = open(log_path, "w+")
    proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        t0 = time.time()
        while time.time() - t0 < 120:
            if proc.poll() is not None:
                raise RuntimeError(f"{label} exited early ({proc.returncode})")
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200:
                    break
            except Exception:
                time.sleep(0.5)
        else:
            raise TimeoutError(f"{label} health timeout")

        load_s = time.time() - t0
        gtt_loaded = gtt_gib()
        mem_loaded = mem_avail_gib()

        payload = {
            "model": "qwen3.6-35b-a3b",
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": 42,
            "cache_prompt": False,
            "logprobs": True,
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        rt0 = time.time()
        resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
        wall_s = time.time() - rt0
        gtt_peak = max(gtt_loaded, gtt_gib())
        mem_min = min(mem_loaded, mem_avail_gib())

        t = resp.get("timings", {})
        choice = resp["choices"][0]
        msg = choice["message"]
        lp_items = (choice.get("logprobs") or {}).get("content") or []
        nan_lp = any(item.get("logprob") is None for item in lp_items) if lp_items else False
        sample = ((msg.get("reasoning_content") or "") + " || " + (msg.get("content") or "")).replace("\n", " ")[:180]
        pp_n = int(t.get("prompt_n", 0) or 0)
        pp_tps = float(t.get("prompt_per_second", 0.0) or 0.0)
        dec_tps = float(t.get("predicted_per_second", 0.0) or 0.0)
        dn = int(t.get("draft_n", 0) or 0)
        da = int(t.get("draft_n_accepted", 0) or 0)
        acc = (100.0 * da / dn) if dn > 0 else 0.0
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=15)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        log_f.close()
        time.sleep(2.0)

    resets_after = gpu_reset_count()
    resets = max(0, resets_after - resets_before) if resets_before >= 0 and resets_after >= 0 else -1
    kv_line = ""
    buf_line = ""
    reallocs = 0
    for ln in open(log_path):
        if "KV buffer size" in ln:
            kv_line = ln.strip()
        if "compute buffer size" in ln and "ROCm0" in ln:
            buf_line = ln.strip()
        if "graph reallocs" in ln:
            try:
                parts = ln.split("=")[1].split("/")
                reallocs += int(parts[0].strip()) + int(parts[1].split()[0].strip())
            except Exception:
                pass

    print(
        f"[{label:22s}] ub={ub:3d} ctx={ctx:6d} | in={pp_n:5d} | wall={wall_s:6.1f}s | "
        f"PP={pp_tps:6.1f} | DEC={dec_tps:5.2f} | acc={acc:5.1f}% | "
        f"GTT={gtt_peak:5.2f} GiB | MemAvail={mem_min:5.2f} GiB | NaN={nan_lp} | resets={resets} | reallocs={reallocs}"
    )
    if kv_line:
        print(f"    {kv_line}")
    if buf_line:
        print(f"    {buf_line}")
    print(f"    sample={sample!r}")
    sys.stdout.flush()
    return {
        "label": label,
        "ub": ub,
        "batch": batch,
        "ctx": ctx,
        "prompt_n": pp_n,
        "load_s": f"{load_s:.1f}",
        "wall_s": f"{wall_s:.2f}",
        "pp_tps": f"{pp_tps:.2f}",
        "dec_tps": f"{dec_tps:.2f}",
        "acc_pct": f"{acc:.2f}",
        "gtt_peak_gib": f"{gtt_peak:.2f}",
        "mem_min_gib": f"{mem_min:.2f}",
        "nan": nan_lp,
        "resets": resets,
        "reallocs": reallocs,
    }


def safe_run(rows, label, ub, batch, ctx, prompt_text):
    try:
        rows.append(run_case(label, ub, batch, ctx, prompt_text))
    except Exception as e:
        print(f"[{label:22s}] FAILED: {e}")
        sys.stdout.flush()
        rows.append({
            "label": label, "ub": ub, "batch": batch, "ctx": ctx,
            "prompt_n": 0, "load_s": "ERR", "wall_s": "ERR", "pp_tps": "0",
            "dec_tps": "0", "acc_pct": "0", "gtt_peak_gib": f"{gtt_gib():.2f}",
            "mem_min_gib": f"{mem_avail_gib():.2f}", "nan": True,
            "resets": gpu_reset_count(), "reallocs": -1,
        })


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    p18k = build_dense_prompt(56000)   # ~18.5K tokens
    p25k = build_dense_prompt(78000)   # ~25.6K tokens
    rows = []

    if mode in ("all", "ub"):
        print("=== PART 1: -ub 64 vs 96 vs 128 on 18.5K & 25.6K cold prefill (c=32768) ===")
        for ub in (64, 96, 128):
            safe_run(rows, f"ub{ub}_18k", ub, 512, 32768, p18k)
        for ub in (64, 128):
            safe_run(rows, f"ub{ub}_25k", ub, 512, 32768, p25k)

    if mode in ("all", "ctx"):
        print("\n=== PART 2: Context scaling (64K / 128K / 256K) with native F16 KV + DFlash2 ===")
        for ctx in (65536, 131072, 262144):
            safe_run(rows, f"ctx{ctx//1024}k_ub64_18k", 64, 512, ctx, p18k)

    if mode == "probe":
        print("=== PROBE PART A: -ub 128 / 192 / 256 at -c 262144 (256K) on 23.6K cold prompt ===")
        p55k = build_dense_prompt(130000)  # ~55K tokens
        probe_rows = []
        best_ub = None
        best_pp = 0.0
        for ub, batch in ((128, 512), (192, 576), (256, 512)):
            safe_run(probe_rows, f"ctx256k_ub{ub}_23k", ub, batch, 262144, p18k)
            r = probe_rows[-1]
            # Stop escalating -ub if we hit any error, NaN, or GPU reset
            if r["wall_s"] == "ERR" or r["nan"] or (isinstance(r["resets"], int) and r["resets"] > 0):
                print(f"Stopping -ub escalation at ub={ub} (wall={r['wall_s']}, nan={r['nan']}, resets={r['resets']})")
                sys.stdout.flush()
                break
            pp_val = float(r["pp_tps"])
            if pp_val > best_pp:
                best_pp = pp_val
                best_ub = (ub, batch)

        if best_ub is not None:
            ub, batch = best_ub
            print(f"\n=== PROBE PART B: Large ~55K-token cold prefill at -c 262144 with winning -ub {ub} -b {batch} ===")
            safe_run(probe_rows, f"ctx256k_ub{ub}_55k", ub, batch, 262144, p55k)

        out_probe = "/home/acid/projects/llama.cpp-APU-upgrade/bench_ub_256k_probe.csv"
        with open(out_probe, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(probe_rows[0].keys()))
            w.writeheader()
            w.writerows(probe_rows)
        print(f"\nSaved {len(probe_rows)} rows to {out_probe}")
        return

    if rows:
        with open(CSV_OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved {len(rows)} rows to {CSV_OUT}")


if __name__ == "__main__":
    main()
