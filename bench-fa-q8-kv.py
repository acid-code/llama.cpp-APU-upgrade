#!/usr/bin/env python3
"""
Head-to-head benchmark of Nathanw1014's fa-tile-dequant-on-load (b781a8d5dc) + RDNA3 DFlash2 routing
on Qwen3.6-35B-A3B-UD-Q4_K_M + DFlash2 (n7 p0.55) at -c 262144 -ub 256 -b 512:
  - Stock b12217 (f16 KV vs q8_0 KV) at 23.6K cold tokens
  - Patched b12217 (q8_0 KV with max_q=8 vs max_q=9999, plus q4_0 KV) at 23.6K cold tokens
  - Deep 55.2K cold gate on winning q8_0 mode (comparing against stock f16's 15.45 tok/s / 30.66 GiB GTT)
"""
import csv
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

STOCK_BIN = "/opt/llama-server/bin/llama-server"
STOCK_LIB = "/opt/llama-server/bin"
PATCH_BIN = "/home/acid/projects/llama.cpp-APU-upgrade/build-fa-q8-kv/bin/llama-server"
PATCH_LIB = "/home/acid/projects/llama.cpp-APU-upgrade/build-fa-q8-kv/bin"
MODEL = "/home/acid/projects/llama.cpp-APU-upgrade/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
DRAFT = "/home/acid/projects/llama.cpp-APU-upgrade/models/qwen36-35b-a3b-dflash-Q8_0.gguf"
README_PATH = "/home/acid/projects/llama.cpp-APU-upgrade/README.md"
PORT = 8097
CSV_OUT = "/home/acid/projects/llama.cpp-APU-upgrade/bench_fa_q8_kv.csv"


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


def run_case(label, bin_path, lib_path, kv_type, prompt_text, max_q=None, ctx=262144, max_tokens=128):
    unload_ollama()
    g0 = gtt_gib()
    m0 = mem_avail_gib()
    if g0 > 4.0 or m0 < 24.0:
        raise RuntimeError(f"Unsafe GPU memory before {label}: GTT={g0:.2f} GiB, MemAvail={m0:.2f} GiB")

    resets_before = gpu_reset_count()
    cmd = [
        bin_path, "-m", MODEL,
        "-md", DRAFT, "-ngld", "999",
        "--spec-draft-n-max", "7", "--spec-draft-n-min", "1", "--spec-draft-p-min", "0.55",
        "-ngl", "999", "-fit", "off", "-lm", "none", "-fa", "on",
        "-ub", "256", "-b", "512", "-t", "12", "-c", str(ctx), "-np", "1",
        "--jinja", "--host", "127.0.0.1", "--port", str(PORT),
    ]
    if kv_type != "f16":
        cmd += ["-ctk", kv_type, "-ctv", kv_type]

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = lib_path
    env["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
    env["HIP_VISIBLE_DEVICES"] = "0"
    if max_q is not None:
        env["GGML_CUDA_FA_TILE_Q_MAX_Q"] = str(max_q)

    log_path = f"/tmp/fa_q8_{label}.log"
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
    reallocs = 0
    for ln in open(log_path):
        if "graph reallocs" in ln:
            try:
                parts = ln.split("=")[1].split("/")
                reallocs += int(parts[0].strip()) + int(parts[1].split()[0].strip())
            except Exception:
                pass

    print(
        f"[{label:22s}] kv={kv_type:4s} max_q={str(max_q):4s} | in={pp_n:5d} | wall={wall_s:6.1f}s | "
        f"PP={pp_tps:6.1f} | DEC={dec_tps:5.2f} | acc={acc:5.1f}% | "
        f"GTT_load={gtt_loaded:5.2f} peak={gtt_peak:5.2f} GiB | MemAvail={mem_min:5.2f} GiB | "
        f"NaN={nan_lp} | resets={resets} | reallocs={reallocs}"
    )
    print(f"    sample={sample!r}")
    sys.stdout.flush()
    return {
        "label": label, "kv": kv_type, "max_q": str(max_q), "ctx": ctx,
        "in_toks": pp_n, "wall_s": f"{wall_s:.2f}", "pp_tps": f"{pp_tps:.2f}",
        "dec_tps": f"{dec_tps:.2f}", "acc_pct": f"{acc:.2f}",
        "gtt_load_gib": f"{gtt_loaded:.2f}", "gtt_peak_gib": f"{gtt_peak:.2f}",
        "mem_min_gib": f"{mem_min:.2f}", "nan": nan_lp, "resets": resets, "reallocs": reallocs,
    }


def main():
    p23k = build_dense_prompt(56000)   # ~24K tokens
    p55k = build_dense_prompt(130000)  # ~55K tokens
    rows = []

    print("=== PART 1: 24K Cold Gate at -c 262144 -ub 256 (Stock vs fa-tile-dequant-on-load) ===")
    cases_23k = [
        ("stock_f16_24k", STOCK_BIN, STOCK_LIB, "f16", None),
        ("stock_q8_24k", STOCK_BIN, STOCK_LIB, "q8_0", None),
        ("patch_q8_max8_24k", PATCH_BIN, PATCH_LIB, "q8_0", 8),
        ("patch_q8_all_24k", PATCH_BIN, PATCH_LIB, "q8_0", 9999),
        ("patch_q4_all_24k", PATCH_BIN, PATCH_LIB, "q4_0", 9999),
    ]
    for label, b, l, kv, mq in cases_23k:
        try:
            rows.append(run_case(label, b, l, kv, p23k, max_q=mq))
        except Exception as e:
            print(f"[{label:22s}] FAILED: {e}")
            sys.stdout.flush()

    # Pick fastest clean q8_0 mode for the deep 55K gate
    q8_candidates = [r for r in rows if r["label"].startswith("patch_q8_") and not r["nan"]]
    if q8_candidates:
        best_q8 = max(q8_candidates, key=lambda r: float(r["dec_tps"]))
        best_mq = int(best_q8["max_q"])
        print(f"\n=== PART 2: Deep 55K Cold Gate at -c 262144 -ub 256 (winning q8_0 max_q={best_mq}) ===")
        try:
            rows.append(run_case(f"patch_q8_mq{best_mq}_55k", PATCH_BIN, PATCH_LIB, "q8_0", p55k, max_q=best_mq))
        except Exception as e:
            print(f"[patch_q8_55k] FAILED: {e}")
            sys.stdout.flush()

    if rows:
        with open(CSV_OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved {len(rows)} rows to {CSV_OUT}")


if __name__ == "__main__":
    main()
