#!/usr/bin/env python3
"""
Safe, memory-guarded llama-server benchmark suite for Part B (exp-coalesced-dflash).

Safeguards:
  - Verifies MemAvailable >= 28 GiB and GPU GTT usage < 2 GiB before every server start.
  - Refuses to launch if another llama-server/ollama model is holding GPU GTT memory.
  - Loads the 21 GiB model ONCE per config and queries all 5 prompts over HTTP.
  - Waits for full GTT reclamation (< 512 MiB) after shutting down each server.
"""
import csv
import glob
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO_DIR = "/home/acid/projects/llama.cpp-APU-upgrade"
SERVER_BIN = os.path.join(REPO_DIR, "build-dspine-v2/bin/llama-server")
MODEL = os.path.join(REPO_DIR, "models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DRAFT = os.path.join(REPO_DIR, "models/qwen36-35b-a3b-dflash-Q8_0.gguf")
PORT = 18089

PROMPTS = [
    ("coding",
     "Write a clean C++17 lock-free single-producer single-consumer ring buffer with cache-line padding and explain memory ordering choices in detail."),
    ("tech_512",
     "Explain unified memory Graphics Translation Table zero-copy DMA and Mixture-of-Experts routing on AMD Ryzen APUs in detail. " * 16),
    ("reasoning",
     "A server has 48 GB unified DDR5 memory at 75 GB/s bandwidth and runs a 256-expert MoE model with 8 active experts per token across 40 layers. Derive the exact expected number of unique experts activated per layer in a batch of B independent tokens under uniform routing, and compute the memory traffic ratio vs batch size B=1 for B=2, 4, 6, 8."),
    ("agent_json",
     "You are a Linux SRE agent debugging an amdgpu MES queue reset on an AMD Radeon 780M running ROCm. Provide a structured JSON troubleshooting plan with commands, expected outputs, and root-cause decision branches."),
    ("python_algo",
     "Implement an LRU cache with O(1) get/put and TTL expiration in Python without using OrderedDict, then analyze edge cases under concurrent access."),
]

N_GEN = 256


def get_mem_available_gib():
    with open("/proc/meminfo", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                kb = int(line.split()[1])
                return kb / (1024.0 * 1024.0)
    return 0.0


def get_gtt_used_gib():
    paths = glob.glob("/sys/class/drm/card*/device/mem_info_gtt_used")
    if not paths:
        return 0.0
    vals = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                vals.append(int(f.read().strip()) / (1024.0 ** 3))
        except OSError:
            pass
    return max(vals) if vals else 0.0


def assert_memory_safe(min_avail_gib=28.0, max_gtt_gib=2.0):
    avail = get_mem_available_gib()
    gtt = get_gtt_used_gib()
    if avail < min_avail_gib or gtt > max_gtt_gib:
        raise RuntimeError(
            f"SAFETY ABORT: MemAvailable={avail:.2f} GiB (need >= {min_avail_gib:.1f}), "
            f"GTT used={gtt:.2f} GiB (need <= {max_gtt_gib:.1f}). "
            f"Check if ollama or another llama-server holds GPU memory."
        )
    return avail, gtt


def wait_gtt_reclaimed(timeout_s=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if get_gtt_used_gib() < 0.5:
            time.sleep(2.0)
            return
        time.sleep(0.5)


def wait_server_ready(proc, log_path, timeout_s=90.0):
    url = f"http://127.0.0.1:{PORT}/health"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early (rc={proc.returncode}), check {log_path}")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"llama-server did not become healthy within {timeout_s}s")


def query_prompt(prompt_text, n_predict=N_GEN):
    url = f"http://127.0.0.1:{PORT}/v1/chat/completions"
    payload = {
        "model": "qwen",
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": n_predict,
        "temperature": 0.0,
        "seed": 42,
        "cache_prompt": False,
        "stream": False,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    timings = data.get("timings", {})
    msg = data.get("choices", [{}])[0].get("message", {})
    text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]

    pred_n = int(timings.get("predicted_n") or 0)
    pred_ms = float(timings.get("predicted_ms") or 0.0)
    pred_tps = float(timings.get("predicted_per_second") or 0.0)
    prompt_tps = float(timings.get("prompt_per_second") or 0.0)
    draft_n = int(timings.get("draft_n") or 0)
    draft_acc = int(timings.get("draft_n_accepted") or 0)

    n_steps = max(pred_n - draft_acc, 1)
    ms_per_step = pred_ms / n_steps if pred_ms > 0 else 0.0
    avg_dft = draft_n / n_steps if draft_n > 0 else 0.0
    acc_pct = (100.0 * draft_acc / draft_n) if draft_n > 0 else 0.0
    mean_len = pred_n / n_steps

    return {
        "tps": pred_tps,
        "prompt_tps": prompt_tps,
        "pred_n": pred_n,
        "pred_ms": pred_ms,
        "ms_per_step": ms_per_step,
        "steps": n_steps,
        "avg_dft": avg_dft,
        "acc_pct": acc_pct,
        "draft_n": draft_n,
        "draft_acc": draft_acc,
        "mean_len": mean_len,
        "hash": text_hash,
    }


def run_config(label, draft_args, extra_env=None, prompts=PROMPTS):
    avail, gtt = assert_memory_safe()
    print(f"\n>>> [{label}] starting llama-server (MemAvail={avail:.1f} GiB, GTT={gtt:.2f} GiB)", flush=True)

    env = os.environ.copy()
    env["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
    env["HIP_VISIBLE_DEVICES"] = "0"
    if extra_env:
        env.update(extra_env)

    cmd = [
        SERVER_BIN,
        "-m", MODEL,
        "-ngl", "999",
        "-fit", "off", "-lm", "none", "-fa", "on",
        "-ub", "64", "-b", "512", "-t", "12", "-c", "4096",
        "--host", "127.0.0.1", "--port", str(PORT),
        "--no-webui",
    ] + draft_args

    safe_tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    log_path = f"/tmp/bench_srv_{safe_tag}.log"
    log_f = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)

    rows = []
    try:
        wait_server_ready(proc, log_path)
        # Warm up 16 tokens once so HIP/KFD caches are steady
        _ = query_prompt("Say ok.", n_predict=16)

        for pname, ptext in prompts:
            r = query_prompt(ptext, n_predict=N_GEN)
            r["config"] = label
            r["prompt"] = pname
            rows.append(r)
            print(
                f"  {pname:<11} | {r['tps']:6.2f} tok/s | {r['ms_per_step']:6.2f} ms/st | "
                f"dft={r['avg_dft']:4.2f} | acc={r['acc_pct']:5.1f}% | len={r['mean_len']:4.2f} | "
                f"pp={r['prompt_tps']:5.1f} | h={r['hash']}",
                flush=True,
            )
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        log_f.close()
        wait_gtt_reclaimed()

    # Extract any coalescing or MoE stats logged at server shutdown
    extra_lines = []
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if any(k in line for k in ("Expert-Coalesced", "DECODE_SUMMARY", "avg_unique_experts/layer")):
                    extra_lines.append(line.strip())
    except OSError:
        pass

    for line in extra_lines:
        print(f"  [log] {line}", flush=True)

    return rows, extra_lines


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "pass1"
    csv_path = os.path.join(REPO_DIR, f"bench_part_b_{stage}.csv")

    base_dflash = ["-md", DRAFT, "-ngld", "999", "--spec-draft-n-min", "1"]

    if stage == "pass1":
        configs = [
            ("no-draft",        [], None),
            ("base n7 p=0.00",  base_dflash + ["--spec-draft-n-max", "7", "--spec-draft-p-min", "0.00"], None),
            ("base n5 p=0.00",  base_dflash + ["--spec-draft-n-max", "5", "--spec-draft-p-min", "0.00"], None),
            ("p_min=0.45",      base_dflash + ["--spec-draft-n-max", "7", "--spec-draft-p-min", "0.45"], None),
            ("p_min=0.55",      base_dflash + ["--spec-draft-n-max", "7", "--spec-draft-p-min", "0.55"], None),
        ]
        prompts = PROMPTS
    elif stage == "pass2":
        configs = [
            ("cost=0.18 (uniform)",     base_dflash + ["--spec-draft-n-max", "7", "--spec-draft-p-min", "0.00", "--spec-draft-exp-cost", "0.18"], {"LLAMA_DFLASH_EXP_PROXY": "uniform"}),
            ("cost=0.22 (hybrid)",      base_dflash + ["--spec-draft-n-max", "7", "--spec-draft-p-min", "0.00", "--spec-draft-exp-cost", "0.22"], None),
            ("cost=0.28 (hybrid)",      base_dflash + ["--spec-draft-n-max", "7", "--spec-draft-p-min", "0.00", "--spec-draft-exp-cost", "0.28"], None),
            ("cost=0.35 (hybrid)",      base_dflash + ["--spec-draft-n-max", "7", "--spec-draft-p-min", "0.00", "--spec-draft-exp-cost", "0.35"], None),
            ("pen=0.35 p=0.25 (hybrid)",base_dflash + ["--spec-draft-n-max", "7", "--spec-draft-p-min", "0.25", "--spec-draft-exp-penalty", "0.35"], None),
        ]
        prompts = PROMPTS
    elif stage == "pass_hipcache":
        configs = [
            ("base n5 p=0.00",           base_dflash + ["--spec-draft-n-max", "5", "--spec-draft-p-min", "0.00"], None),
            ("p_min=0.55",               base_dflash + ["--spec-draft-n-max", "7", "--spec-draft-p-min", "0.55"], None),
            ("cost=0.18 (uniform)",      base_dflash + ["--spec-draft-n-max", "7", "--spec-draft-p-min", "0.00", "--spec-draft-exp-cost", "0.18"], {"LLAMA_DFLASH_EXP_PROXY": "uniform"}),
            ("cost=0.22 (hybrid)",       base_dflash + ["--spec-draft-n-max", "7", "--spec-draft-p-min", "0.00", "--spec-draft-exp-cost", "0.22"], None),
            ("cost=0.28 (hybrid)",       base_dflash + ["--spec-draft-n-max", "7", "--spec-draft-p-min", "0.00", "--spec-draft-exp-cost", "0.28"], None),
            ("pen=0.35 p=0.25 (hybrid)", base_dflash + ["--spec-draft-n-max", "7", "--spec-draft-p-min", "0.25", "--spec-draft-exp-penalty", "0.35"], None),
        ]
        prompts = PROMPTS
    elif stage == "pass_router":
        configs = [
            ("p_min=0.60",              base_dflash + ["--spec-draft-n-max", "7", "--spec-draft-p-min", "0.60"], None),
            ("p_min=0.65",              base_dflash + ["--spec-draft-n-max", "7", "--spec-draft-p-min", "0.65"], None),
            ("cost=0.25 (uniform)",     base_dflash + ["--spec-draft-n-max", "7", "--spec-draft-p-min", "0.00", "--spec-draft-exp-cost", "0.25"], {"LLAMA_DFLASH_EXP_PROXY": "uniform"}),
            ("cost=0.35 (router)",      base_dflash + ["--spec-draft-n-max", "7", "--spec-draft-p-min", "0.00", "--spec-draft-exp-cost", "0.35"], {"LLAMA_DFLASH_EXP_PROXY": "router"}),
            ("p=0.45 pen=0.20 (router)",base_dflash + ["--spec-draft-n-max", "7", "--spec-draft-p-min", "0.45", "--spec-draft-exp-penalty", "0.20"], {"LLAMA_DFLASH_EXP_PROXY": "router"}),
        ]
        prompts = PROMPTS
    elif stage == "telemetry":
        configs = [
            ("base n7 p=0.00",          base_dflash + ["--spec-draft-n-max", "7", "--spec-draft-p-min", "0.00"], {"LLAMA_MOE_STATS": "1"}),
            ("p_min=0.55",              base_dflash + ["--spec-draft-n-max", "7", "--spec-draft-p-min", "0.55"], {"LLAMA_MOE_STATS": "1"}),
            ("cost=0.18 (uniform)",     base_dflash + ["--spec-draft-n-max", "7", "--spec-draft-p-min", "0.00", "--spec-draft-exp-cost", "0.18"], {"LLAMA_MOE_STATS": "1", "LLAMA_DFLASH_EXP_PROXY": "uniform"}),
            ("cost=0.28 (hybrid)",      base_dflash + ["--spec-draft-n-max", "7", "--spec-draft-p-min", "0.00", "--spec-draft-exp-cost", "0.28"], {"LLAMA_MOE_STATS": "1"}),
        ]
        prompts = PROMPTS[:3]
    else:
        raise SystemExit(f"Unknown stage: {stage}")

    all_rows = []
    summaries = []
    for label, dargs, env_x in configs:
        rows = []
        for attempt in range(2):
            try:
                rows, _ = run_config(label, dargs, env_x, prompts=prompts)
                break
            except Exception as e:
                print(f"  [WARN] {label} attempt {attempt + 1} failed: {e}", flush=True)
                wait_gtt_reclaimed()
                time.sleep(4.0)
        all_rows.extend(rows)
        if rows:
            avg_tps = sum(r["tps"] for r in rows) / len(rows)
            avg_ms = sum(r["ms_per_step"] for r in rows) / len(rows)
            avg_dft = sum(r["avg_dft"] for r in rows) / len(rows)
            avg_acc = sum(r["acc_pct"] for r in rows) / len(rows)
            avg_len = sum(r["mean_len"] for r in rows) / len(rows)
            summaries.append((label, avg_tps, avg_ms, avg_dft, avg_acc, avg_len))

    fieldnames = ["config", "prompt", "tps", "prompt_tps", "pred_n", "pred_ms", "ms_per_step", "steps", "avg_dft", "acc_pct", "draft_n", "draft_acc", "mean_len", "hash"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print(f"\n=== SUMMARY ({stage}, saved to {csv_path}) ===", flush=True)
    print(f"{'Config':<26} | {'avg tok/s':>9} | {'ms/step':>7} | {'avg_dft':>7} | {'acc%':>6} | {'mean_len':>8}", flush=True)
    print("-" * 78, flush=True)
    for label, avg_tps, avg_ms, avg_dft, avg_acc, avg_len in summaries:
        print(f"{label:<26} | {avg_tps:9.2f} | {avg_ms:7.2f} | {avg_dft:7.2f} | {avg_acc:5.1f}% | {avg_len:8.2f}", flush=True)


if __name__ == "__main__":
    main()
