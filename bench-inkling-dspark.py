#!/usr/bin/env python3
"""
Memory-guarded llama-server benchmark for Qwen3.6 and Inkling 2 Mini.

The suite follows the APU benchmark conclusions in README.md:
  - keep all model and draft weights on the Radeon 780M
  - use ub=64 and native F16 KV
  - use p_min=0.55 for DFlash2
  - use DSpark p_min=0.00 because dflash.has_confidence_head=False
  - test the official DSpark n_max=3 case first, then n_max=5/7/8 ablations
  - never run a second model or server while a model is in GTT
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
import urllib.request


REPO_DIR = "/home/acid/projects/llama.cpp-APU-upgrade"
QWEN_SERVER_BIN = os.environ.get(
    "QWEN_SERVER_BIN",
    os.path.join(REPO_DIR, "build-dspine-v2/bin/llama-server"),
)
INKLING_SERVER_BIN = os.environ.get(
    "INKLING_SERVER_BIN",
    "/home/acid/.copilot/session-state/8878eae7-8748-404a-b33b-292f70671bc5/files/unsloth-b12588/llama-server",
)
INKLING_LIB_DIR = "/home/acid/.copilot/session-state/8878eae7-8748-404a-b33b-292f70671bc5/files/unsloth-b12588"
PORT = 18090
N_GEN = 256

MODELS = {
    "qwen": os.path.join(REPO_DIR, "models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    "inkling": os.path.join(REPO_DIR, "models/Inkling-2-Mini-UD-Q4_K_XL.gguf"),
    "inkling_uncensored": os.path.join(REPO_DIR, "models/Inkling-2-Mini-Uncensored-AD-Q4_K.gguf"),
}

QWEN_DRAFT = os.path.join(REPO_DIR, "models/qwen36-35b-a3b-dflash-Q8_0.gguf")
INKLING_DRAFT = os.path.join(REPO_DIR, "models/dspark-Inkling-2-Mini-Q8_0.gguf")

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


def get_mem_available_gib():
    with open("/proc/meminfo", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024.0 * 1024.0)
    return 0.0


def get_gtt_used_gib():
    paths = glob.glob("/sys/class/drm/card*/device/mem_info_gtt_used")
    vals = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
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
            "Stop llama-server or ollama and reclaim GTT first."
        )
    return avail, gtt


def wait_gtt_reclaimed(timeout_s=45.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if get_gtt_used_gib() < 0.5:
            time.sleep(2.0)
            return
        time.sleep(0.5)
    raise RuntimeError("GTT memory was not reclaimed before timeout")


def wait_server_ready(proc, log_path, timeout_s=180.0):
    url = f"http://127.0.0.1:{PORT}/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early (rc={proc.returncode}), check {log_path}")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"llama-server did not become healthy within {timeout_s}s")


def query_prompt(prompt_text, n_predict):
    url = f"http://127.0.0.1:{PORT}/v1/chat/completions"
    payload = {
        "model": "benchmark",
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": n_predict,
        "temperature": 0.0,
        "seed": 42,
        "cache_prompt": False,
        "stream": False,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=240.0) as response:
        data = json.loads(response.read().decode("utf-8"))

    timings = data.get("timings", {})
    message = data.get("choices", [{}])[0].get("message", {})
    text = (message.get("reasoning_content") or "") + (message.get("content") or "")

    predicted_n = int(timings.get("predicted_n") or 0)
    predicted_ms = float(timings.get("predicted_ms") or 0.0)
    predicted_tps = float(timings.get("predicted_per_second") or 0.0)
    prompt_tps = float(timings.get("prompt_per_second") or 0.0)
    draft_n = int(timings.get("draft_n") or 0)
    draft_accepted = int(timings.get("draft_n_accepted") or 0)

    steps = max(predicted_n - draft_accepted, 1)
    ms_per_step = predicted_ms / steps if predicted_ms > 0 else 0.0
    acc_pct = (100.0 * draft_accepted / draft_n) if draft_n > 0 else 0.0

    return {
        "tps": predicted_tps,
        "prompt_tps": prompt_tps,
        "pred_n": predicted_n,
        "pred_ms": predicted_ms,
        "ms_per_step": ms_per_step,
        "steps": steps,
        "avg_dft": draft_n / steps if draft_n > 0 else 0.0,
        "acc_pct": acc_pct,
        "draft_n": draft_n,
        "draft_acc": draft_accepted,
        "mean_len": predicted_n / steps,
        "hash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:10],
    }


def run_config(label, model, server_bin, draft_args, extra_env, prompts, n_gen):
    avail, gtt = assert_memory_safe()
    print(f"\n>>> [{label}] MemAvail={avail:.1f} GiB GTT={gtt:.2f} GiB", flush=True)

    env = os.environ.copy()
    env["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
    env["HIP_VISIBLE_DEVICES"] = "0"
    env.update(extra_env)

    cmd = [
        server_bin,
        "-m", model,
        "-ngl", "999",
        "-fit", "off",
        "-lm", "none",
        "-fa", "on",
        "-ub", "64",
        "-b", "512",
        "-t", "12",
        "-c", "4096",
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--no-webui",
    ] + draft_args

    safe_tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    log_path = f"/tmp/bench_inkling_{safe_tag}.log"
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

        rows = []
        try:
            wait_server_ready(proc, log_path)
            query_prompt("Say ok.", 16)

            for prompt_name, prompt_text in prompts:
                row = query_prompt(prompt_text, n_gen)
                row["config"] = label
                row["prompt"] = prompt_name
                rows.append(row)
                print(
                    f"  {prompt_name:<11} | {row['tps']:6.2f} tok/s | "
                    f"{row['ms_per_step']:6.2f} ms/step | dft={row['avg_dft']:4.2f} | "
                    f"acc={row['acc_pct']:5.1f}% | len={row['mean_len']:4.2f} | "
                    f"pp={row['prompt_tps']:5.1f} | h={row['hash']}",
                    flush=True,
                )
        finally:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)

    wait_gtt_reclaimed()
    return rows


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "full"
    if stage not in ("quick", "full"):
        raise SystemExit("Usage: python3 bench-inkling-dspark.py [quick|full]")

    required = [
        QWEN_SERVER_BIN,
        INKLING_SERVER_BIN,
        *MODELS.values(),
        QWEN_DRAFT,
        INKLING_DRAFT,
    ]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(missing))

    prompts = PROMPTS[:3] if stage == "quick" else PROMPTS
    n_gen = 128 if stage == "quick" else N_GEN

    qwen_dflash = [
        "-md", QWEN_DRAFT,
        "-ngld", "999",
        "--spec-type", "draft-dflash",
        "--spec-draft-n-min", "1",
    ]
    inkling_dspark = [
        "-md", INKLING_DRAFT,
        "-ngld", "999",
        "--spec-type", "draft-dspark",
        "--spec-draft-n-min", "1",
    ]

    configs = [
        ("qwen-no-draft", MODELS["qwen"], QWEN_SERVER_BIN, [], {}),
        ("qwen-dflash-n5-p55", MODELS["qwen"], QWEN_SERVER_BIN,
         qwen_dflash + ["--spec-draft-n-max", "5", "--spec-draft-p-min", "0.55"], {}),
        ("qwen-dflash-n7-p55", MODELS["qwen"], QWEN_SERVER_BIN,
         qwen_dflash + ["--spec-draft-n-max", "7", "--spec-draft-p-min", "0.55"], {}),
        ("inkling-no-draft", MODELS["inkling"], INKLING_SERVER_BIN, [], {"LD_LIBRARY_PATH": INKLING_LIB_DIR}),
        ("inkling-dspark-n3-p0", MODELS["inkling"], INKLING_SERVER_BIN,
         inkling_dspark + ["--spec-draft-n-max", "3", "--spec-draft-p-min", "0.00"], {"LD_LIBRARY_PATH": INKLING_LIB_DIR}),
        ("inkling-dspark-n5-p0", MODELS["inkling"], INKLING_SERVER_BIN,
         inkling_dspark + ["--spec-draft-n-max", "5", "--spec-draft-p-min", "0.00"], {"LD_LIBRARY_PATH": INKLING_LIB_DIR}),
        ("inkling-dspark-n7-p0", MODELS["inkling"], INKLING_SERVER_BIN,
         inkling_dspark + ["--spec-draft-n-max", "7", "--spec-draft-p-min", "0.00"], {"LD_LIBRARY_PATH": INKLING_LIB_DIR}),
        ("inkling-dspark-n8-p0", MODELS["inkling"], INKLING_SERVER_BIN,
         inkling_dspark + ["--spec-draft-n-max", "8", "--spec-draft-p-min", "0.00"], {"LD_LIBRARY_PATH": INKLING_LIB_DIR}),
        ("inkling-uncensored-no-draft", MODELS["inkling_uncensored"], INKLING_SERVER_BIN, [], {"LD_LIBRARY_PATH": INKLING_LIB_DIR}),
        ("inkling-uncensored-dspark-n3-p0", MODELS["inkling_uncensored"], INKLING_SERVER_BIN,
         inkling_dspark + ["--spec-draft-n-max", "3", "--spec-draft-p-min", "0.00"], {"LD_LIBRARY_PATH": INKLING_LIB_DIR}),
        ("inkling-uncensored-dspark-n5-p0", MODELS["inkling_uncensored"], INKLING_SERVER_BIN,
         inkling_dspark + ["--spec-draft-n-max", "5", "--spec-draft-p-min", "0.00"], {"LD_LIBRARY_PATH": INKLING_LIB_DIR}),
        ("inkling-uncensored-dspark-n7-p0", MODELS["inkling_uncensored"], INKLING_SERVER_BIN,
         inkling_dspark + ["--spec-draft-n-max", "7", "--spec-draft-p-min", "0.00"], {"LD_LIBRARY_PATH": INKLING_LIB_DIR}),
        ("inkling-uncensored-dspark-n8-p0", MODELS["inkling_uncensored"], INKLING_SERVER_BIN,
         inkling_dspark + ["--spec-draft-n-max", "8", "--spec-draft-p-min", "0.00"], {"LD_LIBRARY_PATH": INKLING_LIB_DIR}),
    ]

    csv_path = os.path.join(REPO_DIR, f"bench-inkling-{stage}.csv")
    all_rows = []
    summaries = []

    for label, model, server_bin, draft_args, extra_env in configs:
        for attempt in range(2):
            try:
                rows = run_config(label, model, server_bin, draft_args, extra_env, prompts, n_gen)
                break
            except Exception as error:
                print(f"  [WARN] {label} attempt {attempt + 1} failed: {error}", flush=True)
                wait_gtt_reclaimed()
                time.sleep(4.0)
        else:
            rows = []

        all_rows.extend(rows)
        if rows:
            summaries.append((
                label,
                sum(row["tps"] for row in rows) / len(rows),
                sum(row["ms_per_step"] for row in rows) / len(rows),
                sum(row["avg_dft"] for row in rows) / len(rows),
                sum(row["acc_pct"] for row in rows) / len(rows),
                sum(row["mean_len"] for row in rows) / len(rows),
            ))

    fieldnames = [
        "config", "prompt", "tps", "prompt_tps", "pred_n", "pred_ms",
        "ms_per_step", "steps", "avg_dft", "acc_pct", "draft_n",
        "draft_acc", "mean_len", "hash",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"\n=== SUMMARY ({stage}, saved to {csv_path}) ===", flush=True)
    print(f"{'Config':<38} | {'avg tok/s':>9} | {'ms/step':>7} | {'avg_dft':>7} | {'acc%':>6} | {'mean_len':>8}", flush=True)
    print("-" * 87, flush=True)
    for label, avg_tps, avg_ms, avg_dft, avg_acc, avg_len in summaries:
        print(
            f"{label:<38} | {avg_tps:9.2f} | {avg_ms:7.2f} | "
            f"{avg_dft:7.2f} | {avg_acc:5.1f}% | {avg_len:8.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
