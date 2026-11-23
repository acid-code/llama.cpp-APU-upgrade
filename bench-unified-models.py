#!/usr/bin/env python3
"""
Unified GT105 APU benchmark suite (mainline b12629 + PR 32704 + PR 32218 + gf_res_prev[10] +
DFlash/DSpark host-sync + set_token_state fix + Part A approx-routing).

Benchmarks (100% iGPU offload, native F16 KV, ub=64, b=512, t=12, strict GTT/RAM guard):
  1. Qwen3.6-35B-A3B Q4_K_M + DFlash2 (n7 p0.55) -- verifies unified mainline matches build-dspine-v2
  2. Inkling-2-Mini Mainline Q4_K_XL (no-draft vs DSpark n1/n2) -- verifies 0 state mismatch + lossless hash
  3. Inkling-2-Mini Uncensored AD-Q4_K (no-draft vs DSpark n1) -- verifies 0 state mismatch + lossless hash
  4. Inkling-2-Mini Mainline Q3_K_XL (no-draft vs DSpark n1 + reasoning_effort low vs high wall-clock)
  5. Flint 39.8B-A2.5B UD-Q4_K_XL (no-draft raw coding/reasoning MoE comparison)
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
UNIFIED_BIN = os.environ.get(
    "UNIFIED_SERVER_BIN",
    "/home/acid/projects/llama-inkling-main/build/bin/llama-server",
)
PORT = int(os.environ.get("BENCH_PORT", "18092"))

MODELS = {
    "qwen36_q4": os.path.join(REPO_DIR, "models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    "inkling_q4": os.path.join(REPO_DIR, "models/Inkling-2-Mini-UD-Q4_K_XL.mainline.gguf"),
    "inkling_unc_q4": os.path.join(REPO_DIR, "models/Inkling-2-Mini-Uncensored-AD-Q4_K.gguf"),
    "inkling_q3": os.path.join(REPO_DIR, "models/Inkling-2-Mini-UD-Q3_K_XL.mainline.gguf"),
    "flint_q4": os.path.join(REPO_DIR, "models/flint-UD-Q4_K_XL.gguf"),
}

DRAFTS = {
    "qwen_dflash2": os.path.join(REPO_DIR, "models/qwen36-35b-a3b-dflash-Q8_0.gguf"),
    "inkling_dspark": os.path.join(REPO_DIR, "models/dspark-Inkling-2-Mini-Q8_0.gguf"),
}

PROMPTS_QUICK = [
    ("coding",
     "Write a clean C++17 lock-free single-producer single-consumer ring buffer with cache-line padding and explain memory ordering choices in detail."),
    ("reasoning",
     "A server has 48 GB unified DDR5 memory at 75 GB/s bandwidth and runs a 256-expert MoE model with 8 active experts per token across 40 layers. Derive the exact expected number of unique experts activated per layer in a batch of B independent tokens under uniform routing, and compute the memory traffic ratio vs batch size B=1 for B=2, 4, 6, 8."),
    ("agent_json",
     "You are a Linux SRE agent debugging an amdgpu MES queue reset on an AMD Radeon 780M running ROCm. Provide a structured JSON troubleshooting plan with commands, expected outputs, and root-cause decision branches."),
]

PROMPTS_FULL = PROMPTS_QUICK + [
    ("tech_512",
     "Explain unified memory Graphics Translation Table zero-copy DMA and Mixture-of-Experts routing on AMD Ryzen APUs in detail. " * 16),
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
    vals = []
    for path in glob.glob("/sys/class/drm/card*/device/mem_info_gtt_used"):
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
            "Stop llama-server / unload ollama models first."
        )
    return avail, gtt


def wait_gtt_reclaimed(timeout_s=45.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if get_gtt_used_gib() < 0.8:
            time.sleep(1.5)
            return
        time.sleep(0.5)
    raise RuntimeError("GTT memory was not reclaimed before timeout")


def wait_server_ready(proc, log_path, timeout_s=180.0):
    url = f"http://127.0.0.1:{PORT}/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early (rc={proc.returncode}); check {log_path}")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"llama-server did not become healthy within {timeout_s}s; check {log_path}")


def query_completion(prompt_text, n_predict):
    payload = {
        "prompt": prompt_text,
        "n_predict": n_predict,
        "temperature": 0.0,
        "seed": 42,
        "cache_prompt": False,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=240.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    wall_s = time.time() - t0
    tim = data.get("timings", {})
    content = data.get("content", "")
    pred_n = int(tim.get("predicted_n") or 0)
    pred_ms = float(tim.get("predicted_ms") or 0.0)
    pred_tps = float(tim.get("predicted_per_second") or 0.0)
    prompt_tps = float(tim.get("prompt_per_second") or 0.0)
    draft_n = int(tim.get("draft_n") or 0)
    draft_acc = int(tim.get("draft_n_accepted") or 0)
    acc_pct = (100.0 * draft_acc / draft_n) if draft_n > 0 else 0.0
    n_steps = max(pred_n - draft_acc, 1)
    return {
        "pred_n": pred_n,
        "pred_tps": pred_tps,
        "prompt_tps": prompt_tps,
        "ms_step": pred_ms / n_steps if pred_ms > 0 else 0.0,
        "acc_pct": acc_pct,
        "mean_len": pred_n / n_steps if pred_n > 0 else 0.0,
        "wall_s": wall_s,
        "sha12": hashlib.sha256(content.encode("utf-8")).hexdigest()[:12],
    }


def query_chat_effort(prompt_text, effort, max_tokens=512):
    payload = {
        "model": "bench",
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 42,
        "cache_prompt": False,
        "stream": False,
        "chat_template_kwargs": {"reasoning_effort": effort},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=240.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    wall_s = time.time() - t0
    tim = data.get("timings", {})
    msg = data.get("choices", [{}])[0].get("message", {})
    return {
        "pred_n": int(tim.get("predicted_n") or 0),
        "pred_tps": float(tim.get("predicted_per_second") or 0.0),
        "wall_s": wall_s,
        "reasoning_chars": len(msg.get("reasoning_content") or ""),
        "content_chars": len(msg.get("content") or ""),
    }


def run_config(cfg, prompts, n_gen, log_dir):
    model_path = MODELS[cfg["model"]]
    if not os.path.exists(model_path):
        print(f"SKIP {cfg['name']}: missing {model_path}", flush=True)
        return None
    draft_path = DRAFTS[cfg["draft"]] if cfg.get("draft") else None
    if draft_path and not os.path.exists(draft_path):
        print(f"SKIP {cfg['name']}: missing {draft_path}", flush=True)
        return None

    assert_memory_safe()
    log_path = os.path.join(log_dir, f"{cfg['name']}.log")
    cmd = [
        UNIFIED_BIN,
        "-m", model_path,
        "-ngl", "999",
        "-fit", "off",
        "-lm", "none",
        "-fa", "on",
        "-ub", "64",
        "-b", "512",
        "-t", "12",
        "-c", "4096",
        "-np", "1",
        "--jinja",
        "--host", "127.0.0.1",
        "--port", str(PORT),
    ]
    if draft_path:
        cmd += [
            "-md", draft_path,
            "-ngld", "999",
            "--spec-draft-n-max", str(cfg["n_max"]),
            "--spec-draft-n-min", "1",
            "--spec-draft-p-min", f"{cfg['p_min']:.2f}",
        ]
        if cfg.get("spec_type"):
            cmd += ["--spec-type", cfg["spec_type"]]

    env = os.environ.copy()
    env["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
    env["HIP_VISIBLE_DEVICES"] = "0"
    env.pop("LLAMA_APPROX_ROUTING", None)

    print(f"\n=== [{cfg['name']}] starting unified llama-server ===", flush=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, preexec_fn=os.setsid)
        try:
            wait_server_ready(proc, log_path)
            gtt_loaded = get_gtt_used_gib()
            rows = []
            for pname, ptext in prompts:
                r = query_completion(ptext, n_gen)
                r["prompt"] = pname
                rows.append(r)
                print(
                    f"  {pname:<11} | PP={r['prompt_tps']:6.1f} | Dec={r['pred_tps']:6.2f} t/s | "
                    f"ms/step={r['ms_step']:6.2f} | Acc={r['acc_pct']:5.1f}% | Len={r['mean_len']:4.2f} | sha={r['sha12']}",
                    flush=True,
                )

            effort_res = None
            if cfg.get("test_effort"):
                e_prompt = "Write a concise Python function to check if a number is prime."
                e_low = query_chat_effort(e_prompt, "low", max_tokens=512)
                e_high = query_chat_effort(e_prompt, "high", max_tokens=512)
                effort_res = {"low": e_low, "high": e_high}
                print(
                    f"  [reasoning_effort] low: {e_low['pred_n']} tok in {e_low['wall_s']:.1f}s "
                    f"(think={e_low['reasoning_chars']}c, ans={e_low['content_chars']}c) | "
                    f"high: {e_high['pred_n']} tok in {e_high['wall_s']:.1f}s "
                    f"(think={e_high['reasoning_chars']}c, ans={e_high['content_chars']}c)",
                    flush=True,
                )
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=15)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait(timeout=5)
                except Exception:
                    pass
            wait_gtt_reclaimed()

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        log_txt = f.read()
    state_err = log_txt.count("state size mismatch")
    realloc_lines = [ln.strip() for ln in log_txt.splitlines() if "graph reallocs" in ln]
    avg_dec = sum(r["pred_tps"] for r in rows) / len(rows)
    avg_pp = sum(r["prompt_tps"] for r in rows) / len(rows)
    avg_acc = sum(r["acc_pct"] for r in rows) / len(rows)
    avg_len = sum(r["mean_len"] for r in rows) / len(rows)
    hashes = {r["prompt"]: r["sha12"] for r in rows}

    print(
        f"  => AVG [{cfg['name']}]: PP={avg_pp:.1f} t/s | Dec={avg_dec:.2f} t/s | "
        f"Acc={avg_acc:.1f}% | Len={avg_len:.2f} | GTT={gtt_loaded:.1f} GiB | state_mismatch={state_err}",
        flush=True,
    )
    if realloc_lines:
        print(f"  => {realloc_lines[-1]}", flush=True)

    return {
        "name": cfg["name"],
        "model": cfg["model"],
        "avg_pp": avg_pp,
        "avg_dec": avg_dec,
        "avg_acc": avg_acc,
        "avg_len": avg_len,
        "gtt_gib": gtt_loaded,
        "state_mismatch": state_err,
        "hashes": hashes,
        "effort": effort_res,
    }


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
    prompts = PROMPTS_FULL if mode == "full" else PROMPTS_QUICK
    n_gen = 256 if mode == "full" else 128

    log_dir = "/tmp/bench_unified_logs"
    os.makedirs(log_dir, exist_ok=True)

    configs = [
        {"name": "qwen36_q4_dflash2_n7_p055", "model": "qwen36_q4",      "draft": "qwen_dflash2",   "n_max": 7, "p_min": 0.55},
        {"name": "flint_q4_nodraft",          "model": "flint_q4",       "draft": None},
        {"name": "inkling_q4_nodraft",        "model": "inkling_q4",     "draft": None},
        {"name": "inkling_q4_dspark_n1",      "model": "inkling_q4",     "draft": "inkling_dspark", "spec_type": "draft-dspark", "n_max": 1, "p_min": 0.00},
        {"name": "inkling_q4_dspark_n2",      "model": "inkling_q4",     "draft": "inkling_dspark", "spec_type": "draft-dspark", "n_max": 2, "p_min": 0.00},
        {"name": "inkling_unc_q4_nodraft",    "model": "inkling_unc_q4", "draft": None},
        {"name": "inkling_unc_q4_dspark_n1",  "model": "inkling_unc_q4", "draft": "inkling_dspark", "spec_type": "draft-dspark", "n_max": 1, "p_min": 0.00},
        {"name": "inkling_q3_nodraft",        "model": "inkling_q3",     "draft": None,             "test_effort": True},
        {"name": "inkling_q3_dspark_n1",      "model": "inkling_q3",     "draft": "inkling_dspark", "spec_type": "draft-dspark", "n_max": 1, "p_min": 0.00},
    ]

    results = []
    for cfg in configs:
        try:
            res = run_config(cfg, prompts, n_gen, log_dir)
            if res:
                results.append(res)
        except Exception as exc:
            print(f"ERROR [{cfg['name']}]: {exc}", flush=True)

    print("\n================ UNIFIED GT105 BENCHMARK SUMMARY ================", flush=True)
    print(f"{'Config':<28} | {'PP t/s':>7} | {'Dec t/s':>7} | {'Acc%':>6} | {'Len':>5} | {'GTT GiB':>7} | {'Mismatch':>8} | {'Lossless?':>10}", flush=True)
    print("-" * 98, flush=True)

    base_hashes = {}
    for r in results:
        if r["name"].endswith("_nodraft"):
            base_hashes[r["model"]] = r["hashes"]

    csv_path = os.path.join(REPO_DIR, f"bench_unified_{mode}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["config", "model", "avg_pp_tps", "avg_dec_tps", "avg_acc_pct", "avg_len", "gtt_gib", "state_mismatch", "lossless"])
        for r in results:
            lossless = "base"
            if not r["name"].endswith("_nodraft") and r["model"] in base_hashes:
                b = base_hashes[r["model"]]
                n_match = sum(1 for k, v in r["hashes"].items() if b.get(k) == v)
                lossless = f"{n_match}/{len(r['hashes'])}"
            print(
                f"{r['name']:<28} | {r['avg_pp']:7.1f} | {r['avg_dec']:7.2f} | {r['avg_acc']:5.1f}% | "
                f"{r['avg_len']:5.2f} | {r['gtt_gib']:7.1f} | {r['state_mismatch']:8d} | {lossless:>10}",
                flush=True,
            )
            w.writerow([r["name"], r["model"], f"{r['avg_pp']:.2f}", f"{r['avg_dec']:.2f}",
                        f"{r['avg_acc']:.2f}", f"{r['avg_len']:.2f}", f"{r['gtt_gib']:.2f}",
                        r["state_mismatch"], lossless])
    print(f"\nWrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
