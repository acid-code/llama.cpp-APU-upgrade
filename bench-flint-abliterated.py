#!/usr/bin/env python3
"""
Benchmark dima6312 vs mradermacher imatrix abliterated Flint-40B-A2.5B GGUFs
across quant sizes (Q4_K_M, IQ4_XS, Q3_K_M) and Flint-DFlash2 (n_max, p_min) grid.
"""

import csv
import importlib
import os
import subprocess
import sys
import time
import urllib.request

REPO_DIR = "/home/acid/projects/llama.cpp-APU-upgrade"
sys.path.insert(0, REPO_DIR)
B = importlib.import_module("bench-unified-models")

B.MODELS.update({
    "dima_q4km":    os.path.join(REPO_DIR, "models/flint-abliterated/dima-Q4_K_M.gguf"),
    "mrad_i1_q4km": os.path.join(REPO_DIR, "models/flint-abliterated/mrad-i1-Q4_K_M.gguf"),
    "mrad_i1_iq4xs":os.path.join(REPO_DIR, "models/flint-abliterated/mrad-i1-IQ4_XS.gguf"),
    "mrad_i1_q3km": os.path.join(REPO_DIR, "models/flint-abliterated/mrad-i1-Q3_K_M.gguf"),
})
B.DRAFTS["flint_dflash2"] = os.path.join(REPO_DIR, "models/Flint-DFlash2-Q8_0.gguf")


def free_ollama_and_agent():
    subprocess.run(["docker", "stop", "local-agent-local-agent-1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=b'{"model":"gemma3:4b","keep_alive":0}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5.0).read()
    except Exception:
        pass
    time.sleep(2.0)


def main():
    log_dir = "/tmp/bench_flint_abliterated"
    os.makedirs(log_dir, exist_ok=True)

    # Stage 1: Compare all 4 models (raw + n7 p0.60 + n7 p0.65 + n6 p0.60)
    models = ["dima_q4km", "mrad_i1_q4km", "mrad_i1_iq4xs", "mrad_i1_q3km"]
    stage1 = []
    for m in models:
        stage1.append({"name": f"{m}_nodraft",        "model": m, "draft": None})
        stage1.append({"name": f"{m}_dflash2_n7_p60", "model": m, "draft": "flint_dflash2", "n_max": 7, "p_min": 0.60})
        stage1.append({"name": f"{m}_dflash2_n7_p65", "model": m, "draft": "flint_dflash2", "n_max": 7, "p_min": 0.65})
        stage1.append({"name": f"{m}_dflash2_n6_p60", "model": m, "draft": "flint_dflash2", "n_max": 6, "p_min": 0.60})

    free_ollama_and_agent()
    results = []
    try:
        for cfg in stage1:
            try:
                r = B.run_config(cfg, B.PROMPTS_QUICK, 128, log_dir)
                if r:
                    results.append(r)
            except Exception as exc:
                print(f"ERROR [{cfg['name']}]: {exc}", flush=True)

        # Stage 2: Fine (n_max, p_min) sweep on the top 2 fastest models from Stage 1
        best_by_model = {}
        for r in results:
            if not r["name"].endswith("_nodraft"):
                m = r["model"]
                if m not in best_by_model or r["avg_dec"] > best_by_model[m]:
                    best_by_model[m] = r["avg_dec"]
        top2 = sorted(best_by_model.keys(), key=lambda m: best_by_model[m], reverse=True)[:2]
        print(f"\n=== Stage 2 fine sweep on top 2 models: {top2} ===", flush=True)

        stage2 = []
        for m in top2:
            stage2.append({"name": f"{m}_dflash2_n7_p55", "model": m, "draft": "flint_dflash2", "n_max": 7, "p_min": 0.55})
            stage2.append({"name": f"{m}_dflash2_n7_p70", "model": m, "draft": "flint_dflash2", "n_max": 7, "p_min": 0.70})
            stage2.append({"name": f"{m}_dflash2_n5_p60", "model": m, "draft": "flint_dflash2", "n_max": 5, "p_min": 0.60})
            stage2.append({"name": f"{m}_dflash2_n6_p65", "model": m, "draft": "flint_dflash2", "n_max": 6, "p_min": 0.65})

        for cfg in stage2:
            try:
                r = B.run_config(cfg, B.PROMPTS_QUICK, 128, log_dir)
                if r:
                    results.append(r)
            except Exception as exc:
                print(f"ERROR [{cfg['name']}]: {exc}", flush=True)
    finally:
        subprocess.run(["docker", "start", "local-agent-local-agent-1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    raw_by_model = {r["model"]: r["avg_dec"] for r in results if r["name"].endswith("_nodraft")}
    print("\n================ FLINT ABLITERATED + DFLASH2 SUMMARY ================", flush=True)
    print(f"{'Config':<28} | {'PP t/s':>7} | {'Dec t/s':>7} | {'vs Raw':>7} | {'Acc%':>6} | {'Len':>5} | {'GTT GiB':>7}", flush=True)
    print("-" * 84, flush=True)
    csv_path = os.path.join(REPO_DIR, "bench_flint_abliterated.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["config", "model", "avg_pp_tps", "avg_dec_tps", "vs_raw_pct", "avg_acc_pct", "avg_len", "gtt_gib"])
        for r in results:
            raw = raw_by_model.get(r["model"], 0.0)
            vs = (100.0 * (r["avg_dec"] / raw - 1.0)) if raw > 0 else 0.0
            print(
                f"{r['name']:<28} | {r['avg_pp']:7.1f} | {r['avg_dec']:7.2f} | {vs:+6.1f}% | "
                f"{r['avg_acc']:5.1f}% | {r['avg_len']:5.2f} | {r['gtt_gib']:7.1f}",
                flush=True,
            )
            w.writerow([r["name"], r["model"], f"{r['avg_pp']:.2f}", f"{r['avg_dec']:.2f}",
                        f"{vs:+.2f}", f"{r['avg_acc']:.2f}", f"{r['avg_len']:.2f}", f"{r['gtt_gib']:.2f}"])
    print(f"\nWrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
