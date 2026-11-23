#!/usr/bin/env python3
import base64
import csv
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

BIN = "/opt/llama-server/uncensored-bin/llama-server"
LIB = "/opt/llama-server/uncensored-bin"
MODEL = "/home/acid/projects/llama.cpp-APU-upgrade/models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"
MMPROJ = "/home/acid/projects/llama.cpp-APU-upgrade/models/mmproj-Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf"
DRAFT = "/home/acid/projects/llama.cpp-APU-upgrade/models/qwen36-35b-a3b-dflash-Q8_0.gguf"
TEST_IMG = os.environ.get("BENCH_TEST_IMG", "")
PORT = 8091
CSV_OUT = "/home/acid/projects/llama.cpp-APU-upgrade/bench_qwen36_uncensored.csv"

PROMPTS = [
    ("code", "Write a concise Python function implementing binary search on a sorted list with edge-case handling and return the index or -1."),
    ("reasoning", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Explain step by step."),
    ("sec_ctf", "Write a concise technical study guide for a Linux binary exploitation CTF explaining step-by-step how a classic stack buffer overflow overwrites the saved return address on x86-64, and how NX, ASLR, and stack canaries mitigate it."),
]

CONFIGS = [
    ("nodraft_nommproj", None, None, False),
    ("dflash2_n7_p55_mmproj", 7, 0.55, True),
    ("dflash2_n7_p60_mmproj", 7, 0.60, True),
    ("dflash2_n7_p50_mmproj", 7, 0.50, True),
    ("dflash2_n5_p55_mmproj", 5, 0.55, True),
]


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


def run_config(name, n_max, p_min, use_mmproj, test_vision=False, ctx=4096):
    unload_ollama()
    g0 = gtt_gib()
    m0 = mem_avail_gib()
    if g0 > 4.0 or m0 < 28.0:
        raise RuntimeError(f"Unsafe GPU memory before {name}: GTT={g0:.2f} GiB, MemAvail={m0:.2f} GiB")

    cmd = [
        BIN, "-m", MODEL,
        "-ngl", "999", "-fit", "off", "-lm", "none", "-fa", "on",
        "-ub", "64", "-b", "512", "-t", "12", "-c", str(ctx), "-np", "1",
        "--jinja", "--host", "127.0.0.1", "--port", str(PORT),
    ]
    if use_mmproj:
        cmd += ["--mmproj", MMPROJ]
    if n_max is not None:
        cmd += [
            "-md", DRAFT, "-ngld", "999",
            "--spec-draft-n-max", str(n_max),
            "--spec-draft-n-min", "1",
            "--spec-draft-p-min", f"{p_min:.2f}",
        ]

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LIB
    env["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
    env["HIP_VISIBLE_DEVICES"] = "0"

    log_path = f"/tmp/qwen36_unc_{name}.log"
    log_f = open(log_path, "w+")
    proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        t0 = time.time()
        while time.time() - t0 < 90:
            if proc.poll() is not None:
                raise RuntimeError(f"{name} exited early ({proc.returncode})")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(0.5)
        else:
            raise TimeoutError(f"{name} health timeout")

        load_s = time.time() - t0
        gtt = gtt_gib()

        pp_list, dec_list, acc_list, len_list = [], [], [], []
        samples = {}
        for pname, prompt in PROMPTS:
            payload = {
                "model": "qwen3.6-35b-a3b-uncensored",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 128,
                "temperature": 0.0,
                "seed": 42,
            }
            req = urllib.request.Request(
                f"http://127.0.0.1:{PORT}/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            resp = json.loads(urllib.request.urlopen(req, timeout=90).read())
            t = resp.get("timings", {})
            pp = float(t.get("prompt_per_second", 0.0))
            dec = float(t.get("predicted_per_second", 0.0))
            dn = int(t.get("draft_n", 0) or 0)
            da = int(t.get("draft_n_accepted", 0) or 0)
            acc = (100.0 * da / dn) if dn > 0 else 0.0
            pp_list.append(pp)
            dec_list.append(dec)
            acc_list.append(acc)
            msg = resp["choices"][0]["message"]
            rc = (msg.get("reasoning_content") or "").replace("\n", " ")[:220]
            ct = (msg.get("content") or "").replace("\n", " ")[:180]
            samples[pname] = f"dec={dec:.2f} | think={rc!r} | ans={ct!r}"

        vision_info = ""
        if test_vision and use_mmproj and os.path.exists(TEST_IMG):
            b64 = base64.b64encode(open(TEST_IMG, "rb").read()).decode()
            v_payload = {
                "model": "qwen3.6-35b-a3b-uncensored",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "List the model names visible in this screenshot concisely."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }],
                "max_tokens": 128,
                "temperature": 0.0,
                "seed": 42,
            }
            vt0 = time.time()
            vreq = urllib.request.Request(
                f"http://127.0.0.1:{PORT}/v1/chat/completions",
                data=json.dumps(v_payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            vresp = json.loads(urllib.request.urlopen(vreq, timeout=120).read())
            velap = time.time() - vt0
            vt = vresp.get("timings", {})
            vmsg = vresp["choices"][0]["message"]
            vout = (vmsg.get("content") or vmsg.get("reasoning_content") or "").replace("\n", " ")[:260]
            vision_info = f"VISION: wall={velap:.2f}s pp={vt.get('prompt_per_second',0):.1f} dec={vt.get('predicted_per_second',0):.2f} | out={vout!r}"

        gtt_peak = max(gtt, gtt_gib())
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
        time.sleep(1.5)

    reallocs = 0
    mean_lens = []
    for ln in open(log_path):
        if "graph reallocs" in ln:
            parts = ln.split("=")[1].split("/")
            try:
                reallocs += int(parts[0].strip()) + int(parts[1].split()[0].strip())
            except Exception:
                pass
        if "mean len =" in ln:
            try:
                mean_lens.append(float(ln.split("mean len =")[1].strip().split()[0]))
            except Exception:
                pass

    avg_pp = sum(pp_list) / len(pp_list)
    avg_dec = sum(dec_list) / len(dec_list)
    avg_acc = sum(acc_list) / len(acc_list)
    avg_len = (sum(mean_lens) / len(mean_lens)) if mean_lens else 1.0
    print(f"[{name:24s}] load={load_s:4.1f}s | PP={avg_pp:6.1f} | DEC={avg_dec:6.2f} tok/s | acc={avg_acc:5.1f}% | len={avg_len:4.2f} | GTT={gtt_peak:5.2f} GiB | reallocs={reallocs}")
    if name in ("nodraft_nommproj", "dflash2_n7_p55_mmproj"):
        for k, v in samples.items():
            print(f"    [{k}] {v}")
    if vision_info:
        print(f"    {vision_info}")
    sys.stdout.flush()
    return {
        "config": name,
        "avg_pp_tps": f"{avg_pp:.2f}",
        "avg_dec_tps": f"{avg_dec:.2f}",
        "avg_acc_pct": f"{avg_acc:.2f}",
        "avg_len": f"{avg_len:.2f}",
        "gtt_gib": f"{gtt_peak:.2f}",
        "reallocs": reallocs,
    }


def main():
    rows = []
    raw_dec = None
    for name, n_max, p_min, use_mmproj in CONFIGS:
        row = run_config(name, n_max, p_min, use_mmproj, test_vision=(name == "dflash2_n7_p55_mmproj"))
        if raw_dec is None:
            raw_dec = float(row["avg_dec_tps"])
        row["vs_raw_pct"] = f"{(float(row['avg_dec_tps']) / raw_dec - 1.0) * 100.0:+.2f}"
        rows.append(row)

    # Quick 32K ctx GTT check on best config
    best = max(rows[1:], key=lambda r: float(r["avg_dec_tps"]))
    print(f"\nWINNER: {best['config']} -> {best['avg_dec_tps']} tok/s ({best['vs_raw_pct']}% vs raw, acc={best['avg_acc_pct']}%, GTT={best['gtt_gib']} GiB)")

    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "avg_pp_tps", "avg_dec_tps", "vs_raw_pct", "avg_acc_pct", "avg_len", "gtt_gib", "reallocs"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
