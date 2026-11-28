#!/usr/bin/env python3
"""
Phase 1 3-Way Apples-to-Apples Benchmark on b12217 (build-dspine-v2, 19e225579, -ub 256 -b 512):
  1. Qwen3.6-35B-A3B-UD-Q4_K_M  (20.61 GiB - current :8080 baseline)
  2. Qwen3.6-35B-A3B-UD-IQ4_XS  (16.51 GiB - pure IQ4_XS size/quality delta on Qwen3.6)
  3. Tiel-Coder-35B-A3B-UD-IQ4_XS (16.51 GiB - Tiel-Coder vs Qwen3.6 at exact same size)

Measures:
  - 4K Speed: Raw vs DFlash2 (n7, p_min sweep)
  - Smartness Suite (5 scored prompts, max_tokens=768, temp=0.0):
      [code_exec]   Write merge_intervals_with_gaps() in Python -> auto-executed against 6 unit tests!
      [bug_hunt]    Find the subtle race/deadlock bug in a short Python/SQLite snippet
      [agent_json]  Strict JSON tool-calling adherence (parsed & schema-validated automatically)
      [general_qa]  Factual & systems knowledge (tests Tiel-Coder's known trivia trade-off)
      [hebrew]      Hebrew instruction-following & concise technical explanation
  - Section 4.2.1 Cold 23.6K Validation Gate (cache_prompt=false, logprobs=true, NaN=False, 0 resets)
"""
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

BIN = "/opt/llama-server/bin/llama-server"
LIB = "/opt/llama-server/bin"
QWEN_Q4KM = "/home/acid/projects/llama.cpp-APU-upgrade/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
QWEN_IQ4XS = "/home/acid/projects/llama.cpp-APU-upgrade/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
TIEL_IQ4XS = "/home/acid/projects/llama.cpp-APU-upgrade/models/Tiel-Coder-35B-A3B-UD-IQ4_XS.gguf"
DRAFT = "/home/acid/projects/llama.cpp-APU-upgrade/models/qwen36-35b-a3b-dflash-Q8_0.gguf"
README_PATH = "/home/acid/projects/llama.cpp-APU-upgrade/README.md"
PORT = 8097
CSV_OUT = "/home/acid/projects/llama.cpp-APU-upgrade/bench_tiel_vs_qwen.csv"
SMART_OUT = "/home/acid/projects/llama.cpp-APU-upgrade/bench_tiel_vs_qwen_smartness.json"

PROMPTS_4K = [
    ("code", "Write a concise Python function implementing an LRU cache with O(1) get/put without using functools.lru_cache, including edge-case handling."),
    ("reasoning", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Explain step by step."),
    ("agent_json", "Return a JSON object with keys 'plan' (3 short steps to debug a high-latency SQLite query) and 'tool_call' ({name, args}) to inspect schema."),
]

SMARTNESS_PROMPTS = [
    (
        "code_exec",
        "Write ONLY a Python code block defining `def merge_with_cooldown(intervals, cooldown):` that takes a list of `[start, end]` integer intervals (possibly unsorted or empty) and a non-negative integer `cooldown`. Merge two intervals if the next starts at or before `prev_end + cooldown` (so `[1,3]` and `[5,7]` merge when `cooldown=2`, but not when `cooldown=1`). Return sorted merged `[start, end]` pairs. Handle empty input and nested intervals.",
    ),
    (
        "bug_hunt",
        "Find the TWO bugs in this Python snippet and give the minimal fix in <= 6 lines:\n```python\nasync def fetch_once(cache, locks, key, loader):\n    if key in cache:\n        return cache[key]\n    lock = locks.setdefault(key, asyncio.Lock())\n    async with lock:\n        val = await loader(key)\n        cache[key] = val\n        del locks[key]\n        return val\n```",
    ),
    (
        "agent_json",
        'Output ONLY valid JSON (no markdown fences, no prose) matching {"tool":"read_file","args":{"path":string,"start_line":integer,"end_line":integer},"reason":string} to inspect lines 120 to 160 of /opt/llama-server/run.sh for batch flags.',
    ),
    (
        "general_qa",
        "Answer concisely in 3 bullet points: (1) What year and treaty established the European Coal and Steel Community? (2) Why does dual-channel DDR5 bottleneck MoE decode on an APU even when compute CUs are idle? (3) What is the difference between a futex and a spinlock in Linux?",
    ),
    (
        "hebrew",
        "הסבר בעברית פשוטה וברורה ב-3 נקודות קצרות: מה ההבדל בין זיכרון RAM רגיל ל-GTT במעבד AMD APU, ולמה חשוב להשאיר מרווח זיכרון פנוי כשמריצים מודל שפה מקומי.",
    ),
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


def build_23k_prompt():
    txt = open(README_PATH, "r", encoding="utf-8", errors="ignore").read()
    chunks = []
    idx = 1
    while sum(len(c) for c in chunks) < 56000:
        chunks.append(f"\n\n--- ARCHIVE SECTION {idx} ---\n" + txt)
        idx += 1
    body = "".join(chunks)[:56000]
    return (
        "Read the following technical documentation carefully and summarize the 3 most important "
        "hardware findings and recommended flags for Radeon 780M in concise bullet points.\n\n"
        + body
    )


def grade_code_exec(answer):
    m = re.search(r"```(?:python)?\s*(.*?)```", answer, re.S)
    code = m.group(1) if m else answer
    ns = {}
    try:
        exec(code, ns)
        fn = ns["merge_with_cooldown"]
        tests = [
            (([], 2), []),
            (([[1, 3], [5, 7]], 2), [[1, 7]]),
            (([[1, 3], [5, 7]], 1), [[1, 3], [5, 7]]),
            (([[5, 10], [1, 12], [2, 4]], 0), [[1, 12]]),
            (([[10, 12], [1, 3], [4, 6]], 1), [[1, 6], [10, 12]]),
            (([[1, 2], [2, 3], [3, 4]], 0), [[1, 4]]),
        ]
        passed = sum(1 for (args, exp) in tests if fn(*args) == exp)
        return f"PASS {passed}/6" if passed == 6 else f"FAIL {passed}/6"
    except Exception as e:
        return f"ERR ({type(e).__name__}: {e})"


def grade_agent_json(answer):
    s = answer.strip()
    try:
        obj = json.loads(s)
        ok = (
            obj.get("tool") == "read_file"
            and isinstance(obj.get("args"), dict)
            and obj["args"].get("path") == "/opt/llama-server/run.sh"
            and obj["args"].get("start_line") == 120
            and obj["args"].get("end_line") == 160
            and isinstance(obj.get("reason"), str)
            and len(obj["reason"]) > 3
        )
        return "STRICT_PASS" if ok else f"SCHEMA_MISMATCH ({list(obj.keys())})"
    except Exception:
        return "INVALID_JSON"


def run_server(label, model_path, n_max, p_min, ctx=4096, ub=256):
    unload_ollama()
    g0 = gtt_gib()
    m0 = mem_avail_gib()
    if g0 > 4.0 or m0 < 24.0:
        raise RuntimeError(f"Unsafe GPU memory before {label}: GTT={g0:.2f} GiB, MemAvail={m0:.2f} GiB")

    cmd = [
        BIN, "-m", model_path,
        "-ngl", "999", "-fit", "off", "-lm", "none", "-fa", "on",
        "-ub", str(ub), "-b", "512", "-t", "12", "-c", str(ctx), "-np", "1",
        "--jinja", "--host", "127.0.0.1", "--port", str(PORT),
    ]
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

    log_path = f"/tmp/tiel3way_{label}.log"
    log_f = open(log_path, "w+")
    proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 120:
        if proc.poll() is not None:
            log_f.close()
            raise RuntimeError(f"{label} exited early ({proc.returncode})")
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200:
                return proc, log_f, log_path, time.time() - t0
        except Exception:
            time.sleep(0.5)
    os.killpg(proc.pid, signal.SIGTERM)
    log_f.close()
    raise TimeoutError(f"{label} health timeout")


def stop_server(proc, log_f):
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


def query(prompt, max_tokens=128, enable_thinking=True):
    payload = {
        "model": "test",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 42,
        "cache_prompt": False,
        "logprobs": True,
    }
    if not enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=300).read())
    wall = time.time() - t0
    t = resp.get("timings", {})
    choice = resp["choices"][0]
    msg = choice["message"]
    lp_items = (choice.get("logprobs") or {}).get("content") or []
    nan_lp = any(item.get("logprob") is None for item in lp_items)
    dn = int(t.get("draft_n", 0) or 0)
    da = int(t.get("draft_n_accepted", 0) or 0)
    return {
        "wall": wall,
        "in_toks": int(t.get("prompt_n", 0) or 0),
        "pp_tps": float(t.get("prompt_per_second", 0.0) or 0.0),
        "dec_tps": float(t.get("predicted_per_second", 0.0) or 0.0),
        "acc_pct": (100.0 * da / dn) if dn > 0 else 0.0,
        "nan": nan_lp,
        "think": msg.get("reasoning_content") or "",
        "content": msg.get("content") or "",
    }


def parse_log_stats(log_path):
    reallocs = 0
    mean_lens = []
    for ln in open(log_path):
        if "graph reallocs" in ln:
            try:
                parts = ln.split("=")[1].split("/")
                reallocs += int(parts[0].strip()) + int(parts[1].split()[0].strip())
            except Exception:
                pass
        if "mean len =" in ln:
            try:
                mean_lens.append(float(ln.split("mean len =")[1].strip().split()[0]))
            except Exception:
                pass
    return reallocs, (sum(mean_lens) / len(mean_lens)) if mean_lens else 1.0


def bench_speed(label, model_path, n_max, p_min, ctx=4096, prompts=PROMPTS_4K, max_tokens=128):
    resets_before = gpu_reset_count()
    proc, log_f, log_path, load_s = run_server(label, model_path, n_max, p_min, ctx=ctx)
    try:
        gtt_peak = gtt_gib()
        res = []
        for _, p in prompts:
            r = query(p, max_tokens=max_tokens)
            gtt_peak = max(gtt_peak, gtt_gib())
            res.append(r)
    finally:
        stop_server(proc, log_f)
    resets = max(0, gpu_reset_count() - resets_before) if resets_before >= 0 else -1
    reallocs, avg_len = parse_log_stats(log_path)
    avg_pp = sum(r["pp_tps"] for r in res) / len(res)
    avg_dec = sum(r["dec_tps"] for r in res) / len(res)
    avg_acc = sum(r["acc_pct"] for r in res) / len(res)
    nan_any = any(r["nan"] for r in res)
    tot_wall = sum(r["wall"] for r in res)
    in_toks = res[-1]["in_toks"]
    print(
        f"[{label:28s}] load={load_s:4.1f}s | in={in_toks:5d} | PP={avg_pp:6.1f} | DEC={avg_dec:6.2f} tok/s | "
        f"acc={avg_acc:5.1f}% | len={avg_len:4.2f} | GTT={gtt_peak:5.2f} GiB | wall={tot_wall:5.1f}s | "
        f"NaN={nan_any} | resets={resets} | reallocs={reallocs}"
    )
    sys.stdout.flush()
    return {
        "label": label, "ctx": ctx, "in_toks": in_toks,
        "avg_pp_tps": f"{avg_pp:.2f}", "avg_dec_tps": f"{avg_dec:.2f}",
        "avg_acc_pct": f"{avg_acc:.2f}", "avg_len": f"{avg_len:.2f}",
        "gtt_gib": f"{gtt_peak:.2f}", "wall_s": f"{tot_wall:.2f}",
        "nan": nan_any, "resets": resets, "reallocs": reallocs,
    }


def bench_smartness(model_tag, model_path, n_max, p_min):
    print(f"\n--- SMARTNESS SUITE: {model_tag} (n_max={n_max}, p_min={p_min}) ---")
    proc, log_f, _, _ = run_server(f"smart_{model_tag}", model_path, n_max, p_min, ctx=8192)
    out = {}
    try:
        for pname, prompt in SMARTNESS_PROMPTS:
            # Give reasoning room so final answer is never truncated
            r = query(prompt, max_tokens=1536, enable_thinking=(pname != "agent_json"))
            ans = r["content"].strip()
            grade = ""
            if pname == "code_exec":
                grade = grade_code_exec(ans)
            elif pname == "agent_json":
                grade = grade_agent_json(ans)
            out[pname] = {
                "dec_tps": round(r["dec_tps"], 2),
                "acc_pct": round(r["acc_pct"], 1),
                "wall_s": round(r["wall"], 1),
                "grade": grade,
                "answer": ans,
                "think_len": len(r["think"]),
            }
            preview = ans.replace("\n", " ")[:220]
            print(f"  [{pname:10s}] dec={r['dec_tps']:5.2f} t/s | acc={r['acc_pct']:4.1f}% | wall={r['wall']:4.1f}s | {grade:16s} | {preview!r}")
            sys.stdout.flush()
    finally:
        stop_server(proc, log_f)
    return out


def main():
    rows = []
    print("=== 1. 4K 3-WAY SPEED & DFLASH2 CORRELATION (b12217, -ub 256 -b 512) ===")
    rows.append(bench_speed("qwen_q4km_nodraft", QWEN_Q4KM, None, None))
    rows.append(bench_speed("qwen_q4km_dflash_n7_p55", QWEN_Q4KM, 7, 0.55))

    rows.append(bench_speed("qwen_iq4xs_nodraft", QWEN_IQ4XS, None, None))
    rows.append(bench_speed("qwen_iq4xs_dflash_n7_p55", QWEN_IQ4XS, 7, 0.55))
    rows.append(bench_speed("qwen_iq4xs_dflash_n7_p60", QWEN_IQ4XS, 7, 0.60))

    rows.append(bench_speed("tiel_iq4xs_nodraft", TIEL_IQ4XS, None, None))
    tiel_best_p = None
    tiel_best_dec = 0.0
    for p_min in (0.50, 0.55, 0.60, 0.65):
        r = bench_speed(f"tiel_iq4xs_dflash_n7_p{int(p_min*100)}", TIEL_IQ4XS, 7, p_min)
        rows.append(r)
        if not r["nan"] and float(r["avg_dec_tps"]) > tiel_best_dec:
            tiel_best_dec = float(r["avg_dec_tps"])
            tiel_best_p = p_min

    tiel_raw = float([r for r in rows if r["label"] == "tiel_iq4xs_nodraft"][0]["avg_dec_tps"])
    tiel_use_draft = tiel_best_p is not None and tiel_best_dec > tiel_raw * 1.03

    print(f"\n=== 2. SECTION 4.2.1 COLD 23.6K VALIDATION GATE (Tiel use_draft={tiel_use_draft}, p_min={tiel_best_p}) ===")
    p23k = [("readme_23k", build_23k_prompt())]
    rows.append(bench_speed("qwen_iq4xs_23k_gate", QWEN_IQ4XS, 7, 0.55, ctx=32768, prompts=p23k, max_tokens=96))
    rows.append(bench_speed(
        "tiel_iq4xs_23k_gate", TIEL_IQ4XS,
        7 if tiel_use_draft else None,
        tiel_best_p if tiel_use_draft else None,
        ctx=32768, prompts=p23k, max_tokens=96,
    ))

    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved speed CSV -> {CSV_OUT}")

    print("\n=== 3. SMARTNESS SUITE (Code Exec, Bug Hunt, Strict JSON, General/Systems QA, Hebrew) ===")
    smart = {}
    smart["qwen_q4km"] = bench_smartness("qwen_q4km", QWEN_Q4KM, 7, 0.55)
    smart["qwen_iq4xs"] = bench_smartness("qwen_iq4xs", QWEN_IQ4XS, 7, 0.55)
    smart["tiel_iq4xs"] = bench_smartness(
        "tiel_iq4xs", TIEL_IQ4XS,
        7 if tiel_use_draft else None,
        tiel_best_p if tiel_use_draft else None,
    )
    with open(SMART_OUT, "w", encoding="utf-8") as f:
        json.dump(smart, f, indent=2, ensure_ascii=False)
    print(f"\nSaved smartness report -> {SMART_OUT}")


if __name__ == "__main__":
    main()
