#!/usr/bin/env python3
"""
Zero-download MoE active-expert sweep (top-8 baseline vs top-7, top-6, top-5) on deployed
b2143c676 (fa-tile-dequant-on-load + DFlash2 + q8_0 KV, -ub 256 -b 512):
  1. 4K 3-prompt speed & DFlash2 acceptance (top_k = 8, 7, 6, 5)
  2. 16-chunk (8,192-token) Perplexity on README.md via llama-perplexity (top_k = 8, 7, 6, 5)
  3. 24K cold gate at -c 262144 (top_k = 8 vs 6)
  4. Smartness Suite (code_exec 6/6, agent_json, general_qa, hebrew) at top_k = 8, 7, 6
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

SRV_BIN = "/opt/llama-server/bin/llama-server"
PPL_BIN = "/opt/llama-server/bin/llama-perplexity"
LIB = "/opt/llama-server/bin"
MODEL = "/home/acid/projects/llama.cpp-APU-upgrade/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
DRAFT = "/home/acid/projects/llama.cpp-APU-upgrade/models/qwen36-35b-a3b-dflash-Q8_0.gguf"
README_PATH = "/home/acid/projects/llama.cpp-APU-upgrade/README.md"
PORT = 8097
CSV_OUT = "/home/acid/projects/llama.cpp-APU-upgrade/bench_topk_qwen36.csv"

PROMPTS_4K = [
    ("code", "Write a concise Python function implementing an LRU cache with O(1) get/put without using functools.lru_cache, including edge-case handling."),
    ("reasoning", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Explain step by step."),
    ("agent_json", "Return a JSON object with keys 'plan' (3 short steps to debug a high-latency SQLite query) and 'tool_call' ({name, args}) to inspect schema."),
]

SMART_PROMPTS = [
    (
        "code_exec",
        False,
        "Write ONLY a Python code block defining `def merge_with_cooldown(intervals, cooldown):` that takes a list of `[start, end]` integer intervals (possibly unsorted or empty) and a non-negative integer `cooldown`. Merge two intervals if the next starts at or before `prev_end + cooldown`. Return sorted merged `[start, end]` pairs. Handle empty input and nested intervals.",
    ),
    (
        "agent_json",
        False,
        'Output ONLY valid JSON (no markdown fences, no prose) matching {"tool":"read_file","args":{"path":string,"start_line":integer,"end_line":integer},"reason":string} to inspect lines 120 to 160 of /opt/llama-server/run.sh for batch flags.',
    ),
    (
        "general_qa",
        False,
        "Answer concisely in 3 bullet points: (1) What year and treaty established the European Coal and Steel Community? (2) Why does dual-channel DDR5 bottleneck MoE decode on an APU? (3) What is the difference between a futex and a spinlock in Linux?",
    ),
    (
        "hebrew",
        False,
        "הסבר בעברית פשוטה וברורה ב-3 נקודות קצרות: מה ההבדל בין זיכרון RAM רגיל ל-GTT במעבד AMD APU, ולמה חשוב להשאיר מרווח זיכרון פנוי כשמריצים מודל שפה מקומי.",
    ),
]


def env_rocm():
    e = os.environ.copy()
    e["LD_LIBRARY_PATH"] = LIB
    e["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
    e["HIP_VISIBLE_DEVICES"] = "0"
    return e


def gtt_gib():
    for d in ("/sys/class/drm/card0/device", "/sys/class/drm/card1/device"):
        p = os.path.join(d, "mem_info_gtt_used")
        if os.path.exists(p):
            return int(open(p).read().strip()) / (1024 ** 3)
    return 0.0


def build_24k_prompt():
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
        return f"ERR ({type(e).__name__})"


def grade_agent_json(answer):
    try:
        obj = json.loads(answer.strip())
        ok = (
            obj.get("tool") == "read_file"
            and isinstance(obj.get("args"), dict)
            and obj["args"].get("path") == "/opt/llama-server/run.sh"
            and obj["args"].get("start_line") == 120
            and obj["args"].get("end_line") == 160
        )
        return "STRICT_PASS" if ok else "SCHEMA_MISMATCH"
    except Exception:
        return "INVALID_JSON"


def run_server(label, top_k, ctx=4096):
    cmd = [
        SRV_BIN, "-m", MODEL,
        "-md", DRAFT, "-ngld", "999",
        "--spec-draft-n-max", "7", "--spec-draft-n-min", "1", "--spec-draft-p-min", "0.55",
        "-ngl", "999", "-fit", "off", "-lm", "none", "-fa", "on",
        "-ctk", "q8_0", "-ctv", "q8_0",
        "-ub", "256", "-b", "512", "-t", "12", "-c", str(ctx), "-np", "1",
        "--jinja", "--host", "127.0.0.1", "--port", str(PORT),
    ]
    if top_k != 8:
        cmd += ["--override-kv", f"qwen35moe.expert_used_count=int:{top_k}"]

    log_path = f"/tmp/topk_{label}.log"
    lf = open(log_path, "w+")
    proc = subprocess.Popen(cmd, env=env_rocm(), stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 90:
        if proc.poll() is not None:
            lf.close()
            raise RuntimeError(f"{label} exited ({proc.returncode})")
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200:
                return proc, lf, log_path
        except Exception:
            time.sleep(0.5)
    os.killpg(proc.pid, signal.SIGTERM)
    lf.close()
    raise TimeoutError(f"{label} health timeout")


def stop_server(proc, lf):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=15)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass
    lf.close()
    time.sleep(1.5)


def query(prompt, max_tokens=128, enable_thinking=True):
    payload = {
        "model": "qwen3.6-35b-a3b",
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
    lp = (choice.get("logprobs") or {}).get("content") or []
    dn = int(t.get("draft_n", 0) or 0)
    da = int(t.get("draft_n_accepted", 0) or 0)
    return {
        "wall": wall,
        "in_toks": int(t.get("prompt_n", 0) or 0),
        "pp_tps": float(t.get("prompt_per_second", 0.0) or 0.0),
        "dec_tps": float(t.get("predicted_per_second", 0.0) or 0.0),
        "acc_pct": (100.0 * da / dn) if dn > 0 else 0.0,
        "nan": any(x.get("logprob") is None for x in lp),
        "content": msg.get("content") or "",
    }


def run_ppl(top_k):
    cmd = [
        PPL_BIN, "-m", MODEL, "-f", README_PATH,
        "-ngl", "999", "-fit", "off", "-lm", "none", "-fa", "on",
        "-ctk", "q8_0", "-ctv", "q8_0",
        "-ub", "256", "-b", "512", "-t", "12", "-c", "512", "--chunks", "16",
    ]
    if top_k != 8:
        cmd += ["--override-kv", f"qwen35moe.expert_used_count=int:{top_k}"]
    t0 = time.time()
    out = subprocess.check_output(cmd, env=env_rocm(), stderr=subprocess.STDOUT, text=True, timeout=300)
    elap = time.time() - t0
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)", out)
    ppl = float(m.group(1)) if m else -1.0
    err = float(m.group(2)) if m else -1.0
    used_m = re.search(r"n_expert_used\s*=\s*(\d+)", out)
    used_k = int(used_m.group(1)) if used_m else -1
    print(f"  [PPL top_k={top_k}] verified_n_expert_used={used_k} | PPL={ppl:.4f} +/- {err:.4f} ({elap:.1f}s)")
    sys.stdout.flush()
    return ppl, err, used_k


def main():
    rows = []
    print("=== 1. 4K 3-PROMPT SPEED + DFLASH2 ACCEPTANCE (q8_0 KV, -ub 256) ===")
    for k in (8, 7, 6, 5):
        proc, lf, lp = run_server(f"4k_k{k}", k, ctx=4096)
        try:
            res = [query(p, max_tokens=128, enable_thinking=True) for _, p in PROMPTS_4K]
            gtt = gtt_gib()
            smart_res = {}
            if k in (8, 7, 6):
                for sname, think_on, sprompt in SMART_PROMPTS:
                    sr = query(sprompt, max_tokens=512, enable_thinking=think_on)
                    g = grade_code_exec(sr["content"]) if sname == "code_exec" else (grade_agent_json(sr["content"]) if sname == "agent_json" else "")
                    smart_res[sname] = (g, sr["content"].replace("\n", " ")[:180])
        finally:
            stop_server(proc, lf)
        avg_pp = sum(r["pp_tps"] for r in res) / len(res)
        avg_dec = sum(r["dec_tps"] for r in res) / len(res)
        avg_acc = sum(r["acc_pct"] for r in res) / len(res)
        nan_any = any(r["nan"] for r in res)
        print(f"[4k_top{k}] PP={avg_pp:6.1f} | DEC={avg_dec:6.2f} tok/s | acc={avg_acc:5.1f}% | GTT={gtt:5.2f} GiB | NaN={nan_any}")
        for sname, (g, prev) in smart_res.items():
            print(f"    [{sname:10s}] {g:12s} | {prev!r}")
        sys.stdout.flush()
        rows.append({
            "phase": "4k", "top_k": k, "pp_tps": f"{avg_pp:.2f}", "dec_tps": f"{avg_dec:.2f}",
            "acc_pct": f"{avg_acc:.2f}", "ppl": "", "gtt_gib": f"{gtt:.2f}", "nan": nan_any,
            "code_grade": smart_res.get("code_exec", ("", ""))[0],
            "json_grade": smart_res.get("agent_json", ("", ""))[0],
        })

    print("\n=== 2. 16-CHUNK (8,192 TOK) TECHNICAL CORPUS PERPLEXITY ===")
    ppl_map = {}
    for k in (8, 7, 6, 5):
        ppl, err, used_k = run_ppl(k)
        ppl_map[k] = ppl
        for r in rows:
            if r["phase"] == "4k" and r["top_k"] == k:
                r["ppl"] = f"{ppl:.4f}"

    print("\n=== 3. 24K COLD GATE AT -c 262144 -ub 256 (top_k = 8 vs 6) ===")
    p24k = build_24k_prompt()
    for k in (8, 6):
        proc, lf, lp = run_server(f"24k_k{k}", k, ctx=262144)
        try:
            r = query(p24k, max_tokens=128, enable_thinking=True)
            gtt = gtt_gib()
        finally:
            stop_server(proc, lf)
        print(f"[24k_top{k}] in={r['in_toks']} | wall={r['wall']:5.1f}s | PP={r['pp_tps']:6.1f} | DEC={r['dec_tps']:5.2f} tok/s | acc={r['acc_pct']:5.1f}% | GTT={gtt:5.2f} GiB | NaN={r['nan']}")
        sys.stdout.flush()
        rows.append({
            "phase": "24k", "top_k": k, "pp_tps": f"{r['pp_tps']:.2f}", "dec_tps": f"{r['dec_tps']:.2f}",
            "acc_pct": f"{r['acc_pct']:.2f}", "ppl": f"{ppl_map.get(k, -1):.4f}", "gtt_gib": f"{gtt:.2f}",
            "nan": r["nan"], "code_grade": "", "json_grade": "",
        })

    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {CSV_OUT}")


if __name__ == "__main__":
    main()
