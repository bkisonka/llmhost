#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Back-to-back замер llmhost (живой стек демона) vs Ollama на одной машине и
одних и тех же .gguf. Linux-версия scripts/bench_vs_ollama.py (методика
та же; стенд — 2 GPU).

  - обе стороны получают одну и ту же матрицу весов (Ollama — импорт
    через Modelfile FROM, делается снаружи: ollama create);
  - генерация: raw-continuation, temperature 0, n_predict 256; ток/с —
    из собственных счётчиков движков (timings llama.cpp через прокси,
    eval_count/eval_duration у Ollama);
  - prefill: уникальные промпты (повторные врут у обоих движков),
    короткий ~101 ток × 3, длинный ~7.9к ток × 2;
  - холодная загрузка: стороны предварительно выгружены, меряется время
    до фактической готовности;
  - между фазами — ожидание фактического освобождения VRAM по
    nvidia-smi, а не фиксированная пауза.

Запуск: python3 scripts/bench_vs_ollama_linux.py [--ollama-model ИМЯ]
Выход: таблица в stdout + JSON в scripts/bench_vs_ollama_linux_last.json
"""
import argparse
import http.client
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

PROXY_PORT = 5050   # OpenAI-порт llmhost (основной демон или изолированный)
CTRL_PORT = 7331
OLLAMA_PORT = 11434  # переопределяется --ollama-port (стенд другой версии)
DATA_DIR = os.path.expanduser("~/.local/share/llmhost")  # -> --data-dir

# Тот же промпт, что в движковом бенче llmhost (bench.cpp): ~150 токенов.
PROMPT = ("The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor "
          "jugs. How vexingly quick daft zebras jump! Sphinx of black quartz, judge my vow. "
          "Bright vixens jump; dozy fowl quack. Quick zephyrs blow, vexing daft Jim. Two "
          "driven jocks help fax my big quiz. Five quacking zephyrs jolt my wax bed. The "
          "job requires extra pluck and zeal from every young wage earner.")
PROMPT_SENTENCES = [x.strip() for x in PROMPT.replace(". ", "|").split("|") if x.strip()]
N_PREDICT = 256

LONG_WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
              "november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee "
              "zulu north south east west river mountain forest desert ocean island valley").split()

N_SHORT = 3
N_LONG = 2

LH_ID = "gemma-4-26B-A4B-it-UD-Q4_K_M"  # переопределяется --lh-id
NUM_CTX = 32768  # обе стороны: как в бенче 03.10


def token():
    with open(os.path.join(DATA_DIR, "token.txt")) as f:
        return f.read().strip()


TOKEN = token()
AUTH = {"Authorization": "Bearer " + TOKEN}


def short_prompt(seed):
    n = len(PROMPT_SENTENCES)
    rot = seed % n
    return ". ".join(PROMPT_SENTENCES[rot:] + PROMPT_SENTENCES[:rot]) + "."


def long_prompt(seed):
    import random
    ws = LONG_WORDS * 70
    random.Random(seed).shuffle(ws)
    return " ".join(ws)


def req(port, method, path, body=None, timeout=600, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    h = dict(headers or {})
    if body is not None:
        h.setdefault("Content-Type", "application/json")
    c.request(method, path, json.dumps(body) if body is not None else None, headers=h)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def gpu_free_mib():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return [int(x) for x in r.stdout.strip().splitlines()]


def wait_vram_clean(min_free=(11000, 21000), timeout_s=120):
    """Ждать фактического освобождения VRAM на ОБОИХ картах (замер 01.10:
    преждевременный старт — тихий спилл и обвал генерации)."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            free = gpu_free_mib()
            if len(free) >= 2 and all(f >= m for f, m in zip(free, min_free)):
                return True
        except Exception:
            return True
        time.sleep(2)
    print("  ! VRAM не освободилась: %s" % gpu_free_mib(), flush=True)
    return False


def lh_wait_state(model_id, want="loaded", timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st, data = req(CTRL_PORT, "GET", "/v0/models", timeout=30, headers=AUTH)
        if st == 200:
            for m in json.loads(data).get("models", []):
                if m.get("id") == model_id and m.get("state") == want:
                    return True
        time.sleep(1)
    return False


def lh_unload_all():
    req(CTRL_PORT, "POST", "/v0/unload", {}, timeout=120, headers=AUTH)
    wait_vram_clean()


def ollama_stop(name, port=None):
    """Выгрузить модель: API generate с keep_alive=0 — работает и для
    системного сервиса, и для стенда на нестандартном порту (CLI ollama
    stop ходит на OLLAMA_HOST по умолчанию и стенд не видит)."""
    p = port if port is not None else OLLAMA_PORT
    req(p, "POST", "/api/generate",
        {"model": name, "prompt": "", "keep_alive": 0}, timeout=120)


def ollama_version(port=None):
    p = port if port is not None else OLLAMA_PORT
    try:
        st, data = req(p, "GET", "/api/version", timeout=30)
        return json.loads(data).get("version", "?")
    except Exception:
        return "?"


def gguf_parts(path):
    """Файл или части шарда по возрастанию."""
    import glob
    if os.path.isfile(path):
        return [path]
    parts = sorted(glob.glob(path + "-00001-of-*.gguf"))
    return parts or [path]


def stream_ttft(port, path, body, headers, is_ndjson):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    h = dict(headers)
    h["Content-Type"] = "application/json"
    c.request("POST", path, json.dumps(body), headers=h)
    r = c.getresponse()
    t0 = time.time()
    ttft = None
    buf = b""
    while True:
        chunk = r.read(4096)
        if not chunk:
            break
        buf += chunk
        if is_ndjson:
            if b'"response"' in buf:
                ttft = time.time() - t0
                break
        else:
            if b"data: " in buf:
                ttft = time.time() - t0
                break
    try:
        while r.read(65536):
            pass
    except Exception:
        pass
    c.close()
    return ttft


def bench_llmhost(model_id):
    lh_unload_all()
    t0 = time.time()
    st, _ = req(CTRL_PORT, "POST", "/v0/load", {"name": model_id}, timeout=30, headers=AUTH)
    if st != 202 and st != 200:
        raise RuntimeError("llmhost /v0/load: %d" % st)
    if not lh_wait_state(model_id):
        raise RuntimeError("llmhost: %s не загрузился" % model_id)
    cold_load = time.time() - t0
    # b11800+: прокси пускает только /v1* и /health — сырой prompt живёт в
    # /v1/completions, тайминги те же. Нонс-хвост в gen-промптах — симметрично
    # стороне Ollama (у обеих слот с KV-переиспользованием, промпты уникальны).
    warm_ttft = stream_ttft(PROXY_PORT, "/v1/completions",
                            {"model": model_id, "prompt": PROMPT, "max_tokens": 64,
                             "temperature": 0, "stream": True},
                            AUTH, is_ndjson=False)
    gens = []
    for i in range(3):
        st, data = req(PROXY_PORT, "POST", "/v1/completions",
                       {"model": model_id, "prompt": PROMPT + " Note %d." % i,
                        "max_tokens": N_PREDICT, "temperature": 0},
                       timeout=600, headers=AUTH)
        t = json.loads(data).get("timings", {})
        if t.get("predicted_per_second"):
            gens.append((t["predicted_per_second"], t.get("prompt_per_second", 0)))
    shorts = []
    for i in range(N_SHORT):
        st, data = req(PROXY_PORT, "POST", "/v1/completions",
                       {"model": model_id, "prompt": short_prompt(i), "max_tokens": 1,
                        "temperature": 0}, timeout=600, headers=AUTH)
        t = json.loads(data).get("timings", {})
        if t.get("prompt_per_second"):
            shorts.append(t["prompt_per_second"])
    longs = []
    for i in range(N_LONG):
        st, data = req(PROXY_PORT, "POST", "/v1/completions",
                       {"model": model_id, "prompt": long_prompt(i), "max_tokens": 1,
                        "temperature": 0}, timeout=600, headers=AUTH)
        t = json.loads(data).get("timings", {})
        if t.get("prompt_per_second"):
            longs.append(t["prompt_per_second"])
    return {"cold_first_req_s": cold_load, "warm_ttft_s": warm_ttft, "runs": gens,
            "prefill_short": shorts, "long_prefill": longs}


def bench_ollama(model_name):
    ollama_stop(model_name)
    wait_vram_clean()
    t0 = time.time()
    st, data = req(OLLAMA_PORT, "POST", "/api/generate",
                   {"model": model_name, "prompt": "Hi", "stream": False,
                    "raw": True,
                    "options": {"num_predict": 4, "temperature": 0,
                                "num_ctx": NUM_CTX}},
                   timeout=900)
    cold_total = time.time() - t0
    j0 = json.loads(data)
    cold_load = j0.get("load_duration", 0) / 1e9 or cold_total
    warm_ttft = stream_ttft(OLLAMA_PORT, "/api/generate",
                            {"model": model_name, "prompt": PROMPT, "stream": True,
                             "raw": True,
                             "options": {"num_predict": 64, "temperature": 0,
                                         "num_ctx": NUM_CTX}},
                            {}, is_ndjson=True)
    gens = []
    for i in range(3):
        # Уникальный хвост-нонс: Ollama 0.20.5 на повторном промпте обрывает
        # генерацию (done:false без счётчиков — баг переиспользования KV,
        # воспроизводится стабильно; первый gen после загрузки всегда цел).
        st, data = req(OLLAMA_PORT, "POST", "/api/generate",
                       {"model": model_name, "prompt": PROMPT + " Note %d." % i,
                        "stream": False, "raw": True,
                        "options": {"num_predict": N_PREDICT, "temperature": 0,
                                    "num_ctx": NUM_CTX}}, timeout=900)
        j = json.loads(data)
        if st != 200 or j.get("error"):
            print("  ! ollama gen: %d %s" % (st, data[:300]), flush=True)
            continue
        if j.get("eval_count") and j.get("eval_duration"):
            gens.append((j["eval_count"] / (j["eval_duration"] / 1e9),
                         j.get("prompt_eval_count", 0) /
                         max(j.get("prompt_eval_duration", 1) / 1e9, 1e-9)))
    shorts = []
    for i in range(N_SHORT):
        st, data = req(OLLAMA_PORT, "POST", "/api/generate",
                       {"model": model_name, "prompt": short_prompt(10 + i),
                        "stream": False, "raw": True,
                        "options": {"num_predict": 1, "temperature": 0,
                                    "num_ctx": NUM_CTX}}, timeout=900)
        j = json.loads(data)
        if j.get("prompt_eval_count") and j.get("prompt_eval_duration"):
            shorts.append(j["prompt_eval_count"] / (j["prompt_eval_duration"] / 1e9))
    longs = []
    for i in range(N_LONG):
        st, data = req(OLLAMA_PORT, "POST", "/api/generate",
                       {"model": model_name, "prompt": long_prompt(100 + i),
                        "stream": False, "raw": True,
                        "options": {"num_predict": 1, "temperature": 0,
                                    "num_ctx": NUM_CTX}}, timeout=900)
        j = json.loads(data)
        if j.get("prompt_eval_count") and j.get("prompt_eval_duration"):
            longs.append(j["prompt_eval_count"] / (j["prompt_eval_duration"] / 1e9))
    return {"cold_first_req_s": cold_load, "warm_ttft_s": warm_ttft, "runs": gens,
            "prefill_short": shorts, "long_prefill": longs}


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0


def main():
    global OLLAMA_PORT, LH_ID, PROXY_PORT, CTRL_PORT, DATA_DIR, TOKEN, AUTH
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama-model", default="bench-gemma4-26b")
    ap.add_argument("--lh-id", default=LH_ID,
                    help="id модели в реестре llmhost для замера")
    ap.add_argument("--ollama-port", type=int, default=OLLAMA_PORT,
                    help="порт Ollama: 11434 системный, 11435 — стенд другой версии")
    ap.add_argument("--proxy-port", type=int, default=PROXY_PORT,
                    help="OpenAI-порт llmhost (5050 основной демон, 17340 изолированный)")
    ap.add_argument("--ctrl-port", type=int, default=CTRL_PORT,
                    help="control-порт llmhost (7331 основной демон, 17341 изолированный)")
    ap.add_argument("--data-dir", default=DATA_DIR,
                    help="каталог данных llmhost (для token.txt)")
    ap.add_argument("--side", choices=("both", "llmhost", "ollama"), default="both",
                    help="какую сторону мерить в этом прогоне")
    ap.add_argument("--json-out", default="bench_vs_ollama_linux_last.json",
                    help="имя файла отчёта в scripts/")
    ap.add_argument("--label", default="",
                    help="метка режима в meta (например, «harness»)")
    args = ap.parse_args()
    OLLAMA_PORT = args.ollama_port
    LH_ID = args.lh_id
    PROXY_PORT = args.proxy_port
    CTRL_PORT = args.ctrl_port
    DATA_DIR = args.data_dir
    TOKEN = token()
    AUTH = {"Authorization": "Bearer " + TOKEN}
    ol_ver = ollama_version()
    print("Ollama %s на :%d" % (ol_ver, OLLAMA_PORT), flush=True)
    st, sdata = req(CTRL_PORT, "GET", "/v0/status", timeout=30, headers=AUTH)
    lh_ver = json.loads(sdata).get("version", "?") if st == 200 else "?"
    st, edata = req(CTRL_PORT, "GET", "/v0/engine", timeout=30, headers=AUTH)
    lh_eng = json.loads(edata).get("release", "?") if st == 200 else "?"
    print("llmhost %s, движок %s" % (lh_ver, lh_eng), flush=True)

    # Чистый старт: резидент Ollama занимает карту, у llmhost не будет места
    ollama_stop(args.ollama_model)
    wait_vram_clean()

    r_lh = r_ol = None
    if args.side in ("both", "llmhost"):
        print("== llmhost (%s%s) ==" % (LH_ID, ", " + args.label if args.label else ""),
              flush=True)
        r_lh = bench_llmhost(LH_ID)
        lh_unload_all()
    if args.side in ("both", "ollama"):
        print("== ollama (%s) ==" % args.ollama_model, flush=True)
        r_ol = bench_ollama(args.ollama_model)
        ollama_stop(args.ollama_model)

    print()
    print("%-10s %-12s %-14s %-10s %-16s %-14s %-14s" %
          ("движок", "холодный, с", "ttft, с", "ген ток/с",
           "prefill ток/с", "длинный ток/с", "прогоны ген."))
    out = {}
    for engine, r in (("llmhost", r_lh), ("ollama", r_ol)):
        if r is None:
            continue
        gen = med([g[0] for g in r["runs"]])
        pre = med(r["prefill_short"] or [g[1] for g in r["runs"]])
        lp = med(r["long_prefill"])
        print("%-10s %-12.1f %-10s %-16.1f %-14.0f %-14.0f %s" %
              (engine, r["cold_first_req_s"],
               ("%.2f" % r["warm_ttft_s"]) if r["warm_ttft_s"] else "—",
               gen, pre, lp,
               [round(g[0], 1) for g in r["runs"]]))
        out[engine] = {
            "cold_first_req_s": round(r["cold_first_req_s"], 2),
            "warm_ttft_s": round(r["warm_ttft_s"], 3) if r["warm_ttft_s"] else None,
            "gen_tps_median": round(gen, 1),
            "prefill_short_median": round(pre, 1),
            "long_prefill_median": round(lp, 1),
            "gen_tps_runs": [round(g[0], 1) for g in r["runs"]],
            "prefill_short_runs": [round(x, 1) for x in r["prefill_short"]],
            "long_prefill_runs": [round(x, 1) for x in r["long_prefill"]],
        }
    out["meta"] = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "model": LH_ID,
        "label": args.label,
        "llmhost_version": lh_ver,
        "llmhost_engine_release": lh_eng,
        "llmhost_ctx": NUM_CTX, "ollama_ctx": NUM_CTX,
        "ollama_version": ol_ver,
        "stand": "Xeon, RTX 3060 12 ГБ + RTX PRO 4000 Blackwell 24 ГБ, драйвер 580.173.02",
    }
    with open(os.path.join(HERE, args.json_out), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nJSON: scripts/bench_vs_ollama_linux_last.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
