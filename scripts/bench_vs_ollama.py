#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Честный замер llmhost vs Ollama на ОДНОЙ машине и ОДНИХ И ТЕХ ЖЕ .gguf.

Методология:
  - обе стороны получают одинаковые GGUF (Ollama — через Modelfile FROM);
  - генерация: raw-continuation без шаблона чата (у Ollama "raw": true,
    у llmhost — /completion), temperature 0, n_predict 256; ток/с берём из
    собственных счётчиков движков (eval_count/eval_duration у Ollama,
    timings у llama.cpp) — без внешнего подсчёта;
  - холодный старт: обе стороны выгружаются, замеряется время до первого
    токена стримом (включает загрузку весов);
  - перекрёстное переключение: A резидентна → запрос B (стрим до первого
    токена), затем обратно A — видно, кто держит обе резидентно;
  - между фазами движки выгружаются, пауза 4 с — VRAM не делится.

Запуск: python scripts/bench_vs_ollama.py [--ollama-dir E:\\ollama-bench]
Выход: таблица в stdout + JSON в scripts/bench_vs_ollama_last.json
"""
import argparse
import http.client
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

PROXY_PORT = 7330
CTRL_PORT = 7331
OLLAMA_PORT = 11434

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


def short_prompt(seed):
    """Тот же объём, что PROMPT (~101 ток), порядок предложений сдвинут:
    первое предложение всегда другое — KV-кэш Ollama не сработает, а у нас
    и так cache_prompt: false. Полная переоценка у обеих сторон."""
    n = len(PROMPT_SENTENCES)
    rot = seed % n
    return ". ".join(PROMPT_SENTENCES[rot:] + PROMPT_SENTENCES[:rot]) + "."


N_SHORT = 3  # прогонов короткого prefill

def long_prompt(seed):
    """~7.9k токенов, уникальный на каждый прогон: KV-кэш обеих сторон не должен
    врать (у Ollama повтор промпта режет работу вдвое при том же prompt_eval_count)."""
    import random
    ws = LONG_WORDS * 70
    random.Random(seed).shuffle(ws)
    return " ".join(ws)

N_LONG = 2  # прогонов длинного prefill


MODELS = {  # ключевое имя → (id llmhost, имя Ollama)
    "gemma-4-12b": ("gemma-4-12b-it-UD-Q5_K_XL", "bench-gemma4-12b"),
    "qwen3-0.6b": ("Qwen3-0.6B-Q8_0", "bench-qwen3-06b"),
}


def token():
    with open(os.path.join(ROOT, "build", "Release", "data", "token.txt")) as f:
        return f.read().strip()


TOKEN = token()


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


def llmhost_unload_all():
    req(CTRL_PORT, "POST", "/v0/unload", {}, timeout=120,
        headers={"Authorization": "Bearer " + TOKEN})
    wait_vram_clean()


def wait_vram_clean(min_free_mib=10200, timeout_s=90):
    """Ждать фактического освобождения VRAM: WDDM отдаёт память убитого
    процесса с задержкой, и «сон 4 с» однажды привёл к тихому спиллу в
    shared-память и падению генерации в 7 раз (замер 01.10)."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True)
            free = int(r.stdout.strip().splitlines()[0])
            if free >= min_free_mib:
                return True
        except Exception:
            return True  # нет nvidia-smi — работаем как раньше
        time.sleep(2)
    print("  ! VRAM не освободилась до %d МиБ — замер может быть занижен" % min_free_mib,
          flush=True)
    return False


def llmhost_wait_state(model_id, want="loaded", timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st, data = req(CTRL_PORT, "GET", "/v0/models", timeout=30,
                       headers={"Authorization": "Bearer " + TOKEN})
        if st == 200:
            for m in json.loads(data).get("models", []):
                if m.get("id") == model_id and m.get("state") == want:
                    return True
        time.sleep(1)
    return False


class Ollama:
    def __init__(self, ollama_dir):
        self.exe = os.path.join(ollama_dir, "app", "ollama.exe")
        self.env = dict(os.environ, OLLAMA_HOST="127.0.0.1:%d" % OLLAMA_PORT,
                        OLLAMA_MODELS=os.path.join(ollama_dir, "models"))

    def cli(self, *args):
        subprocess.run([self.exe] + list(args), env=self.env,
                       capture_output=True, timeout=120)

    def stop(self, name):
        self.cli("stop", name)

    def stop_all(self):
        for _, name in MODELS.values():
            self.stop(name)
        wait_vram_clean()


def stream_ttft(port, path, body, headers, is_ndjson):
    """Время до первого содержательного куска стрима. Возвращает (ttft, ok)."""
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
    # добить соединение (AbortController не нужен: читаем до конца в фоне)
    try:
        while r.read(65536):
            pass
    except Exception:
        pass
    c.close()
    return ttft


def bench_llmhost(model_id):
    """Холодная загрузка /v0/load→loaded, ttft стримом, 3 прогона генерации."""
    auth = {"Authorization": "Bearer " + TOKEN}
    llmhost_unload_all()
    t0 = time.time()
    req(CTRL_PORT, "POST", "/v0/load", {"name": model_id}, timeout=30, headers=auth)
    if not llmhost_wait_state(model_id):
        raise RuntimeError("llmhost: %s не загрузился" % model_id)
    cold_load = time.time() - t0
    # ttft на загруженной (стрим до первого куска)
    warm_ttft = stream_ttft(PROXY_PORT, "/completion",
                            {"prompt": PROMPT, "n_predict": 64, "temperature": 0,
                             "stream": True, "cache_prompt": False},
                            auth, is_ndjson=False)
    # три прогона генерации — точные ток/с из timings
    gens = []
    for _ in range(3):
        st, data = req(PROXY_PORT, "POST", "/completion",
                       {"prompt": PROMPT, "n_predict": N_PREDICT, "temperature": 0,
                        "cache_prompt": False}, timeout=600, headers=auth)
        j = json.loads(data)
        t = j.get("timings", {})
        if t.get("predicted_per_second"):
            gens.append((t["predicted_per_second"], t.get("prompt_per_second", 0),
                         t.get("predicted_ms", 0) / 1000))
    # короткий prefill: уникальные промпты, полная переоценка у обеих сторон
    shorts = []
    for i in range(N_SHORT):
        st, data = req(PROXY_PORT, "POST", "/completion",
                       {"prompt": short_prompt(i), "n_predict": 1, "temperature": 0,
                        "cache_prompt": False}, timeout=600, headers=auth)
        t = json.loads(data).get("timings", {})
        if t.get("prompt_per_second"):
            shorts.append(t["prompt_per_second"])
    # длинный prefill (уникальные промпты — кэш обеих сторон не врёт)
    longs = []
    for i in range(N_LONG):
        st, data = req(PROXY_PORT, "POST", "/completion",
                       {"prompt": long_prompt(i), "n_predict": 1, "temperature": 0,
                        "cache_prompt": False}, timeout=600, headers=auth)
        t = json.loads(data).get("timings", {})
        if t.get("prompt_per_second"):
            longs.append(t["prompt_per_second"])
    return {"cold_first_req_s": cold_load, "warm_ttft_s": warm_ttft, "runs": gens,
            "prefill_short": shorts, "long_prefill": longs}


def bench_ollama(ol, model_name):
    """Холодная загрузка: собственная load_duration Ollama + общий таймер."""
    ol.stop_all()
    t0 = time.time()
    st, data = req(OLLAMA_PORT, "POST", "/api/generate",
                   {"model": model_name, "prompt": "Hi", "stream": False,
                    "raw": True, "options": {"num_predict": 4, "temperature": 0}},
                   timeout=600)
    cold_total = time.time() - t0
    j0 = json.loads(data)
    cold_load = j0.get("load_duration", 0) / 1e9 or cold_total
    # ttft на загруженной
    warm_ttft = stream_ttft(OLLAMA_PORT, "/api/generate",
                            {"model": model_name, "prompt": PROMPT, "stream": True,
                             "raw": True, "options": {"num_predict": 64,
                                                      "temperature": 0}},
                            {}, is_ndjson=True)
    gens = []
    for _ in range(3):
        st, data = req(OLLAMA_PORT, "POST", "/api/generate",
                       {"model": model_name, "prompt": PROMPT, "stream": False,
                        "raw": True, "options": {"num_predict": N_PREDICT,
                                                 "temperature": 0}}, timeout=600)
        j = json.loads(data)
        if j.get("eval_count") and j.get("eval_duration"):
            gens.append((j["eval_count"] / (j["eval_duration"] / 1e9),
                         j.get("prompt_eval_count", 0) /
                         max(j.get("prompt_eval_duration", 1) / 1e9, 1e-9),
                         j["eval_duration"] / 1e9))
    # короткий prefill: уникальные промпты (повторы у них режутся их кэшем:
    # замер 01.10 — один промпт 3.3-33.9 мс при count=101)
    shorts = []
    for i in range(N_SHORT):
        st, data = req(OLLAMA_PORT, "POST", "/api/generate",
                       {"model": model_name, "prompt": short_prompt(i), "stream": False,
                        "raw": True, "options": {"num_predict": 1, "temperature": 0}},
                       timeout=600)
        j = json.loads(data)
        if j.get("prompt_eval_count") and j.get("prompt_eval_duration"):
            shorts.append(j["prompt_eval_count"] / (j["prompt_eval_duration"] / 1e9))
    # длинный prefill: уникальные промпты — их кэш на повторе режет работу
    longs = []
    for i in range(N_LONG):
        st, data = req(OLLAMA_PORT, "POST", "/api/generate",
                       {"model": model_name, "prompt": long_prompt(100 + i), "stream": False,
                        "raw": True, "options": {"num_predict": 1, "temperature": 0}},
                       timeout=600)
        j = json.loads(data)
        if j.get("prompt_eval_count") and j.get("prompt_eval_duration"):
            longs.append(j["prompt_eval_count"] / (j["prompt_eval_duration"] / 1e9))
    return {"cold_first_req_s": cold_load, "warm_ttft_s": warm_ttft, "runs": gens,
            "prefill_short": shorts, "long_prefill": longs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama-dir", default=r"E:\ollama-bench")
    args = ap.parse_args()
    ol = Ollama(args.ollama_dir)

    results = {}
    for key, (lh_id, ol_name) in MODELS.items():
        print("== %s ==" % key, flush=True)
        # llmhost: выгрузить всё у Ollama, замерить, выгрузить у llmhost
        ol.stop_all()
        print("  llmhost…", flush=True)
        results[("llmhost", key)] = bench_llmhost(lh_id)
        llmhost_unload_all()
        print("  ollama…", flush=True)
        results[("ollama", key)] = bench_ollama(ol, ol_name)
        ol.stop_all()

    print()
    print("%-10s %-12s %-18s %-12s %-14s" %
          ("движок", "модель", "холодный 1-й, с", "ttft, с", "ген. ток/с (медиана)"))
    out = {}
    for (engine, key), r in results.items():
        gens = sorted(g[0] for g in r["runs"])
        med = gens[len(gens) // 2] if gens else 0
        pre = sorted(r.get("prefill_short") or [g[1] for g in r["runs"]])
        premed = pre[len(pre) // 2] if pre else 0
        lp = sorted(r.get("long_prefill", []))
        lpmed = lp[len(lp) // 2] if lp else 0
        ttft = r["warm_ttft_s"]
        print("%-10s %-12s %-18.1f %-12s %-14.1f  (prefill %.0f, длинный %.0f)" %
              (engine, key, r["cold_first_req_s"],
               ("%.2f" % ttft) if ttft else "—", med, premed, lpmed))
        out["%s/%s" % (engine, key)] = {
            "cold_first_req_s": round(r["cold_first_req_s"], 2),
            "warm_ttft_s": round(ttft, 3) if ttft else None,
            "gen_tps_median": round(med, 1),
            "prefill_tps_median": round(premed, 1),
            "long_prefill_tps_median": round(lpmed, 1),
            "gen_tps_runs": [round(g, 1) for g in gens],
        }
    # Версии сторон в отчёт: llmhost (демон + релиз движка) и Ollama —
    # чтобы сравнение всегда было атрибутировано.
    try:
        _, sd = req(CTRL_PORT, "GET", "/v0/status", timeout=30,
                    headers={"Authorization": "Bearer " + TOKEN})
        lh_ver = json.loads(sd).get("version", "?")
        _, ed = req(CTRL_PORT, "GET", "/v0/engine", timeout=30,
                    headers={"Authorization": "Bearer " + TOKEN})
        lh_eng = json.loads(ed).get("release", "?")
    except Exception:
        lh_ver = lh_eng = "?"
    try:
        v = subprocess.run([ol.exe, "--version"], env=ol.env, capture_output=True,
                           timeout=30, text=True)
        ov = ((v.stdout or v.stderr or "?").strip().splitlines() or ["?"])[-1]
    except Exception:
        ov = "?"
    out["meta"] = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "llmhost_version": lh_ver,
        "llmhost_engine_release": lh_eng,
        "ollama_version": ov,
        "stand": "Windows-стенд (методика — bench_vs_ollama_linux.py)",
    }
    with open(os.path.join(HERE, "bench_vs_ollama_last.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nJSON: scripts/bench_vs_ollama_last.json")


if __name__ == "__main__":
    sys.exit(main())
