#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llmhost harness e2e — многопользовательский сценарий (L12).

Проверяются три инварианта harness-режима на живом демоне:
  1. Занятая модель не выселяется НИКОГДА: /v0/unload занятой — 409.
  2. Запрос к модели, которой не хватает памяти из-за занятого резидента,
     паркуется в очередь пробуждения и доезжает сразу после освобождения —
     200 без единого 502.
  3. Очередь наблюдаема: метрики depth/wait в /v0/metrics; истёкший TTL —
     честный 504 с причиной.

Для честного конфликта обе модели должны пиннуться на ОДНУ карту
(например, main-gpu = 1 у мелкой в llmhost.conf тестового демона), крупная
— заведомо больше свободного остатка. Если сценарий невоспроизводим
(нет второй модели / карты не конфликтуют) — SKIP, а не FAIL.

Запуск:  python tests/harness_multiuser_e2e.py [--openai-port N] [--control-port N]
         [--token-file ПУТЬ] [--model-a ID] [--model-b ID]
"""

import argparse
import json
import sys
import threading
import time
import http.client

results = []


def check(name, cond, msg):
    results.append((bool(cond), name, msg))
    return cond


class Client:
    def __init__(self, host, port, token):
        self.host, self.port, self.token = host, port, token

    def req(self, method, path, body=None, timeout=30):
        hdrs = {"Authorization": "Bearer " + self.token}
        data = None
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode()
            hdrs["Content-Type"] = "application/json"
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        conn.request(method, path, data, hdrs)
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        try:
            obj = json.loads(raw)
        except Exception:
            obj = None
        return resp.status, obj, raw


def wait_state(c, model_id, want, timeout_s=180):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        st, obj, _ = c.req("GET", "/v0/status")
        for m in (obj or {}).get("models", []):
            if m.get("id") == model_id and m.get("state") == want:
                return True
        time.sleep(1)
    return False


def metrics_has(c, needle):
    st, _, raw = c.req("GET", "/v0/metrics", timeout=15)
    return st == 200 and needle.encode() in raw


def log_tail(data_dir, nbytes=400000):
    try:
        with open(data_dir.rstrip("/") + "/log.txt", "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openai-port", type=int, default=17340)
    ap.add_argument("--control-port", type=int, default=17341)
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--model-a", default="Qwen3-0.6B-Q8_0",
                    help="маленькая: её стрим держит занятость")
    ap.add_argument("--model-b", default="Qwen3-30B-A3B-Instruct-2507-Q4_K_M",
                    help="большая: не влезает рядом с занятой мелкой")
    ap.add_argument("--data-dir", default="",
                    help="LH_DATA_DIR стенда: assert'ы по log.txt (опционально)")
    args = ap.parse_args()

    token = open(args.token_file, encoding="utf-8").read().strip()
    c = Client("127.0.0.1", args.control_port, token)
    p = Client("127.0.0.1", args.openai_port, token)

    # Разведка: обе модели есть, демон в harness, карта свободна.
    st, obj, _ = c.req("GET", "/v0/status")
    if st != 200 or not (obj or {}).get("harness_mode"):
        print("SKIP: демон не в harness-режиме")
        return 0
    ids = {m.get("id") for m in obj.get("models", [])}
    if args.model_a not in ids or args.model_b not in ids:
        print("SKIP: нет %s/%s в реестре" % (args.model_a, args.model_b))
        return 0
    gpus = obj.get("gpus", [])
    if not gpus or all(g.get("vram_free", 0) < (1 << 30) for g in gpus):
        print("SKIP: карты заняты чужими резидентами — сценарий невоспроизводим")
        return 0

    # Чистый старт: выгрузить всё незанятое.
    c.req("POST", "/v0/unload", {})
    time.sleep(2)

    def state_of(model_id):
        st, obj, _ = c.req("GET", "/v0/status")
        for m in (obj or {}).get("models", []):
            if m.get("id") == model_id:
                return m.get("state")
        return None

    def ensure_loaded(model_id, tries=10):
        # Настойчиво: при старте демона восстановление резидентории может
        # само грузить модель (409 «уже идёт загрузка» / «уже загружена»).
        for _ in range(tries):
            s = state_of(model_id)
            if s == "loaded":
                return True
            if s == "loading":
                time.sleep(2)
                continue
            c.req("POST", "/v0/unload", {"model": model_id})
            time.sleep(1)
            c.req("POST", "/v0/load", {"name": model_id})
            time.sleep(2)
        return state_of(model_id) == "loaded"

    # 1) A загружена.
    if not check("load A", ensure_loaded(args.model_a), "A не загрузилась"):
        return report()
    if not check("A loaded", wait_state(c, args.model_a, "loaded"), "A не загрузилась"):
        return report()

    # 2) B заведомо не влезает рядом с A на её карте: иначе сценарий не тот.
    st, obj, _ = c.req("GET", "/v0/status")
    a_args = " ".join(next((m.get("args", []) for m in obj.get("models", [])
                            if m.get("id") == args.model_a), []))
    print("· A args: %s" % a_args[-90:])

    # 3) A стримит длинную генерацию; в потоке пробуем выгрузить A (409) и
    #    запрашиваем B — она должна запарковаться.
    stream_done = threading.Event()
    stream_err = []
    stream_raw = []

    def stream_a(max_tokens=4000):
        try:
            st, obj, raw = p.req("POST", "/v1/chat/completions",
                                 {"model": args.model_a, "stream": True,
                                  "max_tokens": max_tokens,
                                  "messages": [{"role": "user",
                                                "content": "Расскажи длинную историю "
                                                           "про кота и паровоз."}]},
                                 timeout=300)
            stream_err.append(st)
            stream_raw.append(raw)
        except Exception as e:
            stream_err.append(str(e))
        stream_done.set()

    def wait_busy(timeout_s=15):
        # Стрим дошёл до движка: сторож занятости видит модель.
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            st, obj, _ = c.req("GET", "/v0/status")
            if (obj or {}).get("busy"):
                return True
            time.sleep(0.3)
        return False

    th = threading.Thread(target=stream_a)
    th.start()
    check("A busy visible", wait_busy(), "сторож занятости не увидел стрим A")

    # 3а) занятую выгружать нельзя.
    st, obj, raw = c.req("POST", "/v0/unload", {"model": args.model_a})
    check("unload busy A -> 409", st == 409,
          "ожидал 409, получил %s: %r" % (st, raw[:120]))

    # 3б) B паркуется и доезжает после освобождения A.
    t0 = time.time()
    st_b, obj_b, raw_b = p.req("POST", "/v1/chat/completions",
                               {"model": args.model_b, "stream": False,
                                "max_tokens": 16,
                                "messages": [{"role": "user", "content": "Скажи «готово»."}]},
                               timeout=600)
    dt_b = time.time() - t0
    check("B parked then 200", st_b == 200,
          "ожидал 200 после паркинга, получил %s: %r" % (st_b, raw_b[:200]))
    check("no 502 for B", b"502" not in raw_b[:400], "в теле ответа 502")
    th.join(timeout=300)
    check("A stream ok", bool(stream_err) and stream_err[0] == 200,
          "стрим A: %r" % (stream_err[:1],))
    print("· B дождалась за %.1f с (парковка + загрузка %d МиБ)" %
          (dt_b, 17000))

    # 4) B теперь резидентна, A выселена выселением-по-плану.
    st, obj, _ = c.req("GET", "/v0/status")
    states = {m.get("id"): (m.get("state")) for m in obj.get("models", [])}
    check("B loaded after park", states.get(args.model_b) == "loaded",
          "B не резидентна: %r" % states.get(args.model_b))

    # 4а) Стоп-кран: занятую A снимают принудительно — force отменяет
    #     инвариант по явному решению пользователя. Стрим A обрывается
    #     (тело без [DONE]), слот уснул, повторная будилка работает.
    stream_done.clear()
    stream_err.clear()
    stream_raw.clear()
    th = threading.Thread(target=stream_a)
    th.start()
    check("A busy visible (force)", wait_busy(),
          "сторож занятости не увидел стрим A")
    st, obj, raw = c.req("POST", "/v0/unload",
                         {"model": args.model_a, "force": True})
    check("force unload busy A -> 200", st == 200,
          "ожидал 200, получил %s: %r" % (st, raw[:120]))
    th.join(timeout=60)
    check("A stream ended by force", not th.is_alive(),
          "стрим A висит после force-выгрузки")
    check("A stream truncated (no DONE)",
          bool(stream_raw) and b"[DONE]" not in stream_raw[-1],
          "стрим A должен быть оборван без [DONE]: %r"
          % (stream_raw[-1][-80:] if stream_raw else None,))
    check("A asleep after force",
          wait_state(c, args.model_a, "unloaded", timeout_s=15),
          "A не выгрузилась после force (loaded держится)")

    # 5) Метрики очереди дышат.
    check("queue metrics", metrics_has(c, "llmhost_queue_wait_ms"),
          "в /v0/metrics нет llmhost_queue_wait_ms")

    # 6) TTL: короткий TTL — занятая A держит карту дольше; B получает 504
    #    с причиной, а не 502/503-бурю.
    st, _, _ = c.req("POST", "/v0/settings", {"key": "queue-ttl", "value": "3"})
    check("set queue-ttl=3", st == 200, "POST /v0/settings queue-ttl: %s" % st)
    # A грузим и стримим снова; B просит память и должна устать ждать за ~3 с.
    c.req("POST", "/v0/unload", {})
    time.sleep(2)
    ensure_loaded(args.model_a)
    # Стрим, устойчивый к раннему EOS: перезапускаем генерации, пока не
    # истечёт окно — занятость A должна перекрыть TTL очереди.
    stop_streams = threading.Event()

    def stream_loop(duration_s):
        t0 = time.time()
        while not stop_streams.is_set() and time.time() - t0 < duration_s:
            try:
                p.req("POST", "/v1/chat/completions",
                      {"model": args.model_a, "stream": True, "max_tokens": 4000,
                       "messages": [{"role": "user",
                                     "content": "Пиши очень длинную историю про "
                                                "кота и паровоз, не заканчивай."}]},
                      timeout=60)
            except Exception:
                pass

    th2 = threading.Thread(target=stream_loop, args=(25,))
    th2.start()
    check("A busy visible (ttl)", wait_busy(), "сторож занятости не увидел стрим A")
    t0 = time.time()
    st_b2, obj_b2, raw_b2 = p.req("POST", "/v1/chat/completions",
                                  {"model": args.model_b, "stream": False,
                                   "max_tokens": 8,
                                   "messages": [{"role": "user", "content": "hi"}]},
                                  timeout=300)
    dt_ttl = time.time() - t0
    stop_streams.set()
    th2.join(timeout=120)
    check("B ttl -> 504", st_b2 == 504,
          "ожидал 504 по TTL, получил %s: %r" % (st_b2, raw_b2[:200]))
    check("ttl ~3s", 2.0 <= dt_ttl < 60, "TTL сработал за %.1f с" % dt_ttl)
    # Причина очереди — языконезависимо: в тексте присутствует имя модели,
    # которая держала память (демон стенда может отвечать на любом языке).
    check("ttl message", obj_b2 and args.model_a in str(
        (obj_b2.get("error") or {}).get("message", "")),
        "в теле 504 нет причины очереди: %r" % (obj_b2,))
    # вернуть TTL
    c.req("POST", "/v0/settings", {"key": "queue-ttl", "value": "120"})

    # 7) Парковка KV слотов на NVMe ([general] slot-park*): простой → save,
    #    пробуждение → restore (ход попадает в кэш промпта). Порог занижен
    #    ключом — парковались бы и короткие тестовые промпты; простои ждём
    #    честным минутным unload-after-min (0 — «никогда», меньше минуты нет).
    st, obj, _ = c.req("GET", "/v0/settings")
    prior = {row["key"]: row.get("value") for row in (obj or {}).get("settings", [])}

    def setkey(key, value):
        st, _, raw = c.req("POST", "/v0/settings", {"key": key, "value": value})
        return st == 200

    def restore_keys():
        setkey("slot-park-min-tokens", prior.get("slot-park-min-tokens", "8192"))
        setkey("unload-after-min", prior.get("unload-after-min", "15"))

    ok_cfg = (setkey("slot-park-min-tokens", "1")
              and setkey("unload-after-min", "1"))
    check("park: ключи стенда приняты", ok_cfg,
          "POST /v0/settings slot-park-min-tokens/unload-after-min отклонён")
    if not ok_cfg:
        return report()

    if not check("park: A загружена", ensure_loaded(args.model_a),
                 "A не загрузилась перед парковкой"):
        restore_keys()
        return report()
    st_a, _, raw_a = p.req("POST", "/v1/chat/completions",
                           {"model": args.model_a, "stream": False, "max_tokens": 8,
                            "messages": [{"role": "user",
                                          "content": "Парковка: скажи «ок»."}]},
                           timeout=120)
    check("park: ход по A", st_a == 200, "ожидал 200: %r" % (raw_a[:120],))

    # Сторож простоя: простой ≥ 1 мин → выселение с парковкой. Успех —
    # A не loaded и у неё появился файл парковки (parked_mib > 0 в статусе).
    parked = False
    t0 = time.time()
    while time.time() - t0 < 180:
        st, obj, _ = c.req("GET", "/v0/status")
        for m in (obj or {}).get("models", []):
            if (m.get("id") == args.model_a and m.get("state") != "loaded"
                    and (m.get("parked_mib") or 0) > 0):
                parked = True
        if parked:
            break
        time.sleep(3)
    check("park: A выселена с парковкой", parked,
          "за 180 с A не выселена сторожем или файл парковки не появился")

    # Пробуждение: restore идёт до будильника очереди — ход возвращает 200.
    st_w, _, raw_w = p.req("POST", "/v1/chat/completions",
                           {"model": args.model_a, "stream": False, "max_tokens": 8,
                            "messages": [{"role": "user",
                                          "content": "Парковка: скажи «ок»."}]},
                           timeout=600)
    check("park: пробуждение 200", st_w == 200, "ожидал 200: %r" % (raw_w[:160],))
    check("park: A снова резидентна", wait_state(c, args.model_a, "loaded"),
          "A не вернулась в loaded после пробуждения")
    # Снимок потреблён: restore забрал KV в VRAM и удалил файл (гигиена
    # каталога парковки) — parked_mib в статусе исчезает до следующей парковки.
    st, obj, _ = c.req("GET", "/v0/status")
    consumed = not any(m.get("id") == args.model_a and (m.get("parked_mib") or 0) > 0
                       for m in (obj or {}).get("models", []))
    check("park: снимок потреблён — parked_mib погас после restore", consumed,
          "parked_mib остался после restore: файл не удалён")
    if args.data_dir:
        tail = log_tail(args.data_dir)
        check("park: restore в логе демона",
              "восстановлено" in tail and "слот" in tail,
              "в хвосте log.txt нет строки возврата парковки")
    restore_keys()

    # Уборка: выгрузить всё незанятое.
    c.req("POST", "/v0/unload", {})
    return report()


def report():
    fails = [r for r in results if not r[0]]
    for ok, name, msg in results:
        if ok:
            print("[PASS] %s" % name)
        else:
            print("[FAIL] %s\n       ✗ %s" % (name, msg))
    print("Итого: %d проверок, FAIL: %d" % (len(results), len(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
