#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llmhost API e2e — мучаем контракт.

Проверяются два порта запущенного llmhost:
  :7330 — OpenAI-прокси (всё пересылается в llama-server-детей);
  :7331 — control API.

Контракт для клиентов:
  - Bearer-токен из data/token.txt;
  - OpenAI-совместимые /v1/* на 7330;
  - ошибки — JSON {"error": {...}} с message, у стабильных кодов;
  - состояние (какие модели загружены) может меняться, но API не должно
    ни падать (5xx), ни вешаться, ни портить конфиг.

Запуск:  python tests/api_e2e.py [--quick] [-v]
         [--token-file ПУТЬ] [--openai-port N] [--control-port N]
         (по умолчанию токен/конфиг берутся из $LH_DATA_DIR, иначе
         $XDG_DATA_HOME/llmhost, иначе ~/.local/share/llmhost; порты — 7330/7331)
Выход:   код 0 — все проверки прошли; 1 — есть FAIL.

Тесты, меняющие состояние (загрузка nomic-эмбеддера), в конце откатывают:
после прогона должна остаться загруженной та модель, что была при старте.
"""

import argparse
import json
import shutil
import socket
import ssl
import sys
import threading
import time
import http.client
import os
from urllib.parse import quote

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def default_data_dir():
    """Тот же порядок, что у демона (util.cpp app_dir): LH_DATA_DIR →
    $XDG_DATA_HOME/llmhost → ~/.local/share/llmhost."""
    env = os.environ.get("LH_DATA_DIR")
    if env:
        return env
    xdg = os.environ.get("XDG_DATA_HOME")
    return os.path.join(xdg or os.path.expanduser("~/.local/share"), "llmhost")


DATA_DIR = default_data_dir()
TOKEN_FILE = os.path.join(DATA_DIR, "token.txt")

PROXY_HOST = "127.0.0.1"
CTRL_HOST = "127.0.0.1"
PROXY_PORT = 7330
CTRL_PORT = 7331

T_GEN = 180     # таймаут на генерацию
T_API = 20      # таймаут на обычные вызовы

results = []   # (status, name, detail)
VERBOSE = False
QUICK = False


def note(msg):
    if VERBOSE:
        print("        · " + msg)


class Check:
    """Копит внутри одного теста провалы/заметки."""
    def __init__(self, name):
        self.name = name
        self.fails = []
        self.notes = []

    def fail(self, msg):
        self.fails.append(msg)

    def ok(self, cond, msg):
        if not cond:
            self.fail(msg)
        return cond

    def status_ok(self, got, want, what="status"):
        # want: число или набор чисел
        allowed = want if isinstance(want, (set, tuple, list)) else {want}
        if got not in allowed:
            self.fail("%s: ожидал %s, получил %s" % (what, sorted(allowed), got))
            return False
        return True


# ---------------------------------------------------------------- transport --

def raw_request(method, path, body=None, headers=None, port=None,
                token=True, timeout=T_API, host=None):
    """Один HTTP-запрос. Возвращает (status, headers-dict, body-bytes)."""
    if port is None:
        port = PROXY_PORT
    if host is None:
        host = PROXY_HOST if port == PROXY_PORT else CTRL_HOST
    hdrs = {"Accept": "*/*"}
    if token:
        hdrs["Authorization"] = "Bearer " + TOKEN
    if headers:
        hdrs.update(headers)
    if body is not None and isinstance(body, (dict, list)):
        body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        hdrs.setdefault("Content-Type", "application/json")
    elif isinstance(body, str):
        body = body.encode("utf-8")
    # путь тоже должен быть ASCII (percent-encoding — на вызывающем)
    if not path.isascii():
        path = quote(path)
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=hdrs)
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, data
    finally:
        conn.close()


def jreq(method, path, body=None, **kw):
    """Запрос с ожиданием JSON-ответа. Возвращает (status, headers, obj|None, raw)."""
    status, hdrs, data = raw_request(method, path, body, **kw)
    obj = None
    try:
        obj = json.loads(data.decode("utf-8", "replace"))
    except Exception:
        pass
    return status, hdrs, obj, data


def chat_body(**over):
    b = {
        "model": QWEN_ID,
        "messages": [{"role": "user", "content": "Ответь одним словом: привет"}],
        "max_tokens": 16,
    }
    b.update(over)
    return b


def stream_request(body, timeout=T_GEN, read_limit=None, headers=None):
    """Открыть SSE-стрим и вычитать всё (или read_limit байт).
    Возвращает (status, headers, chunks-список bytes, complete-bool)."""
    hdrs = {"Authorization": "Bearer " + TOKEN,
            "Content-Type": "application/json",
            "Accept": "text/event-stream"}
    if headers:
        hdrs.update(headers)
    conn = http.client.HTTPConnection(PROXY_HOST, PROXY_PORT, timeout=timeout)
    chunks, complete = [], False
    try:
        conn.request("POST", "/v1/chat/completions",
                     body=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                     headers=hdrs)
        resp = conn.getresponse()
        status = resp.status
        hds = {k.lower(): v for k, v in resp.getheaders()}
        got = 0
        while True:
            piece = resp.read1(65536)
            if not piece:
                complete = True
                break
            chunks.append(piece)
            got += len(piece)
            if read_limit is not None and got >= read_limit:
                break
        return status, hds, chunks, complete
    finally:
        # close() рвёт соединение, если не дочитали
        conn.close()


def parse_sse(chunks):
    """Собрать SSE-события из кусков: список 'data'-полезных нагрузок (str)."""
    raw = b"".join(chunks).decode("utf-8", "replace")
    events = []
    for block in raw.split("\n\n"):
        for line in block.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                events.append(line[5:].strip())
    return events


def sse_json(events):
    """JSON-объекты из data-полей (без [DONE])."""
    out = []
    for e in events:
        if e == "[DONE]":
            continue
        try:
            out.append(json.loads(e))
        except Exception:
            pass
    return out


# ------------------------------------------------------------------ fixture --

def load_token():
    global TOKEN
    with open(TOKEN_FILE, "r", encoding="utf-8") as f:
        TOKEN = f.read().strip()


def get_models():
    st, _, obj, _ = jreq("GET", "/v0/models", port=CTRL_PORT)
    if st != 200 or not obj or "models" not in obj:
        return None
    return obj["models"]


def router_ids():
    ms = get_models()
    return [m["id"] for m in ms if "id" in m] if ms else []


def find_qwen30b():
    for i in router_ids():
        if "Qwen3-30B-A3B-Instruct-2507-Q4_K_M" in i:
            return i
    return None


def find_embedder():
    ms = get_models() or []
    for m in ms:
        if m.get("embedding") and m.get("id"):
            return m["id"]
    return None


def wait_model_state(model_id, want, timeout_s=120):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        ms = get_models()
        if ms:
            for m in ms:
                if m.get("id") == model_id and m.get("state") == want:
                    return True
        time.sleep(1.0)
    return False


# ==================================================================== tests ==

def test_auth_proxy(c):
    # без токена
    st, h, obj, raw = jreq("POST", "/v1/chat/completions", chat_body(), token=False)
    c.status_ok(st, 401)
    c.ok(obj and "error" in obj, "тело 401 — не JSON с error: %r" % raw[:120])
    if obj and "error" in obj:
        c.ok(isinstance(obj["error"].get("message"), str) and obj["error"]["message"],
             "401 без error.message")
    # неверный токен
    st, _, obj, raw = jreq("POST", "/v1/chat/completions", chat_body(),
                           headers={"Authorization": "Bearer " + "x" * 64})
    c.status_ok(st, 401)
    c.ok(obj and "error" in obj, "401 с неверным токеном — не JSON error")
    # схема не Bearer
    st, _, obj, _ = jreq("GET", "/v1/models",
                         headers={"Authorization": "Basic " + TOKEN})
    c.status_ok(st, 401, "Basic вместо Bearer должен дать 401")


def test_auth_control(c):
    st, _, obj, raw = jreq("GET", "/v0/status", port=CTRL_PORT, token=False)
    c.status_ok(st, 401)
    c.ok(obj and "error" in obj, "control 401 без токена — не JSON error: %r" % raw[:120])
    st, _, _, _ = jreq("GET", "/v0/status", port=CTRL_PORT,
                       headers={"Authorization": "Bearer " + "неверно".encode("utf-8").hex()})
    c.status_ok(st, 401)


def test_ctrl_status(c):
    st, _, obj, raw = jreq("GET", "/v0/status", port=CTRL_PORT)
    c.status_ok(st, 200)
    c.ok(obj is not None, "статус — не JSON: %r" % raw[:120])
    if obj:
        for key in ("version", "state", "harness_mode", "gpu", "models"):
            c.ok(key in obj, "в статусе нет %r" % key)
        c.ok(isinstance(obj.get("models"), list), "models — не список")


def test_ctrl_hardware(c):
    st, _, obj, _ = jreq("GET", "/v0/hardware", port=CTRL_PORT)
    c.status_ok(st, 200)
    c.ok(obj and "gpu" in obj, "hardware без gpu")
    if obj and "gpu" in obj:
        c.ok("name" in obj["gpu"] and "vram_total" in obj["gpu"],
             "в gpu нет name/vram_total")


def test_ctrl_models(c):
    st, _, obj, _ = jreq("GET", "/v0/models", port=CTRL_PORT)
    c.status_ok(st, 200)
    ms = (obj or {}).get("models")
    c.ok(isinstance(ms, list) and ms, "список моделей пуст/не список")
    for m in (ms or []):
        for key in ("path", "name", "state"):
            c.ok(key in m, "у модели нет %r: %r" % (key, sorted(m.keys())))
    ids = [m.get("id") for m in ms or []]
    c.ok(QWEN_ID in ids, "в списке нет загруженного id %r (есть %r)" % (QWEN_ID, ids))


def test_ctrl_credentials(c):
    st, _, obj, _ = jreq("GET", "/v0/credentials", port=CTRL_PORT)
    c.status_ok(st, 200)
    if obj:
        c.ok(obj.get("api_key") == TOKEN, "api_key в /v0/credentials не совпадает с token.txt")
        c.ok(str(obj.get("openai_base_url", "")).endswith("/v1"),
             "openai_base_url не заканчивается на /v1: %r" % obj.get("openai_base_url"))


def test_ctrl_readonly_endpoints(c):
    for path in ("/v0/engine", "/v0/pull", "/v0/bench"):
        st, _, obj, raw = jreq("GET", path, port=CTRL_PORT)
        c.status_ok(st, 200, path)
        c.ok(obj is not None, "%s — не JSON: %r" % (path, raw[:100]))
    # веб-дашборд: статический HTML без auth (секретов в файле нет), JSON-404
    # на /v0/* это не ломает — /v0/* остаются под auth_wrap.
    st, hd, _, raw = jreq("GET", "/", port=CTRL_PORT)
    c.status_ok(st, 200, "GET /")
    c.ok("text/html" in str(hd).lower(), "GET / — не text/html: %r" % str(hd)[:80])
    c.ok(b"<html" in raw.lower() and b"llmhost" in raw, "GET / — не дашборд")


def test_ctrl_errors(c):
    # мусор вместо JSON
    st, _, obj, raw = jreq("POST", "/v0/load", "это не json", port=CTRL_PORT)
    c.status_ok(st, 400)
    c.ok(obj and "error" in obj, "400 на не-JSON — не JSON error: %r" % raw[:100])
    # пустое тело
    st, _, obj, _ = jreq("POST", "/v0/load", {}, port=CTRL_PORT)
    c.status_ok(st, 400)
    # неизвестное имя модели
    st, _, obj, _ = jreq("POST", "/v0/load", {"name": "нет-такой-модели-xyz"}, port=CTRL_PORT)
    c.status_ok(st, 404)
    # unload неизвестной модели
    st, _, obj, _ = jreq("POST", "/v0/unload", {"model": "нет-такой-xyz"}, port=CTRL_PORT)
    c.status_ok(st, 404)
    # unload уже незагруженной — идемпотентно (не 5xx)
    st, _, _, _ = jreq("POST", "/v0/unload", {"model": "Qwen3-0.6B-Q8_0"}, port=CTRL_PORT)
    c.ok(st < 500, "unload незагруженной: %d (5xx недопустим)" % st)
    # движок: неизвестная сборка — 404, без запуска установки
    st, _, obj, _ = jreq("POST", "/v0/engine", {"build": "нет-такой-build"}, port=CTRL_PORT)
    c.status_ok(st, 404)
    # bench с несуществующим путём — 4xx, без запуска
    st, _, obj, _ = jreq("POST", "/v0/bench",
                         {"path": "/nonexistent/e2e/нет-такого-файла.gguf"}, port=CTRL_PORT)
    c.ok(400 <= st < 500, "bench с мусорным путём: %d" % st)
    # pull без repo
    st, _, obj, _ = jreq("POST", "/v0/pull", {}, port=CTRL_PORT)
    c.status_ok(st, 400)
    # pull с кривым repo (без слэша) — не должен начать скачивание
    st, _, obj, _ = jreq("POST", "/v0/pull", {"repo": "мусор-без-слэша"}, port=CTRL_PORT)
    c.ok(400 <= st < 500, "pull с кривым repo: %d" % st)
    # quit — НЕ вызываем ни под каким видом


def test_ctrl_unknown_routes(c):
    st, _, _, raw = jreq("GET", "/v0/net-takogo", port=CTRL_PORT)
    c.status_ok(st, 404)
    c.ok(b"<html" not in raw.lower(), "404 — голый HTML вместо JSON: %r" % raw[:80])
    st, _, _, _ = jreq("DELETE", "/v0/status", port=CTRL_PORT)
    c.status_ok(st, 404)
    # не-JSON POST на существующий маршрут без парсера — не должен ронять
    st, _, _, _ = jreq("POST", "/v0/quit_check" if False else "/v0/unload", "мусор", port=CTRL_PORT)
    c.ok(st < 500, "unload с мусорным телом: %d" % st)


def test_ctrl_oversize(c):
    # лимит control — 1 МиБ: слишком большое тело → 4xx без зависания
    big = "x" * (2 * 1024 * 1024)
    st, _, _, _ = jreq("POST", "/v0/unload", big, port=CTRL_PORT, timeout=30)
    c.ok(400 <= st < 500, "2 МиБ в control: %d (ожидал 4xx)" % st)


def test_ctrl_load_nonexistent(c):
    # путь, которого нет: честная ошибка, а не 202 в никуда
    st, _, obj, _ = jreq("POST", "/v0/load",
                         {"path": "/nonexistent/e2e/нет-такого-файла.gguf"}, port=CTRL_PORT)
    c.ok(400 <= st < 500,
         "load несуществующего файла: %d (202 «в никуда» = молчаливый отказ)" % st)


def test_ctrl_catalog(c):
    st, _, obj, _ = jreq("GET", "/v0/catalog/search?q=qwen&limit=2", port=CTRL_PORT,
                         timeout=30)
    if st >= 500:
        c.notes.append("catalog/search недоступен (сеть HF?): %d" % st)
        return
    c.status_ok(st, 200)
    # limit-мусор не должен ронять
    st2, _, _, _ = jreq("GET", "/v0/catalog/search?q=qwen&limit=abc", port=CTRL_PORT,
                        timeout=30)
    c.ok(st2 < 500, "limit=abc: %d" % st2)
    st3, _, _, _ = jreq("GET", "/v0/catalog/search?q=qwen&limit=-5", port=CTRL_PORT,
                        timeout=30)
    c.ok(st3 < 500, "limit=-5: %d" % st3)
    # без repo — 4xx
    st4, _, _, _ = jreq("GET", "/v0/catalog/model", port=CTRL_PORT, timeout=30)
    c.ok(st4 < 500, "catalog/model без repo: %d" % st4)


def test_ctrl_delete_model(c):
    """DELETE /v0/models: модель уходит с диска вместе со спутниками
    (mmproj, MTP-модуль), повторное удаление — 404. На настоящих моделях не
    проверяем — создаём пустышки в отдельной подпапке корня моделей. Корень =
    общий каталог-префикс всех путей реестра; фикстура обязана быть (а) в
    пределах глубины скана (MAX_DEPTH=4 в modelreg) и (б) вне поддеревьев
    живых моделей — иначе их каталог «усыновляет» MTP-модуль, и он честно
    не считается спутником (semantics companions_of)."""
    st, _, obj, _ = jreq("GET", "/v0/models", port=CTRL_PORT)
    models = (obj or {}).get("models", [])
    if not models:
        c.notes.append("моделей нет — пропуск")
        return
    root = os.path.commonpath([os.path.dirname(m["path"]) for m in models])
    d = os.path.join(root, "zz-e2e-rm-tmp")
    os.makedirs(d, exist_ok=True)
    f = os.path.join(d, "zz-e2e-rm-test.gguf")
    mm = os.path.join(d, "mmproj-F16.gguf")
    mt = os.path.join(d, "MTP", "mtp-x.gguf")
    os.makedirs(os.path.dirname(mt), exist_ok=True)
    # Предусловие (б) — проверяемое, а не данность: живая модель, чей каталог
    # — предок фикстуры (например, модель лежит прямо в корне моделей),
    # держит MTP-модуль «в зоне досягаемости» (mtp_place: поддерево папки
    # модели, тот же охват, что у find_draft). Контракт удаления — «модуль,
    # нужный ещё живой основной модели, не трогаем»: в этом раскладе MTP-
    # спутник обязан ОСТАТЬСЯ, и тест проверяет именно это, а не отказ.
    adopted = False
    for m in models:
        dd = os.path.dirname(m.get("path") or "")
        if not dd or m["path"] == f:
            continue
        try:
            if os.path.commonpath([dd, d]) == dd:
                adopted = True
                break
        except ValueError:
            pass
    for p in (f, mm, mt):
        with open(p, "wb"):
            pass
    try:
        st2, _, obj2, raw = jreq("DELETE", "/v0/models", {"path": f}, port=CTRL_PORT,
                                 timeout=60)
        c.status_ok(st2, 200, "удаление пустышки со спутниками")
        c.ok(obj2 and obj2.get("deleted"), "ответ без deleted: %r" % raw[:120])
        c.ok(not os.path.exists(f), "файл остался на диске")
        c.ok(not os.path.exists(mm), "mmproj-спутник остался на диске")
        if adopted:
            c.ok(os.path.exists(mt),
                 "MTP-спутник удалён, хотя живая модель держит его в зоне досягаемости")
        else:
            c.ok(not os.path.exists(mt), "MTP-спутник остался на диске")
        st3, _, _, _ = jreq("DELETE", "/v0/models", {"path": f}, port=CTRL_PORT)
        c.status_ok(st3, 404, "повторное удаление")
        st4, _, _, _ = jreq("DELETE", "/v0/models", {"path": ""},
                            port=CTRL_PORT)
        c.status_ok(st4, 400, "пустой path")
    finally:
        for p in (f, mm, mt):
            if os.path.exists(p):
                os.remove(p)
        shutil.rmtree(d, ignore_errors=True)
    # Удаление дёргает registry_changed: карта реестра перестраивается сразу,
    # слоты удалённых моделей выселяются. /v1/models синтезируется на прокси
    # из карты (детей не будит) — окно гонки здесь невозможно.


def test_ctrl_aliases(c):
    """Короткие имена установки: список отдаётся, мусор в /v0/pull — 4xx."""
    st, _, obj, raw = jreq("GET", "/v0/aliases", port=CTRL_PORT)
    c.status_ok(st, 200)
    names = [a.get("name") for a in (obj or {}).get("aliases", [])]
    c.ok("qwen3" in names, "в /v0/aliases нет qwen3: %r" % raw[:120])
    for a in (obj or {}).get("aliases", []):
        c.ok(a.get("repo", "").count("/") == 1 and a.get("note"),
             "кривая запись алиаса: %r" % a)
    # несуществующее короткое имя — 400 с подсказкой, не 500 и не старт загрузки
    st2, _, obj2, _ = jreq("POST", "/v0/pull", {"repo": "точно-не-модель"},
                           port=CTRL_PORT)
    c.status_ok(st2, 400, "pull с мусорным именем")
    c.ok(obj2 and "error" in obj2, "400 без конверта")


# ------------------------------------------------------------- proxy: read --

def test_proxy_models(c):
    st, h, obj, raw = jreq("GET", "/v1/models")
    c.status_ok(st, 200)
    c.ok(obj is not None, "тело — не JSON: %r" % raw[:120])
    if obj:
        c.ok(obj.get("object") == "list", "object != list: %r" % obj.get("object"))
        data = obj.get("data")
        c.ok(isinstance(data, list) and data, "data пуст/не список")
        ids = [d.get("id") for d in data or []]
        c.ok(QWEN_ID in ids, "нет %r в /v1/models (есть %r)" % (QWEN_ID, ids))
        for d in data or []:
            c.ok(d.get("object") == "model", "у модели object != model: %r" % d.get("id"))


def test_proxy_health(c):
    # /health у ребёнка публичный; прокси требует токен для всего — но не должен отдавать 5xx
    st, _, _, raw = jreq("GET", "/health", token=False)
    c.ok(st in (200, 401), "/health без токена: %d (%r)" % (st, raw[:80]))
    st, _, obj, _ = jreq("GET", "/health")
    c.ok(st == 200, "/health с токеном: %d" % st)


def test_proxy_options_preflight(c):
    # браузерный preflight: без Authorization, с Origin и Access-Control-Request-*
    st, h, _, raw = jreq("OPTIONS", "/v1/chat/completions", token=False, headers={
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization,content-type",
    })
    c.status_ok(st, (200, 204), "preflight")
    acao = h.get("access-control-allow-origin")
    c.ok(bool(acao), "нет Access-Control-Allow-Origin (браузерные клиенты отвалятся)")
    note("preflight: %d, ACAO=%r, methods=%r" %
         (st, acao, h.get("access-control-allow-methods")))


def test_proxy_cors_on_response(c):
    st, h, _, _ = jreq("GET", "/v1/models", headers={"Origin": "http://localhost:3000"})
    c.status_ok(st, 200)
    c.ok(bool(h.get("access-control-allow-origin")),
         "обычный ответ без CORS-заголовков (браузерному клиенту ответ прочесть нельзя)")


# --------------------------------------------------------- proxy: generate --

def test_chat_happy(c):
    st, h, obj, raw = jreq("POST", "/v1/chat/completions", chat_body(), timeout=T_GEN)
    c.status_ok(st, 200)
    c.ok(obj is not None, "тело — не JSON: %r" % raw[:200])
    if not obj:
        return
    c.ok(obj.get("object") == "chat.completion", "object != chat.completion: %r" % obj.get("object"))
    choices = obj.get("choices")
    c.ok(isinstance(choices, list) and len(choices) == 1, "choices: %r" % choices)
    if choices:
        msg = choices[0].get("message", {})
        c.ok(msg.get("role") == "assistant", "role != assistant: %r" % msg.get("role"))
        c.ok(isinstance(msg.get("content"), str) and msg["content"].strip(),
             "content пуст: %r" % msg.get("content"))
        c.ok(choices[0].get("finish_reason") == "stop",
             "finish_reason: %r" % choices[0].get("finish_reason"))
    u = obj.get("usage") or {}
    c.ok(u.get("prompt_tokens", 0) > 0 and u.get("completion_tokens", 0) > 0,
         "usage пуст: %r" % u)
    c.ok(isinstance(obj.get("id"), str) and obj["id"], "нет id")
    c.ok(isinstance(obj.get("created"), int), "нет created")
    note("ответ: %r" % ((choices or [{}])[0].get("message", {}).get("content", ""))[:60])


def test_chat_no_model(c):
    st, _, obj, raw = jreq("POST", "/v1/chat/completions",
                           chat_body(model=None), timeout=T_GEN)
    c.status_ok(st, 200, "запрос без model (будилка возьмёт загруженную)")
    c.ok(obj is not None, "не JSON: %r" % raw[:120])


def test_chat_any(c):
    st, _, obj, _ = jreq("POST", "/v1/chat/completions",
                         chat_body(model="any"), timeout=T_GEN)
    c.status_ok(st, 200, 'model="any"')


def test_chat_unknown_model(c):
    st, _, obj, raw = jreq("POST", "/v1/chat/completions",
                           chat_body(model="нет-такой-модели-xyz"), timeout=30)
    c.ok(400 <= st < 500, "неизвестная модель: %d (ожидал 4xx)" % st)
    c.ok(obj and "error" in obj and obj["error"].get("message"),
         "тело ошибки без message: %r" % raw[:150])


def test_chat_bad_messages(c):
    cases = [
        ("пустой messages", {"messages": []}),
        # None затирает дефолт chat_body: тело вообще без messages —
        # валидатор обязан дать 400 до будилки модели (не гонка первого форварда).
        ("нет messages", {"messages": None}),
        ("messages строка", {"model": QWEN_ID, "messages": "привет"}),
        ("messages число", {"model": QWEN_ID, "messages": 42}),
        ("сообщение без role", {"messages": [{"content": "привет"}]}),
        ("сообщение без content", {"messages": [{"role": "user"}]}),
        ("messages не объект", {"messages": ["просто строка"]}),
        ("role не строка", {"messages": [{"role": 1, "content": "x"}]}),
        ("content число", {"messages": [{"role": "user", "content": 42}]}),
        ("stream не булев", {"model": QWEN_ID, "messages": [{"role": "user", "content": "x"}], "stream": "да"}),
        ("temperature строка", {"model": QWEN_ID, "messages": [{"role": "user", "content": "x"}], "temperature": "жарко"}),
        ("max_tokens строка", {"model": QWEN_ID, "messages": [{"role": "user", "content": "x"}], "max_tokens": "много"}),
        ("model не строка", {"model": 42, "messages": [{"role": "user", "content": "x"}]}),
        ("тело не JSON", None),  # особый случай: сырой мусор
    ]
    for name, over in cases:
        if over is None:
            st, _, obj, raw = jreq("POST", "/v1/chat/completions", "{не-json", timeout=60)
        else:
            st, _, obj, raw = jreq("POST", "/v1/chat/completions", chat_body(**over),
                                   timeout=60)
        if st == 502:
            # Гонка первого форварда на свежеподнятом демоне: автозагрузка
            # last_model меняет автозагрузку на старте. Один
            # ретрай — инфраструктурная погрешность, не предмет проверки.
            st, _, obj, raw = jreq("POST", "/v1/chat/completions",
                                   "{не-json" if over is None else chat_body(**over),
                                   timeout=60)
        c.ok(400 <= st < 500 or (st == 200),
             "%s: %d (5xx недопустим)" % (name, st))
        note("%s → %d %r" % (name, st, raw[:80]))


def test_completion_embedding_validation(c):
    # Обязательные поля legacy completions и embeddings — проверка до будилки
    # модели: 400 с конвертом ошибки, без 5xx и без зависаний.
    cases = [
        ("completions без prompt", {"model": QWEN_ID}),
        ("completions prompt дробный", {"model": QWEN_ID, "prompt": 1.5}),
        ("embeddings без input", {}),
        ("embeddings input число", {"input": 42}),
    ]
    for name, body in cases:
        st, _, obj, raw = jreq("POST", "/v1/completions" if "prompt" in body
                               else "/v1/embeddings", body, timeout=60)
        c.ok(400 <= st < 500, "%s: %d (ожидался 400)" % (name, st))
        c.ok(isinstance(obj, dict) and isinstance(obj.get("error"), dict),
             "%s: конверт ошибки" % name)
        note("%s → %d" % (name, st))


def test_chat_content_parts(c):
    # OpenAI-формат мультимодального content — текстовые части
    body = chat_body(messages=[{"role": "user", "content": [
        {"type": "text", "text": "Скажи одно слово: тест"}]}])
    st, _, obj, raw = jreq("POST", "/v1/chat/completions", body, timeout=T_GEN)
    c.status_ok(st, 200, "content-части")
    c.ok(obj is not None, "не JSON: %r" % raw[:150])


def test_chat_params(c):
    # тип-мусор в числовых полях — 4xx, не 5xx и не зависание
    for name, over in [
        ("max_tokens строка", {"max_tokens": "много"}),
        ("max_tokens null", {"max_tokens": None}),
        ("max_tokens -5", {"max_tokens": -5}),
        ("temperature 2.0", {"temperature": 2.0}),
        ("temperature -1", {"temperature": -1}),
        ("top_p 5", {"top_p": 5}),
        ("top_k -3", {"top_k": -3}),
        ("stop не массив", {"stop": 7}),
        ("stream строка", {"stream": "да"}),
        ("unknown поле", {"совсем_неизвестное": {"a": [1, 2, 3]}}),
    ]:
        st, _, obj, raw = jreq("POST", "/v1/chat/completions", chat_body(**over),
                               timeout=60)
        c.ok(st < 500, "%s: %d (5xx недопустим)" % (name, st))
        note("%s → %d" % (name, st))


def test_chat_max_tokens_edge(c):
    # max_tokens=1 — ровно один токен, finish_reason=length
    st, _, obj, _ = jreq("POST", "/v1/chat/completions",
                         chat_body(max_tokens=1), timeout=T_GEN)
    c.status_ok(st, 200)
    if obj and obj.get("choices"):
        fr = obj["choices"][0].get("finish_reason")
        c.ok(fr == "length", "finish_reason при max_tokens=1: %r (ожидал length)" % fr)
        c.ok(obj["choices"][0]["message"].get("content") is not None,
             "content отсутствует")
    # max_tokens=0 — либо 4xx, либо пустой ответ, но не 5xx
    st, _, _, _ = jreq("POST", "/v1/chat/completions",
                       chat_body(max_tokens=0), timeout=T_GEN)
    c.ok(st < 500, "max_tokens=0: %d" % st)


def test_chat_stop(c):
    st, _, obj, _ = jreq("POST", "/v1/chat/completions",
                         chat_body(max_tokens=64,
                                   messages=[{"role": "user",
                                              "content": "Напиши по-русски: один, два, три, четыре, пять"}],
                                   stop=["три"]), timeout=T_GEN)
    c.status_ok(st, 200)
    if obj and obj.get("choices"):
        txt = obj["choices"][0]["message"]["content"]
        c.ok("три" not in txt.lower(), "stop-последовательность не сработала: %r" % txt)


def test_chat_unicode(c):
    prompt = "Повтори точно: Привет 🚀 мир — ёлка, ёж, €99"
    st, _, obj, _ = jreq("POST", "/v1/chat/completions",
                         chat_body(messages=[{"role": "user", "content": prompt}],
                                   max_tokens=64), timeout=T_GEN)
    c.status_ok(st, 200)
    if obj and obj.get("choices"):
        txt = obj["choices"][0]["message"]["content"]
        c.ok("ёл" in txt or "ёр" in txt or "🚀" in txt,
             "юникод исказился: %r" % txt[:80])


def test_chat_long_prompt(c):
    # ~30 КиБ промпта: препроцессинг занимает секунды — проверка больших тел
    filler = ("Служебный абзац про устройство мира номер %d. "
              "Здесь ничего важного нет, только объём. ")
    text = "".join(filler % i for i in range(300))
    c.ok(len(text) > 20000, "промпт вдруг короткий")
    st, _, obj, raw = jreq("POST", "/v1/chat/completions",
                           chat_body(messages=[{"role": "user",
                                                "content": text + "\n\nСколько слов «объём» в тексте? Одним словом."}],
                                     max_tokens=32), timeout=T_GEN)
    c.status_ok(st, 200, "длинный промпт")
    c.ok(obj is not None, "не JSON: %r" % raw[:150])


def test_chat_malformed_bodies(c):
    cases = [
        ("обрезанный JSON", '{"model": "'),
        ("пустое тело", ""),
        ("JSON-массив", "[1,2,3]"),
        ("JSON-строка", '"привет"'),
        ("не-JSON текст", "просто текст"),
        ("BOM", "﻿{\"model\": \"%s\"}" % QWEN_ID),
        ("двойной JSON", "{\"a\":1}{\"b\":2}"),
    ]
    for name, body in cases:
        st, _, obj, raw = jreq("POST", "/v1/chat/completions", body, timeout=30)
        c.ok(400 <= st < 500, "%s: %d (ожидал 4xx)" % (name, st))
        if obj:
            c.ok("error" in obj, "%s: 4xx без error: %r" % (name, raw[:80]))


def test_chat_wrong_content_type(c):
    # JSON-тело с text/plain — серверы llama обычно прощают; главное не 5xx
    st, _, _, _ = jreq("POST", "/v1/chat/completions",
                       json.dumps(chat_body()), timeout=T_GEN,
                       headers={"Content-Type": "text/plain"})
    c.ok(st < 500, "text/plain c JSON-телом: %d" % st)
    # Truly invalid UTF-8 in body: normalized 4xx (router b11223 gives 500 → thin layer → 400)
    bad = b'{"model": "' + QWEN_ID.encode() + b'", "messages": [{"role": "user", "content": "\xff\xfe"}]}'
    st, _, obj, raw = jreq("POST", "/v1/chat/completions", bad, timeout=30)
    c.ok(400 <= st < 500, "invalid UTF-8: %d (4xx after normalization)" % st)
    c.ok(obj and "error" in obj, "invalid UTF-8: body without error: %r" % raw[:100])


def test_chat_stream(c):
    body = chat_body(stream=True, max_tokens=48)
    st, h, chunks, complete = stream_request(body)
    c.status_ok(st, 200, "stream-запрос")
    c.ok(complete, "поток оборвался без EOF")
    ctype = h.get("content-type", "")
    c.ok(ctype.startswith("text/event-stream"),
         "content-type стрима: %r" % ctype)
    events = parse_sse(chunks)
    c.ok(len(events) > 2, "мало событий: %d" % len(events))
    c.ok(events[-1] == "[DONE]", "последнее событие не [DONE]: %r" % events[-1:])
    objs = sse_json(events)
    c.ok(len(objs) >= 2, "JSON-событий меньше двух")
    if objs:
        first = objs[0]
        c.ok(first.get("object") == "chat.completion.chunk",
             "object чанка: %r" % first.get("object"))
        deltas = [o["choices"][0].get("delta", {}) for o in objs
                  if o.get("choices")]
        c.ok(deltas and deltas[0].get("role") == "assistant",
             "первый delta без role=assistant: %r" % deltas[:1])
        text = "".join(d.get("content") or "" for d in deltas)
        c.ok(text.strip(), "склеенный текст пуст")
        frs = [o["choices"][0].get("finish_reason") for o in objs if o.get("choices")]
        frs = [f for f in frs if f]
        c.ok(frs and frs[-1] == "stop", "finish_reason стрима: %r" % frs)
        for o in objs:
            for ch in o.get("choices") or []:
                c.ok("message" not in ch, "в чанке message вместо delta")
        note("стрим: %d событий, текст: %r" % (len(objs), text[:50]))


def test_chat_stream_usage(c):
    body = chat_body(stream=True, max_tokens=24,
                     stream_options={"include_usage": True})
    st, _, chunks, complete = stream_request(body)
    c.status_ok(st, 200)
    c.ok(complete, "поток оборвался")
    objs = sse_json(parse_sse(chunks))
    usages = [o.get("usage") for o in objs if o.get("usage")]
    c.ok(usages and usages[-1].get("completion_tokens", 0) >= 1,
         "usage в стриме не пришёл (stream_options.include_usage проигнорирован): %r" % usages)
    if usages:
        last = objs[-1]
        note("usage-чанк choices=%r" % (last.get("choices"),))


def test_chat_stream_disconnect(c):
    # начать стрим и грубо оборвать посреди — сервер должен пережить
    body = chat_body(stream=True, max_tokens=256)
    st, _, chunks, complete = stream_request(body, read_limit=64)
    c.status_ok(st, 200)
    c.ok(not complete, "ожидали обрыв, а поток сам закончился (маловато max_tokens?)")
    # после обрыва — сервер жив и отвечает
    for i in range(3):
        st2, _, obj, raw = jreq("POST", "/v1/chat/completions",
                                chat_body(max_tokens=8), timeout=T_GEN)
        c.status_ok(st2, 200, "запрос после разрыва #%d" % (i + 1))
        c.ok(obj is not None, "не JSON после разрыва: %r" % raw[:100])


def client_gone_count():
    """Суммарный llmhost_errors_total{kind="client_gone"} из /v0/metrics."""
    st, _, data = raw_request("GET", "/v0/metrics", port=CTRL_PORT)
    if st != 200:
        return None
    total = 0
    for line in data.decode("utf-8", "replace").splitlines():
        if line.startswith("llmhost_errors_total") and 'kind="client_gone"' in line:
            try:
                total += int(line.rsplit(" ", 1)[1])
            except (ValueError, IndexError):
                pass
    return total


def wait_busy(want, timeout_s):
    """Ждать busy==true/false в /v0/status. True — дождались."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        st, _, obj, _ = jreq("GET", "/v0/status", port=CTRL_PORT)
        if st == 200 and obj is not None and bool(obj.get("busy")) == want:
            return True
        time.sleep(0.5)
    return False


def test_chat_zombie_cancel(c):
    """Отмена не-стримового запроса посреди генерации. Раньше прокси сидел
    в ожидании upstream-заголовков (для stream=false — до конца генерации)
    и обрыв клиента не видел: модель дожёвывала всё и держала слот занятым.
    Теперь счётчик client_gone растёт сразу, busy снимается."""
    before = client_gone_count()
    c.ok(before is not None, "/v0/metrics недоступен")
    if before is None:
        return
    # Не-стрим с большим max_tokens и ignore_eos (иначе быстрая модель
    # отвечит за 0.4 с — короче интервала опроса busy): тело придёт только
    # в конце; соединение рвём посреди — прокси обязан заметить в петле
    # ожидания заголовков.
    conn = http.client.HTTPConnection(PROXY_HOST, PROXY_PORT, timeout=T_GEN)
    conn.request("POST", "/v1/chat/completions",
                 body=json.dumps(chat_body(max_tokens=3000, ignore_eos=True),
                                 ensure_ascii=False).encode("utf-8"),
                 headers={"Authorization": "Bearer " + TOKEN,
                          "Content-Type": "application/json"})
    busy_seen = wait_busy(True, 90)
    c.ok(busy_seen, "генерация не началась за 90 с (busy не поднялся)")
    conn.close()  # обрыв без чтения: FIN в сокет прокси
    if not busy_seen:
        return
    # счётчик обязан вырасти быстро, а не к естественному концу 3000 токенов
    grown = False
    t0 = time.time()
    while time.time() - t0 < 10:
        now = client_gone_count()
        if now is not None and now > before:
            grown = True
            break
        time.sleep(0.5)
    c.ok(grown, "client_gone не вырос за 10 с после обрыва (зомби жив?)")
    c.ok(wait_busy(False, 30), "busy не снят за 30 с — генерация продолжалась")
    # сервер жив
    st2, _, obj, raw = jreq("POST", "/v1/chat/completions",
                            chat_body(max_tokens=8), timeout=T_GEN)
    c.status_ok(st2, 200, "запрос после зомби-отмены")
    c.ok(obj is not None, "не JSON после зомби-отмены: %r" % raw[:100])


def test_unload_busy_force(c):
    """Force-выгрузка занятой: без force — честный 409 (инвариант цел),
    force на «все» — 400, с force — 200, слот уснул, фоновый запрос рвётся."""
    res = {}

    def gen():
        try:
            res["r"] = jreq("POST", "/v1/chat/completions",
                            chat_body(max_tokens=3000, ignore_eos=True),
                            timeout=T_GEN)
        except Exception as e:
            res["r"] = (0, None, None, str(e).encode("utf-8"))

    th = threading.Thread(target=gen, daemon=True)
    th.start()
    if not c.ok(wait_busy(True, 90), "генерация не началась за 90 с"):
        th.join(T_GEN)
        r = res.get("r")
        c.fail("фоновый запрос: %r %r" %
               ((r or [None])[0], (r[3][:150] if r and len(r) > 3 else None)))
        return
    st, _, _, raw = jreq("POST", "/v0/unload", {"model": QWEN_ID}, port=CTRL_PORT)
    c.ok(st == 409, "unload занятой без force: %d (ожидал 409): %r"
         % (st, raw[:120]))
    st, _, _, raw = jreq("POST", "/v0/unload", {"force": True}, port=CTRL_PORT)
    c.ok(st == 400, "force на «все»: %d (ожидал 400): %r" % (st, raw[:120]))
    st, _, _, raw = jreq("POST", "/v0/unload",
                         {"model": QWEN_ID, "force": "да"}, port=CTRL_PORT)
    c.ok(st == 400, "force не-булевый: %d (ожидал 400): %r" % (st, raw[:120]))
    st, _, obj, raw = jreq("POST", "/v0/unload", {"model": QWEN_ID, "force": True},
                           port=CTRL_PORT, timeout=30)
    c.status_ok(st, 200, "unload с force")
    # models_state отдаёт уснувший слот как "unloaded" (loaded/loading — иначе).
    c.ok(wait_model_state(QWEN_ID, "unloaded", timeout_s=15),
         "модель осталась loaded после force-выгрузки")
    th.join(timeout=30)
    c.ok(not th.is_alive(), "фоновый запрос висит после force-выгрузки")
    r = res.get("r")
    if r:
        c.ok(r[0] == 0 or r[0] >= 400,
             "фоновый запрос: %d (ожидали ошибку/разрыв)" % r[0])
    # вернуть модель (tail-тесты ждут резидента) и проверить демона
    jreq("POST", "/v0/load", {"name": QWEN_ID}, port=CTRL_PORT, timeout=30)
    c.ok(wait_model_state(QWEN_ID, "loaded", timeout_s=300),
         "модель не вернулась после force-выгрузки")
    st2, _, obj2, _ = jreq("POST", "/v1/chat/completions",
                           chat_body(max_tokens=8), timeout=T_GEN)
    c.status_ok(st2, 200, "чат после force-выгрузки и возврата")


def test_completions_legacy(c):
    body = {"model": QWEN_ID, "prompt": "Скажи одно слово: привет",
            "max_tokens": 16}
    st, _, obj, raw = jreq("POST", "/v1/completions", body, timeout=T_GEN)
    c.status_ok(st, 200, "legacy completions")
    c.ok(obj is not None, "не JSON: %r" % raw[:150])
    if obj and obj.get("choices"):
        c.ok(isinstance(obj["choices"][0].get("text"), str),
             "нет text в choices: %r" % obj["choices"][0].keys())
    # prompt массивом строк
    st, _, _, _ = jreq("POST", "/v1/completions",
                       {"model": QWEN_ID, "prompt": ["раз"], "max_tokens": 4},
                       timeout=60)
    c.ok(st < 500, "prompt массивом: %d" % st)


def test_embeddings(c):
    if not EMBEDDER_ID:
        c.notes.append("эмбеддера нет в реестре — пропуск")
        return
    body = {"model": EMBEDDER_ID, "input": "привет, мир"}
    st, _, obj, raw = jreq("POST", "/v1/embeddings", body, timeout=240)
    c.status_ok(st, 200, "embeddings (включая авто-загрузку эмбеддера)")
    c.ok(obj is not None, "не JSON: %r" % raw[:150])
    data = (obj or {}).get("data")
    c.ok(isinstance(data, list) and data, "data пуст: %r" % raw[:150])
    if data:
        emb = data[0].get("embedding")
        c.ok(isinstance(emb, list) and len(emb) > 10 and
             all(isinstance(x, float) for x in emb[:5]),
             "embedding не числовой массив: %r" % (emb[:3] if isinstance(emb, list) else emb))
        note("embedding dim=%d" % (len(emb) if isinstance(emb, list) else -1))
    # input массивом
    st2, _, obj2, _ = jreq("POST", "/v1/embeddings",
                           {"model": EMBEDDER_ID, "input": ["раз", "два"]},
                           timeout=60)
    c.ok(st2 < 500, "embeddings массивом: %d" % st2)
    # не-эмбеддер модель на embeddings — честная >=400 с объяснением
    st3, _, obj3, raw3 = jreq("POST", "/v1/embeddings",
                              {"model": QWEN_ID, "input": "привет"}, timeout=60)
    c.ok(st3 >= 400, "эмбеддинг от чат-модели: %d (ожидал >=400)" % st3)
    c.ok(obj3 and "error" in obj3 and obj3["error"].get("message"),
         "эмбеддинг от чат-модели: тело без error.message: %r" % raw3[:120])


def test_contract_error_envelope(c):
    """Контракт тонкого слоя: ошибки — {"error":{message,type,code}}, коды честные."""
    # parse-ошибки ребёнка (b11223 отдаёт 500 server_error) → нормализованы в 400
    st, _, obj, raw = jreq("POST", "/v1/chat/completions", '{"model": "', timeout=30)
    c.ok(400 <= st < 500, "обрезанный JSON: %d (ожидал 400 после нормализации)" % st)
    c.ok(obj and obj["error"].get("code") == st,
         "error.code != HTTP-статусу: %r" % raw[:150])
    c.ok(obj and obj["error"].get("type") == "invalid_request_error",
         "error.type: %r (ожидал invalid_request_error)" % (obj or {}).get("error", {}).get("type"))
    c.ok(obj and isinstance(obj["error"].get("message"), str) and obj["error"]["message"],
         "error.message пуст")
    # 401 — authentication_error и code
    st, _, obj, raw = jreq("POST", "/v1/chat/completions", chat_body(), token=False)
    c.status_ok(st, 401)
    c.ok(obj and obj["error"].get("type") == "authentication_error" and
         obj["error"].get("code") == 401, "401-конверт: %r" % raw[:150])
    # control — тот же конверт
    st, _, obj, raw = jreq("POST", "/v0/load", "мусор", port=CTRL_PORT)
    c.status_ok(st, 400)
    c.ok(obj and obj["error"].get("code") == 400 and obj["error"].get("type") ==
         "invalid_request_error" and obj["error"].get("message"),
         "control-конверт: %r" % raw[:150])
    # тело ошибки — валидный JSON даже с кавычками внутри message
    st, _, obj2, _ = jreq("POST", "/v0/load", {"path": '/tmp/e2e/кавычки"внутри/x.gguf'},
                          port=CTRL_PORT)
    c.ok(obj2 is not None, "кавычки в message ломают JSON")


def test_contract_wrong_typed_fields(c):
    """Чужие типы в строковых полях — 400, а не 500 с nlohmann-исключением."""
    cases = [
        ("load path число", "/v0/load", {"path": 123}),
        ("load name число", "/v0/load", {"name": 123}),
        ("unload model число", "/v0/unload", {"model": 42}),
        ("engine build число", "/v0/engine", {"build": 7}),
        ("bench path число", "/v0/bench", {"path": 9}),
        ("pull repo число", "/v0/pull", {"repo": 5}),
    ]
    for name, path, body in cases:
        st, _, obj, raw = jreq("POST", path, body, port=CTRL_PORT, timeout=30)
        c.ok(400 <= st < 500, "%s: %d (ожидал 400, не 500): %r" % (name, st, raw[:100]))
        c.ok(obj and "error" in obj, "%s: тело без error" % name)


def test_contract_load_nonexistent(c):
    st, _, obj, raw = jreq("POST", "/v0/load",
                           {"path": "/nonexistent/e2e/net-takogo-faila.gguf"}, port=CTRL_PORT)
    c.status_ok(st, 404, "load несуществующего файла")
    c.ok(obj and obj["error"].get("message"), "тело 404 без message: %r" % raw[:120])
    # и состояние не изменилось — никакой модели не добавилось
    time.sleep(1.0)
    ids = router_ids()
    c.ok(all("faila" not in (i or "") for i in ids), "несуществующая модель попала в реестр")


def test_contract_preflight(c):
    """Браузерный preflight: 204, CORS-заголовки, без токена."""
    st, h, _, _ = jreq("OPTIONS", "/v1/chat/completions", token=False, headers={
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization,content-type",
    })
    c.status_ok(st, (200, 204), "preflight")
    c.ok(h.get("access-control-allow-origin") == "http://localhost:3000",
         "ACAO не эхо Origin: %r" % h.get("access-control-allow-origin"))
    c.ok("POST" in (h.get("access-control-allow-methods") or ""),
         "ACAM без POST: %r" % h.get("access-control-allow-methods"))
    ah = (h.get("access-control-allow-headers") or "").lower()
    c.ok("authorization" in ah and "content-type" in ah,
         "ACAH без authorization/content-type: %r" % ah)
    # Дубль ACAO браузер отвергает целиком («contains multiple values»):
    # preflight падает, и настоящий запрос к модели не уходит вовсе. Считаем
    # сырые пары заголовков — dict выше дубликаты схлопывает (последний
    # побеждает), и удвоение сквозь него проходило незаметно.
    conn = http.client.HTTPConnection(PROXY_HOST, PROXY_PORT, timeout=T_API)
    try:
        conn.request("OPTIONS", "/v1/chat/completions", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        })
        acao = [v for k, v in conn.getresponse().getheaders()
                if k.lower() == "access-control-allow-origin"]
    finally:
        conn.close()
    c.ok(acao == ["http://localhost:3000"],
         "ACAO должен быть ровно один с эхом Origin: %r" % acao)
    # preflight на control — не обязателен, но не должен быть 5xx
    st2, _, _, _ = jreq("OPTIONS", "/v0/status", port=CTRL_PORT, token=False,
                        headers={"Origin": "http://localhost:3000",
                                 "Access-Control-Request-Method": "GET"})
    c.ok(st2 < 500, "control preflight: %d" % st2)
    # чужой Origin: CORS-заголовков быть не должно (эхо — только доверенным)
    st3, h3, _, _ = jreq("OPTIONS", "/v1/chat/completions", token=False, headers={
        "Origin": "https://evil.example",
        "Access-Control-Request-Method": "POST",
    })
    c.status_ok(st3, (200, 204), "preflight чужому Origin")
    c.ok(not h3.get("access-control-allow-origin"),
         "preflight чужому Origin дал ACAO: %r" % h3.get("access-control-allow-origin"))


def test_contract_cors_echo(c):
    st, h, _, _ = jreq("GET", "/v1/models", headers={"Origin": "http://localhost:3000"})
    c.status_ok(st, 200)
    c.ok(h.get("access-control-allow-origin") == "http://localhost:3000",
         "ответ без ACAO-эха: %r" % h.get("access-control-allow-origin"))
    # без Origin — CORS-заголовков не нужно
    st2, h2, _, _ = jreq("GET", "/v1/models")
    c.ok(not h2.get("access-control-allow-origin"),
         "без Origin внезапно есть ACAO: %r" % h2.get("access-control-allow-origin"))
    # чужой Origin — эха быть не должно: посторонний сайт в браузере
    # не имеет права читать ответы локального API
    st3, h3, _, _ = jreq("GET", "/v1/models", headers={"Origin": "https://evil.example"})
    c.status_ok(st3, 200)
    c.ok(not h3.get("access-control-allow-origin"),
         "чужому Origin дали ACAO: %r" % h3.get("access-control-allow-origin"))
    c.ok(not h3.get("vary"), "чужому Origin дали Vary: %r" % h3.get("vary"))


def test_contract_embeddings_no_model(c):
    """embeddings без model — честная 4xx «модель не выбрана», не 501 от чат-модели."""
    st, _, obj, raw = jreq("POST", "/v1/embeddings", {"input": "привет"}, timeout=30)
    c.ok(400 <= st < 500, "embeddings без model: %d" % st)
    c.ok(obj and "error" in obj and obj["error"].get("message"),
         "тело без message: %r" % raw[:150])
    c.ok("embeddings" not in (obj["error"]["message"].lower() if obj and obj.get("error")
                              else "") or st == 400,
         "сообщение вводит в заблуждение: %r" % (obj or {}).get("error", {}).get("message"))


def test_tokenize_apply_template(c):
    st, _, obj, _ = jreq("POST", "/tokenize",
                         {"model": QWEN_ID, "content": "привет"}, timeout=30)
    c.ok(st < 500, "tokenize: %d" % st)
    note("tokenize → %d %r" % (st, (obj or b"")[:80] if not obj else list(obj)[:5]))
    st, _, obj, _ = jreq("POST", "/apply-template", chat_body(max_tokens=4), timeout=30)
    c.ok(st < 500, "apply-template: %d" % st)


def _tiny_png():
    """64x64 PNG: белый круг на зелёном фоне — без внешних библиотек."""
    import base64, struct, zlib
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + \
               struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    rows = b""
    for y in range(64):
        rows += b"\x00"
        for x in range(64):
            rows += b"\xff\xff\xff" if (x-32)**2 + (y-32)**2 <= 20*20 else b"\x20\x4f\x20"
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def test_vision(c):
    """Картинки: модель с mmproj-соседом обязана видеть (авто-attach llmhost);
    регресс — «image input is not supported» без ммпроj. Vision-модели нет — пропуск."""
    import glob as _glob
    st, _, obj, _ = jreq("GET", "/v0/models", port=CTRL_PORT)
    target, tid = None, None
    # Наименьшая модель с mmproj-соседом: крупная (мультимодальный 27B с
    # проекцией ~0,9 ГиБ) вытесняет резидентов и провоцирует качели выселений
    # на маленьких картах, на которых мелкая влезла бы без единой замены.
    best_sz = None
    for m in (obj or {}).get("models", []):
        d = os.path.dirname(m.get("path", ""))
        if d and _glob.glob(os.path.join(d, "mmproj*.gguf")):
            sz = os.path.getsize(m["path"]) if os.path.exists(m.get("path", "")) else 1 << 62
            if best_sz is None or sz < best_sz:
                best_sz, target, tid = sz, m["path"], m.get("id") or m.get("name")
    if target:
        note("vision-модель: %s (%.1f ГБ)" % (tid, best_sz / 2**30))
    if not target:
        c.notes.append("vision-модели с mmproj нет — пропуск")
        return
    # раз mmproj скачан — модель обязана загрузиться с ним и увидеть картинку
    if tid:
        st, _, _, _ = jreq("POST", "/v0/load", {"name": tid}, port=CTRL_PORT, timeout=30)
        c.ok(st in (200, 202, 409), "load vision-модели: %d" % st)
        loaded_v = False
        for _ in range(120):  # до 4 минут
            time.sleep(2)
            st, _, obj2, _ = jreq("GET", "/v0/models", port=CTRL_PORT)
            m = next((x for x in obj2.get("models", [])
                      if (x.get("id") or x.get("name")) == tid), None)
            if m and m.get("state") == "loaded":
                loaded_v = True
                break
            if m and m.get("state") == "unloaded" and m.get("failed"):
                c.ok(False, "vision-модель не загрузилась (failed)")
                return
            # просто unloaded — загрузка ещё не началась (идёт выселение
            # резидентов под неё): терпеливо ждём
    st, _, obj3, raw = jreq("POST", "/v1/chat/completions", {
        "model": tid or "any", "max_tokens": 400, "messages": [{
            "role": "user", "content": [
                {"type": "text", "text": "Опиши одним предложением: какая фигура и какой фон?"},
                {"type": "image_url", "image_url": {"url": _tiny_png()}},
            ]}]}, timeout=T_GEN)
    c.status_ok(st, 200, "vision-запрос")
    body = raw.decode("utf-8", "replace")
    c.ok("not supported" not in body and "mmproj" not in body,
         "картинки не поддержаны — mmproj не приложен: %r" % body[:120])
    if obj3:
        msg = obj3["choices"][0]["message"]
        text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
        c.ok(len(text) > 5, "пустой ответ на картинку")
        note("vision-ответ: %r" % text[:120])


def test_keepalive(c):
    # три запроса в одном соединении (keep-alive прокси)
    conn = http.client.HTTPConnection(PROXY_HOST, PROXY_PORT, timeout=T_API)
    try:
        for i in range(3):
            hdrs = {"Authorization": "Bearer " + TOKEN,
                    "Content-Type": "application/json"}
            conn.request("GET", "/v1/models", headers=hdrs)
            resp = conn.getresponse()
            data = resp.read()
            c.status_ok(resp.status, 200, "keep-alive #%d" % (i + 1))
            c.ok(b"Qwen" in data or b"data" in data, "пустой ответ keep-alive #%d" % (i + 1))
    finally:
        conn.close()


def test_query_string(c):
    st, _, obj, _ = jreq("GET", "/v1/models?foo=bar&baz=1")
    c.status_ok(st, 200, "GET с query-строкой")


def test_concurrent_chat(c):
    n = 4
    outs = [None] * n

    def worker(i):
        t0 = time.time()
        st, _, obj, raw = jreq("POST", "/v1/chat/completions",
                               chat_body(max_tokens=16,
                                         messages=[{"role": "user",
                                                    "content": "Ответь одним словом: число %d" % i}]),
                               timeout=300)
        outs[i] = (st, obj, time.time() - t0, raw)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    t0 = time.time()
    for t in threads: t.start()
    for t in threads: t.join(320)
    wall = time.time() - t0
    for i, o in enumerate(outs):
        if o is None:
            c.fail("concurrent #%d: поток не завершился (зависание >300с)" % i)
            continue
        st, obj, dt, raw = o
        c.status_ok(st, 200, "concurrent #%d" % (i + 1))
        c.ok(obj is not None, "concurrent #%d: не JSON: %r" % (i + 1, raw[:100]))
    c.ok(wall < 280, "4 параллельных запроса заняли %.0f с — похоже на зависание" % wall)
    note("4 запроса за %.1f с" % wall)


def test_state_after_hammer(c):
    st, _, obj, _ = jreq("GET", "/v0/status", port=CTRL_PORT)
    c.status_ok(st, 200)
    c.ok(obj and obj.get("state") in ("loaded", "loading"),
         "после прогона state=%r" % (obj or {}).get("state"))
    # конфиг не побился: llmhost.conf (единый INI) или legacy config.json
    conf_path = os.path.join(DATA_DIR, "llmhost.conf")
    cfg_path = os.path.join(DATA_DIR, "config.json")
    if os.path.exists(conf_path):
        try:
            with open(conf_path, "r", encoding="utf-8") as f:
                c.ok("[general]" in f.read(), "llmhost.conf содержит [general]")
        except Exception as e:
            c.fail("llmhost.conf побит: %s" % e)
    elif os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                json.load(f)
        except Exception as e:
            c.fail("config.json побит: %s" % e)
    else:
        note("ни llmhost.conf, ни config.json нет (демон с умолчаниями) — ок")


# ----------------------------------------------------------------- restore --

def test_restore(c):
    """Вернуть состояние: выгрузить всё, кроме того, что было загружено при старте."""
    ms = get_models() or []
    for m in ms:
        mid, state = m.get("id"), m.get("state")
        if state in ("loaded", "loading") and mid != START_LOADED_ID:
            st, _, _, _ = jreq("POST", "/v0/unload", {"model": mid}, port=CTRL_PORT,
                               timeout=60)
            c.ok(st < 500, "restore: unload %s → %d" % (mid, st))
    if START_LOADED_ID:
        ok = wait_model_state(START_LOADED_ID, "loaded", timeout_s=60)
        if not ok:
            # могла быть вытеснена — вернуть
            st, _, _, _ = jreq("POST", "/v0/load", {"name": START_LOADED_ID},
                               port=CTRL_PORT, timeout=30)
            c.ok(st in (200, 202), "restore: вернуть %s → %d" % (START_LOADED_ID, st))
            c.ok(wait_model_state(START_LOADED_ID, "loaded", 300),
                 "%s не вернулась в loaded" % START_LOADED_ID)


# ---------------------------------------------------------------- image-слот

def test_image_control(c):
    # Контракт image-слота: статус всегда 200; нагрузка на прокси к
    # незагруженному/ненастроенному движку — внятный 503, не 404 allow-list'а.
    st, _, obj, raw = jreq("GET", "/v0/image", port=CTRL_PORT)
    c.status_ok(st, 200, "GET /v0/image")
    c.ok(isinstance(obj, dict) and "state" in obj and "models" in obj,
         "/v0/image — нет state/models: %r" % raw[:120])
    st, _, obj, raw = jreq("POST", "/v1/images/generations", port=PROXY_PORT,
                           body={"prompt": "a red cube", "size": "64x64"})
    c.ok(st in (200, 503),
         "POST /v1/images/generations без движка — %d: %r" % (st, raw[:120]))
    if st == 503:
        c.ok(isinstance(obj, dict) and "error" in obj,
             "503 без конверта ошибки: %r" % raw[:120])
    # нативное API sd-server тоже пропущено через прокси (не 404 allow-list'а)
    st, _, obj, raw = jreq("GET", "/sdcpp/v1/models/status", port=PROXY_PORT)
    c.ok(st in (200, 503),
         "GET /sdcpp/v1/models/status — %d: %r" % (st, raw[:120]))
    # чужой путь под image-префиксом — по-прежнему 404
    st, _, _, _ = jreq("GET", "/sdcpp/v2/nope", port=PROXY_PORT)
    c.ok(st == 404, "GET /sdcpp/v2/nope — %d (ожидался 404)" % st)
    # каталог художников для селектора клиента: синтез на прокси, без ребёнка
    st, _, obj, raw = jreq("GET", "/v1/images/models", port=PROXY_PORT)
    c.status_ok(st, 200, "GET /v1/images/models")
    ids = [m.get("id") for m in (obj or {}).get("data", [])]
    c.ok("z-image-turbo" in ids, "в /v1/images/models нет z-image-turbo: %r" % ids)
    c.ok((obj or {}).get("object") == "list", "не список: %r" % str(raw)[:80])


def test_image_remove_contracts(c):
    # Удаление художника: негативные контракты (реальное удаление файлов —
    # только на scratch-каталоге, сценарий 10 beta12). Каталог встроенных
    # художников приезжает в статусе — отдельного маршрута для него нет.
    st, _, obj, raw = jreq("GET", "/v0/image", port=CTRL_PORT)
    c.status_ok(st, 200, "GET /v0/image")
    c.ok(isinstance(obj, dict) and isinstance(obj.get("aliases"), list),
         "/v0/image — нет aliases (каталог для pull): %r" % raw[:120])
    c.ok("z-image-turbo" in obj.get("aliases", []),
         "в aliases нет z-image-turbo: %r" % obj.get("aliases"))
    st, _, _, _ = jreq("DELETE", "/v0/image", port=CTRL_PORT, body={"model": 123})
    c.ok(400 <= st < 500, "DELETE model:123 — %d (ожидался 4xx)" % st)
    st, _, _, _ = jreq("DELETE", "/v0/image", port=CTRL_PORT,
                       body={"model": "нет такого; rm -rf /"})
    c.ok(st == 400, "DELETE с charset-мусором — %d (ожидался 400)" % st)
    st, _, obj, raw = jreq("DELETE", "/v0/image", port=CTRL_PORT,
                           body={"model": "img-unknown-probe"})
    c.ok(st == 404, "DELETE неизвестного — %d (ожидался 404)" % st)
    c.ok(isinstance(obj, dict) and "error" in obj,
         "404 без конверта ошибки: %r" % str(raw)[:120])
    # pull kind=image: kind-дискриминация и отказ от «file» без сети.
    st, _, _, _ = jreq("POST", "/v0/pull", port=CTRL_PORT,
                       body={"kind": "bogus", "repo": "org/repo"})
    c.ok(st == 400, "pull kind=bogus — %d (ожидался 400)" % st)
    st, _, _, _ = jreq("POST", "/v0/pull", port=CTRL_PORT,
                       body={"kind": "image", "repo": "unsloth/Z-Image-Turbo-GGUF",
                             "file": "x"})
    c.ok(st == 400, "pull kind=image с file — %d (ожидался 400)" % st)
    st, _, _, _ = jreq("POST", "/v0/pull", port=CTRL_PORT, body={"kind": 5})
    c.ok(400 <= st < 500, "pull kind:5 — %d (ожидался 4xx)" % st)
    st, _, obj, _ = jreq("GET", "/v0/pull", port=CTRL_PORT)
    c.ok(st == 200 and obj.get("kind") in ("llm", "image"),
         "статус pull без kind: %r" % obj)


def test_image_edit_feature_gate(c):
    # edits по модели без ref_images в рецепте — честный 400 на прокси ДО
    # ребёнка (апстрим на таком запросе падает GGML_ASSERT'ом, инцидент 09.11).
    st, _, obj, raw = jreq("GET", "/v0/image", port=CTRL_PORT)
    if st != 200 or not (obj or {}).get("configured"):
        print("        (пропущено: image-каталог не настроен)")
        return
    st, _, obj, raw = jreq("POST", "/v1/images/edits", port=PROXY_PORT,
                           body={"model": "z-image-turbo", "prompt": "x"})
    c.ok(st == 400, "edits без ref_images — %d (ожидался 400 от гейта): %r"
         % (st, str(raw)[:120]))
    c.ok(isinstance(obj, dict) and "error" in obj, "тело без error: %r" % str(raw)[:120])


def test_image_generation(c):
    # Живая генерация — только если у изолированного демона настроен движок
    # (image-engine + каталог моделей): без них секция честно пропускается.
    st, _, obj, raw = jreq("GET", "/v0/image", port=CTRL_PORT)
    ready = (st == 200 and (obj or {}).get("configured") and
             (obj or {}).get("engine", {}).get("found") and (obj or {}).get("models"))
    if not ready:
        print("        (пропущено: image-движок не настроен у изолированного демона)")
        return
    st, _, obj, raw = jreq("POST", "/v0/image/load", port=CTRL_PORT, body={"model": ""})
    c.status_ok(st, 202, "POST /v0/image/load")
    state, err = "?", ""
    unloaded_chat = False
    deadline = time.time() + 300
    while time.time() < deadline:
        st, _, obj, raw = jreq("GET", "/v0/image", port=CTRL_PORT)
        state = (obj or {}).get("state")
        err = (obj or {}).get("error") or ""
        if state in ("loaded", "error"):
            break
        time.sleep(1)
    if state == "error" and "занято llama-моделью" in err:
        # Арбитраж карт сработал по дизайну: llama-резидент держит image-gpu.
        # Чат e2e грузит модели на ту же карту — освобождаем и пробуем ещё раз.
        unloaded_chat = True
        jreq("POST", "/v0/unload", port=CTRL_PORT, body={}, timeout=120)
        jreq("POST", "/v0/image/load", port=CTRL_PORT, body={"model": ""})
        state, err = "?", ""
        deadline = time.time() + 300
        while time.time() < deadline:
            st, _, obj, raw = jreq("GET", "/v0/image", port=CTRL_PORT)
            state = (obj or {}).get("state")
            err = (obj or {}).get("error") or ""
            if state in ("loaded", "error"):
                break
            time.sleep(1)
    if state == "error" and "занято llama-моделью" in err:
        # Своих llama-резидентов мы уже выгрузили — карту держит ВНЕШНИЙ
        # процесс (другой работающий демон или движок на той же карте). Это
        # среда, а не код: честный skip.
        print("        (пропущено: image-gpu занята внешним процессом)")
        return
    c.ok(state == "loaded", "image не загрузился: state=%s err=%r" % (state, err))
    if state != "loaded":
        return
    try:
        st, _, obj, raw = jreq("POST", "/v1/images/generations", port=PROXY_PORT,
                               body={"prompt": "a red cube on white background",
                                     "size": "256x256", "n": 1},
                               timeout=600)
        c.status_ok(st, 200, "POST /v1/images/generations")
        data = (obj or {}).get("data") or []
        c.ok(bool(data) and len(data[0].get("b64_json", "")) > 1000,
             "генерация вернула не b64-картинку: %r" % str(raw)[:120])
        # Селектор модели в запросе: клиент просит другую модель — она
        # загружается (рантайм-смена), неизвестная — честный 400.
        st, _, obj, raw = jreq("POST", "/v1/images/generations", port=PROXY_PORT,
                               body={"model": "flux2-klein-4b",
                                     "prompt": "a blue cube", "size": "256x256"},
                               timeout=600)
        c.status_ok(st, 200, "generations с model=flux2-klein-4b")
        st, _, obj, raw = jreq("POST", "/v1/images/generations", port=PROXY_PORT,
                               body={"model": "нет-такой", "prompt": "x"}, timeout=30)
        c.ok(st == 400, "неизвестная модель — %d (ожидался 400)" % st)
        st, _, obj, raw = jreq("GET", "/v0/image", port=CTRL_PORT)
        c.ok((obj or {}).get("model") == "flux2-klein-4b",
             "после запроса с model= активна flux2-klein-4b, а не %r" % (obj or {}).get("model"))
    finally:
        # VRAM по итогам e2e не держим: остановили image-ребёнка, а чат-модель
        # (state_after_hammer ждёт loaded) — вернули, если сами её снимали.
        jreq("POST", "/v0/image/unload", port=CTRL_PORT, body={})
        if unloaded_chat:
            jreq("POST", "/v0/load", {"name": START_LOADED_ID},
                 port=CTRL_PORT, timeout=30)


# ====================================================================== run ==

def run_test(fn):
    c = Check(fn.__name__)
    t0 = time.time()
    try:
        fn(c)
    except (http.client.HTTPException, socket.timeout, ConnectionError, ssl.SSLError) as e:
        c.fail("транспорт: %s: %s" % (type(e).__name__, e))
    except Exception as e:
        c.fail("неожиданное исключение: %s: %s" % (type(e).__name__, e))
    dt = time.time() - t0
    status = "FAIL" if c.fails else "PASS"
    results.append((status, c.name, c.fails, dt))
    print("[%s] %-32s %5.1fs" % (status, c.name, dt))
    for n in c.notes:
        print("         · " + n)
    for f in c.fails:
        print("         ✗ " + f)


def main():
    global VERBOSE, QUICK, QWEN_ID, EMBEDDER_ID, START_LOADED_ID
    global PROXY_PORT, CTRL_PORT, TOKEN_FILE, DATA_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="без генераций и concurrency")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--token-file", default=TOKEN_FILE,
                    help="путь к token.txt (по умолчанию — data-dir демона)")
    ap.add_argument("--openai-port", type=int, default=PROXY_PORT)
    ap.add_argument("--control-port", type=int, default=CTRL_PORT)
    args = ap.parse_args()
    VERBOSE, QUICK = args.verbose, args.quick
    PROXY_PORT, CTRL_PORT, TOKEN_FILE = args.openai_port, args.control_port, args.token_file
    # token.txt живёт в data-dir демона (рядом с llmhost.conf и
    # api_tokens.json) — изолированный стенд указывается --token-file, и
    # conf надо читать оттуда же, а не из $LH_DATA_DIR тестового
    # окружения (иначе тест заглядывает в чужой data-dir, например в основной демон).
    DATA_DIR = os.path.dirname(os.path.abspath(TOKEN_FILE))

    load_token()
    ms = get_models()
    if not ms:
        print("FATAL: control API не отвечает / моделей нет")
        return 2
    START_LOADED_ID = next((m["id"] for m in ms
                            if m.get("state") == "loaded" and m.get("id")), None)
    QWEN_ID = find_qwen30b() or START_LOADED_ID or (router_ids() or [None])[0]
    if not QWEN_ID:
        # Классический режим: id у моделей пустые — работаем по пути
        # (резолв /v0/load и чат классики принимают путь; поле model
        # классика игнорирует, обслуживает резидентный слот).
        for m in ms:
            if m.get("path"):
                QWEN_ID = m["path"]
                break
    EMBEDDER_ID = find_embedder()
    print("token: %d символов; загружено при старте: %r; основная модель: %r; эмбеддер: %r"
          % (len(TOKEN), START_LOADED_ID, QWEN_ID, EMBEDDER_ID))
    if not QWEN_ID:
        print("FATAL: нет ни одной модели в реестре")
        return 2

    fast = [
        test_auth_proxy, test_auth_control,
        test_ctrl_status, test_ctrl_hardware, test_ctrl_models,
        test_ctrl_credentials, test_ctrl_readonly_endpoints,
        test_ctrl_errors, test_ctrl_unknown_routes, test_ctrl_oversize,
        test_ctrl_load_nonexistent, test_ctrl_catalog, test_ctrl_aliases,
        test_ctrl_delete_model,
        test_image_control,
        test_image_remove_contracts,
        test_image_edit_feature_gate,
        test_contract_error_envelope, test_contract_wrong_typed_fields,
        test_contract_load_nonexistent, test_contract_preflight,
        test_contract_cors_echo, test_contract_embeddings_no_model,
        test_proxy_models, test_proxy_health, test_proxy_options_preflight,
        test_proxy_cors_on_response,
        test_chat_bad_messages, test_completion_embedding_validation,
        test_chat_params, test_chat_malformed_bodies,
        test_chat_wrong_content_type, test_query_string, test_keepalive,
    ]
    gen = [
        test_chat_happy, test_chat_no_model, test_chat_any, test_chat_unknown_model,
        test_chat_max_tokens_edge, test_chat_stop, test_chat_unicode,
        test_chat_long_prompt, test_chat_content_parts,
        test_chat_stream, test_chat_stream_usage,
        test_completions_legacy, test_embeddings, test_tokenize_apply_template,
        test_vision, test_image_generation,
    ]
    stress = [
        test_chat_stream_disconnect, test_chat_zombie_cancel, test_unload_busy_force,
        test_concurrent_chat,
    ]
    tail = [test_state_after_hammer, test_restore]

    order = fast + ([] if QUICK else gen) + ([] if QUICK else stress) + tail
    for fn in order:
        run_test(fn)

    fails = [r for r in results if r[0] == "FAIL"]
    print()
    print("=" * 60)
    print("Итого: %d проверок, FAIL: %d" % (len(results), len(fails)))
    for st, name, fails_, dt in fails:
        print("  FAIL %s" % name)
        for f in fails_:
            print("       ✗ %s" % f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
