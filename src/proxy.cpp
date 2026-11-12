#include "proxy.h"
#include "i18n.h"

#include "config.h"
#include "metrics.h"
#include "supervisor.h"
#include "tokens.h"
#include "util.h"

#include <future>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <strings.h>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

using lh::i18n::tr;

namespace lh::proxy {

namespace {

// Доверенный ли источник для CORS-эха: localhost/127.0.0.1 (локальные
// дев-серверы, file:// со своим Origin «null» не относится) плюс явный
// список cors_origins из config.json. Произвольный сайт в браузере не
// должен ни читать ответы локального API, ни — при выключенной
// авторизации — молча гонять локальную модель.
bool trusted_origin(const std::string& origin, const std::vector<std::string>& extra) {
  for (const std::string& o : extra)
    if (strcasecmp(o.c_str(), origin.c_str()) == 0) return true;
  static const char* schemes[] = {"http://", "https://"};
  for (const char* sch : schemes) {
    size_t n = std::strlen(sch);
    if (origin.size() <= n || strncasecmp(origin.c_str(), sch, n) != 0) continue;
    std::string rest = origin.substr(n);
    size_t cut = rest.find_first_of("/?#");
    if (cut != std::string::npos) rest = rest.substr(0, cut);
    std::string host;
    if (!rest.empty() && rest[0] == '[') {
      // IPv6-литерал: «[::1]:8080» и голый «[::1]» — хост до закрывающей
      // скобки (rfind(':') тут резал по последнему :: в адресе).
      size_t rb = rest.find(']');
      if (rb == std::string::npos) return false;
      host = rest.substr(0, rb + 1);
    } else {
      std::transform(rest.begin(), rest.end(), rest.begin(),
                     [](unsigned char c) { return (char)std::tolower(c); });
      size_t port = rest.rfind(':');
      host = port == std::string::npos ? rest : rest.substr(0, port);
    }
    if (host == "localhost" || host == "127.0.0.1" || host == "[::1]") return true;
  }
  return false;
}

// Очередь кусков от upstream-потока в chunked-провайдер ответа.
// С верхней границей: медленный клиент не раздувает память — upstream
// встаёт в backpressure (а при обрыве клиента просыпается по close()).
struct Chan {
  static constexpr size_t CAP = 8u << 20; // 8 МиБ буфера
  std::mutex m;
  std::condition_variable cv;
  std::deque<std::string> q;
  size_t buffered = 0;
  bool closed = false;

  bool push(std::string s) {
    std::unique_lock l(m);
    cv.wait(l, [&] { return closed || buffered < CAP; });
    if (closed) return false;
    buffered += s.size();
    q.push_back(std::move(s));
    cv.notify_one();
    return true;
  }
  void close() {
    { std::lock_guard l(m); closed = true; }
    cv.notify_all();
  }
  bool pop(std::string& out) {
    std::unique_lock l(m);
    cv.wait(l, [&] { return closed || !q.empty(); });
    if (q.empty()) return false;
    out = std::move(q.front());
    q.pop_front();
    buffered -= out.size();
    cv.notify_all(); // освободили место — ждущий push проснётся
    return true;
  }

  // Исходы таймированного pop: данные / тишина (канал жив, upstream молчит) /
  // закрыт (upstream закончил). Тишина — окно для пробы клиента: до первого
  // куска (префилл) и между кусками обрыв иначе не виден.
  enum class Pop { data, timeout, closed };
  Pop pop_for(std::string& out, std::chrono::milliseconds slice) {
    std::unique_lock l(m);
    if (!cv.wait_for(l, slice, [&] { return closed || !q.empty(); }))
      return Pop::timeout;
    if (q.empty()) return Pop::closed;
    out = std::move(q.front());
    q.pop_front();
    buffered -= out.size();
    cv.notify_all();
    return Pop::data;
  }
};

struct Upstream {
  std::mutex m;
  std::condition_variable cv;
  bool headers_done = false;
  bool failed = false;
  std::string fail_why;
  int status = 200;
  std::string ctype;
};

// Стабильный конверт ошибок для внешних клиентов: {"error":{message,type,code}}.
// Тип — словарь OpenAI (invalid_request_error / authentication_error / api_error),
// код всегда равен HTTP-статусу. Этот формат — часть контракта llmhost.
void json_error(httplib::Response& res, int status, const std::string& msg) {
  const char* type = status == 401 ? "authentication_error"
                   : status < 500 ? "invalid_request_error"
                                  : "api_error";
  nlohmann::json j = {{"error", {{"message", msg}, {"type", type}, {"code", status}}}};
  res.status = status;
  res.set_content(j.dump(), "application/json");
}

// Тело ошибки upstream пересобирается в тот же стабильный конверт. Роутер
// (llama.cpp) относит json-ошибки разбора запроса к server_error(500) — для
// клиента это ошибка его запроса: такие переводим в 400. Всё остальное
// передаём как есть, только выравниваем форму; смыслы (501 not_supported и
// т.п.) не выдумываем.
void json_error_upstream(httplib::Response& res, int status, const std::string& body) {
  std::string msg, utype;
  try {
    auto j = nlohmann::json::parse(body);
    if (j.is_object() && j.contains("error")) {
      auto& e = j["error"];
      if (e.is_object()) {
        if (e.contains("message") && e["message"].is_string()) msg = e["message"];
        if (e.contains("type") && e["type"].is_string()) utype = e["type"];
      } else if (e.is_string()) {
        msg = e.get<std::string>();
      }
    }
  } catch (...) {}
  int code = status;
  if (status >= 500) {
    // Разбор/типы тела запроса — на совести клиента, что бы ни писал роутер.
    if (msg.rfind("[json.exception.parse_error", 0) == 0 ||
        msg.rfind("[json.exception.type_error", 0) == 0 ||
        msg.rfind("[json.exception.out_of_range", 0) == 0 ||
        msg.rfind("Failed to parse", 0) == 0)
      code = 400;
  }
  if (msg.empty()) msg = "upstream ответил " + std::to_string(status);
  // llama-server в parallel-режиме (-np N) делит общий контекст на слоты:
  // на запрос доступно c/N, и честный 400 за длинный промпт клиенты читают
  // как «модель урезали» (заморозка beta 12, наблюдение 3.2: будилка ни при
  // чём — слот всегда имел c/N). Код и текст движка не меняем, объяснение
  // подмешиваем.
  if (msg.find("exceeds the available context size") != std::string::npos)
    msg += " — это контекст одного параллельного слота: llama-server делит "
           "общий контекст (-c) на число слотов (-np); уменьшите промпт, "
           "число параллельных слотов (-np) или увеличьте общий контекст (-c)";
  const char* type;
  if (code < 500 && (utype == "invalid_request_error" || utype == "not_found_error" ||
                     utype == "authentication_error" || utype == "permission_error" ||
                     utype == "rate_limit_error"))
    type = utype.c_str();
  else
    type = code == 401 ? "authentication_error"
         : code == 404 ? "not_found_error"
         : code < 500  ? "invalid_request_error"
                       : "api_error";
  nlohmann::json j = {{"error", {{"message", msg}, {"type", type}, {"code", code}}}};
  res.status = code;
  res.set_content(j.dump(), "application/json");
}

// Валидация обязательных полей OpenAI-запроса — ДО подсказки модели и будилки:
// кривое тело не должно будить резидентов и висеть в ожидании upstream
// (пустой messages: llama-server b11429 принимает соединение и молчит —
// клиент висел до read-таймаута, 600 с; e2e test_chat_bad_messages).
// Проверки по типам, не по значениям: специфику значений держит на себе
// движок. Незнакомые поля не трогаем — совместимость с клиентами шире
// нашего словаря. Пустая строка — ок, иначе текст для 400.
std::string validate_openai_request(const std::string& target, const std::string& body) {
  const bool chat = target.rfind("/v1/chat/completions", 0) == 0;
  // Не «compl»: альтернативный токен C++ (compl ≡ ~), ломает парсинг.
  const bool legacy = !chat && target.rfind("/v1/completions", 0) == 0;
  const bool emb = target.rfind("/v1/embeddings", 0) == 0;
  if (!chat && !legacy && !emb) return {}; // прочие пути — как раньше

  nlohmann::json j;
  try {
    j = nlohmann::json::parse(body);
  } catch (...) {
    return tr("тело запроса — не корректный JSON");
  }
  if (!j.is_object()) return tr("тело запроса — не JSON-объект");

  // Общие поля: проверяем тип, если поле присутствует (обязательность —
  // только у endpoint-специфичного ядра: messages/prompt/input). Явный null
  // приравнивается к отсутствию: клиенты сериализуют «не названо» и так,
  // подстановку модели делает роутер (поймано e2e test_chat_no_model).
  auto named = [&j](const char* k) { return j.contains(k) && !j[k].is_null(); };
  if (named("model") && !j["model"].is_string())
    return "\"model\" должен быть строкой";
  if (named("stream") && !j["stream"].is_boolean())
    return "\"stream\" должен быть булевым";
  if (named("user") && !j["user"].is_string())
    return "\"user\" должен быть строкой";
  for (const char* k : {"temperature", "top_p", "presence_penalty", "frequency_penalty"})
    if (named(k) && !j[k].is_number())
      return std::string("\"") + k + "\" должен быть числом";
  for (const char* k : {"max_tokens", "max_completion_tokens", "n", "seed"})
    if (named(k) && !j[k].is_number_integer())
      return std::string("\"") + k + "\" должен быть целым числом";

  if (chat) {
    if (!j.contains("messages") || !j["messages"].is_array())
      return "\"messages\" обязателен и должен быть массивом";
    if (j["messages"].empty())
      return "\"messages\" не может быть пустым";
    for (size_t i = 0; i < j["messages"].size(); ++i) {
      auto& m = j["messages"][i];
      if (!m.is_object())
        return "messages[" + std::to_string(i) + "] — не объект";
      if (!m.contains("role") || !m["role"].is_string() ||
          m["role"].get<std::string>().empty())
        return "messages[" + std::to_string(i) +
               "]: \"role\" обязателен и должен быть непустой строкой";
      // content может отсутствовать (assistant с tool_calls), быть строкой,
      // массивом частей или null — остальное не осмысленно.
      if (m.contains("content") && !m["content"].is_string() &&
          !m["content"].is_array() && !m["content"].is_null())
        return "messages[" + std::to_string(i) +
               "]: \"content\" должен быть строкой, массивом частей или null";
    }
    return {};
  }
  if (legacy) {
    if (!j.contains("prompt")) return "\"prompt\" обязателен";
    auto& p = j["prompt"];
    if (p.is_string() || p.is_number_integer()) return {};
    if (p.is_array()) {
      for (auto& x : p)
        if (!x.is_string() && !x.is_number_integer())
          return tr("элементы \"prompt\" — строки или целые токен-идентификаторы");
      return {};
    }
    return "\"prompt\" должен быть строкой или массивом строк/токенов";
  }
  // embeddings
  if (!j.contains("input")) return "\"input\" обязателен";
  auto& in = j["input"];
  if (in.is_string()) return {};
  if (in.is_array()) {
    for (auto& x : in)
      if (!x.is_string() && !(x.is_array()))
        return tr("элементы \"input\" — строки или массивы токенов");
    return {};
  }
  return "\"input\" должен быть строкой или массивом";
}

// ---- стадии handle(): auth → harness-интерсепты → резолв+будилка → форвард --

// Контекст одного запроса для метрик и форензики.
struct Rq {
  std::string model;
  std::atomic<uint64_t> bytes{0};
  metrics::UsageSniffer sniff;
};
using Done = std::function<void(const char*, int)>;

// Метрики: одно время старта, один done() на каждый исход handle. Счётчик
// байт — по байтам, ПРИНЯТЫМ от upstream (при обрыве клиента чуть больше
// отданного — для наблюдаемости несущественно).
Done make_done(const char* endpoint, std::shared_ptr<Rq> rq) {
  const auto t0 = std::chrono::steady_clock::now();
  return [rq, endpoint, t0](const char* kind, int status) {
    metrics::Outcome o;
    o.endpoint = endpoint;
    o.kind = kind;
    o.status = status;
    o.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    o.bytes = rq->bytes.load();
    o.usage = rq->sniff.finish();
    metrics::observe(rq->model, o);
  };
}

// Auth на /v1: общий ключ openai_auth, ЛИБО именованные токены из БД
// (lhctl tokens). БД с активными токенами тоже форсирует проверку —
// раз ключи созданы, анонимный доступ закрывается.
// true — пускаем; false — ответ уже записан (401/503) и done вызван.
bool check_auth(const httplib::Request& req, httplib::Response& res, const Done& done) {
  bool auth_required = config::current().openai_auth || lh::tokens::count() > 0;
  if (!auth_required) return true;
  // Значение после "Bearer " — общий ключ или один из именованных токенов.
  std::string auth = req.get_header_value("Authorization");
  std::string presented =
      auth.rfind("Bearer ", 0) == 0 ? auth.substr(7) : std::string();
  if (!presented.empty() && lh::tokens::verify(presented))
    return true; // именованный токен принят (last_used обновлён)
  // Fail-closed: токен не создался — API закрыт, а не «пускаем всех».
  switch (config::check_bearer(auth)) {
    case config::BearerVerdict::Ok:
      return true;
    case config::BearerVerdict::Wrong:
      done("auth", 401);
      json_error(res, 401, tr("нужен Bearer-токен (lhctl tokens — как выдать)"));
      return false;
    case config::BearerVerdict::NoToken:
      done("auth", 503);
      json_error(res, 503,
                 "Bearer-токен не создался — API закрыт (подробности в журнале)");
      return false;
  }
  return true;
}

// harness: единого upstream нет — список моделей и health отвечает сам
// прокси (список — из карты реестра, health — из живого слота или «демон
// здоров»). Вызывается ПОСЛЕ auth: авторизация обязательна для всего /v1.
// true — ответ уже записан; false — ехать в upstream (живой /health
// пробрасывается ребёнку — честный статус его слота).
bool handle_harness_get(const httplib::Request& req, httplib::Response& res,
                        const Done& done) {
  if (req.method != "GET") return false;
  if (req.target.rfind("/v1/models", 0) == 0) {
    nlohmann::json j = {{"object", "list"}, {"data", nlohmann::json::array()}};
    for (auto& m : sup::chat_models())
      j["data"].push_back({{"id", m.id},
                           {"object", "model"},
                           {"created", 0},
                           {"owned_by", "llmhost"},
                           {"state", m.state}});
    done("ok", 200);
    res.set_content(j.dump(), "application/json");
    return true;
  }
  if (req.target.rfind("/health", 0) == 0) {
    int p = 0;
    std::string k;
    if (!sup::upstream_for({}, p, k)) {
      // Ни один ребёнок не жив — прокси здоров, моделей в памяти нет.
      done("ok", 200);
      res.set_content(nlohmann::json{{"status", "ok"}}.dump(), "application/json");
      return true;
    }
    // Есть живой слот — пробросим его health (честный статус ребёнка).
  }
  return false;
}

// Резолв модели: валидация тела, подсказка «model» (роутер), занятость и
// будилка. true — можно ехать в upstream; false — ответ уже записан.
// body может быть переписан (route_request/inject_system). busy_model
// создаётся ДО чтения порта: проба KV останавливает ребёнка только когда
// модель не занята; пометь занятость после upstream_for — проба успевает
// снять процесс, и запрос уходит в мёртвый порт.
bool resolve_and_wake(const httplib::Request& req, httplib::Response& res,
                      const Done& done, std::shared_ptr<Rq> rq, std::string& body,
                      std::string& model_id, int& up_port, std::string& up_key,
                      std::shared_ptr<sup::BusyModel>& busy_model,
                      std::string* note_out = nullptr) {
  body = req.body;
  {
    std::string bad = validate_openai_request(req.target, body);
    if (!bad.empty()) {
      done("bad_request", 400);
      json_error(res, 400, bad);
      return false;
    }
  }
  model_id = sup::route_request(body);
  if (model_id.empty() && !body.empty()) {
    // Роутим по полю model ЛЮБОЙ запрос (у эмбеддингов подсказки нет
    // сознательно — но порт ребёнка выбрать надо): достаём id сами.
    try {
      auto j = nlohmann::json::parse(body);
      if (j.is_object() && j.contains("model") && j["model"].is_string())
        model_id = j["model"].get<std::string>();
    } catch (...) {}
  }
  rq->model = model_id;
  if (model_id.empty() && !req.body.empty()) {
    // Ни клиент, ни подсказка не назвали модель — сырой 400 ребёнка
    // («model name is missing») ни о чём не скажет; объясняем сами.
    try {
      auto j = nlohmann::json::parse(body);
      if (j.is_object() && !j.contains("model")) {
        done("bad_request", 400);
        json_error(res, 400,
                   tr("модель не выбрана: укажите \"model\": <id> в запросе — "
                   "список id: GET /v1/models"));
        return false;
      }
    } catch (...) {}
  }

  // Переопределение «system» из model_overrides.json: подложить, если клиент
  // своего не прислал. model_id пуст — вызов no-op.
  sup::inject_system(model_id, body);

  // Контракт демона: чат всегда кэширует промпт. Клиенты встречаются разные —
  // часть шлёт cache_prompt=false (или не знает о поле), и каждый ход агента
  // перекачивает весь контекст заново (урок 30.10: диалог с 80B — 61k токенов
  // префилла при нуле попаданий). Перекрываем клиентское значение; эмбеддерам
  // и прочим без messages не трогаем — их кэш всё равно stateless.
  if (body.find("\"messages\"") != std::string::npos) {
    try {
      auto j = nlohmann::json::parse(body);
      if (j.is_object() && j.contains("messages") && j["messages"].is_array() &&
          j.value("cache_prompt", true) == false) {
        j["cache_prompt"] = true;
        body = j.dump();
      }
    } catch (...) {}
  }

  // Занятость — до чтения порта (см. шапку функции).
  busy_model = std::make_shared<sup::BusyModel>(model_id);

  if (!sup::upstream_for(model_id, up_port, up_key)) {
    // Не загружена — будим и ждём; иначе честная ошибка. Будилка сама
    // выселит незанятых, а при занятых блокерах — запаркует запрос
    // (TTL очереди); истечение — честный 504 с причиной, не 502.
    sup::WakeResult wr = sup::ensure_model_ex(model_id, 300000);
    if (!wr.ok) {
      const int status = wr.not_found ? 404 : wr.queued_timeout ? 504 : 503;
      done(wr.not_found ? "not_found"
               : wr.queued_timeout ? "queue_timeout"
                                   : "unavailable",
           status);
      json_error(res, status,
                 wr.err.empty() ? tr("модель недоступна") : wr.err);
      return false;
    }
    sup::upstream_for(model_id, up_port, up_key);
  }
  // Нота деградации — по слоту уже загруженной модели (обе ветки выше).
  if (note_out) *note_out = sup::model_note(model_id);
  return true;
}

// Форензика ≥400: точное тело запроса (можно переиграть curl'ом) и хвост
// вывода llama-server (голый 500 httplib = исключение в его обработчике).
// Тело — не длиннее 1 МиБ: это промпты пользователя, тонну вложений на диск
// вываливать не стоит.
void dump_failed_request(const std::string& req_body, const std::string& model,
                         int status) {
  constexpr size_t kDumpLimit = 1u << 20;
  std::string dump = req_body;
  if (dump.size() > kDumpLimit)
    dump = dump.substr(0, kDumpLimit) +
           "\n…[обрезано, всего " + std::to_string(req_body.size()) + " байт]\n";
  std::string dir = lh::app_dir();
  lh::write_small_file(dir + "/failed_request.json", dump);
  lh::restrict_file_permissions(dir + "/failed_request.json");
  std::string tail = sup::server_log_tail_for(model);
  if (!tail.empty()) {
    lh::write_small_file(dir + "/failed_server_tail.txt", tail);
    lh::restrict_file_permissions(dir + "/failed_server_tail.txt");
  }
  logf("proxy: upstream %d — тело запроса (%zu байт) в failed_request.json%s",
       status, req_body.size(),
       tail.empty() ? "" : " и failed_server_tail.txt");
}

// Heap-пакет воркера: клиент может уйти посреди генерации, а заголовочный
// таймаут — вернуть handler раньше ответа upstream. Всё, что нужно воркеру
// после запуска, едет в heap: захват по ссылке из стека handle —
// use-after-return (поймано ASan-ом живьём: retry-петля «upstream переехал»
// читала up_port мёртвого кадра — SEGV из реального прогона).
struct Pass {
  std::string method, target, body, ctype, accept;
  std::string model; // для повторного чтения порта слота (harness)
  int port = 0;
  std::string key;
  bool image = false; // image-пасс: ретрай решает image_upstream (автостарт
                      // ребёнка), а не upstream_for чат-слота
};

// Форвард в llama-server: retry гонки первого форварда (будилка отчиталась
// «жив», но роутер мог переехать на новый порт — пауза, перечитали порт,
// повтор; третий отказ честно уходит клиенту 502-м). Статус и ошибки — в
// Upstream, тело — в канал.
void forward_worker(std::shared_ptr<Pass> pass, std::shared_ptr<Chan> chan,
                    std::shared_ptr<std::atomic<bool>> dead,
                    std::shared_ptr<Upstream> up, std::shared_ptr<Rq> rq) {
  httplib::Client cli("127.0.0.1", pass->port);
  cli.set_connection_timeout(std::chrono::seconds(3));
  cli.set_read_timeout(std::chrono::seconds(600));
  cli.set_write_timeout(std::chrono::seconds(30));

  httplib::Request r;
  r.method = pass->method;
  r.path = pass->target; // raw путь вместе с query
  r.body = pass->body;
  r.headers.emplace("Authorization", "Bearer " + pass->key);
  if (!pass->ctype.empty())
    r.headers.emplace("Content-Type", pass->ctype);
  if (!pass->accept.empty())
    r.headers.emplace("Accept", pass->accept);

  r.response_handler = [up, dead](const httplib::Response& resp) {
    {
      std::lock_guard l(up->m);
      up->status = resp.status;
      up->ctype = resp.get_header_value("Content-Type");
      up->headers_done = true;
    }
    up->cv.notify_all();
    return !dead->load();
  };
  r.content_receiver = [chan, dead, rq](const char* d, size_t n, uint64_t, uint64_t) {
    if (dead->load()) return false;
    rq->bytes += n;
    rq->sniff.feed(d, n);
    return chan->push(std::string(d, n)); // false — очередь закрыта/переполнена
  };

  httplib::Result rr = cli.send(r);
  for (int attempt = 0; !rr && !dead->load() && attempt < 2; ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    if (dead->load()) break;
    int p2 = 0;
    if (pass->image) {
      // Image-ребёнок мог умереть на запросе: image_upstream сам перезапустит
      // его (автостарт) и отдаст новый порт. upstream_for сюда ходить не
      // должен — для пустого model он возвращает ЧАТ-слот, и image-запрос
      // уезжал llama-server'у (инцидент 09.11: «переехал 8200 → 8100»).
      std::string ierr;
      if (!sup::image_upstream(p2, ierr) || p2 <= 0) continue;
    } else {
      std::string k2;
      if (!sup::upstream_for(pass->model, p2, k2) || p2 <= 0) continue;
      pass->key = k2;
      r.headers.erase("Authorization");
      r.headers.emplace("Authorization", "Bearer " + pass->key);
    }
    if (p2 != pass->port)
      logf("proxy: upstream переехал %d → %d — повтор форварда", pass->port, p2);
    pass->port = p2;
    cli.set_connection_timeout(std::chrono::seconds(3));
    cli.set_read_timeout(std::chrono::seconds(600));
    cli.set_write_timeout(std::chrono::seconds(30));
    rr = cli.send(r);
  }
  if (rr && rr->status >= 400)
    dump_failed_request(r.body, rq->model, rr->status);
  if (!rr) {
    std::lock_guard l(up->m);
    up->failed = true;
    up->fail_why = httplib::to_string(rr.error());
    up->headers_done = true;
  }
  up->cv.notify_all();
  chan->close();
}

// Отдача клиенту: дождаться заголовков upstream (для stream=false они
// приходят только в конце генерации — лимит 300 с, зависший upstream раньше
// держал клиента до read-таймаута 600 с), ошибки upstream забрать целиком
// в стабильный конверт, остальное — чанкед-стрим. busy_model в захватах
// релизера — guard занятости живёт ДО КОНЦА отдачи, а не до заголовков:
// handle() возвращается сразу после заголовков upstream, и стековый guard
// освободил бы модель в первые миллисекунды генерации. Отпускается последним
// ref'ом shared_ptr — после этого слот честно «незанят».
//
// alive — проба сокета клиента (Request::connection_alive, llmhost-патч
// httplib): пока upstream молчит (не-стрим до конца генерации, префилл до
// первого токена), обрыв клиента виден только ей — зомби-генерация иначе
// доживает до конца и держит слот занятым. pass — для журнала обрыва.
void stream_reply(httplib::Response& res, std::shared_ptr<Upstream> up,
                  std::shared_ptr<Chan> chan, std::shared_ptr<std::atomic<bool>> dead,
                  std::shared_ptr<sup::BusyModel> busy_model, const Done& done,
                  const std::string& note = {},
                  const std::function<bool()>& alive = {},
                  const std::shared_ptr<Pass>& pass = {}) {
  const auto t_start = std::chrono::steady_clock::now();
  {
    std::unique_lock l(up->m);
    constexpr auto kHeadersWait = std::chrono::seconds(300);
    const auto deadline = t_start + kHeadersWait;
    bool headers = false;
    while (!headers) {
      if (alive && !alive()) {
        // Клиент ушёл, пока upstream молчал: зомби-форвард гасим, слот
        // освободится с последним ref'ом busy_model. Ответ не пишем —
        // сокет мёртв (статус 499 — для формы, запись до клиента не дойдёт).
        *dead = true;
        chan->close();
        if (pass)
          logf("proxy: %s %s — клиент оборвал соединение через %.1f с "
               "(генерация отменяется)",
               pass->method.c_str(), pass->target.c_str(),
               std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             t_start)
                   .count());
        done("client_gone", 499);
        res.status = 499;
        return;
      }
      auto now = std::chrono::steady_clock::now();
      if (now >= deadline) break;
      const auto slice = std::min<std::chrono::steady_clock::duration>(
          std::chrono::milliseconds(500), deadline - now);
      headers = up->cv.wait_for(l, slice, [&] { return up->headers_done; });
    }
    if (!headers) {
      *dead = true;
      chan->close(); // разбудить forward_worker, если он встал в push на CAP:
      // без close приёмник зависнет в cv.wait навсегда (поток в g_bg,
      // join_background на остановке демона — тоже)
      done("timeout", 504);
      json_error(res, 504, "upstream не ответил за 300 с — соединение закрыто");
      return;
    }
    if (up->failed) {
      *dead = true;
      chan->close();
      done("unreachable", 502);
      json_error(res, 502, "llama-server недоступен: " + up->fail_why);
      return;
    }
    res.status = up->status;
  }

  std::string ctype = up->ctype;
  if (ctype.empty() || ctype.rfind("text/event-stream", 0) != 0) {
    // Обычные ответы — JSON. SSE оставляем как есть.
    if (ctype.empty()) ctype = "application/json";
  }

  // Ошибки upstream не стримим: тело маленькое — забираем целиком и отдаём в
  // стабильном конверте (json_error_upstream). SSE-потоки не трогаем.
  if (up->status >= 400 && ctype.rfind("application/json", 0) == 0) {
    std::string err_body, piece;
    while (chan->pop(piece) && err_body.size() < (1u << 20))
      err_body += piece;
    *dead = true;
    chan->close(); // дренаж остановлен на 1 МиБ, канал мог остаться полным:
    // пуш forward_worker без close завис бы в cv.wait навсегда
    done("upstream_error", up->status);
    json_error_upstream(res, up->status, err_body);
    return;
  }

  const int up_status = up->status;
  // Нота деградации — на ответе чата, канал по формату ответа: заголовок всем,
  // SSE — комментарий-строка перед потоком (парсеры пропускают строки без
  // «data: »), не-стримовой JSON — поле llmhost_note в теле. Контент ответа
  // ассистента не трогаем. Заголовок ровно один раз: httplib set_header
  // аппендит повторный, а дубль режется браузером.
  const bool ok_up = up_status < 400;
  const bool sse = ctype.rfind("text/event-stream", 0) == 0;
  const bool note_json = ok_up && !sse && !note.empty() &&
                         ctype.rfind("application/json", 0) == 0;
  if (ok_up && !note.empty()) res.set_header("X-Llmhost-Note", note);
  res.set_chunked_content_provider(
      ctype,
      [chan, dead, ok_up, sse, note_json, note, alive](size_t,
                                                       httplib::DataSink& sink) -> bool {
        if (ok_up && sse && !note.empty()) {
          std::string line = note;
          std::replace(line.begin(), line.end(), '\n', ' '); // комментарий — одна строка
          const std::string c = ": llmhost: " + line + "\n\n";
          if (!sink.is_writable() || !sink.write(c.data(), c.size())) {
            *dead = true;
            chan->close();
            return false;
          }
        }
        // Тишина канала — не «конец»: не-стрим и префилл молчат до конца
        // расчёта. Ждём срезами и в паузах пробуем сокет клиента (alive) —
        // иначе обрыв в тишине увидим только на следующем куске, а для
        // не-стрима — на финальной записи, после всей генерации.
        auto gone = [&]() -> bool {
          if (!alive || alive()) return false;
          *dead = true;
          chan->close();
          return true;
        };
        if (!note_json) {
          std::string chunk;
          for (;;) {
            const Chan::Pop st =
                chan->pop_for(chunk, std::chrono::milliseconds(300));
            if (st == Chan::Pop::closed) break;
            if (st == Chan::Pop::timeout) {
              if (gone()) return false;
              continue;
            }
            if (!sink.is_writable()) { // клиент ушёл — рвём и upstream
              *dead = true;
              chan->close();
              return false;
            }
            if (!sink.write(chunk.data(), chunk.size())) {
              *dead = true;
              chan->close();
              return false;
            }
          }
          sink.done();
          return true;
        }
        // Не-стримовой JSON буферизуем (клиент и так ждёт одно тело): поле
        // llmhost_note вставляется перед закрывающей скобкой объекта. Потолок
        // 16 МиБ — перерос: отдаём как есть, нота не стоит поломанного ответа.
        constexpr size_t kNoteBufCap = 16u << 20;
        std::string body, piece;
        bool overflow = false;
        for (;;) {
          const Chan::Pop st =
              chan->pop_for(piece, std::chrono::milliseconds(300));
          if (st == Chan::Pop::closed) break;
          if (st == Chan::Pop::timeout) {
            if (gone()) return false;
            continue;
          }
          if (!sink.is_writable()) {
            *dead = true;
            chan->close();
            return false;
          }
          if (!overflow) {
            if (body.size() + piece.size() <= kNoteBufCap) {
              body += piece;
              continue;
            }
            overflow = true;
            if (!sink.write(body.data(), body.size())) {
              *dead = true;
              chan->close();
              return false;
            }
          }
          if (!sink.write(piece.data(), piece.size())) {
            *dead = true;
            chan->close();
            return false;
          }
        }
        if (overflow) {
          sink.done();
          return true;
        }
        const size_t start = body.find_first_not_of(" \t\r\n");
        const size_t end = body.find_last_not_of(" \t\r\n");
        if (start != std::string::npos && end != std::string::npos &&
            body[start] == '{' && body[end] == '}') {
          const bool empty_obj = end > start && body[end - 1] == '{';
          body.insert(end, (empty_obj ? std::string() : std::string(",")) +
                               "\"llmhost_note\":" + nlohmann::json(note).dump());
        }
        if (!sink.write(body.data(), body.size())) return false;
        sink.done();
        return true;
      },
      [pass, busy_model, dead, done, up_status, t_start](bool ok) {
        *dead = true;
        // ok — всё отдано, false — клиент ушёл или канал умер.
        if (!ok && pass)
          logf("proxy: %s %s — клиент оборвал соединение через %.1f с",
               pass->method.c_str(), pass->target.c_str(),
               std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             t_start)
                   .count());
        done(ok ? "ok" : "client_gone", up_status);
      });
}

// Image generation through the sd-server child: request/response as-is, without
// chat validation and model substitution. No auth on the child — everything is
// covered by the port's authorization; occupancy is marked with ImageBusy
// (the watchdog will not stop the child mid-generation), autostart —
// image_upstream. There is no SSE: upstream headers arrive only with the
// finished image (b64 of a few MiB), headers-wait of 300 s is greater than
// generation time.
// Целевая модель edits-гейта: селектор запроса → активная → кандидат
// автостарта (тот же выбор, что image_load("")). Пусто — каталог не подскажет.
std::string image_edit_target(const nlohmann::json& st, const std::string& want) {
  if (!want.empty()) return want;
  if (st.value("state", "") == "loaded") {
    std::string active = st.value("model", "");
    if (!active.empty()) return active;
  }
  return st.value("autostart_model", "");
}

// Умеет ли модель каталога принимать референсы (фича «ref_images» рецепта).
bool image_model_takes_refs(const nlohmann::json& st, const std::string& id) {
  for (const auto& m : st["models"]) {
    if (m.value("id", "") != id) continue;
    for (const auto& f : m["features"])
      if (f.get<std::string>() == "ref_images") return true;
    return false;
  }
  return false;
}

void handle_image(const httplib::Request& req, httplib::Response& res, Done done,
                  std::shared_ptr<Rq> rq) {
  // Каталог художников для селектора клиента: синтез на прокси, ребёнок
  // не нужен (открытие дропдауна не должно будить движок). Auth — общая
  // портовая; control-API для этого наружу не требуется.
  if (req.method == "GET" && req.path == "/v1/images/models") {
    const nlohmann::json st = sup::image_status_json();
    const bool live = st.value("state", "") == "loaded";
    const std::string active = live ? st.value("model", "") : std::string();
    nlohmann::json data = nlohmann::json::array();
    for (const auto& m : st["models"]) {
      data.push_back({
          {"id", m.value("id", "")},
          {"name", m.value("name", "")},
          {"bytes", m.value("bytes", (uint64_t)0)},
          {"steps", m.value("steps", -1)},
          {"cfg", m.value("cfg", -1.0)},
          {"sampler", m.value("sampler", "")},
          {"loaded", live && m.value("id", "") == active},
      });
    }
    res.status = 200;
    res.set_content(nlohmann::json{{"object", "list"},
                                   {"active", active.empty()
                                                  ? nlohmann::json(nullptr)
                                                  : nlohmann::json(active)},
                                   {"data", data}}
                        .dump(),
                   "application/json");
    done("image_models", 200);
    return;
  }
  // Поле «model» из запроса — селектор клиента работает: какая модель
  // попросили, такая и будет загружена (рантайм-смена за секунды). Пусто —
  // автостарт по image-model из конфига. Ребёнок «model» игнорирует
  // (value()-парсер), так что тело едет дальше как есть. Парсим ДО гейтов:
  // edits-гейт решает по целевой модели до того, как селектор её разбудит.
  std::string want_model;
  try {
    if (!req.body.empty()) {
      auto j = nlohmann::json::parse(req.body);
      if (j.is_object() && j.contains("model") && j["model"].is_string())
        want_model = j["model"].get<std::string>();
    }
  } catch (...) { // не JSON — multipart, модель ищем среди частей
  }
  if (want_model.empty())
    for (const auto& [k, f] : req.files)
      if (f.name == "model") { want_model = f.content; break; }

  // Гейт тела: не-JSON POST-тело до ребёнка не доезжает. Раньше «не разобрать
  // как JSON» молча считалось multipart и форвардилось как есть; теперь —
  // честный 400 на прокси. (Инцидент 09.11: sd-server падал 134-м на тестовой
  // генерации; ребёнок json-исключения ловит, но мусор наружу больше не едет
  // вовсе — поведение ребёнка на неожиданные тела не часть контракта.)
  if (req.method == "POST" && !req.body.empty() && req.files.empty()) {
    try {
      nlohmann::json::parse(req.body);
    } catch (...) {
      done("bad_body", 400);
      json_error(res, 400, tr("тело запроса не разбирается как JSON"));
      return;
    }
  }

  // /v1/images/edits требует от модели фичи «ref_images». Рецепты это
  // заявляют явно (flux2/flux3 — есть, z-image — нет), но апстрим принимает
  // реф без проверки фичи и падает на GGML_ASSERT в z_image.hpp:808 —
  // SIGABRT всего ребёнка (инцидент 09.11: тестовый edits уронил sd-server
  // дважды; воспроизводится одним запросом). Честный 400 на прокси ДО
  // селектора: модель для заведомо неподъёмного запроса не будится.
  if (req.path == "/v1/images/edits") {
    const nlohmann::json st = sup::image_status_json();
    const std::string target = image_edit_target(st, want_model);
    if (!target.empty() && !image_model_takes_refs(st, target)) {
      done("bad_model", 400);
      json_error(res, 400, tr("модель «") + target +
                                tr("» не поддерживает редактирование по "
                                   "изображению (в рецепте нет ref_images) — "
                                   "текст-в-картинку: /v1/images/generations"));
      return;
    }
  }
  {
    if (!want_model.empty()) {
      // id едет в журнал/ошибки — тот же charset, что принимает каталог.
      for (unsigned char ch : want_model) {
        if (!isalnum(ch) && ch != '-' && ch != '_' && ch != '.') {
          done("bad_model", 400);
          json_error(res, 400, tr("id модели — [A-Za-z0-9._-], а не «") + want_model + "»");
          return;
        }
      }
      bool known = false;
      // Именованная локальная: «for (auto& m : f()["models"])» висит на
      // временном json — временный умирает до первой итерации.
      const nlohmann::json st = sup::image_status_json();
      for (const auto& m : st["models"])
        if (m.value("id", "") == want_model) { known = true; break; }
      if (!known) {
        done("bad_model", 400);
        json_error(res, 400, tr("в каталоге image-моделей нет «") + want_model +
                                  tr("» — список: lhctl image"));
        return;
      }
      std::string lerr;
      if (!sup::image_load(want_model, lerr)) {
        done("image_unavailable", 503);
        json_error(res, 503, tr("image engine unavailable: ") + lerr);
        return;
      }
      sup::image_touch();
    }
  }
  int port = 0;
  std::string err;
  if (!sup::image_upstream(port, err)) {
    done("image_unavailable", 503);
    json_error(res, 503, tr("image engine unavailable: ") + err);
    return;
  }
  sup::image_touch();
  auto busy = std::make_shared<sup::ImageBusy>();
  auto chan = std::make_shared<Chan>();
  auto dead = std::make_shared<std::atomic<bool>>(false);
  auto up = std::make_shared<Upstream>();
  auto pass = std::make_shared<Pass>();
  pass->method = req.method;
  pass->target = req.target;
  pass->body = req.body;
  pass->ctype = req.get_header_value("Content-Type");
  // httplib на сервере разбирает multipart в req.files — req.body остаётся
  // пустым (read_content_core). Для сырого форварда собираем тело обратно:
  // /v1/images/edits sd-server ждёт multipart (замер 08.11: форвард с пустым
  // телом давал бессмысленный 400 «prompt required» от ребёнка).
  if (pass->body.empty() && !req.files.empty()) {
    std::string boundary = httplib::detail::make_multipart_data_boundary();
    httplib::MultipartFormDataItems items;
    for (const auto& [k, f] : req.files)
      items.push_back({f.name, f.content, f.filename, f.content_type});
    pass->body = httplib::detail::serialize_multipart_formdata(items, boundary);
    pass->ctype = httplib::detail::serialize_multipart_formdata_get_content_type(boundary);
  }
  pass->accept = req.get_header_value("Accept");
  pass->port = port;
  pass->image = true;
  // The worker survives handle (detach): the generation continues even if the
  // client leaves, ImageBusy lives until the forward completes.
  sup::track_background(
      [pass, chan, dead, up, rq, busy] { forward_worker(pass, chan, dead, up, rq); });
  stream_reply(res, up, chan, dead, /*busy_model=*/nullptr, done, {},
               req.connection_alive, pass);
}

void handle(const httplib::Request& req, httplib::Response& res) {
  auto rq = std::make_shared<Rq>();
  auto done = make_done(metrics::endpoint_of(req.target), rq);

  if (!check_auth(req, res, done)) return;

  // Allow-list поверхности: наружу — только OpenAI-фасад /v1* и /health,
  // методы GET/POST. Прокси подставляет ребёнку его собственный --api-key,
  // поэтому прежний проброс «любой путь любым методом» превращал владение
  // /v1-токеном во владение всем API llama-server: админские /models/load и
  // /models/unload, /props (пути и конфигурация), /slots — всё в обход
  // занятости, гейтов и учёта llmhost. Граница префикса строгая: «/v1» или
  // «/v1/…»; «/v1xyz» не пропускаем.
  {
    const std::string_view target = req.target;
    const std::string_view path = target.substr(0, target.find('?'));
    const bool v1 = path == "/v1" ||
                    (path.size() > 3 && path.substr(0, 3) == "/v1" && path[3] == '/');
    // Image-маршруты — те же два семейства, что у sd-server (OpenAI images +
    // его нативные API), граница префикса строгая, как у /v1.
    auto prefix = [&path](std::string_view p) {
      return path == p || (path.size() > p.size() && path.substr(0, p.size()) == p &&
                           path[p.size()] == '/');
    };
    const bool image_family =
        prefix("/v1/images") || prefix("/sdapi/v1") || prefix("/sdcpp/v1");
    const bool ok_path = v1 || image_family || path == "/health";
    const bool ok_method = req.method == "GET" || req.method == "POST";
    if (!ok_path || !ok_method) {
      done("not_found", 404);
      json_error(res, 404,
                 tr("на OpenAI-порту доступны только /v1/* (GET/POST) и /health"));
      return;
    }
    if (image_family) return handle_image(req, res, done, rq);
  }

  if (handle_harness_get(req, res, done)) return;

  // Подставить «model», если клиент не назвал внятную модель, и пометить
  // занятость модели — сторож простоя не выгрузит её посреди ответа.
  std::string body, model_id;
  int up_port = 0;
  std::string up_key;
  std::string note; // диагноз деградации слота — на ответе чата (см. stream_reply)
  std::shared_ptr<sup::BusyModel> busy_model;
  if (!resolve_and_wake(req, res, done, rq, body, model_id, up_port, up_key,
                        busy_model, &note))
    return;

  // Отмена во время будилки (модель грузилась до 300 с): клиент ушёл —
  // форвард не начинаем. Модель остаётся резидентом (штатно), занятость
  // снимет деструктор busy_model при выходе.
  if (req.connection_alive && !req.connection_alive()) {
    logf("proxy: %s %s — клиент оборвал соединение при будилке «%s»",
         req.method.c_str(), req.target.c_str(), model_id.c_str());
    done("client_gone", 499);
    res.status = 499; // сокет мёртв — запись до клиента не дойдёт
    return;
  }

  sup::touch_model(model_id);

  auto chan = std::make_shared<Chan>();
  auto dead = std::make_shared<std::atomic<bool>>(false);
  auto up = std::make_shared<Upstream>();

  auto pass = std::make_shared<Pass>();
  pass->method = req.method;
  pass->target = req.target;
  pass->body = body;
  pass->ctype = req.get_header_value("Content-Type");
  pass->accept = req.get_header_value("Accept");
  pass->model = model_id;
  pass->port = up_port;
  pass->key = up_key;

  // Воркер переживает handle (detach): клиент может уйти посреди генерации,
  // а заголовочный таймаут — вернуть handler раньше ответа upstream.
  sup::track_background(
      [pass, chan, dead, up, rq] { forward_worker(pass, chan, dead, up, rq); });

  stream_reply(res, up, chan, dead, busy_model, done, note, req.connection_alive,
               pass);
}

std::shared_ptr<httplib::Server> g_svr;
std::thread g_thr;

} // namespace

bool start(int port, const std::string& bind) {
  g_svr = std::make_shared<httplib::Server>();
  auto svr = g_svr;
  // Пул обработчиков по умолчанию = hardware_concurrency−1: живой SSE-стрим
  // и ожидание будилки (до 300 с) держат воркер целиком, и восемь параллельных
  // «спящих» запросов голодали бы весь порт. Явный пул шире — очереди задач
  // httplib с ограничением не имеет, поэтому потолок именно за воркерами.
  svr->new_task_queue = [] { return new httplib::ThreadPool(256); };
  svr->set_payload_max_length(256 * 1024 * 1024); // длинные промпты и вложения
  // Явные лимиты: локальный хулиган не должен вешать потоки вечным чтением
  // запроса. Запись — щедро (SSE-генерация думает минутами до первого токена),
  // чтение — 30 с на порцию данных с лихвой хватает любому живому клиенту.
  svr->set_read_timeout(30, 0);
  svr->set_write_timeout(600, 0);
  auto handler = [](const httplib::Request& req, httplib::Response& res) {
    std::string origin = req.get_header_value("Origin");
    bool allow_cors = !origin.empty() &&
                      trusted_origin(origin, config::current().cors_origins);
    try {
      if (req.method == "OPTIONS" && req.has_header("Origin")) {
        // Preflight браузера: Authorization браузер не шлёт — проверку токена и
        // ход в роутер пропускаем. CORS-заголовки ставим сами: поведение тонкого
        // слоя не зависит от версии роутера. Allow-Origin здесь НЕ ставим:
        // общий хвост ниже ставит его на каждый доверенный ответ, а set_header
        // httplib добавляет второе значение, не перезаписывая первое — дубль
        // браузер отвергает целиком («contains multiple values»), preflight
        // падает, и настоящий запрос к модели так и не уходит.
        res.status = 204;
        if (allow_cors) {
          std::string hdrs = req.get_header_value("Access-Control-Request-Headers");
          res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
          res.set_header("Access-Control-Allow-Headers",
                         hdrs.empty() ? "Authorization, Content-Type" : hdrs);
          res.set_header("Access-Control-Max-Age", "86400");
        }
      } else {
        handle(req, res);
      }
    } catch (const std::exception& e) {
      // Исключение — в журнал (там разберутся), клиенту — общая фраза:
      // text исключений может содержать фрагменты путей и разбора.
      logf("proxy: %s %s — внутренняя ошибка: %s", req.method.c_str(),
           req.target.c_str(), e.what());
      json_error(res, 500, tr("внутренняя ошибка — подробности в журнале демона"));
    }
    // CORS-эхо Origin на любом ответе — но только доверенному источнику
    // (localhost/127.0.0.1 и явный список из конфига): посторонний сайт в
    // браузере ответ прочесть не должен, даже при выключенной авторизации.
    if (allow_cors) {
      res.set_header("Access-Control-Allow-Origin", origin);
      res.set_header("Vary", "Origin");
    }
  };
  // Любой путь и метод уходит в llama-server.
  svr->Get(".*", handler);
  svr->Post(".*", handler);
  svr->Put(".*", handler);
  svr->Delete(".*", handler);
  svr->Options(".*", handler);

  // Результат bind — наружу, синхронно: занятый порт не должен оставлять
  // полуживой демон (процесс жив, OpenAI endpoint мёртв) — старт обязан
  // отказать.
  std::promise<bool> bound;
  g_thr = std::thread([svr, port, bind, &bound]() {
    if (!svr->bind_to_port(bind, port)) {
      logf("proxy: %s:%d занят — OpenAI endpoint не поднят", bind.c_str(), port);
      bound.set_value(false);
      return;
    }
    logf("proxy: слушаю %s:%d", bind.c_str(), port);
    bound.set_value(true);
    svr->listen_after_bind();
  });
  return bound.get_future().get();
}

void stop() {
  if (g_svr) g_svr->stop();
  if (g_thr.joinable()) g_thr.join();
  g_svr.reset();
}

} // namespace lh::proxy
