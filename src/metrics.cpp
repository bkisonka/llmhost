#include "metrics.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <map>
#include <mutex>
#include <sstream>

namespace lh::metrics {

namespace {

// Хвост потока, достаточный для usage: финальное SSE-событие и хвост JSON
// укладываются с запасом, а память — константная на запрос.
constexpr size_t kTail = 4096;

struct Stats {
  std::map<std::string, uint64_t> requests;    // endpoint → счётчик
  std::map<std::string, uint64_t> errors;      // kind → счётчик
  std::map<std::string, uint64_t> bytes;       // endpoint → байты ответа
  std::map<std::string, double> seconds;       // endpoint → сумма длительности
  std::map<std::string, uint64_t> secs_count;  // endpoint → число замеров
  uint64_t prompt = 0, completion = 0;         // токены из usage
  double tps_ewma = 0;                         // ток/с со сглаживанием
  // Очередь пробуждения (harness): ждут ли модель сейчас и сколько уже
  // нажевались. wait_ms — сумма ожиданий в парковке по завершении ожидания.
  uint64_t queue_wait_ms = 0, queue_wait_count = 0, queue_timeouts = 0;
  int queue_depth_now = 0;
};

std::mutex g_mtx;
std::map<std::string, Stats> g_models; // ключ — id модели, "" — модель неизвестна

// Клиентское «model» из /v1 — произвольная строка (default_chat_model
// пропускает запрошенное имя как есть): карта не должна расти со стороны
// клиента. Длинные ключи сводим в «?», а когда неизвестных ключей набралось
// до потолка — все дальнейшие неизвестные тоже идут в «?», известные копят
// как обычно. Без этого любой держатель токена (а при openai_auth=false — любой
// достигающий порта) раздувал RAM демона и число строк в /v0/metrics на всё
// время жизни процесса.
constexpr size_t kMaxModelKeys = 4096;
constexpr size_t kMaxKeyLen = 128;

Stats& slot(const std::string& model) { // вызывается под g_mtx
  std::string key = model.size() > kMaxKeyLen ? std::string("?") : model;
  auto it = g_models.find(key);
  if (it == g_models.end()) {
    if (g_models.size() >= kMaxModelKeys) key = "?";
    it = g_models.find(key);
    if (it == g_models.end()) it = g_models.emplace(std::move(key), Stats{}).first;
  }
  return it->second;
}

// Число после ключа вида "prompt_tokens": встречаем ':', пропускаем разделители.
int64_t grab_number(const std::string& tail, const char* key) {
  size_t p = tail.find(key);
  if (p == std::string::npos) return -1;
  p += strlen(key);
  while (p < tail.size() &&
         (tail[p] == ' ' || tail[p] == ':' || tail[p] == '\n' || tail[p] == '\r' ||
          tail[p] == '\t'))
    ++p;
  if (p >= tail.size() || !std::isdigit((unsigned char)tail[p])) return -1;
  uint64_t v = 0;
  while (p < tail.size() && std::isdigit((unsigned char)tail[p]))
    v = v * 10 + (uint64_t)(tail[p++] - '0');
  return (int64_t)v;
}

// Экранирование значения метки Prometheus: \ " \n \r; прочие control-байты
// (<0x20) — в подчёркивание: exposition-поток не должен рваться ни чем,
// редкие парсеры считают одиночный CR переводом строки. Клиентское «model»
// попадает в метку прямо из тела запроса.
std::string label(const std::string& v) {
  std::string out;
  out.reserve(v.size());
  for (char c : v) {
    switch (c) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      default: out += ((unsigned char)c < 0x20) ? '_' : c;
    }
  }
  return out;
}

void family(std::ostringstream& out, const char* name, const char* type,
            const char* help) {
  out << "# HELP " << name << ' ' << help << "\n# TYPE " << name << ' ' << type
      << '\n';
}

} // namespace

void UsageSniffer::feed(const char* data, size_t n) {
  std::lock_guard l(g_mtx);
  tail_.append(data, std::min(n, kTail));
  if (tail_.size() > kTail) tail_.erase(0, tail_.size() - kTail);
}

Usage UsageSniffer::finish() {
  std::lock_guard l(g_mtx);
  Usage u;
  // OpenAI-имена (/v1/chat/completions) и сырые /completion-имена llama.cpp —
  // кто из них пришёл, тем и живём.
  int64_t p = grab_number(tail_, "\"prompt_tokens\"");
  if (p < 0) p = grab_number(tail_, "\"tokens_evaluated\"");
  int64_t c = grab_number(tail_, "\"completion_tokens\"");
  if (c < 0) c = grab_number(tail_, "\"tokens_predicted\"");
  u.seen = p >= 0 || c >= 0;
  u.prompt = p > 0 ? (uint64_t)p : 0;
  u.completion = c > 0 ? (uint64_t)c : 0;
  return u;
}

const char* endpoint_of(const std::string& target) {
  if (target.find("/chat/completions") != std::string::npos) return "chat";
  if (target.find("/completions") != std::string::npos ||
      target.find("/completion") != std::string::npos)
    return "completion";
  if (target.find("/embeddings") != std::string::npos) return "embeddings";
  return "other";
}

void observe(const std::string& model, const Outcome& o) {
  const char* ep = o.endpoint;
  std::lock_guard l(g_mtx);
  Stats& s = slot(model);
  ++s.requests[ep];
  if (std::strcmp(o.kind, "ok") != 0) ++s.errors[o.kind];
  s.bytes[ep] += o.bytes;
  s.seconds[ep] += o.seconds;
  ++s.secs_count[ep];
  s.prompt += o.usage.prompt;
  s.completion += o.usage.completion;
  if (o.usage.completion > 0 && o.seconds > 0) {
    const double tps = (double)o.usage.completion / o.seconds;
    s.tps_ewma = s.tps_ewma == 0 ? tps : s.tps_ewma * 0.7 + tps * 0.3;
  }
}

void queue_depth(const std::string& model, int waiting) {
  std::lock_guard l(g_mtx);
  slot(model).queue_depth_now = waiting;
}

void queue_wait(const std::string& model, uint64_t ms) {
  std::lock_guard l(g_mtx);
  Stats& s = slot(model);
  s.queue_wait_ms += ms;
  ++s.queue_wait_count;
}

void queue_timeout(const std::string& model) {
  std::lock_guard l(g_mtx);
  ++slot(model).queue_timeouts;
}

nlohmann::json parse_engine(const std::string& text) {
  nlohmann::json o = nlohmann::json::object();
  for (size_t pos = 0; pos < text.size();) {
    size_t eol = text.find('\n', pos);
    if (eol == std::string::npos) eol = text.size();
    const std::string line = text.substr(pos, eol - pos);
    pos = eol + 1;
    if (line.rfind("llamacpp:", 0) != 0) continue;
    const size_t sp = line.find(' ');
    if (sp == std::string::npos) continue;
    std::string key = line.substr(9, sp - 9); // без префикса «llamacpp:»
    const size_t brace = key.find('{');
    if (brace != std::string::npos) key = key.substr(0, brace);
    try {
      size_t used = 0;
      const double v = std::stod(line.substr(sp + 1), &used);
      if (used == line.size() - sp - 1) o[key] = v; // хвост «12x» — мусор, мимо
    } catch (...) {
    } // битая строка телеметрии не роняет рендер
  }
  return o;
}

std::string render(const nlohmann::json& status, const nlohmann::json& models_state,
                   const nlohmann::json& engine) {
  std::lock_guard l(g_mtx);
  std::ostringstream out;

  out << "# HELP llmhost_up llmhost daemon отвечает\n"
         "# TYPE llmhost_up gauge\nllmhost_up 1\n";

  if (status.is_object()) {
    family(out, "llmhost_supervisor_state", "gauge",
           "состояние супервизора (1 у текущего значения state)");
    out << "llmhost_supervisor_state{state=\"" << label(status.value("state", "?"))
        << "\"} 1\n";
    family(out, "llmhost_vram_bytes", "gauge", "VRAM карты (kind=free|total), байты");
    if (status.contains("gpus") && status["gpus"].is_array())
      for (auto& g : status["gpus"]) {
        if (!g.is_object()) continue;
        int idx = g.value("index", -1);
        const std::string card = label(idx >= 0 ? std::to_string(idx) : "?");
        const std::string name = label(g.value("name", std::string("?")));
        out << "llmhost_vram_bytes{card=\"" << card << "\",name=\"" << name
            << "\",kind=\"free\"} " << g.value("vram_free", (uint64_t)0) << '\n';
        out << "llmhost_vram_bytes{card=\"" << card << "\",name=\"" << name
            << "\",kind=\"total\"} " << g.value("vram_total", (uint64_t)0) << '\n';
      }
  }

  if (models_state.is_object()) {
    family(out, "llmhost_model_state", "gauge",
           "живое состояние модели роутера (1 у текущего значения state)");
    for (auto& [id, obj] : models_state.items()) {
      if (!obj.is_object()) continue;
      std::string st = obj.value("state", std::string("unloaded"));
      if (st.empty()) st = "unloaded";
      out << "llmhost_model_state{model=\"" << label(id) << "\",state=\"" << label(st)
          << "\"} 1\n";
    }
  }

  // Точная телеметрия движков (дети llama-server с --metrics): скорости и
  // приём спекции из /metrics живых детей, по-модельно. Секунды декода
  // рядом с токенами — клиент считает мгновенную скорость по rate().
  if (engine.is_array()) {
    family(out, "llmhost_engine_decode_tps", "gauge",
           "средняя скорость декода движка (пожизненная, из /metrics ребёнка)");
    family(out, "llmhost_engine_prompt_tps", "gauge",
           "средняя скорость префилла движка (без кэшированных токенов)");
    family(out, "llmhost_engine_decode_tokens_total", "counter",
           "декодированные токены движка");
    family(out, "llmhost_engine_decode_seconds_total", "counter",
           "время декода движка (мгновенная скорость — по rate() в Prometheus)");
    family(out, "llmhost_engine_prompt_tokens_total", "counter",
           "обработанные промпт-токены движка (без кэшированных)");
    family(out, "llmhost_engine_spec_accepted_total", "counter",
           "принятые спекуляцией черновик-токены");
    family(out, "llmhost_engine_spec_draft_total", "counter",
           "сгенерированные черновиком токены");
    for (auto& e : engine) {
      if (!e.is_object()) continue;
      const std::string m = label(e.value("model", std::string("?")));
      const nlohmann::json& s =
          e.contains("series") && e["series"].is_object() ? e["series"]
                                                          : nlohmann::json::object();
      auto num = [&](const char* k) -> double {
        return s.contains(k) && s[k].is_number() ? s[k].get<double>() : 0.0;
      };
      const double dtps = num("predicted_tokens_seconds");
      const double ptps = num("prompt_tokens_seconds");
      const double dtok = num("tokens_predicted_total");
      const double dsec = num("tokens_predicted_seconds_total");
      const double ptok = num("prompt_tokens_total");
      const double acc = num("spec_decode_num_accepted_tokens_total");
      const double dr = num("spec_decode_num_draft_tokens_total");
      if (dtps > 0)
        out << "llmhost_engine_decode_tps{model=\"" << m << "\"} " << dtps << '\n';
      if (ptps > 0)
        out << "llmhost_engine_prompt_tps{model=\"" << m << "\"} " << ptps << '\n';
      if (dtok > 0)
        out << "llmhost_engine_decode_tokens_total{model=\"" << m << "\"} " << dtok
            << '\n';
      if (dsec > 0)
        out << "llmhost_engine_decode_seconds_total{model=\"" << m << "\"} " << dsec
            << '\n';
      if (ptok > 0)
        out << "llmhost_engine_prompt_tokens_total{model=\"" << m << "\"} " << ptok
            << '\n';
      if (acc > 0)
        out << "llmhost_engine_spec_accepted_total{model=\"" << m << "\"} " << acc
            << '\n';
      if (dr > 0)
        out << "llmhost_engine_spec_draft_total{model=\"" << m << "\"} " << dr << '\n';
    }
  }

  family(out, "llmhost_requests_total", "counter", "запросы через прокси");
  family(out, "llmhost_errors_total", "counter", "ошибки по исходу (kind)");
  family(out, "llmhost_prompt_tokens_total", "counter",
         "токены промпта (из usage ответов)");
  family(out, "llmhost_completion_tokens_total", "counter",
         "сгенерированные токены (из usage)");
  family(out, "llmhost_response_bytes_total", "counter", "байты ответов клиенту");
  family(out, "llmhost_request_seconds", "counter",
         "длительность запросов (_sum/_count — среднее по rate())");
  family(out, "llmhost_generation_tps", "gauge",
         "ток/с генерации (EWMA по завершённым запросам с usage)");
  family(out, "llmhost_queue_depth", "gauge",
         "ждущих в очереди пробуждения сейчас (модель → число)");
  family(out, "llmhost_queue_wait_ms", "counter",
         "ожидание в очереди пробуждения (_sum/_count — среднее по rate())");
  family(out, "llmhost_queue_timeouts_total", "counter",
         "запросы, не дождавшиеся освобождения памяти (TTL очереди)");

  for (auto& [model, s] : g_models) {
    const std::string m = label(model);
    for (auto& [ep, n] : s.requests)
      out << "llmhost_requests_total{model=\"" << m << "\",endpoint=\"" << ep
          << "\"} " << n << '\n';
    for (auto& [kind, n] : s.errors)
      out << "llmhost_errors_total{model=\"" << m << "\",kind=\"" << kind
          << "\"} " << n << '\n';
    if (s.prompt)
      out << "llmhost_prompt_tokens_total{model=\"" << m << "\"} " << s.prompt << '\n';
    if (s.completion)
      out << "llmhost_completion_tokens_total{model=\"" << m << "\"} " << s.completion
          << '\n';
    for (auto& [ep, n] : s.bytes)
      out << "llmhost_response_bytes_total{model=\"" << m << "\",endpoint=\"" << ep
          << "\"} " << n << '\n';
    for (auto& [ep, v] : s.seconds)
      out << "llmhost_request_seconds_sum{model=\"" << m << "\",endpoint=\"" << ep
          << "\"} " << v << '\n';
    for (auto& [ep, n] : s.secs_count)
      out << "llmhost_request_seconds_count{model=\"" << m << "\",endpoint=\"" << ep
          << "\"} " << n << '\n';
    if (s.tps_ewma > 0)
      out << "llmhost_generation_tps{model=\"" << m << "\"} " << s.tps_ewma << '\n';
    if (s.queue_depth_now > 0)
      out << "llmhost_queue_depth{model=\"" << m << "\"} " << s.queue_depth_now << '\n';
    if (s.queue_wait_count) {
      out << "llmhost_queue_wait_ms_sum{model=\"" << m << "\"} " << s.queue_wait_ms
          << '\n';
      out << "llmhost_queue_wait_ms_count{model=\"" << m << "\"} "
          << s.queue_wait_count << '\n';
    }
    if (s.queue_timeouts)
      out << "llmhost_queue_timeouts_total{model=\"" << m << "\"} "
          << s.queue_timeouts << '\n';
  }
  return out.str();
}

} // namespace lh::metrics
