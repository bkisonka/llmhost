#include "quality.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace lh::quality {

namespace {

// Запрос к ребёнку — тот же фасад, что у прогрева (sup_spawn::warmup):
// OpenAI-порт ребёнка, temperature 0 и seed — детерминизм суита.
nlohmann::json chat_once(int port, const std::string& key,
                         const nlohmann::json& messages, const nlohmann::json* tools,
                         int max_tokens, std::string& err) {
  httplib::Client cli("127.0.0.1", port);
  cli.set_connection_timeout(std::chrono::seconds(5));
  cli.set_read_timeout(std::chrono::seconds(300)); // префилл большого needle
  nlohmann::json body = {
      {"messages", messages},
      {"temperature", 0.0},
      {"seed", 424242},
      {"max_tokens", max_tokens},
      {"stream", false},
  };
  if (tools) body["tools"] = *tools;
  httplib::Headers hdrs = {{"Authorization", "Bearer " + key}};
  auto res = cli.Post("/v1/chat/completions", hdrs, body.dump(), "application/json");
  if (!res || res->status != 200) {
    err = res ? "ребёнок ответил " + std::to_string(res->status) +
                    (res->body.empty()
                         ? ""
                         : ": " + res->body.substr(0, 200))
              : "ребёнок недоступен: " + httplib::to_string(res.error());
    return {};
  }
  try {
    auto j = nlohmann::json::parse(res->body);
    const auto& ch = j["choices"].at(0);
    return ch["message"];
  } catch (const std::exception& e) {
    err = std::string("ответ не разбирается: ") + e.what();
    return {};
  }
}

// «профиль reading» ответа: content + reasoning (thinking-модели отвечают
// content'ом после рассуждений; скор ищем по обоим полям — проверка
// детерминированная, текст рассуждений тоже часть ответа модели).
std::string answer_text(const nlohmann::json& msg) {
  std::string out;
  if (msg.is_object()) {
    if (msg.contains("content") && msg["content"].is_string())
      out += msg["content"].get<std::string>();
    if (msg.contains("reasoning_content") && msg["reasoning_content"].is_string())
      out += " " + msg["reasoning_content"].get<std::string>();
  }
  return out;
}

// ---- задачи суита ------------------------------------------------------------

// Needle-recall: факт в середину объёмного текста (глубина ~70%), вопрос в
// конец. Филлер — на контексте замера, с потолком 8192 токенов: задача
// суита — РАЗЛИЧАТЬ ФОРМЫ одной модели (kv q8/f16, кэш MoE), а не мерить
// абсолютную глубину контекста; на 262k-моделях полный ctx делал бы пробу
// минутной в окне простоя (сам ребёнок меряет скорость на своём замере —
// здесь потолок сознательный и записан).
double task_needle(int port, const std::string& key, uint32_t ctx,
                   std::string& terr) {
  const uint32_t cap_tokens = std::min<uint32_t>(std::max<uint32_t>(ctx, 1024u), 8192u);
  // Русский: ~3 символа на токен; абзац ~120 байт ≈ 40 токенов.
  const size_t want_chars = (size_t)cap_tokens * 3;
  const char* filler_fmt = "Служебный абзац про устройство мира номер %d. "
                           "Здесь ничего важного нет, только объём текста. ";
  const char* needle = "Координата станции наблюдения: ЗЕЛЁНЫЙ-2718. ";
  std::string text;
  int i = 0;
  size_t needle_pos = (size_t)(want_chars * 7 / 10);
  while (text.size() < want_chars) {
    if (text.size() < needle_pos && text.size() + 48 > needle_pos) {
      text += needle; // глубина ~70% объёма
      continue;
    }
    char buf[256];
    std::snprintf(buf, sizeof buf, filler_fmt, ++i);
    text += buf;
  }
  nlohmann::json msgs = nlohmann::json::array({nlohmann::json{
      {"role", "user"},
      {"content", text +
                      "\n\nКакая координата станции наблюдения упоминается в "
                      "тексте? Назови её точно, одним словом-числом."}}});
  std::string err;
  auto msg = chat_once(port, key, msgs, nullptr, 64, err);
  if (msg.is_null()) { terr = err; return -1.0; }
  // Проверка по числу needle'а: регистр букв ответа — территория модели
  // (std::tolower байтом UTF-8-кириллицу не берёт), филлер номеров не
  // достигает 2718 (абзацев при потолке 8192 токенов ~510), так что число
  // однозначно указывает на вычитанный факт.
  const std::string a = answer_text(msg);
  return a.find("2718") != std::string::npos ? 1.0 : 0.0;
}

// Tool-call: небольшая схема (два инструмента), запрос требует вызов первого
// с конкретным аргументом. Проверка — имя вызова и аргумент; текст ответа не
// важен.
double task_tools(int port, const std::string& key, std::string& terr) {
  nlohmann::json tools = nlohmann::json::array();
  tools.push_back({
      {"type", "function"},
      {"function",
       {{"name", "get_weather"},
        {"description", "Текущая погода в городе"},
        {"parameters",
         {{"type", "object"},
          {"properties",
           {{"city", {{"type", "string"}}}, {"unit", {{"type", "string"}}}}},
          {"required", nlohmann::json::array({"city"})}}}}},
  });
  tools.push_back({
      {"type", "function"},
      {"function",
       {{"name", "get_time"},
        {"description", "Текущее время в часовом поясе"},
        {"parameters",
         {{"type", "object"},
          {"properties", {{"tz", {{"type", "string"}}}}},
          {"required", nlohmann::json::array({"tz"})}}}}},
  });
  nlohmann::json msgs = nlohmann::json::array(
      {nlohmann::json{{"role", "user"},
                      {"content", "Какая погода в Казани? Используй инструменты."}}});
  std::string err;
  auto msg = chat_once(port, key, msgs, &tools, 128, err);
  if (msg.is_null()) { terr = err; return -1.0; }
  if (!msg.is_object() || !msg.contains("tool_calls") || !msg["tool_calls"].is_array() ||
      msg["tool_calls"].empty())
    return 0.0; // модель не вызвала инструмент — корректность 0
  const auto& call = msg["tool_calls"][0];
  if (!call.is_object() || !call.contains("function")) return 0.0;
  const std::string name = call["function"].value("name", "");
  if (name != "get_weather") return 0.0; // wrong tool
  std::string args = call["function"].contains("arguments")
                         ? call["function"]["arguments"].dump()
                         : std::string();
  if (call["function"].contains("arguments") &&
      call["function"]["arguments"].is_string())
    args = call["function"]["arguments"].get<std::string>();
  return args.find("азань") != std::string::npos ? 1.0 : 0.0; // регистр первой буквы — территория модели
}

// Код-челлендж: детерминированная проверка ответа БЕЗ исполнения кода —
// арифметика с фиксированным результатом и строгой формой JSON.
double task_code(int port, const std::string& key, std::string& terr) {
  nlohmann::json msgs = nlohmann::json::array({nlohmann::json{
      {"role", "user"},
      {"content", "Вычисли 17 + 25 и верни ровно JSON вида {\"sum\": N}, "
                  "где N — результат. Никакого текста, только JSON."}}});
  std::string err;
  auto msg = chat_once(port, key, msgs, nullptr, 128, err);
  if (msg.is_null()) { terr = err; return -1.0; }
  const std::string a = answer_text(msg);
  // Форма важна так же, как число: JSON ровно с полем sum = 42.
  const size_t b0 = a.find('{'), b1 = a.find('}');
  if (b0 == std::string::npos || b1 == std::string::npos || b1 <= b0) return 0.0;
  try {
    auto j = nlohmann::json::parse(a.substr(b0, b1 - b0 + 1));
    if (j.is_object() && j.contains("sum") && j["sum"].is_number_integer() &&
        j["sum"].get<long long>() == 42)
      return 1.0;
  } catch (...) {}
  return 0.0;
}

// Согласие: одна задача рассуждению, где ответ канонический (точный счёт в
// структурированном тексте) — расхождение двух temperature=0 ответов signal
// ошибки рассуждения, извлечение и tool-call её не ловят.
const char* kAgreementTask =
    "В списке: яблоко, груша, слива, яблоко, груша, айва, слива, яблоко, айва, "
    "яблоко. Сколько раз встречается слово «яблоко»? Ответь только числом.";

} // namespace

bool run_suite(int port, const std::string& key, uint32_t ctx, Scores& out,
               std::string& err) {
  std::string nerr, terr, cerr;
  out.needle = task_needle(port, key, ctx, nerr);
  out.tools = task_tools(port, key, terr);
  out.code = task_code(port, key, cerr);
  int taken = 0;
  for (double v : {out.needle, out.tools, out.code})
    if (v >= 0.0) ++taken;
  if (taken == 0) {
    // Честные причины: по задаче на каждую.
    err = "ни одна задача суита не снялась (needle: " + nerr + "; tools: " + terr +
          "; code: " + cerr + ")";
    return false;
  }
  out.tasks = taken;
  return true;
}

bool run_agreement(int worker_port, const std::string& worker_key,
                   int verifier_port, const std::string& verifier_key,
                   double& score, std::string& err) {
  nlohmann::json msgs = nlohmann::json::array(
      {nlohmann::json{{"role", "user"}, {"content", kAgreementTask}}});
  std::string werr, verr;
  auto w = chat_once(worker_port, worker_key, msgs, nullptr, 96, werr);
  auto v = chat_once(verifier_port, verifier_key, msgs, nullptr, 96, verr);
  if (w.is_null() || v.is_null()) {
    err = "согласие не снялось: worker (" + werr + "), verifier (" + verr + ")";
    return false; // «нет», а не ноль — одному из детей ответ взять не удалось
  }
  // Ответ — число; сравниваем канонически: последняя числовая лексема
  // каждого ответа (thinking-модели пишут «Ответ: 4» после рассуждений).
  auto num_of = [](const std::string& s) -> std::string {
    std::string cur, last;
    for (char c : s) {
      if (std::isdigit((unsigned char)c)) cur += c;
      else if (!cur.empty()) { last = cur; cur.clear(); }
    }
    if (!cur.empty()) last = cur;
    return last;
  };
  const std::string wn = num_of(answer_text(w)), vn = num_of(answer_text(v));
  if (wn.empty() || vn.empty()) {
    err = "согласие не снялось: ответ без числа";
    return false;
  }
  score = wn == vn ? 1.0 : 0.0;
  return true;
}

} // namespace lh::quality
