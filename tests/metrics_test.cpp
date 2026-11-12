// Тест агрегатора метрик: счётчики по исходам, снифер usage (в т.ч. usage,
// разрезанный между чанками, и сырые /completion-имена), экранирование меток.
#include "metrics.h"
#include "fmt.h"

#include <nlohmann/json.hpp>

#include <cassert>
#include <cstdio>
#include <set>
#include <string>

using namespace lh;

static bool has_line(const std::string& text, const std::string& line) {
  return text.find(line + '\n') != std::string::npos;
}

int main() {
  // endpoint по пути: константная кардинальность, что бы ни прислал клиент.
  assert(std::string(metrics::endpoint_of("/v1/chat/completions")) == "chat");
  assert(std::string(metrics::endpoint_of("/v1/completions")) == "completion");
  assert(std::string(metrics::endpoint_of("/completion")) == "completion");
  assert(std::string(metrics::endpoint_of("/v1/embeddings")) == "embeddings");
  assert(std::string(metrics::endpoint_of("/v1/models")) == "other");

  // ---- снифер usage -------------------------------------------------------
  // usage разрезан между чанками; длины — strlen, чтобы feed не дочитал мусор.
  metrics::UsageSniffer s;
  s.feed("data: {\"choices\":[{\"delta\":{\"content\":\"прив\"}]}",
         std::strlen("data: {\"choices\":[{\"delta\":{\"content\":\"прив\"}]}"));
  s.feed("er\",\"usage\":{\"prompt_tokens\":120,\"comp",
         std::strlen("er\",\"usage\":{\"prompt_tokens\":120,\"comp"));
  s.feed("letion_tokens\":37}}\n\ndata: [DONE]\n\n",
         std::strlen("letion_tokens\":37}}\n\ndata: [DONE]\n\n"));
  metrics::Usage u = s.finish();
  assert(u.seen && u.prompt == 120 && u.completion == 37);

  // Сырые имена /completion llama.cpp.
  metrics::UsageSniffer raw;
  raw.feed("{\"tokens_predicted\": 42, \"tokens_evaluated\": 7, \"trailing\": 1}",
           std::strlen("{\"tokens_predicted\": 42, \"tokens_evaluated\": 7, \"trailing\": 1}"));
  u = raw.finish();
  assert(u.seen && u.prompt == 7 && u.completion == 42);

  // Без usage — честный «не видел» (embeddings, ошибки, стримы без include_usage).
  metrics::UsageSniffer none;
  none.feed("data: {\"choices\":[{\"delta\":{\"content\":\"...\"}}]}\n\n",
            std::strlen("data: {\"choices\":[{\"delta\":{\"content\":\"...\"}}]}\n\n"));
  u = none.finish();
  assert(!u.seen);

  // Длинный поток: хвост ограничен, старое выталкивается — usage в конце ловится.
  metrics::UsageSniffer long_stream;
  std::string chunk(2048, 'x');
  for (int i = 0; i < 10; ++i) long_stream.feed(chunk.data(), chunk.size());
  long_stream.feed("{\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":6}}",
                   std::strlen("{\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":6}}"));
  u = long_stream.finish();
  assert(u.seen && u.prompt == 5 && u.completion == 6);

  // ---- счётчики и рендер --------------------------------------------------
  metrics::observe("m1", {metrics::endpoint_of("/v1/chat/completions"), "ok", 200,
                          2.0, 800, {100, 60}});
  metrics::observe("m1", {metrics::endpoint_of("/v1/chat/completions"), "ok", 200,
                          4.0, 900, {100, 120}});
  metrics::observe("m1", {"chat", "upstream_error", 500, 0.1, 64, {}});
  metrics::observe("m1", {"chat", "client_gone", 200, 1.0, 10, {}});
  metrics::observe("m2", {"embeddings", "ok", 200, 0.05, 300, {}});
  metrics::observe("", {"other", "auth", 401, 0.0, 0, {}});

  const std::string out =
      metrics::render(nlohmann::json{{"state", "loaded"},
                                     {"gpus", nlohmann::json::array({
                                          nlohmann::json{{"index", 1},
                                                         {"name", "RTX \"PRO\" 4000"},
                                                         {"vram_free", 1u << 30},
                                                         {"vram_total", 2u << 30}},
                                      })}},
                      nlohmann::json{{"m1", {{"state", "loaded"}}},
                                     {"m2", {{"state", "unloaded"}}}});

  assert(has_line(out, "llmhost_up 1"));
  assert(has_line(out, "llmhost_supervisor_state{state=\"loaded\"} 1"));
  // Кавычки в имени карты экранированы — рендер не ломает формат.
  assert(has_line(out, "llmhost_vram_bytes{card=\"1\",name=\"RTX \\\"PRO\\\" 4000\",kind=\"free\"} 1073741824"));
  assert(has_line(out, "llmhost_model_state{model=\"m1\",state=\"loaded\"} 1"));
  assert(has_line(out, "llmhost_model_state{model=\"m2\",state=\"unloaded\"} 1"));
  assert(has_line(out, "llmhost_requests_total{model=\"m1\",endpoint=\"chat\"} 4"));
  assert(has_line(out, "llmhost_requests_total{model=\"m2\",endpoint=\"embeddings\"} 1"));
  assert(has_line(out, "llmhost_errors_total{model=\"m1\",kind=\"upstream_error\"} 1"));
  assert(has_line(out, "llmhost_errors_total{model=\"m1\",kind=\"client_gone\"} 1"));
  assert(has_line(out, "llmhost_errors_total{model=\"\",kind=\"auth\"} 1"));
  assert(has_line(out, "llmhost_prompt_tokens_total{model=\"m1\"} 200"));
  assert(has_line(out, "llmhost_completion_tokens_total{model=\"m1\"} 180"));
  assert(has_line(out, "llmhost_response_bytes_total{model=\"m1\",endpoint=\"chat\"} 1774"));
  assert(has_line(out, "llmhost_request_seconds_sum{model=\"m1\",endpoint=\"chat\"} 7.1"));
  assert(has_line(out, "llmhost_request_seconds_count{model=\"m1\",endpoint=\"chat\"} 4"));
  // EWMA: первый замер 60/2=30, второй 120/4=30 → 30; float-печать 30.
  assert(has_line(out, "llmhost_generation_tps{model=\"m1\"} 30"));

  // HELP/TYPE на месте у всех семейств.
  assert(out.find("# TYPE llmhost_requests_total counter") != std::string::npos);
  assert(out.find("# TYPE llmhost_generation_tps gauge") != std::string::npos);
  // Модель с кавычкой в имени не рвёт формат.
  metrics::observe("we\"ird\nname", {"chat", "ok", 200, 1.0, 0, {}});
  const std::string out2 = metrics::render(nlohmann::json::object(),
                                           nlohmann::json::object());
  assert(has_line(out2, "llmhost_requests_total{model=\"we\\\"ird\\nname\",endpoint=\"chat\"} 1"));

  // CR и control-байты в метке тоже экранированы/заменены — поток
  // exposition не рвётся (одиночный CR некоторые парсеры считают переводом).
  metrics::observe("bad\rname\x01", {"chat", "ok", 200, 1.0, 0, {}});
  const std::string out_cr = metrics::render(nlohmann::json::object(),
                                             nlohmann::json::object());
  assert(out_cr.find('\r') == std::string::npos);
  assert(out_cr.find('\x01') == std::string::npos);
  assert(out_cr.find("bad\\rname_") != std::string::npos);

  // Кардинальность: клиентское «model» — произвольная строка, карта не должна
  // расти бесконечно (DoS памяти и числа строк метрик). Сверх потолка
  // неизвестные модели стекаются в «?», длинные ключи идут в «?» сразу.
  for (int i = 0; i < 6000; ++i)
    metrics::observe("flood" + std::to_string(i), {"chat", "ok", 200, 0.1, 0, {}});
  metrics::observe(std::string(200, 'x'), {"chat", "ok", 200, 0.1, 0, {}});
  const std::string out3 = metrics::render(nlohmann::json::object(),
                                           nlohmann::json::object());
  {
    std::set<std::string> models;
    for (size_t p = 0; (p = out3.find("model=\"", p)) != std::string::npos;) {
      p += 7;
      size_t e = out3.find('"', p);
      if (e == std::string::npos) break;
      models.insert(out3.substr(p, e - p));
      p = e;
    }
    assert(models.size() <= 4100); // потолок 4096 + бакет «?» (+ пара срезанных
                                   // экранированных меток от простого парсера)
    assert(models.count("?") == 1);
    assert(models.count(std::string(200, 'x')) == 0);
  }

  // ---- телеметрия движка: разбор /metrics ребёнка и рендер серий ---------
  {
    const std::string child =
        "# HELP llamacpp:tokens_predicted_total x\n"
        "# TYPE llamacpp:tokens_predicted_total counter\n"
        "llamacpp:tokens_predicted_total 4127\n"
        "llamacpp:prompt_tokens_seconds{foo=\"bar\"} 504.03\n"
        "llamacpp:predicted_tokens_seconds 40.54\n"
        "llamacpp:tokens_predicted_seconds_total 101.375\n"
        "llamacpp:spec_decode_num_accepted_tokens_total{position=\"0\"} 2983\n"
        "llamacpp:spec_decode_num_draft_tokens_total 3394\n"
        "garbage line\n"
        "llamacpp:broken 12x\n";
    auto s = metrics::parse_engine(child);
    // метки отбрасываются, битые значения не попадают
    assert(s["tokens_predicted_total"] == 4127.0);
    assert(s["prompt_tokens_seconds"] == 504.03);
    assert(s["predicted_tokens_seconds"] == 40.54);
    assert(s["tokens_predicted_seconds_total"] == 101.375);
    assert(s["spec_decode_num_accepted_tokens_total"] == 2983.0);
    assert(s["spec_decode_num_draft_tokens_total"] == 3394.0);
    assert(s.size() == 6);
    nlohmann::json engine = nlohmann::json::array();
    engine.push_back(nlohmann::json{{"model", "M"}, {"series", s}});
    const std::string out4 = metrics::render(nlohmann::json::object(),
                                             nlohmann::json::object(), engine);
    assert(has_line(out4, "llmhost_engine_decode_tps{model=\"M\"} 40.54"));
    assert(has_line(out4, "llmhost_engine_prompt_tps{model=\"M\"} 504.03"));
    assert(has_line(out4, "llmhost_engine_decode_tokens_total{model=\"M\"} 4127"));
    assert(has_line(out4, "llmhost_engine_decode_seconds_total{model=\"M\"} 101.375"));
    assert(has_line(out4, "llmhost_engine_spec_accepted_total{model=\"M\"} 2983"));
    assert(has_line(out4, "llmhost_engine_spec_draft_total{model=\"M\"} 3394"));
    // без движков — серии не рождаются
    const std::string out5 = metrics::render(nlohmann::json::object(),
                                             nlohmann::json::object());
    assert(out4.find("llmhost_engine_decode_tps") != std::string::npos);
    assert(out5.find("llmhost_engine_decode_tps{") == std::string::npos);
  }

  // ---- клиентский разбор /v0/metrics (fmt.h) ------------------------------
  {
    const std::string prom =
        "# HELP llmhost_requests_total x\n"
        "llmhost_requests_total{model=\"A\",endpoint=\"chat\"} 3\n"
        "llmhost_requests_total{model=\"A\",endpoint=\"completion\"} 5\n"
        "llmhost_prompt_tokens_total{model=\"A\"} 1475512\n"
        "llmhost_engine_decode_tokens_total{model=\"A\"} 4127\n"
        "llmhost_engine_decode_seconds_total{model=\"A\"} 101.375\n"
        "llmhost_engine_decode_tps{model=\"A\"} 40.54\n"
        "llmhost_engine_spec_accepted_total{model=\"A\"} 2983\n"
        "llmhost_engine_spec_draft_total{model=\"A\"} 3394\n"
        "llmhost_up 1\n"
        "llmhost_vram_bytes{card=\"0\",kind=\"free\"} 11953504256\n";
    auto rows = fmt::parse_metrics(prom);
    assert(rows.size() == 1);
    auto it = rows.find("A");
    assert(it != rows.end());
    // запросы суммируются по эндпоинтам, безметочные серии пропущены
    assert(it->second.reqs == 8);
    assert(it->second.prompt_tok == 1475512);
    assert(it->second.dec_tok == 4127.0);
    assert(it->second.dec_sec == 101.375);
    assert(it->second.dec_tps == 40.54);
    assert(it->second.spec_acc == 2983.0);
    assert(it->second.spec_draft == 3394.0);
    // мгновенная скорость из дельт + спарклайн
    double inst = 0;
    if (it->second.dec_sec > 0) inst = it->second.dec_tok / it->second.dec_sec;
    assert(inst > 40.0 && inst < 41.0);
    const std::string sp = fmt::sparkline({0.0, 1.0, 4.0, 8.0});
    assert(sp.size() == 10); // « »(1 байт) + три графемы по 3 байта UTF-8
    assert(sp != fmt::sparkline({}));
    assert(fmt::fmt_u64(4127.0) == "4127");
  }

  std::puts("metrics_test: ok");
  return 0;
}
