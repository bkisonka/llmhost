// fmt.h — человекочитаемые величины и общие карточки для lhctl (CLI, watch
// и полноэкранный интерфейс). Header-only: разделяются между единицами
// трансляции lhctl.cpp и tui.cpp.
#pragma once

#include "i18n.h"
#include "termui.h"
#include "version.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>
#include <vector>

namespace lh::fmt {

using lh::i18n::tr;

inline std::string human_bytes(uint64_t b) {
  char buf[32];
  if (b >= 1073741824ull)
    std::snprintf(buf, sizeof(buf), tr("%.1f ГБ"), b / 1073741824.0);
  else if (b >= 1048576ull)
    std::snprintf(buf, sizeof(buf), tr("%.0f МБ"), b / 1048576.0);
  else
    std::snprintf(buf, sizeof(buf), tr("%llu КБ"), (unsigned long long)(b / 1024));
  return buf;
}

inline std::string human_ctx(uint64_t ctx) {
  char buf[32];
  if (ctx >= 1024)
    std::snprintf(buf, sizeof(buf), tr("%lluК"), (unsigned long long)(ctx / 1024));
  else
    std::snprintf(buf, sizeof(buf), "%llu", (unsigned long long)ctx);
  return buf;
}

inline std::string human_count(uint64_t n) {
  char buf[32];
  if (n >= 1000000)
    std::snprintf(buf, sizeof(buf), tr("%.1fМ"), n / 1000000.0);
  else if (n >= 1000)
    std::snprintf(buf, sizeof(buf), tr("%.0fК"), n / 1000.0);
  else
    std::snprintf(buf, sizeof(buf), "%llu", (unsigned long long)n);
  return buf;
}

// Состояние модели одним цветным словом.
// «Модель в памяти» по записи /v0/models: router/harness дают state,
// классика — флаг loaded (state пуст). Канон одного предиката — раньше жил
// в пяти копиях tui.
inline bool state_loaded(const std::string& st, bool classic_loaded) {
  return st == "loaded" || (st.empty() && classic_loaded);
}

inline std::string state_cell(const std::string& st, bool failed, bool classic_loaded) {
  // Формулировки — как в меню трея Windows-версии.
  using namespace lh::ui;
  if (failed) return c::err(tr("(не загрузилась)"));
  if (st == "loaded" || (st.empty() && classic_loaded)) return c::ok(tr("(в памяти)"));
  if (st == "loading") return c::warn(tr("(грузится…)"));
  if (st == "error") return c::err(tr("(ошибка)"));
  if (st == "unloaded") return c::dim("—");
  if (st.empty()) return c::dim("—");
  return c::dim(st);
}

// Вложить таблицу в панель: рендер уже в ширину внутреннего поля рамки.
inline void panel_table(lh::ui::Panel& p, const lh::ui::Table& t) {
  std::string s = t.render(lh::ui::width() - 6);
  for (size_t pos = 0; pos < s.size();) {
    size_t nl = s.find('\n', pos);
    if (nl == std::string::npos) nl = s.size();
    std::string row = s.substr(pos, nl - pos);
    if (!row.empty()) p.line("  " + row);
    pos = nl + 1;
  }
}

// Карточка модели (show и интерфейс управления).
// Человекочитаемые пары из аргументов спавна ребёнка (/v0/status models[]
// .args): то, что планировщик решил, — в том виде, в каком поехало движку.
inline std::vector<std::pair<std::string, std::string>> slot_args_pairs(
    const nlohmann::json& args) {
  std::map<std::string, std::string> m;
  if (!args.is_array()) return {}; // не-массив в «args» — пусто, не type_error
  for (size_t i = 0; i + 1 < args.size(); ++i) {
    // get<std::string> на числе/null бросает type_error вне try — terminate
    // тула. Источник формально наш демон, но контракт JSON есть контракт.
    if (!args[i].is_string() || !args[i + 1].is_string()) continue;
    std::string k = args[i].get<std::string>();
    if (!k.empty() && k[0] == '-') m[k] = args[i + 1].get<std::string>();
  }
  const char* names[][2] = {
      {"-c", tr("контекст")}, {"-ngl", tr("слоёв на GPU")}, {"--n-cpu-moe", tr("блоков экспертов в RAM")},
      {"-ot", tr("тензоры вне GPU")}, {"-ctk", "KV K"}, {"-ctv", "KV V"},
      {"--model-draft", tr("черновик")}, {"--spec-type", tr("тип спекуляции")},
      {"--gpu-layers-draft", tr("слоёв черновика на GPU")}, {"--device-draft", tr("черновик на карте")},
      {"--device", tr("устройство")},
      {"-ts", tr("сплит карт")}, {"--mmproj", tr("vision-проекция")}, {"-ub", "ubatch"},
      {"-np", tr("слотов parallel")}, {"--main-gpu", tr("дом-карта")},
  };
  std::vector<std::pair<std::string, std::string>> out;
  for (auto& row : names) {
    auto it = m.find(row[0]);
    if (it != m.end()) out.push_back({row[1], it->second});
  }
  return out;
}

inline std::string render_show(const nlohmann::json& m) {
  using namespace lh::ui;
  Panel p(tr("Модель: ") + m.value("id", m.value("name", "?")));
  p.kv(tr("Имя"), m.value("name", "—"));
  p.kv(tr("Файл"), m.value("path", "—"));
  p.kv(tr("Размер"), human_bytes(m.value("size", (uint64_t)0)));
  p.kv(tr("Архитектура"), m.value("arch", "?") + tr("  ·  квант ") + m.value("quant", "?"));
  if (m.value("n_layers", 0)) p.kv(tr("Слоёв"), std::to_string(m.value("n_layers", 0)));
  if (m.value("n_ctx_train", (uint64_t)0))
    p.kv(tr("Контекст"), tr("до ") + human_ctx(m.value("n_ctx_train", (uint64_t)0)) + tr(" токенов"));
  std::string st = m.value("state", "");
  if (!st.empty() || m.value("loaded", false)) {
    std::string cell = state_cell(st, m.value("failed", false), m.value("loaded", false));
    int gpu_i = m.value("gpu", -1);
    if (gpu_i >= 0) cell += c::dim(tr("  ·  карта ") + std::to_string(gpu_i));
    p.kv(tr("Состояние"), cell);
  }
  auto fit = m.value("fit", nlohmann::json::object());
  if (!fit.is_null() && fit.value("ctx", (uint64_t)0)) {
    std::string plan = human_ctx(fit.value("ctx", (uint64_t)0));
    if (fit.value("max_ctx", (uint64_t)0))
      plan += c::dim(tr(" (потолок ") + human_ctx(fit.value("max_ctx", (uint64_t)0)) + ")");
    if (!fit.value("fits", false)) plan = c::warn(plan + tr(" — не влезает"));
    p.kv(tr("План VRAM"), human_bytes(fit.value("vram_needed", (uint64_t)0)) +
                          c::dim(" · ctx ") + plan);
  }
  if (int octx = m.value("ctx_override", 0))
    p.kv(tr("Ручной ctx"), c::accent(std::to_string(octx)) +
                           c::dim(tr("  ·  снять: lhctl ctx <модель> auto")));
  const std::string mo = m.value("mtp_mode", "");
  if (!mo.empty()) {
    const std::string ovr = m.value("mtp_override", "");
    p.kv("MTP", c::accent(mo) +
                    (ovr.empty() ? c::dim(tr("  ·  глобальный [general]"))
                                 : c::dim(tr("  ·  переопределение секции")) +
                                       c::dim(tr("  ·  сменить: M в карточке"))));
  }
  // Vision-режим карточки: mmproj — проекция, которая прицепится (пусто —
  // выключено или не найдено); переключение — V в карточке / lhctl vision.
  {
    const std::string vo = m.value("vision_mode", "auto");
    const std::string vm = m.value("mmproj", "");
    const bool ovr = !m.value("vision_override", "").empty();
    std::string cell;
    if (vo == "off")
      cell = c::accent("off") + c::dim(tr("  ·  без --mmproj  ·  включить: V (lhctl vision <модель> auto)"));
    else if (!vm.empty())
      cell = c::accent("auto") + c::dim("  ·  «" + vm.substr(vm.find_last_of("/\\") + 1) +
                                        tr("»  ·  выключить: V (lhctl vision <модель> off)"));
    else
      cell = c::accent("auto") + c::dim(tr("  ·  проекций рядом нет"));
    if (ovr) cell += c::dim(tr("  ·  переопределение секции"));
    p.kv("Vision", cell);
  }
  // Thinking-режим карточки: auto — движок сам детектит по шаблону;
  // off/on — форс --reasoning; переключение — T в карточке / lhctl thinking.
  {
    const std::string to = m.value("think_override", "");
    const std::string tm = m.value("think_mode", "auto");
    std::string cell;
    if (tm == "off")
      cell = c::accent("off") +
             c::dim(tr("  ·  без рассуждений  ·  вернуть: T (lhctl thinking <модель> auto)"));
    else if (tm == "on")
      cell = c::accent("on") +
             c::dim(tr("  ·  форс --reasoning on  ·  выключить: T (lhctl thinking <модель> off)"));
    else
      cell = c::accent("auto") +
             c::dim(tr("  ·  решает движок по шаблону  ·  выключить: T (lhctl thinking <модель> off)"));
    if (!to.empty()) cell += c::dim(tr("  ·  переопределение секции"));
    p.kv("Thinking", cell);
  }
  if (m.value("embedding", false)) p.kv(tr("Эмбеддер"), c::ok(tr("да (/v1/embeddings)")));
  return p.render();
}

// Панель движка (engine и интерфейс управления).
inline std::string render_engine(const nlohmann::json& j) {
  using namespace lh::ui;
  Panel p(tr("Движок llama.cpp"));
  p.kv(tr("Пин"), c::accent(j.value("release", "?")));
  p.kv(tr("Активная"), c::accent(j.value("active", "—")));
  if (!j.value("selected", "").empty())
    p.kv(tr("Выбрана"), c::accent(j.value("selected", "")) +
                        c::dim(tr("  ·  сбросить: lhctl engine use auto")));
  std::string st = j.value("state", "idle");
  if (st != "idle")
    p.kv(tr("Установка"), c::warn(st + " «" + j.value("build", "") + "»"));
  if (st == "downloading" || st == "extracting") {
    uint64_t done = j.value("done", (uint64_t)0), total = j.value("total", (uint64_t)0);
    double frac = total ? (double)done / (double)total : 0.0;
    p.kv(tr("Прогресс"), c::by_frac(frac, bar(frac, 20)) + "  " + human_bytes(done) +
                          c::dim(tr(" из ") + human_bytes(total)));
  }
  if (st == "error" && !j.value("error", "").empty())
    p.kv(tr("Ошибка"), c::err(j.value("error", "")));
  if (j.value("mismatch", false))
    p.line("  " + c::warn(
                   tr("⚠ пин и установленный движок расходятся — "
                      "«установлено» ниже показывает факт")));
  p.hr();
  Table t;
  t.col(tr("СБОРКА")).col(tr("РАЗМЕР"), Table::Right).col(tr("СТАТУС"));
  for (auto& b : j.value("builds", nlohmann::json::array())) {
    std::string status;
    if (b.value("installed", false)) {
      status = c::ok(tr("✓ установлена"));
      std::string rel = b.value("installed_release", "");
      if (!rel.empty())
        status += rel == j.value("release", "")
                      ? c::dim("  ·  " + rel)
                      : c::warn("  ·  " + rel + tr(" ≠ пин ") + j.value("release", ""));
      status += c::dim(b.value("pinned", false) ? tr("  ·  пиннута") : "");
    }
    else if (b.value("pinned", false))
      status = c::accent(tr("пиннута")) + c::dim(tr("  ·  не установлена"));
    else
      status = c::warn(tr("БЕЗ ПИНА"));
    if (b.value("id", "") == j.value("active", "") && !j.value("active", "").empty())
      status += c::dim(tr("  ← работает"));
    t.row({b.value("id", "?"), human_bytes(b.value("size", (uint64_t)0)), status});
  }
  panel_table(p, t);
  p.line("  " + c::dim(tr("установить: lhctl engine install <сборка>  ·  выбрать: lhctl engine use <сборка>")));
  p.line("  " + c::dim(tr("обновить пин: lhctl engine update")));
  return p.render();
}

// Строка прогресса скачивания: спиннер, имя, бар, проценты, скорость и ETA.
inline std::string download_line(const std::string& name, bool have_total, double frac,
                                 uint64_t done, uint64_t total, double speed_bps,
                                 const std::string& stage) {
  using namespace lh::ui;
  const int w = width();
  std::string stats;
  if (speed_bps > 1) {
    char sp[40];
    std::snprintf(sp, sizeof(sp), tr("%.1f МБ/с"), speed_bps / 1048576.0);
    stats = sp;
    if (have_total && frac < 1.0 && total > done) {
      double sec = (double)(total - done) / speed_bps;
      char eta[40];
      if (sec >= 60)
        std::snprintf(eta, sizeof(eta), tr("  ·  ~%lu мин %lu с"), (unsigned long)(sec / 60),
                      (unsigned long)sec % 60);
      else
        std::snprintf(eta, sizeof(eta), tr("  ·  ~%lu с"), (unsigned long)sec);
      stats += eta;
    }
  }
  if (have_total)
    stats = human_bytes(done) + c::dim("/" + human_bytes(total)) +
            (stats.empty() ? "" : "  ·  " + stats);
  int nw = std::min((int)dwidth(name), w * 2 / 5);
  int bar_w = w - 6 - nw - 6 - (int)dwidth(stats);
  if (bar_w < 8) bar_w = 8;
  std::string mid;
  if (have_total) {
    char pct[8];
    std::snprintf(pct, sizeof(pct), "%3d%%", (int)(frac * 100 + 0.5));
    mid = c::by_frac(1.0 - frac, bar(frac, bar_w)) + " " + pct;
  } else {
    mid = c::accent(stage.empty() ? "…" : stage);
  }
  return " " + std::string(lh::ui::spinner()) + " " +
         lh::ui::pad(lh::ui::trunc(name, (size_t)nw), (size_t)nw) + "  " + mid + "  " + stats;
}

inline std::string error_message(const std::string& body) {
  try {
    auto j = nlohmann::json::parse(body);
    if (j.contains("error") && j["error"].contains("message") &&
        j["error"]["message"].is_string())
      return j["error"]["message"].get<std::string>();
  } catch (...) {}
  return body;
}

// ---- Метрики /v0/metrics: разбор и спарклайн -------------------------------
// Общий для lhctl (metrics [-w]) и TUI (экран метрик): один разбор текста
// Prometheus демона в строки по-модельно.

struct MetricRow {
  uint64_t reqs = 0;        // llmhost_requests_total (сумма по эндпоинтам)
  uint64_t prompt_tok = 0;  // llmhost_prompt_tokens_total (из usage, с кэшем)
  uint64_t compl_tok = 0;   // llmhost_completion_tokens_total (из usage)
  double dec_tok = 0;       // llmhost_engine_decode_tokens_total
  double dec_sec = 0;       // llmhost_engine_decode_seconds_total
  double dec_tps = 0;       // llmhost_engine_decode_tps (пожизненная средняя)
  double prompt_tps = 0;    // llmhost_engine_prompt_tps
  double spec_acc = 0;      // llmhost_engine_spec_accepted_total
  double spec_draft = 0;    // llmhost_engine_spec_draft_total
};

// Один ключ серии → поле строки. Кардинальность меток константна, model
// берётся из метки; серии без модели и семьи без строк (состояния, очередь,
// VRAM) — мимо: строка таблицы рождается только у счётчиков трафика.
inline void metric_feed(const std::string& key, const std::string& model,
                        double v, std::map<std::string, MetricRow>& out) {
  if (model.empty()) return;
  MetricRow* r = nullptr;
  if (key == "llmhost_requests_total") { r = &out[model]; r->reqs += (uint64_t)v; }
  else if (key == "llmhost_prompt_tokens_total") { r = &out[model]; r->prompt_tok = (uint64_t)v; }
  else if (key == "llmhost_completion_tokens_total") { r = &out[model]; r->compl_tok = (uint64_t)v; }
  else if (key == "llmhost_engine_decode_tokens_total") { r = &out[model]; r->dec_tok = v; }
  else if (key == "llmhost_engine_decode_seconds_total") { r = &out[model]; r->dec_sec = v; }
  else if (key == "llmhost_engine_decode_tps") { r = &out[model]; r->dec_tps = v; }
  else if (key == "llmhost_engine_prompt_tps") { r = &out[model]; r->prompt_tps = v; }
  else if (key == "llmhost_engine_spec_accepted_total") { r = &out[model]; r->spec_acc = v; }
  else if (key == "llmhost_engine_spec_draft_total") { r = &out[model]; r->spec_draft = v; }
}

// Разбор текста /v0/metrics: строки «family{labels} value»; модель — из
// метки model="...". HELP/TYPE и чужие семейства пропускаются.
inline std::map<std::string, MetricRow> parse_metrics(const std::string& text) {
  std::map<std::string, MetricRow> out;
  for (size_t pos = 0; pos < text.size();) {
    size_t eol = text.find('\n', pos);
    if (eol == std::string::npos) eol = text.size();
    const std::string line = text.substr(pos, eol - pos);
    pos = eol + 1;
    if (line.rfind("llmhost_", 0) != 0) continue;
    const size_t sp = line.find(' ');
    if (sp == std::string::npos) continue;
    std::string key = line.substr(0, sp);
    const size_t brace = key.find('{');
    if (brace == std::string::npos) continue; // безметочные семейства не нужны
    std::string labels = key.substr(brace + 1, key.size() - brace - 2);
    key = key.substr(0, brace);
    // единственная интересная метка — model="..."; кардинальность меток
    // константна, кавычки внутри имени модельного id исключены sanitize'ом
    const size_t mk = labels.find("model=\"");
    if (mk == std::string::npos) continue;
    const size_t v0 = mk + 7;
    const size_t v1 = labels.find('"', v0);
    if (v1 == std::string::npos) continue;
    const std::string model = labels.substr(v0, v1 - v0);
    try {
      metric_feed(key, model, std::stod(line.substr(sp + 1)), out);
    } catch (...) {}
  }
  return out;
}

// Спарклайн из выборки мгновенных скоростей: нормировка на максимум,
// восемь градаций. Пустая выборка — пустая строка.
inline std::string sparkline(const std::vector<double>& v) {
  static const char* bars[] = {" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};
  if (v.empty()) return {};
  double mx = 0;
  for (double x : v) mx = x > mx ? x : mx;
  if (mx <= 0) return {};
  std::string s;
  for (double x : v) {
    int b = (int)(x / mx * 8.0 + 0.5);
    if (b > 8) b = 8;
    s += bars[b];
  }
  return s;
}

// Дробный счётчик Prometheus в целую строку («4127»).
inline std::string fmt_u64(double v) {
  char b[32];
  std::snprintf(b, sizeof b, "%.0f", v);
  return b;
}

}  // namespace lh::fmt
