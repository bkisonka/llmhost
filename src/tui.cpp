// tui — полноэкранный интерфейс управления llmhost в терминале.
//
// Референс — Windows-GUI этого же репозитория: каталог (catalogui.cpp) —
// поиск по HuggingFace + два списка (результаты → кванты выбранной модели)
// + «Скачать выбранное»/«Загрузить скачанное» + прогресс и статус-строка;
// чат (chatui.cpp) — лента диалога, ввод, «Отправить»→«Стоп»; формулировки
// состояний — как в меню трея («в памяти», «грузится…», «не загрузилась»).
//
// Запуск: `lhctl` без аргументов в терминале. Данные обновляются сами
// (~1 с), resize окна обрабатывается, терминал восстанавливается при
// выходе. Рисуется средствами termui; никакого ncurses.
//
// Клавиши: ↑↓/PgUp/PgDn — выбор, Enter/F3 — карточка, F4 — чат с моделью,
// F5 — загрузить, F7 — каталог (Enter у результата — кванты, F5 — скачать,
// F6 — загрузить скачанное), F6 — токены /v1, F8 — выгрузить, Ctrl-U — все,
// F9 — движок, i — изображения/художники (Enter — загрузить, p — скачать,
// Ctrl-D — удалить), Ctrl-D — удалить (с подтверждением), Ctrl-X — остановить
// скачивание (с любого экрана), F10/q — выход.
#include "tui.h"
#include "sse.h"

#include "i18n.h"
#include "fmt.h"
#include "termui.h"
#include "version.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <deque>
#include <set>
#include <tuple>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

using nlohmann::json;
using namespace lh::ui;
using namespace lh::fmt;

namespace lh::tui {
namespace {

#ifndef _WIN32
volatile sig_atomic_t g_winch = 0;
void on_winch(int) { g_winch = 1; }
#endif

enum class Scr { Main, Catalog, Chat, Tokens, TokenInput, Card, CtxInput, Engine, EngMenu,
                 EngInput, ConfirmRm, Help, ModelStat, Images, Metrics };

struct MRow {
  std::string id, path, st, quant;
  bool failed = false, classic_loaded = false, emb = false, fits = true;
  int gpu = -1;
  uint64_t size = 0, ctx = 0, mctx = 0, vneed = 0;
  // Спутники удаления: mmproj/MTP-модули и части шарда («ммproj-F16 (861 МБ)»).
  uint64_t comp_bytes = 0;
  std::vector<std::string> comp_desc;
};

// Вердикт каталога — те же слова, что в catalogui.cpp.
std::string verdict_text(const json& fit) {
  bool fits = fit.value("fits", false), partial = fit.value("partial", false),
       dense = fit.value("dense_partial", false);
  uint64_t need = fit.value("vram_needed", (uint64_t)0);
  if (!fit.value("gpu_known", false)) return c::warn(tr("? VRAM неизвестна"));
  if (fits) return c::ok(tr("✓ влезает, ") + human_bytes(need));
  if (partial || dense) return c::warn(tr("~ частично, ") + human_bytes(need));
  return c::err(tr("× не влезает, ") + human_bytes(need));
}

std::string rule(int w) {
  std::string r;
  for (int i = 0; i < w; ++i) r += "─";
  return c::dim(r);
}

// Курсорная строка: инверсия на всю ширину и без вложенных красок — любая
// внутренняя ESC[0m обрывает инверсию посреди строки (выглядело как «другой
// шрифт»), поэтому краски снимаются, строка добивается до края экрана.
std::string cursor_line(const std::string& line, int w) {
  return c::hl(pad(trunc(plain(line), (size_t)(w - 1)), (size_t)(w - 1)));
}

std::string keybar(std::initializer_list<std::pair<const char*, const char*>> keys) {
  std::string line;
  bool first = true;
  for (auto& [key, label] : keys) {
    if (!first) line += c::dim(" │ ");
    line += c::accent(key) + " " + label;
    first = false;
  }
  return line;
}

// Панель клавиш, вписывающаяся в ширину w: пункты вытесняются с хвоста,
// последний («F10 выход») не теряется никогда. dwidth понимает ANSI.
std::string keybar_fit(const std::vector<std::pair<std::string, std::string>>& keys,
                       int w) {
  if (keys.empty()) return {};
  auto cell = [&](const std::pair<std::string, std::string>& kv) {
    return kv.first + " " + kv.second;
  };
  auto total = [&](size_t n) { // ячейки при n видимых + последний всегда
    size_t cells = 0;
    for (size_t i = 0; i < n; ++i) {
      if (i) cells += 3;
      cells += dwidth(cell(keys[i]));
    }
    if (n < keys.size()) cells += 3 + dwidth(cell(keys.back()));
    return cells;
  };
  size_t fit = keys.size();
  while (fit > 1 && total(fit) > (size_t)(w < 4 ? 4 : w - 1)) --fit;
  std::string line;
  bool first = true;
  for (size_t i = 0; i < fit; ++i) {
    if (!first) line += c::dim(" │ ");
    line += c::accent(keys[i].first) + " " + keys[i].second;
    first = false;
  }
  if (fit < keys.size()) {
    if (!first) line += c::dim(" │ ");
    line += c::accent(keys.back().first) + " " + keys.back().second;
  }
  return line;
}

struct Tui {
  httplib::Client& cli;
  httplib::Headers hdrs;
  int port;

  Scr scr = Scr::Main;
  json status = json::object(), models = json::object(), engine = json::object();
  bool alive = false;
  std::vector<MRow> rows;
  int cursor = 0, scroll = 0;
  int card_scroll = 0, engine_scroll = 0, stat_scroll = 0;
  unsigned char kprev = 0;  // предыдущий байт Key::Char — окно UTF-8 для key_seq
  std::vector<std::string> card_lines, engine_lines, stat_lines;
  std::string card_mtp;  // эффективный mtp-mode карточки — для цикла по M
  std::string card_vision;  // эффективный vision-режим карточки — для цикла по V
  std::string card_think;   // эффективный thinking-режим карточки — для цикла по T
  // Метрики движка (экран m): последняя выборка /v0/metrics и история
  // (токены, секунды) для мгновенных скоростей — спарклайн по дельтам.
  std::map<std::string, lh::fmt::MetricRow> met_rows;
  std::map<std::string, std::deque<std::pair<double, double>>> met_hist;
  std::chrono::steady_clock::time_point met_at{};
  // Меню команд движка (F9 → F5): пункты из живого состояния, установка —
  // по одной через очередь (демон ставит в один поток, занят = 409).
  struct EngAction {
    std::string title, cmd, kind, arg;
  };
  std::vector<EngAction> eng_menu;
  int eng_mcur = 0, eng_scroll = 0;
  std::vector<std::string> eng_queue;
  bool eng_watch = false; // установка в полёте (даже когда очередь пуста)
  std::string eng_input;
  // «Актуальный релиз» для шапки меню движка: тянем один раз при открытии
  // меню (демон кэширует GitHub на 10 минут), в тике — чтобы не морозить UI.
  std::string eng_latest, eng_latest_note;
  bool eng_latest_ok = false, eng_latest_pending = false;
  std::string notice;
  bool notice_err = false;
  std::chrono::steady_clock::time_point notice_until;
  std::string rm_id, rm_path;
  uint64_t rm_size = 0;
  std::vector<std::string> rm_comp; // спутники удаления для подтверждения
  bool rm_is_image = false;  // ConfirmRm общий: llm-модель или художник (Ctrl-D
                             // в разделе «Изображения») — приговор разный
  bool rm_force_unload = false; // ConfirmRm: занятая модель — снять принудительно
                                // (F8/u по занятой дало 409; force обрывает все
                                // генерации модели — решение за пользователем)
  // Изображения (художники): курсор по каталогу + инлайн-ввод источника pull.
  int img_cur = 0, img_scroll = 0;
  bool img_typing = false;
  std::string img_input;
  std::string img_rm_id;
  // Ручной контекст (карточка → C): модель, текущий override, план, ввод.
  std::string ctx_path, ctx_id, ctx_input;
  int ctx_cur = 0, ctx_plan = 0;
  json pull = json::object();
  json image = json::object();  // image-слот (/v0/image)
  bool pull_was = false;
  int openai_port = 0;
  std::string openai_key;
  // Свежесть данных шапки/списка: refresh() молча держит прошлый снимок,
  // если демон не ответил, — без счётчика такое замирание неотличимо от
  // правды (инцидент F8 29.10: «выгружаю…» при списке «(в памяти)»).
  std::chrono::steady_clock::time_point fresh_at{};
  bool fresh_once = false;

  // Каталог (референс catalogui): поиск → результаты → кванты.
  std::string cat_query, cat_repo;
  json cat_results = json::array(), cat_detail = json::object();
  int cat_rcur = 0, cat_rscroll = 0, cat_qcur = 0, cat_qscroll = 0;
  int cat_stage = 0;  // 0 строка поиска, 1 результаты, 2 кванты
  std::string cat_status = tr("введите запрос и нажмите Enter");
  std::vector<int> cat_kinds;  // 0 обычный, 1 ★рекомендую, 2 mmproj/mtp, 3 мимо

  // Чат (референс chatui): лента + ввод; «Отправить» → «Стоп».
  std::vector<std::string> tr_text;
  std::vector<int> tr_style;  // 0 ответ, 1 мысли/пометки, 2 пользователь
  json chat_history = json::array();
  std::string chat_input, chat_model_id, chat_model_path, chat_pending;
  bool chat_generating = false, chat_abort = false;
  int chat_scroll = 0;

  // Токены /v1.
  json tokens_list = json::array();
  int tok_cur = 0, tok_scroll = 0;
  std::string token_name;

  json get(const char* path) {
    auto r = cli.Get(path, hdrs);
    if (!r || r->status != 200) return json();
    try {
      return json::parse(r->body);
    } catch (...) {
      return json();
    }
  }

  void say(std::string s, bool err = false) {
    notice = std::move(s);
    notice_err = err;
    notice_until = std::chrono::steady_clock::now() + std::chrono::seconds(6);
  }

  void refresh() {
    bool ok = true;
    auto st = get("/v0/status");
    alive = !st.is_null();
    if (alive) status = st; else ok = false;
    auto m = get("/v0/models");
    if (!m.is_null()) models = m; else ok = false;
    // Скачивание демона показываем независимо от того, кем запущено.
    // Image-слот: состояние + каталог (не влияет на «свежесть» данных чата).
    image = get("/v0/image");
    auto p = get("/v0/pull");
    if (!p.is_null()) {
      std::string s = p.value("state", "idle");
      if (s == "downloading" || s == "extracting" || s == "resolving") {
        pull = p;
        pull_was = true;
      } else if (pull_was) {
        if (s == "done") say("✓ скачано: " + p.value("repo", ""));
        else if (s == "stopped") say(tr("скачивание остановлено"));
        else if (s == "error") {
          std::string e = p.value("error", std::string());
          say(e.empty() ? tr("ошибка скачивания") : e, true);
        }
        pull = json::object();
        pull_was = false;
      }
    }
    // Полный успех — точка отсчёта свежести; иначе счётчик тикает дальше.
    if (ok) {
      fresh_at = std::chrono::steady_clock::now();
      fresh_once = true;
    }
    // Экран метрик живёт на своей выборке (дороговато для каждого тика).
    if (scr == Scr::Metrics) metrics_sample();
  }

  // Выборка /v0/metrics: разбор в строки по-модельно + история (токены,
  // секунды декода) — мгновенная скорость для спарклайна считается из дельт.
  // Текст Prometheus — не JSON, поэтому свой GET с разбором руками.
  void metrics_sample() {
    auto r = cli.Get("/v0/metrics", hdrs);
    if (!r || r->status != 200 || r->body.empty()) return;
    met_rows = lh::fmt::parse_metrics(r->body);
    for (auto& [id, row] : met_rows) {
      auto& h = met_hist[id];
      h.emplace_back(row.dec_tok, row.dec_sec);
      while (h.size() > 60) h.pop_front();
    }
    met_at = std::chrono::steady_clock::now();
  }

  void collect() {
    std::vector<MRow> next;
    for (auto& m : models.value("models", json::array())) {
      MRow r;
      r.path = m.value("path", "");
      r.id = m.value("id", "");
      if (r.id.empty()) {  // классический режим: стем файла
        size_t sl = r.path.find_last_of("/\\");
        r.id = sl == std::string::npos ? r.path : r.path.substr(sl + 1);
        if (r.id.size() > 5 && strcasecmp(r.id.c_str() + r.id.size() - 5, ".gguf") == 0)
          r.id.erase(r.id.size() - 5);
      }
      if (r.id.empty()) continue;
      r.st = m.value("state", "");
      r.failed = m.value("failed", false);
      r.classic_loaded = m.value("loaded", false);
      r.emb = m.value("embedding", false);
      r.gpu = m.value("gpu", -1);
      r.size = m.value("size", (uint64_t)0);
      r.quant = m.value("quant", "—");
      auto f = m.value("fit", json::object());
      r.ctx = f.value("ctx", (uint64_t)0);
      r.mctx = f.value("max_ctx", (uint64_t)0);
      r.vneed = f.value("vram_needed", (uint64_t)0);
      r.fits = f.value("fits", false);
      for (auto& c : m.value("companions", json::array())) {
        uint64_t sz = c.value("size", (uint64_t)0);
        r.comp_bytes += sz;
        std::string cp = c.value("path", "");
        size_t cs = cp.find_last_of("/\\");
        if (cs != std::string::npos) cp = cp.substr(cs + 1);
        std::string kind = c.value("kind", "");
        std::string label = kind == "mmproj"  ? "mmproj"
                            : kind == "mtp"   ? "MTP"
                            : kind == "shard" ? tr("шард")
                                              : ".part";
        r.comp_desc.push_back(label + " " + cp + " (" + human_bytes(sz) + ")");
      }
      next.push_back(std::move(r));
    }
    std::sort(next.begin(), next.end(), [](auto& a, auto& b) {
      auto rank = [](auto& r) {
        return state_loaded(r.st, r.classic_loaded) ? 0
               : r.st == "loading" ? 1 : 2;
      };
      return std::make_pair(rank(a), a.id) < std::make_pair(rank(b), b.id);
    });
    rows = std::move(next);
    if (cursor >= (int)rows.size()) cursor = (int)rows.size() - 1;
    if (cursor < 0) cursor = 0;
  }

  // ---- Действия -------------------------------------------------------------
  // Каждое действие рисует намерение ДО сетевого вызова и применяет
  // оптимистичное состояние строки ПОСЛЕ 2xx: refresh() умеет молча
  // залипать на старом снимке (таймауты до 30 с), и список не должен
  // противоречить сообщению, пока свежих данных нет.
  void action_load() {
    if (rows.empty()) { say(tr("реестр пуст — нечего загружать"), true); return; }
    auto& r = rows[cursor];
    if (state_loaded(r.st, r.classic_loaded)) {
      say("«" + r.id + tr("» уже в памяти"));
      return;
    }
    say(tr("загружаю «") + r.id + "»…");
    draw();
    auto res = cli.Post("/v0/load", hdrs, json{{"path", r.path}}.dump(), "application/json");
    if (!res) { say(tr("демон не отвечает"), true); draw(); return; }
    if (res->status >= 400) { say(error_message(res->body), true); draw(); return; }
    r.st = "loading";
    draw();
  }

  void action_unload(bool all) {
    if (all) {
      say(tr("выгружаю все модели…"));
      draw();
      auto res = cli.Post("/v0/unload", hdrs, "{}", "application/json");
      if (!res || res->status >= 400) { say(tr("выгрузка не прошла"), true); draw(); return; }
      // Занятые (генерация) демон оставляет — busy_kept в ответе.
      std::set<std::string> kept;
      auto rr = json::parse(res->body, nullptr, /*allow_exceptions=*/false);
      if (rr.is_object())
        for (auto& b : rr.value("busy_kept", json::array()))
          if (b.is_string()) kept.insert(b.get<std::string>());
      for (auto& r : rows) {
        if (!state_loaded(r.st, r.classic_loaded) && r.st != "loading") continue;
        if (kept.count(r.id)) continue;
        r.st = "unloaded";
        r.classic_loaded = false;
      }
      draw();
      return;
    }
    if (rows.empty()) { say(tr("реестр пуст"), true); return; }
    auto& r = rows[cursor];
    if (r.st != "loaded" && r.st != "loading" && !r.classic_loaded) {
      say("«" + r.id + tr("» не в памяти"));
      return;
    }
    say(tr("выгружаю «") + r.id + "»…");
    draw();
    auto res = cli.Post("/v0/unload", hdrs, json{{"model", r.id}}.dump(), "application/json");
    if (!res) { say(tr("демон не отвечает"), true); draw(); return; }
    if (res->status == 409) {
      // Занята генерацией — выключатель: предложить принудительную выгрузку
      // (Confirmed отдельным экраном, Esc отменяет). Зомби-генерация после
      // отмены клиента больше не живёт (ранний client_gone), но стоп-кран
      // нужен и для живых стримов — решение за пользователем.
      rm_force_unload = true;
      rm_id = r.id;
      scr = Scr::ConfirmRm;
      draw();
      return;
    }
    if (res->status >= 400) { say(error_message(res->body), true); draw(); return; }
    r.st = "unloaded";
    r.classic_loaded = false;
    draw();
  }

  void action_unload_force_confirm() {
    scr = Scr::Main;
    say(tr("снимаю принудительно «") + rm_id + "»…");
    draw();
    auto res = cli.Post("/v0/unload", hdrs,
                        json{{"model", rm_id}, {"force", true}}.dump(),
                        "application/json");
    rm_force_unload = false;
    if (!res) { say(tr("демон не отвечает"), true); return; }
    if (res->status >= 400) { say(error_message(res->body), true); return; }
    for (auto& r : rows)
      if (r.id == rm_id) { r.st = "unloaded"; r.classic_loaded = false; }
    say(tr("«") + rm_id + tr("» снята принудительно — генерации модели обрываются"));
    draw();
  }

  void action_ctx_apply() {
    std::string s = ctx_input;
    for (char& c : s)
      if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
    long long v = 0;
    bool remove = s == "auto" || s == "0";
    if (!remove) {
      char* end = nullptr;
      v = strtoll(s.c_str(), &end, 10);
      if (!end || *end != '\0' || v <= 0) { say(tr("нужны токены числом или auto"), true); return; }
      if (v < 256 || v > 2097152) { say(tr("диапазон 256…2097152 токенов"), true); return; }
    }
    auto res = cli.Post("/v0/models/ctx", hdrs,
                        json{{"path", ctx_path}, {"ctx", remove ? 0 : (int)v}}.dump(),
                        "application/json");
    if (!res) { say(tr("демон не отвечает"), true); return; }
    if (res->status >= 400) { say(error_message(res->body), true); return; }
    refresh();
    if (remove) {
      say(tr("ручной ctx снят — умная настройка решает сама"));
      return;
    }
    // Демон проверил, что при таком контексте проекция ещё помещается;
    // непустая note — «такой ctx исключает vision», показываем с алармом.
    std::string note;
    try {
      note = json::parse(res->body).value("note", "");
    } catch (...) {
    }
    if (!note.empty())
      say(note, true);
    else
      say("ctx-size = " + std::to_string(v) + tr(" — пресет пересобирается"));
  }

  void action_rm_begin() {
    if (rows.empty()) { say(tr("реестр пуст"), true); return; }
    auto& r = rows[cursor];
    rm_is_image = false;
    rm_id = r.id;
    rm_path = r.path;
    rm_size = r.size + r.comp_bytes;
    rm_comp = r.comp_desc;
    if (rm_comp.size() > 6) { // шард-группа может быть длинной — панель не раздувать
      rm_comp.resize(5);
      rm_comp.push_back(tr("… и ещё ") + std::to_string(r.comp_desc.size() - 5));
    }
    scr = Scr::ConfirmRm;
  }

  void action_rm_confirm() {
    auto res =
        cli.Delete("/v0/models", hdrs, json{{"path", rm_path}}.dump(), "application/json");
    scr = Scr::Main;
    if (!res) { say(tr("демон не отвечает"), true); return; }
    if (res->status >= 400) { say(error_message(res->body), true); return; }
    std::string msg = tr("удалено: ") + rm_id;
    try {
      auto r = json::parse(res->body);
      size_t n = r.value("files", json::array()).size();
      if (n > 1) msg += " + " + std::to_string(n - 1) + tr(" файлов");
      msg += tr(" — освобождено ") + human_bytes(r.value("bytes", (uint64_t)0));
    } catch (...) {}
    say(msg);
  }

  // ---- Изображения (художники) ---------------------------------------------
  // Тот же контракт действий, что у llama-списка: намерение до сети,
  // оптимистичное состояние после 2xx, refresh() принесёт правду.
  std::string img_active() const { return image.value("model", std::string()); }
  std::string img_state() const { return image.value("state", "unloaded"); }

  void img_load() {
    auto models = image.value("models", json::array());
    if (models.empty()) { say(tr("каталог image-моделей пуст — скачайте: p"), true); return; }
    if (img_cur >= (int)models.size()) img_cur = 0;
    std::string id = models[img_cur].value("id", "");
    say(tr("загружаю «") + id + "»…");
    draw();
    auto res = cli.Post("/v0/image/load", hdrs, json{{"model", id}}.dump(),
                        "application/json");
    if (!res) { say(tr("демон не отвечает"), true); draw(); return; }
    if (res->status >= 400) { say(error_message(res->body), true); draw(); return; }
    draw(); // 202: слот станет loading в ближайшем тике
  }

  void img_unload() {
    say(tr("останавливаю image-движок…"));
    draw();
    auto res = cli.Post("/v0/image/unload", hdrs, "{}", "application/json");
    if (!res || res->status >= 400) { say(tr("выгрузка не прошла"), true); draw(); return; }
    draw();
  }

  void img_rm_begin() {
    auto models = image.value("models", json::array());
    if (models.empty() || img_cur >= (int)models.size()) {
      say(tr("каталог пуст"), true);
      return;
    }
    img_rm_id = models[img_cur].value("id", "");
    rm_is_image = true;
    scr = Scr::ConfirmRm;
  }

  void img_rm_confirm() {
    std::string id = img_rm_id;
    rm_is_image = false;
    scr = Scr::Images;
    auto res =
        cli.Delete("/v0/image", hdrs, json{{"model", id}}.dump(), "application/json");
    if (!res) { say(tr("демон не отвечает"), true); return; }
    if (res->status >= 400) { say(error_message(res->body), true); return; }
    std::string msg = tr("удалено: ") + id;
    try {
      auto r = json::parse(res->body);
      msg += tr(" — освобождено ") + human_bytes(r.value("bytes", (uint64_t)0));
    } catch (...) {}
    say(msg);
  }

  void img_pull_start() {
    std::string name = img_input;
    img_input.clear();
    img_typing = false;
    if (name.empty()) return;
    auto res = cli.Post("/v0/pull", hdrs, json{{"repo", name}, {"kind", "image"}}.dump(),
                        "application/json");
    if (!res) { say(tr("демон не отвечает"), true); return; }
    if (res->status >= 400 && res->status != 409) {
      say(error_message(res->body), true);
      return;
    }
    say(res->status == 409 ? tr("уже качается другое — прогресс внизу")
                           : tr("скачивание «") + name + tr("» началось — прогресс внизу"));
    draw();
  }

  // ---- Каталог (catalogui) ----------------------------------------------------
  void cat_start_search() {
    if (cat_query.empty()) return;
    cat_status = tr("ищу…");
    auto r = cli.Get("/v0/catalog/search",
                     httplib::Params{{"q", cat_query}, {"limit", "30"}}, hdrs);
    if (!r || r->status != 200) {
      cat_status = tr("поиск не удался: ") + (r ? error_message(r->body) : tr("демон не отвечает"));
      return;
    }
    try {
      cat_results = json::parse(r->body).value("results", json::array());
    } catch (...) {
      cat_results = json::array();
    }
    cat_rcur = cat_rscroll = 0;
    cat_detail = json::object();
    cat_stage = cat_results.empty() ? 0 : 1;
    auto gpu = get("/v0/status").value("gpu", json::object());
    uint64_t vf = gpu.value("vram_free", (uint64_t)0),
             vt = gpu.value("vram_total", (uint64_t)0);
    cat_status = cat_results.empty()
                     ? tr("ничего не найдено")
                     : tr("найдено: ") + std::to_string(cat_results.size()) +
                           tr(" — вердикты по состоянию на момент поиска") +
                           (vf ? tr(" (свободно ") + human_bytes(vf) + tr(" из ") +
                                     human_bytes(vt) + ")"
                               : "") +
                           tr("; обновите поиск после выгрузки моделей");
  }

  void cat_fetch_detail() {
    if (cat_rcur >= (int)cat_results.size()) return;
    std::string repo = cat_results[cat_rcur].value("repo", "");
    cat_repo = repo;
    cat_status = tr("читаю ") + repo + "…";
    auto r = cli.Get("/v0/catalog/model", httplib::Params{{"repo", repo}}, hdrs);
    if (!r || r->status != 200) {
      cat_status = tr("не читается: ") + (r ? error_message(r->body) : tr("демон не отвечает"));
      return;
    }
    try {
      cat_detail = json::parse(r->body);
    } catch (...) {
      cat_status = tr("не читается: ответ не JSON");
      return;
    }
    cat_qcur = cat_qscroll = 0;
    // Автовыбор рекомендованного кванта — как ListView_SetItemState в catalogui.
    std::string recommended = cat_detail.value("recommended", "");
    auto files = cat_detail.value("files", json::array());
    cat_kinds.clear();
    for (auto& f : files) {
      bool rec = !recommended.empty() && f.value("filter", "") == recommended;
      bool aux = f.value("kind", "quant") != "quant";
      cat_kinds.push_back(aux ? 2 : rec ? 1 : 0);
      if (rec) cat_qcur = (int)cat_kinds.size() - 1;
    }
    bool precise = cat_detail.value("meta", json::object()).value("precise", false);
    std::string rec_note;
    for (auto& f : files)
      if (!recommended.empty() && f.value("filter", "") == recommended) {
        rec_note = tr(" · ★ Рекомендую: ") + f.value("quant", "?") +
                   tr(" — наибольший квант, влезающий целиком");
        break;
      }
    cat_status = std::string(precise ? tr("оценка по реальному заголовку GGUF")
                                     : tr("заголовок GGUF не дочитан — оценка грубая, по имени")) +
                 rec_note;
    cat_stage = 2;
  }

  void cat_pull() {
    auto files = cat_detail.value("files", json::array());
    if (cat_qcur >= (int)files.size()) return;
    std::string filter = files[cat_qcur].value("filter", "");
    auto res = cli.Post("/v0/pull", hdrs,
                        json{{"repo", cat_repo}, {"file", filter}}.dump(),
                        "application/json");
    if (!res) { cat_status = tr("демон не отвечает"); return; }
    if (res->status >= 400 && res->status != 409) {
      cat_status = error_message(res->body);
      return;
    }
    cat_status = res->status == 409 ? tr("уже качается другое — прогресс внизу")
                                    : tr("скачивание началось — прогресс внизу");
  }

  void cat_load_downloaded() {
    auto files = cat_detail.value("files", json::array());
    if (cat_qcur >= (int)files.size()) return;
    auto& f = files[cat_qcur];
    if (f.value("kind", "quant") == "mtp") {
      cat_status = tr("MTP-модуль отдельно не грузят — он предложится при загрузке основной модели");
      return;
    }
    if (!f.value("downloaded", false)) {
      cat_status = tr("файл ещё не скачан — сначала «Скачать выбранное»");
      return;
    }
    std::string p = f.value("path", "");
    size_t sl = p.find_last_of("/\\");
    std::string file = sl == std::string::npos ? p : p.substr(sl + 1);
    size_t dot = file.find_last_of('.');
    if (dot != std::string::npos) file = file.substr(0, dot);
    auto res = cli.Post("/v0/load", hdrs, json{{"name", file}}.dump(), "application/json");
    if (!res) { cat_status = tr("демон не отвечает"); return; }
    if (res->status >= 400) { cat_status = error_message(res->body); return; }
    cat_status = tr("загрузка модели началась — статус на главном экране");
  }

  // ---- Чат (chatui) -------------------------------------------------------------
  // chat_scroll — сколько строк ленты спрятано снизу. 0 значит «хвост»:
  // новый текст виден сразу. follow сбрасывает к хвосту (своё сообщение);
  // чужой допис во время чтения истории хвост не отнимает.
  void tr_add(const std::string& s, int style, bool follow = true) {
    tr_text.push_back(s);
    tr_style.push_back(style);
    // Долгая сессия не должна расти без предела: старые абзацы теряются,
    // контекст истории (chat_history) при этом сохраняется.
    if (tr_text.size() > 2000) {
      tr_text.erase(tr_text.begin(), tr_text.begin() + 1000);
      tr_style.erase(tr_style.begin(), tr_style.begin() + 1000);
      tr_text.insert(tr_text.begin(), tr("… (старая часть ленты обрезана)"));
      tr_style.insert(tr_style.begin(), 1);
    }
    if (follow) chat_scroll = 0;
  }

  bool chat_scroll_key(Key k) {
    switch (k) {
      case Key::Up: ++chat_scroll; return true;
      case Key::Down:
        if (chat_scroll > 0) --chat_scroll;
        return true;
      case Key::PgUp: chat_scroll += std::max(1, height() / 2); return true;
      case Key::PgDn:
        chat_scroll -= std::max(1, height() / 2);
        if (chat_scroll < 0) chat_scroll = 0;
        return true;
      case Key::Home: chat_scroll = 1 << 30; return true;
      case Key::End: chat_scroll = 0; return true;
      default: return false;
    }
  }

  bool chat_loaded() {
    for (auto& r : rows)
      if (r.id == chat_model_id)
        return state_loaded(r.st, r.classic_loaded);
    return false;
  }

  void open_chat() {
    if (rows.empty()) { say(tr("реестр пуст"), true); return; }
    auto& r = rows[cursor];
    auto loaded = [](const MRow& x) {
      return state_loaded(x.st, x.classic_loaded);
    };
    // F4 — чат с моделью В ПАМЯТИ: выбранная не загружена — переключаемся на
    // загруженную; сама грузится (поставленная «загружается» — ждём как
    // обычно). Заводить новую загрузку — только когда в памяти совсем пусто.
    const MRow* pick = &r;
    if (!loaded(r) && r.st != "loading") {
      for (auto& x : rows)
        if (loaded(x)) { pick = &x; break; }
    }
    if (pick != &r)
      say("«" + r.id + tr("» не в памяти — чат с «") + pick->id + "»");
    bool same = chat_model_id == pick->id;
    chat_model_id = pick->id;
    chat_model_path = pick->path;
    scr = Scr::Chat;
    if (!same) {
      // другая модель — прежняя лента и история к ней не относятся
      chat_history = json::array();
      if (!tr_text.empty()) tr_add("", 0);
      tr_add(tr("Диалог с «") + pick->id + "».", 1);
    }
    if (tr_text.empty()) {
      tr_add(tr("Диалог с «") + pick->id + "».", 1);
      tr_add(tr("Enter — отправить, Esc — стоп генерации / выход, /clear — новая сессия, "
             "/bye — закрыть чат."),
             1);
    }
    if (!loaded(*pick)) {
      auto res = cli.Post("/v0/load", hdrs, json{{"path", pick->path}}.dump(),
                          "application/json");
      if (res && res->status < 400) tr_add(tr("Модель загружается — подождите…"), 1);
    }
  }

  void chat_send() {
    std::string text = chat_input;
    chat_input.clear();
    while (!text.empty() && (text.back() == ' ' || text.back() == '\r')) text.pop_back();
    if (text.empty()) return;
    if (text == "/clear") {
      chat_history = json::array();
      tr_text.clear();
      tr_style.clear();
      tr_add(tr("Сессия сброшена."), 1);
      return;
    }
    if (text == "/bye") {
      scr = Scr::Main;
      return;
    }
    tr_add(text, 2);
    chat_pending = text;
    try_send_pending();
  }

  void try_send_pending() {
    if (chat_pending.empty() || chat_generating) return;
    if (!chat_loaded()) {
      if (!chat_model_path.empty())
        cli.Post("/v0/load", hdrs, json{{"path", chat_model_path}}.dump(),
                 "application/json");
      return;
    }
    std::string text = chat_pending;
    chat_pending.clear();
    chat_generate(text);
  }

  // Сам диалог: SSE-стрим в ленту; Esc во время генерации — стоп («Отправить»
  // в chatui превращается в «Стоп» — здесь то же самое).
  void chat_generate(const std::string& question) {
    json msgs = json::array();
    for (auto& m : chat_history) msgs.push_back(m);
    msgs.push_back({{"role", "user"}, {"content", question}});
    json body = {{"messages", msgs}, {"stream", true}};
    if (!chat_model_id.empty()) body["model"] = chat_model_id;

    httplib::Client oc("127.0.0.1", openai_port);
    oc.set_connection_timeout(std::chrono::seconds(5));
    oc.set_read_timeout(std::chrono::seconds(600));
    httplib::Request rq;
    rq.method = "POST";
    rq.path = "/v1/chat/completions";
    rq.body = body.dump();
    if (!openai_key.empty()) rq.headers.emplace("Authorization", "Bearer " + openai_key);
    rq.headers.emplace("Content-Type", "application/json");

    std::string answer, reason;
    bool done = false;
    lh::sse::Chat sse;
    auto last_draw = std::chrono::steady_clock::now();
    chat_generating = true;
    chat_abort = false;
    tr_add("", 0);
    const size_t placeholder = tr_text.size() - 1;
    size_t reason_row = static_cast<size_t>(-1);
    size_t answer_row = static_cast<size_t>(-1);
    // Первая пришедшая часть занимает заготовку. Вторая — новая строка
    // ниже, а не замена: мысли и ответ чередуются в стриме и раньше
    // затирали одну и ту же строку ленты.
    auto claim = [&](size_t& row, int style) {
      if (row != static_cast<size_t>(-1)) return;
      if (reason_row == static_cast<size_t>(-1) &&
          answer_row == static_cast<size_t>(-1)) {
        row = placeholder;
        if (row < tr_style.size()) tr_style[row] = style;
        return;
      }
      tr_add("", style, chat_scroll == 0);
      row = tr_text.size() - 1;
    };

    sse.on_reasoning = [&](const std::string& r2) {
      claim(reason_row, 1);
      reason += r2;
      if (reason_row < tr_text.size()) {
        tr_text[reason_row] = reason;
        tr_style[reason_row] = 1;
      }
    };
    sse.on_content = [&](const std::string& c) {
      claim(answer_row, 0);
      answer += c;
      if (answer_row < tr_text.size()) {
        tr_text[answer_row] = answer;
        tr_style[answer_row] = 0;
      }
    };

    rq.content_receiver = [&](const char* d, size_t n, uint64_t, uint64_t) -> bool {
      if (done) return true;
      bool typed = false;
      for (;;) {
        char c0 = 0;
        Key k = read_key(0, c0);
        if (k == Key::None) break;
        if (k == Key::Esc) {
          chat_abort = true;
          return false;  // httplib рвёт соединение — генерация остановлена
        }
        // Пока главный цикл сидит в этом стриме, прокрутка и набор живут здесь.
        if (chat_scroll_key(k)) { typed = true; continue; }
        if (k == Key::Backspace) {
          while (!chat_input.empty() && ((unsigned char)chat_input.back() & 0xC0) == 0x80)
            chat_input.pop_back();
          if (!chat_input.empty()) chat_input.pop_back();
          typed = true;
        } else if (k == Key::Char && (unsigned char)c0 >= 0x20) {
          chat_input += c0;
          typed = true;
        }
      }
      if (!sse.feed(d, n)) done = true; // [DONE]; соединение рвать не нужно
      auto now = std::chrono::steady_clock::now();
      if (typed ||
          std::chrono::duration_cast<std::chrono::milliseconds>(now - last_draw).count() >=
              60) {
        last_draw = now;
        draw();
      }
      return true;
    };
    auto res = oc.send(rq);
    chat_generating = false;

    if (chat_abort) {
      chat_abort = false;
      tr_add(tr("генерация остановлена"), 1, false);
      return;
    }
    if (!res || res->status != 200) {
      std::string e = res ? error_message(sse.raw_tail()) : tr("нет ответа");
      tr_add(tr("ошибка: ") + e, 1, false);
      return;
    }
    if (answer.empty() && reason.empty()) {
      tr_add(tr("пустой ответ модели"), 1, false);
      return;
    }
    chat_history.push_back({{"role", "user"}, {"content", question}});
    chat_history.push_back(
        {{"role", "assistant"}, {"content", answer.empty() ? reason : answer}});
    tr_add("", 0, false);
  }

  // ---- Токены /v1 ---------------------------------------------------------------
  void tokens_reload() {
    auto r = get("/v0/tokens");
    tokens_list = r.is_null() ? json::array() : r.value("tokens", json::array());
    if (tok_cur >= (int)tokens_list.size()) tok_cur = (int)tokens_list.size() - 1;
    if (tok_cur < 0) tok_cur = 0;
  }

  void token_add() {
    auto res = cli.Post("/v0/tokens", hdrs, json{{"name", token_name}}.dump(),
                        "application/json");
    if (!res) { say(tr("демон не отвечает"), true); return; }
    if (res->status >= 400) { say(error_message(res->body), true); return; }
    tokens_reload();
    say(tr("токен создан — значение в списке"));
  }

  void token_revoke() {
    if (tok_cur < 0 || tok_cur >= (int)tokens_list.size()) return;
    std::string name = tokens_list[tok_cur].value("name", "");
    auto res = cli.Delete("/v0/tokens", hdrs, json{{"name", name}}.dump(),
                          "application/json");
    if (!res) { say(tr("демон не отвечает"), true); return; }
    if (res->status >= 400) { say(error_message(res->body), true); return; }
    tokens_reload();
    say(tr("отозван: ") + name);
  }

  // ---- Экраны ---------------------------------------------------------------
  // Строка событий строится из живых данных: «грузится…» держится ровно
  // столько, сколько грузится; уведомление живёт 6 с после действия.
  std::string event_line() {
    for (auto& r : rows)
      if (r.st == "loading")
        return " " + std::string(spinner()) + " " + c::warn(tr("грузится «") + r.id + "»…");
    if (!pull.is_null() && !pull.empty()) {
      uint64_t done = pull.value("done", (uint64_t)0),
               total = pull.value("total", (uint64_t)0);
      std::string stt = pull.value("state", "");
      const char* stage = stt == "extracting" ? tr("распаковка tar.gz…")
                          : pull.value("stopping", false) ? tr("остановка…")
                                                          : tr("резолв…");
      return download_line(pull.value("file", ""), total > 0,
                           total ? (double)done / (double)total : 0.0, done, total,
                           pull.value("speed", 0.0), stage);
    }
    if (!notice.empty() && std::chrono::steady_clock::now() < notice_until)
      return " " + (notice_err ? c::err(notice) : c::ok(notice));
    return "";
  }

  std::string header_line() {
    std::string st = status.value("state", "");
    bool busy = status.value("busy", false);
    std::string t;
    if (!alive) t = c::err(tr("демон не отвечает на 127.0.0.1:") + std::to_string(port));
    else if (st == "loaded") t = c::ok(tr("работает"));
    else if (st == "loading") t = c::warn(tr("загрузка модели…"));
    else t = tr("простой");
    if (busy) t += c::accent(tr("  ·  идёт генерация"));
    int loaded = 0;
    for (auto& r : rows)
      if (state_loaded(r.st, r.classic_loaded)) ++loaded;
    if (alive) t += c::dim(tr("  ·  в памяти: ") + std::to_string(loaded));
    // Image-слот (sd-server): короткий чип — модель в VRAM видна сразу.
    if (alive && image.is_object()) {
      const std::string ist = image.value("state", "");
      if (ist == "loaded")
        t += c::ok(tr("  ·  изо: ") + image.value("model", ""));
      else if (ist == "loading")
        t += c::warn(tr("  ·  изо: загрузка…"));
      else if (ist == "error")
        t += c::err(tr("  ·  изо: ошибка"));
    }
    // Данные на экране — последний удачный снимок: если refresh не отвечает,
    // честно показываем возраст вместо молчаливой старины.
    if (alive && fresh_once) {
      const int age = (int)std::chrono::duration_cast<std::chrono::seconds>(
                          std::chrono::steady_clock::now() - fresh_at).count();
      if (age >= 5)
        t += c::warn(age >= 120 ? tr("  ·  данные от ") + std::to_string(age / 60) +
                                      tr(" мин назад")
                                : tr("  ·  данные от ") + std::to_string(age) +
                                      tr(" с назад"));
    }
    // Шапка — плоской композицией: инверсия с вложенными красками обрывалась
    // первым же внутренним ESC[0m посреди строки (выглядело как «другой шрифт»).
    return " " + c::accent("llmhost " + status.value("version", LH_VERSION)) +
           "  ·  " + t +
           c::dim(openai_port ? "  ·  API: http://127.0.0.1:" +
                                    std::to_string(openai_port) + "/v1"
                              : "");
  }

  std::vector<std::string> draw_main() {
    const int W = width(), H = height();
    std::vector<std::string> out;
    out.push_back(header_line());

    const int state_w = 17, card_w = 5, size_w = 9, quant_w = 8, vram_w = 9;
    int id_w = W - state_w - card_w - size_w - quant_w - vram_w - 6 * 3 - 1;
    for (auto& r : rows) id_w = std::max(id_w, std::min((int)dwidth(r.id), W * 2 / 5));
    if (id_w < 8) id_w = 8;
    out.push_back(" " +
                  c::dim(pad(tr("МОДЕЛЬ"), id_w) + " │ " + pad(tr("СОСТОЯНИЕ"), state_w) +
                         " │ " + padl(tr("КАРТА"), card_w) + " │ " +
                         padl(tr("РАЗМЕР"), size_w) + " │ " + pad(tr("КВАНТ"), quant_w) +
                         " │ " + padl("VRAM", vram_w)));
    out.push_back(" " + c::dim(horiz(id_w + 2) + "┼" + horiz(state_w + 2) + "┼" +
                               horiz(card_w + 2) + "┼" + horiz(size_w + 2) + "┼" +
                               horiz(quant_w + 2) + "┼" + horiz(vram_w + 2)));

    // Аппаратный блок всегда виден: строки карт + RAM/бэкенд.
    int hw = alive ? (int)status.value("gpus", json::array()).size() + 2 : 0;
    int vis = H - (int)out.size() - hw - 3;
    if (vis < 1) vis = 1;
    if (cursor < scroll) scroll = cursor;
    if (cursor >= scroll + vis) scroll = cursor - vis + 1;
    if (scroll < 0) scroll = 0;
    if (rows.empty())
      out.push_back(" " + c::dim(tr("реестр пуст — F7 открывает каталог моделей")));
    for (int i = scroll; i < (int)rows.size() && i < scroll + vis; ++i) {
      auto& r = rows[i];
      std::string vram = "—";
      if (r.vneed) vram = r.fits ? human_bytes(r.vneed) : c::warn(human_bytes(r.vneed));
      std::string line = " " + pad(trunc(r.id, (size_t)id_w), (size_t)id_w) + " │ " +
                         pad(state_cell(r.st, r.failed, r.classic_loaded), state_w) + " │ " +
                         padl(r.gpu >= 0 ? std::to_string(r.gpu) : "—", card_w) + " │ " +
                         padl(human_bytes(r.size), size_w) + " │ " + pad(r.quant, quant_w) +
                         " │ " + padl(vram, vram_w);
      if (r.emb) line += c::dim(tr("  эмбеддер"));
      line = trunc(line, W - 1);
      out.push_back(i == cursor ? cursor_line(line, W) : line);
    }
    if (scroll + vis < (int)rows.size())
      out.push_back(c::dim(tr("   ↓ ещё ") + std::to_string((int)rows.size() - scroll - vis)));

    // Что и где загружено, сколько памяти (слова трея Windows-версии).
    if (alive) {
      auto gpus = status.value("gpus", json::array());
      size_t nw = 0;
      for (auto& g : gpus) nw = std::max(nw, dwidth(g.value("name", "?")));
      nw = std::min(nw, (size_t)W / 2);
      int bw = W - 10 - (int)nw - 17;
      if (bw > 30) bw = 30;
      if (bw < 8) bw = 8;
      for (auto& g : gpus) {
        uint64_t total = g.value("vram_total", (uint64_t)0),
                 fr = g.value("vram_free", (uint64_t)0);
        uint64_t used = total > fr ? total - fr : 0;
        double frac = total ? (double)used / (double)total : 0.0;
        out.push_back(" " + pad(std::to_string(g.value("index", 0)), 2) + " " +
                      c::dim(pad(trunc(g.value("name", "?"), nw), nw)) + "  " +
                      c::by_frac(frac, bar(frac, bw)) + "  " +
                      padl(human_bytes(used) + c::dim("/" + human_bytes(total)), 15));
      }
      // Строка image-движка: что загружено и куда смотрит.
      if (image.is_object() && !image.value("models", json::array()).empty()) {
        const std::string ist = image.value("state", "");
        std::string isum = tr("изо sd-server: ");
        if (ist == "loaded")
          isum += image.value("model", "") + c::dim(tr(" (порт ") +
                                                  std::to_string(image.value("port", 0)) + ")");
        else if (ist == "loading")
          isum += tr("загрузка…");
        else if (ist == "error")
          isum += c::err(image.value("error", tr("ошибка")));
        else
          isum += c::dim(std::string(tr("простой (автостарт: карта ")) +
                         std::to_string(image.value("gpu", 0)) + ")");
        isum = trunc(isum, (size_t)W - 2);
        out.push_back(" " + (ist == "error" ? isum : c::dim(isum)));
      }
      auto gpu = status.value("gpu", json::object());
      uint64_t ram_t = gpu.value("ram_total", (uint64_t)0),
               ram_a = gpu.value("ram_avail", (uint64_t)0);
      std::string tail;
      if (ram_t) {
        uint64_t used = ram_t > ram_a ? ram_t - ram_a : 0;
        tail = "RAM " + human_bytes(used) + "/" + human_bytes(ram_t);
      }
      std::string cc = gpu.value("cc", "");
      if (gpu.value("nvidia", false))
        tail += std::string(tail.empty() ? "" : "  ·  ") + "CUDA" +
                (cc.empty() ? "" : " cc " + cc);
      // gpu.build — «что может железо»; движок — какая сборка реально едет.
      auto eng = status.value("engine", json::object());
      std::string ebuild = eng.value("build", "");
      if (!ebuild.empty()) {
        tail += tr("  ·  движок ") + c::accent(ebuild);
        if (!eng.value("selected", "").empty())
          tail += c::dim("*"); // выбрана вручную, не по железу
      } else if (!gpu.value("build", "").empty())
        tail += tr("  ·  движок ") + gpu.value("build", "");
      if (!tail.empty()) out.push_back(" " + c::dim(tail));
    }

    out.push_back(event_line());
    // F-клавиши по возрастанию, Ctrl-хвостом: keybar_fit вытесняет с конца
    // (кроме «выход»), так что на узких терминалах первыми теряются
    // Ctrl-комбинации, а не F6/F7.
    out.push_back(keybar_fit({{"F1", tr("помощь")}, {"F2", tr("статус")},
                                    {"F3", tr("карточка")},
                                    {"F4", tr("чат")}, {"F5", tr("загрузить")},
                                    {"F6", tr("токены")}, {"F7", tr("каталог")},
                                    {"F8", tr("выгрузить")}, {"F9", tr("движок")},
                                    {"i", tr("изо")}, {"m", tr("метрики")},
                                    {"Ctrl-D", tr("удалить")}, {"Ctrl-U", tr("все")},
                                    {"Ctrl-X", tr("стоп")},
                                    {"F10", tr("выход")}},
                                   W));
    return out;
  }

  // Каталог: раскладка окна catalogui — поиск сверху, результаты, кванты
  // выбранной модели, статус-строка внизу.
  std::vector<std::string> draw_catalog() {
    const int W = width(), H = height();
    std::vector<std::string> out;
    out.push_back(c::hl(tr(" llmhost — каталог моделей ")));
    out.push_back(" " + c::dim(tr("Поиск по HuggingFace (GGUF): ")) + c::accent(cat_query) +
                  (cat_stage == 0 ? c::accent("█") : ""));

    int body = H - (int)out.size() - 4;  // статус + события + keybar + запас
    int rh = body * 45 / 100, qh = body - rh - 2;
    if (rh < 3) rh = 3;
    if (qh < 3) qh = 3;

    out.push_back(" " +
                  (cat_stage == 1 ? c::accent(tr("РЕЗУЛЬТАТЫ"))
                                  : c::dim(tr("РЕЗУЛЬТАТЫ"))) +
                  " " + c::dim(tr("(↑↓, Enter/Tab — к квантам)")));
    std::vector<std::string> rrows;
    for (auto& r : cat_results) {
      auto fit = r.value("fit", json::object());
      std::string v =
          !fit.value("known", false) ? c::warn(tr("? размер неизвестен")) : verdict_text(fit);
      double p = fit.value("params", 0.0);
      std::string params = "—";
      char b[16];
      if (p >= 1e9) { std::snprintf(b, 16, "%.1fB", p / 1e9); params = b; }
      else if (p > 0) { std::snprintf(b, 16, "%.0fM", p / 1e6); params = b; }
      rrows.push_back(" " +
                      pad(trunc(r.value("repo", "?"), (size_t)(W * 3 / 5)),
                          (size_t)(W * 3 / 5)) +
                      "  " + c::dim(pad(params, 6)) + "  " +
                      padl(human_count(r.value("downloads", (uint64_t)0)), 6) + "  " + v);
    }
    if (rrows.empty()) rrows.push_back(" " + c::dim(tr("(пока пусто)")));
    {
      if (cat_rcur >= (int)rrows.size()) cat_rcur = (int)rrows.size() - 1;
      if (cat_rcur < 0) cat_rcur = 0;
      if (cat_rcur < cat_rscroll) cat_rscroll = cat_rcur;
      if (cat_rcur >= cat_rscroll + rh) cat_rscroll = cat_rcur - rh + 1;
      if (cat_rscroll < 0) cat_rscroll = 0;
      for (int i = cat_rscroll; i < (int)rrows.size() && i < cat_rscroll + rh; ++i)
        out.push_back(i == cat_rcur && cat_stage == 1 ? cursor_line(rrows[i], W) : rrows[i]);
    }

    out.push_back(" " +
                  (cat_stage == 2
                       ? c::accent(tr("КВАНТЫ «") + cat_repo + "»")
                       : c::dim(tr("КВАНТЫ ") + (cat_repo.empty()
                                                 ? tr("(выберите результат)")
                                                 : "«" + cat_repo + "»"))) +
                  " " + c::dim(tr("(F5 — скачать выбранное, F6 — загрузить скачанное)")));
    std::vector<std::string> qrows;
    cat_kinds.clear();
    if (cat_detail.contains("files")) {
      std::string recommended = cat_detail.value("recommended", "");
      for (auto& f : cat_detail.value("files", json::array())) {
        bool is_mmproj = f.value("kind", "quant") == "mmproj";
        bool is_mtp = f.value("kind", "quant") == "mtp";
        std::string quant = f.value("quant", "?");
        if (f.value("parts", 1) > 1) quant += " ×" + std::to_string(f.value("parts", 1));
        bool is_rec = !recommended.empty() && f.value("filter", "") == recommended;
        if (is_rec) quant = "★ " + quant;
        auto fj = f.value("fit", json::object());
        uint32_t mctx = fj.value("max_ctx", 0);
        std::string status;
        if (is_mmproj) {
          status = f.value("downloaded", false)
                       ? tr("скачано ✓ — нужна для картинок (вместе с моделью)")
                       : tr("vision-проекция — нужна для картинок (качается вместе с моделью)");
        } else if (is_mtp) {
          status = f.value("downloaded", false)
                       ? tr("скачано ✓ — включится при загрузке модели")
                       : tr("MTP-модуль — ускоряет генерацию этой модели (~+25%)");
        } else {
          bool moe_cpu = fj.value("moe_cpu", false),
               dense = fj.value("dense_partial", false);
          bool downloaded = f.value("downloaded", false);
          if (downloaded) status = tr("скачано ✓");
          else if (moe_cpu) status = tr("влезет (MoE: эксперты на CPU)");
          else if (dense)
            status = tr("влезет частично — ") + std::to_string(fj.value("ngl", 0)) + tr(" из ") +
                     std::to_string(fj.value("n_layers", 0)) +
                     tr(" слоёв на GPU, хвост в RAM (медленнее)");
          else if (fj.value("fits", false)) status = tr("можно качать");
          else if (fj.value("partial", false)) status = tr("частично — меньше контекст");
          else status = tr("мимо");
        }
        int kind = (is_mmproj || is_mtp) ? 2
                   : is_rec             ? 1
                   : (!fj.value("fits", false) && !fj.value("partial", false) &&
                      !fj.value("dense_partial", false) && !f.value("downloaded", false))
                         ? 3
                         : 0;
        cat_kinds.push_back(kind);
        std::string colored_quant = kind == 1 ? c::ok(quant)
                                    : kind == 2 ? c::dim(quant)
                                    : kind == 3 ? c::err(quant)
                                                : quant;
        qrows.push_back(" " + pad(colored_quant, 18) + "  " +
                        padl(human_bytes(f.value("size", (uint64_t)0)), 9) + "  " +
                        pad((is_mmproj || is_mtp) ? "—" : verdict_text(fj), 26) + "  " +
                        padl((is_mmproj || is_mtp)
                                 ? "—"
                                 : (mctx ? tr("до ") + std::to_string(mctx) + tr(" ток.") : "—"),
                             10) +
                        "  " + c::dim(status));
      }
    }
    if (qrows.empty())
      qrows.push_back(" " + c::dim(cat_repo.empty()
                                       ? tr("(выберите результат — Enter или Tab)")
                                       : tr("(читаю…)")));
    {
      if (cat_qcur >= (int)qrows.size()) cat_qcur = (int)qrows.size() - 1;
      if (cat_qcur < 0) cat_qcur = 0;
      if (cat_qcur < cat_qscroll) cat_qscroll = cat_qcur;
      if (cat_qcur >= cat_qscroll + qh) cat_qscroll = cat_qcur - qh + 1;
      if (cat_qscroll < 0) cat_qscroll = 0;
      for (int i = cat_qscroll; i < (int)qrows.size() && i < cat_qscroll + qh; ++i)
        out.push_back(i == cat_qcur && cat_stage == 2 ? cursor_line(qrows[i], W) : qrows[i]);
    }

    out.push_back(" " + c::accent(tr("статус: ")) + trunc(cat_status, (size_t)(W - 10)));
    out.push_back(event_line());
    out.push_back(keybar(
        {{"Enter", cat_stage == 0 ? tr("искать") : cat_stage == 1 ? tr("кванты") : tr("скачать")},
         {"Tab", tr("списки")}, {"F5", tr("скачать")}, {"F6", tr("загрузить")}, {"Esc", tr("назад")}}));
    return out;
  }

  // Чат: лента + ввод, как в chatui («Отправить» → «Стоп»).
  std::vector<std::string> draw_chat() {
    const int W = width(), H = height();
    std::vector<std::string> all;
    for (size_t i = 0; i < tr_text.size(); ++i) {
      int style = i < tr_style.size() ? tr_style[i] : 0;
      // « Вы> » — 5 ячеек, обычный отступ — 2. Режем текст так, чтобы
      // вместе с отступом ряд не вылезал за ширину и не переносился терминалом.
      int indent = style == 2 ? 5 : 2;
      int wrap_w = W - indent;
      if (wrap_w < 1) wrap_w = 1;
      for (auto& piece : hard_wrap(tr_text[i], (size_t)wrap_w)) {
        if (style == 1) all.push_back("  " + c::dim(piece));
        else if (style == 2) all.push_back(" " + c::accent(tr("Вы>")) + " " + piece);
        else all.push_back("  " + piece);
      }
    }
    int body = H - 3; // шапка, ввод, клавиши
    if (body < 1) body = 1;
    auto window = [&](int rows) {
      int max_scroll = (int)all.size() > rows ? (int)all.size() - rows : 0;
      int scroll = chat_scroll;
      if (scroll > max_scroll) scroll = max_scroll;
      if (scroll < 0) scroll = 0;
      int from = (int)all.size() > rows ? (int)all.size() - rows - scroll : 0;
      if (from < 0) from = 0;
      return std::tuple<int, int>{from, scroll};
    };
    auto [from, scroll] = window(body);
    bool hint = from > 0 && body > 1;
    if (hint) {
      --body;
      std::tie(from, scroll) = window(body);
      hint = from > 0;
    }
    chat_scroll = scroll;
    std::vector<std::string> out;
    out.push_back(c::hl(tr(" llmhost — чат: ") +
                     (chat_model_id.empty() ? tr("модель по умолчанию") : chat_model_id) +
                     (chat_generating ? tr("  ·  генерирую… (Esc — стоп)") : "") + " "));
    if (hint) out.push_back(c::dim(tr("  ↑ ещё ") + std::to_string(from) + tr(" — ↑ или PgUp")));
    for (int i = from; i < from + body && i < (int)all.size(); ++i) out.push_back(all[i]);
    while ((int)out.size() < H - 2) out.push_back("");
    out.push_back(" " + c::accent(tr("Вы>")) + " " + chat_input +
                  (chat_generating ? c::warn(tr("  …Esc — стоп")) : c::accent("█")));
    out.push_back(keybar({{"Enter", chat_generating ? "…" : tr("отправить")},
                             {"Esc", chat_generating ? tr("стоп") : tr("назад")},
                             {"/clear", tr("новая сессия")}, {"/bye", tr("закрыть чат")},
                             {"↑↓", tr("лента")}}));
    return out;
  }

  // Экран метрик: таблица по-модельно + спарклайн мгновенной скорости
  // декода (дельты токенов/секунд по истории выборок). Модель без телеметрии
  // движка (эмбеддер, только что проснувшийся ребёнок) — прочерки.
  std::vector<std::string> draw_metrics() {
    const int W = width(), H = height();
    std::vector<std::string> out;
    out.push_back(c::hl(tr(" llmhost — метрики движка ")));
    out.push_back(" " + c::dim(tr("источник: /metrics детей через демон; обновление "
                                  "каждые 2 с; промпт прокси — с кэшированными, "
                                  "движок — без")));
    out.push_back("");
    out.push_back(" " + c::dim(pad(tr("модель"), 36) + "  " +
                               pad(tr("декод тк/с"), 10) + " " +
                               pad(tr("префилл"), 9) + " " +
                               pad(tr("приём"), 6) + " " +
                               pad(tr("декод ткн"), 9) + " " +
                               pad(tr("промпт"), 10) + " " + tr("запр.")));
    for (auto& [id, r] : met_rows) {
      char b1[32], b2[32], b3[32];
      std::snprintf(b1, sizeof b1, "%.1f", r.dec_tps);
      std::snprintf(b2, sizeof b2, "%.1f", r.prompt_tps);
      std::snprintf(b3, sizeof b3, "%.2f",
                    r.spec_draft > 0 ? r.spec_acc / r.spec_draft : 0.0);
      out.push_back(" " + pad(trunc(id, 36), 36) + "  " +
                    pad(r.dec_tps > 0 ? b1 : "—", 10) + " " +
                    pad(r.prompt_tps > 0 ? b2 : "—", 9) + " " +
                    pad(r.spec_draft > 0 ? b3 : "—", 6) + " " +
                    pad(r.dec_tok > 0 ? fmt_u64(r.dec_tok) : "—", 9) + " " +
                    pad(r.prompt_tok > 0 ? fmt_u64((double)r.prompt_tok) : "—", 10) +
                    " " + (r.reqs > 0 ? fmt_u64((double)r.reqs) : "—"));
      // Спарклайн мгновенной скорости: дельты по истории выборок.
      auto h = met_hist.find(id);
      if (h != met_hist.end() && h->second.size() >= 2) {
        std::vector<double> inst;
        for (size_t i = 1; i < h->second.size(); ++i) {
          const double dt = h->second[i].first - h->second[i - 1].first;
          const double ds = h->second[i].second - h->second[i - 1].second;
          if (ds >= 0.5) inst.push_back(dt / ds);
        }
        if (!inst.empty()) {
          char b4[32];
          std::snprintf(b4, sizeof b4, "%.1f", inst.back());
          out.push_back("   " + c::dim(sparkline(inst)) + " " +
                        c::dim(std::string(tr("сейчас ")) + b4 + " тк/с"));
        }
      }
    }
    if (met_rows.empty())
      out.push_back(" " + c::dim(tr("(нет моделей с трафиком — счётчики пусты)")));
    while ((int)out.size() < H - 2) out.push_back("");
    out.push_back(event_line());
    out.push_back(keybar({{"Esc", tr("назад")}, {"m", tr("обновить")},
                                {"F10", tr("выход")}}));
    (void)W;
    return out;
  }

  std::vector<std::string> draw_tokens() {
    const int W = width(), H = height();
    std::vector<std::string> out;
    out.push_back(c::hl(tr(" llmhost — токены доступа к /v1 ")));
    out.push_back(" " + c::dim(tr("Именованные ключи OpenAI-порта (api_tokens.json); пока "
                               "они есть, /v1 отвечает только по ним.")));
    std::vector<std::string> all;
    for (auto& t : tokens_list) {
      all.push_back(pad(trunc(t.value("name", "?"), 20), 20) + "  " +
                    c::dim(t.value("token", "?")) + "  " +
                    c::dim(tr("создан ") + t.value("created_at", "—")) +
                    (t.value("last_used", "").empty()
                         ? ""
                         : c::dim(tr("  ·  использован ") + t.value("last_used", ""))));
    }
    if (all.empty()) all.push_back(" " + c::dim(tr("(токенов нет — Enter создаёт первый)")));
    int vis = H - (int)out.size() - 3;
    if (vis < 1) vis = 1;
    if (tok_cur < tok_scroll) tok_scroll = tok_cur;
    if (tok_cur >= tok_scroll + vis) tok_scroll = tok_cur - vis + 1;
    if (tok_scroll < 0) tok_scroll = 0;
    for (int i = tok_scroll; i < (int)all.size() && i < tok_scroll + vis; ++i)
      out.push_back(i == tok_cur ? cursor_line(all[i], W) : all[i]);
    while ((int)out.size() < H - 2) out.push_back("");
    out.push_back(event_line());
    out.push_back(
        keybar({{"Enter", tr("создать")}, {"Del", tr("отозвать")}, {"Esc", tr("назад")}}));
    (void)W;
    return out;
  }

  std::vector<std::string> draw_token_input() {
    std::vector<std::string> out;
    out.push_back(c::hl(tr(" llmhost — новый токен /v1 ")));
    out.push_back("");
    out.push_back(tr(" Имя (буквы/цифры/-_. ): ") + token_name + c::accent("█"));
    out.push_back(" " + c::dim(tr("Значение сгенерируется само и показывается в списке —")));
    out.push_back(" " + c::dim(tr("клиенты передают его как Bearer-ключ OpenAI-порта.")));
    out.push_back(keybar({{"Enter", tr("создать")}, {"Esc", tr("отмена")}}));
    return out;
  }

  // Раздел «Изображения»: каталог художников (image::scan), статус слота,
  // загрузка/выгрузка/удаление и инлайн-ввод источника pull. Прогресс скачивания
  // рисуется общим event_line() (он знает про /v0/pull независимо от kind).
  std::vector<std::string> draw_images() {
    const int W = width(), H = height();
    std::vector<std::string> out;
    out.push_back(c::hl(tr(" llmhost — изображения (художники) ")));
    if (img_typing) {
      out.push_back("");
      out.push_back(tr(" Имя каталога или org/repo: ") + img_input + c::accent("█"));
      out.push_back(" " + c::dim(tr("Конвенция своего репо: recipes/*.json + diffusion_models/ +")));
      out.push_back(" " + c::dim(tr("text_encoders/ + vae/ — по одному файлу на слот.")));
    } else {
      const std::string st = img_state();
      if (st == "loaded")
        out.push_back(std::string(" ") + tr("движок: загружена «") + c::ok(img_active()) +
                      tr("», порт ") + std::to_string(image.value("port", 0)));
      else if (st == "loading")
        out.push_back(" " + c::warn(tr("движок: загрузка…")));
      else if (st == "error")
        out.push_back(" " + c::err(tr("движок: ошибка — ") + image.value("error", std::string())));
      else
        out.push_back(" " + c::dim(tr("движок: не загружен (автостарт при первом запросе)")));
      auto models = image.value("models", json::array());
      std::vector<std::string> list;
      for (auto& m : models) {
        char gb[32];
        // gb — готовая строка (%s): printf с char[] на %.1f падал сегфолтом
        // (та же грабля, что в print_image_status).
        std::snprintf(gb, sizeof(gb), "%.1f", m.value("bytes", 0ull) / 1073741824.0);
        std::string id = m.value("id", "");
        const bool resident = st == "loaded" && img_active() == id;
        std::string line = " " + pad(trunc(id, 16), 16) + "  " +
                           pad(trunc(m.value("name", ""), 26), 26) + "  " +
                           padl(std::string(gb) + tr(" ГиБ"), 9) + "  " +
                           c::dim(tr("шаги ") + std::to_string(m.value("steps", -1)) + " · " +
                                  m.value("sampler", "—"));
        if (resident) line += "  " + c::ok(tr("· в памяти"));
        list.push_back(std::move(line));
      }
      if (list.empty())
        list.push_back(" " + c::dim(tr("(каталог пуст — скачайте: p)")));
      int vis = H - (int)out.size() - 3;
      if (vis < 1) vis = 1;
      if (img_cur < img_scroll) img_scroll = img_cur;
      if (img_cur >= img_scroll + vis) img_scroll = img_cur - vis + 1;
      if (img_scroll < 0) img_scroll = 0;
      for (int i = img_scroll; i < (int)list.size() && i < img_scroll + vis; ++i)
        out.push_back(i == img_cur ? cursor_line(list[i], W) : list[i]);
    }
    while ((int)out.size() < H - 2) out.push_back("");
    out.push_back(event_line());
    out.push_back(img_typing
                      ? keybar({{"Enter", tr("скачать")}, {"Esc", tr("отмена")}})
                      : keybar({{"Enter/F5", tr("загрузить")},
                                {"F8", tr("выгрузить")},
                                {"p", tr("скачать")},
                                {"Ctrl-D", tr("удалить")},
                                {"Esc", tr("назад")}}));
    (void)W;
    return out;
  }

  // F2: подробный статус модели из /v0/status — слоты, живые аргументы
  // ребёнка, состояния. Курсорная модель — первой, остальные загруженные следом.
  void open_model_stat() {
    if (rows.empty()) { say(tr("реестр пуст"), true); return; }
    const std::string sel_id = rows[cursor].id;
    auto st = get("/v0/status");
    json sel, loaded_arr = json::array();
    for (auto& m : st.value("models", json::array())) {
      if (m.value("state", "") != "loaded") continue;
      if (m.value("id", "") == sel_id) sel = m;
      else loaded_arr.push_back(m);
    }
    stat_lines.clear();
    auto push_slot = [&](const json& m) {
      stat_lines.push_back(c::hl(tr("Модель: ") + m.value("id", "?")));
      stat_lines.push_back(
          tr(" порт ") + std::to_string(m.value("port", 0)) +
          (m.value("pid", 0) ? "  ·  pid " + std::to_string(m.value("pid", 0)) : ""));
      if (m.contains("error") && !m["error"].get<std::string>().empty())
        stat_lines.push_back(" " + c::err(tr("ошибка: ") + m["error"].get<std::string>()));
      for (auto& kv : lh::fmt::slot_args_pairs(m["args"]))
        stat_lines.push_back(" " + c::dim(pad(kv.first + ":", 26)) + kv.second);
      stat_lines.push_back("");
    };
    if (!sel.is_null() && sel.value("state", "") == "loaded") push_slot(sel);
    for (auto& m : loaded_arr) push_slot(m);
    if (stat_lines.empty()) {
      stat_lines.push_back(c::dim(tr("загруженных моделей нет — F5 загрузит выбранную")));
      stat_lines.push_back("");
      stat_lines.push_back(c::dim(tr("модель курсора: ") + sel_id));
    }
    stat_scroll = 0;
    scr = Scr::ModelStat;
  }

  std::vector<std::string> draw_model_stat() {
    std::vector<std::string> out;
    out.push_back(c::hl(tr(" llmhost — статус модели ")));
    int vis = height() - 3;
    if (vis < 1) vis = 1;
    if (stat_scroll > (int)stat_lines.size()) stat_scroll = (int)stat_lines.size();
    for (int i = stat_scroll; i < stat_scroll + vis && i < (int)stat_lines.size(); ++i)
      out.push_back(stat_lines[i]);
    while ((int)out.size() < height() - 2) out.push_back("");
    out.push_back(keybar({{"F2/Esc", tr("назад")}, {"↑↓", tr("прокрутка")}}));
    return out;
  }

  std::vector<std::string> draw_card() {
    const int W = width(), H = height();
    std::vector<std::string> out;
    out.push_back(c::hl(tr(" llmhost — карточка модели ")));
    int vis = H - 3;
    if (card_scroll > (int)card_lines.size()) card_scroll = (int)card_lines.size();
    for (int i = card_scroll; i < (int)card_lines.size() && i < card_scroll + vis; ++i)
      out.push_back(card_lines[i]);
    while ((int)out.size() < H - 2) out.push_back("");
    out.push_back(event_line());
    out.push_back(keybar({{"Esc", tr("назад")},
                                {"C", "ctx"},
                                {"M", tr("mtp-режим")},
                                {"V", "vision"},
                                {"T", "thinking"},
                                {"↑↓", tr("прокрутка")}}));
    (void)W;
    return out;
  }

  std::vector<std::string> draw_engine() {
    const int W = width(), H = height();
    std::vector<std::string> out;
    out.push_back(c::hl(tr(" llmhost — движок llama.cpp ")));
    int vis = H - 3;
    if (engine_scroll > (int)engine_lines.size()) engine_scroll = (int)engine_lines.size();
    for (int i = engine_scroll; i < (int)engine_lines.size() && i < engine_scroll + vis; ++i)
      out.push_back(engine_lines[i]);
    while ((int)out.size() < H - 2) out.push_back("");
    out.push_back(event_line());
    out.push_back(keybar({{"F5/Enter", tr("команды")}, {"Esc", tr("назад")}}));
    (void)W;
    return out;
  }

  std::vector<std::string> draw_eng_menu() {
    // Шапка (установлен/актуальный) закреплена сверху и видна всегда; пункты
    // при нехватке высоты сжимаются до одной строки, иначе — окно со скроллом.
    const int W = width(), H = height();
    std::vector<std::string> out;
    out.push_back(c::hl(tr(" llmhost — команды движка ")));
    for (auto& l : eng_menu_header()) out.push_back(l);
    out.push_back("");
    int avail = H - (int)out.size() - 2; // событие + подсказки клавиш
    if (avail < 1) avail = 1;
    const int n = (int)eng_menu.size();
    auto item_pretty = [&](int i) {
      const auto& a = eng_menu[i];
      return std::vector<std::string>{
          (i == eng_mcur ? c::accent(" ▶ ") : "   ") +
              (i == eng_mcur ? c::accent(a.title) : a.title),
          c::dim("     " + a.cmd), ""};
    };
    auto item_compact = [&](int i) {
      const auto& a = eng_menu[i];
      std::string body = (i == eng_mcur ? c::accent(" ▶ ") : "   ") +
                      (i == eng_mcur ? c::accent(a.title) : a.title) +
                         c::dim("  ·  " + a.cmd);
      return std::vector<std::string>{lh::ui::trunc(body, (size_t)(W - 1))};
    };
    // Формат: красивый (3 строки на пункт) → компактный (1) → окно скролла.
    bool compact = n * 3 > avail;
    if (n <= avail) {
      for (int i = 0; i < n; ++i) {
        auto v = compact ? item_compact(i) : item_pretty(i);
        out.insert(out.end(), v.begin(), v.end());
      }
    } else {
      int window = avail;
      if (eng_mcur < eng_scroll) eng_scroll = eng_mcur;
      if (eng_mcur >= eng_scroll + window) eng_scroll = eng_mcur - window + 1;
      if (eng_scroll > n - window) eng_scroll = n - window;
      if (eng_scroll < 0) eng_scroll = 0;
      for (int i = eng_scroll; i < eng_scroll + window && i < n; ++i) {
        auto v = item_compact(i);
        out.insert(out.end(), v.begin(), v.end());
      }
    }
    while ((int)out.size() < H - 2) out.push_back("");
    // Перелив — не иллюзия: кадр с лишними строками уезжает за верх экрана.
    if ((int)out.size() > H - 2) out.resize(H - 2);
    out.push_back(event_line());
    out.push_back(keybar({{"↑↓", tr("выбор")}, {"Enter", tr("выполнить")}, {"Esc", tr("назад")}}));
    (void)W;
    return out;
  }

  std::vector<std::string> draw_eng_input() {
    const int W = width(), H = height();
    Panel p(tr("Обновить на конкретный релиз"));
    p.line(tr("Тег релиза llama.cpp (bNNNNN), пусто — последний:"));
    p.line("  " + c::accent(eng_input) + "_");
    p.hr();
    p.line(c::dim(tr("пин переписывает manifest.json (SHA256 из GitHub API);")));
    p.line(c::dim(tr("установка — следом, активной сборкой")));
    std::string body = p.render(std::min(W, 72));
    std::vector<std::string> all;
    for (size_t pos = 0; pos < body.size();) {
      size_t nl = body.find('\n', pos);
      if (nl == std::string::npos) nl = body.size();
      std::string line = body.substr(pos, nl - pos);
      if (!line.empty()) all.push_back(line);
      pos = nl + 1;
    }
    int top = (H - (int)all.size()) / 2;
    if (top < 0) top = 0;
    std::vector<std::string> out((size_t)top, "");
    for (auto& l : all) out.push_back(l);
    while ((int)out.size() < H - 1) out.push_back("");
    out.push_back(keybar({{"Enter", tr("пин + установить")}, {"Esc", tr("отмена")}}));
    (void)W;
    return out;
  }

  std::vector<std::string> draw_ctx_input() {
    const int W = width(), H = height();
    Panel p(tr("Ручной контекст"));
    p.line(tr("Модель: ") + ctx_id);
    p.line(c::dim(tr("  сейчас: ")) + (ctx_cur ? std::to_string(ctx_cur) + tr(" токенов (вручную)")
                                         : tr("авто — умная настройка")) +
           (ctx_plan ? c::dim(tr("  ·  план ") + std::to_string(ctx_plan)) : ""));
    p.line("");
    p.line(tr("Новые токены (число) или auto — снять ручной ctx:"));
    p.line("  " + c::accent(ctx_input) + "_");
    std::string body = p.render(std::min(W, 72));
    std::vector<std::string> all;
    for (size_t pos = 0; pos < body.size();) {
      size_t nl = body.find('\n', pos);
      if (nl == std::string::npos) nl = body.size();
      std::string line = body.substr(pos, nl - pos);
      if (!line.empty()) all.push_back(line);
      pos = nl + 1;
    }
    int top = (H - (int)all.size()) / 2;
    if (top < 0) top = 0;
    std::vector<std::string> out((size_t)top, "");
    for (auto& l : all) out.push_back(l);
    while ((int)out.size() < H - 1) out.push_back("");
    out.push_back(keybar({{"Enter", tr("применить")}, {"Esc", tr("отмена")}}));
    (void)W;
    return out;
  }

  std::vector<std::string> draw_confirm() {
    const int W = width(), H = height();
    Panel p(tr("Подтверждение"));
    if (rm_force_unload) {
      p.line(tr("«") + rm_id + tr("» занята генерацией — снять принудительно?"));
      p.line(c::err(tr("ВСЕ генерации этой модели обрываются")) +
             c::dim(tr(" — для всех клиентов.")));
    } else if (rm_is_image) {
      p.line(tr("Удалить художника «") + img_rm_id + tr("» с диска"));
      p.line(c::dim(tr("  файлы слотов, рецепт (если свободен) и запись привязки")));
      p.line(c::err(tr("БЕЗВОЗВРАТНО")) +
             c::dim(tr("? Загруженную модель демон сначала остановит.")));
    } else {
      p.line(tr("Удалить «") + rm_id + "» (" + human_bytes(rm_size) + tr(") с диска"));
      for (auto& c : rm_comp) p.line(c::dim(tr("  вместе с ней: ") + c));
      p.line(c::err(tr("БЕЗВОЗВРАТНО")) + c::dim(tr("? Файл .gguf нельзя будет восстановить.")));
    }
    std::string body = p.render(std::min(W, 72));
    std::vector<std::string> all;
    for (size_t pos = 0; pos < body.size();) {
      size_t nl = body.find('\n', pos);
      if (nl == std::string::npos) nl = body.size();
      std::string line = body.substr(pos, nl - pos);
      if (!line.empty()) all.push_back(line);
      pos = nl + 1;
    }
    int top = (H - (int)all.size()) / 2;
    if (top < 0) top = 0;
    std::vector<std::string> out((size_t)top, "");
    for (auto& l : all) out.push_back(l);
    while ((int)out.size() < H - 1) out.push_back("");
    out.push_back(keybar({{"Enter", rm_force_unload ? tr("снять") : tr("удалить")},
                                {"Esc", tr("отмена")}}));
    (void)W;
    return out;
  }

  std::vector<std::string> draw_help() {
    const int W = width(), H = height();
    std::vector<std::string> out;
    out.push_back(c::hl(tr(" llmhost — управление (клавиши) ")));
    for (std::string l : {
             tr("ГЛАВНЫЙ ЭКРАН"),
             tr("  ↑↓, PgUp/PgDn     выбор модели"),
             tr("  Enter, F3         карточка модели (C — ручной ctx: число или auto;\n"
                "                    M — mtp-режим, V — vision, T — thinking)"),
             tr("  F4                чат с выбранной моделью (лента; Enter — отправить,"),
             tr("                    Esc во время генерации — стоп; ↑↓/PgUp — лента;"),
             tr("                    /clear, /bye)"),
             tr("  F5                загрузить выбранную"),
             tr("  F6                токены доступа к /v1 (создать/отозвать)"),
             tr("  F7                каталог моделей (поиск, кванты, скачать/загрузить)"),
             tr("  F8, u             выгрузить выбранную"),
             tr("  F9 → F5           команды движка: обновить пин/установить/переключить"),
             tr("  i                 художники (изображения): каталог, загрузка/удаление"),
             tr("  Ctrl-D            удалить модель с диска (mmproj/MTP и шарды — с ней)"),
             tr("  Ctrl-U            выгрузить все"),
             tr("  Ctrl-X            остановить идущее скачивание (любой экран)"),
             tr("  F10, q, Ctrl-C    выход (демон продолжает работать)"),
             tr("КАТАЛОГ (как окно каталога Windows-версии)"),
             tr("  Enter             у результата — показать кванты; у кванта — скачать"),
             tr("  F5 / F6           скачать выбранное / загрузить скачанное"),
             tr("ИЗОБРАЖЕНИЯ (художники)"),
             tr("  Enter/F5          загрузить выбранного (к нему идут запросы /v1/images)"),
             tr("  F8, u             остановить движок (выгрузить)"),
             tr("  p                 скачать: имя каталога или org/repo с раскладкой"),
             tr("                    recipes/ + diffusion_models/ + text_encoders/ + vae/"),
             tr("  Ctrl-D            удалить с диска (загруженную демон выгрузит сам)"),
             "",
             tr("Слова — те же, что в Windows-версии: «в памяти», «грузится…»,"),
             tr("«не загрузилась». Данные обновляются сами раз в секунду."),
         }) {
      out.push_back(" " + c::dim(l));
    }
    while ((int)out.size() < H - 2) out.push_back("");
    out.push_back(event_line());
    out.push_back(keybar({{"Esc", tr("назад")}}));
    (void)W;
    return out;
  }

  void draw() {
    std::vector<std::string> out;
    switch (scr) {
      case Scr::Main: out = draw_main(); break;
      case Scr::Catalog: out = draw_catalog(); break;
      case Scr::Chat: out = draw_chat(); break;
      case Scr::Tokens: out = draw_tokens(); break;
      case Scr::TokenInput: out = draw_token_input(); break;
      case Scr::Card: out = draw_card(); break;
      case Scr::ModelStat: out = draw_model_stat(); break;
      case Scr::Metrics: out = draw_metrics(); break;
      case Scr::Engine: out = draw_engine(); break;
      case Scr::EngMenu: out = draw_eng_menu(); break;
      case Scr::EngInput: out = draw_eng_input(); break;
      case Scr::CtxInput: out = draw_ctx_input(); break;
      case Scr::ConfirmRm: out = draw_confirm(); break;
      case Scr::Help: out = draw_help(); break;
      case Scr::Images: out = draw_images(); break;
    }
    draw_lines(out);
  }

  // true — выход из интерфейса.
  // Окно UTF-8 (kprev) живёт в Tui: handle получает его значение НА ВХОДЕ
  // (prev) и обновляет член на выходе — цикл лишь пробрасывает.
  bool handle(Key k, char ch) {
    const unsigned char prev = kprev;
    kprev = k == Key::Char ? (unsigned char)ch : 0;
    if (k == Key::None) return false;
    if (k == Key::F10 || k == Key::CtrlC) return true;
    // Остановка скачивания доступна с любого экрана: строка прогресса общая.
    if (k == Key::CtrlX && !pull.is_null() && !pull.empty()) {
      auto res = cli.Delete("/v0/pull", hdrs, "", "text/plain");
      if (!res) say(tr("демон не отвечает"), true);
      else if (res->status >= 400) say(error_message(res->body), true);
      else say(tr("остановка скачивания…"));
      return false;
    }
    switch (scr) {
      case Scr::Main: {
        switch (k) {
          case Key::Up: if (cursor > 0) --cursor; break;
          case Key::Down: if (cursor + 1 < (int)rows.size()) ++cursor; break;
          case Key::PgUp: cursor -= height() / 2; if (cursor < 0) cursor = 0; break;
          case Key::PgDn: cursor += height() / 2; break;
          case Key::Home: cursor = 0; break;
          case Key::End: cursor = (int)rows.size() - 1; break;
          case Key::Enter: case Key::F3: open_card(); break;
          case Key::F2: open_model_stat(); break;
          case Key::F4: open_chat(); break;
          case Key::F5: action_load(); break;
          case Key::F7: scr = Scr::Catalog; break;
          case Key::F6: tokens_reload(); scr = Scr::Tokens; break;
          case Key::F8: action_unload(false); break;
          case Key::CtrlU: action_unload(true); break;
          case Key::CtrlD: action_rm_begin(); break;
          case Key::F9: open_engine(); break;
          case Key::F1: scr = Scr::Help; break;
          case Key::Char:
            if (ch == 'l') action_load();
            else if (ch == 'u') action_unload(false);
            else if (ch == 'c') open_chat();
            else if (ch == 'm' || key_seq(prev, ch, "ь")) scr = Scr::Metrics;
            else if (ch == 'i') { img_typing = false; img_input.clear(); scr = Scr::Images; }
            else if (ch == 'q' || key_seq(prev, ch, "й")) return true;
            break;
          default: break;
        }
        break;
      }
      case Scr::Catalog: {
        switch (k) {
          case Key::Up:
            if (cat_stage == 1) { if (cat_rcur > 0) --cat_rcur; }
            else if (cat_stage == 2) { if (cat_qcur > 0) --cat_qcur; }
            break;
          case Key::Down:
            if (cat_stage == 1) {
              if (cat_rcur + 1 < (int)cat_results.size()) ++cat_rcur;
            } else if (cat_stage == 2) {
              ++cat_qcur;
            }
            break;
          case Key::Tab:
            if (cat_stage == 0) cat_stage = 1;
            else if (cat_stage == 1) cat_fetch_detail();
            else cat_stage = 1;
            break;
          case Key::Enter:
            if (cat_stage == 0) cat_start_search();
            else if (cat_stage == 1) cat_fetch_detail();
            else cat_pull();
            break;
          case Key::F5: cat_pull(); break;
          case Key::F6: cat_load_downloaded(); break;
          case Key::Esc:
            if (cat_stage == 0) scr = Scr::Main;
            else cat_stage = cat_stage == 2 ? 1 : 0;
            break;
          case Key::Char:
            if (cat_stage == 0) {
              if ((unsigned char)ch >= 0x20) cat_query += ch;
            } else if (ch == 'q') scr = Scr::Main;
            break;
          case Key::Backspace:
            if (cat_stage == 0) {
              while (!cat_query.empty() && (cat_query.back() & 0xC0) == 0x80)
                cat_query.pop_back();
              if (!cat_query.empty()) cat_query.pop_back();
            }
            break;
          default: break;
        }
        break;
      }
      case Scr::Chat: {
        if (chat_generating) {
          if (k == Key::Esc) chat_abort = true;  // подхватит receiver
          return false;
        }
        switch (k) {
          case Key::Esc: scr = Scr::Main; break;
          case Key::Enter: chat_send(); break;
          case Key::Backspace: {
            while (!chat_input.empty() && (chat_input.back() & 0xC0) == 0x80)
              chat_input.pop_back();
            if (!chat_input.empty()) chat_input.pop_back();
            break;
          }
          case Key::Up: case Key::Down: case Key::PgUp: case Key::PgDn:
          case Key::Home: case Key::End:
            chat_scroll_key(k);
            break;
          case Key::Char:
            if ((unsigned char)ch >= 0x20) chat_input += ch;
            break;
          default: break;
        }
        break;
      }
      case Scr::Tokens: {
        switch (k) {
          case Key::Esc: scr = Scr::Main; break;
          case Key::Enter: token_name.clear(); scr = Scr::TokenInput; break;
          case Key::Del: token_revoke(); break;
          case Key::Up: if (tok_cur > 0) --tok_cur; break;
          case Key::Down: if (tok_cur + 1 < (int)tokens_list.size()) ++tok_cur; break;
          default: break;
        }
        break;
      }
      case Scr::TokenInput: {
        if (k == Key::Enter) {
          token_add();
          scr = Scr::Tokens;
        } else if (k == Key::Esc) {
          scr = Scr::Tokens;
        } else if (k == Key::Backspace) {
          while (!token_name.empty() && (token_name.back() & 0xC0) == 0x80)
            token_name.pop_back();
          if (!token_name.empty()) token_name.pop_back();
        } else if (k == Key::Char && (unsigned char)ch >= 0x20) {
          token_name += ch;
        }
        break;
      }
      case Scr::ModelStat: {
        switch (k) {
          case Key::Esc: case Key::F2: scr = Scr::Main; break;
          case Key::Up: if (stat_scroll > 0) --stat_scroll; break;
          case Key::Down: ++stat_scroll; break;
          case Key::PgUp: stat_scroll -= height() / 2; if (stat_scroll < 0) stat_scroll = 0; break;
          case Key::PgDn: stat_scroll += height() / 2; break;
          default: break;
        }
        break;
      }
      case Scr::Card: {
        switch (k) {
          case Key::Esc: scr = Scr::Main; break;
          case Key::Up: if (card_scroll > 0) --card_scroll; break;
          case Key::Down: ++card_scroll; break;
          case Key::PgUp: card_scroll -= height() / 2; if (card_scroll < 0) card_scroll = 0; break;
          case Key::PgDn: card_scroll += height() / 2; break;
          case Key::Char:
            if (ch == 'C' || ch == 'c' || key_seq(prev, ch, "С") ||
                key_seq(prev, ch, "с")) {
              if (ctx_path.empty()) { say(tr("модель не выбрана"), true); break; }
              ctx_input.clear();
              scr = Scr::CtxInput;
            } else if (ch == 'M' || ch == 'm' || key_seq(prev, ch, "Ь") ||
                       key_seq(prev, ch, "ь")) {
              action_mtp_cycle();
            } else if (ch == 'V' || ch == 'v' ||
                       key_seq(prev, ch, "м") || key_seq(prev, ch, "М")) {
              action_vision_cycle();
            } else if (ch == 'T' || ch == 't' ||
                       key_seq(prev, ch, "е") || key_seq(prev, ch, "Е")) {
              action_think_cycle();
            }
            break;
          default: break;
        }
        break;
      }
      case Scr::CtxInput: {
        if (k == Key::Enter) {
          action_ctx_apply();
          scr = Scr::Card;
        } else if (k == Key::Esc) {
          scr = Scr::Card;
        } else if (k == Key::Backspace) {
          while (!ctx_input.empty() && (ctx_input.back() & 0xC0) == 0x80)
            ctx_input.pop_back();
          if (!ctx_input.empty()) ctx_input.pop_back();
        } else if (k == Key::Char && (unsigned char)ch >= 0x20) {
          ctx_input += ch;
        }
        break;
      }
      case Scr::Engine: {
        if (k == Key::Esc) scr = Scr::Main;
        // F5 — основная клавиша; Enter и m — запасные (терминалы шлют F5
        // по-разному, Enter не портит ни один).
        else if (k == Key::F5 || k == Key::Enter ||
                 (k == Key::Char && (ch == 'm' || ch == 'M' ||
                                     key_seq(prev, ch, "ь") ||
                                     key_seq(prev, ch, "Ь"))))
          open_eng_menu();
        break;
      }
      case Scr::EngMenu: {
        if (k == Key::Esc || k == Key::F5) scr = Scr::Engine;
        else if (k == Key::Up) { if (eng_mcur > 0) --eng_mcur; }
        else if (k == Key::Down) { if (eng_mcur + 1 < (int)eng_menu.size()) ++eng_mcur; }
        else if (k == Key::PgUp) eng_mcur = 0;
        else if (k == Key::PgDn) eng_mcur = (int)eng_menu.size() - 1;
        else if (k == Key::Enter && eng_mcur < (int)eng_menu.size()) eng_menu_exec(eng_menu[eng_mcur]);
        break;
      }
      case Scr::EngInput: {
        if (k == Key::Enter) {
          std::string tag = eng_input;
          eng_input.clear();
          std::string rel = engine_pin(tag);
          if (!rel.empty()) {
            say(tr("пин манифеста: ") + rel);
            eng_queue = {engine.value("active", "")};
            scr = Scr::Engine;
          } else {
            scr = Scr::EngMenu;
          }
        } else if (k == Key::Esc) {
          scr = Scr::EngMenu;
        } else if (k == Key::Backspace) {
          while (!eng_input.empty() && (eng_input.back() & 0xC0) == 0x80)
            eng_input.pop_back();
          if (!eng_input.empty()) eng_input.pop_back();
        } else if (k == Key::Char && (unsigned char)ch >= 0x20 && eng_input.size() < 32) {
          eng_input += ch;
        }
        break;
      }
      case Scr::ConfirmRm: {
        const bool yes = k == Key::Enter ||
                         (k == Key::Char && (ch == 'y' || ch == 'Y' ||
                                             key_seq(prev, ch, "д") || key_seq(prev, ch, "Д")));
        const bool no = k == Key::Esc ||
                        (k == Key::Char && (ch == 'n' || ch == 'N' || key_seq(prev, ch, "н")));
        if (rm_force_unload) {
          if (yes) action_unload_force_confirm();
          else if (no) { rm_force_unload = false; scr = Scr::Main; }
        } else if (rm_is_image) {
          if (yes) img_rm_confirm();
          else if (no) { rm_is_image = false; scr = Scr::Images; }
        } else {
          if (yes) action_rm_confirm();
          else if (no) scr = Scr::Main;
        }
        break;
      }
      case Scr::Metrics: {
        if (k == Key::Esc) scr = Scr::Main;
        else if (k == Key::Char && (ch == 'm' || ch == 'M' ||
                                    key_seq(prev, ch, "ь") || key_seq(prev, ch, "Ь")))
          metrics_sample(); // вручную — не дожидаясь 2-секундного тика
        break;
      }
      case Scr::Images: {
        if (img_typing) {
          if (k == Key::Enter) img_pull_start();
          else if (k == Key::Esc) { img_typing = false; img_input.clear(); }
          else if (k == Key::Backspace) {
            while (!img_input.empty() && (img_input.back() & 0xC0) == 0x80)
              img_input.pop_back();
            if (!img_input.empty()) img_input.pop_back();
          } else if (k == Key::Char && (unsigned char)ch >= 0x20) {
            img_input += ch;
          }
          break;
        }
        switch (k) {
          case Key::Esc: scr = Scr::Main; break;
          case Key::Enter: case Key::F5: img_load(); break;
          case Key::F8: case Key::Del: img_unload(); break;
          case Key::CtrlD: img_rm_begin(); break;
          case Key::Up: if (img_cur > 0) --img_cur; break;
          case Key::Down:
            if (img_cur + 1 < (int)image.value("models", json::array()).size()) ++img_cur;
            break;
          case Key::Char:
            if (ch == 'p' || key_seq(prev, ch, "з")) {
              img_typing = true;
              img_input.clear();
            } else if (ch == 'u') {
              img_unload();
            }
            break;
          default: break;
        }
        break;
      }
      case Scr::Help: {
        if (k == Key::Esc || k == Key::Enter || (k == Key::Char && ch == 'q')) scr = Scr::Main;
        break;
      }
    }
    return false;
  }

  void open_card() {
    if (rows.empty()) { say(tr("реестр пуст"), true); return; }
    const std::string& id = rows[cursor].id;
    json entry;
    for (auto& m : models.value("models", json::array()))
      if (m.value("id", "") == id) { entry = m; break; }
    if (entry.is_null()) { say(tr("запись не найдена"), true); return; }
    ctx_path = entry.value("path", "");
    ctx_id = entry.value("id", "");
    ctx_cur = entry.value("ctx_override", 0);
    ctx_plan = entry.value("fit", json::object()).value("ctx", 0);
    card_mtp = entry.value("mtp_mode", "");
    card_vision = entry.value("vision_mode", "auto");
    card_think = entry.value("think_mode", "auto");
    std::string body = render_show(entry);
    card_lines.clear();
    for (size_t pos = 0; pos < body.size();) {
      size_t nl = body.find('\n', pos);
      if (nl == std::string::npos) nl = body.size();
      std::string line = body.substr(pos, nl - pos);
      if (!line.empty()) card_lines.push_back(line);
      pos = nl + 1;
    }
    // Живой слот: что реально поехало движку (/v0/status .args ребёнка).
    if (entry.value("state", "") == "loaded") {
      auto st = get("/v0/status");
      for (auto& sm : st.value("models", json::array())) {
        if (sm.value("id", "") != entry.value("id", "") || !sm.contains("args"))
          continue;
        card_lines.push_back("");
        card_lines.push_back(c::hl(tr("Слот: живые аргументы ребёнка")));
        card_lines.push_back(
            tr(" порт ") + std::to_string(sm.value("port", 0)) +
            (sm.value("pid", 0)
                 ? c::dim("  ·  pid " + std::to_string(sm.value("pid", 0)))
                 : ""));
        for (auto& kv : lh::fmt::slot_args_pairs(sm["args"]))
          card_lines.push_back(" " + c::dim(pad(kv.first + ":", 26)) + kv.second);
      }
    }
    card_scroll = 0;
    scr = Scr::Card;
  }

  // Цикл пер-модельного mtp-mode: auto → on → off → auto. Пишет переопределение
  // через демон; слот уходит в сон, первый запрос поднимет ребёнка заново.
  void action_mtp_cycle() {
    if (ctx_path.empty()) { say(tr("модель не выбрана"), true); return; }
    const char* next = card_mtp == "on" ? "off" : card_mtp == "off" ? "auto" : "on";
    auto res = cli.Post("/v0/models/mtp", hdrs,
                        json{{"path", ctx_path}, {"mode", next}}.dump(),
                        "application/json");
    if (!res) { say(tr("демон не отвечает"), true); return; }
    if (res->status >= 400) { say(error_message(res->body), true); return; }
    refresh();
    // Перестроить карточку с новым режимом и живым слотом.
    if (!rows.empty()) {
      const std::string id = rows[cursor].id;
      for (auto& m : models.value("models", json::array()))
        if (m.value("id", "") == id) {
          std::string body = render_show(m);
          card_lines.clear();
          for (size_t pos = 0; pos < body.size();) {
            size_t nl = body.find('\n', pos);
            if (nl == std::string::npos) nl = body.size();
            std::string line = body.substr(pos, nl - pos);
            if (!line.empty()) card_lines.push_back(line);
            pos = nl + 1;
          }
          break;
        }
    }
    card_mtp = next;
    say(std::string("mtp-mode = ") + next + tr(" — слот в сон"));
  }

  // Цикл пер-модельного vision-режима по V: auto → off → auto. auto — своя
  // проекция прицепится сама (родство имён); off — ребёнок без --mmproj.
  void action_vision_cycle() {
    if (ctx_path.empty()) { say(tr("модель не выбрана"), true); return; }
    const char* next = card_vision == "off" ? "auto" : "off";
    auto res = cli.Post("/v0/models/vision", hdrs,
                        json{{"path", ctx_path}, {"mode", next}}.dump(),
                        "application/json");
    if (!res) { say(tr("демон не отвечает"), true); return; }
    if (res->status >= 400) { say(error_message(res->body), true); return; }
    refresh();
    // Перестроить карточку с новым режимом и живым слотом.
    if (!rows.empty()) {
      const std::string id = rows[cursor].id;
      for (auto& m : models.value("models", json::array()))
        if (m.value("id", "") == id) {
          std::string body = render_show(m);
          card_lines.clear();
          for (size_t pos = 0; pos < body.size();) {
            size_t nl = body.find('\n', pos);
            if (nl == std::string::npos) nl = body.size();
            std::string line = body.substr(pos, nl - pos);
            if (!line.empty()) card_lines.push_back(line);
            pos = nl + 1;
          }
          break;
        }
    }
    card_vision = next;
    say(std::string("vision-mode = ") + next + tr(" — слот в сон"));
  }

  // Цикл пер-модельного thinking-режима по T: auto → off → on → auto.
  // auto — движок сам детектит по шаблону; off/on — форс --reasoning.
  void action_think_cycle() {
    if (ctx_path.empty()) { say(tr("модель не выбрана"), true); return; }
    const char* next = card_think == "off" ? "on"
                       : card_think == "on" ? "auto" : "off";
    auto res = cli.Post("/v0/models/thinking", hdrs,
                        json{{"path", ctx_path}, {"mode", next}}.dump(),
                        "application/json");
    if (!res) { say(tr("демон не отвечает"), true); return; }
    if (res->status >= 400) { say(error_message(res->body), true); return; }
    refresh();
    // Перестроить карточку с новым режимом и живым слотом.
    if (!rows.empty()) {
      const std::string id = rows[cursor].id;
      for (auto& m : models.value("models", json::array()))
        if (m.value("id", "") == id) {
          std::string body = render_show(m);
          card_lines.clear();
          for (size_t pos = 0; pos < body.size();) {
            size_t nl = body.find('\n', pos);
            if (nl == std::string::npos) nl = body.size();
            std::string line = body.substr(pos, nl - pos);
            if (!line.empty()) card_lines.push_back(line);
            pos = nl + 1;
          }
          break;
        }
    }
    card_think = next;
    say(std::string("thinking-mode = ") + next + tr(" — слот в сон"));
  }

  void render_engine_lines() {
    std::string body = render_engine(engine);
    engine_lines.clear();
    for (size_t pos = 0; pos < body.size();) {
      size_t nl = body.find('\n', pos);
      if (nl == std::string::npos) nl = body.size();
      std::string line = body.substr(pos, nl - pos);
      if (!line.empty()) engine_lines.push_back(line);
      pos = nl + 1;
    }
  }

  void open_engine() {
    engine = get("/v0/engine");
    if (engine.is_null()) { say(tr("демон не отвечает"), true); return; }
    render_engine_lines();
    engine_scroll = 0;
    scr = Scr::Engine;
  }

  // ---- меню команд движка (F9 → F5) ---------------------------------------
  void open_eng_menu() {
    if (engine.is_null()) engine = get("/v0/engine");
    if (engine.is_null()) { say(tr("демон не отвечает"), true); return; }
    eng_menu.clear();
    const std::string active = engine.value("active", "");
    eng_menu.push_back({tr("Свежий b-релиз: перепинить манифест + установить активную") +
                            (active.empty() ? std::string() : " «" + active + "»"),
                        "lhctl engine update", "update", ""});
    eng_menu.push_back({tr("Только перепинить манифест (бинарники не качаются)"),
                        "lhctl engine update --pin-only", "pin_only", ""});
    eng_menu.push_back({tr("Свежий b-релиз на ВСЕ пиннутые сборки"),
                        "lhctl engine update --all", "update_all", ""});
    eng_menu.push_back({tr("Конкретный релиз… (ввод тега bNNNNN)"),
                        "lhctl engine update --release bNNNNN", "update_release", ""});
    for (auto& b : engine.value("builds", json::array())) {
      if (!b.value("pinned", false)) continue;
      const std::string id = b.value("id", "");
      // Ветки явным списком (выбор между cuda-ветками прямо в меню): активная
      // видна в шапке без пункта; установленная — переключение и
      // переустановка из текущего пина; только пиннутая — установка.
      if (id == active) {
        eng_menu.push_back({tr("Ветка «") + id + tr("»: переустановить из текущего пина"),
                            "lhctl engine install " + id, "install", id});
        continue;
      }
      if (b.value("installed", false)) {
        eng_menu.push_back({tr("Ветка «") + id + tr("»: переключиться"),
                            "lhctl engine use " + id, "use", id});
        eng_menu.push_back({tr("Ветка «") + id + tr("»: переустановить из текущего пина"),
                            "lhctl engine install " + id, "install", id});
      } else {
        eng_menu.push_back({tr("Ветка «") + id + tr("»: пиннута, не установлена — установить"),
                            "lhctl engine install " + id, "install", id});
      }
    }
    if (!engine.value("selected", "").empty())
      eng_menu.push_back({tr("Сборка по железу (снять ручной выбор)"),
                          "lhctl engine use auto", "use", "auto"});
    eng_mcur = eng_mcur >= (int)eng_menu.size() ? 0 : eng_mcur;
    eng_latest_pending = true; // шапку «установлен / актуальный» доносит тик
    scr = Scr::EngMenu;
  }

  void eng_latest_fetch() {
    eng_latest_pending = false;
    auto res = cli.Get("/v0/engine/latest", hdrs);
    if (!res || res->status != 200) {
      eng_latest_ok = false;
      eng_latest.clear();
      eng_latest_note = tr("GitHub недоступен");
      return;
    }
    try {
      auto j = json::parse(res->body);
      if (j.contains("release")) {
        eng_latest = j.value("release", "");
        eng_latest_ok = true;
        eng_latest_note.clear();
      } else {
        eng_latest.clear();
        eng_latest_ok = false;
        std::string e = j.value("error", std::string());
        eng_latest_note = e.empty() ? tr("GitHub недоступен") : e;
      }
    } catch (...) {
      eng_latest_ok = false;
      eng_latest_note = tr("ответ не разобрался");
    }
  }

  // Шапка меню: установленный vs актуальный релиз (источник — GitHub).
  std::vector<std::string> eng_menu_header() {
    std::vector<std::string> head;
    std::string st = engine.value("state", "idle");
    if (st == "downloading" || st == "extracting") {
      uint64_t done = engine.value("done", (uint64_t)0),
               total = engine.value("total", (uint64_t)0);
      double frac = total ? (double)done / (double)total : 0.0;
      head.push_back(c::warn(tr("Установка «") + engine.value("build", "") + "»: " +
                             c::by_frac(frac, bar(frac, 20)) + "  " +
                             human_bytes(done) + tr(" из ") + human_bytes(total) +
                             tr(" — остальные команды недоступны")));
      head.push_back("");
    }
    std::string installed = "?";
    const std::string active = engine.value("active", "");
    for (auto& b : engine.value("builds", json::array()))
      if (b.value("id", "") == active && !active.empty()) {
        installed = b.value("installed_release", "");
        if (installed.empty()) installed = "?";
      }
    std::string line = tr("Установлен: ") + installed +
                       (active.empty() ? "" : " («" + active + "»)");
    // Ветки явным списком: юзер просил выбор между cuda-ветками «по-честному»
    // — что пиннуто, что едет, видно в шапке до входа в подменю (новая ветка
    // появляется сама после пина свежего релиза — достройка из ассетов).
    std::string branches;
    for (auto& b : engine.value("builds", json::array())) {
      if (!b.value("pinned", false)) continue;
      const std::string id = b.value("id", "");
      if (!branches.empty()) branches += "  ·  ";
      if (id == active) branches += id + tr(" — активная");
      else branches += id + (b.value("installed", false) ? tr(" — установлена")
                                                          : tr(" — пиннута, не установлена"));
    }
    if (eng_latest_pending) {
      line += tr("  ·  актуальный: проверяю…");
      head.push_back(c::dim(line));
      if (!branches.empty()) head.push_back(tr("Ветки: ") + branches);
      return head;
    }
    if (!eng_latest_ok) {
      line += tr("  ·  актуальный: ") + c::warn(tr("не узнать"));
      head.push_back(c::dim(line));
      head.push_back(c::dim("      " + eng_latest_note));
      if (!branches.empty()) head.push_back(tr("Ветки: ") + branches);
      return head;
    }
    line += tr("  ·  актуальный: ") + c::accent(eng_latest);
    if (installed == "?")
      head.push_back(line), head.push_back(c::dim(tr("      установленный релиз не определён")));
    else if (installed != eng_latest)
      head.push_back(line), head.push_back(c::warn(tr("      доступно обновление — первый пункт меню")));
    else
      head.push_back(line), head.push_back(c::ok(tr("      актуален")));
    if (!branches.empty()) head.push_back(tr("Ветки: ") + branches);
    return head;
  }

  // Пин манифеста (синхронно, секунды). Пустая строка — ошибка (скажена).
  std::string engine_pin(const std::string& tag) {
    auto res = cli.Post("/v0/engine/pin", hdrs,
                        json{{"release", tag.empty() ? "latest" : tag}}.dump(),
                        "application/json");
    if (!res) { say(tr("демон не отвечает"), true); return ""; }
    if (res->status >= 400) { say(error_message(res->body), true); return ""; }
    try {
      return json::parse(res->body).value("release", "");
    } catch (...) { return ""; }
  }

  void engine_install(const std::string& build) {
    auto res = cli.Post("/v0/engine", hdrs, json{{"build", build}}.dump(),
                        "application/json");
    if (res && res->status < 400) {
      say(tr("ставлю «") + build + tr("» — прогресс на экране движка"));
      scr = Scr::Engine;
    } else {
      say(error_message(res ? res->body : tr("демон не отвечает")), true);
    }
  }

  void eng_menu_exec(const EngAction& a) {
    std::string st = engine.value("state", "idle");
    if (st == "downloading" || st == "extracting")
      return; // прогресс и так живёт в шапке меню и в строке событий каждую секунду
    if (a.kind == "update") {
      std::string rel = engine_pin("latest");
      if (rel.empty()) return;
      say(tr("пин манифеста: ") + rel);
      eng_queue = {engine.value("active", "")};
      open_engine();
    } else if (a.kind == "pin_only") {
      std::string rel = engine_pin("latest");
      if (rel.empty()) return;
      say(tr("пин манифеста: ") + rel + tr(" — установка отдельно: F5 → Установить"));
      engine = get("/v0/engine");
      render_engine_lines();
    } else if (a.kind == "update_all") {
      std::string rel = engine_pin("latest");
      if (rel.empty()) return;
      say(tr("пин манифеста: ") + rel);
      eng_queue.clear();
      for (auto& b : engine.value("builds", json::array()))
        if (b.value("pinned", false)) eng_queue.push_back(b.value("id", ""));
      open_engine();
    } else if (a.kind == "update_release") {
      eng_input.clear();
      scr = Scr::EngInput;
    } else if (a.kind == "install") {
      eng_queue = {a.arg};
      engine_install(a.arg);
    } else if (a.kind == "use") {
      auto res = cli.Post("/v0/engine/build", hdrs,
                          json{{"build", a.arg == "auto" ? "" : a.arg}}.dump(),
                          "application/json");
      if (!res) { say(tr("демон не отвечает"), true); return; }
      if (res->status >= 400) { say(error_message(res->body), true); return; }
      say(a.arg == "auto" ? tr("выбор снят — дети перезапускаются на сборке по железу")
                          : tr("выбрана «") + a.arg + tr("» — дети перезапускаются"));
      engine = get("/v0/engine");
      render_engine_lines();
    }
  }

  // Установки: тик раз в секунду — запускает очередь, следит за прогрессом
  // и финалом, держит экран движка живым (иначе панель показывает застывшее
  // «idle», а установка идёт молча — поймано юзером живьём). eng_watch —
  // установка в полёте, даже когда очередь уже пуста.
  void engine_tick() {
    bool on_screen = scr == Scr::Engine || scr == Scr::EngMenu;
    if (!on_screen && eng_queue.empty() && !eng_watch) return;
    auto prev = engine.value("state", "idle");
    engine = get("/v0/engine");
    if (engine.is_null()) return;
    if (scr == Scr::Engine) render_engine_lines();
    std::string st = engine.value("state", "idle");
    std::string build = engine.value("build", "");
    if (st == "downloading" || st == "extracting") {
      eng_watch = true;
      uint64_t done = engine.value("done", (uint64_t)0),
               total = engine.value("total", (uint64_t)0);
      int pct = total ? (int)(done * 100 / total) : 0;
      say(tr("установка «") + build + "»: " + human_bytes(done) + tr(" из ") +
          human_bytes(total) + " (" + std::to_string(pct) + "%)");
      return;
    }
    if (!eng_queue.empty() && st == "idle") {
      std::string next = eng_queue.front();
      eng_queue.erase(eng_queue.begin());
      auto res = cli.Post("/v0/engine", hdrs, json{{"build", next}}.dump(),
                          "application/json");
      if (res && res->status < 400) {
        say(tr("ставлю «") + next + "»…" +
            (eng_queue.empty()
                 ? ""
                 : tr(" (в очереди ещё ") + std::to_string(eng_queue.size()) + ")"));
      } else {
        say(error_message(res ? res->body : tr("демон не отвечает")), true);
        eng_queue.clear();
        eng_watch = false;
      }
      return;
    }
    if (eng_watch && (st == "idle" || st == "error")) {
      eng_watch = false;
      if (st == "error") {
        std::string e = engine.value("error", std::string());
        say(e.empty() ? tr("ошибка установки") : e, true);
      } else if (prev != "idle") {
        say(tr("✓ сборка «") + build + tr("» установлена"));
      }
    }
  }


  void fetch_credentials() {
    auto c = get("/v0/credentials");
    if (c.is_null()) return;
    std::string url = c.value("openai_base_url", "");
    size_t colon = url.rfind(':');
    if (colon != std::string::npos) openai_port = std::atoi(url.c_str() + colon + 1);
    openai_key = c.value("api_key", std::string());
  }
};

}  // namespace

int run(int port, const std::string& token) {
  httplib::Client cli("127.0.0.1", port);
  cli.set_connection_timeout(std::chrono::seconds(3));
  cli.set_read_timeout(std::chrono::seconds(30));
  httplib::Headers hdrs;
  if (!token.empty()) hdrs = {{"Authorization", "Bearer " + token}};

#ifndef _WIN32
  struct sigaction sa {};
  sa.sa_handler = on_winch;
  sigaction(SIGWINCH, &sa, nullptr);
#endif

  RawTerminal rt;
  Tui t{cli, hdrs, port};
  t.fetch_credentials();
  t.refresh();
  t.collect();
  t.draw();
  auto last = std::chrono::steady_clock::now();
  for (;;) {
    char ch = 0;
    Key k = read_key(250, ch);
#ifndef _WIN32
    if (g_winch) {
      g_winch = 0;
      t.draw();
      continue;
    }
#endif
    if (k != Key::None) {
      if (t.handle(k, ch)) break;
      t.refresh();
      t.collect();
      t.draw();
      last = std::chrono::steady_clock::now();
      continue;
    }
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count() >= 1000) {
      t.refresh();
      t.collect();
      if (!t.chat_pending.empty()) t.try_send_pending();
      t.engine_tick();
      if (t.eng_latest_pending && t.scr == Scr::EngMenu) t.eng_latest_fetch();
      t.draw();
      last = now;
    }
  }
  return 0;
}

}  // namespace lh::tui
