// lhctl — консольный клиент control API llmhost (наш ответ на `ollama ps`).
//   lhctl status             — что загружено, VRAM, режим (в tty — приборная
//                              панель с барами; -w — живьём, resize-безопасно)
//   lhctl models             — реестр моделей с состояниями
//   lhctl search <запрос>    — поиск по каталогу HuggingFace с оценкой VRAM
//   lhctl load <файл|имя>    — загрузить модель (.gguf или имя из реестра)
//   lhctl pull <имя> [файл]  — скачать модель (короткое имя или org/repo)
//   lhctl engine             — движок: пин манифеста, сборки, состояние
//   lhctl engine install <сборка> — установить сборку из текущего пина
//   lhctl engine update [сборка]  — пин свежего релиза llama.cpp + установка
//   lhctl unload [id]        — выгрузить одну (id роутера) или все
//                              (--force — занятую принудительно: обрывает
//                               все генерации модели)
//   lhctl quit               — завершить llmhost
//   lhctl completion bash|powershell — скрипт автодополнения (TAB подсказывает
//                                      имена моделей живого реестра)
// Опции: --port N (по умолчанию 7331), --token XXX (по умолчанию
// data\token.txt рядом с lhctl.exe — та же установка, что и llmhost).
// Значения по умолчанию можно задать и окружением LHCTL_PORT / LHCTL_TOKEN —
// сгенерированные скрипты автодополнения разделяют их с самим lhctl.
// В терминале вывод — панели/таблицы termui; пайпы и --json/--raw получают
// машинный вывод прежнего формата.
#include "util.h"
#include "version.h"
#include "i18n.h"
#include "termui.h"
#include "fmt.h"
#include "sse.h"
#include "tui.h"

#include <httplib.h>
#include <nlohmann/json.hpp>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/select.h>
#include <strings.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <optional>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using nlohmann::json;

#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif

// Автодополнение для bash (Git Bash, WSL, Linux): имена моделей тянутся
// живьём из `lhctl models --raw` — построчно, без JSON, сервер выключен —
// TAB просто молчит (2>/dev/null).
static const char kBashCompletion[] = R"lh(
# lhctl tab-completion for bash (Git Bash, WSL, Linux).
# Install: source <(lhctl completion bash)  — or add this line to ~/.bashrc.
_lhctl_completions() {
  local cur prev opts models
  opts="status models show search rm chat ctx mtp image set load pull engine tokens unload quit completion help"
  COMPREPLY=()
  cur="${COMP_WORDS[COMP_CWORD]}"
  prev="${COMP_WORDS[COMP_CWORD-1]}"
  case "$prev" in
    load|unload|show|rm|chat|ctx|mtp)
      # --raw: one model id per line, no JSON to parse. Server down —
      # empty output, silent TAB.
      models=$(lhctl models --raw 2>/dev/null)
      COMPREPLY=($(compgen -W "$models" -- "$cur"))
      ;;
    pull)
      models="stop $(lhctl pull --raw 2>/dev/null)"
      COMPREPLY=($(compgen -W "$models" -- "$cur"))
      ;;
    image)
      models=$(lhctl image --raw 2>/dev/null)
      COMPREPLY=($(compgen -W "$models" -- "$cur"))
      ;;
    engine)
      COMPREPLY=($(compgen -W "install update" -- "$cur"))
      ;;
    completion)
      COMPREPLY=($(compgen -W "bash powershell" -- "$cur"))
      ;;
    *)
      if [[ $COMP_CWORD -eq 1 ]]; then
        COMPREPLY=($(compgen -W "$opts" -- "$cur"))
      fi
      ;;
  esac
  return 0
}
complete -o bashdefault -o default -F _lhctl_completions lhctl lhctl.exe
)lh";

// Автодополнение для Windows PowerShell / pwsh (Register-ArgumentCompleter);
// имена моделей — через `lhctl models --raw` (построчный вывод, никакого
// JSON-парсинга — одинаково в PS 5.1 и pwsh). Комментарии в скрипте —
// по-английски: PS 5.1 читает .ps1 без BOM в ANSI, и piped
// `| Add-Content $PROFILE` исказил бы кириллицу (коду это безвредно, но
// пусть профиль выглядит опрятно).
static const char kPsCompletion[] = R"lh(
# lhctl tab-completion for PowerShell. Install:
#   lhctl completion powershell | Add-Content $PROFILE
# then restart the console — or run the same piped command with
# Invoke-Expression to enable it in the current session only.
Register-ArgumentCompleter -Native -CommandName lhctl, lhctl.exe -ScriptBlock {
  param($wordToComplete, $commandAst, $cursorPosition)
  # Флаги и их значения (--port N, --token X) не участвуют в определении
  # подкоманды — иначе 'lhctl --port 7331 load <TAB>' не находит 'load'.
  $raw = @($commandAst.CommandElements | ForEach-Object { $_.ToString() })
  $elems = @()
  for ($i = 0; $i -lt $raw.Count; $i++) {
    if ($raw[$i] -match '^--(port|token)$') { $i++; continue }
    if ($raw[$i] -match '^-') { continue }
    $elems += $raw[$i]
  }
  $sub = if ($elems.Count -ge 2) { [string]$elems[1] } else { '' }
  $cmds = 'status','models','show','search','rm','chat','ctx','mtp','set','load','pull','engine','tokens','unload','quit','completion','help'
  function Add-Cand([string]$t) {
    if ($t -like "$wordToComplete*") {
      [System.Management.Automation.CompletionResult]::new($t, $t, 'ParameterValue', $t)
    }
  }
  if ($elems.Count -le 1) { $cmds | ForEach-Object { Add-Cand $_ }; return }
  switch ($sub) {
    { $_ -in 'load','unload','show','rm','chat','ctx','mtp','pull' } {
      if ($elems.Count -le 3) {
        try {
          # --raw: one id per line; for pull these are short install names.
          # stop — штатная остановка идущего скачивания, слово зарезервировано.
          if ($sub -eq 'pull') { Add-Cand 'stop' }
          $src = if ($sub -eq 'pull') { 'pull' } else { 'models' }
          (& lhctl $src --raw 2>$null) | ForEach-Object { Add-Cand $_ }
        } catch { }
      }
      break
    }
    'completion' {
      if ($elems.Count -le 3) { 'bash','powershell' | ForEach-Object { Add-Cand $_ } }
      break
    }
    'engine' {
      if ($elems.Count -le 3) { 'install','update' | ForEach-Object { Add-Cand $_ } }
      break
    }
    'image' {
      if ($elems.Count -le 3) {
        try { (& lhctl image --raw 2>$null) | Where-Object { $_ -notmatch ' ' } | ForEach-Object { Add-Cand $_ } } catch { }
      }
      break
    }
    default {
      if ($elems.Count -le 2) { $cmds | ForEach-Object { Add-Cand $_ } }
    }
  }
}
)lh";

// ---- lhctl chat: диалог/one-shot через OpenAI-прокси (аналог `ollama run`) --

namespace {

using namespace lh::fmt;  // human_bytes, state_cell, render_show, download_line…

bool g_vt = false; // терминал понимает ANSI — «мысли» модели приглушаем цветом

#ifndef _WIN32
// SIGWINCH в режиме watch: resize окна → немедленная перерисовка.
volatile sig_atomic_t g_winch = 0;
void on_winch(int) { g_winch = 1; }
#endif

void emit(const std::string& s) {
  fwrite(s.data(), 1, s.size(), stdout);
  fflush(stdout);
}

// ---- termui-рендеры: только для интерактивного терминала --------------------
// Всё, что ниже, вызывается под lh::ui::pretty(); пайпы и скрипты получают
// прежний машинный вывод, так что контракты команд не меняются.

// Сводка-«приборная панель»: состояние, карты с барами, модели.
std::string render_status(const json& j) {
  using namespace lh::ui;
  const int w = width();
  std::string state = j.value("state", "");
  bool busy = j.value("busy", false);
  Panel p("llmhost " + j.value("version", LH_VERSION));

  std::string st_txt;
  if (state == "loaded") st_txt = c::ok(tr("модели загружены, готов к запросам"));
  else if (state == "loading") st_txt = c::warn(tr("идёт загрузка модели…"));
  else if (state == "error") st_txt = c::err(tr("ошибка: ") + j.value("error", "?"));
  else st_txt = c::dim(tr("модели выгружены"));
  if (busy) st_txt += c::accent(tr("  ·  идёт генерация"));
  p.kv(tr("Состояние"), st_txt);

  p.kv(tr("Режим"), tr("harness: по llama-server на каждую модель"));
  int umin = j.value("unload_after_min", 0);
  int mmax = j.value("models_max", 0);
  std::string sleep_txt = umin > 0 ? tr("выгрузка после ") + std::to_string(umin) + tr(" мин простоя")
                                   : tr("выгрузка по простою выключена");
  if (mmax > 0) sleep_txt += c::dim(tr("  ·  резидентно до ") + std::to_string(mmax) + tr(" моделей"));
  p.kv(tr("Сон"), sleep_txt);
  p.hr();

  // Карты: бар заполнения VRAM на каждую.
  auto gpus = j.value("gpus", json::array());
  // Имя карты подгоняем под ширину: строка = индекс(2)+пробелы(4)+имя+бар(≥8)
  // + значение VRAM(16) и обязана влезать во внутреннее поле панели (w-4).
  int name_max = (w - 4) - 32;
  if (name_max < 8) name_max = 8;
  size_t name_w = 0;
  for (auto& g : gpus) name_w = std::max(name_w, dwidth(g.value("name", "?")));
  name_w = std::min({name_w, (size_t)name_max, (size_t)w / 2});
  int bw = w - 10 - (int)name_w - 18;
  if (bw > 34) bw = 34;  // длиннее — бар выглядит пустым
  if (bw < 8) bw = 8;
  for (auto& g : gpus) {
    uint64_t total = g.value("vram_total", (uint64_t)0),
             free = g.value("vram_free", (uint64_t)0);
    uint64_t used = total > free ? total - free : 0;
    double frac = total ? (double)used / (double)total : 0.0;
    p.line(" " + lh::ui::pad(std::to_string(g.value("index", 0)), 2) + " " +
           c::dim(lh::ui::pad(lh::ui::trunc(g.value("name", "?"), name_w), name_w)) + "  " +
           c::by_frac(frac, bar(frac, bw)) + "  " +
           lh::ui::padl(human_bytes(used) + c::dim("/" + human_bytes(total)), 16));
  }
  auto gpu = j.value("gpu", json::object());
  uint64_t ram_t = gpu.value("ram_total", (uint64_t)0),
           ram_a = gpu.value("ram_avail", (uint64_t)0);
  if (ram_t) {
    uint64_t used = ram_t > ram_a ? ram_t - ram_a : 0;
    double frac = (double)used / (double)ram_t;
    p.line(" " + lh::ui::pad(" ", 2) + " " + c::dim(lh::ui::pad("RAM", name_w)) + "  " +
           c::by_frac(frac, bar(frac, bw)) + "  " +
           lh::ui::padl(human_bytes(used) + c::dim("/" + human_bytes(ram_t)), 16));
  }
  std::string backend, cc = gpu.value("cc", "");
  if (gpu.value("nvidia", false))
    backend = "CUDA" + (cc.empty() ? "" : " (cc " + cc + ")");
  if (gpu.value("vulkan", false))
    backend += backend.empty() ? "Vulkan ✓" : "  ·  Vulkan ✓";
  if (!backend.empty()) p.line(" " + c::dim(lh::ui::pad(tr("Железо:"), name_w + 3)) + backend);
  // Что реально запускается: gpu.build — лишь «на что способно железо»
  // (vulkan там — «лоадер есть»), дети могут ехать на CUDA-сборке.
  auto eng = j.value("engine", json::object());
  std::string ebuild = eng.value("build", "");
  if (!ebuild.empty()) {
    std::string extra;
    std::string rel = eng.value("release", "");
    if (ebuild != "custom" && !rel.empty()) extra += " " + rel;
    extra += eng.value("selected", "").empty()
                 ? c::dim(tr(" · по железу"))
                 : c::dim(tr(" · выбрана: ") + eng.value("selected", ""));
    p.line(" " + c::dim(lh::ui::pad(tr("Движок:"), name_w + 3)) + c::accent(ebuild) + extra);
  }

  // Основная модель (классический режим / последняя используемая).
  auto prim = j.value("model", json::object());
  if (!prim.value("file", "").empty()) {
    p.hr();
    p.kv(tr("Основная"), prim.value("file", "?"));
  }

  auto models = j.value("models", json::array());
  if (!models.empty()) {
    p.hr();
    lh::ui::Table t;
    t.col(tr("МОДЕЛЬ")).col(tr("СОСТОЯНИЕ")).col(tr("РОЛЬ"), lh::ui::Table::Left, 10);
    struct Row { int rank; std::vector<std::string> cells; };
    std::vector<Row> rows;
    for (auto& m : models) {
      std::string id = m.value("id", "");
      if (id.empty()) continue;
      std::string role = m.value("embedding", false) ? c::dim(tr("эмбеддер")) : "";
      // Ранг — из состояния ДО рендера: текст ячейки зависит от языка
      // (перевод не содержит русских подстрок — искать в нём нельзя).
      std::string st = m.value("state", "");
      if (st.empty() && m.value("loaded", false)) st = "loaded";
      int rank = st == "loaded" ? 0 : st == "loading" ? 1 : 2;
      rows.push_back({rank,
                      {id, state_cell(m.value("state", ""), m.value("failed", false), false),
                       role}});
    }
    std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) {
      return std::make_pair(a.rank, a.cells[0]) < std::make_pair(b.rank, b.cells[0]);
    });
    for (auto& r : rows) t.row(std::move(r.cells));
    panel_table(p, t);
  }
  // Загруженные слоты — подробно: что реально поехало движку (args ребёнка).
  bool details = false;
  for (auto& m : models) {
    if (m.value("state", "") != "loaded" || !m.contains("args")) continue;
    if (!details) {
      p.hr();
      details = true;
    }
    p.kv(tr("Слот ") + m.value("id", "?"),
         tr("порт ") + std::to_string(m.value("port", 0)) +
             (m.value("pid", 0) ? c::dim("  ·  pid " + std::to_string(m.value("pid", 0)))
                                : ""));
    for (auto& kv : slot_args_pairs(m.value("args", json::array())))
      p.line(" " + c::dim(pad(kv.first + ":", 26)) + kv.second);
  }
  return p.render(w);
}

// Реестр моделей: таблица с планом VRAM от планировщика.
std::string render_models(const json& j) {
  using namespace lh::ui;
  Table t;
  t.col(tr("МОДЕЛЬ")).col(tr("СОСТОЯНИЕ")).col(tr("КАРТА"), Table::Right).col(tr("РАЗМЕР"), Table::Right)
      .col(tr("КВАНТ")).col(tr("ПЛАН CTX"), Table::Right).col(tr("VRAM-ОЦЕНКА"), Table::Right);
  struct Row { int rank; std::vector<std::string> cells; };
  std::vector<Row> rows;
  for (auto& m : j.value("models", json::array())) {
    std::string id = m.value("id", "");
    if (id.empty()) continue;
    std::string st = m.value("state", "");
    int rank = st == "loaded" ? 0 : st == "loading" ? 1 : 2;
    if (st.empty() && m.value("loaded", false)) rank = 0;
    int gpu_i = m.value("gpu", -1);
    auto fit = m.value("fit", json::object());
    std::string plan = "—", vram = "—";
    if (!fit.is_null()) {
      uint64_t ctx = fit.value("ctx", (uint64_t)0), mctx = fit.value("max_ctx", (uint64_t)0);
      if (ctx) plan = mctx ? human_ctx(ctx) + c::dim("/" + human_ctx(mctx)) : human_ctx(ctx);
      uint64_t need = fit.value("vram_needed", (uint64_t)0);
      if (need) vram = fit.value("fits", false) ? human_bytes(need)
                                                : c::warn(human_bytes(need));
    }
    rows.push_back({rank,
                    {id, state_cell(st, m.value("failed", false), m.value("loaded", false)),
                     gpu_i >= 0 ? std::to_string(gpu_i) : "—",
                     human_bytes(m.value("size", (uint64_t)0)),
                     m.value("quant", "—"), plan, vram}});
  }
  std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) {
    return std::make_pair(a.rank, a.cells[0]) < std::make_pair(b.rank, b.cells[0]);
  });
  for (auto& r : rows) t.row(std::move(r.cells));
  return t.render();
}

// Короткие имена для pull.
std::string render_aliases(const json& j) {
  using namespace lh::ui;
  Table t;
  t.col(tr("ИМЯ")).col(tr("РЕПОЗИТОРИЙ"), Table::Left, 48).col(tr("ПРИМЕЧАНИЕ"), Table::Left, 44);
  for (auto& a : j.value("aliases", json::array()))
    t.row({c::accent(a.value("name", "")), a.value("repo", ""), c::dim(a.value("note", ""))});
  return t.render();
}

// Результаты поиска по каталогу HuggingFace.
std::string render_search(const json& j, const std::string& query) {
  using namespace lh::ui;
  Panel p(tr("Поиск: «") + query + "»");
  auto gpu = j.value("gpu", json::object());
  if (!gpu.empty())
    p.kv(tr("Железо"), gpu.value("name", "?") + c::dim(tr("  ·  свободно ")) +
                       human_bytes(gpu.value("vram_free", (uint64_t)0)) +
                       c::dim(tr(" из ") + human_bytes(gpu.value("vram_total", (uint64_t)0))));
  p.hr();
  Table t;
  t.col(tr("РЕПОЗИТОРИЙ")).col(tr("РАЗМЕР"), Table::Right).col("VRAM", Table::Right)
      .col(tr("ПОМЕЩАЕТСЯ")).col(tr("СКАЧИВАНИЙ"), Table::Right).col("★", Table::Right);
  for (auto& r : j.value("results", json::array())) {
    auto f = r.value("fit", json::object());
    std::string verdict;
    if (f.value("fits", false)) verdict = c::ok(tr("✓ целиком"));
    else if (f.value("partial", false)) verdict = c::warn(tr("частично"));
    else if (f.value("dense_partial", false)) verdict = c::warn(tr("частично (плотная)"));
    else if (!f.value("gpu_known", true)) verdict = c::dim(tr("неизвестно"));
    else verdict = c::err(tr("✗ не влезает"));
    uint64_t sz = f.value("size_est", (uint64_t)0), vneed = f.value("vram_needed", (uint64_t)0);
    if (!sz && !vneed) verdict = c::dim(tr("нет оценки"));  // метаданных не было
    t.row({r.value("repo", "?"),
           sz ? human_bytes(sz) : "—",
           vneed ? human_bytes(vneed) : "—",
           verdict, human_count(r.value("downloads", (uint64_t)0)),
           std::to_string(r.value("likes", 0))});
  }
  panel_table(p, t);
  p.line("  " + c::dim(tr("установить: lhctl pull <org/repo>")));
  return p.render();
}

// Запись из /v0/models по id (роутер), general.name или стему файла — как /v0/load.
// Найти запись модели в /v0/models: id, general.name, стем файла или полный
// путь. Единый матчинг для show/rm/ctx/mtp/load/chat — раньше inline-копии в
// show/rm/ctx/mtp не матчили полный путь, и «lhctl rm <путь>» вёл себя иначе,
// чем «lhctl load <путь>».
std::optional<json> find_model(const std::string& body, const std::string& want) {
  try {
    auto doc = json::parse(body);
    for (auto& m : doc.value("models", json::array())) {
      std::string p = m.value("path", "");
      std::string file = p;
      size_t slash = file.find_last_of("/\\");
      if (slash != std::string::npos) file = file.substr(slash + 1);
      size_t dot = file.find_last_of('.');
      if (dot != std::string::npos) file = file.substr(0, dot);
      if (m.value("id", "") != want && m.value("name", "") != want &&
          strcasecmp(file.c_str(), want.c_str()) != 0 && p != want)
        continue;
      return m;
    }
  } catch (...) {}
  return std::nullopt;
}

bool find_model_entry(const std::string& body, const std::string& want,
                      std::string& path_out, std::string& state_out) {
  auto m = find_model(body, want);
  if (!m) return false;
  path_out = m->value("path", "");
  std::string st = m->value("state", "");
  if (st.empty() && m->value("loaded", false)) st = "loaded";
  state_out = st;
  return true;
}

// Опрос до готовности — общий для ensure_model и «lhctl load <файл>»
// (раньше цикл был в двух экземплярах с расхождениями: версия load видела
// состояние error из /v0/status, версия ensure_model — нет). Спиннер в tty,
// точки в пайпе; 15 минут потолок; «unloaded» после 6 с — честный отказ
// (первые секунды unloaded — норма, пока ребёнок не стартовал). Для файлов,
// которых ещё нет в реестре, и классики — состояния error/loaded из
// /v0/status.
static int wait_loaded(int port, httplib::Client& cli, const httplib::Headers& hdrs,
                       const std::string& want) {
  const bool tty = lh::ui::pretty();
  auto tick = [&](unsigned long sec) {
    if (tty)
      std::fprintf(stderr, tr("\r\x1b[2K%s загружаю «%s»  ·  %lu с"),
                   lh::ui::spinner(), want.substr(0, 60).c_str(), sec);
    else
      std::fputc('.', stderr);
    std::fflush(stderr);
  };
  tick(0);
  auto t0 = std::chrono::steady_clock::now();
  for (;;) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    unsigned long sec =
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - t0)
            .count();
    tick(sec);
    if (sec > 900) {
      std::fprintf(stderr, "%s\n", tr("\nзагрузка идёт дольше 15 минут — сдаюсь"));
      return 3;
    }
    auto st = cli.Get("/v0/models", hdrs);
    if (st && st->status == 200) {
      std::string p2, s2;
      if (find_model_entry(st->body, want, p2, s2)) {
        if (s2 == "loaded") {
          if (tty)
            std::fprintf(stderr, "\r\x1b[2K%s%s\n",
                         lh::ui::c::ok("✓").c_str(),
                         (tr(" загружена: «") + want.substr(0, 60) + tr("»  ·  ") +
                          std::to_string(sec) + tr(" с")).c_str());
          else
            std::fprintf(stderr, "%s\n", tr(" готово"));
          return 0;
        }
        if (s2 == "unloaded" && sec > 6) {
          std::fprintf(stderr, "%s\n",
                       tr("\r\x1b[2K\nмодель не загрузилась — lhctl status покажет причину"));
          return 3;
        }
      }
    }
    auto cs = cli.Get("/v0/status", hdrs);
    if (cs && cs->status == 200) {
      try {
        auto j = json::parse(cs->body);
        if (j.value("state", "") == "error") {
          std::fprintf(stderr, "\n%s\n", j.value("error", tr("модель не загрузилась")).c_str());
          return 3;
        }
        if (j.value("state", "") == "loaded" &&
            j.value("model", json::object()).value("path", "") == want) {
          if (tty)
            std::fprintf(stderr, "\r\x1b[2K%s%s\n", lh::ui::c::ok("✓").c_str(),
                         (tr(" загружена  ·  ") + std::to_string(sec) + tr(" с")).c_str());
          else
            std::fprintf(stderr, "%s\n", tr(" готово"));
          return 0;
        }
      } catch (...) {}
    }
  }
}

// Догрузить модель, если не резидентна: /v0/load + опрос состояния. В tty —
// перерисовываемая строка со спиннером и секундами (ширина перечитывается —
// resize окна не ломает строку), в пайпе — прежние точки.
// Коды выхода — по семантике lhctl: 0 готова, 1 сервер, 2 токен, 3 запрос, 4 аргументы.
int ensure_model(int port, httplib::Client& cli, const httplib::Headers& hdrs,
                 const std::string& want) {
  auto res = cli.Get("/v0/models", hdrs);
  if (!res) {
    std::fprintf(stderr, tr("llmhost недоступен на 127.0.0.1:%d\n"), port);
    return 1;
  }
  if (res->status == 401) {
    std::fprintf(stderr, "%s", tr("lhctl: неверный токен (401) — см. data/token.txt\n"));
    return 2;
  }
  std::string path, state;
  if (!find_model_entry(res->body, want, path, state)) {
    std::fprintf(stderr, tr("модель «%s» не найдена — список: lhctl models\n"), want.c_str());
    return 4;
  }
  if (state == "loaded") return 0;

  auto r = cli.Post("/v0/load", hdrs, json{{"path", path}}.dump(), "application/json");
  if (!r || r->status >= 400) {
    std::fprintf(stderr, "%s\n",
                 (tr("загрузка не началась: ") +
                  (r ? error_message(r->body) : tr("нет ответа"))).c_str());
    return 3;
  }
  return wait_loaded(port, cli, hdrs, want);
}

// Один ход: вопрос → SSE-стрим. Ответ печатается сразу по мере генерации;
// размышления (reasoning_content) показываются приглушённо (show_reasoning).
// true — успех, история пополнена.
bool chat_turn(httplib::Client& cli, const httplib::Headers& hdrs,
               const std::string& model, json& history, const std::string& question,
               bool show_reasoning) {
  json msgs = json::array();
  for (auto& m : history) msgs.push_back(m);
  msgs.push_back({{"role", "user"}, {"content", question}});
  json body = {{"messages", msgs}, {"stream", true}};
  if (!model.empty()) body["model"] = model;

  std::string answer, reason;
  bool done = false, sep = false, marker = false;
  lh::sse::Chat sse;
  sse.on_reasoning = [&](const std::string& r2) {
    reason += r2;
    if (show_reasoning) {
      if (g_vt) emit("\x1b[2m" + r2 + "\x1b[0m");
      else if (!marker) { emit(tr("(думаю…)\n")); marker = true; }
    }
  };
  sse.on_content = [&](const std::string& c) {
    if (!sep && !reason.empty() && show_reasoning) { emit("\n"); sep = true; }
    answer += c;
    emit(c);
  };
  httplib::Request r; // send()+content_receiver: Post с приёмником в этой
  r.method = "POST";  // версии httplib перегрузки не имеет — паттерн прокси
  r.path = "/v1/chat/completions";
  r.body = body.dump();
  for (auto& h : hdrs) r.headers.emplace(h.first, h.second);
  r.headers.emplace("Content-Type", "application/json");
  r.content_receiver = [&](const char* d, size_t n, uint64_t, uint64_t) -> bool {
    if (done) return true;
    if (!sse.feed(d, n)) done = true; // [DONE]; соединение рвать не нужно
    return true;
  };
  auto res = cli.send(r);

  if (!res) {
    std::fprintf(stderr, "%s\n",
                 (tr("ошибка запроса: ") + httplib::to_string(res.error())).c_str());
    return false;
  }
  if (res->status != 200) {
    std::fprintf(stderr, "%s\n", error_message(sse.raw_tail()).c_str());
    return false;
  }
  if (answer.empty() && reason.empty()) {
    std::fprintf(stderr, "%s\n", tr("пустой ответ модели"));
    return false;
  }
  history.push_back({{"role", "user"}, {"content", question}});
  history.push_back({{"role", "assistant"},
                     {"content", answer.empty() ? reason : answer}});
  return true;
}

// ---- Многострочная вставка в чате -------------------------------------------
// Цикл чата — «строка → запрос → генерация», и вставка ломала его дважды:
// терминал кладёт все строки вставки в буфер разом, каждая становилась
// ОТДЕЛЬНЫМ запросом (модель отвечала на каждую строку, 2-3 ответа подряд),
// а набранное во время генерации — вплоть до /bye — молча ждало очереди за
// ними. Пока stdin — терминал и данные уже готовы, дочитываем их в ТО ЖЕ
// сообщение: вставка = один запрос. Пайпы не склеиваем — это построчный
// протокол для скриптов. Быстрый путь (одна строка, ввода больше нет) —
// без единой лишней миллисекунды.
#ifndef _WIN32
bool input_pending() {
  fd_set rfds;
  FD_ZERO(&rfds);
  FD_SET(0, &rfds);
  timeval tv{0, 0};
  return select(1, &rfds, nullptr, nullptr, &tv) > 0;
}

// Окно 150 мс между строками: буфер tty ~4 КиБ, большая вставка приходит
// кусками — пауза между ними ещё не конец вставки. Ctrl-D посреди вставки —
// «отослать как есть»; строка /bye внутри буфера — не текст, а команда:
// дошлило текущее сообщение, после ответа выходим.
std::string drain_paste(std::string msg, bool& quit_after) {
  for (;;) {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(0, &rfds);
    timeval tv{0, 150000};
    if (select(1, &rfds, nullptr, nullptr, &tv) <= 0) break;
    std::string line;
    if (!std::getline(std::cin, line)) break;
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
    if (line == "/bye" || line == "/exit" || line == "/quit") {
      quit_after = true;
      break;
    }
    msg += "\n" + line;
  }
  return msg;
}
#endif

int chat_run(int port, const std::string& token, std::string model,
             const std::string& question) {
  if (model == "any") model.clear(); // «any» = по умолчанию, как в поле model

  // Управление (load/models) — на control-порту; сам диалог — на OpenAI-порту,
  // его узнаём из /v0/credentials (флаг --port задаёт именно control).
  httplib::Client ctrl("127.0.0.1", port);
  ctrl.set_connection_timeout(std::chrono::seconds(5));
  ctrl.set_read_timeout(std::chrono::seconds(30));
  httplib::Headers chdrs;
  if (!token.empty()) chdrs = {{"Authorization", "Bearer " + token}};
  auto cred = ctrl.Get("/v0/credentials", chdrs);
  if (!cred) {
    std::fprintf(stderr, tr("llmhost недоступен на 127.0.0.1:%d\n"), port);
    return 1;
  }
  if (cred->status == 401) {
    std::fprintf(stderr, "%s", tr("lhctl: неверный токен (401) — см. data/token.txt\n"));
    return 2;
  }
  int openai_port = 0;
  std::string key = token; // приоритет у явного токена
  try {
    auto j = json::parse(cred->body);
    std::string url = j.value("openai_base_url", ""); // http://127.0.0.1:7330/v1
    size_t colon = url.rfind(':');
    if (colon != std::string::npos) openai_port = std::atoi(url.c_str() + colon + 1);
    if (key.empty()) key = j.value("api_key", std::string());
  } catch (...) {}
  if (openai_port <= 0) {
    std::fprintf(stderr, "%s\n", tr("не удалось узнать OpenAI-порт у llmhost"));
    return 3;
  }

  httplib::Client cli("127.0.0.1", openai_port);
  cli.set_connection_timeout(std::chrono::seconds(5));
  cli.set_read_timeout(std::chrono::seconds(300)); // генерация думает минутами
  cli.set_write_timeout(std::chrono::seconds(30));
  httplib::Headers hdrs;
  if (!key.empty()) hdrs = {{"Authorization", "Bearer " + key}};

  if (!model.empty()) {
    int rc = ensure_model(port, ctrl, chdrs, model);
    if (rc != 0) return rc;
  }

  if (!question.empty()) {
    // One-shot: чистый ответ в stdout — для пайпов и скриптов; мысли скрыты.
    json history = json::array();
    return chat_turn(cli, hdrs, model, history, question, /*show_reasoning=*/false)
               ? 0
               : 3;
  }

#ifdef _WIN32
  HANDLE hout = GetStdHandle(STD_OUTPUT_HANDLE);
  DWORD cm = 0;
  if (GetConsoleMode(hout, &cm))
    g_vt = SetConsoleMode(hout, cm | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0;
#else
  // POSIX: ANSI-последовательности терминал понимает всегда, но не в пайпе
  // и не при NO_COLOR/TERM=dumb — критерий тот же, что в lh::ui::init
  // (раньше флаг здесь не ставился вовсе, и «мысли» теряли приглушение).
  g_vt = lh::ui::color();
#endif

  std::string title = model.empty()
                          ? tr("Чат с llmhost — модель по умолчанию (последняя используемая)")
                          : tr("Чат с llmhost — модель: ") + model;
  std::fprintf(stderr, "%s\n%s",
               title.c_str(),
               tr("/bye — выход, /clear — новая сессия, Ctrl+D — выход.\n"
                  "Многострочная вставка уходит одним сообщением.\n"));

  json history = json::array();
  std::string line;
  for (;;) {
    std::fputs(tr("\nВы> "), stdout);
    std::fflush(stdout);
    if (!std::getline(std::cin, line)) break; // EOF: Ctrl+D или конец пайпа
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
    if (line.empty()) continue;
    if (line == "/bye" || line == "/exit" || line == "/quit") break;
    if (line == "/clear") {
      history = json::array();
      std::fputs(tr("(сессия сброшена)\n"), stdout);
      continue;
    }
    bool quit_after = false;
#ifndef _WIN32
    if (isatty(0) && input_pending()) line = drain_paste(std::move(line), quit_after);
#endif
    std::fputc('\n', stdout);
    chat_turn(cli, hdrs, model, history, line, /*show_reasoning=*/true);
    if (quit_after) break; // /bye, набранный во время генерации: текущее — дошлём, потом выходим
  }
  return 0;
}

} // namespace

static void usage() {
  std::puts(tr(
      "lhctl — управление llmhost из консоли\n"
      "  lhctl                    полноэкранный интерфейс управления (в духе mc:\n"
      "                           реестр с курсором, загрузка/выгрузка, поиск,\n"
      "                           прогресс; F1 — помощь) — только в терминале\n"
      "  lhctl status [-w сек]    приборная панель (загруженные модели, VRAM-бары);\n"
      "                           -w — обновлять живьём (resize окна не ломает)\n"
      "  lhctl models [--raw]     список моделей реестра (--raw — id построчно)\n"
      "  lhctl show <модель>      карточка модели (архитектура, квант, контекст)\n"
      "  lhctl ctx <модель> [N|auto]\n"
      "                           ручной контекст модели: пишет ctx-size в\n"
      "                           llmhost.conf и пересобирает пресет сразу;\n"
      "                           без значения — показать текущее\n"
      "  lhctl mtp <модель> [auto|on|off]\n"
      "                           MTP-спекуляция модели (дополнение токенов\n"
      "                           черновиком): auto — как решает умная настройка,\n"
      "                           on/off — форс; без значения — показать текущее\n"
      "  lhctl vision <модель> [auto|off]\n"
      "                           vision-режим модели (картинки через mmproj):\n"
      "                           auto — своя проекция прицепится сама, off —\n"
      "                           грузить без неё; без значения — показать текущее\n"
        "  lhctl thinking <модель> [auto|on|off]\n"
        "                           thinking-режим модели (рассуждения перед\n"
        "                           ответом): auto — решает движок по шаблону,\n"
        "                           on/off — форс; без значения — показать текущее\n"
        "  lhctl metrics [-w сек]\n"
        "                           точные метрики движка по-модельно (скорости\n"
        "                           декода и префилла, приём спекции, токены):\n"
        "                           без -w — один срез, с -w — живая таблица\n"
        "                           со спарклайном мгновенной скорости\n"
      "  lhctl image [load [модель] | pull [имя] | rm <модель> | unload]\n"
      "                           генерация изображений (движок sd-server):\n"
      "                           без аргумента — статус и каталог моделей,\n"
      "                           load — загрузить (к модели обращается\n"
      "                           /v1/images/* OpenAI-порта), unload — остановить,\n"
      "                           pull — скачать художника (без имени — каталог\n"
      "                           встроенных; или org/repo с раскладкой\n"
      "                           recipes/ + категории), rm — удалить с диска\n"
      "  lhctl set [ключ [значение]]\n"
      "                           настройки [general] llmhost.conf (единый\n"
      "                           конфиг): без аргументов — вся таблица,\n"
      "                           с аргументами — записать; применяется на\n"
      "                           лету (порты — после рестарта демона)\n"
      "  lhctl search <запрос>    поиск по каталогу HuggingFace: размер, оценка\n"
      "                           VRAM, влезет ли на ваши карты\n"
      "  lhctl tokens [list]      именованные токены /v1 (значение — в колонке)\n"
      "  lhctl tokens add <имя>   выдать токен (значение генерируется и\n"
      "                           сохраняется; пока токены есть, /v1 — только по ним)\n"
      "  lhctl tokens rm <имя>    отозвать токен\n"
      "  lhctl rm <модель>        удалить модель С ДИСКА безвозвратно: вместе\n"
      "                           с mmproj/MTP и шардами, скачанными с ней\n"
      "                           (спросит подтверждение, --yes — без вопроса)\n"
      "  lhctl chat [модель] [вопрос]\n"
      "                           чат: без вопроса — диалог со стримом (/bye —\n"
      "                           выход); с вопросом — один ответ; без модели —\n"
      "                           последняя используемая\n"
      "  lhctl load <файл|имя>    загрузить модель (.gguf или имя из реестра)\n"
      "  lhctl pull [имя] [файл]  скачать модель: короткое имя (без аргумента —\n"
      "                           список; квант подберётся под вашу карту) или\n"
      "                           org/repo; файл — имя или wildcard-фильтр\n"
      "  lhctl pull stop          остановить идущее скачивание (недокачанный\n"
      "                           .part остаётся — повторный pull докачит)\n"
      "  lhctl engine             движок: пин манифеста, сборки, что ставится\n"
      "  lhctl engine install <сборка>\n"
      "                           установить сборку из текущего пина (скачивание\n"
      "                           с проверкой SHA256; модели должны быть\n"
      "                           выгружены)\n"
      "  lhctl engine update [сборка]\n"
      "                           обновить движок: пин последнего релиза\n"
      "                           llama.cpp и установка активной сборки\n"
      "                           (--release bNNNNN — конкретный релиз,\n"
      "                           --pin-only — только манифест без установки,\n"
      "                           --all — все пиннутые сборки)\n"

      "  lhctl unload [id]        выгрузить одну (id) или все\n"
      "                           (--force — занятую принудительно:\n"
      "                           обрывает все генерации модели)\n"
      "  lhctl quit               завершить llmhost\n"
      "  lhctl version            версия\n"
      "  lhctl completion bash|powershell\n"
      "                           скрипт автодополнения (TAB подсказывает модели)\n"
      "Опции: --port N (7331), --token XXX (иначе data/token.txt рядом с lhctl\n"
      "или $XDG_DATA_HOME/llmhost/token.txt), --json — машинный вывод даже в\n"
      "терминале.\n"
      "Окружение LHCTL_PORT / LHCTL_TOKEN задаёт значения по умолчанию.\n"
      "Коды выхода: 0 — успех; 1 — сервер недоступен; 2 — неверный токен (401);\n"
      "3 — ошибка запроса (404/500); 4 — ошибка аргументов."));
}

// ---- контекст команды: общий бойлерплейс main() без протаскивания 10 аргументов
  struct CmdCtx {
    httplib::Client& cli;
    httplib::Headers hdrs;
    int port = 7331;
    std::string token, arg, arg2, release_tag;
    bool raw = false, yes = false, pin_only = false, all_builds = false;
    bool force = false;
    bool watch = false;
    httplib::Result res; // ответ «хвостовых» команд — печатают сырой JSON

  // 0 — ответ в порядке; 1 — сервер недоступен; 2 — неверный токен.
  // Раньше эти три строки были отрукописаны в каждой команде.
  int gate() {
    if (!res) {
      std::fprintf(stderr, tr("llmhost недоступен на 127.0.0.1:%d (%s)\n"), port,
                   httplib::to_string(res.error()).c_str());
      return 1;
    }
    if (res->status == 401) {
      std::fprintf(stderr, "%s", tr("lhctl: неверный токен (401) — см. data/token.txt\n"));
      return 2;
    }
    return 0;
  }
};

// Сентинел run_*-функций: ответ в c.res — печать сырого JSON общим хвостом.
constexpr int kTail = -1;

// Общий хвост не-tty путей (status/models/quit/tokens/search/load/unload):
// печать JSON-тела как есть. Ошибки уже отфильтрованы.
static int print_raw(CmdCtx& c) {
  if (!c.res) {
    std::fprintf(stderr, tr("llmhost недоступен на 127.0.0.1:%d (%s)\n"), c.port,
                 httplib::to_string(c.res.error()).c_str());
    return 1;
  }
  if (c.res->status == 401) {
    std::fprintf(stderr, "%s", tr("lhctl: неверный токен (401) — см. data/token.txt\n"));
    return 2;
  }
  if (c.res->status < 200 || c.res->status >= 300) {
    // Тело ошибки — в stderr: stdout остаётся для успешных ответов.
    std::fprintf(stderr, "%s\n", c.res->body.c_str());
    return 3;
  }
  try {
    std::puts(json::parse(c.res->body).dump(2).c_str());
  } catch (...) {
    std::puts(c.res->body.c_str()); // не JSON — печатаем как есть
  }
  return 0;
}

static int run_status(CmdCtx& c) {
  if (lh::ui::pretty() && c.watch) {
    // Живая приборная панель: перерисовка раз в interval секунд, SIGWINCH
    // (resize окна) будит немедленно. Выход — Ctrl+C.
    int interval = 2;
    if (!c.arg.empty()) {
      try { interval = std::stoi(c.arg); } catch (...) {}
    }
    if (interval < 1) interval = 1;
#ifndef _WIN32
    struct sigaction sa{};
    sa.sa_handler = on_winch;
    sigaction(SIGWINCH, &sa, nullptr);
#endif
    lh::ui::clear_screen();
    for (;;) {
      auto r = c.cli.Get("/v0/status", c.hdrs);
      if (r && r->status == 200) {
        try {
          std::fputs(render_status(json::parse(r->body)).c_str(), stdout);
        } catch (...) {}
      } else {
        lh::ui::Panel p("llmhost");
        p.kv(tr("Состояние"), lh::ui::c::err(tr("демон не отвечает на 127.0.0.1:") +
                                             std::to_string(c.port)));
        std::fputs(p.render().c_str(), stdout);
      }
      std::fflush(stdout);
      for (int slept = 0; slept < interval * 5; ++slept) {
#ifndef _WIN32
        if (g_winch) break;
#endif
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
      }
#ifndef _WIN32
      g_winch = 0;
#endif
      lh::ui::clear_screen();
    }
  }
  c.res = c.cli.Get("/v0/status", c.hdrs);
  if (!c.watch && lh::ui::pretty() && c.res && c.res->status >= 200 && c.res->status < 300) {
    try {
      std::fputs(render_status(json::parse(c.res->body)).c_str(), stdout);
      return 0;
    } catch (...) {}
  }

  return kTail; // сырой JSON — общий хвост
}
static int run_models(CmdCtx& c) {
  c.res = c.cli.Get("/v0/models", c.hdrs);
  if (!c.raw && c.res && c.res->status >= 200 && c.res->status < 300) {
    if (lh::ui::pretty()) {
      try {
        std::fputs(render_models(json::parse(c.res->body)).c_str(), stdout);
        return 0;
      } catch (...) { /* упадём в старый вывод ниже */ }
    }
    // Человекочитаемый вывод (пайпы): id, состояние и карта по раскладке.
    try {
      auto j = json::parse(c.res->body);
      for (auto& m : j.value("models", json::array())) {
        std::string id = m.value("id", std::string());
        if (id.empty()) continue;
        std::string line = id;
        std::string st = m.value("state", "");
        if (st == "loaded") line += tr("  (в памяти)");
        else if (st == "loading") line += tr("  (грузится…)");
        else if (m.value("failed", false)) line += tr("  (не загрузилась)");
        int gpu = m.value("gpu", -1);
        if (gpu >= 0) line += tr("  [карта ") + std::to_string(gpu) + "]";
        std::puts(line.c_str());
      }
    } catch (...) { std::puts(c.res->body.c_str()); }
    return 0;
  }
  if (c.raw) {
    // Дисциплина для completer-скриптов: stdout — только идентификаторы,
    // любые ошибки — в stderr (их прячут 2>/dev/null и 2>$null).
    if (!c.res) {
      std::fprintf(stderr, tr("lhctl: сервер недоступен на 127.0.0.1:%d (%s)\n"), c.port,
                   httplib::to_string(c.res.error()).c_str());
      return 1;
    }
    if (c.res->status < 200 || c.res->status >= 300) {
      if (c.res->status == 401) {
        std::fprintf(stderr, "%s", tr("lhctl: неверный токен (401) — см. data/token.txt\n"));
        return 2;
      }
      std::fprintf(stderr, tr("lhctl: сервер ответил %d\n"), c.res->status);
      return 3;
    }
    try {
      auto j = json::parse(c.res->body);
      // Сначала загруженные, затем «грузящиеся», остальные; по алфавиту внутри.
      std::vector<std::pair<int, std::string>> rows;
      for (auto& m : j.value("models", json::array())) {
        std::string id = m.value("id", std::string());
        if (id.empty()) {
          // Классический режим: id нет — имя файла без .gguf.
          std::string p = m.value("path", "");
          size_t slash = p.find_last_of("/\\");
          id = slash == std::string::npos ? p : p.substr(slash + 1);
          if (id.size() > 5 && strcasecmp(id.c_str() + id.size() - 5, ".gguf") == 0)
            id.erase(id.size() - 5);
        }
        if (id.empty()) continue;
        std::string st = m.value("state", "");
        int rank = st == "loaded" ? 0 : st == "loading" ? 1 : 2;
        if (st.empty() && m.value("loaded", false)) rank = 0; // классический режим
        rows.push_back({rank, std::move(id)});
      }
      std::sort(rows.begin(), rows.end());
      for (auto& [rank, id] : rows) std::puts(id.c_str());
    } catch (...) { /* не JSON — печатать нечего */ }
    return 0;
  }

  return kTail; // сырой JSON — общий хвост
}
static int run_quit(CmdCtx& c) {
  c.res = c.cli.Post("/v0/quit", c.hdrs, "{}", "application/json");

  return kTail; // сырой JSON — общий хвост
}
static int run_completion(CmdCtx& c) {
  if (c.arg == "bash") {
    std::fputs(kBashCompletion, stdout);
    return 0;
  }
  if (c.arg == "powershell") {
    std::fputs(kPsCompletion, stdout);
    return 0;
  }
  std::fprintf(stderr, "%s", tr("completion: укажите bash или powershell\n"));
  return 4;

  return kTail; // недостижимо: все пути выше возвращают код
}
static int run_show(CmdCtx& c) {
  if (c.arg.empty()) {
    std::fprintf(stderr, "%s", tr("show: укажите модель (lhctl models — список)\n"));
    return 4;
  }
  c.res = c.cli.Get("/v0/models", c.hdrs);
  if (c.res && c.res->status == 200) {
    auto m = find_model(c.res->body, c.arg);
    if (!m) {
      std::fprintf(stderr, tr("модель «%s» не найдена — lhctl models покажет список\n"),
                   c.arg.c_str());
      return 3;
    }
    {
      if (lh::ui::pretty()) {
        std::fputs(render_show(*m).c_str(), stdout);
        return 0;
      }
      double gb = m->value("size", (uint64_t)0) / 1073741824.0;
      std::printf("%s%s\n", tr("модель:    "), m->value("id", m->value("name", "?")).c_str());
      std::printf("%s%s\n", tr("файл:      "), m->value("path", "").c_str());
      std::printf(tr("размер:    %.1f ГБ\n"), gb);
      std::printf(tr("архитектура: %s, квант %s\n"), m->value("arch", "?").c_str(),
                  m->value("quant", "?").c_str());
      if (m->value("n_layers", 0)) std::printf(tr("слоёв:     %d\n"), m->value("n_layers", 0));
      if (m->value("n_ctx_train", (uint64_t)0))
        std::printf(tr("контекст обучения: %llu токенов\n"),
                    (unsigned long long)m->value("n_ctx_train", (uint64_t)0));
      std::string st = m->value("state", "");
      if (!st.empty()) {
        std::printf("%s%s", tr("состояние: "), st.c_str());
        int gpu = m->value("gpu", -1);
        if (gpu >= 0) std::printf(tr(" (карта %d)"), gpu);
        std::printf("\n");
      } else if (m->value("loaded", false)) {
        std::printf("%s\n", tr("состояние: загружена"));
      }
      if (m->value("embedding", false)) std::printf("%s\n", tr("эмбеддер:  да (/v1/embeddings)"));
      {
        const std::string vo = m->value("vision_mode", "auto");
        const std::string vm = m->value("mmproj", "");
        if (vo == "off")
          std::printf("%s\n", tr("vision:     off (переопределение) — без --mmproj"));
        else if (!vm.empty())
          std::printf("%s\n",
                      (tr("vision:     auto — «") +
                       vm.substr(vm.find_last_of("/\\") + 1) + tr("» прицепится сама")).c_str());
        else
          std::printf("%s\n", tr("vision:     auto — проекций рядом нет"));
      }
      {
        const std::string to = m->value("think_override", "");
        if (!to.empty())
          std::printf(tr("thinking:   %s (переопределение)\n"), to.c_str());
        else
          std::printf("%s\n", tr("thinking:   auto — решает движок по шаблону"));
      }
      return 0;
    }
  }
  // ошибки сервера — общим хвостом ниже

  return kTail; // недостижимо: все пути выше возвращают код
}
static int run_rm(CmdCtx& c) {
  if (c.arg.empty()) {
    std::fprintf(stderr, "%s", tr("rm: укажите модель (lhctl models — список)\n"));
    return 4;
  }
  c.res = c.cli.Get("/v0/models", c.hdrs);
  if (!c.res || c.res->status != 200) {
    if (c.res && c.res->status == 401) {
      std::fprintf(stderr, "%s", tr("lhctl: неверный токен (401) — см. data/token.txt\n"));
      return 2;
    }
    std::fprintf(stderr, tr("llmhost недоступен на 127.0.0.1:%d\n"), c.port);
    return 1;
  }
  std::string path;
  double gb = 0;
  uint64_t comp_bytes = 0;
  std::vector<std::string> comp_lines; // «mmproj: … (861 МБ)» для подтверждения
  if (auto m = find_model(c.res->body, c.arg)) {
    path = m->value("path", "");
    gb = m->value("size", (uint64_t)0) / 1073741824.0;
    for (auto& sp : m->value("companions", json::array())) {
      uint64_t sz = sp.value("size", (uint64_t)0);
      std::string cp = sp.value("path", "");
      size_t cs = cp.find_last_of("/\\");
      if (cs != std::string::npos) cp = cp.substr(cs + 1);
      std::string kind = sp.value("kind", "");
      std::string label = kind == "mmproj"  ? "mmproj"
                          : kind == "mtp"   ? tr("MTP-модуль")
                          : kind == "shard" ? tr("шард")
                                            : tr("недокачка");
      // Размер модели в списке — уже вся шард-группа. Части в сумму
      // подтверждения второй раз не входят, но в перечне файлов остаются.
      if (kind != "shard") comp_bytes += sz;
      comp_lines.push_back("  " + label + ": " + cp + " (" +
                           human_bytes(sz) + ")");
    }
  }
  if (path.empty()) {
    std::fprintf(stderr, tr("модель «%s» не найдена — lhctl models покажет список\n"),
                 c.arg.c_str());
    return 3;
  }
  if (!c.yes) {
    if (!isatty(STDIN_FILENO)) {
      // Не интерактивный ввод (пайп/скрипт): не-tty без явного --yes не
      // удаляем НИЧЕГО — иначе «echo | lhctl rm» молча сносил бы модели.
      std::fprintf(stderr,
                   "rm: ввод не интерактивен — подтвердите флагом --yes\n");
      return 4;
    }
    std::printf(tr("Удалить «%s» (%s"), c.arg.c_str(), human_bytes((uint64_t)(gb * 1073741824.0)).c_str());
    if (comp_bytes) std::printf(tr(" + спутники %s"), human_bytes(comp_bytes).c_str());
    std::printf("%s", tr(") безвозвратно? [y/N]\n"));
    for (auto& l : comp_lines) std::printf("%s\n", l.c_str());
    std::fflush(stdout);
    std::string line;
    if (!std::getline(std::cin, line) ||
        !(line == "y" || line == "Y" || line == "yes" || line == "д" || line == "Д")) {
      std::printf("%s\n", tr("отменено"));
      return 0;
    }
  }
  c.res = c.cli.Delete("/v0/models", c.hdrs, json{{"path", path}}.dump(), "application/json");
  if (!c.res) { std::fprintf(stderr, "%s", tr("llmhost недоступен\n")); return 1; }
  if (c.res->status == 401) {
    std::fprintf(stderr, "%s", tr("lhctl: неверный токен (401) — см. data/token.txt\n"));
    return 2;
  }
  if (c.res->status != 200) {
    std::fprintf(stderr, "%s\n", error_message(c.res->body).c_str());
    return 3;
  }
  try {
    auto r = json::parse(c.res->body);
    std::printf(tr("удалено безвозвратно: %s — файлов %zu, освобождено %s\n"),
                c.arg.c_str(), r.value("files", json::array()).size(),
                human_bytes(r.value("bytes", (uint64_t)0)).c_str());
  } catch (...) {
    std::printf(tr("удалено безвозвратно: %s\n"), c.arg.c_str());
  }
  return 0;

  return kTail; // недостижимо: все пути выше возвращают код
}
static int run_ctx(CmdCtx& c) {
  // Ручной контекст модели: без значения — показать; число — записать
  // ctx-size в llmhost.conf; auto — убрать override.
  if (c.arg.empty()) {
    std::fprintf(stderr, "%s", tr("ctx: укажите модель (lhctl models — список)\n"));
    return 4;
  }
  c.res = c.cli.Get("/v0/models", c.hdrs);
  if (!c.res || c.res->status != 200) {
    if (c.res && c.res->status == 401) {
      std::fprintf(stderr, "%s", tr("lhctl: неверный токен (401) — см. data/token.txt\n"));
      return 2;
    }
    std::fprintf(stderr, tr("llmhost недоступен на 127.0.0.1:%d\n"), c.port);
    return 1;
  }
  std::string path, mid;
  int cur = 0, plan = 0;
  if (auto m = find_model(c.res->body, c.arg)) {
    path = m->value("path", "");
    mid = m->value("id", m->value("name", std::string()));
    cur = m->value("ctx_override", 0);
    plan = m->value("fit", json::object()).value("ctx", 0);
  }
  if (path.empty()) {
    std::fprintf(stderr, tr("модель «%s» не найдена — lhctl models покажет список\n"),
                 c.arg.c_str());
    return 3;
  }
  if (c.arg2.empty()) {
    if (cur > 0)
      std::printf(tr("«%s»: ручной ctx-size %d (план умной настройки %s)\n"),
                  mid.c_str(), cur,
                  plan ? std::to_string(plan).c_str() : "—");
    else
      std::printf(tr("«%s»: ручной ctx нет — умная настройка решает сама (план %s)\n"),
                  mid.c_str(), plan ? std::to_string(plan).c_str() : "—");
    std::printf(tr("установить: lhctl ctx %s <токены>  ·  убрать: lhctl ctx %s auto\n"),
                c.arg.c_str(), c.arg.c_str());
    return 0;
  }
  long long v = 0;
  bool remove = c.arg2 == "auto" || c.arg2 == "0";
  if (!remove) {
    char* end = nullptr;
    v = strtoll(c.arg2.c_str(), &end, 10);
    if (!end || *end != '\0' || v <= 0) {
      std::fprintf(stderr, "ctx: значение — число токенов или auto (получено «%s»)\n",
                   c.arg2.c_str());
      return 4;
    }
    if (v < 256 || v > 2097152) {
      std::fprintf(stderr, "%s", tr("ctx: разумный диапазон 256…2097152 токенов\n"));
      return 4;
    }
  }
  c.res = c.cli.Post("/v0/models/ctx", c.hdrs,
                 json{{"path", path}, {"ctx", remove ? 0 : (int)v}}.dump(),
                 "application/json");
  if (!c.res) { std::fprintf(stderr, "%s", tr("llmhost недоступен\n")); return 1; }
  if (c.res->status == 401) {
    std::fprintf(stderr, "%s", tr("lhctl: неверный токен (401) — см. data/token.txt\n"));
    return 2;
  }
  if (c.res->status != 200) {
    std::fprintf(stderr, "%s\n", error_message(c.res->body).c_str());
    return 3;
  }
  try {
    auto r = json::parse(c.res->body);
    std::string section = r.value("section", mid);
    if (remove) {
      std::string tail = plan ? tr(" (план ") + std::to_string(plan) + ")" : "";
      std::printf(tr("ручной ctx снят: «%s» — умная настройка решает сама%s\n"),
                  mid.c_str(), tail.c_str());
    } else {
      std::printf("ctx-size = %lld → llmhost.conf [%s]\n", v, section.c_str());
      std::printf("%s\n", tr("пресет пересобирается — дети перезапустятся плавно"));
      // Демон проверил, что при таком контексте проекция ещё помещается;
      // непустая note — «такой ctx исключает vision».
      const std::string note = r.value("note", "");
      if (!note.empty()) std::printf("%s\n", (tr("предупреждение: ") + note).c_str());
    }
  } catch (...) {
    std::printf("%s\n", tr(remove ? "ручной ctx снят" : "ctx-size записан"));
  }
  return 0;

  return kTail; // недостижимо: все пути выше возвращают код
}
static int run_mtp(CmdCtx& c) {
  // Пер-модельный mtp-mode: без значения — показать; auto/on/off — записать
  // переопределение (auto убирает строку, возвращая глобальный [general]).
  if (c.arg.empty()) {
    std::fprintf(stderr, "%s", tr("mtp: укажите модель (lhctl models — список)\n"));
    return 4;
  }
  c.res = c.cli.Get("/v0/models", c.hdrs);
  if (!c.res || c.res->status != 200) {
    if (c.res && c.res->status == 401) {
      std::fprintf(stderr, "%s", tr("lhctl: неверный токен (401) — см. data/token.txt\n"));
      return 2;
    }
    std::fprintf(stderr, tr("llmhost недоступен на 127.0.0.1:%d\n"), c.port);
    return 1;
  }
  std::string path, mid, cur, glob;
  if (auto m = find_model(c.res->body, c.arg)) {
    path = m->value("path", "");
    mid = m->value("id", m->value("name", std::string()));
    cur = m->value("mtp_override", "");
    glob = m->value("mtp_mode", "");
  }
  if (path.empty()) {
    std::fprintf(stderr, tr("модель «%s» не найдена — lhctl models покажет список\n"),
                 c.arg.c_str());
    return 3;
  }
  if (c.arg2.empty()) {
    if (cur.empty())
      std::printf(tr("«%s»: mtp-mode не задан — глобальный «%s»\n"), mid.c_str(),
                  glob.c_str());
    else
      std::printf(tr("«%s»: mtp-mode = %s (переопределение)\n"), mid.c_str(), cur.c_str());
    std::printf(tr("установить: lhctl mtp %s auto|on|off\n"), c.arg.c_str());
    return 0;
  }
  std::string mode = c.arg2;
  for (char& ch : mode)
    if (ch >= 'A' && ch <= 'Z') ch = (char)(ch - 'A' + 'a');
  if (mode != "auto" && mode != "on" && mode != "off") {
    std::fprintf(stderr, "mtp: значение — auto, on или off (получено «%s»)\n",
                 c.arg2.c_str());
    return 4;
  }
  c.res = c.cli.Post("/v0/models/mtp", c.hdrs,
                 json{{"path", path}, {"mode", mode}}.dump(), "application/json");
  if (!c.res) { std::fprintf(stderr, "%s", tr("llmhost недоступен\n")); return 1; }
  if (c.res->status == 401) {
    std::fprintf(stderr, "%s", tr("lhctl: неверный токен (401) — см. data/token.txt\n"));
    return 2;
  }
  if (c.res->status != 200) {
    std::fprintf(stderr, "%s\n", error_message(c.res->body).c_str());
    return 3;
  }
  try {
    auto r = json::parse(c.res->body);
    std::string section = r.value("section", mid);
    if (mode == "auto")
      std::printf(tr("mtp-mode снят: «%s» — решает глобальный [general]\n"), mid.c_str());
    else
      std::printf(tr("mtp-mode = %s → llmhost.conf [%s]\n"), mode.c_str(), section.c_str());
    std::printf("%s\n", tr("слот в сон — первый запрос поднимет ребёнка с новым режимом"));
  } catch (...) {
    std::printf("%s\n", tr("mtp-mode записан"));
  }
  return 0;

  return kTail; // недостижимо: все пути выше возвращают код
}
static int run_vision(CmdCtx& c) {
  // Пер-модельный vision-режим: без значения — показать; auto/off — записать
  // переопределение (auto убирает строку: своя проекция прицепляется сама).
  if (c.arg.empty()) {
    std::fprintf(stderr, "%s", tr("vision: укажите модель (lhctl models — список)\n"));
    return 4;
  }
  c.res = c.cli.Get("/v0/models", c.hdrs);
  if (!c.res || c.res->status != 200) {
    if (c.res && c.res->status == 401) {
      std::fprintf(stderr, "%s", tr("lhctl: неверный токен (401) — см. data/token.txt\n"));
      return 2;
    }
    std::fprintf(stderr, tr("llmhost недоступен на 127.0.0.1:%d\n"), c.port);
    return 1;
  }
  std::string path, mid, cur, mm;
  if (auto m = find_model(c.res->body, c.arg)) {
    path = m->value("path", "");
    mid = m->value("id", m->value("name", std::string()));
    cur = m->value("vision_override", "");
    mm = m->value("mmproj", "");
  }
  if (path.empty()) {
    std::fprintf(stderr, tr("модель «%s» не найдена — lhctl models покажет список\n"),
                 c.arg.c_str());
    return 3;
  }
  if (c.arg2.empty()) {
    if (cur.empty())
      std::printf(tr("«%s»: vision auto — проекция прицепится сама%s\n"), mid.c_str(),
                  mm.empty() ? tr(" (проекций рядом нет)")
                             : (tr(" (проекция «") +
                                mm.substr(mm.find_last_of("/\\") + 1) + "»)")
                                   .c_str());
    else
      std::printf(tr("«%s»: vision = %s (переопределение) — без --mmproj\n"),
                  mid.c_str(), cur.c_str());
    std::printf(tr("установить: lhctl vision %s auto|off\n"), c.arg.c_str());
    return 0;
  }
  std::string mode = c.arg2;
  for (char& ch : mode)
    if (ch >= 'A' && ch <= 'Z') ch = (char)(ch - 'A' + 'a');
  if (mode != "auto" && mode != "off") {
    std::fprintf(stderr, "vision: значение — auto или off (получено «%s»)\n",
                 c.arg2.c_str());
    return 4;
  }
  c.res = c.cli.Post("/v0/models/vision", c.hdrs,
                 json{{"path", path}, {"mode", mode}}.dump(), "application/json");
  if (!c.res) { std::fprintf(stderr, "%s", tr("llmhost недоступен\n")); return 1; }
  if (c.res->status == 401) {
    std::fprintf(stderr, "%s", tr("lhctl: неверный токен (401) — см. data/token.txt\n"));
    return 2;
  }
  if (c.res->status != 200) {
    std::fprintf(stderr, "%s\n", error_message(c.res->body).c_str());
    return 3;
  }
  try {
    auto r = json::parse(c.res->body);
    std::string section = r.value("section", mid);
    if (mode == "auto")
      std::printf(tr("vision снят: «%s» — своя проекция прицепится сама\n"), mid.c_str());
    else
      std::printf(tr("vision-mode = off → llmhost.conf [%s] — ребёнок без --mmproj\n"),
                  section.c_str());
    std::printf("%s\n", tr("слот в сон — первый запрос поднимет ребёнка с новым режимом"));
  } catch (...) {
    std::printf("%s\n", tr("vision-mode записан"));
  }
  return 0;

  return kTail; // недостижимо: все пути выше возвращают код
}
static int run_thinking(CmdCtx& c) {
  // Пер-модельный thinking-режим: без значения — показать; auto/on/off —
  // записать переопределение (auto убирает строку: движок сам детектит).
  if (c.arg.empty()) {
    std::fprintf(stderr, "%s", tr("thinking: укажите модель (lhctl models — список)\n"));
    return 4;
  }
  c.res = c.cli.Get("/v0/models", c.hdrs);
  if (!c.res || c.res->status != 200) {
    if (c.res && c.res->status == 401) {
      std::fprintf(stderr, "%s", tr("lhctl: неверный токен (401) — см. data/token.txt\n"));
      return 2;
    }
    std::fprintf(stderr, tr("llmhost недоступен на 127.0.0.1:%d\n"), c.port);
    return 1;
  }
  std::string path, mid, cur;
  if (auto m = find_model(c.res->body, c.arg)) {
    path = m->value("path", "");
    mid = m->value("id", m->value("name", std::string()));
    cur = m->value("think_override", "");
  }
  if (path.empty()) {
    std::fprintf(stderr, tr("модель «%s» не найдена — lhctl models покажет список\n"),
                 c.arg.c_str());
    return 3;
  }
  if (c.arg2.empty()) {
    if (cur.empty())
      std::printf(tr("«%s»: thinking auto — решает движок по шаблону\n"), mid.c_str());
    else
      std::printf(tr("«%s»: thinking = %s (переопределение)\n"), mid.c_str(),
                  cur.c_str());
    std::printf(tr("установить: lhctl thinking %s auto|on|off\n"), c.arg.c_str());
    return 0;
  }
  std::string mode = c.arg2;
  for (char& ch : mode)
    if (ch >= 'A' && ch <= 'Z') ch = (char)(ch - 'A' + 'a');
  if (mode != "auto" && mode != "on" && mode != "off") {
    std::fprintf(stderr, "thinking: значение — auto, on или off (получено «%s»)\n",
                 c.arg2.c_str());
    return 4;
  }
  c.res = c.cli.Post("/v0/models/thinking", c.hdrs,
                 json{{"path", path}, {"mode", mode}}.dump(), "application/json");
  if (!c.res) { std::fprintf(stderr, "%s", tr("llmhost недоступен\n")); return 1; }
  if (c.res->status == 401) {
    std::fprintf(stderr, "%s", tr("lhctl: неверный токен (401) — см. data/token.txt\n"));
    return 2;
  }
  if (c.res->status != 200) {
    std::fprintf(stderr, "%s\n", error_message(c.res->body).c_str());
    return 3;
  }
  try {
    auto r = json::parse(c.res->body);
    std::string section = r.value("section", mid);
    if (mode == "auto")
      std::printf("%s\n", tr("thinking снят: режим решает движок по шаблону"));
    else
      std::printf(tr("thinking-mode = %s → llmhost.conf [%s] — ребёнок с --reasoning %s\n"),
                  mode.c_str(), section.c_str(), mode.c_str());
    std::printf("%s\n", tr("слот в сон — первый запрос поднимет ребёнка с новым режимом"));
  } catch (...) {
    std::printf("%s\n", tr("thinking-mode записан"));
  }
  return 0;

  return kTail; // недостижимо: все пути выше возвращают код
}
// Таблица метрик одного среза: разбор текста Prometheus демона в строки
// по-модельно. prev — прошлый срез для мгновенных скоростей (watch-режим);
// пустой prev — только пожизненные средние.
static void metrics_table(const std::map<std::string, lh::fmt::MetricRow>& rows,
                          const std::map<std::string, lh::fmt::MetricRow>& prev) {
  std::printf("%s\n", tr("модель                                   декод тк/с  "
                        "префилл  приём  декод ткн  промпт ткн   запр."));
  for (auto& [id, r] : rows) {
    char b1[32], b2[32], b3[32], inst[32];
    std::snprintf(b1, sizeof b1, "%.1f", r.dec_tps);
    std::snprintf(b2, sizeof b2, "%.1f", r.prompt_tps);
    std::snprintf(b3, sizeof b3, "%.2f",
                  r.spec_draft > 0 ? r.spec_acc / r.spec_draft : 0.0);
    // мгновенная скорость: дельта токенов / дельта секунд между срезами
    double now_tps = 0;
    auto p = prev.find(id);
    if (p != prev.end() && r.dec_sec - p->second.dec_sec >= 0.5)
      now_tps = (r.dec_tok - p->second.dec_tok) / (r.dec_sec - p->second.dec_sec);
    if (now_tps > 0) std::snprintf(inst, sizeof inst, "%.1f", now_tps);
    else std::snprintf(inst, sizeof inst, "—");
    std::printf("%-40s %9s %9s %7s %10s %11s %6s\n",
                lh::ui::trunc(id, 40).c_str(), b1, b2,
                r.spec_draft > 0 ? b3 : "—",
                lh::fmt::fmt_u64(r.dec_tok).c_str(),
                lh::fmt::fmt_u64((double)r.prompt_tok).c_str(),
                lh::fmt::fmt_u64((double)r.reqs).c_str());
    if (now_tps > 0)
      std::printf("%40s  %s\n", "",
                  (std::string(tr("мгновенно ")) + inst + " тк/с").c_str());
  }
  if (rows.empty()) std::printf("%s\n", tr("(нет моделей с трафиком — счётчики пусты)"));
}

static int run_metrics(CmdCtx& c) {
  // lhctl metrics [-w сек]: срез /v0/metrics; с -w — живая таблица
  // со спарклайном мгновенной скорости декода.
  const bool watch = c.arg == "-w";
  unsigned period = 2;
  if (watch && !c.arg2.empty()) {
    period = (unsigned)std::atoi(c.arg2.c_str());
    if (period == 0 || period > 60) period = 2;
  }
  std::map<std::string, lh::fmt::MetricRow> prev;
  std::map<std::string, std::deque<double>> hist; // мгновенные скорости
  for (;;) {
    c.res = c.cli.Get("/v0/metrics", c.hdrs);
    if (!c.res || c.res->status != 200) {
      if (c.res && c.res->status == 401) {
        std::fprintf(stderr, "%s", tr("lhctl: неверный токен (401) — см. data/token.txt\n"));
        return 2;
      }
      std::fprintf(stderr, tr("llmhost недоступен на 127.0.0.1:%d\n"), c.port);
      return 1;
    }
    auto rows = lh::fmt::parse_metrics(c.res->body);
    if (watch) std::fprintf(stderr, "\x1b[2J\x1b[H");
    std::printf("%s%s\n", tr("llmhost — метрики движка"),
                watch ? tr("  ·  обновление каждые N с (Ctrl-C — выход)") : "");
    std::printf("%s\n", tr("источник: /metrics детей; декод/префилл — из таймингов "
                           "движка, промпт — из usage прокси (с кэшированными)"));
    metrics_table(rows, prev);
    if (!watch) return 0;
    // Спарклайн мгновенных скоростей за историю наблюдения.
    for (auto& [id, r] : rows) {
      auto p = prev.find(id);
      if (p != prev.end() && r.dec_sec - p->second.dec_sec >= 0.5) {
        auto& h = hist[id];
        h.push_back((r.dec_tok - p->second.dec_tok) / (r.dec_sec - p->second.dec_sec));
        while (h.size() > 48) h.pop_front();
      }
      auto it = hist.find(id);
      if (it != hist.end() && it->second.size() >= 2) {
        const std::vector<double> v(it->second.begin(), it->second.end());
        std::printf("  %s  %s\n", lh::fmt::sparkline(v).c_str(), id.c_str());
      }
    }
    std::fflush(stdout);
    prev = rows;
    std::this_thread::sleep_for(std::chrono::seconds(period));
  }
}

static int image_status_once(CmdCtx& c, nlohmann::json& out) {
  c.res = c.cli.Get("/v0/image", c.hdrs);
  if (!c.res || c.res->status != 200) {
    if (c.res && c.res->status == 401) {
      std::fprintf(stderr, "%s", tr("lhctl: неверный токен (401) — см. data/token.txt\n"));
      return 2;
    }
    std::fprintf(stderr, tr("llmhost недоступен на 127.0.0.1:%d\n"), c.port);
    return 1;
  }
  try {
    out = nlohmann::json::parse(c.res->body);
  } catch (...) {
    std::fprintf(stderr, "%s", tr("image: битый ответ демона\n"));
    return 1;
  }
  return 0;
}
static void print_image_status(const nlohmann::json& j) {
  const std::string st = j.value("state", "unloaded");
  std::printf(tr("движок:  sd-server %s\n"),
              j.value("engine", nlohmann::json::object()).value("found", false)
                  ? tr("найден")
                  : tr("НЕ найден (image-engine или engine/sd/)"));
  std::printf(tr("карта:   image-gpu=%d, простой ≤ %d мин\n"),
              j.value("gpu", 0), j.value("idle_min", 0));
  if (st == "loaded")
    std::printf(tr("состояние: загружена «%s» (порт %d)\n"),
                j.value("model", "").c_str(), j.value("port", 0));
  else if (st == "loading")
    std::printf("%s", tr("состояние: идёт загрузка…\n"));
  else if (st == "error")
    std::printf(tr("состояние: ошибка — %s\n"), j.value("error", "?").c_str());
  else
    std::printf("%s", tr("состояние: не загружена (автостарт при первом запросе)\n"));
  for (const auto& m : j.value("models", nlohmann::json::array())) {
    char gb[32];
    std::snprintf(gb, sizeof(gb), "%.1f", m.value("bytes", 0ull) / 1073741824.0);
    // gb — готовая строка (%s): %.1f с char[] здесь и падал сегфолтом.
    std::printf(tr("модель:  %s — %s, %s ГиБ, шаги %d, CFG %.2f %s\n"),
                m.value("id", "").c_str(), m.value("name", "").c_str(), gb,
                m.value("steps", -1), m.value("cfg", -1.0),
                m.value("sampler", "").c_str());
  }
}
static int pull_follow(CmdCtx& c); // хвост прогресса — общий с run_pull (ниже)
static int run_image(CmdCtx& c) {
  // Image-слот (sd-server): без аргумента — статус (или id построчно с --raw
  // для автодополнения); load [модель] — загрузить (пустая модель —
  // автостарт-выбор демона) и дождаться готовности; pull [имя] — скачать
  // художника; rm — удалить с диска; unload — остановить ребёнка.
  // Генерация — не здесь: /v1/images/* OpenAI-порта.
  if (c.arg.empty()) {
    nlohmann::json j;
    int rc = image_status_once(c, j);
    if (rc) return rc;
    if (c.raw) {
      // --raw: id каталога построчно — автодополнение load/rm (как models --raw).
      for (const auto& m : j.value("models", nlohmann::json::array()))
        std::puts(m.value("id", "").c_str());
      return 0;
    }
    print_image_status(j);
    return 0;
  }
  if (c.arg == "unload") {
    c.res = c.cli.Post("/v0/image/unload", c.hdrs, "{}", "application/json");
    if (!c.res || c.res->status != 200) {
      std::fprintf(stderr, "%s", tr("image: unload не удался\n"));
      return 1;
    }
    std::printf("%s\n", tr("image-движок остановлен"));
    return 0;
  }
  if (c.arg == "pull") {
    // Художники: без имени — каталог встроенных; с именем — скачать (имя
    // каталога или org/repo с раскладкой-конвенцией). Прогресс — общий /v0/pull.
    if (c.arg2.empty()) {
      nlohmann::json j;
      int rc = image_status_once(c, j);
      if (rc) return rc;
      if (c.raw) {
        for (const auto& a : j.value("aliases", nlohmann::json::array()))
          std::puts(a.get<std::string>().c_str());
        return 0;
      }
      std::puts(tr("Каталог художников для «lhctl image pull <имя>»:\n"));
      for (const auto& a : j.value("aliases", nlohmann::json::array()))
        std::printf("  %s\n", a.get<std::string>().c_str());
      std::puts(tr("\nСвой источник — репозиторий org/repo с раскладкой: recipes/*.json + "
                   "diffusion_models/ + text_encoders/ + vae/ (по одному файлу на слот)."));
      return 0;
    }
    json body = {{"repo", c.arg2}, {"kind", "image"}};
    auto r = c.cli.Post("/v0/pull", c.hdrs, body.dump(), "application/json");
    if (!r) {
      std::fprintf(stderr, tr("llmhost недоступен на 127.0.0.1:%d\n"), c.port);
      return 1;
    }
    if (r->status == 401) {
      std::fprintf(stderr, "%s", tr("lhctl: неверный токен (401) — см. data/token.txt\n"));
      return 2;
    }
    if (r->status >= 400 && r->status != 409) {
      std::fprintf(stderr, "%s\n", error_message(r->body).c_str());
      return 3;
    }
    // 202 — качаем; 409 — занято другим pull'ом: проводим до конца его.
    return pull_follow(c);
  }
  if (c.arg == "rm") {
    // Удаление художника с диска: файлы слотов + рецепт (если свободен) и
    // запись привязки; загруженную демон выгрузит сам. Подтверждение — как rm.
    if (c.arg2.empty()) {
      std::fprintf(stderr, "%s", tr("image rm: укажите модель (lhctl image — список)\n"));
      return 4;
    }
    nlohmann::json j;
    int rc = image_status_once(c, j);
    if (rc) return rc;
    std::string name;
    uint64_t bytes = 0;
    bool known = false;
    for (const auto& m : j.value("models", nlohmann::json::array())) {
      if (m.value("id", "") == c.arg2) {
        known = true;
        name = m.value("name", "");
        bytes = m.value("bytes", (uint64_t)0);
        break;
      }
    }
    if (!known) {
      std::fprintf(stderr, tr("в каталоге image-моделей нет «%s» — lhctl image покажет список\n"),
                   c.arg2.c_str());
      return 3;
    }
    if (!c.yes) {
      if (!isatty(STDIN_FILENO)) {
        // Не интерактивный ввод (пайп/скрипт): не-tty без явного --yes не
        // удаляем НИЧЕГО — тот же контракт, что у rm.
        std::fprintf(stderr, "%s", tr("image rm: ввод не интерактивен — подтвердите флагом --yes\n"));
        return 4;
      }
      std::printf(tr("Удалить «%s» (%s) с диска безвозвратно? [y/N]\n"),
                  c.arg2.c_str(), human_bytes(bytes).c_str());
      std::fflush(stdout);
      std::string line;
      if (!std::getline(std::cin, line) ||
          !(line == "y" || line == "Y" || line == "yes" || line == "д" || line == "Д")) {
        std::printf("%s\n", tr("отменено"));
        return 0;
      }
    }
    c.res = c.cli.Delete("/v0/image", c.hdrs, json{{"model", c.arg2}}.dump(),
                         "application/json");
    if (!c.res) { std::fprintf(stderr, "%s", tr("llmhost недоступен\n")); return 1; }
    if (c.res->status == 401) {
      std::fprintf(stderr, "%s", tr("lhctl: неверный токен (401) — см. data/token.txt\n"));
      return 2;
    }
    if (c.res->status != 200) {
      std::fprintf(stderr, "%s\n", error_message(c.res->body).c_str());
      return 3;
    }
    try {
      auto r = json::parse(c.res->body);
      std::string tail;
      if (r.contains("recipe") && !r["recipe"].is_null())
        tail = tr(", рецепт ") + r.value("recipe", std::string()) + tr(" удалён");
      std::printf(tr("удалено безвозвратно: %s — файлов %zu, освобождено %s%s\n"),
                  c.arg2.c_str(), r.value("files", json::array()).size(),
                  human_bytes(r.value("bytes", (uint64_t)0)).c_str(), tail.c_str());
    } catch (...) {
      std::printf(tr("удалено безвозвратно: %s\n"), c.arg2.c_str());
    }
    return 0;
  }
  if (c.arg != "load") {
    std::fprintf(stderr, "%s",
                 tr("image: статус — без аргумента; load [модель], pull [имя], "
                    "rm <модель>, unload\n"));
    return 4;
  }
  nlohmann::json body = nlohmann::json::object();
  if (!c.arg2.empty()) body["model"] = c.arg2;
  c.res = c.cli.Post("/v0/image/load", c.hdrs, body.dump(), "application/json");
  if (!c.res || (c.res->status != 200 && c.res->status != 202)) {
    if (c.res && (c.res->status == 400 || c.res->status == 404 || c.res->status == 409))
      std::fprintf(stderr, tr("image: %s\n"), c.res->body.c_str());
    else
      std::fprintf(stderr, "%s", tr("image: load не удался\n"));
    return 1;
  }
  std::printf("%s", tr("image: загрузка…\n"));
  for (int i = 0; i < 240; ++i) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    nlohmann::json j;
    int rc = image_status_once(c, j);
    if (rc) return rc;
    const std::string st = j.value("state", "unloaded");
    if (st == "loaded") {
      std::printf(tr("image: «%s» загружена (порт %d)\n"),
                  j.value("model", "").c_str(), j.value("port", 0));
      return 0;
    }
    if (st == "error") {
      std::fprintf(stderr, tr("image: ошибка — %s\n"), j.value("error", "?").c_str());
      return 1;
    }
  }
  std::fprintf(stderr, "%s", tr("image: загрузка не завершилась за 240 с\n"));
  return 1;
}
static int run_set(CmdCtx& c) {
  // Настройки [general] llmhost.conf — один источник правды. Без аргументов
  // — вся таблица ключей; с аргументами — записать (валидация на демоне,
  // прямой файл lhctl не трогает) и применить на лету, если ключ умеет.
  if (c.arg.empty()) {
    c.res = c.cli.Get("/v0/settings", c.hdrs);
    if (!c.res || c.res->status != 200) {
      if (c.res && c.res->status == 401) {
        std::fprintf(stderr, "%s", tr("lhctl: неверный токен (401) — см. data/token.txt\n"));
        return 2;
      }
      std::fprintf(stderr, tr("llmhost недоступен на 127.0.0.1:%d\n"), c.port);
      return 1;
    }
    try {
      auto r = json::parse(c.res->body);
      auto arr = r.value("settings", json::array());
      std::printf(tr("[general] в %s:\n"), r.value("conf", "").c_str());
      size_t wkey = 0;
      for (auto& e : arr) wkey = std::max(wkey, e.value("key", "").size());
      for (auto& e : arr) {
        std::string k = e.value("key", ""), v = e.value("value", "");
        bool rr = e.value("restart_required", false);
        std::printf("  %-*s = %-10s %s\n", (int)wkey, k.c_str(), v.c_str(),
                    rr ? tr("· после рестарта демона") : "");
      }
    } catch (...) {
      std::fprintf(stderr, "%s\n", tr("set: ответ демона не читается"));
      return 3;
    }
    return 0;
  }
  if (c.arg2.empty()) {
    std::fprintf(stderr, "%s\n", tr("set: укажите значение (lhctl set — список ключей)"));
    return 4;
  }
  c.res = c.cli.Post("/v0/settings", c.hdrs, json{{"key", c.arg}, {"value", c.arg2}}.dump(),
                 "application/json");
  if (!c.res) { std::fprintf(stderr, "%s", tr("llmhost недоступен\n")); return 1; }
  if (c.res->status == 401) {
    std::fprintf(stderr, "%s", tr("lhctl: неверный токен (401) — см. data/token.txt\n"));
    return 2;
  }
  if (c.res->status != 200) {
    std::fprintf(stderr, "%s\n", error_message(c.res->body).c_str());
    return 3;
  }
  try {
    auto r = json::parse(c.res->body);
    std::string key = r.value("key", c.arg), value = r.value("value", c.arg2);
    std::printf("%s = %s → llmhost.conf [general]\n", key.c_str(), value.c_str());
    if (r.value("restart_required", false))
      std::printf("%s\n", tr("вступит после рестарта: systemctl --user restart llmhostd"));
    else
      std::printf("%s\n", tr("применено — пресет пересобран, дети перезапустятся плавно"));
  } catch (...) {
    std::printf("%s\n", tr("записано"));
  }
  return 0;

  return kTail; // недостижимо: все пути выше возвращают код
}
// Хвост скачивания, общий llama-pull и image-pull: опрос /v0/pull до done/error
// с прогресс-строкой. 0 — скачано, 3 — ошибка (текст уже выведен).
static int pull_follow(CmdCtx& c) {
  for (;;) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    auto s = c.cli.Get("/v0/pull", c.hdrs);
    if (!s || s->status != 200) continue; // короткий сбой — попробуем ещё
    try {
      auto j = json::parse(s->body);
      std::string st = j.value("state", "");
      uint64_t done = j.value("done", (uint64_t)0), total = j.value("total", (uint64_t)0);
      double speed = j.value("speed", 0.0);
      if (lh::ui::pretty()) {
        if (total > 0)
          lh::ui::progress_update(download_line(j.value("file", ""), true,
                                                (double)done / (double)total, done, total,
                                                speed, ""));
        else
          lh::ui::progress_update(download_line(j.value("file", ""), false, 0, 0, 0, speed,
                                                tr("резолв репозитория…")));
      } else if (total > 0) {
        char line[256];
        std::snprintf(line, sizeof(line), tr("\r  ↓ %-40s %2llu%%   %6.1f МБ/с   "),
                      j.value("file", "").substr(0, 40).c_str(),
                      (unsigned long long)(done * 100 / total), speed / 1048576.0);
        std::fputs(line, stdout);
        std::fflush(stdout);
      } else {
        char line[256];
        std::snprintf(line, sizeof(line), tr("\r  ↓ %-40s %6.1f МБ/с   "),
                      j.value("file", "").substr(0, 40).c_str(), speed / 1048576.0);
        std::fputs(line, stdout);
        std::fflush(stdout);
      }
      if (st == "done") {
        if (lh::ui::pretty())
          lh::ui::progress_end(lh::ui::c::ok(tr("✓ скачано: ")) + j.value("repo", ""));
        else
          std::printf(tr("\rскачано: %s\n"), j.value("repo", "").c_str());
        return 0;
      }
      if (st == "error") {
        if (lh::ui::pretty()) std::fputc('\r', stdout), std::fputs("\x1b[2K", stdout);
        std::printf("%.*s\n", 120, j.value("error", tr("ошибка скачивания")).c_str());
        return 3;
      }
    } catch (...) {}
  }
}
static int run_pull(CmdCtx& c) {
  if (c.arg.empty()) {
    // Список коротких имён — из API (живой источник истины).
    auto r = c.cli.Get("/v0/aliases", c.hdrs);
    if (!r) {
      std::fprintf(stderr, tr("llmhost недоступен на 127.0.0.1:%d\n"), c.port);
      return 1;
    }
    if (r->status == 401) {
      std::fprintf(stderr, "%s", tr("lhctl: неверный токен (401) — см. data/token.txt\n"));
      return 2;
    }
    if (r->status != 200) {
      std::fprintf(stderr, tr("aliases: сервер ответил %d\n"), r->status);
      return 3;
    }
    try {
      auto arr = json::parse(r->body).value("aliases", json::array());
      if (c.raw) {
        for (auto& a : arr) std::puts(a.value("name", "").c_str());
      } else if (lh::ui::pretty()) {
        std::puts(tr("Короткие имена для «lhctl pull <имя>» — квант под вашу карту "
                  "выбирается сам:\n"));
        std::fputs(render_aliases(json{{"aliases", arr}}).c_str(), stdout);
        std::puts(tr("\nПолные репозитории (org/repo) тоже принимаются; файл — "
                  "вторым аргументом или wildcard-фильтром."));
      } else {
        std::puts(tr("Короткие имена для «lhctl pull <имя>» (квант под вашу карту "
                  "выбирается сам):\n"));
        for (auto& a : arr)
          std::printf("  %-14s %-48s %s\n", a.value("name", "").c_str(),
                      a.value("repo", "").c_str(), a.value("note", "").c_str());
        std::puts(tr("\nПолные репозитории (org/repo) тоже принимаются; файл — "
                  "вторым аргументом или wildcard-фильтром."));
      }
    } catch (...) {
      std::fprintf(stderr, "%s\n", tr("aliases: ответ не разобрался"));
      return 3;
    }
    return 0;
  }
  if (c.arg == "stop") {
    // Штатная остановка идущего скачивания: .part остаётся на диске,
    // повторный pull докачивает с места останова.
    auto r = c.cli.Delete("/v0/pull", c.hdrs, "", "text/plain");
    if (!r) {
      std::fprintf(stderr, tr("llmhost недоступен на 127.0.0.1:%d\n"), c.port);
      return 1;
    }
    if (r->status == 401) {
      std::fprintf(stderr, "%s", tr("lhctl: неверный токен (401) — см. data/token.txt\n"));
      return 2;
    }
    if (r->status >= 400) {
      std::fprintf(stderr, "%s\n", error_message(r->body).c_str());
      return 3;
    }
    std::puts(tr("остановлено: недокачанный .part остался, повторный pull докачит"));
    return 0;
  }
  // Скачивание: короткое имя (qwen3) или org/repo; файл/фильтр — опционально.
  json body = {{"repo", c.arg}};
  if (!c.arg2.empty()) body["file"] = c.arg2;
  auto r = c.cli.Post("/v0/pull", c.hdrs, body.dump(), "application/json");
  if (!r) {
    std::fprintf(stderr, tr("llmhost недоступен на 127.0.0.1:%d\n"), c.port);
    return 1;
  }
  if (r->status == 401) {
    std::fprintf(stderr, "%s", tr("lhctl: неверный токен (401) — см. data/token.txt\n"));
    return 2;
  }
  if (r->status >= 400 && r->status != 409) {
    std::fprintf(stderr, "%s\n", error_message(r->body).c_str());
    return 3;
  }
  // 202 — качаем; 409 — уже качается что-то другое: проводим до конца его.
  return pull_follow(c);
}
static int run_engine(CmdCtx& c) {
  // ---- движок: статус / install / update ---------------------------------
  if (c.arg.empty() || c.arg == "status") {
    auto r = c.cli.Get("/v0/engine", c.hdrs);
    if (!r) {
      std::fprintf(stderr, tr("llmhost недоступен на 127.0.0.1:%d\n"), c.port);
      return 1;
    }
    if (r->status == 401) {
      std::fprintf(stderr, "%s", tr("lhctl: неверный токен (401) — см. data/token.txt\n"));
      return 2;
    }
    if (r->status != 200) {
      std::fprintf(stderr, "engine: сервер ответил %d\n", r->status);
      return 3;
    }
    try {
      auto j = json::parse(r->body);
      if (lh::ui::pretty()) {
        std::fputs(render_engine(j).c_str(), stdout);
        return 0;
      }
      std::printf(tr("пин манифеста:   %s\n"), j.value("release", "?").c_str());
      std::printf(tr("активная сборка: %s\n"), j.value("active", "?").c_str());
      if (!j.value("selected", "").empty())
        std::printf(tr("выбрана:         %s (снять: lhctl engine use auto)\n"),
                    j.value("selected", "").c_str());
      std::string st = j.value("state", "idle");
      if (st != "idle")
        std::printf(tr("установка: %s «%s»\n"), st.c_str(), j.value("build", "").c_str());
      for (auto& b : j.value("builds", json::array())) {
        uint64_t sz = b.value("size", (uint64_t)0);
        char size_str[32];
        if (sz >= 1073741824ull)
          std::snprintf(size_str, sizeof(size_str), tr("%.1f ГБ"), sz / 1073741824.0);
        else
          std::snprintf(size_str, sizeof(size_str), tr("%.0f МБ"), sz / 1048576.0);
        std::string rel = b.value("installed_release", "");
        std::string tail;
        if (b.value("installed", false) && !rel.empty()) tail = " (" + rel + ")";
        std::printf("  %-10s %8s  %s %s%s\n", b.value("id", "?").c_str(), size_str,
                    b.value("pinned", false) ? tr("пиннута") : tr("БЕЗ ПИНА"),
                    b.value("installed", false) ? tr("установлена") : "", tail.c_str());
      }
      if (j.value("mismatch", false))
        std::printf("%s\n",
            tr("РАСХОЖДЕНИЕ: пин и установленный движок расходятся — скобки выше показывают факт"));
    } catch (...) {
      std::fprintf(stderr, "%s\n", tr("engine: ответ не разобрался"));
      return 3;
    }
    return 0;
  }
  if (c.arg == "use") {
    // Явный выбор сборки движка: дети перезапускаются плавно на новой сборке.
    if (c.arg2.empty()) {
      std::fprintf(stderr, "%s\n", tr("engine use: укажите сборку (lhctl engine — список) или auto"));
      return 4;
    }
    auto ur = c.cli.Post("/v0/engine/build", c.hdrs, json{{"build", c.arg2}}.dump(),
                       "application/json");
    if (!ur) { std::fprintf(stderr, "llmhost недоступен на 127.0.0.1:%d\n", c.port); return 1; }
    if (ur->status == 401) {
      std::fprintf(stderr, "%s", tr("lhctl: неверный токен (401) — см. data/token.txt\n"));
      return 2;
    }
    if (ur->status >= 400) {
      std::fprintf(stderr, "%s\n", error_message(ur->body).c_str());
      return 3;
    }
    try {
      auto j = json::parse(ur->body);
      std::string sel = j.value("selected", "?");
      std::string act = j.value("active", "?");
      if (sel == "auto")
        std::printf(tr("выбор снят — сборка по железу: %s\n"), act.c_str());
      else
        std::printf(tr("выбрана сборка %s (активная: %s) — дети перезапускаются\n"),
                    sel.c_str(), act.c_str());
      std::printf("%s\n", tr("сменить обратно: lhctl engine use auto"));
    } catch (...) {
      std::printf("%s\n", tr("готово"));
    }
    return 0;
  }
  if (c.arg != "install" && c.arg != "update") {
    std::fprintf(stderr,
                 tr("engine: неизвестная подкоманда «%s» (status|use|install|update)\n"),
                 c.arg.c_str());
    return 4;
  }
  if (c.arg == "install" && c.arg2.empty()) {
    std::fprintf(stderr, "%s\n", tr("engine install: укажите сборку (lhctl engine — список)"));
    return 4;
  }
  if (c.arg == "install" && (c.pin_only || c.all_builds || !c.release_tag.empty())) {
    // install ставит ТЕКУЩИЙ пин; флаги пина осмысленны только в update
    std::fprintf(stderr,
                 "%s\n", tr("engine install: флаги --release/--pin-only/--all не применяются"));
    return 4;
  }
  if (c.arg == "update" && !c.arg2.empty() && c.all_builds) {
    std::fprintf(stderr, "%s\n", tr("engine update: указана сборка и --all одновременно"));
    return 4;
  }
  std::vector<std::string> targets;
  if (c.arg == "install") {
    targets.push_back(c.arg2);
  } else if (!c.arg2.empty()) {
    targets.push_back(c.arg2);
  }

  if (c.arg == "update") {
    // Состояние до пина: нужно и для сравнения версий, и для выбора цели
    // (активная или все пиннутые), когда она не указана явно.
    std::string rel_before;
    auto cur = c.cli.Get("/v0/engine", c.hdrs);
    if (!cur) {
      std::fprintf(stderr, tr("llmhost недоступен на 127.0.0.1:%d\n"), c.port);
      return 1;
    }
    if (cur->status == 401) {
      std::fprintf(stderr, "%s", tr("lhctl: неверный токен (401) — см. data/token.txt\n"));
      return 2;
    }
    if (cur->status != 200) {
      std::fprintf(stderr, "engine: сервер ответил %d\n", cur->status);
      return 3;
    }
    try {
      auto j = json::parse(cur->body);
      rel_before = j.value("release", "");
      if (targets.empty()) {
        if (c.all_builds) {
          for (auto& b : j.value("builds", json::array()))
            if (b.value("pinned", false)) targets.push_back(b.value("id", ""));
        } else {
          std::string act = j.value("active", "");
          if (act.empty()) {
            std::fprintf(stderr,
                         "%s\n",
                         tr("engine: демон не определил активную сборку — укажите "
                            "явно: lhctl engine update <сборка>"));
            return 4;
          }
          targets.push_back(act);
        }
      }
    } catch (...) {
      std::fprintf(stderr, "engine: ответ не разобрался\n");
      return 3;
    }

    auto pr = c.cli.Post("/v0/engine/pin", c.hdrs,
                       json{{"release", c.release_tag.empty() ? "latest" : c.release_tag}}
                           .dump(),
                       "application/json");
    if (!pr) {
      std::fprintf(stderr, tr("llmhost недоступен на 127.0.0.1:%d\n"), c.port);
      return 1;
    }
    if (pr->status == 401) {
      std::fprintf(stderr, "%s", tr("lhctl: неверный токен (401) — см. data/token.txt\n"));
      return 2;
    }
    if (pr->status != 200) {
      std::fprintf(stderr, "%s\n", error_message(pr->body).c_str());
      return 3;
    }
    std::string rel_after, pin_path, pin_added;
    try {
      auto pj = json::parse(pr->body);
      rel_after = pj.value("release", "");
      pin_path = pj.value("path", "");
      pin_added = pj.value("added", "");
    } catch (...) {}
    if (rel_after == rel_before) {
      std::printf(
          tr("пин уже %s — обновление не требуется\n"
             "(переустановить текущий пин: lhctl engine install <сборка>)\n"),
          rel_after.c_str());
      return 0;
    }
    std::printf(tr("пин манифеста: %s → %s\n"), rel_before.c_str(), rel_after.c_str());
    if (!pin_path.empty())
      std::printf(tr("файл: %s\n"), pin_path.c_str());
    if (!pin_added.empty())
      std::printf(tr("достроены из ассетов релиза: %s\n"), pin_added.c_str());
    if (c.pin_only) return 0;
  }

  if (targets.empty()) {
    std::fprintf(stderr, "%s\n", tr("engine: нет пиннутых сборок для установки"));
    return 4;
  }
  for (auto& t : targets) {
    auto ir = c.cli.Post("/v0/engine", c.hdrs, json{{"build", t}}.dump(),
                       "application/json");
    if (!ir) {
      std::fprintf(stderr, tr("llmhost недоступен на 127.0.0.1:%d\n"), c.port);
      return 1;
    }
    if (ir->status == 401) {
      std::fprintf(stderr, "%s", tr("lhctl: неверный токен (401) — см. data/token.txt\n"));
      return 2;
    }
    if (ir->status >= 400) {
      std::fprintf(stderr, "%s: %s\n", t.c_str(), error_message(ir->body).c_str());
      return 3;
    }
    std::printf(tr("ставлю «%s»\n"), t.c_str());
    bool seen_active = false;
    int idle_ticks = 0;
    for (;;) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      auto s = c.cli.Get("/v0/engine", c.hdrs);
      if (s && s->status == 401) {
        std::fprintf(stderr, "%s\n", tr("lhctl: неверный токен (401)"));
        return 2;
      }
      if (!s || s->status != 200) continue; // короткий сбой — попробуем ещё
      try {
        auto j = json::parse(s->body);
        std::string st = j.value("state", "idle");
        if (st == "error") {
          if (lh::ui::pretty()) lh::ui::progress_end("");
          std::fprintf(stderr, "%s\n", (tr("ошибка: ") + j.value("error", "?")).c_str());
          return 3;
        }
        if (st == "downloading" || st == "extracting") {
          seen_active = true;
          uint64_t done = j.value("done", (uint64_t)0),
                   total = j.value("total", (uint64_t)0);
          double speed = j.value("speed", 0.0);
          if (lh::ui::pretty()) {
            if (st == "downloading" && total > 0)
              lh::ui::progress_update(download_line(t, true, (double)done / (double)total,
                                                    done, total, speed, ""));
            else if (st == "downloading")
              lh::ui::progress_update(download_line(t, false, 0, 0, 0, speed, ""));
            else
              lh::ui::progress_update(download_line(t, false, 0, 0, 0, 0,
                                                    tr("распаковка tar.gz…")));
          } else if (st == "downloading" && total > 0) {
            char line[160];
            std::snprintf(line, sizeof(line), tr("\r  ↓ %-10s %2llu%%   %6.1f МБ/с   "),
                          t.c_str(), (unsigned long long)(done * 100 / total),
                          speed / 1048576.0);
            std::fputs(line, stdout);
            std::fflush(stdout);
          } else if (st == "downloading") {
            char line[160];
            std::snprintf(line, sizeof(line), tr("\r  ↓ %-10s %6.1f МБ/с   "), t.c_str(),
                          speed / 1048576.0);
            std::fputs(line, stdout);
            std::fflush(stdout);
          } else {
            char line[160];
            std::snprintf(line, sizeof(line), tr("\r  ↓ %-10s распаковка…        "),
                          t.c_str());
            std::fputs(line, stdout);
            std::fflush(stdout);
          }
        } else if (seen_active) {
          // idle после активной фазы — установка закончилась
          if (lh::ui::pretty())
            lh::ui::progress_end(lh::ui::c::ok(tr("✓ готово: ")) + t);
          else
            std::printf(tr("\rготово: «%s»%70s\n"), t.c_str(), "");
          break;
        } else if (++idle_ticks > 30) {
          // POST прошёл, а состояние так и не сдвинулось — застряли
          std::fprintf(stderr,
                       "%s\n", tr("\nустановка не началась за 30 с — смотрите lhctl engine"));
          return 3;
        }
      } catch (...) {}
    }
  }
  return 0;

  return kTail; // недостижимо: все пути выше возвращают код
}
static int run_tokens(CmdCtx& c) {
  // Именованные токены доступа к /v1 (словарь в api_tokens.json демона).
  if (c.arg.empty() || c.arg == "list") {
    c.res = c.cli.Get("/v0/tokens", c.hdrs);
    if (lh::ui::pretty() && c.res && c.res->status >= 200 && c.res->status < 300) {
      try {
        auto arr = json::parse(c.res->body).value("tokens", json::array());
        lh::ui::Table t;
        t.col(tr("ИМЯ")).col(tr("ТОКЕН")).col(tr("СОЗДАН")).col(tr("ПОСЛЕДНИЙ ИСПОЛЬЗОВАН"));
        for (auto& e : arr)
          t.row({lh::ui::c::accent(e.value("name", "?")), e.value("token", "?"),
                 e.value("created_at", "—"), e.value("last_used", "—")});
        if (!t.rows()) std::puts(tr("токенов нет — lhctl tokens add <имя>"));
        else std::fputs(t.render().c_str(), stdout);
        return 0;
      } catch (...) {}
    }
    // не-tty — сырой JSON общим хвостом ниже
  } else if (c.arg == "add") {
    if (c.arg2.empty()) {
      std::fprintf(stderr, "%s\n", tr("tokens add: укажите имя (lhctl tokens add rag2)"));
      return 4;
    }
    c.res = c.cli.Post("/v0/tokens", c.hdrs, json{{"name", c.arg2}}.dump(),
                   "application/json");
    if (lh::ui::pretty() && c.res && c.res->status >= 200 && c.res->status < 300) {
      try {
        auto j = json::parse(c.res->body);
        lh::ui::Panel p(tr("Токен создан: ") + j.value("name", "?"));
        p.kv(tr("Значение"), lh::ui::c::bold(j.value("token", "?")));
        p.line("  " + lh::ui::c::dim(tr("клиенты передают его как Bearer-ключ OpenAI-порта;")));
        p.line("  " + lh::ui::c::dim(tr("значение хранится и показывается снова: lhctl tokens")));
        std::fputs(p.render().c_str(), stdout);
        return 0;
      } catch (...) {}
    }
  } else if (c.arg == "rm" || c.arg == "remove") {
    if (c.arg2.empty()) {
      std::fprintf(stderr, "%s\n", tr("tokens rm: укажите имя (lhctl tokens — список)"));
      return 4;
    }
    c.res = c.cli.Delete("/v0/tokens", c.hdrs, json{{"name", c.arg2}}.dump(),
                     "application/json");
    if (lh::ui::pretty() && c.res && c.res->status >= 200 && c.res->status < 300) {
      std::printf(tr("отозван: %s\n"), c.arg2.c_str());
      return 0;
    }
  } else {
    std::fprintf(stderr, tr("tokens: неизвестная подкоманда «%s» (list|add|rm)\n"),
                 c.arg.c_str());
    return 4;
  }

  return kTail; // сырой JSON — общий хвост
}
static int run_chat(CmdCtx& c) {
  // Один аргумент с пробелом — это вопрос с моделью по умолчанию
  // («lhctl chat \"расскажи…\"»); два аргумента — модель и вопрос.
  std::string model = c.arg, question = c.arg2;
  if (!c.arg.empty() && c.arg2.empty() && c.arg.find(' ') != std::string::npos) {
    model.clear();
    question = c.arg;
  }
  return chat_run(c.port, c.token, std::move(model), std::move(question));

  return kTail; // недостижимо: все пути выше возвращают код
}
static int run_load(CmdCtx& c) {
  if (c.arg.empty()) {
    std::fprintf(stderr, "%s\n", tr("load: укажите .gguf-файл или имя модели"));
    return 4;
  }
  if (lh::ui::pretty()) {
    // Интерактивная загрузка: спиннер + опрос до готовности (то же, что
    // делает чат перед ходом).
    if (lh::file_exists(c.arg)) {
      // Произвольный файл — по пути, как раньше; потом опрос.
      auto r2 = c.cli.Post("/v0/load", c.hdrs, json{{"path", c.arg}}.dump(),
                         "application/json");
      if (!r2) {
        std::fprintf(stderr, tr("llmhost недоступен на 127.0.0.1:%d\n"), c.port);
        return 1;
      }
      if (r2->status == 401) {
        std::fprintf(stderr, "%s", tr("lhctl: неверный токен (401) — см. data/token.txt\n"));
        return 2;
      }
      if (r2->status >= 400) {
        std::fprintf(stderr, "%s\n",
                     (tr("загрузка не началась: ") + error_message(r2->body)).c_str());
        return 3;
      }
      return wait_loaded(c.port, c.cli, c.hdrs, c.arg);
    }
    return ensure_model(c.port, c.cli, c.hdrs, c.arg); // имя/id — общий путь чата
  }
  json body;
  // Существующий файл — по пути; иначе ищем по имени в реестре.
  if (lh::file_exists(c.arg))
    body = {{"path", c.arg}};
  else
    body = {{"name", c.arg}};
  c.res = c.cli.Post("/v0/load", c.hdrs, body.dump(), "application/json");

  return kTail; // сырой JSON — общий хвост
}
static int run_search(CmdCtx& c) {
  // Поиск по каталогу HuggingFace: таблица с оценкой «влезет ли в VRAM».
  if (c.arg.empty()) {
    std::fprintf(stderr, "%s\n", tr("search: укажите запрос (lhctl search qwen3)"));
    return 4;
  }
  c.res = c.cli.Get("/v0/catalog/search", httplib::Params{{"q", c.arg}, {"limit", "16"}}, c.hdrs);
  if (lh::ui::pretty() && c.res && c.res->status >= 200 && c.res->status < 300) {
    try {
      std::fputs(render_search(json::parse(c.res->body), c.arg).c_str(), stdout);
      return 0;
    } catch (...) {}
  }
  // не-tty: сырой JSON общим хвостом ниже

  return kTail; // сырой JSON — общий хвост
}
static int run_unload(CmdCtx& c) {
  // --force — снять занятую принудительно (стоп-кран: обрывает все генерации
  // модели). На «все» (без id) демон отвечает 400 — люк только для одной.
  json b = json::object();
  if (!c.arg.empty()) b["model"] = c.arg;
  if (c.force) b["force"] = true;
  c.res = c.cli.Post("/v0/unload", c.hdrs, b.dump(), "application/json");

  return kTail; // сырой JSON — общий хвост
}
int main(int argc, char** argv) {
  // Значения по умолчанию из окружения: автодополнение и сам lhctl разделяют
  // одни и те же LHCTL_PORT / LHCTL_TOKEN.
  int port = 7331;
  {
    const char* e = getenv("LHCTL_PORT");
    if (e && *e) {
      int p = atoi(e);
      if (p > 0 && p < 65536) port = p;
    }
  }
  std::string token;
  {
    const char* e = getenv("LHCTL_TOKEN");
    if (e && *e) token = e;
  }
  std::string cmd, arg, arg2, release_tag;
  bool raw = false, yes_flag = false, pin_only = false, all_builds = false;
  bool force_flag = false;
  bool json_flag = false, watch = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--port" && i + 1 < argc) {
      port = atoi(argv[++i]);
    } else if (a == "--token" && i + 1 < argc) {
      token = argv[++i];
    } else if (a == "--release" && i + 1 < argc) {
      release_tag = argv[++i]; // engine update: конкретный релиз вместо последнего
    } else if (a == "--pin-only") {
      pin_only = true; // engine update: перепинить манифест без установки
    } else if (a == "--all") {
      all_builds = true; // engine update: все пиннутые сборки, не только активная
    } else if (a == "--raw") {
      raw = true; // models/pull: список построчно — источник для автодополнения
    } else if (a == "--json") {
      json_flag = true; // машинный JSON даже в терминале
    } else if (a == "-w" || a == "--watch") {
      watch = true; // status: живая приборная панель
    } else if (a == "--yes") {
      yes_flag = true; // rm: без вопроса (скрипты)
    } else if (a == "--force") {
      force_flag = true; // unload: снять занятую принудительно (стоп-кран)
    } else if (cmd.empty()) {
      cmd = a;
    } else if (arg.empty()) {
      arg = a;
    } else if (arg2.empty()) {
      arg2 = a; // chat: модель + вопрос
    } else {
      std::fprintf(stderr, tr("лишний аргумент: %s\n"), a.c_str());
      return 4;
    }
  }
  lh::ui::init(raw || json_flag);

  if (token.empty()) {
    // Токен той же установки: data рядом с lhctl, иначе XDG-локация демона.
    token = lh::read_small_file(lh::exe_dir() + "/data/token.txt");
    if (token.empty()) {
      const char* xdg = getenv("XDG_DATA_HOME");
      std::string base = (xdg && *xdg && xdg[0] == '/')
                             ? std::string(xdg) + "/llmhost"
                             : std::string(getenv("HOME") ? getenv("HOME") : "") +
                                   "/.local/share/llmhost";
      token = lh::read_small_file(base + "/token.txt");
    }
    if (token.empty()) {
      // Windows-раскладка с общим каталогом установки — на всякий случай.
      token = lh::read_small_file(lh::exe_dir() + "/../llmhost/data/token.txt");
    }
  }

  httplib::Client cli("127.0.0.1", port);
  cli.set_connection_timeout(std::chrono::seconds(3));
  cli.set_read_timeout(std::chrono::seconds(30));
  httplib::Headers hdrs;
  if (!token.empty()) hdrs = {{"Authorization", "Bearer " + token}};

  if (cmd.empty()) {
    // Без аргументов в терминале — полноэкранный интерфейс управления;
    // в пайпе/скрипте — справка, как всегда.
    if (lh::ui::pretty()) return lh::tui::run(port, token);
    usage();
    return 4;
  }
  if (cmd == "help") {
    usage();
    return 0;
  }

  if (cmd == "version") {
    std::puts("lhctl/llmhost " LH_VERSION);
    return 0;
  }

  CmdCtx c{cli, hdrs};
  c.port = port;
  c.token = token;
  c.arg = arg;
  c.arg2 = arg2;
  c.release_tag = release_tag;
  c.raw = raw;
  c.yes = yes_flag;
  c.force = force_flag;
  c.pin_only = pin_only;
  c.all_builds = all_builds;
  c.watch = watch;

  int rc = kTail;
  if (cmd == "status") {
    rc = run_status(c);
  } else if (cmd == "models") {
    rc = run_models(c);
  } else if (cmd == "quit") {
    rc = run_quit(c);
  } else if (cmd == "completion") {
    rc = run_completion(c);
  } else if (cmd == "show") {
    rc = run_show(c);
  } else if (cmd == "rm") {
    rc = run_rm(c);
  } else if (cmd == "ctx") {
    rc = run_ctx(c);
  } else if (cmd == "mtp") {
    rc = run_mtp(c);
  } else if (cmd == "vision") {
    rc = run_vision(c);
  } else if (cmd == "thinking") {
    rc = run_thinking(c);
  } else if (cmd == "metrics") {
    rc = run_metrics(c);
  } else if (cmd == "image") {
    rc = run_image(c);
  } else if (cmd == "set") {
    rc = run_set(c);
  } else if (cmd == "pull") {
    rc = run_pull(c);
  } else if (cmd == "engine") {
    rc = run_engine(c);
  } else if (cmd == "tokens") {
    rc = run_tokens(c);
  } else if (cmd == "chat") {
    rc = run_chat(c);
  } else if (cmd == "load") {
    rc = run_load(c);
  } else if (cmd == "search") {
    rc = run_search(c);
  } else if (cmd == "unload") {
    rc = run_unload(c);
  } else {
    std::fprintf(stderr, tr("неизвестная команда: %s\n"), cmd.c_str());
    usage();
    return 4;
  }

  return rc == kTail ? print_raw(c) : rc;
}