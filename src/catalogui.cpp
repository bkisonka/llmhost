#include "catalogui.h"

#include "catalog.h"
#include "config.h"
#include "engine.h"
#include "modelreg.h"
#include "mtp.h"
#include "pull.h"
#include "supervisor.h"
#include "util.h"
#include "winstyle.h"

#include <windows.h>
#include <commctrl.h>

#include <nlohmann/json.hpp>

#include <atomic>
#include <string>
#include <thread>

namespace lh::catalogui {

namespace {

using nlohmann::json;

constexpr wchar_t CAT_CLASS[] = L"llmhost_catalog";

constexpr int IDC_GO = 101;
constexpr int IDC_Q = 102;
constexpr int IDC_RESULTS = 103;
constexpr int IDC_FILES = 104;
constexpr int IDC_PULL = 105;
constexpr int IDC_LOAD = 106;
constexpr int IDC_PROG = 107;
constexpr int IDC_STATUS = 108;

constexpr UINT WM_CAT_SEARCH = WM_APP + 11;       // LPARAM → new std::string (json)
constexpr UINT WM_CAT_SEARCH_ERR = WM_APP + 12;
constexpr UINT WM_CAT_DETAIL = WM_APP + 13;
constexpr UINT WM_CAT_DETAIL_ERR = WM_APP + 14;
constexpr UINT WM_CAT_LOAD_ERR = WM_APP + 15; // отказ sup::load — в статус-строку

HWND g_hwnd = nullptr;
HWND g_results = nullptr, g_files = nullptr;
json g_search, g_detail;             // последние ответы каталога
int g_file_sel = -1;                 // выбранный квант
std::vector<int> g_file_kinds;       // роль строки: 0 обычный, 1 рекомендую, 2 mmproj, 3 мимо
std::string g_last_repo;             // для которого уже показаны кванты
std::atomic<bool> g_searching{false}, g_detail_busy{false};
bool g_pull_was_busy = false;        // переход «скачивание шло» → обновить флаги

std::wstring utf8_display(const std::string& s) { return utf8_to_wide(s); }

std::wstring gb_text(uint64_t b) {
  wchar_t buf[32];
  if (b >= 1ull << 30) swprintf(buf, 32, L"%.1f ГБ", b / 1073741824.0);
  else if (b >= 1ull << 20) swprintf(buf, 32, L"%.0f МБ", b / 1048576.0);
  else swprintf(buf, 32, L"%llu КБ", (unsigned long long)(b >> 10));
  return buf;
}

std::wstring params_text(double p) {
  wchar_t buf[32];
  if (p <= 0) return L"—";
  if (p >= 1e9) swprintf(buf, 32, L"%.1fB", p / 1e9);
  else swprintf(buf, 32, L"%.0fM", p / 1e6);
  return buf;
}

// Светофор: ✓ — целиком, ~ — частично (меньший контекст или часть слоёв в RAM),
// ✗ — мимо, ? — нет данных.
struct Verdict {
  std::wstring text;
  bool fits = false, partial = false;
};
Verdict verdict(const json& fit) {
  Verdict v;
  v.fits = fit.value("fits", false);
  v.partial = fit.value("partial", false);
  bool dense = fit.value("dense_partial", false);
  uint64_t need = fit.value("vram_needed", (uint64_t)0);
  if (!fit.value("gpu_known", false)) v.text = L"? VRAM неизвестна";
  else if (v.fits) v.text = L"✓ влезает, " + gb_text(need);
  else if (v.partial || dense) v.text = L"~ частично, " + gb_text(need);
  else v.text = L"✗ не влезает, " + gb_text(need);
  return v;
}

void set_status(const std::wstring& text) {
  if (g_hwnd) SetWindowTextW(GetDlgItem(g_hwnd, IDC_STATUS), text.c_str());
}

HWND add_ctrl(const wchar_t* cls, const wchar_t* text, DWORD style, int x, int y, int w, int h,
              int id) {
  HWND c = CreateWindowExW(0, cls, text, WS_CHILD | WS_VISIBLE | style, x, y, w, h, g_hwnd,
                           (HMENU)(INT_PTR)id, nullptr, nullptr);
  SendMessageW(c, WM_SETFONT, (WPARAM)lh::ui_font(), TRUE);
  return c;
}

// ---- лейаут: тянется вместе с окном ------------------------------------------

// Дизайн — клиентская область 640×560: строка поиска сверху, два списка
// делят середину, кнопки/прогресс/статус прижаты к низу.
constexpr int DESIGN_W = 640, DESIGN_H = 560;
HWND g_lbl1 = nullptr, g_lbl2 = nullptr; // подписи списков (без id)

void relayout(HWND hwnd) {
  RECT rc{};
  GetClientRect(hwnd, &rc);
  int W = rc.right, H = rc.bottom;
  if (W < 200 || H < 200) return;

  auto place = [](HWND h, int x, int y, int w, int ht) {
    if (h) MoveWindow(h, x, y, w, ht, TRUE);
  };
  place(g_lbl1, 16, 12, W - 32, 18);
  place(GetDlgItem(hwnd, IDC_Q), 16, 32, W - 16 - 108, 24);
  place(GetDlgItem(hwnd, IDC_GO), W - 16 - 90, 31, 90, 26);

  // Середина: результаты 190 / файлы 170 от свободного места.
  int avail = (H - 104) - 64; // низ списков отдаём кнопкам (H-104)
  int total = avail - 28;     // подпись второго списка + отступ
  int rh = total * 190 / 360, fh = total - rh;
  place(g_results, 16, 64, W - 16, rh);
  place(g_lbl2, 16, 64 + rh + 10, W - 32, 18);
  place(g_files, 16, 64 + rh + 10 + 18, W - 16, fh);

  place(GetDlgItem(hwnd, IDC_PULL), 16, H - 96, 160, 28);
  place(GetDlgItem(hwnd, IDC_LOAD), 186, H - 96, 170, 28);
  place(GetDlgItem(hwnd, IDC_PROG), 16, H - 58, W - 32, 20);
  place(GetDlgItem(hwnd, IDC_STATUS), 16, H - 30, W - 32, 20);
}

void insert_column(HWND lv, int idx, const wchar_t* title, int w) {
  LVCOLUMNW c{};
  c.mask = LVCF_TEXT | LVCF_WIDTH | LVCF_SUBITEM;
  c.pszText = const_cast<LPWSTR>(title);
  c.cx = w;
  c.iSubItem = idx;
  ListView_InsertColumn(lv, idx, &c);
}

void set_row(HWND lv, int i, int col, const std::wstring& text) {
  ListView_SetItemText(lv, i, col, const_cast<LPWSTR>(text.c_str()));
}

// ---- рабочие потоки: сеть не должна держать окно -----------------------------

void post_json(UINT msg, const std::string& payload) {
  PostMessageW(g_hwnd, msg, 0, (LPARAM)new std::string(payload));
}

void start_search(const std::string& q) {
  g_searching = true;
  std::thread([q] {
    std::string err;
    std::string out = catalog::search_json(q, 30, err);
    g_searching = false;
    logf("catui: поиск «%s» — %s", q.c_str(), err.empty() ? "готово" : err.c_str());
    post_json(err.empty() ? WM_CAT_SEARCH : WM_CAT_SEARCH_ERR, err.empty() ? out : err);
  }).detach();
}

void start_detail(const std::string& repo) {
  g_detail_busy = true;
  std::thread([repo] {
    std::string err;
    std::string out = catalog::detail_json(repo, err);
    g_detail_busy = false;
    logf("catui: детали «%s» — %s", repo.c_str(), err.empty() ? "готово" : err.c_str());
    post_json(err.empty() ? WM_CAT_DETAIL : WM_CAT_DETAIL_ERR, err.empty() ? out : err);
  }).detach();
}

// ---- отрисовка списков -------------------------------------------------------

void show_search(const std::string& payload) {
  try {
    g_search = json::parse(payload);
  } catch (...) { return; }
  ListView_DeleteAllItems(g_results);
  g_last_repo.clear();
  ListView_DeleteAllItems(g_files);
  auto rows = g_search.value("results", json::array());
  int i = 0;
  for (auto& r : rows) {
    LVITEMW it{};
    it.mask = LVIF_TEXT;
    it.iItem = i;
    std::wstring repo = utf8_display(r.value("repo", ""));
    it.pszText = repo.data();
    int idx = ListView_InsertItem(g_results, &it);
    auto fit = r.value("fit", json::object());
    set_row(g_results, idx, 1, params_text(fit.value("params", 0.0)));
    set_row(g_results, idx, 2,
            utf8_display(std::to_string(r.value("downloads", (uint64_t)0))));
    // В поиске оценка грубая: по имени. Не угадали параметры — честное «?».
    std::wstring vtext;
    if (!fit.value("known", false)) vtext = L"? размер неизвестен";
    else vtext = verdict(fit).text;
    set_row(g_results, idx, 3, vtext);
    ++i;
  }
  // Вердикты зависят от свободной VRAM на момент поиска — сказать это прямо.
  auto gpu = g_search.value("gpu", json::object());
  uint64_t vf = gpu.value("vram_free", 0ull), vt = gpu.value("vram_total", 0ull);
  std::wstring vram = vf ? L" (свободно " + gb_text(vf) + L" из " + gb_text(vt) + L" ГБ)" : L"";
  set_status(i ? L"найдено: " + std::to_wstring(i) +
                     L" — вердикты по состоянию на момент поиска" + vram +
                     L"; обновите поиск после выгрузки моделей"
               : L"ничего не найдено");
}

void show_detail(const std::string& payload) {
  try {
    g_detail = json::parse(payload);
  } catch (...) { return; }
  ListView_DeleteAllItems(g_files);
  g_file_sel = -1;
  g_file_kinds.clear();
  // Пометка «draft для X»: в реестре есть модель той же архитектуры, но
  // в ≥4 раза крупнее этого файла — значит файл годится ей в черновики.
  std::string arch = g_detail.value("meta", json::object()).value("arch", "");
  std::vector<lh::modelreg::Model> locals;
  if (!arch.empty()) {
    for (auto& l : lh::modelreg::scan(config::models_dir_resolved(config::current()))) {
      if (l.arch != arch || !l.meta.n_layers) continue;
      if (lh::modelreg::is_shard_continuation(l.path)) continue;
      auto copy = l;
      int i = 0, c = 1;
      lh::modelreg::shard_info(copy.path, i, c);
      if (c > 1) copy.size = lh::modelreg::shard_group_bytes(copy.path);
      locals.push_back(std::move(copy));
    }
  }
  auto files = g_detail.value("files", json::array());
  int i = 0;
  int recommend_idx = -1;
  std::string recommended = g_detail.value("recommended", "");
  std::wstring rec_quant;
  for (auto& f : files) {
    bool is_mmproj = f.value("kind", "quant") == "mmproj";
    bool is_mtp = f.value("kind", "quant") == "mtp";
    LVITEMW it{};
    it.mask = LVIF_TEXT;
    it.iItem = i;
    std::wstring quant = utf8_display(f.value("quant", "?"));
    if (f.value("parts", 1) > 1) quant += L" ×" + std::to_wstring(f.value("parts", 1));
    bool is_rec = !recommended.empty() && f.value("filter", "") == recommended;
    if (is_rec) {
      quant.insert(0, L"★ ");
      rec_quant = quant;
    }
    it.pszText = quant.data();
    int idx = ListView_InsertItem(g_files, &it);

    int kind = 0; // 0 обычный · 1 рекомендованный · 2 mmproj/mtp · 3 не влезает
    if (is_mmproj || is_mtp) kind = 2;
    else if (is_rec) kind = 1;
    set_row(g_files, idx, 1, gb_text(f.value("size", (uint64_t)0)));
    json fj = f.value("fit", json::object());
    Verdict v = verdict(fj);
    set_row(g_files, idx, 2, (is_mmproj || is_mtp) ? L"—" : v.text);
    uint32_t mctx = fj.value("max_ctx", 0u);
    set_row(g_files, idx, 3,
            (is_mmproj || is_mtp) ? L"—"
                                  : (mctx ? (std::wstring(L"до ") + std::to_wstring(mctx) + L" ток.") : L"—"));
    std::wstring status;
    if (is_mmproj) {
      status = f.value("downloaded", false)
                   ? L"скачано ✓ — нужна для картинок (вместе с моделью)"
                   : L"vision-проекция — нужна для картинок (качается вместе с моделью)";
    } else if (is_mtp) {
      status = f.value("downloaded", false)
                   ? L"скачано ✓ — включится при загрузке модели"
                   : L"MTP-модуль — ускоряет генерацию этой модели (~+25%)";
    } else {
      bool moe_cpu = fj.value("moe_cpu", false);
      bool dense = fj.value("dense_partial", false);
      bool downloaded = f.value("downloaded", false);
      if (downloaded) status = L"скачано ✓";
      else if (moe_cpu) status = L"влезет (MoE: эксперты на CPU)";
      else if (dense)
        status = L"влезет частично — " + std::to_wstring(fj.value("ngl", 0)) + L" из " +
                 std::to_wstring(fj.value("n_layers", 0)) +
                 L" слоёв на GPU, хвост в RAM (медленнее)";
      else if (v.fits) status = L"можно качать";
      else if (v.partial) status = L"частично — меньше контекст";
      else status = L"мимо";
      if (!downloaded && !v.fits && !v.partial && !dense) kind = 3;
      // Годится в черновики к уже скачанной крупной модели той же семьи?
      if (!locals.empty()) {
        uint64_t sz = f.value("size", (uint64_t)0);
        for (auto& big : locals) {
          if (big.size >= sz * 4) {
            status += L"  · draft для «" + utf8_display(big.name) + L"»";
            break;
          }
        }
      }
    }
    set_row(g_files, idx, 4, status);
    g_file_kinds.push_back(kind);
    if (is_rec) recommend_idx = idx;
    ++i;
  }
  if (recommend_idx >= 0)
    ListView_SetItemState(g_files, recommend_idx, LVIS_SELECTED | LVIS_FOCUSED,
                          LVIS_SELECTED | LVIS_FOCUSED);
  bool precise = g_detail.value("meta", json::object()).value("precise", false);
  int mm = g_detail.value("mmproj_files", 0);
  std::wstring mmnote =
      mm ? L" · vision-проекции (mmproj) внизу списка — нужны только для картинок"
         : L"";
  int mtpn = g_detail.value("mtp_files", 0);
  std::wstring mtpnote =
      mtpn ? L" · MTP-модуль внизу — ускоритель генерации, предложим при загрузке модели"
           : L"";
  std::wstring recw = rec_quant;
  std::wstring recnote = recw.empty()
                             ? L""
                             : L" · ★ Рекомендую: " + recw +
                                   L" — наибольший квант, влезающий целиком";
  set_status((precise ? L"оценка по реальному заголовку GGUF"
                      : L"заголовок GGUF не дочитан — оценка грубая, по имени модели") +
             recnote + mmnote + mtpnote);
}

// Что выбрали в списке квантов: фильтр для pull и локальный путь первого шарда.
bool selected_group(std::string& filter, std::wstring& local_path) {
  if (g_file_sel < 0 || !g_detail.is_object()) return false;
  auto files = g_detail.value("files", json::array());
  if (g_file_sel >= (int)files.size()) return false;
  auto& f = files[g_file_sel];
  filter = f.value("filter", "");
  std::string repo = g_detail.value("repo", "");
  config::Config cfg = config::current();
  std::wstring dir = config::models_dir_resolved(cfg);
  local_path = dir + L"\\" + utf8_to_wide(repo) + L"\\" + utf8_to_wide(f.value("path", ""));
  for (wchar_t& c : local_path)
    if (c == L'/') c = L'\\';
  return !filter.empty();
}

void update_buttons() {
  if (!g_hwnd) return;
  bool net_busy = pull::busy() || engine::busy();
  EnableWindow(GetDlgItem(g_hwnd, IDC_PULL), !net_busy && g_file_sel >= 0);
  std::string filter;
  std::wstring local;
  bool have = selected_group(filter, local);
  EnableWindow(GetDlgItem(g_hwnd, IDC_LOAD),
               have && !net_busy && lh::file_exists(local) &&
                   sup::state() != sup::St::Loading);
}

void start_pull() {
  std::string filter;
  std::wstring local;
  if (!selected_group(filter, local)) return;
  std::string repo = g_detail.value("repo", "");
  std::string e;
  if (!pull::start(repo, filter, e)) set_status(utf8_display(e));
  else set_status(L"скачивание началось — прогресс внизу");
  update_buttons();
}

void start_load() {
  std::string filter;
  std::wstring local;
  if (!selected_group(filter, local) || !lh::file_exists(local)) return;
  // MTP-модуль сам по себе не запускается: он прицепится к основной модели.
  if (g_detail.value("files", json::array())[g_file_sel].value("kind", "quant") == "mtp") {
    set_status(L"MTP-модуль отдельно не грузят — он предложится при загрузке основной модели");
    return;
  }
  std::thread([p = local] {
    std::string e;
    if (sup::load(p, e)) {
      lh::mtp::after_load(p); // модуль уже скачан/доступен? — спросим про ускорение
      return;
    }
    logf("catalog: загрузка не удалась: %s", e.c_str());
    post_json(WM_CAT_LOAD_ERR, e); // не молчим: отказ виден в статус-строке
  }).detach();
  set_status(L"загрузка модели началась — статус в трее");
}

void progress(uint64_t done, uint64_t total) {
  HWND p = GetDlgItem(g_hwnd, IDC_PROG);
  if (!total) return;
  SendMessageW(p, PBM_SETRANGE32, 0, 1000);
  SendMessageW(p, PBM_SETPOS, (WPARAM)(1000.0 * done / total), 0);
}

void tick() {
  // Прогресс и статус — раз в полсекунды; заодно состояние кнопок.
  if (pull::busy()) {
    auto p = json::parse(pull::status_json(), nullptr, false);
    if (!p.is_null()) {
      progress(p.value("done", 0ull), p.value("total", 0ull));
      uint64_t total = p.value("total", 0ull), done = p.value("done", 0ull);
      double speed = p.value("speed", 0.0);
      wchar_t buf[256];
      std::wstring repo = utf8_display(p.value("repo", ""));
      std::wstring file = utf8_display(p.value("file", ""));
      if (total)
        swprintf(buf, 256, L"Скачивание %s: %.0f%% (%.1f МБ/с) — %s", repo.c_str(),
                 100.0 * done / total, speed / 1048576.0, file.c_str());
      else
        swprintf(buf, 256, L"Скачивание %s…", repo.c_str());
      set_status(buf);
    }
    g_pull_was_busy = true;
  } else if (engine::busy()) {
    auto e = json::parse(engine::status_json(), nullptr, false);
    if (!e.is_null()) {
      progress(e.value("done", 0ull), e.value("total", 0ull));
      set_status(L"Скачивание llama-server…");
    }
    g_pull_was_busy = true;
  } else {
    if (g_pull_was_busy) {
      // Скачивание кончилось — флаги «скачано» могли поменяться.
      g_pull_was_busy = false;
      progress(0, 1);
      if (!g_last_repo.empty()) start_detail(g_last_repo);
    }
  }
  update_buttons();
}

LRESULT CALLBACK wndproc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
  g_hwnd = hwnd;
  switch (msg) {
    case WM_CREATE: {
      g_lbl1 = CreateWindowExW(0, L"STATIC", L"Поиск по HuggingFace (GGUF):", WS_CHILD |
                              WS_VISIBLE, 0, 0, 0, 0, g_hwnd, nullptr, nullptr, nullptr);
      SendMessageW(g_lbl1, WM_SETFONT, (WPARAM)lh::ui_font(), TRUE);
      add_ctrl(L"EDIT", nullptr, WS_TABSTOP | WS_BORDER | ES_AUTOHSCROLL, 0, 0, 0, 0, IDC_Q);
      ws::button(g_hwnd, IDC_GO, 0, 0, 90, 26, L"Найти", nullptr, /*def=*/true);

      g_results = add_ctrl(WC_LISTVIEWW, nullptr,
                           WS_TABSTOP | LVS_REPORT | LVS_SHOWSELALWAYS | LVS_SINGLESEL |
                               WS_BORDER,
                           0, 0, 0, 0, IDC_RESULTS);
      ListView_SetExtendedListViewStyle(g_results, LVS_EX_FULLROWSELECT);
      insert_column(g_results, 0, L"Репозиторий", 290);
      insert_column(g_results, 1, L"Параметры", 70);
      insert_column(g_results, 2, L"Скачиваний", 80);
      insert_column(g_results, 3, L"Влезет", 160);

      g_lbl2 = CreateWindowExW(0, L"STATIC", L"Кванты выбранной модели:", WS_CHILD |
                              WS_VISIBLE, 0, 0, 0, 0, g_hwnd, nullptr, nullptr, nullptr);
      SendMessageW(g_lbl2, WM_SETFONT, (WPARAM)lh::ui_font(), TRUE);
      g_files = add_ctrl(WC_LISTVIEWW, nullptr,
                         WS_TABSTOP | LVS_REPORT | LVS_SHOWSELALWAYS | LVS_SINGLESEL |
                             WS_BORDER,
                         0, 0, 0, 0, IDC_FILES);
      ListView_SetExtendedListViewStyle(g_files, LVS_EX_FULLROWSELECT);
      insert_column(g_files, 0, L"Квант", 100);
      insert_column(g_files, 1, L"Размер", 90);
      insert_column(g_files, 2, L"Нужно VRAM", 100);
      insert_column(g_files, 3, L"Контекст", 110);
      insert_column(g_files, 4, L"Статус", 180);

      ws::button(g_hwnd, IDC_PULL, 0, 0, 160, 28, L"Скачать выбранное");
      ws::button(g_hwnd, IDC_LOAD, 0, 0, 170, 28, L"Загрузить скачанное");
      HWND prog = add_ctrl(PROGRESS_CLASSW, nullptr, PBS_SMOOTH, 0, 0, 0, 0, IDC_PROG);
      SendMessageW(prog, PBM_SETRANGE32, 0, 1000);
      add_ctrl(L"STATIC", L"", SS_LEFT | SS_ENDELLIPSIS, 0, 0, 0, 0, IDC_STATUS);
      relayout(hwnd); // WM_SIZE пришёл до создания детей — раскладываем сами
      return 0;
    }
    case WM_SIZE:
      relayout(hwnd);
      ws::round_window(hwnd);
      return 0;
    case WM_GETMINMAXINFO: {
      // Минимум — дизайн-размер (внешний): меньше списки схлопнутся.
      RECT r{0, 0, DESIGN_W, DESIGN_H};
      AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
      ws::min_size(lp, r.right - r.left, r.bottom - r.top);
      return 0;
    }
    case WM_DRAWITEM:
      ws::draw_button((const DRAWITEMSTRUCT*)lp);
      return TRUE;
    case WM_TIMER:
      if (wp == 1) tick();
      return 0;
    case WM_CAT_SEARCH:
    case WM_CAT_SEARCH_ERR:
    case WM_CAT_DETAIL:
    case WM_CAT_DETAIL_ERR:
    case WM_CAT_LOAD_ERR: {
      std::string* s = (std::string*)lp;
      if (msg == WM_CAT_SEARCH) show_search(*s);
      else if (msg == WM_CAT_DETAIL) show_detail(*s);
      else if (msg == WM_CAT_LOAD_ERR) set_status(L"загрузка не удалась: " + utf8_display(*s));
      else set_status((msg == WM_CAT_SEARCH_ERR ? L"поиск не удался: " : L"не читается: ") +
                      utf8_display(*s));
      delete s;
      return 0;
    }
    case WM_NOTIFY: {
      auto* nm = (NMLISTVIEW*)lp;
      // Цвета списка квантов: ★ рекомендую — зелёным, mmproj — серым,
      // заведомо не влезающие — тёмно-красным.
      if (nm->hdr.code == NM_CUSTOMDRAW && nm->hdr.idFrom == IDC_FILES) {
        auto* cd = (NMLVCUSTOMDRAW*)lp;
        if (cd->nmcd.dwDrawStage == CDDS_PREPAINT) return CDRF_NOTIFYITEMDRAW;
        if (cd->nmcd.dwDrawStage == CDDS_ITEMPREPAINT) {
          int ri = (int)cd->nmcd.dwItemSpec;
          if (ri >= 0 && ri < (int)g_file_kinds.size()) {
            switch (g_file_kinds[ri]) {
              case 1: cd->clrText = RGB(0, 128, 60); return CDRF_NEWFONT;
              case 2: cd->clrText = RGB(112, 112, 112); return CDRF_NEWFONT;
              case 3: cd->clrText = RGB(178, 34, 34); return CDRF_NEWFONT;
            }
          }
          return CDRF_DODEFAULT;
        }
      }
      if (nm->hdr.idFrom == IDC_RESULTS && nm->hdr.code == LVN_ITEMCHANGED &&
          (nm->uChanged & LVIF_STATE) && (nm->uNewState & LVIS_SELECTED)) {
        if (!g_search.is_object()) return 0;
        auto rows = g_search.value("results", json::array());
        if (nm->iItem < 0 || nm->iItem >= (int)rows.size()) return 0;
        std::string repo = rows[nm->iItem].value("repo", "");
        if (repo == g_last_repo || g_detail_busy) return 0;
        g_last_repo = repo;
        ListView_DeleteAllItems(g_files);
        set_status(L"читаю " + utf8_display(repo) + L"…");
        start_detail(repo);
      } else if (nm->hdr.idFrom == IDC_FILES && nm->hdr.code == LVN_ITEMCHANGED &&
                 (nm->uChanged & LVIF_STATE) && (nm->uNewState & LVIS_SELECTED)) {
        g_file_sel = nm->iItem;
        update_buttons();
      }
      return 0;
    }
    case WM_COMMAND: {
      int id = LOWORD(wp);
      if (id == IDC_GO) {
        wchar_t buf[256] = {};
        GetWindowTextW(GetDlgItem(g_hwnd, IDC_Q), buf, 256);
        std::string q = wide_to_utf8(buf);
        if (!q.empty() && !g_searching) {
          set_status(L"ищу…");
          start_search(q);
        }
      } else if (id == IDC_PULL) {
        start_pull();
      } else if (id == IDC_LOAD) {
        start_load();
      }
      return 0;
    }
    case WM_CLOSE:
      ShowWindow(hwnd, SW_HIDE);
      return 0;
    default:
      return DefWindowProcW(hwnd, msg, wp, lp);
  }
}

} // namespace

void open() {
  static const bool comctl = [] {
    INITCOMMONCONTROLSEX icc{sizeof(icc), ICC_LISTVIEW_CLASSES | ICC_PROGRESS_CLASS};
    InitCommonControlsEx(&icc);
    return true;
  }();
  (void)comctl;

  if (g_hwnd && IsWindow(g_hwnd)) {
    ShowWindow(g_hwnd, SW_SHOWNORMAL);
    SetForegroundWindow(g_hwnd);
    return;
  }

  WNDCLASSEXW wc{};
  wc.cbSize = sizeof(wc);
  wc.lpfnWndProc = wndproc;
  wc.hInstance = GetModuleHandleW(nullptr);
  wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
  wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
  wc.hIcon = wc.hIconSm = LoadIconW(GetModuleHandleW(nullptr), MAKEINTRESOURCEW(1));
  wc.lpszClassName = CAT_CLASS;
  RegisterClassExW(&wc);

  // Размер окна с запасом на рамку и заголовок: клиент ~640×560; границы
  // перетаскиваемые — лейаут тянется в relayout().
  DWORD style = WS_OVERLAPPEDWINDOW;
  RECT r{0, 0, DESIGN_W, DESIGN_H};
  AdjustWindowRect(&r, style, FALSE);
  CreateWindowExW(0, CAT_CLASS, L"llmhost — каталог моделей", style,
                  CW_USEDEFAULT, CW_USEDEFAULT, r.right - r.left, r.bottom - r.top,
                  nullptr, nullptr, wc.hInstance, nullptr);
  if (g_hwnd) {
    ws::round_window(g_hwnd);
    SetTimer(g_hwnd, 1, 500, nullptr);
    ShowWindow(g_hwnd, SW_SHOWNORMAL); // создание без WS_VISIBLE — окно скрыто
    SetForegroundWindow(g_hwnd);
  }
}

bool filter(MSG& msg) {
  return g_hwnd && IsWindow(g_hwnd) && IsDialogMessageW(g_hwnd, &msg) != 0;
}

} // namespace lh::catalogui
