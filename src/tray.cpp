#include "tray.h"

#include "bench.h"
#include "catalogui.h"
#include "chatui.h"
#include "config.h"
#include "engine.h"
#include "modelreg.h"
#include "mtp.h"
#include "pull.h"
#include "supervisor.h"
#include "util.h"
#include "winstyle.h"
#include "wizard.h"

#include <nlohmann/json.hpp>
#include <shellapi.h>
#include <shlobj.h>
#include <commdlg.h>

#include <string>
#include <vector>
#include <thread>

namespace lh::tray {

namespace {

constexpr UINT WM_APP_TRAY = WM_APP + 1;   // клики по иконке
constexpr UINT WM_APP_STATE = WM_APP + 2;  // статус поменялся (из чужих потоков)
constexpr UINT WM_APP_QUIT = WM_APP + 3;

constexpr int IDM_MODEL_BASE = 2000;       // + индекс в списке моделей
constexpr int IDM_DEL_BASE = 2100;         // + индекс в подменю удаления
constexpr int IDM_PART_BASE = 2200;        // недокачанное: докачать
constexpr int IDM_PARTDEL_BASE = 2300;     // недокачанное: удалить .part
constexpr int IDM_UNLOAD = 9001;
constexpr int IDM_OPEN_DIR = 9002;
constexpr int IDM_SETTINGS = 9003;
constexpr int IDM_QUIT = 9004;
constexpr int IDM_ENGINE = 9005;
constexpr int IDM_CATALOG = 9006;
constexpr int IDM_BENCH = 9007;
constexpr int IDM_CHAT = 9008;
constexpr int IDM_CLI = 9009;
constexpr int IDM_WIZARD = 9010;
constexpr int IDM_ENGINE_BASE = 9020;  // + индекс сборки в engine::list()

constexpr wchar_t TRAY_CLASS[] = L"llmhost_tray";
constexpr wchar_t SET_CLASS[] = L"llmhost_settings";

// Ресурс 1 — app.ico (src/app.rc).
HICON app_icon() {
  HICON h = LoadIconW(GetModuleHandleW(nullptr), MAKEINTRESOURCEW(1));
  return h ? h : LoadIconW(nullptr, IDI_APPLICATION);
}

HWND g_hwnd = nullptr;
HWND g_set_hwnd = nullptr;
NOTIFYICONDATAW g_nid{};
// Снимок списка моделей на момент постройки меню: WM_COMMAND приходит позже,
// а пересканированная в момент клика папка могла уже измениться (индекс
// указал бы на другую модель). Оба касания — из UI-потока.
std::vector<std::pair<std::wstring, std::string>> g_menu_models; // (path, quant)
std::vector<std::wstring> g_del_models; // снимок путей для подменю удаления
struct PartEntry { std::wstring path, repo, file; };
std::vector<PartEntry> g_parts;         // снимок недокачанных для подменю

std::wstring models_dir_resolved() {
  return config::models_dir_resolved(config::current());
}

// ---- статус-строка для тултипа и заголовка меню -----------------------------

std::wstring gb(uint64_t bytes) { // «5,2»
  wchar_t buf[32];
  swprintf(buf, 32, L"%.1f", bytes / 1073741824.0);
  std::wstring s = buf;
  size_t dot = s.find(L'.');
  if (dot != std::wstring::npos) s[dot] = L',';
  return s;
}

// VRAM для подсказки: сумма по всем картам (мульти-GPU), иначе поле gpu.
void vram_sums(const nlohmann::json& j, uint64_t& total, uint64_t& free_vram) {
  total = free_vram = 0;
  auto gpus = j.value("gpus", nlohmann::json::array());
  if (!gpus.empty()) {
    for (auto& g : gpus) {
      total += g.value("vram_total", 0ull);
      free_vram += g.value("vram_free", 0ull);
    }
    return;
  }
  auto gpu = j.value("gpu", nlohmann::json::object());
  total = gpu.value("vram_total", 0ull);
  free_vram = gpu.value("vram_free", 0ull);
}

std::wstring status_line() {
  // Прогресс скачиваний и замера важнее всего остального.
  if (bench::busy()) {
    auto b = nlohmann::json::parse(bench::status_json(), nullptr, false);
    if (!b.is_null()) {
      std::wstring cur = utf8_to_wide(b.value("current", ""));
      if (cur.empty()) cur = L"…";
      return L"Замер движков: " + cur;
    }
  }
  if (pull::busy()) {
    auto p = nlohmann::json::parse(pull::status_json(), nullptr, false);
    if (!p.is_null()) {
      uint64_t done = p.value("done", 0ull), total = p.value("total", 0ull);
      wchar_t buf[192];
      if (total)
        swprintf(buf, 192, L"Скачивание %s: %.0f%%",
                 utf8_to_wide(p.value("repo", "")).c_str(), 100.0 * done / total);
      else
        swprintf(buf, 192, L"Скачивание %s…", utf8_to_wide(p.value("repo", "")).c_str());
      return buf;
    }
  }
  if (engine::busy()) {
    auto e = nlohmann::json::parse(engine::status_json(), nullptr, false);
    if (!e.is_null()) {
      uint64_t done = e.value("done", 0ull), total = e.value("total", 0ull);
      wchar_t buf[96];
      if (total) swprintf(buf, 96, L"Скачивание llama-server: %.0f%%", 100.0 * done / total);
      else swprintf(buf, 96, L"Скачивание llama-server…");
      return buf;
    }
  }

  auto j = sup::status_json();
  std::string st = j.value("state", "?");
  // Резидентных моделей может быть несколько — перечисляем.
  uint64_t total = 0, free_vram = 0;
  vram_sums(j, total, free_vram);
  std::wstring vram;
  if (total > 0 && free_vram <= total)
    vram = L" · VRAM " + gb(total - free_vram) + L"/" + gb(total) + L" ГБ";
  if (st == "loading") return L"Загрузка модели…" + vram;
  if (st == "loaded") {
    std::wstring names;
    int n = 0;
    for (auto& m : j.value("models", nlohmann::json::array()))
      if (m.value("state", "") == "loaded") {
        if (n) names += L", ";
        names += utf8_to_wide(m.value("id", ""));
        ++n;
      }
    return L"Загружено " + std::to_wstring(n) + L": " + names + vram;
  }
  if (st == "error") return L"Ошибка: " + utf8_to_wide(j.value("error", ""));
  return L"Модели не загружены — запрос к API разбудит нужную" + vram;
}

// Адрес OpenAI-эндпоинта — то, что вписывают во внешние клиенты (порт из
// настроек). Контрольный :7331 внутрь не тащим: он не для моделей.
std::wstring api_line() {
  wchar_t buf[96];
  swprintf(buf, 96, L"API: http://127.0.0.1:%d/v1", config::current().openai_port);
  return buf;
}

void update_tooltip() {
  std::wstring text = L"llmhost — " + status_line();
  wcsncpy_s(g_nid.szTip, text.c_str(), _TRUNCATE);
  g_nid.uFlags = NIF_TIP;
  Shell_NotifyIconW(NIM_MODIFY, &g_nid);
}

// ---- автозагрузка -----------------------------------------------------------

bool autostart_on() {
  HKEY k;
  if (RegOpenKeyExW(HKEY_CURRENT_USER, L"Software\\Microsoft\\Windows\\CurrentVersion\\Run",
                    0, KEY_READ, &k) != ERROR_SUCCESS)
    return false;
  DWORD type = 0, size = 0;
  bool present = RegQueryValueExW(k, L"llmhost", nullptr, &type, nullptr, &size) == ERROR_SUCCESS;
  RegCloseKey(k);
  return present;
}

void apply_autostart(bool on) {
  HKEY k;
  if (RegCreateKeyExW(HKEY_CURRENT_USER, L"Software\\Microsoft\\Windows\\CurrentVersion\\Run",
                      0, nullptr, 0, KEY_SET_VALUE, nullptr, &k, nullptr) != ERROR_SUCCESS)
    return;
  if (on) {
    std::wstring cmd = L"\"" + exe_path() + L"\"";
    RegSetValueExW(k, L"llmhost", 0, REG_SZ, (const BYTE*)cmd.c_str(),
                   (DWORD)((cmd.size() + 1) * sizeof(wchar_t)));
  } else {
    RegDeleteValueW(k, L"llmhost");
  }
  RegCloseKey(k);
}

// ---- меню -------------------------------------------------------------------

void load_async(const std::wstring& path) {
  std::thread([p = path] {
    std::string e;
    if (sup::load(p, e)) {
      mtp::after_load(p); // есть MTP-модуль и решение не принято? — замер и вопрос
      return;
    }
    logf("tray: загрузка не удалась: %s", e.c_str());
    // Раньше отказ уходил только в лог — клик выглядел как ноль реакции.
    std::wstring msg = utf8_to_wide(e);
    tray::balloon(e.find("уже загружена") != std::string::npos
                      ? msg
                      : L"Загрузка не удалась: " + msg);
  }).detach();
}

// ---- удаление модели ---------------------------------------------------------

// Удалить модель: выгрузить (запущенный llama-server держит файл через mmap —
// занятый файл Windows не удалит), затем отправить файлы в корзину. Шардированная
// модель (…-00001-of-00003.gguf) удаляется вся. Реестр подхватит вотчер
// супервизора за пару секунд. Пустая строка — успех.
std::wstring delete_model(const std::wstring& path) {
  // Резидентную модель сначала выгрузить: занятый файл Windows не удалит.
  // Занятая генерацией не выселяется — удаление честно отменяем.
  std::string id = sup::id_for_path(path);
  if (!id.empty()) {
    std::string uerr;
    sup::harness_unload_models(id, uerr);
    if (!uerr.empty()) return L"Модель занята генерацией — выгрузить нельзя";
    for (int i = 0; i < 40; ++i) { // до 10 с: ребёнок выходит не мгновенно
      auto st = sup::models_state();
      std::string s = st.contains(id) ? st[id].value("state", "unloaded") : "unloaded";
      if (s != "loaded" && s != "loading") break;
      Sleep(250);
    }
  }

  // Группа шардов: «base-00001-of-00003.gguf» → все «base-*.gguf» рядом.
  size_t slash = path.find_last_of(L"\\/");
  std::wstring dir = slash == std::wstring::npos ? L"." : path.substr(0, slash);
  std::wstring file = slash == std::wstring::npos ? path : path.substr(slash + 1);
  std::vector<std::wstring> files{path};
  const size_t of = file.rfind(L"-of-");
  auto five_digits = [&](size_t end) { // 5 цифр, кончаясь на end (не включая)
    if (end < 5) return false;
    for (size_t k = end - 5; k < end; ++k)
      if (!iswdigit(file[k])) return false;
    return true;
  };
  if (of != std::wstring::npos && of >= 5 && of + 9 <= file.size() &&
      five_digits(of) && five_digits(of + 4 + 5) && file.substr(of + 4 + 5) == L".gguf") {
    std::wstring prefix = file.substr(0, of);
    WIN32_FIND_DATAW fd{};
    HANDLE h = FindFirstFileW((dir + L"\\" + prefix + L"-*.gguf").c_str(), &fd);
    if (h != INVALID_HANDLE_VALUE) {
      files.clear();
      do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
          files.push_back(dir + L"\\" + fd.cFileName);
      } while (FindNextFileW(h, &fd));
      FindClose(h);
    }
  }
  if (files.empty()) files.push_back(path);

  // В корзину, не насмерть: ошиблись — восстановимо из проводника.
  std::wstring list;
  for (auto& f : files) list += f;
  list += L'\0';
  list += L'\0';
  SHFILEOPSTRUCTW op{};
  op.hwnd = nullptr;
  op.wFunc = FO_DELETE;
  op.pFrom = list.c_str();
  op.fFlags = FOF_ALLOWUNDO | FOF_SILENT | FOF_NOCONFIRMATION | FOF_NOERRORUI;
  if (SHFileOperationW(&op) != 0 || op.fAnyOperationsAborted)
    return L"файл не удаляется — выгрузите модель и убедитесь, что её не читают";
  return {};
}

void show_menu() {
  HMENU m = CreatePopupMenu();
  HMENU models = CreatePopupMenu();

  // Заголовок со статусом — некликабельный; второй строкой — адрес API.
  AppendMenuW(m, MF_STRING | MF_DISABLED, 0, status_line().c_str());
  AppendMenuW(m, MF_STRING | MF_DISABLED, 0, api_line().c_str());
  AppendMenuW(m, MF_SEPARATOR, 0, nullptr);

  auto list = modelreg::scan(models_dir_resolved());
  std::wstring loaded_path;
  sup::St st = sup::state(loaded_path);
  nlohmann::json live = sup::models_state();
  g_menu_models.clear();
  for (size_t i = 0; i < list.size() && g_menu_models.size() < 40; ++i) {
    // MTP-модуль — не самостоятельная модель: в меню не показываем, но индекс
    // g_menu_models держим (id пунктов = база + индекс скана).
    if (modelreg::is_mtp_module(list[i].path) ||
        modelreg::is_shard_continuation(list[i].path)) {
      g_menu_models.push_back({L"", std::string()});
      continue;
    }
    std::wstring label = utf8_to_wide(list[i].name);
    if (!list[i].quant.empty() && list[i].quant != "?") label += L"  [" + utf8_to_wide(list[i].quant) + L"]";
    // Галочка у каждой загруженной (резидентов несколько).
    std::string id = sup::id_for_path(list[i].path);
    const nlohmann::json* s = nullptr;
    if (!id.empty() && live.is_object() && live.contains(id) && live[id].is_object())
      s = &live[id];
    std::string ms = s ? s->value("state", "unloaded") : "unloaded";
    if (ms == "loaded") {
      AppendMenuW(models, MF_STRING | MF_CHECKED, IDM_MODEL_BASE + (int)i, label.c_str());
    } else {
      if (ms == "loading") label += L"  (грузится…)";
      else if (s && s->value("failed", false)) label += L"  (не загрузилась)";
      AppendMenuW(models, MF_STRING, IDM_MODEL_BASE + (int)i, label.c_str());
    }
    g_menu_models.push_back({list[i].path, list[i].quant});
  }
  if (g_menu_models.empty())
    AppendMenuW(models, MF_STRING | MF_DISABLED, 0, L"(папка пуста — см. настройки)");
  InsertMenuW(m, 0xFFFFFFFF, MF_POPUP | MF_BYPOSITION, (UINT_PTR)models, L"Модель");

  // Недокачанные загрузки: .part-хвосты после обрыва/перезапуска — докачать
  // или убрать. Только когда они есть; во время активной докачки пункт гасится.
  auto parts = pull::scan_parts(models_dir_resolved());
  if (!parts.empty()) {
    HMENU partsm = CreatePopupMenu();
    g_parts.clear();
    int n = 0;
    for (auto& p : parts) {
      AppendMenuW(partsm, MF_STRING | MF_DISABLED, 0,
                  (L"«" + p.file + L"» — недокачано (" + gb(p.size) + L" ГБ)").c_str());
      bool can_resume = !p.repo.empty() && !pull::busy();
      AppendMenuW(partsm, MF_STRING | (can_resume ? 0 : MF_GRAYED), IDM_PART_BASE + n,
                  L"Докачать");
      AppendMenuW(partsm, MF_STRING, IDM_PARTDEL_BASE + n, L"Удалить загрузку");
      AppendMenuW(partsm, MF_SEPARATOR, 0, nullptr);
      g_parts.push_back({p.path, p.repo, p.file});
      if (++n >= 10) break;
    }
    InsertMenuW(m, 0xFFFFFFFF, MF_POPUP | MF_BYPOSITION, (UINT_PTR)partsm,
                L"Недокачанные загрузки");
  }

  // Подменю удаления: файл уходит в корзину, из папки/меню/пресета — тоже.
  HMENU del = CreatePopupMenu();
  g_del_models.clear();
  for (size_t i = 0; i < list.size() && g_del_models.size() < 40; ++i) {
    std::wstring label = utf8_to_wide(list[i].name);
    if (!list[i].quant.empty() && list[i].quant != "?")
      label += L"  [" + utf8_to_wide(list[i].quant) + L"]";
    AppendMenuW(del, MF_STRING, IDM_DEL_BASE + (int)i, label.c_str());
    g_del_models.push_back(list[i].path);
  }
  if (g_del_models.empty())
    AppendMenuW(del, MF_STRING | MF_DISABLED, 0, L"(папка пуста)");
  InsertMenuW(m, 0xFFFFFFFF, MF_POPUP | MF_BYPOSITION, (UINT_PTR)del, L"Удалить модель");

  // Выгрузить можно только загруженные; спящая уже выгружена — не даём
  // «мёртвого» пункта, из-за которого кажется, что реакция не срабатывает.
  if (st == sup::St::Loaded)
    AppendMenuW(m, MF_STRING, IDM_UNLOAD, L"Выгрузить все");
  else if (st == sup::St::Asleep)
    AppendMenuW(m, MF_STRING | MF_GRAYED, IDM_UNLOAD, L"Выгружено (разбудит запрос)");
  else
    AppendMenuW(m, MF_STRING | MF_GRAYED, IDM_UNLOAD, L"Выгрузить");
  AppendMenuW(m, MF_SEPARATOR, 0, nullptr);
  // Подменю движка: сборки как радио (что реально выберется: явный
  // engine_build, иначе — по железу: NVIDIA → CUDA), бенч — рядом.
  {
    HMENU eng = CreatePopupMenu();
    std::string chosen = config::current().engine_build;
    if (chosen.empty()) chosen = engine::default_build();
    if (engine::busy())
      AppendMenuW(eng, MF_STRING | MF_DISABLED, 0, L"Скачивание движка…");
    // Копия, а не ссылка: engine::list() возвращает вектор по значению, и
    // «const auto& b = engine::list()[i]» держал висячую ссылку на временный —
    // когда куча переиспользовала блок под строку подписи, название сборки
    // затиралось и в меню оставалось голое «(установлен)».
    const std::vector<engine::BuildInfo> builds = engine::list();
    for (size_t i = 0; i < builds.size() && i < 8; ++i) {
      const engine::BuildInfo& b = builds[i];
      if (!b.pinned && !b.installed) continue; // не пиннута — ставить нечем
      std::wstring label = utf8_to_wide(b.id);
      if (b.installed)
        label += chosen == b.id ? L"  ✓ используется" : L" (установлен)";
      else
        label += L" — скачать ~" + std::to_wstring(b.size >> 20) + L" МБ";
      AppendMenuW(eng, MF_STRING | (chosen == b.id ? MF_CHECKED : 0),
                  IDM_ENGINE_BASE + (int)i, label.c_str());
    }
    AppendMenuW(eng, MF_SEPARATOR, 0, nullptr);
    if (bench::busy())
      AppendMenuW(eng, MF_STRING | MF_DISABLED, 0, L"Замер движков идёт…");
    else
      AppendMenuW(eng, MF_STRING, IDM_BENCH, L"Замерить скорость движков…");
    AppendMenuW(m, MF_POPUP, (UINT_PTR)eng, L"Движок ▸");
  }
  AppendMenuW(m, MF_STRING, IDM_CATALOG, L"Каталог моделей…");
  AppendMenuW(m, MF_STRING, IDM_WIZARD, L"Быстрая установка модели…");
  AppendMenuW(m, MF_STRING, IDM_CHAT, L"Чат…");
  AppendMenuW(m, MF_STRING, IDM_OPEN_DIR, L"Открыть папку моделей");
  AppendMenuW(m, MF_STRING, IDM_CLI, L"Открыть CLI (lhctl)…");
  AppendMenuW(m, MF_STRING, IDM_SETTINGS, L"Настройки…");
  AppendMenuW(m, MF_SEPARATOR, 0, nullptr);
  AppendMenuW(m, MF_STRING, IDM_QUIT, L"Выход");

  POINT p;
  GetCursorPos(&p);
  SetForegroundWindow(g_hwnd); // чтобы меню закрылось по клику мимо
  TrackPopupMenu(m, TPM_RIGHTBUTTON, p.x, p.y, 0, g_hwnd, nullptr);
  DestroyMenu(m);
  PostMessageW(g_hwnd, WM_NULL, 0, 0);
}

// ---- окно настроек ----------------------------------------------------------

namespace setwin {

constexpr int IDC_LLAMA = 101, IDC_DIR = 102, IDC_PORT1 = 103, IDC_PORT2 = 104,
              IDC_IDLE = 105, IDC_AUTOSTART = 106, IDC_SAVE = 107, IDC_CANCEL = 108,
              IDC_BROWSE_DIR = 109, IDC_BROWSE_EXE = 110, IDC_AUTO = 111,
              IDC_MODELSMAX = 114, IDC_GPUPIN = 115;

HWND ctrl(int id) { return GetDlgItem(g_set_hwnd, id); }

// Дизайн-координаты контролов (при создании) и дизайн-размер клиентской
// области: при изменении размеров окна всё масштабируется пропорционально.
std::vector<std::pair<HWND, RECT>> g_design;
int g_dw = 0, g_dh = 0;

void remember(HWND h, int x, int y, int w, int ht) {
  g_design.push_back({h, {x, y, x + w, y + ht}});
}

void relayout(HWND hwnd) {
  if (!g_dw || !g_dh) return;
  RECT rc{};
  GetClientRect(hwnd, &rc);
  double fx = (double)rc.right / g_dw, fy = (double)rc.bottom / g_dh;
  for (auto& [h, r] : g_design)
    MoveWindow(h, (int)(r.left * fx), (int)(r.top * fy),
               (int)((r.right - r.left) * fx), (int)((r.bottom - r.top) * fy), TRUE);
}

void add_edit(int id, int x, int y, int w) {
  HWND h = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", nullptr,
                           WS_CHILD | WS_VISIBLE | WS_TABSTOP | ES_AUTOHSCROLL,
                           x, y, w, 24, g_set_hwnd, (HMENU)(INT_PTR)id, nullptr, nullptr);
  SendMessageW(h, WM_SETFONT, (WPARAM)lh::ui_font(), TRUE);
  remember(h, x, y, w, 24);
}

void add_static(const wchar_t* text, int x, int y, int w) {
  HWND h = CreateWindowExW(0, L"STATIC", text, WS_CHILD | WS_VISIBLE,
                           x, y, w, 18, g_set_hwnd, nullptr, nullptr, nullptr);
  SendMessageW(h, WM_SETFONT, (WPARAM)lh::ui_font(), TRUE);
  remember(h, x, y, w, 18);
}

void add_button(const wchar_t* text, int id, int x, int y, int w, bool def = false) {
  HWND h = lh::ws::button(g_set_hwnd, id, x, y, w, 26, text, nullptr, def);
  remember(h, x, y, w, 26);
}

void add_check(const wchar_t* text, int id, int x, int y, int w = 300) {
  HWND h = CreateWindowExW(0, L"BUTTON", text,
                           WS_CHILD | WS_VISIBLE | WS_TABSTOP | BS_AUTOCHECKBOX,
                           x, y, w, 20, g_set_hwnd, (HMENU)(INT_PTR)id, nullptr, nullptr);
  SendMessageW(h, WM_SETFONT, (WPARAM)lh::ui_font(), TRUE);
  remember(h, x, y, w, 20);
}

std::wstring edit_text(int id) {
  wchar_t buf[1024] = {};
  GetWindowTextW(ctrl(id), buf, 1024);
  return buf;
}

void set_edit(int id, const std::wstring& text) { SetWindowTextW(ctrl(id), text.c_str()); }

std::wstring browse_folder(HWND parent) {
  wchar_t buf[MAX_PATH] = {};
  BROWSEINFOW bi{};
  bi.hwndOwner = parent;
  bi.lpszTitle = L"Папка с моделями (.gguf)";
  bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;
  PIDLIST_ABSOLUTE pidl = SHBrowseForFolderW(&bi);
  if (!pidl) return {};
  SHGetPathFromIDListW(pidl, buf);
  CoTaskMemFree(pidl);
  return buf;
}

std::wstring browse_file(HWND parent) {
  wchar_t buf[1024] = {};
  OPENFILENAMEW of{};
  of.lStructSize = sizeof(of);
  of.hwndOwner = parent;
  of.lpstrFilter = L"llama-server.exe\0llama-server.exe\0Все файлы\0*.*\0";
  of.lpstrFile = buf;
  of.nMaxFile = 1024;
  of.Flags = OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;
  of.lpstrTitle = L"Где лежит llama-server.exe";
  if (!GetOpenFileNameW(&of)) return {};
  return buf;
}

void load_fields() {
  config::Config cfg = config::current();
  set_edit(IDC_LLAMA, cfg.llama_server);
  set_edit(IDC_DIR, models_dir_resolved());
  set_edit(IDC_PORT1, std::to_wstring(cfg.openai_port));
  set_edit(IDC_PORT2, std::to_wstring(cfg.control_port));
  set_edit(IDC_IDLE, std::to_wstring(cfg.unload_after_min));
  set_edit(IDC_MODELSMAX, std::to_wstring(cfg.models_max));
  SendMessageW(ctrl(IDC_AUTOSTART), BM_SETCHECK,
               autostart_on() ? BST_CHECKED : BST_UNCHECKED, 0);
  SendMessageW(ctrl(IDC_AUTO), BM_SETCHECK,
               config::current().auto_params ? BST_CHECKED : BST_UNCHECKED, 0);
  SendMessageW(ctrl(IDC_GPUPIN), BM_SETCHECK,
               config::current().gpu_pin ? BST_CHECKED : BST_UNCHECKED, 0);
}

bool save_fields() {
  config::Config cfg = config::current();
  config::Config old = cfg;

  cfg.llama_server = edit_text(IDC_LLAMA);
  cfg.models_dir = edit_text(IDC_DIR);
  try {
    cfg.openai_port = std::stoi(edit_text(IDC_PORT1));
    cfg.control_port = std::stoi(edit_text(IDC_PORT2));
    cfg.unload_after_min = std::stoi(edit_text(IDC_IDLE));
    cfg.models_max = std::stoi(edit_text(IDC_MODELSMAX));
  } catch (...) {
    MessageBoxW(g_set_hwnd, L"Порты, минуты и число моделей — это числа.", L"llmhost",
                MB_ICONWARNING);
    return false;
  }
  if (cfg.models_max < 0 || cfg.models_max > 16) cfg.models_max = 2;
  bool want_autostart =
      SendMessageW(ctrl(IDC_AUTOSTART), BM_GETCHECK, 0, 0) == BST_CHECKED;
  cfg.auto_params = SendMessageW(ctrl(IDC_AUTO), BM_GETCHECK, 0, 0) == BST_CHECKED;
  cfg.gpu_pin = SendMessageW(ctrl(IDC_GPUPIN), BM_GETCHECK, 0, 0) == BST_CHECKED;

  config::save(cfg);
  config::set_current(cfg);
  apply_autostart(want_autostart);

  if (old.openai_port != cfg.openai_port || old.control_port != cfg.control_port)
    MessageBoxW(g_set_hwnd,
                L"Смена портов вступит в силу после перезапуска llmhost.",
                L"llmhost", MB_ICONINFORMATION);
  // Настройки планирования читаются каждой загрузкой слота — применяются без
  // дёрганья детей. Вотчер реестра догонит смену models-dir сам.
  (void)old;
  return true;
}

LRESULT CALLBACK wndproc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
  g_set_hwnd = hwnd;
  switch (msg) {
    case WM_CREATE: {
      int x = 16, w = 448;
      add_static(L"llama-server.exe (пусто — установленная сборка, рядом с программой, PATH):",
                 x, 12, w);
      add_edit(IDC_LLAMA, x, 32, w - 36);
      add_button(L"…", IDC_BROWSE_EXE, x + w - 32, 31, 32);

      add_static(L"Папка с моделями (.gguf):", x, 68, w);
      add_edit(IDC_DIR, x, 88, w - 36);
      add_button(L"…", IDC_BROWSE_DIR, x + w - 32, 87, 32);

      add_static(L"Порт OpenAI API:", x, 124, 130);
      add_edit(IDC_PORT1, x, 144, 130);
      add_static(L"Порт control API:", x + 160, 124, 130);
      add_edit(IDC_PORT2, x + 160, 144, 130);
      add_static(L"Выгружать после простоя (мин, 0 — никогда):", x, 180, 300);
      add_edit(IDC_IDLE, x, 200, 130);
      add_static(L"Моделей в памяти (роутер, 0 — без лимита):", x + 150, 180, 290);
      add_edit(IDC_MODELSMAX, x + 150, 200, 130);
      add_check(L"Запускать вместе с Windows (запись в реестре этой машины)", IDC_AUTOSTART,
                x, 232, 430);
      add_check(L"Умная настройка: контекст «из модели», KV q8_0, MoE на CPU (рекомендуется)",
                IDC_AUTO, x, 256);
      add_check(L"Раскладывать модели по картам по очереди (мульти-GPU)",
                IDC_GPUPIN, x, 280, 430);
      add_button(L"Сохранить", IDC_SAVE, x, 306, 110, /*def=*/true);
      add_button(L"Отмена", IDC_CANCEL, x + 122, 306, 110);
      return 0;
    }
    case WM_SIZE:
      relayout(hwnd);
      ws::round_window(hwnd);
      return 0;
    case WM_GETMINMAXINFO:
      ws::min_size(lp, 480, 448); // меньше дизайна контролы слипнутся
      return 0;
    case WM_DRAWITEM:
      ws::draw_button((const DRAWITEMSTRUCT*)lp);
      return TRUE;
    case WM_COMMAND: {
      int id = LOWORD(wp);
      if (id == IDC_BROWSE_DIR) {
        std::wstring d = browse_folder(hwnd);
        if (!d.empty()) set_edit(IDC_DIR, d);
      } else if (id == IDC_BROWSE_EXE) {
        std::wstring f = browse_file(hwnd);
        if (!f.empty()) set_edit(IDC_LLAMA, f);
      } else if (id == IDC_SAVE) {
        if (save_fields()) ShowWindow(hwnd, SW_HIDE);
      } else if (id == IDC_CANCEL) {
        ShowWindow(hwnd, SW_HIDE);
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

void ensure_window() {
  if (g_set_hwnd && IsWindow(g_set_hwnd)) {
    load_fields();
    ShowWindow(g_set_hwnd, SW_SHOWNORMAL);
    SetForegroundWindow(g_set_hwnd);
    return;
  }
  WNDCLASSEXW wc{};
  wc.cbSize = sizeof(wc);
  wc.lpfnWndProc = wndproc;
  wc.hInstance = GetModuleHandleW(nullptr);
  wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
  wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
  wc.hIcon = wc.hIconSm = app_icon();
  wc.lpszClassName = SET_CLASS;
  RegisterClassExW(&wc);
  // Перетаскиваемые границы — окно масштабируется, контролы следуют.
  DWORD style = WS_OVERLAPPEDWINDOW;
  CreateWindowExW(0, SET_CLASS, L"llmhost — настройки", style,
                  CW_USEDEFAULT, CW_USEDEFAULT, 480, 448, nullptr, nullptr, wc.hInstance,
                  nullptr);
  RECT crc{};
  GetClientRect(g_set_hwnd, &crc);
  g_dw = crc.right; // дизайн-размер: относительно него тянется relayout
  g_dh = crc.bottom;
  ws::round_window(g_set_hwnd);
  load_fields();
  ShowWindow(g_set_hwnd, SW_SHOWNORMAL);
}

} // namespace setwin

// ---- окно трея --------------------------------------------------------------

LRESULT CALLBACK tray_wndproc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
  switch (msg) {
    case WM_APP_TRAY:
      if (lp == WM_RBUTTONUP || lp == WM_LBUTTONUP || lp == WM_LBUTTONDBLCLK)
        show_menu();
      return 0;
    case WM_APP_STATE:
      update_tooltip();
      return 0;
    case WM_TIMER:
      // Скачивания и простой меняются сами по себе — тултип живёт своей жизнью.
      if (wp == 1) update_tooltip();
      return 0;
    case WM_APP_QUIT:
      sup::shutdown();
      Shell_NotifyIconW(NIM_DELETE, &g_nid);
      PostQuitMessage(0);
      return 0;
    case WM_COMMAND: {
      int id = LOWORD(wp);
      if (id >= IDM_MODEL_BASE && id < IDM_MODEL_BASE + 40) {
        int i = id - IDM_MODEL_BASE;
        // Из снимка, снятого при постройке меню, — не из свежего скана папки.
        if (i >= 0 && i < (int)g_menu_models.size()) {
          auto& pick = g_menu_models[i];
          if (pick.second == "не gguf") {
            MessageBoxW(hwnd, utf8_to_wide("Файл не разбирается как GGUF.").c_str(),
                        L"llmhost", MB_ICONWARNING);
          } else {
            load_async(pick.first);
          }
        }
      } else if (id == IDM_UNLOAD) {
        sup::stop(true);
      } else if (id >= IDM_DEL_BASE && id < IDM_DEL_BASE + 40) {
        int i = id - IDM_DEL_BASE;
        if (i >= 0 && i < (int)g_del_models.size()) {
          std::wstring path = g_del_models[i];
          size_t slash = path.find_last_of(L"\\/");
          std::wstring file = slash == std::wstring::npos ? path : path.substr(slash + 1);
          if (MessageBoxW(hwnd,
                          (L"Удалить модель «" + file +
                           L"»?\n\nФайл уйдёт в корзину (восстановимо из проводника); "
                           L"модель исчезнет из меню и пресета роутера.")
                              .c_str(),
                          L"llmhost — удаление модели", MB_OKCANCEL | MB_ICONWARNING) == IDOK) {
            std::thread([p = path] {
              std::wstring err = delete_model(p);
              if (err.empty()) {
                logf("tray: модель удалена (в корзину): %s", wide_to_utf8(p).c_str());
                tray::balloon(L"Модель удалена (в корзину) — списки обновятся через пару секунд.");
              } else {
                logf("tray: удаление не удалось: %s", wide_to_utf8(err).c_str());
                tray::balloon(L"Удаление не удалось: " + err);
              }
            }).detach();
          }
        }
      } else if (id >= IDM_PART_BASE && id < IDM_PART_BASE + 10) {
        // Докачать: pull::start продолжает с места останова (curl -C -),
        // SHA256 после докачки гарантирует целостность.
        int i = id - IDM_PART_BASE;
        if (i < 0 || i >= (int)g_parts.size()) return 0;
        PartEntry p = g_parts[i];
        std::thread([p] {
          std::string e;
          if (!pull::start(lh::wide_to_utf8(p.repo), lh::wide_to_utf8(p.file), e)) {
            logf("tray: докачка не началась: %s", e.c_str());
            tray::balloon(L"Докачка не началась: " + utf8_to_wide(e));
          } else {
            tray::balloon(L"Докачка «" + p.file + L"» продолжена — прогресс в подсказке значка.");
          }
        }).detach();
      } else if (id >= IDM_PARTDEL_BASE && id < IDM_PARTDEL_BASE + 10) {
        // Убрать .part-хвост: файл частичный, заново качать с нуля, поэтому
        // без подтверждения — цена ошибки копеечная.
        int i = id - IDM_PARTDEL_BASE;
        if (i < 0 || i >= (int)g_parts.size()) return 0;
        std::wstring path = g_parts[i].path;
        std::thread([path] {
          if (DeleteFileW(path.c_str())) {
            logf("tray: недокачанная загрузка удалена: %s", lh::wide_to_utf8(path).c_str());
            tray::balloon(L"Недокачанная загрузка удалена.");
          } else {
            logf("tray: .part не удаляется (ошибка %lu)", GetLastError());
            tray::balloon(L"Файл занят — докачка ещё идёт? Попробуйте позже.");
          }
        }).detach();
      } else if (id >= IDM_ENGINE_BASE && id < IDM_ENGINE_BASE + 8) {
        // Выбор сборки движка: не установленная — скачать и включить,
        // установленная — включить. Роутер перезапустится плавно.
        int i = id - IDM_ENGINE_BASE;
        auto builds = engine::list();
        if (i < 0 || i >= (int)builds.size() || engine::busy()) return 0;
        std::string build = builds[i].id;
        bool have = builds[i].installed;
        std::thread([build, have] {
          std::string e;
          if (!have) {
            tray::balloon(L"Скачиваю движок «" + utf8_to_wide(build) +
                          L"» — прогресс в подсказке значка.");
            if (!engine::install(build, e)) {
              tray::balloon(L"Движок не установился: " + utf8_to_wide(e));
              return;
            }
          }
          config::Config cfg = config::current();
          cfg.engine_build = build;
          config::set_current(cfg);
          config::save(cfg);
          // Дети поднимутся с новым exe при следующей загрузке.
          tray::balloon(L"Движок: " + utf8_to_wide(build) +
                        (have ? L"" : L" (установлен и выбран)"));
        }).detach();
      } else if (id == IDM_BENCH) {
        std::string e;
        if (!bench::start(L"", e))
          MessageBoxW(hwnd, utf8_to_wide(e).c_str(), L"llmhost", MB_ICONWARNING);
      } else if (id == IDM_CATALOG) {
        catalogui::open();
      } else if (id == IDM_CHAT) {
        // Чат без языковой модели бессмысленен: вместо пустого окна —
        // предупреждение с подсказкой, где её взять.
        bool have_llm = false;
        for (auto& m : modelreg::scan(models_dir_resolved()))
          if (m.meta.n_layers && !sup::embedding_arch(m.arch)) { have_llm = true; break; }
        if (!have_llm) {
          MessageBoxW(hwnd,
                      utf8_to_wide("В папке моделей нет ни одной языковой модели (GGUF) — "
                                   "чат пока не с кем вести.\n\n"
                                   "Скачайте модель в каталоге («Каталог моделей» в меню) "
                                   "или скопируйте .gguf-файл в папку моделей — она появится "
                                   "в списке через пару секунд.").c_str(),
                      L"llmhost — чат", MB_ICONINFORMATION);
        } else {
          chatui::open();
        }
      } else if (id == IDM_OPEN_DIR) {
        std::wstring dir = models_dir_resolved();
        lh::ensure_dir(dir);
        ShellExecuteW(hwnd, L"explore", dir.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
      } else if (id == IDM_WIZARD) {
        wizard::show_async();
      } else if (id == IDM_CLI) {
        // Консоль с готовым окружением: cwd в папке программы, lhctl в PATH
        // этой сессии, автодополнение включено, статус на экране.
        std::wstring dir = lh::exe_dir();
        if (!lh::file_exists(dir + L"\\lhctl.exe")) {
          MessageBoxW(hwnd, L"lhctl.exe не найден рядом с программой — соберите проект.",
                      L"llmhost", MB_ICONWARNING);
          return 0;
        }
        auto q = [](std::wstring s) { // кавычки для PS-одинарных строк
          size_t p = 0;
          while ((p = s.find(L'\'', p)) != std::wstring::npos)
            s.replace(p, 1, L"''"), p += 2;
          return s;
        };
        std::wstring args =
            L"-NoExit -ExecutionPolicy Bypass -Command \""
            L"$Host.UI.RawUI.WindowTitle = 'llmhost CLI'; "
            L"$env:PATH = '" + q(dir) + L";' + $env:PATH; "
            L"Set-Location -LiteralPath '" + q(dir) + L"'; "
            L"lhctl completion powershell | Out-String | Invoke-Expression; "
            L"Clear-Host; lhctl status\"";
        ShellExecuteW(hwnd, L"open", L"powershell.exe", args.c_str(), dir.c_str(),
                      SW_SHOWNORMAL);
      } else if (id == IDM_SETTINGS) {
        setwin::ensure_window();
      } else if (id == IDM_QUIT) {
        SendMessageW(hwnd, WM_APP_QUIT, 0, 0);
      }
      return 0;
    }
    case WM_DESTROY:
      Shell_NotifyIconW(NIM_DELETE, &g_nid);
      PostQuitMessage(0);
      return 0;
    default:
      return DefWindowProcW(hwnd, msg, wp, lp);
  }
}

} // namespace

bool init(HINSTANCE hInst) {
  WNDCLASSEXW wc{};
  wc.cbSize = sizeof(wc);
  wc.lpfnWndProc = tray_wndproc;
  wc.hInstance = hInst;
  wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
  wc.lpszClassName = TRAY_CLASS;
  if (!RegisterClassExW(&wc)) return false;

  g_hwnd = CreateWindowExW(0, TRAY_CLASS, L"llmhost", 0, 0, 0, 0, 0, HWND_MESSAGE,
                           nullptr, hInst, nullptr);
  if (!g_hwnd) return false;

  g_nid.cbSize = sizeof(g_nid);
  g_nid.hWnd = g_hwnd;
  g_nid.uID = 1;
  g_nid.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP; // без NIF_ICON слот пустой
  g_nid.uCallbackMessage = WM_APP_TRAY;
  g_nid.hIcon = app_icon();
  wcsncpy_s(g_nid.szTip, L"llmhost", _TRUNCATE);
  if (!Shell_NotifyIconW(NIM_ADD, &g_nid)) return false;

  update_tooltip();
  // Раз в секунду: проценты скачиваний и прочие плавающие статусы.
  SetTimer(g_hwnd, 1, 1000, nullptr);
  return true;
}

void run() {
  MSG msg;
  while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
    if ((g_set_hwnd && IsDialogMessageW(g_set_hwnd, &msg)) || catalogui::filter(msg) ||
        chatui::filter(msg))
      continue;
    TranslateMessage(&msg);
    DispatchMessageW(&msg);
  }
}

void notify_state() {
  if (g_hwnd) PostMessageW(g_hwnd, WM_APP_STATE, 0, 0);
}

void balloon(const std::wstring& text) {
  if (!g_hwnd) return;
  // Своя структура, не копия g_nid: balloon зовут рабочие потоки, а тултип
  // в g_nid пишет UI-поток — гонка на общей структуре рвала текст.
  NOTIFYICONDATAW nid{};
  nid.cbSize = sizeof(nid);
  nid.hWnd = g_hwnd;
  nid.uID = 1;
  nid.uFlags = NIF_INFO;
  wcsncpy_s(nid.szInfo, text.c_str(), _TRUNCATE);
  wcsncpy_s(nid.szInfoTitle, L"llmhost", _TRUNCATE);
  nid.dwInfoFlags = NIIF_INFO;
  Shell_NotifyIconW(NIM_MODIFY, &nid);
}

void quit() {
  if (g_hwnd) PostMessageW(g_hwnd, WM_APP_QUIT, 0, 0);
}

// Публично для control API (DELETE /v0/models): выгрузка + группа шардов +
// корзина — та же семантика, что у пункта меню трея.
std::wstring delete_model_to_bin(const std::wstring& path) { return delete_model(path); }

} // namespace lh::tray
