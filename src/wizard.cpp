#include "wizard.h"

#include "catalog_alias.h"
#include "config.h"
#include "engine.h"
#include "hw.h"
#include "modelreg.h"
#include "pull.h"
#include "tray.h"
#include "util.h"
#include "winstyle.h"

#include <commctrl.h>
#include <windows.h>

#include <thread>

namespace lh::wizard {

namespace {

constexpr wchar_t WIZ_CLASS[] = L"llmhostWizard";
constexpr int IDC_INSTALL = 101, IDC_LATER = 102, IDC_MODEL = 103,
              IDC_STATUS = 104, IDC_BAR = 105, IDC_HINT = 106;

HWND g_hwnd = nullptr;
HWND g_combo = nullptr, g_status = nullptr, g_bar = nullptr, g_hint = nullptr,
     g_engine_line = nullptr, g_gpu_line = nullptr, g_install_btn = nullptr;

// Фаза установки: ничего → движок → модель → готово/ошибка. Worker ставит,
// таймер окна читает публичные статусы (engine/pull сами потокобезопасны).
struct Phase {
  enum V { None, Engine, Model, Done, Failed } v = None;
  std::wstring note;
};
Phase g_phase;
std::wstring g_chosen_alias; // что ставим (для итогового сообщения)

// Дефолтный выбор по видеопамяти: побольше карта — побольше модель.
const alias::Entry& default_alias() {
  hw::Gpu gpu = hw::query();
  uint64_t vram = gpu.vram_total;
  const alias::Entry* pick = nullptr;
  auto want = [](const alias::Entry& e, const char* name) {
    return std::string(e.name) == name;
  };
  for (const alias::Entry& e : alias::kAll) {
    if (vram >= (10ull << 30) && want(e, "qwen3")) pick = &e;
    if (vram >= (5ull << 30) && vram < (10ull << 30) && want(e, "qwen3-4b"))
      pick = &e;
  }
  if (!pick) {
    // Слабая карта или CPU: 3B Ministral (Q4 ~2 ГиБ) — влезет почти везде.
    for (const alias::Entry& e : alias::kAll)
      if (want(e, "ministral")) pick = &e;
  }
  return *pick;
}

void set_status(const std::wstring& s) { SetWindowTextW(g_status, s.c_str()); }

void install_worker(bool engine_needed) {
  auto fail = [](const std::wstring& why) { g_phase = {Phase::Failed, why}; };
  if (engine_needed) {
    g_phase = {Phase::Engine, {}};
    // Сборка по железу: NVIDIA → CUDA (быстрее), остальным — Vulkan.
    std::string e;
    if (!engine::install(engine::default_build(), e)) {
      fail(utf8_to_wide("Движок не установился: " + e));
      return;
    }
  }
  g_phase = {Phase::Model, {}};
  std::string e;
  if (!pull::start(wide_to_utf8(g_chosen_alias), "", e)) {
    fail(utf8_to_wide("Скачивание не началось: " + e));
    return;
  }
  // Завершение увидит таймер по pull::status_json() — прогресс честнее
  // показывать из UI-потока, worker закончился.
}

LRESULT CALLBACK wndproc(HWND h, UINT msg, WPARAM w, LPARAM l) {
  switch (msg) {
    case WM_DRAWITEM: {
      // Плоские кнопки winstyle рисует родитель — без этого текста нет.
      if (auto* d = (DRAWITEMSTRUCT*)l; d && d->CtlType == ODT_BUTTON)
        lh::ws::draw_button(d);
      return TRUE;
    }
    case WM_CREATE: {
      HINSTANCE inst = ((LPCREATESTRUCTW)l)->hInstance;

      auto label = [&](const wchar_t* text, int x, int y, int w, int ht = 36) {
        HWND c = CreateWindowExW(0, L"STATIC", text, WS_CHILD | WS_VISIBLE, x, y, w,
                                 ht, h, nullptr, nullptr, nullptr);
        SendMessageW(c, WM_SETFONT, (WPARAM)lh::ui_font(), TRUE);
        return c;
      };

      label(L"Добро пожаловать! До первой модели два шага — оба скачаются "
            L"сами, под вашу видеокарту.", 20, 16, 520, 34);
      g_gpu_line = label(L"", 20, 54, 520, 18);
      g_engine_line = label(L"", 20, 78, 520, 18);

      label(L"Какую модель поставить?", 20, 108, 520, 18);
      g_combo = CreateWindowExW(0, L"COMBOBOX", nullptr,
                                WS_CHILD | WS_VISIBLE | WS_TABSTOP | CBS_DROPDOWNLIST |
                                    WS_VSCROLL,
                                20, 130, 400, 300, h, (HMENU)(INT_PTR)IDC_MODEL, inst,
                                nullptr);
      SendMessageW(g_combo, WM_SETFONT, (WPARAM)lh::ui_font(), TRUE);
      const alias::Entry& def = default_alias();
      int def_idx = 0;
      for (const alias::Entry& e : alias::kAll) {
        std::wstring line = utf8_to_wide(e.name) + L" — " + utf8_to_wide(e.note);
        int i = (int)SendMessageW(g_combo, CB_ADDSTRING, 0, (LPARAM)line.c_str());
        if (std::string(e.name) == std::string(def.name)) def_idx = i;
      }
      SendMessageW(g_combo, CB_SETCURSEL, def_idx, 0);
      g_hint = label(L"Квант подберётся автоматически (рекомендация каталога "
                     L"под вашу карту); для vision-семейств скачается и mmproj.",
                     20, 156, 520, 30);

      g_status = label(L"", 20, 196, 520, 18);
      g_bar = CreateWindowExW(0, PROGRESS_CLASSW, nullptr,
                              WS_CHILD | WS_VISIBLE | PBS_SMOOTH, 20, 218, 520, 14, h,
                              (HMENU)(INT_PTR)IDC_BAR, inst, nullptr);

      g_install_btn = lh::ws::button(h, IDC_INSTALL, 300, 250, 140, 30,
                                     L"Установить и начать", nullptr, true);
      lh::ws::button(h, IDC_LATER, 448, 250, 92, 30, L"Позже", nullptr, false);

      SetWindowTextW(g_status, L"Нажмите «Установить и начать» — дальше всё само.");
      SetTimer(h, 1, 500, nullptr);
      return 0;
    }
    case WM_TIMER: {
      if (g_phase.v == Phase::Failed) {
        set_status(g_phase.note.empty()
                       ? L"Не получилось — подробности в data\\log.txt."
                       : g_phase.note.c_str());
        SetWindowTextW(g_install_btn, L"Повторить");
        EnableWindow(g_install_btn, TRUE);
        return 0;
      }
      // Прогресс: движок или модель — откуда фаза, оттуда и проценты.
      if (g_phase.v == Phase::Engine) {
        std::string s = engine::status_json();
        size_t d = s.find("\"done\":"), t = s.find("\"total\":");
        if (d != std::string::npos && t != std::string::npos) {
          uint64_t done = strtoull(s.c_str() + d + 7, nullptr, 10);
          uint64_t total = strtoull(s.c_str() + t + 8, nullptr, 10);
          if (total > 0) {
            SendMessageW(g_bar, PBM_SETRANGE32, 0, (LPARAM)total);
            SendMessageW(g_bar, PBM_SETPOS, (WPARAM)done, 0);
            wchar_t buf[96];
            swprintf_s(buf, L"Скачиваю движок (%S): %llu из %llu МБ…",
                       engine::default_build().c_str(),
                       (unsigned long long)(done >> 20),
                       (unsigned long long)(total >> 20));
            set_status(buf);
          }
        }
        if (s.find("\"state\":\"idle\"") != std::string::npos ||
            s.find("\"state\":\"error\"") != std::string::npos) {
          // install() в worker завершил фазу Engine — дальше Model
        }
      } else if (g_phase.v == Phase::Model) {
        std::string s = pull::status_json();
        size_t d = s.find("\"done\":"), t = s.find("\"total\":"), st = s.find("\"state\":\"");
        if (d != std::string::npos && t != std::string::npos) {
          uint64_t done = strtoull(s.c_str() + d + 7, nullptr, 10);
          uint64_t total = strtoull(s.c_str() + t + 8, nullptr, 10);
          if (total > 0) {
            SendMessageW(g_bar, PBM_SETRANGE32, 0, (LPARAM)total);
            SendMessageW(g_bar, PBM_SETPOS, (WPARAM)done, 0);
          }
          wchar_t buf[128];
          size_t fs = s.find("\"file\":\"");
          std::string file;
          if (fs != std::string::npos) {
            size_t fe = s.find('"', fs + 9);
            file = s.substr(fs + 9, fe == std::string::npos ? 0 : fe - fs - 9);
          }
          swprintf_s(buf, L"Скачиваю модель: %S (%llu%%)", file.substr(0, 48).c_str(),
                     total ? (unsigned long long)(done * 100 / total) : 0ull);
          set_status(buf);
        }
        if (st != std::string::npos) {
          std::string state = s.substr(st + 9, s.find('"', st + 9) - (st + 9));
          if (state == "done") {
            g_phase = {Phase::Done, {}};
            SendMessageW(g_bar, PBM_SETRANGE32, 0, 100);
            SendMessageW(g_bar, PBM_SETPOS, 100, 0);
            set_status(L"Готово! Модель появится в меню «Модель ▸» через пару "
                       L"секунд; чат — «Чат…» или lhctl chat.");
            SetWindowTextW(g_install_btn, L"Готово");
            EnableWindow(g_install_btn, TRUE);
            tray::balloon(L"Первая модель установлена — меню «Модель ▸», чтобы "
                          L"загрузить, или сразу «Чат…».");
          } else if (state == "error") {
            g_phase = {Phase::Failed, {}};
            set_status(L"Не получилось — подробности в data\\log.txt. Повторить "
                       L"можно кнопкой.");
            SetWindowTextW(g_install_btn, L"Повторить");
            EnableWindow(g_install_btn, TRUE);
          }
        }
      }
      return 0;
    }
    case WM_COMMAND: {
      int id = LOWORD(w);
      if (id == IDC_LATER && g_phase.v != Phase::Engine && g_phase.v != Phase::Model) {
        config::Config cfg = config::current();
        cfg.wizard_dismissed = true;
        config::set_current(cfg);
        config::save(cfg);
        DestroyWindow(h);
      } else if (id == IDC_INSTALL) {
        if (g_phase.v == Phase::Done) {
          DestroyWindow(h);
          return 0;
        }
        int sel = (int)SendMessageW(g_combo, CB_GETCURSEL, 0, 0);
        if (sel >= 0 && sel < (int)alias::kCount) g_chosen_alias = utf8_to_wide(alias::kAll[sel].name);
        EnableWindow(g_install_btn, FALSE);
        bool engine_needed = engine::find_server(config::current()).empty();
        std::thread([engine_needed] { install_worker(engine_needed); }).detach();
      } else if (id == IDC_MODEL && HIWORD(w) == CBN_SELENDOK) {
        int sel = (int)SendMessageW(g_combo, CB_GETCURSEL, 0, 0);
        if (sel >= 0 && sel < (int)alias::kCount)
          SetWindowTextW(g_hint, (L"Репозиторий: " +
                                  utf8_to_wide(alias::kAll[sel].repo) +
                                  L" — квант подберётся под вашу карту")
                                     .c_str());
      }
      return 0;
    }
    case WM_CLOSE:
      if (g_phase.v == Phase::Engine || g_phase.v == Phase::Model) return 0; // заняты
      DestroyWindow(h);
      return 0;
    case WM_DESTROY:
      KillTimer(h, 1);
      g_hwnd = nullptr;
      PostQuitMessage(0);
      return 0;
  }
  return DefWindowProcW(h, msg, w, l);
}

void fill_environment() {
  hw::Gpu gpu = hw::query();
  wchar_t buf[160];
  if (gpu.vram_total > 0)
    swprintf_s(buf, L"Видеокарта: %s (%.1f ГБ)%s", gpu.name.c_str(),
               gpu.vram_total / 1073741824.0,
               gpu.vulkan ? L"" : L" — Vulkan не найден");
  else
    swprintf_s(buf, L"Видеокарта не найдена — модель будет работать на CPU (медленно)");
  SetWindowTextW(g_gpu_line, buf);

  std::wstring exe = engine::find_server(config::current());
  std::string def = engine::default_build();
  uint64_t def_mb = 0;
  for (auto& b : engine::list())
    if (b.id == def) def_mb = b.size >> 20;
  SetWindowTextW(g_engine_line, exe.empty()
      ? (L"Шаг 1. Движок: не установлен — скачается " +
         utf8_to_wide(def) + (def_mb ? L", ~" + std::to_wstring(def_mb) + L" МБ" : L"") +
         L" (под вашу карту).").c_str()
      : L"Шаг 1. Движок: уже установлен.");
}

void run() {
  WNDCLASSEXW wc{};
  wc.cbSize = sizeof(wc);
  wc.lpfnWndProc = wndproc;
  wc.hInstance = GetModuleHandleW(nullptr);
  wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
  wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
  wc.lpszClassName = WIZ_CLASS;
  RegisterClassExW(&wc);
  DWORD style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
  HWND h = CreateWindowExW(0, WIZ_CLASS, L"llmhost — первый запуск", style,
                           CW_USEDEFAULT, CW_USEDEFAULT, 576, 330, nullptr, nullptr,
                           nullptr, nullptr);
  g_hwnd = h;
  ws::round_window(h);
  fill_environment();
  ShowWindow(h, SW_SHOWNORMAL);
  SetForegroundWindow(h);
  MSG msg;
  while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
    if (!IsDialogMessageW(h, &msg)) {
      TranslateMessage(&msg);
      DispatchMessageW(&msg);
    }
  }
}

} // namespace

bool needed() {
  config::Config cfg = config::current();
  if (cfg.wizard_dismissed) return false;
  if (engine::find_server(cfg).empty()) return true;
  return modelreg::scan(config::models_dir_resolved(cfg)).empty();
}

void show_async() {
  std::thread([] {
    if (g_hwnd && IsWindow(g_hwnd)) {
      ShowWindow(g_hwnd, SW_SHOWNORMAL);
      SetForegroundWindow(g_hwnd);
      return;
    }
    run();
  }).detach();
}

} // namespace lh::wizard
