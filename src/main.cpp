// llmhost — трей-приложение: держит llama-server и даёт системе
// OpenAI-совместимый endpoint (:openai_port) и control API (:control_port).
#include "config.h"
#include "control.h"
#include "bench.h"
#include "engine.h"
#include "proxy.h"
#include "supervisor.h"
#include "tray.h"
#include "util.h"
#include "wizard.h"

#include <winsock2.h>
#include <windows.h>
#include <commctrl.h>

#include <thread>

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE, LPWSTR, int) {
  // Visual styles (comctl32 v6 из манифеста) — тема контролов Windows 10.
  // ICC_WIN95_CLASSES — заодно класс тултипов (адреса ссылок в чате).
  INITCOMMONCONTROLSEX icc{sizeof(icc), ICC_STANDARD_CLASSES | ICC_WIN95_CLASSES};
  InitCommonControlsEx(&icc);

  // Один экземпляр на систему.
  HANDLE guard = CreateMutexW(nullptr, TRUE, L"Local\\llmhost.instance");
  if (guard && GetLastError() == ERROR_ALREADY_EXISTS) {
    MessageBoxW(nullptr, L"llmhost уже запущен (значок в трее).", L"llmhost", MB_ICONINFORMATION);
    return 0;
  }

  lh::config::migrate_legacy();
  // CUDA по умолчанию сортирует карты по скорости (FASTEST_FIRST) — индексы
  // CUDA не совпадают с nvidia-smi и нашими настройками (проверено пробой:
  // 3060+Blackwell → Blackwell = cuda:0). PCI_BUS_ID делает индексацию
  // единой; Vulkan и машины с одной картой от переменной не зависят.
  // Дети роутера наследуют окружение — фикс действует на всю иерархию.
  SetEnvironmentVariableW(L"CUDA_DEVICE_ORDER", L"PCI_BUS_ID");
  lh::config::Config cfg = lh::config::load();
  lh::config::set_current(cfg);
  if (cfg.debug_console) AllocConsole();

  lh::logf("llmhost стартует: openai :%d, control :%d, модели: %s",
           cfg.openai_port, cfg.control_port,
           lh::config::models_dir_resolved(cfg).c_str());

  // Установка движка должна начинаться на тихой системе: свободный слот,
  // тихий бенч, заглушенный роутер с детьми. Хук вместо прямых вызовов вверх:
  // engine остаётся файловым менеджером (см. engine.h Quiesce).
  lh::engine::set_quiesce([](std::string& err) {
    lh::sup::St slot = lh::sup::state();
    if (slot == lh::sup::St::Loading || slot == lh::sup::St::Loaded) {
      err = "сначала выгрузите модель — установке нужен свободный llama-server";
      return false;
    }
    if (lh::bench::busy()) {
      err = "дождитесь конца замера движков";
      return false;
    }
    // Дети-модели держат файлы движка — на время подмены глушим их всех.
    lh::sup::stop_router();
    return true;
  });
  lh::sup::init();
  lh::proxy::start(cfg.openai_port, cfg.openai_bind);
  lh::control::start(cfg.control_port);

  // Пин vs диск: расхождение не должно жить молча — один честный лог на
  // старте; живой факт — GET /v0/engine (installed_release, mismatch).
  std::thread([] {
    const std::string pin = lh::engine::pinned_release();
    if (pin.empty()) return;
    for (auto& b : lh::engine::list())
      if (b.installed && !b.installed_release.empty() && b.installed_release != pin)
        lh::logf("engine: РАСХОЖДЕНИЕ: пин %s, установлено %s (сборка «%s»)",
                 pin.c_str(), b.installed_release.c_str(), b.id.c_str());
  }).detach();

  if (!lh::tray::init(hInst)) {
    lh::logf("tray: не создался — выход");
    // Тот же хвост остановки, что на нормальном выходе (перенос SIGABRT-фикса
    // с отказа старта демона на эту ветку): proxy/control и сторожа уже
    // подняты, их joinable-потоки не должны доживать до статических
    // деструкторов — иначе terminate вместо честного return 1.
    lh::sup::shutdown();
    lh::proxy::stop();
    lh::control::stop();
    lh::sup::reap_background();
    WSACleanup();
    if (guard) CloseHandle(guard);
    return 1;
  }
  // Первый запуск (нет движка или моделей) — визард «за два клика», не ждём,
  // пока новичок найдёт README. Отказ («Позже») помнится в конфиге.
  if (lh::wizard::needed()) lh::wizard::show_async();
  lh::tray::run(); // до WM_QUIT (Выход в меню или POST /v0/quit)

  lh::sup::shutdown();
  lh::proxy::stop();
  lh::control::stop();
  lh::sup::reap_background();
  WSACleanup();
  if (guard) CloseHandle(guard);
  return 0;
}
