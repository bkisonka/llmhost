// llmhostd — headless-демон: держит llama-server и даёт системе
#include "i18n.h"
// OpenAI-совместимый endpoint (:openai_port) и control API (:control_port).
// Стиль Ollama на сервере: никакого GUI, лог — в файл (LH_LOG_STDERR=1 — эхо
// в stderr для foreground-запуска). Остановка: SIGTERM/SIGINT или /v0/quit.
#include "config.h"
#include "control.h"
#include "bench.h"
#include "engine.h"
#include "proc.h"
#include "proxy.h"
#include "pull.h"
#include "quit.h"
#include "supervisor.h"
#include "util.h"
#include "tokens.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <signal.h>
#include <sys/file.h>
#include <unistd.h>
#endif

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <thread>

using lh::i18n::tr;

namespace {

// Сигналы и /v0/quit сходятся в один атомарный флаг; main опрашивает его.
void on_signal(int) {
  lh::quit::request();
}

#ifdef _WIN32
BOOL WINAPI on_ctrl(DWORD) {
  lh::quit::request();
  return TRUE; // доходимы в главном цикле: shutdown() сам почистит детей
}

// Один экземпляр на систему: LockFileEx на pid-файле (аналог flock).
// Хендлы Windows не наследуются без явного запроса — наш CreateProcess
// передаёт детям только stdout/stderr, так что posix-ловушки «сирота
// держит унаследованный fd замка» здесь нет по построению.
bool acquire_instance_lock() {
  std::string path = lh::app_dir() + "/llmhostd.pid";
  HANDLE h = CreateFileW(lh::utf8_to_wide(path).c_str(),
                         GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                         OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE) {
    fprintf(stderr, "llmhostd: не открыть pid-файл %s: %lu\n", path.c_str(),
            GetLastError());
    return false;
  }
  OVERLAPPED ov{};
  if (!LockFileEx(h, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0,
                  &ov)) {
    fprintf(stderr, "llmhostd: уже запущен (pid-файл %s занят).\n", path.c_str());
    CloseHandle(h);
    return false;
  }
  SetFilePointer(h, 0, nullptr, FILE_BEGIN);
  SetEndOfFile(h);
  char buf[32];
  int n = snprintf(buf, sizeof(buf), "%lu\n", GetCurrentProcessId());
  DWORD w = 0;
  WriteFile(h, buf, (DWORD)n, &w, nullptr);
  // хендл намеренно не закрывается: держит замок до конца процесса
  static HANDLE keep = h;
  (void)keep;
  return true;
}
#else
// Один экземпляр на систему: flock на pid-файле.
// Файл держим открытым всё время работы — замок живёт вместе с процессом.
bool acquire_instance_lock() {
  std::string path = lh::app_dir() + "/llmhostd.pid";
  // O_CLOEXEC обязателен: замок живёт до конца процесса, и унаследованный
  // ребёнком fd пережил бы смерть демона — сирота-llama-server держала бы
  // one-instance замок, и новый демон (тот, что должен её подмести) не
  // смог бы стартовать (поймано живьём при проверке свипа).
  int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
  if (fd < 0) {
    fprintf(stderr, "llmhostd: не открыть pid-файл %s: %s\n", path.c_str(), strerror(errno));
    return false;
  }
  if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
    if (errno == EWOULDBLOCK) {
      fprintf(stderr, "llmhostd: уже запущен (pid-файл %s занят).\n", path.c_str());
      return false;
    }
    fprintf(stderr, "llmhostd: flock %s: %s\n", path.c_str(), strerror(errno));
    return false;
  }
  if (::ftruncate(fd, 0) == 0) {
    char buf[32];
    int n = snprintf(buf, sizeof(buf), "%d\n", (int)getpid());
    ssize_t ignored = ::write(fd, buf, (size_t)n);
    (void)ignored;
  }
  return true; // fd намеренно не закрывается: держит замок до конца процесса
}
#endif

} // namespace

int main() {
  // Как можно раньше: фатальный сигнал обязан оставить отпечаток в журнале —
  // без него «демон молча умер» в поле неотличим от «не запустился» (реальный
  // случай: SIGFPE сразу после «control: слушаю», журнал обрывался без следа).
  lh::install_crash_telemetry();
  if (!acquire_instance_lock()) return 0;

  // Сигналы — через атомарный флаг, не через самозапись в g_quit из потока:
  // обработчик минимальный, вся логика — в главном цикле.
#ifdef _WIN32
  signal(SIGTERM, on_signal);
  signal(SIGINT, on_signal); // и Ctrl+C; закрытие консоли — через on_ctrl выше
  SetConsoleCtrlHandler(on_ctrl, TRUE);
#else
  struct sigaction sa{};
  sa.sa_handler = on_signal;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGTERM, &sa, nullptr);
  sigaction(SIGINT, &sa, nullptr);
  signal(SIGPIPE, SIG_IGN); // отвалившийся клиент httplib не должен ронять демон
#endif

  // CUDA по умолчанию сортирует карты по скорости (FASTEST_FIRST) — индексы
  // CUDA не совпадают с nvidia-smi и нашими настройками (проверено пробой:
  // 3060+Blackwell → Blackwell = cuda:0). PCI_BUS_ID делает индексацию
  // единой; Vulkan и машины с одной картой от переменной не зависят.
  // Дети роутера наследуют окружение — фикс действует на всю иерархию.
#ifdef _WIN32
  _putenv("CUDA_DEVICE_ORDER=PCI_BUS_ID");
#else
  ::setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 1);
#endif

  lh::config::migrate_legacy();
  lh::config::Config cfg = lh::config::load();
  // Внешняя служба (привязка мимо localhost) — Bearer-ключ не отключается:
  // без него порт читала бы вся сеть. Чиним громко, а не отказываемся
  // стартовать: headless-сервер, который не поднялся, хуже требуемого ключа.
  if (!lh::config::is_local_bind(cfg.openai_bind) && !cfg.openai_auth) {
    lh::logf("proxy: привязка %s — открытый доступ запрещён, openai_auth включён "
             "принудительно",
             cfg.openai_bind.c_str());
    cfg.openai_auth = true;
  }
  lh::config::set_current(cfg);

  // Сироты жёстко убитого прошлого демона (вне systemd-юнита pdeathsig нет):
  // наши llama-server'ы под движковым каталогом добиваются ДО любых спавнов,
  // иначе автовосстановление модели заспавнится в занятую VRAM.
  lh::tokens::init(lh::app_dir() + "/api_tokens.json");
  if (lh::proc::sweep_orphans(lh::app_dir() + "/engine") > 0)
    lh::sup::note_external_kill();

  lh::logf("llmhostd стартует: openai %s:%d, control 127.0.0.1:%d, модели: %s; конфиг: %s",
           cfg.openai_bind.c_str(), cfg.openai_port, cfg.control_port,
           lh::config::models_dir_resolved(cfg).c_str(),
           lh::config::conf_path().c_str());

  // Установка движка должна начинаться на тихой системе: свободный слот,
  // тихий бенч, заглушенный роутер с детьми. Хук вместо прямых вызовов вверх:
  // engine остаётся файловым менеджером (см. engine.h Quiesce).
  lh::engine::set_quiesce([](std::string& err) {
    lh::sup::St slot = lh::sup::state();
    if (slot == lh::sup::St::Loading || slot == lh::sup::St::Loaded) {
      err = tr("сначала выгрузите модель — установке нужен свободный llama-server");
      return false;
    }
    if (lh::bench::busy()) {
      err = tr("дождитесь конца замера движков");
      return false;
    }
    // Дети-модели держат файлы движка — на время подмены глушим их всех.
    lh::sup::stop_router();
    return true;
  });
  lh::sup::init();
  if (!lh::proxy::start(cfg.openai_port, cfg.openai_bind) ||
      !lh::control::start(cfg.control_port)) {
    // Полуживой демон хуже честного отказа: без OpenAI endpoint'а или без
    // control API обслуживать нечего — завершаемся с внятной ошибкой,
    // systemd поднимет заново (порт, возможно, освободится).
    lh::logf("llmhostd: endpoint не поднялся — отказ старта");
    // Хвост остановки тот же, что на сигнале: joinable-потоки (HTTP-серверы,
    // сторожа) не должны доживать до статических деструкторов — иначе
    // terminate/CoreDump вместо честного exit(1).
    lh::sup::shutdown();
    lh::proxy::stop();
    lh::control::stop();
    lh::sup::reap_background();
    return 1;
  }

  // Пин vs диск: расхождение не должно жить молча (04.10 пересборка затёрла
  // обновлённый пин затравкой — демон неделю отчитывался не тем релизом).
  // Один честный лог на старте; живой факт — GET /v0/engine (installed_release).
  std::thread([] {
    const std::string pin = lh::engine::pinned_release();
    if (pin.empty()) return;
    for (auto& b : lh::engine::list())
      if (b.installed && !b.installed_release.empty() && b.installed_release != pin)
        lh::logf("engine: РАСХОЖДЕНИЕ: пин %s, установлено %s (сборка «%s»)",
                 pin.c_str(), b.installed_release.c_str(), b.id.c_str());
  }).detach();

  // Главный цикл: HTTP-серверы крутятся в своих потоках, здесь только ждём
  // остановку (сигнал или /v0/quit).
  while (!lh::quit::requested())
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

  lh::logf("llmhostd останавливается");
  // install/pull/bench работают в отсоединённых потоках (single-flight busy):
  // дождись их завершения с потолком, иначе они допишут свои g_state/конфиг/
  // детей уже после хвоста остановки — тот же класс выхода процесса, что
  // закрытый SIGABRT-фикс про joinable-потоки. Долгая докачка после 10 с
  // прервётся systemd (KillMode=control-group), это честно.
  for (int waited = 0;
       (lh::engine::busy() || lh::pull::busy() || lh::bench::busy()) && waited < 100;
       ++waited)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  if (lh::engine::busy() || lh::pull::busy() || lh::bench::busy())
    lh::logf("llmhostd: установка/скачивание/бенч не завершились за 10 с — "
             "остановка продолжается, группа cgroup добьёт детей");
  lh::sup::shutdown();
  // HTTP-потоки — после остановки детей: живой запрос ещё держит
  // BusyModel и будит очередь. Дождаться их надо до статических
  // деструкторов, иначе g_wake_lot разрушается под notify (TSan).
  lh::proxy::stop();
  lh::control::stop();
  // Обработчик /v0/load мог породить поток уже после первого reap.
  lh::sup::reap_background();
  return 0;
}
