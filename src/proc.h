// Запуск llama-server и ожидание готовности; общий для супервизора и бенчмарка.
// POSIX: каждый ребёнок — в собственной процессной группе (setpgid); stop_tree()
// убивает всё дерево потомков, жёсткую смерть демона покрывают systemd
// KillMode=control-group и sweep_orphans() на старте. Windows: тот же контракт
// через Job Object kill-on-close (свой джоб на ребёнка — stop_tree, общий —
// «дети не переживают родителя»); sweep_orphans() добирает осиротевшие exe
// по пути из каталога движков. stderr пишется в кольцо — по нему различаем
// «нехватку памяти» (лестница -ngl/-c) и падение по черновику от прочих смертей.
#pragma once
#ifdef _WIN32
typedef unsigned long lh_proc_pid_t; // DWORD: pid() и внутренности Child
#endif
#include <sys/types.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace lh::proc {

#ifdef _WIN32
using pid_t = lh_proc_pid_t;
#endif

class Child {
public:
  Child();
  ~Child();
  Child(const Child&) = delete;
  Child& operator=(const Child&) = delete;

  // stderr-хвост, общий с читающим потоком.
  struct Log {
    std::mutex m;
    std::string tail;
  };

  // args — argv БЕЗ argv[0]: exe станет и путём запуска, и argv[0].
  // Пути с пробелами — просто элемент массива, квотинг не нужен.
  // extra_env — дополнение к унаследованному окружению (существующая
  // переменная ЗАМЕНЯЕТСЯ, не дублируется): image-движку ставится
  // CUDA_VISIBLE_DEVICES — без этого ggml-CUDA создаёт контексты на всех
  // картах и держит по ~200 МиБ на чужих.
  // false — не запустился.
  bool start(const std::string& exe, const std::vector<std::string>& args,
             const std::vector<std::pair<std::string, std::string>>& extra_env = {});

  bool alive();
  // Код выхода (WEXITSTATUS, а по сигналу — 128+sig; Windows — ExitProcess);
  // valid=false — ещё жив.
  bool exited(int& code);
  void stop();      // SIGTERM → до 5 с → SIGKILL → забрать зомби (Windows: Terminate)
  // То же плюс всё ДЕРЕВО потомков: у роутера дети-модели — SIGKILL по
  // процессной группе (POSIX) / TerminateJobObject (Windows), иначе оставались
  // бы сиротами после смерти роутера.
  void stop_tree();

  pid_t pid();
  // Последние ~16 КБ stderr.
  std::string log_tail();
  // argv запуска (без argv[0]) — диагностика спавна (/v0/status в harness).
  std::vector<std::string> args();

private:
  std::mutex m_;
#ifdef _WIN32
  HANDLE proc_ = nullptr; // хендл процесса ребёнка
  HANDLE job_ = nullptr;  // персональный джоб: всё дерево одним TerminateJobObject
  pid_t pid_ = 0;
#else
  pid_t pid_ = -1;
  bool reaped_ = true;   // false — жив или зомби; reap берёт на себя stop()/alive()
  int exit_code_ = 0;
#endif
  std::shared_ptr<Log> log_;
  std::vector<std::string> args_;       // argv без argv[0] — для args()
};

enum class Wait { Healthy, Oom, Exited, Timeout };

// Ждёт GET path (по умолчанию /health) на 127.0.0.1:port; параллельно следит
// за смертью процесса. Oom — процесс умер или не отвечает, а в stderr видна
// нехватка памяти. Путь параметризован: у sd-server нет /health, готовность —
// GET /.
Wait wait_health(Child& c, int port, int timeout_ms,
                 const std::string& path = "/health");

// В stderr есть признаки нехватки памяти — супервизор понижает -ngl/-c.
bool looks_oom(const std::string& log);

// В stderr есть признаки падения ПО ЧЕРНОВИКУ (загрузка draft-файла, создание
// MTP-контекста, размерный GGML_ASSERT головы) — в том числе когда это выглядит
// как OOM: там OOM — cudaMalloc буферов головы, не ствола.
bool looks_draft_failure(const std::string& log);

// Последняя содержательная строка stderr — для понятной ошибки.
std::string last_meaningful_line(const std::string& log);

// Последние n строк лога (без пустых) — для журнала при падении ребёнка:
// содержательная строка часто лишь «loading model», причина глубже.
std::vector<std::string> tail_lines(const std::string& log, int n);

// Подметание сирот после жёсткой смерти демона (POSIX: когда systemd-юнита нет;
// Windows: страховка сверх Job kill-on-close): все процессы, чей exe лежит под
// under_dir (наши установленные движки), убиваются всей группой — у роутера в
// его группе сидят и дети-модели. Вызывается на старте демона ДО любых спавнов;
// чужие бинари не трогает — путь exe сверяется с префиксом. Возвращает число
// убитых.
int sweep_orphans(const std::string& under_dir);

} // namespace lh::proc
