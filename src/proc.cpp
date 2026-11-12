#include "proc.h"

#include "util.h"

#include <httplib.h>

#ifdef _WIN32
#include <psapi.h>
#else
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <cctype>
#ifdef _WIN32
#include <cwctype>
#endif
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <thread>

namespace lh::proc {

namespace fs = std::filesystem;

namespace {

std::string lowered(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return (char)std::tolower(c); });
  return s;
}

#ifdef _WIN32

std::wstring lowered(std::wstring s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](wchar_t c) { return (wchar_t)towlower(c); });
  return s;
}

std::wstring wide_of(const std::string& utf8) {
  if (utf8.empty()) return {};
  int n = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), (int)utf8.size(),
                              nullptr, 0);
  std::wstring w((size_t)(n > 0 ? n : 0), L'\0');
  if (n > 0)
    MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), (int)utf8.size(), w.data(), n);
  return w;
}

std::string narrow_of(const std::wstring& w) {
  if (w.empty()) return {};
  int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0,
                              nullptr, nullptr);
  std::string s((size_t)(n > 0 ? n : 0), '\0');
  if (n > 0)
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), s.data(), n,
                        nullptr, nullptr);
  return s;
}

// Квотинг аргумента для командной строки Windows: кавычки вокруг, внутренние
// кавычки экранируются как \", бэкслэши перед кавычками удваиваются.
std::wstring quote_arg(const std::wstring& s) {
  std::wstring out = L"\"";
  size_t slashes = 0;
  for (wchar_t ch : s) {
    if (ch == L'\\') { ++slashes; continue; }
    if (ch == L'"') {
      out.append(slashes * 2 + 2, L'\\');
      out.push_back(L'"');
    } else {
      out.append(slashes, L'\\');
      out.push_back(ch);
    }
    slashes = 0;
  }
  out.append(slashes * 2, L'\\'); // хвостовые бэкслэши — перед закрывающей кавычкой
  out.push_back(L'"');
  return out;
}

// Все дети — в одном Job Object с kill-on-close: жёсткая смерть llmhost
// закрывает хендлы, и ядро убивает llama-server'ы (контракт «дети не
// переживают родителя» вне graceful-путей; graceful — stop()/stop_tree()).
HANDLE root_job() {
  static HANDLE j = [] {
    HANDLE h = CreateJobObjectW(nullptr, nullptr);
    if (h) {
      JOBOBJECT_EXTENDED_LIMIT_INFORMATION lim{};
      lim.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
      SetInformationJobObject(h, JobObjectExtendedLimitInformation, &lim,
                              sizeof(lim));
    }
    return h;
  }();
  return j;
}

// Читает stderr ребёнка в кольцо 16 КБ; закрывает хендл на EOF (смерть).
void reader_thread(HANDLE rd, std::shared_ptr<Child::Log> log) {
  char buf[4096];
  DWORD got = 0;
  std::string acc;
  while (ReadFile(rd, buf, sizeof(buf), &got, nullptr) && got) {
    {
      std::lock_guard l(log->m);
      acc.append(buf, got);
      if (acc.size() > 32 * 1024) acc.erase(0, acc.size() - 16 * 1024);
      log->tail = acc;
    }
  }
  CloseHandle(rd);
}

#else

// Читает stderr ребёнка в кольцо 16 КБ; закрывает fd на EOF (смерть ребёнка).
void reader_thread(int rd, std::shared_ptr<Child::Log> log) {
  char buf[4096];
  std::string acc;
  for (;;) {
    ssize_t got = ::read(rd, buf, sizeof(buf));
    if (got < 0 && errno == EINTR) continue; // сигнал демону не должен рвать
                                             // чтение: закрытый пайп = SIGPIPE
                                             // в живого ребёнка
    if (got <= 0) break;
    {
      std::lock_guard l(log->m);
      acc.append(buf, (size_t)got);
      // Кольцо: хвост 16 КБ — для разбора ошибок больше не нужно.
      if (acc.size() > 32 * 1024) acc.erase(0, acc.size() - 16 * 1024);
      log->tail = acc;
    }
  }
  ::close(rd);
}

// Код выхода: как вышел (WEXITSTATUS), а по сигналу — конвенция shell 128+sig.
int exit_code_from(int st) {
  if (WIFEXITED(st)) return WEXITSTATUS(st);
  if (WIFSIGNALED(st)) return 128 + WTERMSIG(st);
  return 0;
}

#endif // _WIN32

} // namespace

#ifdef _WIN32

Child::Child() : log_(std::make_shared<Log>()) {}

Child::~Child() {
  // stop() обычно зовут явно; здесь — страховка от утечки хендлов.
  stop();
}

bool Child::start(const std::string& exe, const std::vector<std::string>& args,
                  const std::vector<std::pair<std::string, std::string>>& extra_env) {
  std::lock_guard l(m_);
  args_ = args;
  if (proc_) return false;

  std::wstring wexe = wide_of(exe);
  std::wstring cmd = quote_arg(wexe);
  for (const auto& a : args) cmd += L" " + quote_arg(wide_of(a));

  // Блок окружения «K=V\0…\0\0»: текущее окружение + замены extra_env.
  std::wstring env_block;
  if (!extra_env.empty()) {
    std::map<std::wstring, std::wstring> env;
    LPWCH block = ::GetEnvironmentStringsW();
    if (block) {
      for (LPWCH p = block; *p;) {
        std::wstring kv(p);
        if (kv.size() > 1 && kv[0] != L'=') {
          size_t eq = kv.find(L'=');
          if (eq != std::wstring::npos) env[kv.substr(0, eq)] = kv.substr(eq + 1);
        }
        p += kv.size() + 1;
      }
      ::FreeEnvironmentStringsW(block);
    }
    for (const auto& [k, v] : extra_env) env[wide_of(k)] = wide_of(v);
    for (const auto& [k, v] : env) env_block += k + L"=" + v + L"\0";
    env_block.push_back(L'\0');
  }

  SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
  HANDLE rd = nullptr, wr = nullptr;
  if (!CreatePipe(&rd, &wr, &sa, 0)) return false;
  SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0); // читать будет только наш поток

  STARTUPINFOW si{};
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdInput = nullptr;
  si.hStdOutput = wr;
  si.hStdError = wr;
  PROCESS_INFORMATION pi{};
  BOOL ok = CreateProcessW(wexe.c_str(), cmd.data(), nullptr, nullptr, TRUE,
                           CREATE_NO_WINDOW,
                           env_block.empty() ? nullptr : env_block.data(), nullptr,
                           &si, &pi);
  CloseHandle(wr); // наш конец записи: при смерти ребёнка EOF закроет чтение
  if (!ok) {
    CloseHandle(rd);
    return false;
  }
  CloseHandle(pi.hThread);
  proc_ = pi.hProcess;
  pid_ = pi.dwProcessId;
  if (root_job()) AssignProcessToJobObject(root_job(), pi.hProcess);
  // Персональный джоб (вложенный, Win8+): stop_tree убивает всё дерево
  // ребёнка разом, не трогая остальных детей llmhost в общем джобе.
  job_ = CreateJobObjectW(nullptr, nullptr);
  if (job_) {
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION lim{};
    lim.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    SetInformationJobObject(job_, JobObjectExtendedLimitInformation, &lim,
                            sizeof(lim));
    AssignProcessToJobObject(job_, pi.hProcess);
  }

  std::thread(reader_thread, rd, log_).detach();
  return true;
}

bool Child::alive() {
  std::lock_guard l(m_);
  if (!proc_) return false;
  DWORD code = 0;
  return GetExitCodeProcess(proc_, &code) && code == STILL_ACTIVE;
}

bool Child::exited(int& code) {
  std::lock_guard l(m_);
  if (!proc_) { code = 0; return true; }
  DWORD raw = 0;
  if (!GetExitCodeProcess(proc_, &raw)) { code = 0; return true; }
  if (raw == STILL_ACTIVE) return false;
  code = (int)raw;
  return true;
}

void Child::stop() {
  // Выносим хендлы из-под замка: повторный stop() увидит nullptr и ничего
  // не сделает. Graceful-сигналов Windows не даёт — TerminateProcess сразу;
  // дети в джобе умирают вместе с ним.
  HANDLE p = nullptr, j = nullptr;
  {
    std::lock_guard l(m_);
    p = proc_;
    proc_ = nullptr;
    pid_ = 0;
    j = job_;
    job_ = nullptr;
    if (j) TerminateJobObject(j, 0); // дерево — до TerminateProcess, не после
  }
  if (j) CloseHandle(j);
  if (!p) return;
  TerminateProcess(p, 0);
  WaitForSingleObject(p, 5000);
  CloseHandle(p);
}

void Child::stop_tree() {
  {
    std::lock_guard l(m_);
    if (job_) TerminateJobObject(job_, 0); // роутер и все его дети-модели разом
  }
  stop(); // страховка + освобождение хендлов
}

pid_t Child::pid() {
  std::lock_guard l(m_);
  return pid_;
}

std::vector<std::string> Child::args() {
  std::lock_guard l(m_);
  return args_;
}

std::string Child::log_tail() {
  std::lock_guard l(log_->m);
  return log_->tail;
}

#else // POSIX

Child::Child() : log_(std::make_shared<Log>()) {}

Child::~Child() {
  // stop() обычно зовут явно; здесь — страховка от сирот и зомби.
  stop();
}

bool Child::start(const std::string& exe, const std::vector<std::string>& args,
                  const std::vector<std::pair<std::string, std::string>>& extra_env) {
  std::lock_guard l(m_);
  args_ = args;
  if (!reaped_) return false;

  int fds[2];
  if (::pipe(fds) != 0) return false;

  // argv собираем ДО fork: в ребёнке между fork и exec — только
  // async-signal-safe вызовы, никакой аллокации.
  std::vector<char*> argv;
  argv.reserve(args.size() + 2);
  argv.push_back(const_cast<char*>(exe.c_str()));
  for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
  argv.push_back(nullptr);

  // envp — тоже до fork (та же причина). Замена существующей переменной, не
  // дубль: CUDA видит обе строки «VAR=a» и «VAR=b» непредсказуемо.
  std::vector<std::string> env_storage;
  std::vector<char*> envp;
  {
    for (char** e = environ; e && *e; ++e) env_storage.emplace_back(*e);
    for (const auto& [k, v] : extra_env) {
      std::string kv = k + "=" + v;
      bool replaced = false;
      for (auto& s : env_storage) {
        if (s.size() > k.size() && s.compare(0, k.size(), k) == 0 && s[k.size()] == '=') {
          s = kv;
          replaced = true;
          break;
        }
      }
      if (!replaced) env_storage.push_back(std::move(kv));
    }
    envp.reserve(env_storage.size() + 1);
    for (auto& s : env_storage) envp.push_back(s.data());
    envp.push_back(nullptr);
  }

  pid_t pid = ::fork();
  if (pid < 0) {
    ::close(fds[0]);
    ::close(fds[1]);
    return false;
  }
  if (pid == 0) {
    // Замена Job Object kill-on-close — НЕ через PR_SET_PDEATHSIG: сигнал
    // привязан к смерти ПОРОДИВШЕГО ПОТОКА, а не процесса. Дети у нас
    // спавнятся из коротких воркер-потоков (лестница загрузки отчиталась
    // «готово» и закончилась) — pdeathsig убивал живого llama-server сразу
    // после успешного /health (поймано живьём: healthy в 1 с, зомби в 2 с).
    // Полная семантика «дети не переживают родителя» собирается иначе:
    // graceful-пути — stop_tree (SIGTERM/…/v0/quit), жёсткая смерть демона —
    // systemd KillMode=control-group + подметание сирот на старте (фаза 4).
    if (::getppid() == 1) _exit(127); // родитель успел умереть между fork и exec
    ::setpgid(0, 0);                  // своя группа: stop_tree убьёт всё дерево
    ::dup2(fds[1], STDOUT_FILENO);
    ::dup2(fds[1], STDERR_FILENO);
    int devnull = ::open("/dev/null", O_RDONLY);
    if (devnull >= 0) {
      ::dup2(devnull, STDIN_FILENO);
      ::close(devnull);
    }
    ::close(fds[0]);
    ::close(fds[1]);
    ::execve(exe.c_str(), argv.data(), envp.data());
    _exit(127); // exec не удался (нет файла/прав) — код 127, как у shell
  }

  ::close(fds[1]); // наш конец записи: при смерти ребёнка EOF закроет чтение
  pid_ = pid;
  reaped_ = false;
  exit_code_ = 0;
  std::thread(reader_thread, fds[0], log_).detach();
  return true;
}

bool Child::alive() {
  std::lock_guard l(m_);
  if (reaped_) return false;
  int st = 0;
  pid_t r = ::waitpid(pid_, &st, WNOHANG);
  if (r == 0) return true;
  exit_code_ = r == pid_ ? exit_code_from(st) : 0; // r<0 — ECHILD: забрал кто-то другой
  reaped_ = true;
  return false;
}

bool Child::exited(int& code) {
  std::lock_guard l(m_);
  if (!reaped_) {
    int st = 0;
    pid_t r = ::waitpid(pid_, &st, WNOHANG);
    if (r == pid_) { exit_code_ = exit_code_from(st); reaped_ = true; }
    else if (r < 0) { exit_code_ = 0; reaped_ = true; }
  }
  if (!reaped_) return false; // ещё жив — code не трогаем
  code = exit_code_;
  return true;
}

void Child::stop() {
  // Берём обязанность reap'а на себя вне замка: повторный stop() и alive()
  // увидят reaped_ и не будут конкурировать за waitpid.
  pid_t p = -1;
  {
    std::lock_guard l(m_);
    if (!reaped_ && pid_ > 0) p = pid_;
    pid_ = -1;
    reaped_ = true;
  }
  if (p <= 0) return;
  ::kill(p, SIGTERM);
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  for (;;) {
    int st = 0;
    pid_t r = ::waitpid(p, &st, WNOHANG);
    if (r != 0) return; // забрали (или ECHILD — reap сделал читавший поток… нет,
                        // waitpid только наш; r<0 — процесс уже не существует)
    if (std::chrono::steady_clock::now() > deadline) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  ::kill(p, SIGKILL);
  int st = 0;
  ::waitpid(p, &st, 0);
}

void Child::stop_tree() {
  pid_t p = -1;
  {
    std::lock_guard l(m_);
    if (!reaped_ && pid_ > 0) p = pid_;
  }
  if (p > 0) ::kill(-p, SIGKILL); // вся группа разом: роутер и дети-модели
  stop();                         // SIGTERM/KILL самому + забрать зомби
}

pid_t Child::pid() {
  std::lock_guard l(m_);
  return pid_ > 0 ? pid_ : 0;
}

std::vector<std::string> Child::args() {
  std::lock_guard l(m_);
  return args_;
}

std::string Child::log_tail() {
  std::lock_guard l(log_->m);
  return log_->tail;
}

#endif // _WIN32/POSIX — Child

Wait wait_health(Child& c, int port, int timeout_ms, const std::string& path) {
  httplib::Client cli("127.0.0.1", port);
  cli.set_connection_timeout(std::chrono::seconds(1));
  cli.set_read_timeout(std::chrono::seconds(2));
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  bool seen_exit = false;
  for (;;) {
    int code = 0;
    if (c.exited(code)) seen_exit = true;
    auto res = cli.Get(path.c_str());
    if (res && res->status == 200) return Wait::Healthy;
    if (std::chrono::steady_clock::now() > deadline) {
      std::string tail = c.log_tail();
      if (seen_exit || looks_oom(tail)) return Wait::Oom;
      return Wait::Timeout;
    }
    if (seen_exit) {
      // Дать stderr догнать и решить: OOM или другое падение.
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      return looks_oom(c.log_tail()) ? Wait::Oom : Wait::Exited;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  }
}

bool looks_oom(const std::string& log_in) {
  std::string log = lowered(log_in);
  static const char* marks[] = {
      "out of memory", "failed to allocate", "allocation failure", "alloc failed",
      "vkallocatememory", "cuda out of memory", "ggml_backend_alloc", "not enough memory",
      "cuda_error_out_of_memory", "vk::outofmemory",
  };
  for (const char* m : marks)
    if (log.find(m) != std::string::npos) return true;
  return false;
}

// Падение ПО ЧЕРНОВИКУ: загрузка draft-файла или создание MTP-контекста.
// Важно различать с OOM ствола: OOM под MTP — это cudaMalloc compute-буферов
// головы (07.10: чужая голова в TURBO ловила его 14 раз подряд, failover винил
// память ствола и крутил 502-бурю), GGML_ASSERT — размерная несовместимость.
// Строки-маркеры из репродукции на ручном llama-server; ищутся по lowered-логу.
bool looks_draft_failure(const std::string& log_in) {
  std::string log = lowered(log_in);
  static const char* marks[] = {
      "failed to create mtp context", "failed to load draft",
      "draft model load failed", "mtp input row width",
  };
  for (const char* m : marks)
    if (log.find(m) != std::string::npos) return true;
  return false;
}

std::string last_meaningful_line(const std::string& log) {
  size_t end = log.find_last_not_of("\r\n \t");
  if (end == std::string::npos) return {};
  size_t start = log.find_last_of("\r\n", end);
  std::string line = log.substr(start == std::string::npos ? 0 : start + 1,
                                end - (start == std::string::npos ? -1 : start));
  // Строки-повторители прогресса не несут смысла.
  if (line.size() < 4 || line.find('=') != std::string::npos) return {};
  return line;
}

std::vector<std::string> tail_lines(const std::string& log, int n) {
  std::vector<std::string> out;
  size_t pos = log.size();
  while ((int)out.size() < n && pos > 0) {
    size_t end = log.find_last_not_of("\r\n \t", pos - 1);
    if (end == std::string::npos) break;
    size_t start = log.find_last_of("\r\n", end);
    std::string line = log.substr(start == std::string::npos ? 0 : start + 1,
                                  end - (start == std::string::npos ? -1 : start));
    pos = start == std::string::npos ? 0 : start;
    if (!line.empty()) out.push_back(line);
  }
  std::reverse(out.begin(), out.end());
  return out;
}

#ifdef _WIN32

int sweep_orphans(const std::string& under_dir) {
  if (under_dir.empty()) return 0;
  wchar_t canon[4096];
  DWORD n = GetFullPathNameW(wide_of(under_dir).c_str(), 4096, canon, nullptr);
  if (!n || n >= 4096) return 0; // движок ещё не ставился
  std::wstring root(canon, n);
  while (root.size() > 3 && (root.back() == L'\\' || root.back() == L'/'))
    root.pop_back();
  std::wstring rlow = lowered(root);
  for (auto& ch : rlow)
    if (ch == L'/') ch = L'\\';

  // Джобы kill-on-close делают настоящих сирот почти невозможными, свип —
  // страховка: EnumProcesses + префикс-матч exe по каталогу движков.
  DWORD pids[4096] = {}, needed = 0;
  if (!EnumProcesses(pids, sizeof(pids), &needed)) return 0;
  int killed = 0;
  for (unsigned i = 0; i < needed / sizeof(DWORD); ++i) {
    if (pids[i] <= 4 || pids[i] == GetCurrentProcessId()) continue;
    HANDLE p = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_TERMINATE,
                           FALSE, pids[i]);
    if (!p) continue; // чужие/системные — прав нет, они и не наши
    WCHAR path[4096] = {};
    DWORD sz = 4096;
    if (QueryFullProcessImageNameW(p, 0, path, &sz) && sz) {
      std::wstring exe = lowered(std::wstring(path, sz));
      for (auto& ch : exe)
        if (ch == L'/') ch = L'\\';
      if (exe.rfind(rlow + L"\\", 0) == 0) { // не наш бинарь — не трогаем
        if (TerminateProcess(p, 0)) {
          ++killed;
          logf("orphan: убит осиротевший процесс %lu (%s)", pids[i],
               narrow_of(std::wstring(path, sz)).c_str());
        } else {
          logf("orphan: процесс %lu (%s) не убивается: %lu", pids[i],
               narrow_of(std::wstring(path, sz)).c_str(), GetLastError());
        }
      }
    }
    CloseHandle(p);
  }
  if (killed == 0)
    logf("orphan: осиротевших процессов под %s нет", under_dir.c_str());
  return killed;
}

#else // POSIX

int sweep_orphans(const std::string& under_dir) {
  if (under_dir.empty()) return 0;
  char resolved[4096];
  if (!::realpath(under_dir.c_str(), resolved)) return 0; // движок ещё не ставился
  const std::string root = resolved;                      // канонический, без хвостовых слэшей

  // pgid → exe (для лога). Читаем /proc: readlink на чужие процессы не даёт
  // прав — они и не наши; умершие между readdir и readlink просто пропустятся.
  std::map<pid_t, std::string> groups;
  std::error_code ec;
  for (fs::directory_iterator it("/proc", ec), end; !ec && it != end; it.increment(ec)) {
    const std::string name = it->path().filename().string();
    if (name.empty() || name[0] < '0' || name[0] > '9') continue;
    pid_t pid = (pid_t)::atoi(name.c_str());
    if (pid <= 1 || pid == ::getpid()) continue;
    char buf[4096];
    ssize_t n = ::readlink(("/proc/" + name + "/exe").c_str(), buf, sizeof(buf) - 1);
    if (n <= 0) continue;
    buf[n] = '\0';
    if (std::string(buf).rfind(root + "/", 0) != 0) continue; // не наш бинарь
    pid_t g = ::getpgid(pid);
    if (g <= 1 || g == ::getpgrp()) continue;
    groups.emplace(g, buf);
  }
  int killed = 0;
  for (const auto& [g, exe] : groups) {
    if (::kill(-g, SIGKILL) == 0) {
      ++killed;
      logf("orphan: убита осиротевшая группа %d (%s)", (int)g, exe.c_str());
    } else {
      logf("orphan: группа %d (%s) не убивается: %s", (int)g, exe.c_str(),
           strerror(errno));
    }
  }
  if (killed == 0) logf("orphan: осиротевших процессов под %s нет", root.c_str());
  return killed;
}

#endif // _WIN32/POSIX — sweep_orphans

} // namespace lh::proc
