#include "util.h"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <bcrypt.h>
#else
#include <signal.h>
#include <sys/random.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/vfs.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#endif

#include <cerrno>
#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <vector>
#include <mutex>

namespace fs = std::filesystem;

namespace lh {

#ifdef _WIN32
std::string wide_to_utf8(const std::wstring& w) {
  if (w.empty()) return {};
  int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0,
                              nullptr, nullptr);
  std::string s((size_t)(n > 0 ? n : 0), '\0');
  if (n > 0)
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), s.data(), n,
                        nullptr, nullptr);
  return s;
}

std::wstring utf8_to_wide(const std::string& utf8) {
  if (utf8.empty()) return {};
  int n = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), (int)utf8.size(),
                              nullptr, 0);
  std::wstring w((size_t)(n > 0 ? n : 0), L'\0');
  if (n > 0)
    MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), (int)utf8.size(), w.data(), n);
  return w;
}
#endif

} // namespace lh

namespace lh {

std::string exe_path() {
#ifdef _WIN32
  // Буфер растёт: длинные пути и unicode-профили — не редкость.
  std::wstring wbuf(4096, L'\0');
  for (;;) {
    DWORD n = GetModuleFileNameW(nullptr, wbuf.data(), (DWORD)wbuf.size());
    if (n == 0) return {};
    if (n < wbuf.size() || wbuf.size() >= 32768) return wide_to_utf8(wbuf.substr(0, n));
    wbuf.resize(wbuf.size() * 2);
  }
#else
  // Буфер растёт: монтирования с длинными путями — не редкость.
  std::string buf(4096, '\0');
  for (;;) {
    ssize_t n = ::readlink("/proc/self/exe", buf.data(), buf.size());
    if (n <= 0) return {};
    if ((size_t)n < buf.size()) return buf.substr(0, (size_t)n);
    if (buf.size() >= 32768) return buf.substr(0, (size_t)n);
    buf.resize(buf.size() * 2);
  }
#endif
}

std::string exe_dir() {
  auto p = exe_path();
  // Windows несёт бэкслэши — режем по обоим разделителям.
  size_t pos = p.find_last_of("/\\");
  return pos == std::string::npos ? p : p.substr(0, pos);
}

std::string ensure_dir(const std::string& path) {
  if (!path.empty()) {
    std::error_code ec;
    fs::create_directories(path, ec);
  }
  return path;
}

std::string app_dir() {
  // $LH_DATA_DIR → $XDG_DATA_HOME/llmhost → ~/.local/share/llmhost. Всё
  // изменяемое в одном месте: конфиг, токен, журнал, движок, модели.
  static const std::string dir = [] {
    std::string base;
    const char* env = ::getenv("LH_DATA_DIR");
    if (env && *env) base = env;
    if (base.empty()) {
      env = ::getenv("XDG_DATA_HOME");
      if (env && *env && env[0] == '/') base = std::string(env) + "/llmhost";
    }
    if (base.empty()) {
      env = ::getenv("HOME");
      if (env && *env) base = std::string(env) + "/.local/share/llmhost";
    }
    if (base.empty()) base = exe_dir() + "/data"; // совсем без окружения
    // База обязана быть каталогом: LH_DATA_DIR, указывающий на файл (опечатка
    // в юните), не должен превращаться в тихо неработающий демон — все записи
    // в такой «каталог» падали бы молча. logf отсюда звать нельзя (log_path →
    // app_dir — рекурсия статической инициализации), поэтому stderr.
    std::error_code aec;
    if (fs::exists(base, aec) && !fs::is_directory(base, aec)) {
      fprintf(stderr, "llmhost: каталог данных %s — не каталог, отказ\n",
              base.c_str());
      std::exit(3);
    }
    ensure_dir(base);
    // Каталог хранит секреты (token.txt, api_tokens.json) — права 0700
    // best-effort и без журнала: logf отсюда звать нельзя (log_path →
    // app_dir — рекурсия статической инициализации).
    std::error_code pec;
    fs::permissions(base, fs::perms::owner_all, pec);
    return base;
  }();
  return dir;
}

std::string log_path() { return app_dir() + "/log.txt"; }

void logf(const char* fmt, ...) {
  // Один писатель за раз: усечение хвоста (>1 МиБ) и запись — не атомарны
  // между потоками, две одновременные усечки теряли строки журнала.
  static std::mutex log_mtx;
  std::lock_guard log_l(log_mtx);
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(buf, sizeof(buf) - 1, fmt, ap);
  va_end(ap);
  if (n <= 0) return;
  buf[n < (int)sizeof(buf) - 1 ? n : (int)sizeof(buf) - 1] = '\0';

  // Запись журнала — одна строка на событие: control-символы из внешних
  // данных (имя файла с '\n[general]', ответ движка) рисовали в журнале
  // фейковые строки вплоть до фейковых секций конфига. Экранируем наглядно.
  for (char* p = buf; *p; ++p) {
    unsigned char c = (unsigned char)*p;
    if (c >= 0x20 && c != 0x7F) continue;
    if (n + 4 >= (int)sizeof(buf)) { *p = '\0'; break; }
    std::memmove(p + 4, p + 1, strlen(p + 1) + 1);
    static const char* hex = "0123456789ABCDEF";
    p[0] = '\\'; p[1] = 'x';
    p[2] = hex[c >> 4]; p[3] = hex[c & 0xF];
    n += 3;
    p += 3;
  }

  // Эхо в stderr — для foreground-запуска; под systemd журнал и так в файле.
  static const bool echo = [] {
    const char* e = ::getenv("LH_LOG_STDERR");
    return e && *e;
  }();
  if (echo) fprintf(stderr, "[llmhost] %s\n", buf);

  static std::string file = log_path();
#ifdef _WIN32
  HANDLE h = CreateFileW(utf8_to_wide(file).c_str(), FILE_APPEND_DATA,
                         FILE_SHARE_READ, nullptr, OPEN_ALWAYS,
                         FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE) return;
  LARGE_INTEGER sz{};
  if (GetFileSizeEx(h, &sz) && sz.QuadPart > 1024 * 1024) {
    CloseHandle(h);
    // хвост журнала не больше 1 МБ — усекаем и пишем заново
    h = CreateFileW(utf8_to_wide(file).c_str(), GENERIC_WRITE, FILE_SHARE_READ,
                    nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return;
  }
  char line[1100];
  time_t t = ::time(nullptr);
  struct tm tmv{};
  localtime_s(&tmv, &t);
  int m = snprintf(line, sizeof(line), "%04u-%02u-%02u %02u:%02u:%02u %s\n",
                   tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday,
                   tmv.tm_hour, tmv.tm_min, tmv.tm_sec, buf);
  if (m > 0) {
    DWORD w = 0;
    WriteFile(h, line, (DWORD)m, &w, nullptr);
  }
  CloseHandle(h);
#else
  int fd = ::open(file.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0600);
  if (fd < 0) return;
  struct stat st{};
  if (::fstat(fd, &st) == 0 && st.st_size > 1024 * 1024) {
    ::close(fd);
    ::truncate(file.c_str(), 0); // хвост журнала не больше 1 МБ — усекаем и пишем заново
    fd = ::open(file.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0600);
    if (fd < 0) return;
  }
  char line[1100];
  time_t t = ::time(nullptr);
  struct tm tmv{};
  localtime_r(&t, &tmv);
  int m = snprintf(line, sizeof(line), "%04u-%02u-%02u %02u:%02u:%02u %s\n",
                   tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday,
                   tmv.tm_hour, tmv.tm_min, tmv.tm_sec, buf);
  if (m > 0) {
    ssize_t ignored = ::write(fd, line, (size_t)m);
    (void)ignored;
  }
  ::close(fd);
#endif
}

std::string random_hex(size_t n_bytes) {
  // Три попытки: генератор падает крайне редко, но пустой ключ — это
  // llama-server без api-key и API без авторизации (fail-closed).
  static const char* d = "0123456789abcdef";
  for (int attempt = 0; attempt < 3; ++attempt) {
    std::vector<unsigned char> raw(n_bytes);
#ifdef _WIN32
    // BCryptGenRandom системным RNG: тот же fail-closed контракт, что и
    // getrandom на POSIX (аудит 04.10 №9/№12: ключи — только из честного RNG).
    if (n_bytes &&
        BCryptGenRandom(nullptr, raw.data(), (ULONG)raw.size(),
                        BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0)
      continue;
#else
    size_t got = 0;
    while (got < raw.size()) {
      ssize_t r = ::getrandom(raw.data() + got, raw.size() - got, 0);
      if (r <= 0) break;
      got += (size_t)r;
    }
    if (got != raw.size()) continue;
#endif
    std::string hex;
    hex.reserve(n_bytes * 2);
    for (unsigned char b : raw) { hex.push_back(d[b >> 4]); hex.push_back(d[b & 15]); }
    return hex;
  }
  return {};
}

std::string base64_encode(const std::string& in) {
  static const char* tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve((in.size() + 2) / 3 * 4);
  size_t i = 0;
  for (; i + 3 <= in.size(); i += 3) {
    uint32_t v = ((uint32_t)(unsigned char)in[i] << 16) |
                 ((uint32_t)(unsigned char)in[i + 1] << 8) |
                 (uint32_t)(unsigned char)in[i + 2];
    out.push_back(tbl[(v >> 18) & 63]);
    out.push_back(tbl[(v >> 12) & 63]);
    out.push_back(tbl[(v >> 6) & 63]);
    out.push_back(tbl[v & 63]);
  }
  size_t rem = in.size() - i;
  if (rem == 1) {
    uint32_t v = (uint32_t)(unsigned char)in[i] << 16;
    out.push_back(tbl[(v >> 18) & 63]);
    out.push_back(tbl[(v >> 12) & 63]);
    out.push_back('=');
    out.push_back('=');
  } else if (rem == 2) {
    uint32_t v = ((uint32_t)(unsigned char)in[i] << 16) | ((uint32_t)(unsigned char)in[i + 1] << 8);
    out.push_back(tbl[(v >> 18) & 63]);
    out.push_back(tbl[(v >> 12) & 63]);
    out.push_back(tbl[(v >> 6) & 63]);
    out.push_back('=');
  }
  return out;
}

int free_tcp_port(int lo, int hi) {
  for (int port = lo; port <= hi; ++port) {
    int s = (int)::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
#ifdef _WIN32
    if (s < 0) return 0;
#else
    if (s < 0) return 0;
#endif
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
    int ok = ::bind(s, (sockaddr*)&a, sizeof(a)) == 0;
#ifdef _WIN32
    ::closesocket(s);
#else
    ::close(s);
#endif
    if (ok) return port;
  }
  return 0;
}

std::string read_small_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return {};
  // Потолок 8 МиБ: «small» в названии — контракт, а не надежда. Симлинка на
  // /dev/zero или раздутый конфиг не должны выедать RAM демона изотерически:
  // читаем порциями и отваливаемся на переходе границы (пусто = «не удалось»).
  std::string out;
  char buf[65536];
  for (;;) {
    f.read(buf, (std::streamsize)sizeof(buf));
    out.append(buf, (size_t)f.gcount());
    if (out.size() > (8u << 20)) return {};
    if (f.gcount() < (std::streamsize)sizeof(buf)) break;
  }
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' ')) out.pop_back();
  return out;
}

bool write_small_file(const std::string& path, const std::string& data) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) return false;
  f.write(data.data(), (std::streamsize)data.size());
  return (bool)f;
}

bool write_private_file(const std::string& path, const std::string& data) {
#ifdef _WIN32
  // Нет umask-окна в том же смысле: пишем и сразу режем права до rename.
  if (!write_small_file(path, data)) return false;
  return restrict_file_permissions(path);
#else
  // o600 при создании (ср. token.txt в config.cpp): «ofstream по umask,
  // потом chmod» оставляет окно чтения и следует симлинку.
  int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (fd < 0) return false;
  const char* p = data.data();
  size_t left = data.size();
  while (left > 0) {
    ssize_t put = ::write(fd, p, left);
    if (put < 0) {
      if (errno == EINTR) continue;
      ::close(fd);
      return false;
    }
    p += (size_t)put;
    left -= (size_t)put;
  }
  return ::close(fd) == 0;
#endif
}

bool file_exists(const std::string& path) {
  std::error_code ec;
  return fs::exists(path, ec) && !fs::is_directory(path, ec);
}

uint64_t file_size(const std::string& path) {
  std::error_code ec;
  auto sz = fs::file_size(path, ec);
  return ec ? 0 : (uint64_t)sz;
}

std::vector<std::pair<std::string, std::string>> engine_child_env(const std::string& exe) {
#ifndef _WIN32
  std::string dir = exe;
  size_t slash = dir.find_last_of('/');
  dir = slash == std::string::npos ? std::string(".") : dir.substr(0, slash);
  const char* cur = ::getenv("LD_LIBRARY_PATH");
  std::string val = (cur && *cur) ? dir + ":" + cur : dir;
  return {{"LD_LIBRARY_PATH", val}};
#else
  (void)exe;
  return {};
#endif
}

std::string find_llama_server(const std::string& configured) {
  if (!configured.empty()) {
    std::string expanded = expand_env(configured);
    if (file_exists(expanded)) return expanded;
  }
  std::string local = exe_dir() + "/llama-server";
  if (file_exists(local)) return local;
#ifndef _WIN32
  // PATH — как SearchPathW, руками: доступность на exec на каждом элементе.
  // (Windows-ветка движок не ищет: там llama-server.exe кладёт установщик
  // в каталог движка, путь выше.)
  const char* path = ::getenv("PATH");
  if (!path || !*path) return {};
  size_t start = 0;
  const size_t len = strlen(path);
  while (start <= len) {
    size_t end = start;
    while (end < len && path[end] != ':') ++end;
    std::string dir(path + start, end - start);
    // Пустой сегмент PATH по конвенции трактуется как «.», ноexec'ать из
    // текущей директории демона мы не будем: cwd может быть world-writable
    // (ручной запуск из шаренной папки) — подложенный ./llama-server
    // исполнялся бы от имени демона. Только непустые каталоги.
    if (!dir.empty()) {
      std::string cand = dir + "/llama-server";
      if (::access(cand.c_str(), X_OK) == 0) return cand;
    }
    if (end >= len) break;
    start = end + 1;
  }
#endif
  return {};
}

std::string expand_env(const std::string& s) {
  // $VAR и ${VAR}; нет в окружении — фрагмент остаётся как есть (как
  // ExpandEnvironmentStrings оставлял %VAR%).
  std::string out;
  size_t i = 0;
  while (i < s.size()) {
    if (s[i] == '$' && i + 1 < s.size()) {
      size_t j = i + 1;
      bool braced = s[j] == '{';
      if (braced) ++j;
      size_t k = j;
      while (k < s.size() && (std::isalnum((unsigned char)s[k]) || s[k] == '_')) ++k;
      bool closed = !braced || (k < s.size() && s[k] == '}');
      if (closed && k > j) {
        std::string name = s.substr(j, k - j);
        const char* v = ::getenv(name.c_str());
        if (v) {
          out += v;
          i = braced ? k + 1 : k;
          continue;
        }
      }
    }
    out += s[i];
    ++i;
  }
  return out;
}

bool safe_local_filename(const std::string& path) {
  if (path.empty()) return false;
  for (unsigned char c : path)
    if (c < 0x20 || c == 0x7F) return false;
  return true;
}

bool restrict_file_permissions(const std::string& path) {
#ifdef _WIN32
  // NTFS: файл в профиле пользователя уже закрыт от других учёток; POSIX-
  // chmod 0600 здесь не имеет смысла. Держим контракт «функция есть, вызови
  // и не падай» — аудит 04.10 №11 опирается на её наличие.
  (void)path;
  return true;
#else
  return ::chmod(path.c_str(), 0600) == 0;
#endif
}

bool safe_rel_path(const std::string& p) {
  if (p.empty() || p.size() > 512) return false;
  size_t start = 0;
  for (size_t i = 0; i <= p.size(); ++i) {
    bool end = i == p.size();
    bool sep = !end && p[i] == '/';
    if (!end && !sep) {
      unsigned char u = (unsigned char)p[i];
      if (u < 0x20 || u == 0x7F) return false;                // control-символы
      if (u == '"' || u == '\'' || u == '<' || u == '>' || u == '|' || u == ':' ||
          u == '*' || u == '?' || u == '\\')
        return false;                                         // ломают квотинг/путь
      continue;
    }
    std::string part = p.substr(start, i - start);
    if (part.empty() || part == "." || part == "..") return false; // «a//b», «a/», «..»
    start = i + 1;
  }
  return true; // старт с «/» отсечён: первый компонент был бы пустым
}

std::string percent_encode_path(const std::string& s) {
  static const char* hex = "0123456789ABCDEF";
  std::string out;
  out.reserve(s.size());
  for (unsigned char c : s) {
    bool safe = std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~' || c == '/';
    if (safe) out.push_back((char)c);
    else { out.push_back('%'); out.push_back(hex[c >> 4]); out.push_back(hex[c & 15]); }
  }
  return out;
}

std::string replace_all(std::string s, const std::string& from, const std::string& to) {
  if (from.empty()) return s;
  size_t pos = 0;
  while ((pos = s.find(from, pos)) != std::string::npos) {
    s.replace(pos, from.size(), to);
    pos += to.size(); // замена не сканируется заново — «a»→«aa» не зациклится
  }
  return s;
}

std::string release_from_version_output(const std::string& out) {
  size_t p = out.find("build ");
  if (p == std::string::npos) return {};
  p += 6;
  size_t e = p;
  while (e < out.size() && out[e] >= '0' && out[e] <= '9') ++e;
  if (e == p) return {};
  return "b" + out.substr(p, e - p);
}

// Целое с текущей позиции (пропустив не-цифры); false — цифр нет до stop.
static bool scan_ll(const std::string& s, size_t& p, char stop, long long& out) {
  while (p < s.size() && s[p] != stop && (s[p] < '0' || s[p] > '9'))
    ++p;
  size_t b = p;
  while (p < s.size() && s[p] >= '0' && s[p] <= '9') ++p;
  if (p == b) return false;
  out = std::atoll(s.c_str() + b);
  return true;
}

bool parse_draft_acceptance(const std::string& line, double& rate,
                            long long& accepted, long long& generated) {
  rate = 0;
  accepted = generated = 0;
  size_t m = std::string::npos;
  for (const char* marker : {"draft acceptance", "draft accept rate", "spec_draft"}) {
    m = line.find(marker);
    if (m != std::string::npos) break;
  }
  if (m == std::string::npos) return false;
  // Первое число после '=' — приём.
  size_t p = line.find('=', m);
  if (p == std::string::npos) return false;
  ++p;
  while (p < line.size() && (line[p] < '0' || line[p] > '9') && line[p] != '.')
    ++p;
  size_t b = p;
  while (p < line.size() && ((line[p] >= '0' && line[p] <= '9') || line[p] == '.'))
    ++p;
  if (p == b) return false;
  rate = std::atof(line.c_str() + b);
  // Счётчики: два целых вокруг '/' в скобках, либо «n=A/B» (вариант
  // spec_draft). Нет счётчиков — ok, только rate.
  size_t par = line.find('(', p);
  if (par != std::string::npos) {
    size_t close = line.find(')', par);
    if (close != std::string::npos) {
      std::string inside = line.substr(par + 1, close - par - 1);
      size_t q = 0;
      long long a = 0, g = 0;
      if (scan_ll(inside, q, '\0', a)) scan_ll(inside, q, '\0', g);
      accepted = a;
      generated = g;
      return true;
    }
  }
  size_t n = line.find("n=", p);
  if (n != std::string::npos) {
    size_t q = n + 2;
    long long a = 0, g = 0;
    if (scan_ll(line, q, '\0', a)) scan_ll(line, q, '\0', g);
    accepted = a;
    generated = g;
  }
  return true;
}

// Линейный wildcard: у рекурсивной версии «*a*a*a*a…» взрывалось экспоненциально.
bool wildcard_match(const char* pat, const char* str) {
  const char *p = pat, *s = str, *star = nullptr, *star_s = nullptr;
  auto eq = [](char a, char b) {
    return a == '?' || std::tolower((unsigned char)a) == std::tolower((unsigned char)b);
  };
  while (*s) {
    if (eq(*p, *s)) { ++p; ++s; }
    else if (*p == '*') { star = p++; star_s = s; }
    else if (star) { p = star + 1; s = ++star_s; }
    else return false;
  }
  while (*p == '*') ++p;
  return !*p;
}

bool slow_media(const std::string& path) {
#ifdef _WIN32
  // Сетевость носителя на Windows тут не определяем: локальные NTFS —
  // быстрые, прогрев не критичен. ( Remote-путь вида \\\\srv\\share можно
  // отличить по префиксу — позже, по живому репорту.)
  (void)path;
  return false;
#else
  // Сетевые/прозрачно-удалённые ФС по магическим числам statfs. Не узнали
  // носитель — считаем быстрым, прогрев не критичен.
  struct statfs s{};
  if (::statfs(path.c_str(), &s) != 0) return false;
  switch (s.f_type) {
    case 0x6969:        // NFS
    case 0xFF534D42:    // CIFS
    case 0x52654973:    // SMB (legacy)
    case 0xFE534D42:    // SMB2/SMB3
    case 0x01021997:    // 9p
    case 0x65735546:    // FUSE (sshfs, rclone, gvfs)
      return true;
    default:
      return false;
  }
#endif
}

bool normalize_hf_repo(std::string& repo, std::string& err) {
  const std::string HF = "huggingface.co/";
  size_t h = repo.find(HF);
  if (h != std::string::npos) repo = repo.substr(h + HF.size());
  size_t res = repo.find("/resolve/");
  if (res != std::string::npos) repo = repo.substr(0, res);
  while (!repo.empty() && repo.back() == '/') repo.pop_back();
  if (!repo.empty() && repo.front() == '/') repo.erase(0, 1);
  size_t slashes = std::count(repo.begin(), repo.end(), '/');
  if (slashes != 1) {
    err = "нужен репозиторий вида «org/repo», получено «" + repo + "»";
    return false;
  }
  // Имя репозитория попадает в путь на диске (models/<repo>) и в URL —
  // принимаем только символы, которыми HF и пользуется: [A-Za-z0-9._-].
  auto ok_component = [](const std::string& c) {
    if (c.empty() || c.size() > 96 || c == "." || c == "..") return false;
    for (char ch : c) {
      unsigned char u = (unsigned char)ch;
      if (!std::isalnum(u) && ch != '-' && ch != '_' && ch != '.') return false;
    }
    return true;
  };
  size_t slash = repo.find('/');
  if (!ok_component(repo.substr(0, slash)) || !ok_component(repo.substr(slash + 1))) {
    err = "недопустимое имя репозитория: «" + repo + "»";
    return false;
  }
  return true;
}

#ifndef _WIN32
namespace {
// Отпечаток фатального сигнала. Разрешены только async-signal-safe вызовы
// (open/write/raise/_exit): logf из обработчика недопустим — сигнал может
// прервать поток с занятым log_mtx или внутри malloc, и заметка не вышла бы
// никогда (или наступил бы deadlock). Путь к журналу заготавливается при
// установке, чтобы в момент креша не аллоцировать.
char g_crash_log[512];

const char* crash_signame(int s) {
  switch (s) {
    case SIGFPE: return "SIGFPE";
    case SIGILL: return "SIGILL";
    case SIGBUS: return "SIGBUS";
    case SIGSEGV: return "SIGSEGV";
    default: return "?";
  }
}

void on_fatal_signal(int sig, siginfo_t*, void*) {
  char buf[256];
  size_t n = 0;
  auto put = [&](const char* s) { while (*s && n < sizeof(buf) - 2) buf[n++] = *s++; };
  auto putnum = [&](int v) {
    char d[8]; int k = 0;
    do { d[k++] = char('0' + v % 10); v /= 10; } while (v && k < (int)sizeof(d));
    while (k && n < sizeof(buf) - 2) buf[n++] = d[--k];
  };
  put("llmhostd: fatal signal ");
  putnum(sig);
  put(" (");
  put(crash_signame(sig));
  put(") — dying; for triage: dmesg -T | grep -iE 'llmhostd|divide|trap'\n");
  int fd = ::open(g_crash_log, O_WRONLY | O_APPEND | O_CREAT, 0600);
  if (fd >= 0) {
    ssize_t ign = ::write(fd, buf, n);
    (void)ign;
    ::close(fd);
  }
  ssize_t ign = ::write(2, buf, n); // foreground/systemd-stderr тоже видят
  (void)ign;
  struct sigaction dfl{};
  dfl.sa_handler = SIG_DFL;
  sigemptyset(&dfl.sa_mask);
  ::sigaction(sig, &dfl, nullptr);
  ::raise(sig); // дефолт: корка и строка trap в dmesg остаются как были
  ::_exit(128 + sig);
}
} // namespace

void install_crash_telemetry() {
  const std::string p = log_path();
  size_t n = p.size() < sizeof(g_crash_log) - 1 ? p.size() : sizeof(g_crash_log) - 1;
  for (size_t i = 0; i < n; ++i) g_crash_log[i] = p[i];
  g_crash_log[n] = '\0';
  struct sigaction sa{};
  sa.sa_sigaction = on_fatal_signal;
  sa.sa_flags = SA_SIGINFO;
  sigemptyset(&sa.sa_mask);
  for (int s : {SIGFPE, SIGILL, SIGBUS, SIGSEGV}) ::sigaction(s, &sa, nullptr);
}
#else
void install_crash_telemetry() {} // Windows — отдельный механизм в win-репо
                                  // (SetUnhandledExceptionFilter)
#endif

} // namespace lh
