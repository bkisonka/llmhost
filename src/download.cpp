#include "download.h"

#include "proc.h"
#include "util.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstring>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <thread>
#include <vector>

namespace lh::dl {

namespace fs = std::filesystem;

namespace {

// Системный curl — ТОЛЬКО фиксированный путь: поиск в PATH открывал подмену,
// а curl получает HF-токен.
const char* kCurl = "/usr/bin/curl";

std::string find_curl() {
  if (file_exists(kCurl)) return kCurl;
  return {};
}

// Приватный временный файл: имя с random_hex (в общем /tmp предсказуемое
// имя — готовая мишень для симлинка), O_CREAT|O_EXCL (гонку создания не
// оставляем вовсе) и 0600. Пусто — не удалось. Через это проходят HF-токен
// (header-файл), тело запроса и тело ответа: до фикса всё лежало с 0644 —
// токен был читаем другими пользователями машины всё время скачивания.
std::string make_private_tmp(const char* tag) {
  std::error_code ec;
  std::string dir = fs::temp_directory_path(ec).string();
  if (ec || dir.empty()) dir = "/tmp";
  for (int attempt = 0; attempt < 3; ++attempt) {
    const std::string rnd = random_hex(8);
    if (rnd.size() < 16) return {}; // генератор случайности умер — fail-closed
    std::string path = dir + "/llmhost-" + tag + "-" + rnd + ".tmp";
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd >= 0) {
      ::close(fd);
      return path;
    }
    if (errno != EEXIST) return {};
  }
  return {};
}

// Токен уходит только на huggingface.co и его CDN-поддомены. Хост — по
// префиксу до первого «/?:#», а не подстрокой: подстрока матчится и в
// query чужого URL («?ref=huggingface.co»), и в сторонние зеркала —
// hf-mirror.com работает третьих рук и токена не получает вовсе.
bool hf_url(const std::string& url) {
  // Только https: под http токен HF уезжал бы в открытом виде (заголовок
  // прикладывается к запросу целиком, -H @файл).
  for (const char* scheme : {"https://"}) {
    size_t n = strlen(scheme);
    if (url.compare(0, n, scheme) != 0) continue;
    std::string host = url.substr(n);
    size_t end = host.find_first_of("/?#");
    if (end != std::string::npos) host = host.substr(0, end);
    size_t portc = host.rfind(':');
    if (portc != std::string::npos) host = host.substr(0, portc);
    std::transform(host.begin(), host.end(), host.begin(),
                   [](unsigned char c) { return (char)tolower(c); });
    const std::string base = "huggingface.co";
    return host == base ||
           (host.size() > base.size() + 1 &&
            host.compare(host.size() - base.size() - 1, base.size() + 1,
                         "." + base) == 0);
  }
  return false;
}

// Заголовок авторизации HuggingFace — во временный файл, а не в аргумент:
// -H @файл (curl ≥7.55). Токен виден в /proc/<pid>/cmdline другим
// пользователям машины — туда ему дорогу закрываем.
struct HfHeader {
  std::string path; // пусто — токена нет или хост не HF

  explicit HfHeader(const std::string& url) {
    if (!hf_url(url))
      return;
    const char* tok = getenv("HF_TOKEN");
    if (!tok || !*tok || strchr(tok, '"') || strchr(tok, '\n')) return;
    path = make_private_tmp("hfhdr");
    if (path.empty()) return;
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { path.clear(); return; }
    std::string line = "Authorization: Bearer " + std::string(tok) + "\n";
    fwrite(line.data(), 1, line.size(), f);
    fclose(f);
  }
  // Аргументы curl: -H @файл — два элемента argv, квотинг не нужен.
  void extend(std::vector<std::string>& args) const {
    if (!path.empty()) { args.push_back("-H"); args.push_back("@" + path); }
  }
  ~HfHeader() {
    if (!path.empty()) fs::remove(path);
  }
};

// Общий прогон curl для мелких запросов: ждёт выхода, возвращает true при коде 0.
bool run_curl(const std::vector<std::string>& args, proc::Child& child, std::string& err) {
  if (!child.start(kCurl, args)) {
    err = std::string("curl не запустился: ") + strerror(errno);
    return false;
  }
  int code = 0;
  while (!child.exited(code)) std::this_thread::sleep_for(std::chrono::milliseconds(150));
  if (code != 0) {
    std::string line = proc::last_meaningful_line(child.log_tail());
    err = "curl: код " + std::to_string(code) + (line.empty() ? "" : " — " + line);
    return false;
  }
  return true;
}

} // namespace

// SHA256 системным sha256sum как ребёнком: нуль зависимостей, hex — первое
// слово его вывода. Пусто — не посчитали (fail-closed: вызывающий отвергнет).
// Наружу (download.h) — нужна pull'у для перепроверки уже лежащих на диске
// файлов; внутренний потребитель — проверка .part перед rename в dest.
std::string sha256_file(const std::string& path) {
  proc::Child child;
  if (!child.start("/usr/bin/sha256sum", {path})) return {};
  int code = 0;
  while (!child.exited(code)) std::this_thread::sleep_for(std::chrono::milliseconds(50));
  if (code != 0) return {};
  std::string tail = child.log_tail();
  size_t sp = tail.find(' ');
  if (sp == std::string::npos) return {};
  std::string hex = tail.substr(0, sp);
  if (hex.size() != 64) return {};
  for (char c : hex)
    if (!isxdigit((unsigned char)c)) return {};
  return hex;
}

bool download(const std::string& url, const std::string& dest, const std::string& sha256_hex,
              const ProgressFn& on_progress, std::string& err) {
  std::string part = dest + ".part";
  // curl сам создаёт -o файл с umask-правами и по symlink'у записывает в
  // чужой файл нашими правами: в разделяемом каталоге моделей .part
  // pre-create'им 0600 (без O_EXCL — существующий нужен для докачки -C -),
  // симлинк сносим.
  {
    std::error_code sec;
    if (fs::is_symlink(part, sec)) fs::remove(part, sec);
    int pfd = ::open(part.c_str(), O_WRONLY | O_CREAT, 0600);
    if (pfd >= 0) ::close(pfd);
  }
  if (find_curl().empty()) { err = "curl не найден (/usr/bin/curl)"; return false; }
  HfHeader hf(url);
  // Без --retry: при внутреннем ретрае curl перезапускает передачу с исходного
  // offset, обнуляя докачанное (проверено на 8.7.1 — пила в прогрессе и лишние
  // гигабайты). Ретраим сами: свежий запуск с -C - резюмится с текущего размера.
  std::vector<std::string> args = {
      // =https — только https: и первый запрос, и все редиректы (-L), иначе
      // 302 на чужой хост увёл бы Bearer-токен HF.
      "-sS", "-L", "--fail", "--proto", "=https", "--connect-timeout", "20",
      "--max-time", "14400",
      "--speed-limit", "10240", "--speed-time", "60",
      "-C", "-", "-o", part};
  hf.extend(args);
  args.push_back(url);

  uint64_t prev_size = file_size(part);
  int code = 0;
  std::string line;
  for (int attempt = 1; attempt <= 5; ++attempt) {
    proc::Child child;
    if (!child.start(kCurl, args)) {
      err = std::string("curl не запустился: ") + strerror(errno);
      return false;
    }

    // Прогресс по размеру .part: curl сам пишет в файл; total знает вызывающий.
    double speed = 0;
    uint64_t done = prev_size, mark_bytes = prev_size;
    auto mark_t = std::chrono::steady_clock::now();
    bool cancelled = false;
    while (child.alive()) {
      uint64_t now = file_size(part);
      if (now != done) {
        auto t = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t - mark_t).count();
        if (dt >= 0.5) {
          double inst = dt > 0 ? (now - mark_bytes) / dt : 0;
          speed = speed ? speed * 0.6 + inst * 0.4 : inst;
          mark_bytes = now;
          mark_t = t;
        }
        done = now;
        if (on_progress && !on_progress(Progress{done, 0, speed})) {
          cancelled = true;
          break;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    if (cancelled) {
      child.stop();
      err = "остановлено";
      return false;
    }
    if (!child.exited(code)) child.stop(); // страховка
    if (code == 0) break;
    line = proc::last_meaningful_line(child.log_tail());

    uint64_t now_size = file_size(part);
    logf("download: попытка %d: curl код %d, part %llu → %llu байт", attempt, code,
         (unsigned long long)prev_size, (unsigned long long)now_size);
    // Дальше пробовать нет смысла: за попытку не сдвинулось ни на байт
    // (обрыв до первых данных — 404/сеть) или попытки кончились.
    if (attempt == 5 || now_size <= prev_size) {
      err = "curl: код " + std::to_string(code) + (line.empty() ? "" : " — " + line);
      return false;
    }
    prev_size = now_size;
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  }

  if (!sha256_hex.empty()) {
    std::string got = sha256_file(part);
    if (got != sha256_hex) {
      fs::remove(part);
      err = "файл повреждён: SHA256 " + got + ", ожидался " + sha256_hex;
      return false;
    }
  }
  std::error_code ec;
  fs::rename(part, dest, ec);
  if (ec) {
    err = "переименование не удалось: " + ec.message();
    return false;
  }
  return true;
}

// Мелкие запросы: тело — в файл, ошибка HTTP — код выхода curl (с --fail) и строка в stderr.
static bool curl_to_memory(const std::string& url, const std::string& body, std::string& out,
                           std::string& err) {
  std::string out_path = make_private_tmp("http");
  std::string in_path;
  if (out_path.empty()) { err = "временный файл не создался"; return false; }
  HfHeader hf(url);
  std::vector<std::string> args = {"-sS", "-L", "--fail", "--proto", "=https",
                                   "--connect-timeout", "20", "--max-time", "60",
                                   "-o", out_path};
  if (!body.empty()) {
    in_path = make_private_tmp("req");
    if (in_path.empty()) {
      err = "временный файл не создался";
      fs::remove(out_path); // tmp уже создан — не бросать труп в /tmp
      return false;
    }
    FILE* f = fopen(in_path.c_str(), "wb");
    if (!f) {
      err = "временный файл не открылся";
      fs::remove(out_path);
      fs::remove(in_path);
      return false;
    }
    fwrite(body.data(), 1, body.size(), f);
    fclose(f);
    args.push_back("-H");
    args.push_back("Content-Type: application/json");
    args.push_back("--data-binary");
    args.push_back("@" + in_path);
  }
  hf.extend(args);
  args.push_back(url);

  proc::Child child;
  bool ok = run_curl(args, child, err);
  if (!in_path.empty()) fs::remove(in_path);
  if (!ok) {
    fs::remove(out_path);
    return false;
  }
  FILE* f = fopen(out_path.c_str(), "rb");
  fs::remove(out_path);
  if (!f) { err = "ответ curl не читается"; return false; }
  std::string raw;
  char buf[8192];
  size_t got;
  while ((got = fread(buf, 1, sizeof(buf), f)) > 0) raw.append(buf, got);
  fclose(f);
  // Скобки явные (<< и так сильнее relational — но читать без них нельзя).
  // Ограничение постфактум: curl уже записал ответ в tmp; само тело малых
  // API-запросов capped и --max-time (60 с).
  if (raw.size() > (16u << 20)) { err = "ответ слишком большой"; return false; }
  out = raw;
  return true;
}

bool get_string(const std::string& url, std::string& out, std::string& err) {
  return curl_to_memory(url, "", out, err);
}

bool post_json(const std::string& url, const std::string& body, std::string& out, std::string& err) {
  return curl_to_memory(url, body, out, err);
}

bool get_partial(const std::string& url, size_t max_bytes, std::string& out, std::string& err) {
  if (!max_bytes) { err = "нулевой лимит выборки"; return false; } // иначе 0-1 ниже уйдёт в underflow
  if (find_curl().empty()) { err = "curl не найден (/usr/bin/curl)"; return false; }
  std::string out_path = make_private_tmp("part");
  if (out_path.empty()) { err = "временный файл не создался"; return false; }
  HfHeader hf(url);
  std::vector<std::string> args = {
      "-sS", "-L", "--fail", "--proto", "=https", "--connect-timeout", "20",
      "--max-time", "60",
      "-r", "0-" + std::to_string(max_bytes - 1),
      // Страховка: если Range проигнорирован и сервер несёт весь файл — оборвать.
      "--max-filesize", std::to_string(max_bytes),
      "-o", out_path};
  hf.extend(args);
  args.push_back(url);

  proc::Child child;
  if (!child.start(kCurl, args)) {
    err = std::string("curl не запустился: ") + strerror(errno);
    return false;
  }
  int code = 0;
  while (!child.exited(code)) std::this_thread::sleep_for(std::chrono::milliseconds(150));

  // 0 — Range отработал; 63 — сервер прислал больше лимита, curl оборвал,
  // но начало уже в файле. Остальные коды — нечего разбирать.
  std::string raw;
  if (code == 0 || code == 63) {
    FILE* f = fopen(out_path.c_str(), "rb");
    if (f) {
      std::vector<char> buf(65536);
      size_t got;
      while (raw.size() < max_bytes && (got = fread(buf.data(), 1, buf.size(), f)) > 0)
        raw.append(buf.data(), got);
      fclose(f);
    }
  }
  fs::remove(out_path);
  if (raw.size() < 64) {
    std::string line = proc::last_meaningful_line(child.log_tail());
    err = "Range-выборка не удалась: curl: код " + std::to_string(code) +
          (line.empty() ? "" : " — " + line);
    return false;
  }
  out = std::move(raw);
  return true;
}

} // namespace lh::dl
