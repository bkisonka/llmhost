#include "tokens.h"

#include "util.h"

#include <ctime>
#include <filesystem>
#include <map>
#include <mutex>

namespace lh::tokens {
namespace {

namespace fs = std::filesystem;

struct Entry {
  std::string token, created_at, last_used;
};

std::mutex g_mtx;
std::map<std::string, Entry> g_map;  // имя → токен; строк до сотен не бывает
std::string g_path;
std::chrono::steady_clock::time_point g_last_flush;

// Сравнение за константное время — токены ходят по сети, как и общий ключ
// (config::secure_equal), только здесь кандидатов несколько — сканируем все.
bool ct_equal(const std::string& a, const std::string& b) {
  volatile unsigned char diff =
      (unsigned char)((a.size() ^ b.size()) & 0xFF);
  size_t n = std::max(a.size(), b.size());
  for (size_t i = 0; i < n; ++i) {
    unsigned char x = i < a.size() ? (unsigned char)a[i] : 0;
    unsigned char y = i < b.size() ? (unsigned char)b[i] : 0;
    diff |= x ^ y;
  }
  return diff == 0;
}

std::string now_iso() {
  char buf[32];
  std::time_t t = std::time(nullptr);
  std::tm tm{};
#ifdef _WIN32
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
  return buf;
}

void load_locked() {
  g_map.clear();
  std::string raw = lh::read_small_file(g_path);
  if (raw.empty()) return;
  try {
    auto j = nlohmann::json::parse(raw);
    for (auto& t : j.value("tokens", nlohmann::json::array())) {
      std::string name = t.value("name", "");
      if (name.empty() || t.value("token", "").empty()) continue;
      g_map[name] = {t.value("token", ""), t.value("created_at", ""),
                     t.value("last_used", "")};
    }
  } catch (...) {
    lh::logf("tokens: %s не читается — начинаю с пустого словаря", g_path.c_str());
  }
}

void save_locked() {
  nlohmann::json arr = nlohmann::json::array();
  for (auto& [name, e] : g_map)
    arr.push_back({{"name", name},
                   {"token", e.token},
                   {"created_at", e.created_at},
                   {"last_used", e.last_used}});
  std::string out = nlohmann::json{{"schema", 1}, {"tokens", arr}}.dump(2);
  // Атомарно, как config::save: tmp + rename, сбой rename — прямая запись.
  // Файл содержит значения токенов в открытом виде — tmp создаётся сразу с
  // 0600 (write_private_file): прежний «umask-файл, потом chmod» оставлял
  // окно чтения локальному пользователю, а предсказуемое имя .tmp было
  // мишенью симлинка; в fallback путь chmod до записи делал no-op над
  // несуществующим файлом и api_tokens.json мог остаться 0644.
  std::string tmp = g_path + ".tmp." + lh::random_hex(6);
  std::error_code ec;
  if (!lh::write_private_file(tmp, out)) {
    lh::logf("tokens: не записать %s", tmp.c_str());
    fs::remove(tmp, ec);
    return;
  }
  fs::rename(tmp, g_path, ec);
  if (ec) {
    ec.clear();
    if (!lh::write_private_file(g_path, out))
      lh::logf("tokens: не записать %s", g_path.c_str());
  }
}

}  // namespace

void init(const std::string& json_path) {
  std::lock_guard l(g_mtx);
  g_path = json_path;
  load_locked();
  lh::logf("tokens: словарь открыт: %s (%zu шт.)", g_path.c_str(), g_map.size());
}

bool add(const std::string& name, std::string& token, std::string& err) {
  if (name.empty() || name.size() > 64 ||
      name.find_first_not_of(
          "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_. ") !=
          std::string::npos) {
    err = "имя токена: 1–64 символа, буквы/цифры/-_. ";
    return false;
  }
  std::lock_guard l(g_mtx);
  if (g_map.count(name)) {
    err = "имя токена «" + name + "» уже занято";
    return false;
  }
  token = "lh-" + lh::random_hex(24);
  g_map[name] = {token, now_iso(), ""};
  save_locked();
  lh::logf("tokens: создан токен «%s»", name.c_str());
  return true;
}

bool remove(const std::string& name, std::string& err) {
  std::lock_guard l(g_mtx);
  if (!g_map.erase(name)) {
    err = "токен «" + name + "» не найден";
    return false;
  }
  save_locked();
  lh::logf("tokens: отозван токен «%s»", name.c_str());
  return true;
}

nlohmann::json list() {
  std::lock_guard l(g_mtx);
  nlohmann::json arr = nlohmann::json::array();
  for (auto& [name, e] : g_map)
    arr.push_back({{"name", name},
                   {"token", e.token},
                   {"created_at", e.created_at},
                   {"last_used", e.last_used}});
  return arr;
}

int count() {
  std::lock_guard l(g_mtx);
  return (int)g_map.size();
}

bool verify(const std::string& token) {
  if (token.empty()) return false;
  std::lock_guard l(g_mtx);
  for (auto& [name, e] : g_map) {
    if (ct_equal(e.token, token)) {
      e.last_used = now_iso();
      // last_used сбрасываем на диск не чаще раза в минуту: /v1 дёргает
      // verify на каждом запросе, а запись на каждый чат жалко.
      if (std::chrono::steady_clock::now() - g_last_flush > std::chrono::seconds(60)) {
        g_last_flush = std::chrono::steady_clock::now();
        save_locked();
      }
      return true;
    }
  }
  return false;
}

}  // namespace lh::tokens
