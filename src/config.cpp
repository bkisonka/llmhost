#include "config.h"

#include <nlohmann/json.hpp>
#include "i18n.h"
#include "util.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <set>

namespace lh::config {

namespace fs = std::filesystem;

std::string conf_path() { return app_dir() + "/llmhost.conf"; }
std::string state_path() { return app_dir() + "/state.json"; }
std::string token_path() { return app_dir() + "/token.txt"; }

std::string default_models_dir() { return app_dir() + "/models"; }

std::string resolve_path(const std::string& stored) {
  if (stored.size() < 2) return stored;                     // пустое/одиночный символ
  if (stored[0] == '$') return stored;                      // $переменные
  if (stored[0] == '/') return stored;                      // абсолютный
  return exe_dir() + "/" + stored;                          // относительный — от папки exe
}

std::string store_path(const std::string& absolute) {
  std::string base = exe_dir();
  if (absolute.rfind(base + "/", 0) == 0)
    return absolute.substr(base.size() + 1); // внутри папки программы — храним относительно
  return absolute;
}

std::string models_dir_resolved(const Config& c) {
  return c.models_dir.empty() ? default_models_dir() : resolve_path(c.models_dir);
}

bool is_local_bind(const std::string& bind) {
  return bind == "127.0.0.1" || bind == "localhost" || bind == "::1" ||
         bind == "[::1]";
}

// --- мелкие парсеры значений ------------------------------------------------

static std::string trim(const std::string& s) {
  size_t b = s.find_first_not_of(" \t\r");
  if (b == std::string::npos) return {};
  size_t e = s.find_last_not_of(" \t\r");
  return s.substr(b, e - b + 1);
}

// Целое целиком (со знаком): «12» да, «12x», «», «1 2» — нет.
static bool whole_int(const std::string& s, long long& out) {
  if (s.empty()) return false;
  size_t i = (s[0] == '-' || s[0] == '+') ? 1 : 0;
  if (i >= s.size()) return false;
  if (s.size() - i > 18) return false; // длиннее — atoll насыщается до LLONG_MAX,
  // и «валидное» значение уезжает в atoi-потребителей обрезком
  for (size_t j = i; j < s.size(); ++j)
    if (!isdigit((unsigned char)s[j])) return false;
  out = std::atoll(s.c_str());
  return true;
}

// Логическое в широкой записи (писатель пишет true/false, человек может всякое).
static bool parse_bool(const std::string& s, bool& out) {
  static const char* yes[] = {"true", "1", "yes", "on"};
  static const char* no[] = {"false", "0", "no", "off"};
  for (const char* y : yes) if (s == y) { out = true; return true; }
  for (const char* n : no) if (s == n) { out = false; return true; }
  return false;
}

// Печатные символы без управляющих: значения едут в argv и в журнал.
static bool printable(const std::string& s) {
  for (unsigned char c : s)
    if (iscntrl(c)) return false;
  return true;
}

static std::string bool_str(bool v) { return v ? "true" : "false"; }

static std::string join_cors(const std::vector<std::string>& v) {
  std::string out;
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) out += ",";
    out += v[i];
  }
  return out;
}

static std::vector<std::string> split_cors(const std::string& s) {
  std::vector<std::string> out;
  size_t pos = 0;
  while (pos <= s.size()) {
    size_t comma = s.find(',', pos);
    std::string part =
        trim(s.substr(pos, comma == std::string::npos ? std::string::npos
                                                      : comma - pos));
    if (!part.empty()) out.push_back(part);
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  return out;
}

// --- канонический список ключей [general] -----------------------------------
// Порядок = порядок в файле при создании и в выдаче /v0/settings.

std::vector<std::pair<std::string, std::string>> settings_table(const Config& c) {
  return {
    {"openai-port", std::to_string(c.openai_port)},
    {"control-port", std::to_string(c.control_port)},
    {"openai-bind", c.openai_bind},
    {"openai-auth", bool_str(c.openai_auth)},
    {"cors-origins", join_cors(c.cors_origins)},
    {"models-max", std::to_string(c.models_max)},
    {"parallel", std::to_string(c.parallel)},
    {"moe-cache-mib", std::to_string(c.moe_cache_mib)},
    {"moe-cpu-share",
     [](double v) {
       char b[32];
       std::snprintf(b, sizeof(b), "%.2f", v);
       return std::string(b);
     }(c.moe_cpu_share)},
    {"queue-ttl", std::to_string(c.queue_ttl)},
    {"queue-depth", std::to_string(c.queue_depth)},
    {"slot-park", bool_str(c.slot_park)},
    {"slot-park-min-tokens", std::to_string(c.slot_park_min_tokens)},
    {"slot-park-gib", std::to_string(c.slot_park_gib)},
    {"unload-after-min", std::to_string(c.unload_after_min)},
    {"auto-params", bool_str(c.auto_params)},
    {"ctx", std::to_string(c.ctx)},
    {"ngl", std::to_string(c.ngl)},
    {"ubatch", std::to_string(c.ubatch)},
    {"threads", std::to_string(c.threads)},
    {"gpu-pin", bool_str(c.gpu_pin)},
    {"auto-draft", bool_str(c.auto_draft)},
    {"mtp-mode", c.mtp_mode},
    {"draft-accept-floor",
     [](double v) {
       char b[32];
       std::snprintf(b, sizeof(b), "%.2f", v);
       return std::string(b);
     }(c.draft_accept_floor)},
    {"draft-accept-min-tokens", std::to_string(c.draft_accept_min_tokens)},
    {"models-dir", store_path(c.models_dir)},
    {"llama-server", store_path(c.llama_server)},
    {"extra-args", c.extra_args},
    {"engine-build", c.engine_build},
    {"engine-release", c.engine_release},
    {"image-engine", store_path(c.image_engine)},
    {"image-models-dir", store_path(c.image_models_dir)},
    {"image-gpu", std::to_string(c.image_gpu)},
    {"image-model", c.image_model},
    {"image-idle-min", std::to_string(c.image_idle_min)},
    {"image-extra-args", c.image_extra_args},
    {"lang", c.lang},
  };
}

bool is_general_key(const std::string& key); // ниже, используется validate_conf

bool needs_daemon_restart(const std::string& key) {
  return key == "openai-port" || key == "control-port" || key == "openai-bind" ||
         key == "openai-auth" || key == "cors-origins";
}

bool check_setting(const std::string& key, const std::string& value,
                   const std::vector<std::string>& allowed_builds,
                   std::string& norm, std::string& err) {
  long long n = 0;
  bool b = false;
  if (key == "openai-port" || key == "control-port") {
    if (!whole_int(value, n) || n < 1 || n > 65535) {
      err = "целое 1…65535, а не «" + value + "»";
      return false;
    }
    norm = std::to_string(n);
    return true;
  }
  if (key == "unload-after-min") {
    if (!whole_int(value, n) || n < 0) { err = "целое ≥ 0"; return false; }
    norm = std::to_string(n);
    return true;
  }
  if (key == "ctx") {
    // Верх равный /v0/models/ctx (2097152): без потолка atoll-насыщенное
    // значение доезжало до atoi в set_field (UB, обрезка) и в argv ребёнка.
    if (!whole_int(value, n) || n < 256 || n > 2097152) {
      err = "целое в пределах 256..2097152";
      return false;
    }
    norm = std::to_string(n);
    return true;
  }
  if (key == "ngl") {
    if (!whole_int(value, n) || n < 0) { err = "целое ≥ 0"; return false; }
    norm = std::to_string(n);
    return true;
  }
  if (key == "ubatch") {
    if (!whole_int(value, n) || n != 0 && n < 32) { err = "0 (дефолт движка) или ≥ 32"; return false; }
    norm = std::to_string(n);
    return true;
  }
  if (key == "threads") {
    // CPU-потоки ребёнка: 0 — дефолт движка (hardware_concurrency), иначе
    // явное -t; на 8-ядерном Ryzen физические ядра бьют SMT-16 (замер 31.10).
    if (!whole_int(value, n) || n != 0 && (n < 1 || n > 256)) { err = "0 (дефолт движка) или 1…256"; return false; }
    norm = std::to_string(n);
    return true;
  }
  if (key == "models-max") {
    if (!whole_int(value, n) || n < 0 || n > 16) { err = "целое 0…16"; return false; }
    norm = std::to_string(n);
    return true;
  }
  if (key == "parallel") {
    if (!whole_int(value, n) || n < 1 || n > 8) { err = "целое 1…8"; return false; }
    norm = std::to_string(n);
    return true;
  }
  if (key == "moe-cache-mib") {
    // Кэш экспертов MoE: -1 — авто (план строит кандидатов и выбирает по
    // прогнозу резидентности), 0 — выкл, N — вручную (МиБ; физику ограничит
    // бюджет карты).
    if (!whole_int(value, n) || n < -1 || n > 65536) { err = "целое -1…65536 (-1 — авто, 0 — выключить, N МиБ)"; return false; }
    norm = std::to_string(n);
    return true;
  }
  if (key == "moe-cpu-share") {
    // Доля промахов кэша MoE на параллельном CPU (PR #30966); сетка учитывает
    // её в модели промахов. Верх 0.9 — при 1.0 модель промахов слепеет к
    // PCIe-загрузке и завышает кэш.
    size_t end = 0;
    double d;
    try {
      d = std::stod(value, &end);
    } catch (...) {
      err = "дробь 0…0.9, а не «" + value + "»";
      return false;
    }
    if (end != value.size() || d < 0.0 || d > 0.9) {
      err = "дробь 0…0.9, а не «" + value + "»";
      return false;
    }
    char b[32];
    std::snprintf(b, sizeof(b), "%.2f", d);
    norm = b;
    return true;
  }
  if (key == "queue-ttl") {
    // 0 — парковки нет: занято → отказ сразу. Верхний потолок — чтобы клиент
    // не висел сутками; заголовочный таймаут прокси и так 300 с.
    if (!whole_int(value, n) || n < 0 || n > 3600) { err = "целое 0…3600 (секунды, 0 — выключить)"; return false; }
    norm = std::to_string(n);
    return true;
  }
  if (key == "queue-depth") {
    if (!whole_int(value, n) || n < 1 || n > 256) { err = "целое 1…256"; return false; }
    norm = std::to_string(n);
    return true;
  }
  if (key == "slot-park-min-tokens") {
    // Ниже порога возврат с диска дороже перепрефилла (ср. замер dio 30.10).
    if (!whole_int(value, n) || n < 0 || n > 2097152) {
      err = "целое 0…2097152 (0 — паркуем любые слоты)";
      return false;
    }
    norm = std::to_string(n);
    return true;
  }
  if (key == "slot-park-gib") {
    if (!whole_int(value, n) || n < 0 || n > 4096) {
      err = "целое 0…4096 ГиБ (0 — без предела)";
      return false;
    }
    norm = std::to_string(n);
    return true;
  }
  if (key == "draft-accept-floor") {
    // Дробь 0…1; 0 — сторож выключен.
    size_t end = 0;
    double d;
    try {
      d = std::stod(value, &end);
    } catch (...) {
      err = "дробь 0…1, а не «" + value + "»";
      return false;
    }
    if (end != value.size() || d < 0.0 || d > 1.0) {
      err = "дробь 0…1, а не «" + value + "»";
      return false;
    }
    char b[32];
    std::snprintf(b, sizeof(b), "%.2f", d);
    norm = b;
    return true;
  }
  if (key == "draft-accept-min-tokens") {
    if (!whole_int(value, n) || n < 100 || n > 1000000) {
      err = "целое 100…1000000";
      return false;
    }
    norm = std::to_string(n);
    return true;
  }
  if (key == "openai-auth" || key == "auto-params" || key == "gpu-pin" ||
      key == "auto-draft" || key == "slot-park") {
    if (!parse_bool(value, b)) { err = "true/false, а не «" + value + "»"; return false; }
    norm = bool_str(b);
    return true;
  }
  if (key == "mtp-mode") {
    if (value != "auto" && value != "on" && value != "off") {
      err = "auto|on|off, а не «" + value + "»";
      return false;
    }
    norm = value;
    return true;
  }
  if (key == "lang") {
    if (value != "auto" && value != "ru" && value != "en") {
      err = "auto|ru|en, а не «" + value + "»";
      return false;
    }
    norm = value;
    return true;
  }
  if (key == "engine-build") {
    // Строгий набор: значение едет в build_dir() = путь каталога движка —
    // «..», «/» и прочее туда попадать не должны.
    for (unsigned char c : value)
      if (!isalnum(c) && c != '_' && c != '-') {
        err = "имя сборки — [a-z0-9_-] («cuda12»/«vulkan»)";
        return false;
      }
    if (!value.empty() && !allowed_builds.empty() &&
        std::find(allowed_builds.begin(), allowed_builds.end(), value) ==
            allowed_builds.end()) {
      err = "в manifest.json нет сборки «" + value + "»";
      return false;
    }
    norm = value;
    return true;
  }
  if (key == "engine-release") {
    if (!value.empty()) {
      bool ok = value.size() >= 2 && value[0] == 'b';
      for (size_t i = 1; ok && i < value.size(); ++i)
        ok = isdigit((unsigned char)value[i]);
      if (!ok) { err = "тег вида «b11429» или пусто"; return false; }
    }
    norm = value;
    return true;
  }
  if (key == "image-gpu") {
    // Индекс карты image-движка: едет в CUDA_VISIBLE_DEVICES ребёнка — мусор
    // в путь допускать нельзя. Верх — с запасом под существующие хосты.
    if (!whole_int(value, n) || n < 0 || n > 15) { err = "целое 0…15 (индекс карты)"; return false; }
    norm = std::to_string(n);
    return true;
  }
  if (key == "image-idle-min") {
    // 0 — image-ребёнок живёт до явного /v0/image/unload.
    if (!whole_int(value, n) || n < 0 || n > 10080) { err = "целое 0…10080 (минуты, 0 — никогда)"; return false; }
    norm = std::to_string(n);
    return true;
  }
  if (key == "openai-bind" || key == "models-dir" || key == "llama-server" ||
      key == "extra-args" || key == "cors-origins" || key == "image-engine" ||
      key == "image-models-dir" || key == "image-model" || key == "image-extra-args") {
    if (!printable(value)) { err = "управляющие символы в значении"; return false; }
    if (key == "openai-bind" && value.empty()) { err = "адрес привязки не может быть пуст"; return false; }
    if (key == "cors-origins") {
      for (auto& o : split_cors(value))
        if (o.find_first_of(" \t") != std::string::npos) {
          err = "источник «" + o + "» — без пробелов";
          return false;
        }
    }
    norm = value;
    return true;
  }
  err = "не знаю ключ «" + key + "»";
  return false;
}

// Известные ключи [general] — канонический список ровно из settings_table
// (раньше здесь лежала рукописная копия: новый ключ требовал правки в двух
// местах, и рассинхрон ловился только дрейф-гвардом config_test).
bool is_general_key(const std::string& key) {
  static const std::set<std::string> kKnown = [] {
    std::set<std::string> s;
    for (auto& [k, v] : settings_table(Config{})) s.insert(k);
    return s;
  }();
  return kKnown.count(key) > 0;
}

bool set_field(Config& c, const std::string& key, const std::string& norm) {
  if (key == "openai-port") { c.openai_port = std::atoi(norm.c_str()); return true; }
  if (key == "control-port") { c.control_port = std::atoi(norm.c_str()); return true; }
  if (key == "unload-after-min") { c.unload_after_min = std::atoi(norm.c_str()); return true; }
  if (key == "ctx") { c.ctx = std::atoi(norm.c_str()); return true; }
  if (key == "ngl") { c.ngl = std::atoi(norm.c_str()); return true; }
  if (key == "ubatch") { c.ubatch = std::atoi(norm.c_str()); return true; }
  if (key == "threads") { c.threads = std::atoi(norm.c_str()); return true; }
  if (key == "models-max") { c.models_max = std::atoi(norm.c_str()); return true; }
  if (key == "parallel") { c.parallel = std::atoi(norm.c_str()); return true; }
  if (key == "moe-cache-mib") { c.moe_cache_mib = std::atoi(norm.c_str()); return true; }
  if (key == "moe-cpu-share") { c.moe_cpu_share = std::atof(norm.c_str()); return true; }
  if (key == "openai-auth") { c.openai_auth = norm == "true"; return true; }
  if (key == "queue-ttl") { c.queue_ttl = std::atoi(norm.c_str()); return true; }
  if (key == "queue-depth") { c.queue_depth = std::atoi(norm.c_str()); return true; }
  if (key == "slot-park") { c.slot_park = norm == "true"; return true; }
  if (key == "slot-park-min-tokens") { c.slot_park_min_tokens = std::atoi(norm.c_str()); return true; }
  if (key == "slot-park-gib") { c.slot_park_gib = std::atoi(norm.c_str()); return true; }
  if (key == "auto-params") { c.auto_params = norm == "true"; return true; }
  if (key == "gpu-pin") { c.gpu_pin = norm == "true"; return true; }
  if (key == "auto-draft") { c.auto_draft = norm == "true"; return true; }
  if (key == "mtp-mode") { c.mtp_mode = norm; return true; }
  if (key == "lang") { c.lang = norm; return true; }
  if (key == "draft-accept-floor") { c.draft_accept_floor = std::atof(norm.c_str()); return true; }
  if (key == "draft-accept-min-tokens") { c.draft_accept_min_tokens = std::atoi(norm.c_str()); return true; }
  if (key == "openai-bind") { c.openai_bind = norm; return true; }
  if (key == "models-dir") { c.models_dir = norm; return true; }
  if (key == "llama-server") { c.llama_server = norm; return true; }
  if (key == "extra-args") { c.extra_args = norm; return true; }
  if (key == "engine-build") { c.engine_build = norm; return true; }
  if (key == "engine-release") { c.engine_release = norm; return true; }
  if (key == "image-engine") { c.image_engine = norm; return true; }
  if (key == "image-models-dir") { c.image_models_dir = norm; return true; }
  if (key == "image-gpu") { c.image_gpu = std::atoi(norm.c_str()); return true; }
  if (key == "image-model") { c.image_model = norm; return true; }
  if (key == "image-idle-min") { c.image_idle_min = std::atoi(norm.c_str()); return true; }
  if (key == "image-extra-args") { c.image_extra_args = norm; return true; }
  if (key == "cors-origins") { c.cors_origins = split_cors(norm); return true; }
  return false;
}

// --- разбор/запись INI -------------------------------------------------------

// Строка внутри [general]: «ключ = значение»? Выдаёт ключ (может быть пусто).
static std::string key_of(const std::string& line, size_t eq) {
  return trim(line.substr(0, eq));
}

Config load_from(const std::string& path) {
  Config c;
  std::string raw = read_small_file(path);
  if (raw.empty()) return c;
  bool in_general = false;
  lh::for_each_ini_line(raw, [&](const lh::IniLine& l) {
    if (l.section) {
      in_general = l.closed && trim(l.name) == "general";
      return;
    }
    if (!l.has_eq) {
      logf("config: %s:%d: не «ключ = значение» — пропускаю", path.c_str(), l.lineno);
      return;
    }
    if (!in_general) return; // per-model секции — не наши
    std::string norm, err;
    if (!check_setting(l.key, l.value, {}, norm, err)) {
      if (err.find("не знаю") == std::string::npos)
        logf("config: %s:%d: %s — беру умолчание", path.c_str(), l.lineno, err.c_str());
      return; // неизвестный ключ — молча (за будущими версиями), битый — warn+дефолт
    }
    set_field(c, l.key, norm);
  });
  return c;
}

std::mutex& conf_writers_mtx() {
  static std::mutex m;
  return m;
}

void save_to(const std::string& path, const Config& c) {
  // Один писатель за раз: контроль API, вотчер, bench и пер-модельные секции
  // (sup_overrides) пишут один файл — под общим conf_writers_mtx (config.h).
  std::lock_guard l(conf_writers_mtx());
  auto want = settings_table(c);

  std::string raw = read_small_file(path);
  std::vector<std::string> lines;
  if (!raw.empty()) {
    size_t pos = 0;
    while (pos <= raw.size()) {
      size_t eol = raw.find('\n', pos);
      lines.push_back(raw.substr(pos, eol == std::string::npos ? std::string::npos
                                                               : eol - pos));
      if (eol == std::string::npos) break;
      pos = eol + 1;
    }
    while (!lines.empty() && lines.back().empty()) lines.pop_back(); // хвостовые \n
  }

  // Границы [general]: gi — заголовок, gend — следующая секция/конец.
  int gi = -1, gend = (int)lines.size();
  for (int i = 0; i < (int)lines.size(); ++i) {
    std::string t = trim(lines[i]);
    if (t.size() >= 2 && t[0] == '[') {
      if (gi < 0) {
        size_t close = t.find(']');
        if (close != std::string::npos && trim(t.substr(1, close - 1)) == "general")
          gi = i;
      } else {
        gend = i;
        break;
      }
    }
  }

  if (gi >= 0) {
    // Обновить существующие ключи, собрать недостающие.
    std::vector<std::string> missing;
    for (auto& [k, v] : want) {
      bool found = false;
      for (int i = gi + 1; i < gend && !found; ++i) {
        size_t eq = lines[i].find('=');
        if (eq == std::string::npos) continue;
        if (key_of(lines[i], eq) == k) {
          lines[i] = k + " = " + v;
          found = true;
        }
      }
      if (!found) missing.push_back(k + " = " + v);
    }
    if (!missing.empty())
      lines.insert(lines.begin() + gi + 1, missing.begin(), missing.end());
  } else {
    // Секции нет — создаём блоком в начале файла (перед возможными
    // комментариями/секциями).
    std::vector<std::string> head;
    if (raw.empty())
      head = {"# Настройки llmhost. Правится вручную или через lhctl set;",
              "# [general] применяется демоном на лету (порты — после рестарта),",
              "# остальные секции — per-model переопределения."};
    head.push_back("[general]");
    for (auto& [k, v] : want) head.push_back(k + " = " + v);
    head.push_back("");
    lines.insert(lines.begin(), head.begin(), head.end());
  }

  std::string out;
  for (auto& l : lines) out += l + "\n";
  // Уникальный tmp (идиом profiles.cpp): имя «конф.tmp» — предсказуемая
  // мишень симлинка и место, где осколки чужой прерванной записи
  // перетираются следующим писателем.
  std::string tmp = path + ".tmp." + lh::random_hex(6);
  std::error_code ec;
  if (!lh::write_small_file(tmp, out) || (fs::rename(tmp, path, ec), ec)) {
    logf("config: не записался %s", path.c_str());
    fs::remove(tmp, ec);
  }
}

// --- state.json (рантайм-состояние, НЕ настройки) ----------------------------

static State load_state_from(const std::string& path) {
  State s;
  std::string raw = lh::read_small_file(path);
  if (raw.empty()) return s;
  try {
    auto j = nlohmann::json::parse(raw);
    s.last_model = j.value("last_model", "");
    s.mtp_choices = j.value("mtp_choices", std::map<std::string, bool>{});
    s.resident = j.value("resident", std::vector<std::string>{});
  } catch (const std::exception& e) {
    logf("state: %s не читается — работаем с пустым (%s)", path.c_str(), e.what());
  }
  return s;
}

static void save_state_to(const std::string& path, const State& s) {
  std::string out = nlohmann::json{
      {"last_model", s.last_model},
      {"mtp_choices", s.mtp_choices},
      {"resident", s.resident},
  }.dump(2) + "\n";
  // Писатели из разных потоков (note_model_used у прокси/супервизора и
  // mtp::after_load): без замка они писали бы один .tmp вперемешку — битый
  // файл уехал бы в rename. Мьютекс один на процесс.
  static std::mutex g_state_write_mtx;
  std::lock_guard l(g_state_write_mtx);
  // Атомарно, как conf: обрыв питания не должен терять last_model.
  std::string tmp = path + ".tmp";
  std::error_code ec;
  if (lh::write_small_file(tmp, out) && (fs::rename(tmp, path, ec), !ec)) return;
  if (!lh::write_small_file(path, out))
    logf("state: не записался %s", path.c_str());
}

State load_state() { return load_state_from(state_path()); }
void save_state(const State& s) {
  ensure_dir(app_dir());
  save_state_to(state_path(), s);
}

// --- миграция config.json → llmhost.conf + state.json ------------------------

void migrate_legacy() { migrate_legacy_at(conf_path(), app_dir() + "/config.json", state_path()); }

bool migrate_legacy_at(const std::string& conf_file, const std::string& json_file,
                       const std::string& state_file) {
  std::string raw = lh::read_small_file(json_file);
  if (raw.empty()) return false; // нет legacy — нечего делать
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(raw);
    if (!j.is_object()) throw std::runtime_error("не объект");
  } catch (const std::exception& e) {
    logf("config: config.json не читается (%s) — переезда нет, файл оставлен как есть",
         e.what());
    return false;
  }

  // Старые поля (формат config.json) → Config.
  Config old;
  old.openai_port = j.value("openai_port", old.openai_port);
  old.control_port = j.value("control_port", old.control_port);
  old.openai_bind = j.value("openai_bind", old.openai_bind);
  old.unload_after_min = j.value("unload_after_min", old.unload_after_min);
  old.ctx = j.value("ctx", old.ctx);
  old.ngl = j.value("ngl", old.ngl);
  old.openai_auth = j.value("openai_auth", old.openai_auth);
  old.auto_params = j.value("auto_params", old.auto_params);
  old.models_max = j.value("models_max", old.models_max);
  old.parallel = j.value("parallel", old.parallel);
  old.gpu_pin = j.value("gpu_pin", old.gpu_pin);
  old.auto_draft = j.value("auto_draft", old.auto_draft);
  old.mtp_mode = j.value("mtp_mode", old.mtp_mode);
  old.ubatch = j.value("ubatch", old.ubatch);
  old.threads = j.value("threads", old.threads);
  old.models_dir = j.value("models_dir", "");
  old.llama_server = j.value("llama_server", "");
  old.engine_build = j.value("engine_build", "");
  old.engine_release = j.value("engine_release", "");
  old.cors_origins = j.value("cors_origins", std::vector<std::string>{});

  // conf уже с [general]? Ключи, уже записанные там (инсталлером или юзером),
  // не перетираем — переносим только недостающее.
  Config merged = load_from(conf_file);
  std::vector<std::string> present;
  {
    std::string conf_raw = lh::read_small_file(conf_file);
    bool in_general = false;
    lh::for_each_ini_line(conf_raw, [&](const lh::IniLine& l) {
      if (l.section) {
        // «[» без «]» — не заголовок секции: «[general» без скобки выдавал
        // себя за general (npos-1 съедал остаток строки).
        in_general = l.closed && trim(l.name) == "general";
        return;
      }
      if (!in_general || !l.has_eq) return;
      present.push_back(l.key);
    });
  }
  auto has = [&](const std::string& k) {
    return std::find(present.begin(), present.end(), k) != present.end();
  };
  for (auto& [k, v] : settings_table(old))
    if (!has(k)) {
      std::string norm, err;
      if (check_setting(k, v, {}, norm, err)) set_field(merged, k, norm);
    }
  ensure_dir(app_dir());
  save_to(conf_file, merged);

  // Рантайм-состояние — в state.json (не настройки).
  if (!lh::file_exists(state_file)) {
    State st;
    st.last_model = j.value("last_model", "");
    st.mtp_choices = j.value("mtp_choices", std::map<std::string, bool>{});
    save_state_to(state_file, st);
  }

  // Legacy убираем с дороги (как model_overrides.json при его переезде):
  // иначе он «вечный» и врёт, что ещё источник.
  std::error_code ec;
  fs::rename(json_file, json_file + ".migrated", ec);
  logf("config: МИГРАЦИЯ config.json → llmhost.conf%s",
       ec ? " (legacy-файл не переименован)" : " + state.json");
  return true;
}

// --- валидация (гейт перед рискованными записями, например пином движка) ----

// Известные ключи per-model секций: то, что знает set_override_field.
// Неизвестные ключи секций не наша забота (за будущими версиями).
static std::string check_override_value(const std::string& key, const std::string& value) {
  long long n = 0;
  auto digits_or = [&](const char* extra) {
    for (unsigned char c : value)
      if (!isdigit(c) && !strchr(extra, c)) return false;
    return true;
  };
  if (key == "ctx-size" || key == "n-gpu-layers" || key == "main-gpu")
    return whole_int(value, n) ? std::string() : key + ": целое, а не «" + value + "»";
  if (key == "tensor-split")
    return digits_or(",.") ? std::string() : key + ": цифры и запятые, а не «" + value + "»";
  if (key == "mtp-mode") {
    std::string low = value;
    for (char& ch : low)
      if (ch >= 'A' && ch <= 'Z') ch = (char)(ch - 'A' + 'a');
    if (low != "auto" && low != "on" && low != "off")
      return "mtp-mode: значение — auto, on или off, а не «" + value + "»";
    return "";
  }
  if (key == "cache-type") {
    if (value == "f16" || value == "q8_0" || value == "q4_0" || value.empty())
      return {};
    return key + ": f16|q8_0|q4_0, а не «" + value + "»";
  }
  if (key == "spec-type") {
    for (unsigned char c : value)
      if (!isalnum(c) && c != '_' && c != '-')
        return key + ": латиница/цифры/_/-, а не «" + value + "»";
    return {};
  }
  if (key == "model-draft" || key == "system")
    return printable(value) ? std::string() : key + ": управляющие символы";
  return {}; // неизвестный ключ секции — не наша забота
}

std::vector<std::string> validate_conf(const std::vector<std::string>& allowed_builds) {
  return validate_conf_at(conf_path(), allowed_builds);
}

std::vector<std::string> validate_conf_at(
    const std::string& path, const std::vector<std::string>& allowed_builds) {
  std::vector<std::string> errs;
  std::string raw = read_small_file(path);
  if (raw.empty()) return errs; // нет файла — всё умолчания, валидно
  std::string section;
  bool in_section = false;
  lh::for_each_ini_line(raw, [&](const lh::IniLine& l) {
    if (l.section) {
      if (!l.closed || l.name.empty()) {
        errs.push_back("строка " + std::to_string(l.lineno) + ": секция без имени/скобки");
        return;
      }
      section = trim(l.name);
      in_section = true;
      return;
    }
    if (!l.has_eq) {
      errs.push_back("строка " + std::to_string(l.lineno) + ": не «ключ = значение»");
      return;
    }
    if (!in_section) {
      errs.push_back("строка " + std::to_string(l.lineno) + ": ключ вне секции");
      return;
    }
    if (section == "general") {
      if (!is_general_key(l.key)) return; // неизвестное — за будущими версиями
      std::string norm, err;
      if (!check_setting(l.key, l.value, allowed_builds, norm, err))
        errs.push_back("строка " + std::to_string(l.lineno) + ": " + l.key + ": " + err);
    } else {
      std::string err = check_override_value(l.key, l.value);
      if (!err.empty()) errs.push_back("строка " + std::to_string(l.lineno) + ": " + err);
    }
  });
  return errs;
}

// --- приложение -------------------------------------------------------------

Config load() { return load_from(conf_path()); }

void save(const Config& c) {
  ensure_dir(app_dir());
  save_to(conf_path(), c);
}

namespace {

std::mutex g_cfg_mtx;
Config g_cfg;

} // namespace

void set_current(const Config& c) {
  { std::lock_guard l(g_cfg_mtx); g_cfg = c; }
  // Язык интерфейса — на лету: старт демона и hot-правка lang вотчером
  // проходят через эту единственную точку применения хинта.
  lh::i18n::set_lang_hint(c.lang.c_str());
}
Config current() { std::lock_guard l(g_cfg_mtx); return g_cfg; }

// Один статик на процесс (раньше proxy и control держали по своему — на
// первом запуске гонка давала два разных токена и вечный 401).
std::string client_token() {
  static const std::string t = [] {
    ensure_dir(app_dir());
    std::string cur = read_small_file(token_path());
    if (cur.size() >= 32) {
      // Токен мог быть создан до сужения прав — подтягиваем при каждом старте.
      restrict_file_permissions(token_path());
      return cur;
    }
    std::string fresh = random_hex(32);
    if (fresh.size() < 32) {
      logf("token: генератор недоступен (getrandom) — API закрыт (fail-closed)");
      return fresh;
    }
    // Сразу 0600 на создании: ofstream-вариант оставлял окно с правами по
    // umask (0644) между записью и chmod'ом.
    int fd = ::open(token_path().c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    bool written = false;
    if (fd >= 0) {
      ssize_t put = ::write(fd, fresh.data(), fresh.size());
      written = put == (ssize_t)fresh.size();
      ::close(fd);
    }
    if (!written)
      logf("token: не записался токен — авторизация действует до перезапуска");
    else if (!restrict_file_permissions(token_path()))
      logf("token: права файла не сужены до 0600");
    return fresh;
  }();
  return t;
}

// Сравнение за постоянное время: длина не секрет (фиксированный префикс и
// hex-токен), а побайтовый XOR не даёт таймингового оракула по содержимому.
bool secure_equal(const std::string& a, const std::string& b) {
  if (a.size() != b.size()) return false;
  const volatile unsigned char* x = reinterpret_cast<const volatile unsigned char*>(a.data());
  const volatile unsigned char* y = reinterpret_cast<const volatile unsigned char*>(b.data());
  unsigned char acc = 0;
  for (size_t i = 0; i < a.size(); ++i) acc |= x[i] ^ y[i];
  return acc == 0;
}

BearerVerdict check_bearer(const std::string& auth_header) {
  const std::string t = client_token();
  if (t.empty()) return BearerVerdict::NoToken;
  return secure_equal("Bearer " + t, auth_header) ? BearerVerdict::Ok
                                                  : BearerVerdict::Wrong;
}

} // namespace lh::config
