// Конфиг-слой llmhost.conf (INI): roundtrip настроек через [general],
// вливание с сохранением комментариев и per-model секций, миграция
// config.json → conf + state.json, битые значения → warn + дефолт,
// validate_conf/check_setting (гейт пина движка).
// Плюс portable-пути: относительные живут от папки exe.
#include "config.h"
#include "util.h"

#include <cassert>
#include <cstdio>
#include <filesystem>

using namespace lh;
using namespace lh::config;

static std::string tmp(const char* name) {
  return (std::filesystem::temp_directory_path() / name).string();
}

int main() {
  std::string path = tmp("llmhost_config_test.conf");
  std::filesystem::remove(path);

  // Нет файла — умолчания.
  Config d = load_from(path);
  assert(d.openai_port == 7330 && d.control_port == 7331 && d.unload_after_min == 15);
  assert(d.ctx == 4096 && d.ngl == 999 && d.openai_auth);
  assert(d.auto_params);
  assert(d.models_max == 2);
  assert(d.parallel == 1); // слоты выключены по умолчанию; замер «N клиентов» 28.10 — в ROADMAP
  assert(d.moe_cache_mib == 0); // кэш MoE-экспертов выключен по умолчанию (план включит по ключу)
  assert(d.moe_cpu_share == 0.0); // параллельный CPU-счёт промахов выключен (поведение апстрима)
  assert(d.slot_park && d.slot_park_min_tokens == 8192 && d.slot_park_gib == 64); // парковка KV включена
  assert(d.gpu_pin);       // раскладка по картам включена (на 1 карте не видна)
  assert(is_local_bind(d.openai_bind)); // по умолчанию — только localhost

  // Roundtrip со всеми ключами [general], включая кириллицу в путях.
  Config c;
  c.openai_port = 8000;
  c.control_port = 8001;
  c.openai_bind = "0.0.0.0"; // внешняя служба
  c.unload_after_min = 0;
  c.ctx = 8192;
  c.ngl = 33;
  c.openai_auth = false;
  c.models_max = 3;
  c.parallel = 3;
  c.moe_cache_mib = 2048;
  c.moe_cpu_share = 0.6;
  c.slot_park = false;
  c.slot_park_min_tokens = 1024;
  c.slot_park_gib = 8;
  c.gpu_pin = false;
  c.auto_draft = false;
  c.mtp_mode = "off";
  c.lang = "en";
  c.ubatch = 2048;
  c.models_dir = "/data/Модели/Qwen";
  c.llama_server = "/opt/tools/llama-server";
  c.engine_build = "cuda12";
  c.engine_release = "b11429";
  c.cors_origins = {"https://myapp.example", "http://192.168.1.5:8080"};
  save_to(path, c);
  Config r = load_from(path);
  assert(r.openai_port == 8000 && r.control_port == 8001 && r.unload_after_min == 0);
  assert(r.openai_bind == "0.0.0.0" && !is_local_bind(r.openai_bind));
  assert(r.ctx == 8192 && r.ngl == 33 && !r.openai_auth);
  assert(r.models_max == 3 && r.parallel == 3 && r.moe_cache_mib == 2048);
  assert(r.moe_cpu_share == 0.6);
  assert(!r.slot_park && r.slot_park_min_tokens == 1024 && r.slot_park_gib == 8);
  assert(!r.gpu_pin && !r.auto_draft && r.mtp_mode == "off" && r.ubatch == 2048);
  assert(r.lang == "en");
  assert(r.models_dir == c.models_dir && r.llama_server == c.llama_server);
  assert(r.engine_build == "cuda12" && r.engine_release == "b11429");
  assert(r.cors_origins.size() == 2);
  assert(r.cors_origins[0] == "https://myapp.example");
  assert(r.cors_origins[1] == "http://192.168.1.5:8080");

  // Вливание: комментарии и per-model секции сохраняются, недостающие ключи
  // дописываются в [general], существующие обновляются по месту.
  write_small_file(
      path,
      "# мой комментарий\n"
      "; и такой\n"
      "[general]\n"
      "ctx = 4096\n"
      "openai-port = 8000\n"
      "\n"
      "[qwen36]\n"
      "ctx-size = 65536\n"
      "main-gpu = 1\n");
  Config m;
  m.openai_port = 9000;   // есть в файле — обновится
  m.ctx = 16384;          // есть — обновится
  m.models_max = 4;       // нет — допишется
  save_to(path, m);
  std::string raw = read_small_file(path);
  assert(raw.find("# мой комментарий") != std::string::npos);
  assert(raw.find("; и такой") != std::string::npos);
  assert(raw.find("[qwen36]") != std::string::npos);
  assert(raw.find("ctx-size = 65536") != std::string::npos);
  assert(raw.find("main-gpu = 1") != std::string::npos);
  Config mr = load_from(path);
  assert(mr.openai_port == 9000 && mr.ctx == 16384 && mr.models_max == 4);
  // [general] в начале файла, до per-model секций.
  assert(raw.find("[general]") < raw.find("[qwen36]"));

  // Битые значения — по-ключево warn + дефолт, соседние ключи живут.
  write_small_file(path,
                   "[general]\n"
                   "openai-port = abc\n"
                   "ctx = -1\n"
                   "mtp-mode = фигня\n"
                   "lang = фигня\n"
                   "openai-auth = да\n"
                   "control-port = 7500\n");
  Config bad = load_from(path);
  assert(bad.openai_port == 7330 && bad.ctx == 4096 && bad.mtp_mode == "auto");
  assert(bad.lang == "auto");
  assert(bad.openai_auth);       // «да» не знаем — умолчание true
  assert(bad.control_port == 7500);

  // Валидатор: гейт перед пином движка.
  write_small_file(path,
                   "loose-key = 1\n"
                   "\n"
                   "[general]\n"
                   "ctx = abc\n"
                   "engine-release = hello\n"
                   "engine-build = cudaa\n"
                   "\n"
                   "[qwen36]\n"
                   "ctx-size = 65x36\n");
  auto errs = validate_conf_at(path, {"cuda12", "vulkan"});
  bool has_ctx = false, has_rel = false, has_build = false, has_loose = false,
       has_override = false;
  for (auto& e : errs) {
    has_ctx = has_ctx || e.find("ctx:") != std::string::npos;
    has_rel = has_rel || e.find("engine-release") != std::string::npos;
    has_build = has_build || e.find("engine-build") != std::string::npos;
    has_loose = has_loose || e.find("вне секции") != std::string::npos;
    has_override = has_override || e.find("ctx-size") != std::string::npos;
  }
  assert(has_ctx && has_rel && has_build && has_loose && has_override);
  // Чистый conf + неизвестный ключ (за будущими версиями) — валидно.
  write_small_file(path,
                   "[general]\n"
                   "ctx = 8192\n"
                   "engine-release = b11429\n"
                   "engine-build = cuda12\n"
                   "future-key = whatever\n"
                   "\n"
                   "[qwen36]\n"
                   "ctx-size = 65536\n"
                   "cache-type = q8_0\n");
  // Пер-модельный mtp-mode: значения auto/on/off валидны, прочее — ошибка.
  write_small_file(path,
                   "[general]\n"
                   "mtp-mode = auto\n"
                   "\n"
                   "[qwen36]\n"
                   "mtp-mode = off\n");
  assert(validate_conf_at(path, {"cuda12", "vulkan"}).empty());
  write_small_file(path,
                   "[qwen36]\n"
                   "mtp-mode = always\n");
  {
    auto e2 = validate_conf_at(path, {"cuda12", "vulkan"});
    assert(e2.size() == 1 && e2[0].find("mtp-mode") != std::string::npos);
  }

  // check_setting/set_field — писатель /v0/settings и lhctl set.
  std::string norm, err;
  assert(check_setting("ctx", "8192", {}, norm, err) && norm == "8192");
  assert(!check_setting("ctx", "8192x", {}, norm, err));
  assert(check_setting("engine-build", "vulkan", {"cuda12", "vulkan"}, norm, err) &&
         norm == "vulkan");
  assert(!check_setting("engine-build", "cudaa", {"cuda12", "vulkan"}, norm, err));
  assert(check_setting("engine-build", "", {}, norm, err) && norm.empty());
  assert(check_setting("auto-draft", "on", {}, norm, err) && norm == "true");
  assert(check_setting("engine-release", "b99999", {}, norm, err));
  assert(!check_setting("engine-release", "b99x", {}, norm, err));
  assert(check_setting("cors-origins", "https://a.example, https://b.example", {},
                       norm, err));
  assert(check_setting("threads", "8", {}, norm, err) && norm == "8");
  assert(check_setting("threads", "0", {}, norm, err) && norm == "0");
  assert(!check_setting("threads", "300", {}, norm, err));
  assert(!check_setting("threads", "8x", {}, norm, err));
  Config sf;
  assert(set_field(sf, "ctx", "8192") && sf.ctx == 8192);
  assert(set_field(sf, "threads", "8") && sf.threads == 8);
  assert(set_field(sf, "threads", "0") && sf.threads == 0);
  assert(set_field(sf, "auto-draft", "false") && !sf.auto_draft);
  assert(set_field(sf, "cors-origins", "https://a.example,https://b.example") &&
         sf.cors_origins.size() == 2);
  assert(!set_field(sf, "unknown-key", "1"));

  // Дрейф-гвард: каждый ключ таблицы обязан быть известен писателю и
  // валидатору — мусорное значение отклоняется с объяснением, а не «не знаю»
  // (новый ключ settings_table без ветки в check_setting = дыра в валидации).
  for (auto& [k, v] : settings_table(Config{})) {
    (void)v;
    std::string n2, e2;
    assert(!check_setting(k, "\x07", {}, n2, e2));
    assert(e2.find("не знаю") == std::string::npos);
  }

  // Таблица [general]: канонический порядок и store_path для путей.
  Config t;
  t.models_dir = exe_dir() + "/data/models";
  auto tab = settings_table(t);
  bool saw_models_rel = false, saw_ctx = false;
  for (auto& [k, v] : tab) {
    saw_models_rel = saw_models_rel || (k == "models-dir" && v == "data/models");
    saw_ctx = saw_ctx || k == "ctx";
  }
  assert(saw_models_rel && saw_ctx);

  // Пути portable: относительные живут от папки exe, абсолютные и $переменные
  // не трогаются; при сохранении путь внутри папки программы ужимается до
  // относительного (точку монтирования меняют — конфиг переживает).
  std::string exe = exe_dir();
  assert(resolve_path("data/models") == exe + "/data/models");
  assert(resolve_path("/abs/path") == "/abs/path");
  assert(resolve_path("$HOME/x") == "$HOME/x");
  assert(store_path(exe + "/data/models") == "data/models");
  assert(store_path("/elsewhere") == "/elsewhere");
  assert(store_path("data/already") == "data/already");

  // Миграция config.json → [general] conf + state.json; идемпотентность.
  std::string mig_conf = tmp("llmhost_config_test_migrate.conf");
  std::string mig_json = tmp("llmhost_config_test_migrate.json");
  std::string mig_state = tmp("llmhost_config_test_migrate_state.json");
  std::filesystem::remove(mig_conf);
  std::filesystem::remove(mig_json);
  std::filesystem::remove(mig_state);
  write_small_file(mig_json,
                   R"({"openai_port":5050,"openai_bind":"0.0.0.0","ctx":32768,)"
                   R"("engine_build":"cuda12","models_dir":"/data/models",)"
                   R"("last_model":"/data/models/qwen.gguf",)"
                   R"("mtp_choices":{"qwen":false}})");
  // У миграции нет своего состояния — пишем пустышку, чтобы видеть перенос.
  assert(migrate_legacy_at(mig_conf, mig_json, mig_state));
  assert(std::filesystem::exists(mig_json + ".migrated"));
  Config mg = load_from(mig_conf);
  assert(mg.openai_port == 5050 && mg.openai_bind == "0.0.0.0");
  assert(mg.ctx == 32768 && mg.engine_build == "cuda12");
  assert(mg.models_dir == "/data/models");
  std::string st_raw = read_small_file(mig_state);
  assert(st_raw.find("/data/models/qwen.gguf") != std::string::npos);
  assert(st_raw.find("qwen") != std::string::npos);
  // Повторный запуск: json больше нет — ничего не делает.
  assert(!migrate_legacy_at(mig_conf, mig_json, mig_state));
  // Миграция не перетирает ключи, уже записанные в conf (инсталлер/юзер).
  std::filesystem::remove(mig_conf);
  std::filesystem::remove(mig_json + ".migrated");
  write_small_file(mig_conf, "[general]\nctx = 4096\n");
  write_small_file(mig_json, R"({"ctx":999,"openai_port":7000})");
  assert(migrate_legacy_at(mig_conf, mig_json, mig_state));
  Config mg2 = load_from(mig_conf);
  assert(mg2.ctx == 4096);      // в conf уже был — не тронут
  assert(mg2.openai_port == 7000); // в conf не было — перенесён

  std::filesystem::remove(path);
  std::filesystem::remove(mig_conf);
  std::filesystem::remove(mig_json + ".migrated");
  std::filesystem::remove(mig_state);
  std::puts("config_test: ok");
  return 0;
}
