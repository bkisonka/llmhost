#include "sup_internal.h"

#include "util.h"

#include <nlohmann/json.hpp>

#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>

namespace fs = std::filesystem;

namespace lh::sup::detail {

// Per-model переопределения — наш аналог Modelfile: llmhost.conf (INI, обычная
// Linux-практика), секция на модель:
//   [Qwen3-0.6B-Q8_0]
//   ctx-size = 16384
//   n-gpu-layers = 40
//   system = Ты — ассистент бухгалтерии.
// Ключ секции — id роутера (имя файла без .gguf). ctx-size — общий на модель
// (делится между слотами parallel). Переопределения сильнее умной настройки;
// файл читается при каждой перегенерации пресета, правки подхватывает вотчер
// (размер и mtime входят в отпечаток реестра). «system» прокси подкладывает
// в запросы клиентов, не приславших свой system. Legacy: model_overrides.json
// читается, только если llmhost.conf нет (переезд без правки старых установок).
std::string overrides_conf_path() { return lh::app_dir() + "/llmhost.conf"; }
std::string overrides_path() { return lh::app_dir() + "/model_overrides.json"; }

int kv_force_of(const ModelOverrides* o, const std::string& hint) {
  if (!o || o->cache_type.empty()) return 0;
  if (o->cache_type == "f16") return 1;
  if (o->cache_type == "q8_0") return 2;
  if (o->cache_type == "q4_0") return 3;
  logf("load: %s: cache-type «%s» не знаю (f16/q8_0/q4_0) — решает умная настройка",
       hint.c_str(), o->cache_type.c_str());
  return 0;
}

// Одна точка присвоения поля переопределения — общий фильтр для INI и JSON:
// значения едут в командную строку ребёнка, поэтому у каждого поля свой
// допуск. Возврат false — ключ не знает (вызывающий молча пропускает:
// незнакомые ключи оставляем будущим версиям).
bool set_override_field(ModelOverrides& o, const std::string& key,
                        const std::string& value) {
  auto whole_int = [](const std::string& s, int& out) {
    if (s.empty()) return false;
    size_t i = (s[0] == '-' || s[0] == '+') ? 1 : 0;
    if (i >= s.size()) return false;
    if (s.size() - i > 9) return false; // atoi на 10+-значном — UB/обрезка;
    // длиннее int и писать в переопределение некуда
    for (size_t j = i; j < s.size(); ++j)
      if (!isdigit((unsigned char)s[j])) return false;
    out = std::atoi(s.c_str());
    return true;
  };
  if (key == "ctx-size") return whole_int(value, o.ctx_size);
  if (key == "n-gpu-layers") return whole_int(value, o.ngl);
  if (key == "main-gpu") return whole_int(value, o.main_gpu);
  if (key == "moe-cache-mib") {
    // 0 — решает [general]; отрицательное — авто-сетка для этой модели.
    return whole_int(value, o.moe_cache_mib);
  }
  if (key == "system") {
    o.system = value;
    return true;
  }
  if (key == "model-draft") {
    o.model_draft = value;
    return true;
  }
  if (key == "tensor-split") {
    // только цифры, запятые и точки (пропорции)
    std::string clean;
    for (char c : value)
      if (isdigit((unsigned char)c) || c == ',' || c == '.') clean += c;
    o.tensor_split = clean;
    return true;
  }
  if (key == "mtp-mode") {
    std::string clean;
    for (char c : value)
      if (isalnum((unsigned char)c)) clean += (char)tolower((unsigned char)c);
    if (clean != "auto" && clean != "on" && clean != "off") return false;
    o.mtp_mode = clean;
    return true;
  }
  if (key == "vision-mode") {
    // auto — своя проекция прицепляется сама (родство имён, find_mmproj);
    // off — ребёнок едет без --mmproj, что бы ни лежало рядом.
    std::string clean;
    for (char c : value)
      if (isalnum((unsigned char)c)) clean += (char)tolower((unsigned char)c);
    if (clean != "auto" && clean != "off") return false;
    o.vision_mode = clean == "auto" ? std::string() : clean;
    return true;
  }
  if (key == "thinking-mode") {
    // auto — движок сам детектит thinking по шаблону; on/off — форс
    // --reasoning ребёнку (шаблонам с enable_thinking отдаёт kwarg).
    std::string clean;
    for (char c : value)
      if (isalnum((unsigned char)c)) clean += (char)tolower((unsigned char)c);
    if (clean != "auto" && clean != "on" && clean != "off") return false;
    o.think_mode = clean == "auto" ? std::string() : clean;
    return true;
  }
  if (key == "draft-card" || key == "expert-card") {
    // «spare» — груз на запасную карту; «main»/пусто — как раньше (карта
    // ствола / всё на CPU). Незнакомое значение — не знаем, отказ.
    std::string clean;
    for (char c : value)
      if (isalnum((unsigned char)c)) clean += (char)tolower((unsigned char)c);
    if (clean != "spare" && clean != "main") return false;
    if (key == "draft-card") o.draft_card = clean == "main" ? std::string() : clean;
    else o.expert_card = clean == "main" ? std::string() : clean;
    return true;
  }
  if (key == "spec-type" || key == "cache-type") {
    std::string clean;
    for (char c : value)
      if (isalnum((unsigned char)c) || c == '_' || c == '-')
        clean += (char)tolower((unsigned char)c);
    if (key == "spec-type") o.spec_type = clean.empty() ? "draft-simple" : clean;
    else o.cache_type = clean;
    return true;
  }
  return false;
}

// Разбор INI: [секция] + «ключ = значение»; комментарии — строка целиком
// с # или ; (инлайн-комментарии не поддерживаем: «system» может их содержать).
std::map<std::string, ModelOverrides> parse_overrides_ini(const std::string& raw) {
  std::map<std::string, ModelOverrides> out;
  std::string section;
  ModelOverrides cur;
  bool have = false;
  auto flush = [&]() {
    if (have) out[sanitize_id(section)] = cur;
  };
  lh::for_each_ini_line(raw, [&](const lh::IniLine& l) {
    if (l.section) {
      // «[» без «]»/без имени — не заголовок: мусорную строку пропускаем,
      // текущая секция живёт (семантика пер-модельных секций).
      if (!l.closed || l.name.empty()) return;
      flush();
      section = l.name;
      cur = ModelOverrides();
      have = true;
      return;
    }
    if (!l.has_eq || !have) return;
    set_override_field(cur, l.key, l.value);
  });
  flush();
  return out;
}

std::map<std::string, ModelOverrides> load_overrides() {
  std::map<std::string, ModelOverrides> out;
  // Источник — по наличию файла: llmhost.conf главный, JSON — legacy-переезд.
  // exists без error_code бросил бы filesystem_error в потоке прокси/watcher
  // (чужой EACCES/ELOOP на пути к каталогу) — только ec-версия.
  std::error_code ec;
  const bool has_conf = fs::exists(overrides_conf_path(), ec);
  // Гроссбук «уже отлогированного источника»: load_overrides зовут из многих
  // потоков (вотчер, прокси, control) — дедуп без мьютекса был бы гонкой.
  static std::mutex logged_mtx;
  static std::string logged_src;
  auto log_src_once = [&](const std::string& sig, const std::string& msg) {
    std::lock_guard l(logged_mtx);
    if (sig != logged_src) {
      logged_src = sig;
      logf("%s", msg.c_str());
    }
  };
  std::string src_sig;
  if (has_conf) {
    std::string raw = lh::read_small_file(overrides_conf_path());
    out = parse_overrides_ini(raw);
    src_sig = "conf|" + std::to_string(lh::file_size(overrides_conf_path())) + "|" +
              std::to_string((uint64_t)fs::last_write_time(overrides_conf_path(), ec)
                                 .time_since_epoch()
                                 .count());
    log_src_once(src_sig, "router: переопределения из llmhost.conf (" +
                              std::to_string(out.size()) + " секций)");
    return out;
  }
  std::string raw = lh::read_small_file(overrides_path());
  if (raw.empty()) return out;
  try {
    auto j = nlohmann::json::parse(raw);
    if (!j.is_object()) return out;
    for (auto& [key, v] : j.items()) {
      if (!v.is_object()) continue;
      ModelOverrides o;
      for (auto& [k, val] : v.items()) {
        std::string s;
        if (val.is_string()) s = val.get<std::string>();
        else if (val.is_number_integer()) s = std::to_string(val.get<long long>());
        else if (val.is_boolean()) s = val.get<bool>() ? "true" : "false";
        else continue; // вложенное — не наше
        set_override_field(o, k, s);
      }
      out[sanitize_id(key)] = o;
    }
    // Переезд завершается сам: conf нет, а json с содержимым — переносим
    // один раз в llmhost.conf и убираем legacy-файл с дороги. Иначе любой
    // создатель пустого conf (вчера — enable_mtp) молча хоронит все
    // переопределения legacy (поймано живьём: override qwen36 из рабочего
    // инстанса пропал на сутки, пока json считался «переезжевшим»).
    if (!out.empty()) {
      static std::once_flag migrate_once;
      std::call_once(migrate_once, [&] {
        std::string ini;
        int fields = 0;
        for (auto& [id, o] : out) {
          ini += "[" + id + "]\n";
          auto add = [&](const char* key, const std::string& v) {
            // INI читается построчно — многострочные значения не переносятся.
            if (v.empty() || v.find('\n') != std::string::npos) return;
            ini += std::string(key) + " = " + v + "\n";
            ++fields;
          };
          if (o.ctx_size > 0) add("ctx-size", std::to_string(o.ctx_size));
          if (o.ngl > 0) add("n-gpu-layers", std::to_string(o.ngl));
          if (o.main_gpu >= 0) add("main-gpu", std::to_string(o.main_gpu));
          if (o.moe_cache_mib != 0)
            add("moe-cache-mib", std::to_string(o.moe_cache_mib));
          add("tensor-split", o.tensor_split);
          add("model-draft", o.model_draft);
          add("spec-type", o.spec_type);
          add("cache-type", o.cache_type);
          add("system", o.system);
        }
        std::lock_guard conf_l(config::conf_writers_mtx()); // тот же файл,
        // что правят save_to и секции — единый локер писателей conf (config.h)
        const std::string tmp = overrides_conf_path() + ".tmp." + lh::random_hex(6);
        std::error_code ec2;
        if (!lh::write_small_file(tmp, ini) || !lh::write_small_file(
                overrides_path() + ".migrated", lh::read_small_file(overrides_path()))) {
          logf("router: миграция json → llmhost.conf не удалась — legacy остаётся "
               "источником");
          return;
        }
        fs::rename(tmp, overrides_conf_path(), ec2);
        if (ec2) {
          logf("router: миграция json → llmhost.conf не удалась: %s", ec2.message().c_str());
          return;
        }
        fs::rename(overrides_path(), overrides_path() + ".migrated", ec2);
        logf("router: МИГРАЦИЯ model_overrides.json → llmhost.conf: %zu секций, "
             "%d полей%s", out.size(), fields,
             ec2 ? " (legacy-файл не переименован)" : "");
      });
    }
    src_sig = "json|" + std::to_string(lh::file_size(overrides_path())) + "|" +
              std::to_string((uint64_t)fs::last_write_time(overrides_path(), ec)
                                 .time_since_epoch()
                                 .count());
    log_src_once(src_sig,
                 "router: переопределения из model_overrides.json (legacy) — " +
                     std::to_string(out.size()) +
                     " секций; новое место — llmhost.conf");
  } catch (const std::exception& e) {
    logf("router: model_overrides.json не читается — работаем без переопределений (%s)",
         e.what());
  }
  return out;
}

// Секция переопределений модели — sanitize_id от стема файла: те же ключи,
// что в пресете и в разборе INI.
std::string override_section_of(const std::string& model_path) {
  std::string base = model_path;
  size_t s = base.find_last_of("/\\");
  if (s != std::string::npos) base = base.substr(s + 1);
  if (base.size() > 5 && strcasecmp(base.c_str() + base.size() - 5, ".gguf") == 0)
    base.erase(base.size() - 5);
  return sanitize_id(base);
}

// Эффективный mtp-mode: переопределение сильнее глобального [general].
std::string mtp_mode_of(const ModelOverrides* o, const config::Config& cfg) {
  if (o && !o->mtp_mode.empty()) return o->mtp_mode;
  return cfg.mtp_mode;
}

// Секция переопределений по модели. У шард-группы id — групповое имя
// («Qwen3.8-Flash-Next-UD-Q4_K_XL»), а секции в конфигах живут и по стему
// первой части («…-00001-of-00004» — так пишет lhctl ctx и пишут руки),
// и по групповому имени. Ищем сперва id, затем стем файла — иначе секция
// из конфига молча перестаёт действовать (ctx-size падал с 65536 до 8192).
const ModelOverrides* overrides_for(const std::map<std::string, ModelOverrides>& ov,
                                    const std::string& id,
                                    const std::string& model_path) {
  auto it = ov.find(id);
  if (it != ov.end()) return &it->second;
  const std::string sec = override_section_of(model_path);
  if (sec != id) {
    it = ov.find(sec);
    if (it != ov.end()) return &it->second;
  }
  return nullptr;
}

} // namespace lh::sup::detail

namespace lh::sup {

// ---- запись пер-модельных переопределений (переехало из supervisor.cpp; ----
// ---- чтение живёт выше, в detail — один файл на обе стороны секции)      ----

// Запись/удаление одного строкового ключа в секции переопределений модели.
// value пуст — строка удаляется («умолчание»). После записи — немедленный
// apply (пресет и дети перезапускаются плавно, не дожидаясь вотчера) и,
// в harness, увод свободного слота в сон: новый ключ применит спавн.
static bool set_model_override_kv(const std::string& model_path, const std::string& key,
                                  const std::string& value, std::string& err) {
  std::lock_guard apply_l(detail::g_apply_mtx); // запись + тут же apply под одним замком
  // Единый локер read-modify-write llmhost.conf (config.h): без него эта
  // правка секции могла перемешаться с save_to [general] или enable_mtp —
  // у всех свой (или никакой) замок и один общий tmp. Порядок: g_apply →
  // conf_writers.
  std::lock_guard conf_l(config::conf_writers_mtx());
  const std::string section = detail::override_section_of(model_path);
  const std::string conf = detail::overrides_conf_path();
  const std::string raw = lh::read_small_file(conf);

  auto trim = [](std::string s) {
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r'))
      s.pop_back();
    size_t b = s.find_first_not_of(" \t");
    return b == std::string::npos ? std::string() : s.substr(b);
  };
  std::vector<std::string> lines;
  {
    size_t pos = 0;
    while (pos <= raw.size()) {
      size_t eol = raw.find('\n', pos);
      lines.push_back(raw.substr(pos, eol == std::string::npos
                                           ? std::string::npos
                                           : eol - pos));
      if (eol == std::string::npos) break;
      pos = eol + 1;
    }
  }
  size_t sec = std::string::npos, key_line = std::string::npos;
  for (size_t i = 0; i < lines.size(); ++i) {
    const std::string t = trim(lines[i]);
    if (!t.empty() && t[0] == '[') {
      if (sec != std::string::npos) break;
      size_t close = t.find(']');
      if (close != std::string::npos && close >= 2 &&
          detail::sanitize_id(t.substr(1, close - 1)) == section)
        sec = i;
      continue;
    }
    if (sec != std::string::npos && key_line == std::string::npos) {
      size_t eq = t.find('=');
      if (eq != std::string::npos && trim(t.substr(0, eq)) == key) key_line = i;
    }
  }

  if (!value.empty()) {
    const std::string line = key + " = " + value;
    if (key_line != std::string::npos)
      lines[key_line] = line;
    else if (sec != std::string::npos)
      lines.insert(lines.begin() + sec + 1, line);
    else if (raw.empty())
      lines = {"[" + section + "]", line, ""};
    else {
      if (!lines.back().empty()) lines.push_back("");
      lines.push_back("[" + section + "]");
      lines.push_back(line);
    }
  } else if (key_line == std::string::npos) {
    return true; // убирать нечего — уже умолчание
  } else {
    lines.erase(lines.begin() + key_line);
  }

  std::string out;
  for (size_t i = 0; i < lines.size(); ++i) {
    if (i) out += '\n';
    out += lines[i];
  }
  const std::string tmp = conf + ".tmp." + lh::random_hex(6); // предсказуемый
  // «.tmp» — мишень симлинка и место обрывков чужой записи (идиом profiles.cpp)
  std::error_code ec;
  if (!lh::write_small_file(tmp, out)) {
    err = "не удалось записать " + tmp;
    fs::remove(tmp, ec);
    return false;
  }
  fs::rename(tmp, conf, ec);
  if (ec) {
    err = "не удалось заменить " + conf + ": " + ec.message();
    fs::remove(tmp, ec);
    return false;
  }
  logf("supervisor: %s: %s = %s (%s)", section.c_str(), key.c_str(),
       value.empty() ? "умолчание" : value.c_str(), conf.c_str());
  apply_registry_change_locked();
  {
    std::string sid;
    {
      std::lock_guard l(detail::g_mtx);
      for (auto& [i, s] : detail::g_slots)
        if (detail::path_eq(s.path, model_path)) { sid = i; break; }
    }
    if (!sid.empty()) {
      bool busy;
      {
        std::lock_guard lb(detail::g_busy_mtx);
        busy = detail::g_busy_set.count(sid) > 0;
      }
      if (busy)
        logf("harness: «%s» занята — %s применится при следующей загрузке",
             sid.c_str(), key.c_str());
      else {
        logf("harness: «%s» в сон — поднимется с новым %s по первому запросу",
             sid.c_str(), key.c_str());
        detail::stop_hslot(sid, true);
      }
    }
  }
  return true;
}

bool set_model_mtp(const std::string& model_path, const std::string& mode,
                   std::string& err) {
  // auto — умолчание: строку из секции убираем, чтобы конфиг не рос зря.
  return set_model_override_kv(model_path, "mtp-mode",
                               mode == "auto" ? std::string() : mode, err);
}

bool set_model_vision(const std::string& model_path, const std::string& mode,
                      std::string& err) {
  // auto — умолчание: строку из секции убираем, чтобы конфиг не рос зря.
  return set_model_override_kv(model_path, "vision-mode",
                               mode == "auto" ? std::string() : mode, err);
}

bool set_model_thinking(const std::string& model_path, const std::string& mode,
                        std::string& err) {
  // auto — умолчание: строку из секции убираем, чтобы конфиг не рос зря.
  return set_model_override_kv(model_path, "thinking-mode",
                               mode == "auto" ? std::string() : mode, err);
}

bool set_model_ctx(const std::string& model_path, int ctx, std::string& err) {
  // Был отдельный редактор секции — дословная копия set_model_override_kv
  // с другим литералом ключа; осталась только обёртка.
  return set_model_override_kv(model_path, "ctx-size",
                               ctx > 0 ? std::to_string(ctx) : std::string(), err);
}

bool set_model_cache(const std::string& model_path, int mib, std::string& err) {
  // Пер-модельный размер кэша MoE: 0 — снять override ([general] снова
  // решает), N — МиБ вручную. Применится при следующей загрузке модели.
  return set_model_override_kv(model_path, "moe-cache-mib",
                               mib != 0 ? std::to_string(mib) : std::string(),
                               err);
}

int override_ctx_for_path(const std::string& model_path) {
  auto ov = detail::load_overrides();
  auto it = ov.find(detail::override_section_of(model_path));
  return it == ov.end() ? 0 : it->second.ctx_size;
}

std::string override_section_for(const std::string& model_path) {
  return detail::override_section_of(model_path);
}

std::string override_mtp_for_path(const std::string& model_path) {
  auto ov = detail::load_overrides();
  const detail::ModelOverrides* o = detail::overrides_for(ov, std::string(), model_path);
  return o ? o->mtp_mode : std::string();
}

std::string override_vision_for_path(const std::string& model_path) {
  auto ov = detail::load_overrides();
  const detail::ModelOverrides* o = detail::overrides_for(ov, std::string(), model_path);
  return o ? o->vision_mode : std::string();
}

std::string override_think_for_path(const std::string& model_path) {
  auto ov = detail::load_overrides();
  const detail::ModelOverrides* o = detail::overrides_for(ov, std::string(), model_path);
  return o ? o->think_mode : std::string();
}

std::string vision_mmproj_for_path(const std::string& model_path) {
  // Что прицепится к модели сейчас: переопределение (off — ничего) + родство
  // имён из find_mmproj. Карточке и предупреждению ctx — один ответ с load-путём.
  auto ov = detail::load_overrides();
  const detail::ModelOverrides* o = detail::overrides_for(ov, std::string(), model_path);
  return detail::mmproj_for_load(model_path, o, nullptr);
}

bool has_draft_override(const std::string& id) {
  auto ov = detail::load_overrides();
  auto it = ov.find(id);
  return it != ov.end() && !it->second.model_draft.empty();
}

// Включить MTP-черновик модели. Текстуальное слияние в llmhost.conf (чужие
// секции, ключи и комментарии не трогаем), затем регенерация пресета +
// перезапуск роутера — то же, что делает вотчер на 4 с позже, только сразу
// и с перезагрузкой модели, чтобы спекуляция заработала не дожидаясь
// следующего запроса.
bool enable_mtp(const std::string& id, const std::string& draft_stem,
                const std::string& main_path, std::string& err) {
  const std::string conf = detail::overrides_conf_path();
  // Единый локер read-modify-write llmhost.conf (config.h): раньше enable_mtp
  // не брал никакого замка и его RMW перемешивался с save_to и редактором
  // секций через общий tmp. Снимаем до apply_registry_change — тот берёт
  // g_apply_mtx, а порядок вложенности всегда g_apply → conf_writers.
  std::unique_lock conf_l(config::conf_writers_mtx());
  std::vector<std::string> lines;
  {
    std::string raw = lh::read_small_file(conf);
    size_t pos = 0;
    while (pos <= raw.size()) {
      size_t eol = raw.find('\n', pos);
      std::string line =
          raw.substr(pos, eol == std::string::npos ? std::string::npos : eol - pos);
      while (!line.empty() && line.back() == '\r') line.pop_back();
      lines.push_back(line);
      if (eol == std::string::npos) break;
      pos = eol + 1;
    }
    // хвост без перевода строки оставляем как есть — добавим свой с '\n'
    if (!lines.empty() && !raw.empty() && raw.back() != '\n')
      lines.back() += "\x01KEEP"; // маркер «без \n» — при сборке не добавляем
    if (raw.empty()) lines.clear(); // пустой conf — без фантомных пустых строк
  }
  const std::string mtp_header = "[" + id + "]";
  size_t mtp_sec = lines.size();
  for (size_t i = 0; i < lines.size(); ++i)
    if (lines[i] == mtp_header) {
      mtp_sec = i;
      break;
    }
  if (mtp_sec == lines.size()) {
    // секции нет — в конец; маркер «без \n» снимаем, добавляя свой перевод
    if (!lines.empty()) {
      std::string& last = lines.back();
      if (last.size() >= 6 && last.compare(last.size() - 6, 6, "\x01KEEP") == 0) {
        last.erase(last.size() - 6);
        lines.push_back("");
      } else {
        lines.push_back("");
      }
    }
    lines.push_back(mtp_header);
  }
  // тело секции — до следующей строки, начинающейся с '['
  size_t mtp_end = mtp_sec + 1;
  while (mtp_end < lines.size() &&
         (lines[mtp_end].empty() || lines[mtp_end][0] != '['))
    ++mtp_end;
  for (auto& [k, v] :
       std::vector<std::pair<std::string, std::string>>{
           {"model-draft", draft_stem}, {"spec-type", "draft-mtp"}}) {
    const std::string prefix = k + " ";
    bool replaced = false;
    for (size_t i = mtp_sec + 1; i < mtp_end; ++i) {
      if (lines[i].rfind(prefix, 0) == 0 || lines[i].rfind(k + "=", 0) == 0) {
        lines[i] = k + " = " + v;
        replaced = true;
        break;
      }
    }
    if (!replaced)
      lines.insert(lines.begin() + mtp_end, k + " = " + v), ++mtp_end;
  }
  std::string out;
  for (size_t i = 0; i < lines.size(); ++i) {
    std::string l = lines[i];
    // Маркер «нет переводной строки в хвосте» ставится читателем суффиксом
    // ровно в последнюю строку. Искать его по всему файлу было ошибкой:
    // легальная строка значения, содержащая байт 0x01+"KEEP", обрезалась и
    // склеивалась со следующей (порча конфига).
    bool no_nl = i + 1 == lines.size() && l.size() >= 6 &&
                 l.compare(l.size() - 6, 6, "\x01KEEP") == 0;
    if (no_nl) l.erase(l.size() - 6);
    if (i + 1 < lines.size() || !no_nl) out += l + "\n";
    else out += l;
  }
  std::string tmp = conf + ".tmp." + lh::random_hex(6);
  std::error_code ec;
  bool saved = lh::write_small_file(tmp, out);
  if (saved) {
    fs::rename(tmp, conf, ec);
    saved = !ec;
  }
  if (!saved) fs::remove(tmp, ec);
  conf_l.unlock();
  if (!saved) {
    err = "не удалось записать " + conf;
    return false;
  }
  // Ключи прочтёт load_harness при следующей загрузке. Перезагружаем модель
  // сразу, чтобы спекуляция включилась не по первому запросу.
  apply_registry_change();
  if (!load(main_path, err)) {
    err = "модель не перезагрузилась: " + err;
    return false;
  }
  return true;
}

} // namespace lh::sup
