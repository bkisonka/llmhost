// Пер-модельные переопределения: структура, таблица ключей, парсер INI и
// чистый редактор секции. Header-only, без зависимостей от супервизора —
// юнит-тест (tests/overrides_test.cpp) гоняет грамматику в отрыве от демона;
// сам файл переопределений (llmhost.conf) правят писатели в sup_overrides.cpp
// под общими замками. Header-only — юнит на парсер/писатель и
// дрейф-гвард ключей гоняется в отрыве от демона.
#pragma once
#include "util.h" // for_each_ini_line — единственная INI-грамматика проекта

#include <cctype>
#include <map>
#include <string>

namespace lh::sup::detail {

// id попадает в INI-заголовок секции: имя файла на NTFS может содержать «]»,
// «[» и даже переводы строк — ломали бы пресет (инъекция секций). Безопасный
// ASCII, остальное — в «_».
inline std::string sanitize_id(const std::string& base) {
  std::string id;
  id.reserve(base.size());
  for (char c : base)
    id += (isalnum((unsigned char)c) || c == '.' || c == '-' || c == '_') ? c : '_';
  return id.empty() ? "model" : id;
}

// Пер-модельные переопределения — наш аналог Modelfile: llmhost.conf (INI,
// обычная Linux-практика), секция на модель:
//   [Qwen3-0.6B-Q8_0]
//   ctx-size = 16384
//   n-gpu-layers = 40
//   system = Ты — ассистент бухгалтерии.
// Ключ секции — id (имя файла без .gguf). ctx-size — общий на модель
// (делится между слотами parallel). Переопределения сильнее умной настройки.
struct ModelOverrides {
  int ctx_size = 0;   // 0 — как рассчитала умная настройка
  int ngl = 0;        // 0 — как рассчитано
  std::string system; // пусто — без инъекции
  int main_gpu = -1;  // привязка к карте (мульти-GPU); -1 — авто
  std::string tensor_split; // пропорции "0,1"; пусто — авто (все веса main-gpu)
  std::string model_draft;  // draft-модель (id/имя/путь); пусто — без спекуляций
  std::string spec_type;    // draft-simple (дефолт); eagle3/ngram-* — что движок знает
  std::string cache_type;   // f16/q8_0/q4_0; пусто — решает умная настройка
  std::string mtp_mode;     // auto/on/off; пусто — глобальный mtp-mode из [general]
  std::string vision_mode;  // auto/off; пусто — auto: свою проекцию прицеплять
  std::string think_mode;   // auto/on/off; пусто — auto: движок сам детектит
                            // thinking по шаблону модели
                            // (урок 29.10: «первая попавшаяся mmproj*» — не родство);
                            // off — ребёнок едет без --mmproj, что бы ни лежало рядом
  std::string draft_card;   // «spare» — черновик на запасную карту (--device-draft);
                            // пусто/main — карта ствола, как раньше
  std::string expert_card;  // «spare» — доля экспертов MoE на запасной карте
                            // (дополнительное -ot-правило перед exps=CPU); пусто — нет
  int moe_cache_mib = 0;    // N — кэш этой модели вручную (МиБ; оптимум у моделей
                            // разный: Flash-Next 8192, Next-80B 10752 — замер 30.10);
                            // 0 — выключить кэш для модели (статическая раскладка;
                            // на 262k-контексте статика была вдвое быстрее кэша);
                            // -1 — убрать решение, решает [general]
  bool moe_cache_set = false; // ключ был в секции (INI): отличает явный 0 («выкл»)
                              // от отсутствия ключа («решает [general]»); не
                              // персистится — восстанавливается парсером
  bool precision_first = false; // фиделити-контракт агента: KV f16
                                // (если cache-type не задан явно),
                                // thinking on, нота в карточке, удвоенное окно
                                // простоя. --no-context-shift ставится ВСЕМ
                                // чат-детям (см. sup_spawn) — флаг его
                                // не отличает
};

// Таблица ключей секции — дрейф-гвард: парсер обязан знать ровно её; писатели
// (set_model_*), миграция legacy-JSON и юнит-тест берут имена отсюда же.
// Ключ вне таблицы будущим версиям не оставлен сознательно: незнакомый ключ
// в секции молча пропускается парсером, но таблица обязана его называть —
// иначе дрейф (писатель пишет то, что читатель не поймёт).
constexpr const char* kOverrideKeys[] = {
    "ctx-size",  "n-gpu-layers", "main-gpu", "moe-cache-mib", "system",
    "model-draft", "tensor-split", "mtp-mode", "vision-mode", "thinking-mode",
    "precision-first", "draft-card", "expert-card", "spec-type", "cache-type"};

// Одна точка присвоения поля переопределения — общий фильтр для INI и JSON:
// значения едут в командную строку ребёнка, поэтому у каждого поля свой
// допуск. Возврат false — ключ не знает (вызывающий молча пропускает:
// незнакомые ключи оставляем будущим версиям).
inline bool set_override_field(ModelOverrides& o, const std::string& key,
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
    // N — кэш вручную (МиБ); 0 — выключить кэш для модели (статическая
    // раскладка); -1 — убрать решение (решает [general]). Явный 0 от
    // отсутствия ключа отличает moe_cache_set.
    int v = 0;
    if (!whole_int(value, v) || v < -1 || v > 65536) return false;
    o.moe_cache_mib = v;
    o.moe_cache_set = v >= 0;
    return true;
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
  if (key == "precision-first") {
    // Фиделити-контракт. Включение —
    // пер-модельное осознанное решение владельца; авто-профили и умная
    // настройка флаг не ставят никогда.
    std::string clean;
    for (char c : value)
      if (isalnum((unsigned char)c)) clean += (char)tolower((unsigned char)c);
    if (clean == "on" || clean == "true" || clean == "1") {
      o.precision_first = true;
      return true;
    }
    if (clean == "off" || clean == "false" || clean == "0") {
      o.precision_first = false;
      return true;
    }
    return false;
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
inline std::map<std::string, ModelOverrides> parse_overrides_ini(
    const std::string& raw) {
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

// Чистый редактор секции (без замоков, файлов и apply — их делает писатель в
// sup_overrides.cpp): raw → новый текст, где в секции section стоит
// «key = value»; value пуст — ключ удалён («умолчание»). Чужие секции,
// комментарии и порядок строк сохраняются. Секции нет — дописывается в конец.
inline std::string edit_override_section(const std::string& raw,
                                         const std::string& section,
                                         const std::string& key,
                                         const std::string& value) {
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
  size_t sec = std::string::npos, key_line = std::string::npos,
         sec_last_key = std::string::npos;
  for (size_t i = 0; i < lines.size(); ++i) {
    const std::string t = trim(lines[i]);
    if (!t.empty() && t[0] == '[') {
      if (sec != std::string::npos) break; // конец целевой секции
      size_t close = t.find(']');
      if (close != std::string::npos && close >= 2 &&
          sanitize_id(t.substr(1, close - 1)) == section)
        sec = i;
      continue;
    }
    if (sec == std::string::npos) continue;
    // Ключ секции: не пусто, не комментарий, есть «=». Комментарии и пустые
    // строки остаются в хвосте секции — новый ключ встаёт после последнего
    // ключа, а не сразу после заголовка.
    const bool comment = !t.empty() && (t[0] == '#' || t[0] == ';');
    const size_t eq = t.find('=');
    if (!t.empty() && !comment && eq != std::string::npos) {
      sec_last_key = i;
      if (key_line == std::string::npos && trim(t.substr(0, eq)) == key)
        key_line = i;
    }
  }

  if (!value.empty()) {
    const std::string line = key + " = " + value;
    if (key_line != std::string::npos)
      lines[key_line] = line;
    else if (sec != std::string::npos)
      lines.insert(lines.begin() + (sec_last_key == std::string::npos
                                        ? sec + 1
                                        : sec_last_key + 1),
                   line);
    else if (raw.empty())
      lines = {"[" + section + "]", line, ""};
    else {
      if (!lines.back().empty()) lines.push_back("");
      lines.push_back("[" + section + "]");
      lines.push_back(line);
    }
  } else if (key_line == std::string::npos) {
    return raw; // убирать нечего — уже умолчание
  } else {
    lines.erase(lines.begin() + key_line);
  }

  std::string out;
  for (size_t i = 0; i < lines.size(); ++i) {
    if (i) out += '\n';
    out += lines[i];
  }
  return out;
}

} // namespace lh::sup::detail
