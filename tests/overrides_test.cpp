// Грамматика пер-модельных переопределений (overrides_kv.h, header-only):
// парсер INI, редактор секции, дрейф-гвард ключей, precision-first. Живёт
// в отрыве от демона — sup_internal.h не включается, единственный юнит
// грамматики вне демона.
#include "overrides_kv.h"

#include <cassert>
#include <cstdio>
#include <string>

using lh::sup::detail::edit_override_section;
using lh::sup::detail::kOverrideKeys;
using lh::sup::detail::ModelOverrides;
using lh::sup::detail::parse_overrides_ini;
using lh::sup::detail::sanitize_id;
using lh::sup::detail::set_override_field;

int main() {
  // Дрейф-гвард: парсер знает ровно таблицу — каждый ключ таблицы принимается
  // каноническим значением, у таблицы нет дублей.
  {
    for (size_t i = 0; i < sizeof(kOverrideKeys) / sizeof(kOverrideKeys[0]); ++i) {
      ModelOverrides o;
      // Каждому ключу — значение, которое он обязан принять (поля с допуском:
      // специфика значений — на ключе; «on» принимают не все — vision-mode
      // знает только auto/off, draft/expert-card — только spare/main).
      const char* probe = "ctx-size" == std::string(kOverrideKeys[i])    ? "16384"
                          : "n-gpu-layers" == std::string(kOverrideKeys[i]) ? "40"
                          : "main-gpu" == std::string(kOverrideKeys[i])     ? "1"
                          : "moe-cache-mib" == std::string(kOverrideKeys[i]) ? "8192"
                          : "tensor-split" == std::string(kOverrideKeys[i])  ? "3,1"
                          : "spec-type" == std::string(kOverrideKeys[i])     ? "draft-mtp"
                          : "cache-type" == std::string(kOverrideKeys[i])    ? "f16"
                          : "vision-mode" == std::string(kOverrideKeys[i])   ? "off"
                          : "draft-card" == std::string(kOverrideKeys[i])    ? "spare"
                          : "expert-card" == std::string(kOverrideKeys[i])   ? "spare"
                                                                             : "on";
      assert(set_override_field(o, kOverrideKeys[i], probe));
      // Дублей в таблице нет: каждый ключ встречается ровно один раз.
      for (size_t j = i + 1; j < sizeof(kOverrideKeys) / sizeof(kOverrideKeys[0]); ++j)
        assert(std::string(kOverrideKeys[i]) != std::string(kOverrideKeys[j]));
    }
    // Незнакомый ключ (будущие версии) — пропуск без порчи полей.
    ModelOverrides o;
    o.ctx_size = 777;
    assert(!set_override_field(o, "future-key", "1"));
    assert(o.ctx_size == 777);
  }

  // Парсер: полный набор ключей одной секции.
  {
    const char* ini =
        "# комментарий\n"
        "[Qwen3-0.6B-Q8_0]\n"
        "ctx-size = 16384\n"
        "n-gpu-layers = 40\n"
        "main-gpu = 1\n"
        "moe-cache-mib = 8192\n"
        "system = Ты — ассистент бухгалтерии.\n"
        "model-draft = qwen3-0.6b\n"
        "tensor-split = 3,1\n"
        "mtp-mode = on\n"
        "vision-mode = off\n"
        "thinking-mode = off\n"
        "precision-first = on\n"
        "draft-card = spare\n"
        "expert-card = spare\n"
        "spec-type = draft-mtp\n"
        "cache-type = f16\n"
        "; тоже комментарий\n";
    auto m = parse_overrides_ini(ini);
    assert(m.size() == 1);
    const ModelOverrides& o = m.at("Qwen3-0.6B-Q8_0");
    assert(o.ctx_size == 16384 && o.ngl == 40 && o.main_gpu == 1);
    assert(o.moe_cache_mib == 8192 && o.moe_cache_set);
    assert(o.system == "Ты — ассистент бухгалтерии.");
    assert(o.model_draft == "qwen3-0.6b" && o.tensor_split == "3,1");
    assert(o.mtp_mode == "on" && o.vision_mode == "off" && o.think_mode == "off");
    assert(o.precision_first);
    assert(o.draft_card == "spare" && o.expert_card == "spare");
    assert(o.spec_type == "draft-mtp" && o.cache_type == "f16");
  }

  // precision-first: допуски значения, мусор — отказ.
  {
    for (const char* v : {"on", "true", "1", "ON"}) {
      ModelOverrides o;
      assert(set_override_field(o, "precision-first", v));
      assert(o.precision_first);
    }
    // Грамматика знает ровно on|true|1 и off|false|0 — «yes» не синтаксис.
    for (const char* v : {"off", "false", "0", "OFF", "yes", "Yes"}) {
      ModelOverrides o;
      o.precision_first = true;
      bool accepted = set_override_field(o, "precision-first", v);
      if (accepted) assert(!o.precision_first);
      else assert(o.precision_first); // отказ не портит поле
    }
    ModelOverrides o;
    assert(!set_override_field(o, "precision-first", "maybe"));
    assert(!set_override_field(o, "precision-first", ""));
    assert(!o.precision_first);
    // Парсер: «off» в секции — явный off (эквивалент умолчанию).
    auto m = parse_overrides_ini("[a]\nprecision-first = off\n");
    assert(!m.at("a").precision_first);
  }

  // Редактор секции: вставка, замена, удаление, чужие секции не тронуты.
  {
    const std::string conf = "[general]\nctx = 4096\n\n[a]\nctx-size = 1024\n"
                             "# важный комментарий\n[b]\nsystem = привет\n";
    // Вставка нового ключа в существующую секцию.
    std::string out = edit_override_section(conf, "a", "precision-first", "on");
    assert(out.find("[a]\nctx-size = 1024\nprecision-first = on\n") != std::string::npos);
    assert(out.find("[general]\nctx = 4096") != std::string::npos);
    assert(out.find("system = привет") != std::string::npos);
    assert(out.find("# важный комментарий") != std::string::npos);
    // Замена существующего.
    out = edit_override_section(conf, "a", "ctx-size", "2048");
    assert(out.find("ctx-size = 2048") != std::string::npos &&
           out.find("ctx-size = 1024") == std::string::npos);
    // Удаление (value пуст) — ключ исчез, остальное живо.
    out = edit_override_section(conf, "a", "ctx-size", "");
    assert(out.find("ctx-size") == std::string::npos);
    assert(out.find("# важный комментарий") != std::string::npos);
    // Секции нет — дописывается в конец; пустой файл — минимальный каркас.
    out = edit_override_section(conf, "c", "precision-first", "on");
    assert(out.find("[c]\nprecision-first = on") != std::string::npos);
    out = edit_override_section("", "c", "precision-first", "on");
    assert(out == "[c]\nprecision-first = on\n");
    // Удалять нечего — файл не меняется.
    assert(edit_override_section(conf, "a", "precision-first", "") == conf);
  }

  // Парсер + редактор: замкнутый круг (записали — прочитали то же).
  {
    std::string conf = edit_override_section("", "Model-X", "precision-first", "on");
    auto m = parse_overrides_ini(conf);
    assert(m.at("Model-X").precision_first);
    conf = edit_override_section(conf, "Model-X", "precision-first", "");
    m = parse_overrides_ini(conf);
    assert(!m.count("Model-X") || !m.at("Model-X").precision_first);
  }

  // sanitize_id: инъекция секций невозможна, пустое имя — «model».
  {
    assert(sanitize_id("q[we]rty") == "q_we_rty");
    assert(sanitize_id("") == "model");
    assert(sanitize_id("ok.name-1_x") == "ok.name-1_x");
    // Имя файла с переводом строки и заголовком внутри: секцию писатель
    // пишет только из sanitize_id (скобки и переводы строки — в «_»), поэтому
    // «[evil]» внутри имени файла не становится заголовком секции.
    std::string evil = "a\n[evil]";
    const std::string safe = sanitize_id(evil);
    assert(safe.find('\n') == std::string::npos &&
           safe.find('[') == std::string::npos);
    auto m = parse_overrides_ini("[" + safe + "]\nctx-size = 1\n");
    assert(m.size() == 1 && !m.count("evil") && m.count(safe) == 1);
    // Незакрытый заголовок — не секция: ключи мимо неё не попадают.
    auto u = parse_overrides_ini("[unclosed\nctx-size = 1\n");
    assert(u.empty());
  }

  std::puts("overrides_test: ok");
  return 0;
}
