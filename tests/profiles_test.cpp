// Тест хранилища профилей: ключ «id+форма», миграция плоского формата,
// флаг амбициозной попытки, штамп релиза движка. Файлы — во временном
// LH_DATA_DIR.
#include "profiles.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

using namespace lh;

int main() {
  char tmpl[] = "/tmp/lh-profiles-test-XXXXXX";
  char* dir = mkdtemp(tmpl);
  assert(dir);
  setenv("LH_DATA_DIR", dir, 1);

  using profiles::Form;
  using profiles::ModelProfile;

  // ---- make_key -----------------------------------------------------------
  const Form f1{true, true, 2048, true, false, 0, 0};
  const Form f2{true, true, 2048, true, true, 0, 0};    // эксперты в RAM — другой слот
  const Form f3{false, true, 512, false, false, 28, 0}; // MoE-сплит — свой слот
  const Form f4{false, true, 512, false, false, 0, 28}; // плотный сплит — тоже свой
  assert(profiles::make_key("m", f1) != profiles::make_key("m", f2));
  assert(profiles::make_key("m", f1) != profiles::make_key("m", f3));
  assert(profiles::make_key("m", f3) != profiles::make_key("m", f4)); // ncmoe ≠ ngl
  // «|» в id не переживает санитайзер — составной ключ не перетечёт.
  assert(profiles::make_key("a|b", f1) == profiles::make_key("a_b", f1));
  assert(profiles::make_key("m", f1).find("|o1k1c1e0m0n0u2048") != std::string::npos);

  // ---- put/lookup по формам ----------------------------------------------
  ModelProfile p;
  p.out_gpu = true;
  p.cuda = true;
  p.kv_q8 = true;
  p.ubatch = 2048;
  p.ctx = 32768;
  p.overhead = 3000u << 20;
  profiles::put("m", f1, p);
  assert(profiles::lookup("m", f1) &&
         profiles::lookup("m", f1)->overhead == (3000u << 20));
  assert(!profiles::lookup("m", f2)); // замеры форм не смешиваются
  assert(!profiles::lookup("m", f3));
  assert(!profiles::lookup("другая-модель", f1));

  // ---- кэш MoE и блоки на запасной карте — часть формы -------------------
  // Арена SLRU живёт в тех же гигабайтах, что веса: замер с кэшем не должен
  // ложиться в слот без кэша (отравленный оверхед) и наоборот. Блоки -ot
  // меняют host-состав — тоже свой слот. Запись обязана соответствовать
  // форме (experts в RAM — как делает record_profile живой загрузки).
  const Form fc{true, true, 2048, true, true, 0, 0, 6144, 0};  // кэш 6 ГиБ
  const Form fot{true, true, 2048, true, true, 0, 0, 0, 7};    // 7 блоков на запасной
  assert(profiles::make_key("m", f2) != profiles::make_key("m", fc));
  assert(profiles::make_key("m", fc) != profiles::make_key("m", fot));
  p.exps_cpu = true;
  profiles::put("m", fc, p);
  assert(profiles::lookup("m", fc) && !profiles::lookup("m", f2));

  // ---- пара «с кэшем / без» для вывода локальности ------------------------
  // moe_pair_of требует обе скорости, равный контекст и достаточную выборку;
  // возвращает составы форм (host-часть) для обращения T(C). Скорость
  // пишется поверх полной записи формы (ctx из замера загрузки — как в бою).
  profiles::put("m", f2, p);
  profiles::note_speed("m", f2, 20.0, 256);   // база: эксперты в RAM, без кэша
  profiles::note_speed("m", fc, 22.0, 256);   // проба с кэшем, тот же ctx
  profiles::CacheProbe cp, bp;
  assert(profiles::moe_pair_of("m", cp, bp));
  assert(cp.moe_cache == 6144 && cp.exps_cpu && bp.moe_cache == 0);
  assert(cp.ctx == 32768 && bp.ctx == 32768);
  assert(cp.tps == 22.0 && bp.tps == 20.0);
  // контексты разошлись — T_аттн не сокращается, пары нет
  p.ctx = 4096;
  profiles::put("m", f2, p);
  profiles::note_speed("m", f2, 30.0, 256);
  assert(!profiles::moe_pair_of("m", cp, bp));
  // короткая выборка кэш-стороны не считается
  p.ctx = 32768;
  profiles::put("m", f2, p);
  profiles::note_speed("m", f2, 20.0, 256);
  profiles::note_speed("m", fc, 22.0, 64);
  assert(!profiles::moe_pair_of("m", cp, bp));
  // класс KV разошёлся (политика дала кэш-форме f16) — T_аттн не сокращается
  profiles::note_speed("m", fc, 22.0, 256);
  const Form f16c{true, false, 2048, true, true, 0, 0, 6144, 0};
  p.kv_q8 = false;
  profiles::put("m", f16c, p);
  profiles::note_speed("m", f16c, 21.5, 256); // свежее — стала бы «последней»
  assert(!profiles::moe_pair_of("m", cp, bp));
  p.kv_q8 = true;

  // ---- флаг амбициозной попытки ------------------------------------------
  profiles::note_failed_out("m", f1); // OOM в этой форме
  assert(!profiles::lookup("m", f1)); // флаг = «замера нет»
  profiles::put("m", f2, p);          // другая форма — флаг f1 не трогает
  assert(!profiles::lookup("m", f1));
  assert(profiles::lookup("m", f2));
  profiles::put("m", f1, p); // успешный замер этой же формы снимает флаг
  assert(profiles::lookup("m", f1));

  // ---- миграция плоского формата -----------------------------------------
  {
    std::ofstream out(std::string(dir) + "/model_profiles.json");
    out << R"({
  "legacy-model": {"v": 2, "out_gpu": true, "cuda": true, "kv_q8": true,
                   "ubatch": 512, "ctx": 65536, "overhead": 1500000000,
                   "ram_bytes": 0, "exps_cpu": false, "ngl": 0, "updated": 100.0}
})";
  }
  profiles::load(); // старый файл: ключ = id, форма — из полей записи
  const Form legacy{true, true, 512, true, false, 0, 0};
  const std::optional<ModelProfile> lp = profiles::lookup("legacy-model", legacy);
  assert(lp && lp->ctx == 65536 && lp->overhead == 1500000000);
  profiles::put("legacy-model", legacy, *lp); // сохранится уже составным ключом
  profiles::load(); // перечитали v2-файл — запись нашлась по тому же слоту
  assert(profiles::lookup("legacy-model", legacy));

  // ---- штамп релиза движка -----------------------------------------------
  profiles::note_engine("b11379");
  assert(!profiles::lookup("m", f1)); // все замеры снесены
  assert(!profiles::lookup("legacy-model", legacy));
  profiles::note_engine("b11379"); // повтор — штамп тот же, тихо
  // Записи чужой семантики замера (v1) отбрасываются при чтении.
  {
    std::ofstream out(std::string(dir) + "/model_profiles.json");
    out << R"({"engine": "b11379", "profiles": {"old": {"v": 1, "ctx": 4096, )"
           R"("overhead": 1000000, "out_gpu": true, "kv_q8": true, "ubatch": 512, )"
           R"("cuda": true, "exps_cpu": false, "ngl": 0}}})";
  }
  profiles::load();
  assert(!profiles::lookup("old", Form{true, true, 512, true, false, 0, 0}));

  assert(!profiles::kv_q8_by_speed(true, 465, 128, 510, 128)); // f16 быстрее на ~10%
  assert(profiles::kv_q8_by_speed(true, 560, 128, 500, 128));     // q8 быстрее на 12%
  assert(profiles::kv_q8_by_speed(true, 100, 128, 105, 128));     // меньше 8% — политика
  assert(!profiles::kv_q8_by_speed(false, 0, 0, 500, 128));        // нет второй выборки
  assert(profiles::kv_q8_by_speed(true, 600, 128, 500, 64));      // короткая выборка не считается

  // ---- репутация draft: переживает save/load, стирается сменой движка ----
  profiles::note_bad_draft("/m/mtp-A-Q8_0.gguf", "dead");
  profiles::note_bad_draft("/m/mtp-B-Q8_0.gguf", "shy");
  profiles::note_bad_draft("/m/mtp-C-Q8_0.gguf", "weird"); // неизвестный kind — мимо
  assert(profiles::bad_drafts("dead").size() == 1);
  assert(profiles::bad_drafts("shy").size() == 1);
  assert(profiles::bad_drafts("weird").empty());
  profiles::save();
  profiles::load(); // рестарт демона: вердикты читаются с диска
  assert(profiles::bad_drafts("dead").size() == 1);
  assert(profiles::bad_drafts("shy").size() == 1);
  profiles::note_engine("b99999"); // другой релиз — вердикты этому движку не переносятся
  assert(profiles::bad_drafts("dead").empty());
  assert(profiles::bad_drafts("shy").empty());

  std::filesystem::remove_all(dir);
  std::puts("profiles_test: ok");
  return 0;
}
