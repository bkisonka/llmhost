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
  // кэш-замер есть — чередующаяся проба по нему решает мерять базу
  assert(profiles::has_cached_form_speed("m"));
  assert(!profiles::has_cached_form_speed("без-пробы"));
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

  // ---- телеметрия петли самоизмерения -------------------------------------
  // loop_of агрегирует уже записанные данные: формы, пара «кэш/база» с
  // временем, точка покрытия с локалью и вердикт авто-сетки. Всё переживает
  // save/load — «петля мертва» меряется снаружи по возрасту пары.
  {
    const Form fbase{true, true, 2048, true, true, 0, 0, 0, 0};     // база: эксперты в RAM
    const Form fcache{true, true, 2048, true, true, 0, 0, 6144, 0}; // форма с кэшем
    ModelProfile q;
    q.out_gpu = true;
    q.cuda = true;
    q.kv_q8 = true;
    q.ubatch = 2048;
    q.exps_cpu = true;
    q.ctx = 32768;
    q.overhead = 1u << 30;
    profiles::put("loop-model", fbase, q);
    profiles::put("loop-model", fcache, q);
    profiles::note_speed("loop-model", fbase, 20.0, 256);
    profiles::note_speed("loop-model", fcache, 22.0, 256);
    profiles::LoopState ls = profiles::loop_of("loop-model");
    assert(ls.moe && ls.forms == 2); // обе формы с осевшей скоростью
    assert(ls.pair && ls.pair_ctx == 32768 && ls.pair_ts > 0);
    assert(!profiles::loop_of("legacy-model").moe); // плотная — петли нет
    assert(!profiles::loop_of("несуществует").moe &&
           profiles::loop_of("несуществует").forms == 0);

    // вердикты авто-сетки: свежий перезаписывает старый, чужой kind — мимо
    profiles::note_moe_grid("loop-model", "probe", 8023, 0.0);
    ls = profiles::loop_of("loop-model");
    assert(ls.verdict == "probe" && ls.verdict_ts > 0);
    profiles::note_moe_grid("loop-model", "grid", 8023, 10.2);
    ls = profiles::loop_of("loop-model");
    assert(ls.verdict == "grid" && ls.verdict_mib == 8023);
    assert(ls.verdict_ms > 10.19 && ls.verdict_ms < 10.21);
    profiles::note_moe_grid("loop-model", "weird", 1, 0.0);
    assert(profiles::loop_of("loop-model").verdict == "grid");

    // точка покрытия с выведенной локалью — a доезжает до телеметрии
    profiles::note_moe_point("loop-model", 6144, 0.9, 4.23);
    ls = profiles::loop_of("loop-model");
    assert(ls.a > 4.22 && ls.a < 4.24);
    assert(ls.point_mib == 6144 && ls.point_hit > 0.89 && ls.point_hit < 0.91);

    // ---- sustained-сэмплы: контекст замера отдельно от калибровочного -----
    // note_speed с speed_ctx перезаписывает скорость (живой замер важнее
    // warmup) и пишет контекст ЗАМЕРА; калибровочный ctx записи не трогается
    // (надбавка осталась замерена на своём контексте). Пара «кэш/база»
    // сверяет контексты замера — sustained-пара честна к ctx сессий.
    {
      profiles::note_speed("loop-model", fcache, 55.0, 1024, 262144);
      profiles::note_speed("loop-model", fbase, 20.0, 1024, 262144);
      profiles::CacheProbe scp, sbp;
      assert(profiles::moe_pair_of("loop-model", scp, sbp));
      assert(scp.ctx == 262144 && sbp.ctx == 262144); // контексты замера
      const std::optional<ModelProfile> pc = profiles::lookup("loop-model", fcache);
      assert(pc && pc->ctx == 32768);        // калибровка не сдвинулась
      assert(pc && pc->gen_tps == 55.0 && pc->gen_n == 1024); // перезапись
      assert(pc && pc->speed_ctx == 262144);
      // контексты разошлись — пара честно не собирается, хотя калибровочный
      // ctx записей одинаков
      profiles::note_speed("loop-model", fbase, 20.0, 1024, 8192);
      assert(!profiles::moe_pair_of("loop-model", scp, sbp));
      // перезапись без speed_ctx (0) прежнюю скорость не сносит на «неизвестно»
      profiles::note_speed("loop-model", fbase, 20.0, 1024, 262144);
      profiles::note_speed("loop-model", fbase, 21.0, 2048); // warmup поверх sustained
      const std::optional<ModelProfile> pb = profiles::lookup("loop-model", fbase);
      assert(pb && pb->speed_ctx == 262144);
      // sustained-shell (полной записи нет — record_profile отказал) видна
      // паре по speed_ctx: форма чистая, контекст замера есть
      const Form fsh{true, true, 2048, true, true, 0, 0, 2048, 0};
      const Form fsp{true, true, 2048, true, true, 0, 0, 0, 0};
      profiles::note_speed("shell", fsh, 30.0, 512, 16384);
      profiles::note_speed("shell", fsp, 20.0, 512, 16384);
      assert(profiles::has_cached_form_speed("shell"));
      profiles::CacheProbe scp2, sbp2;
      assert(profiles::moe_pair_of("shell", scp2, sbp2));
      assert(scp2.ctx == 16384 && sbp2.ctx == 16384);
      // shell без контекста замера (warmup в пустую форму — не бывает, но
      // контракт проверяем) пару не собирает
      const Form fsh2{true, true, 2048, true, true, 0, 0, 4096, 0};
      profiles::note_speed("noshell", fsh2, 30.0, 512);
      assert(!profiles::has_cached_form_speed("noshell"));
    }

    // рестарт демона: вердикт и точка читаются с диска, пара на месте
    profiles::save();
    profiles::load();
    ls = profiles::loop_of("loop-model");
    assert(ls.moe && ls.forms == 2 && ls.pair);
    assert(ls.verdict == "grid" && ls.verdict_mib == 8023);
    assert(ls.a > 4.22 && ls.a < 4.24);
  }

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
  // Вердикт сетки — замер этого движка: снесён вместе с формами (точка
  // покрытия остаётся — свойство модели, как и раньше).
  assert(!profiles::loop_of("loop-model").pair);
  assert(profiles::loop_of("loop-model").verdict.empty());
  assert(profiles::loop_of("loop-model").forms == 0);
  assert(profiles::loop_of("loop-model").a > 4.22);

  // ---- гейт качества ------------------------------------------------------
  // ε = 2σ по истории прогонов; метрика «нет» гейт не закрывает; σ ещё не
  // калибрована (< 2 прогонов) — тоже не закрывает (капризничающий гейт
  // отключили бы через месяц). Суит детерминирован — повторные прогоны
  // сходятся (σ = 0, ε = 0), регресс ловится любой.
  {
    const Form fr{true, true, 2048, true, false, 0, 0};
    Form fc{true, true, 2048, true, false, 0, 0};
    fc.moe_cache = 4096; // кандидат — соседний грант кэша
    const std::string id = "qm";
    // Ни у резидента, ни у кандидата данных — гейт открыт («нет» ≠ провал).
    assert(profiles::quality_gate(id, fr, fc));
    // Прогона мало (σ не калибрована) — гейт открыт, даже при явном регрессе.
    profiles::note_quality(id, fr, {1.0, 1.0, 1.0, -1});
    profiles::note_quality(id, fc, {0.0, 0.0, 0.0, -1});
    assert(profiles::quality_gate(id, fr, fc));

    // Детерминированные повторные прогоны: σ = 0, ε = 0 — регресс ловится.
    const std::string id2 = "qc";
    assert(profiles::quality_gate(id2, fr, fc));
    profiles::note_quality(id2, fr, {1.0, 1.0, 1.0, -1});
    profiles::note_quality(id2, fr, {1.0, 1.0, 1.0, -1});
    profiles::note_quality(id2, fc, {1.0, 0.0, 1.0, -1});
    // У кандидата один прогон — ε не вычислить, гейт открыт.
    assert(profiles::quality_gate(id2, fr, fc));
    profiles::note_quality(id2, fc, {1.0, 0.0, 1.0, -1});
    std::string why;
    assert(!profiles::quality_gate(id2, fr, fc, &why));
    assert(why.find("tools") != std::string::npos);
    // Форма против себя — сравнение с собой не требуется, гейт открыт.
    assert(profiles::quality_gate(id2, fr, fr));
    // Метрика «нет» у кандидата (верификатор не назначен) — не закрывает.
    Form fc2{true, true, 2048, true, false, 0, 0};
    fc2.moe_cache = 2048;
    profiles::note_quality(id2, fc2, {1.0, -1, 1.0, -1});
    profiles::note_quality(id2, fc2, {1.0, -1, 1.0, -1});
    assert(profiles::quality_gate(id2, fr, fc2));
    // Все кандидаты регрессируют → резидент остаётся: то же решение гейта.
    profiles::note_quality(id2, fc2, {0.0, 0.0, 0.0, -1});
    profiles::note_quality(id2, fc2, {0.0, 0.0, 0.0, -1});
    assert(!profiles::quality_gate(id2, fr, fc2));
    // Качество переживает рестарт и сносится сменой движка (пин в
    // идентичности — через установку пиннутого релиза).
    profiles::save();
    profiles::load();
    assert(!profiles::quality_gate(id2, fr, fc));
    profiles::note_engine("b88888");
    profiles::QualityState qs;
    assert(!profiles::quality_of(id2, fr, qs));
    assert(profiles::quality_gate(id2, fr, fc));
    // Вердикт «quality» читается из вердикта сетки (стоп-условие
    // исследователя доходит).
    profiles::note_moe_grid(id2, "quality", 0, 0.0);
    assert(profiles::loop_of(id2).verdict == "quality");
  }

  std::filesystem::remove_all(dir);
  std::puts("profiles_test: ok");
  return 0;
}
