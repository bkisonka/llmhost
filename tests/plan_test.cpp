// Лестница попыток: умный и ручной режимы, откат KV, gating MoE-on-CPU.
#include "plan.h"

#include <cstdio>
#include <cassert>

using lh::plan::build;
using A = lh::plan::Attempt;

int main() {
  // Ручной режим: как раньше, без умных фич — прогрева нет даже на флэшке.
  {
    auto a = build(false, 999, 4096, true, false, false, /*ncmoe=*/0, /*slow=*/true, 32);
    assert(a.size() >= 2);
    assert(a[0].ngl == 999 && a[0].ctx == 4096);
    assert(!a[0].kv_q8 && !a[0].exps_cpu && a[0].ncmoe == 0 && !a[0].warm);
    // ngl режется первым
    assert(a[1].ngl == 499 && a[1].ctx == 4096);
  }

  // Умный, плотная модель, влезает: сначала q8, затем откат, затем деградация.
  {
    auto a = build(true, 32, 8192, true, false, false, 0, false, 32);
    assert(a.size() == 6);
    assert(a[0].kv_q8 && !a[0].exps_cpu && a[0].ngl == 32 && a[0].ctx == 8192);
    assert(!a[1].kv_q8 && a[1].ctx == 8192);
    assert(a[2].ngl == 16);
    assert(a[3].ngl == 8);
    assert(a[4].ngl == 4 && a[4].ctx == 8192);
    assert(a[5].ngl == 2 && a[5].ctx == 8192);
    // FA сопровождает только KV q8_0 (сама по себе скорости не добавляет).
    assert(a[0].fa && !a[1].fa && !a[2].fa && !a[5].fa);
  }

  // MoE, не влезает, RAM хватает, раскладки нет: полный вынос держится через
  // всю лестницу — деградация наследует размещение самой глубокой попытки.
  // Прежний сброс exps в деградации (30.10) ронял MoE в плотный сплит без
  // кэша: 19 ток/с против ~60-100 у exps-формы на той же модели.
  {
    auto a = build(true, 48, 4096, false, true, true, 0, false, 48);
    assert(a[0].exps_cpu && a[0].kv_q8);
    assert(a[1].exps_cpu && !a[1].kv_q8);
    assert(a[2].exps_cpu);
  }

  // MoE с частичной раскладкой (первые 12 блоков в RAM): попытки 1-2 держат
  // её; не влезла — полу-шаг к полному выносу (30 из 48), затем полный,
  // затем обычная деградация слоями.
  {
    auto a = build(true, 49, 4096, false, true, true, 12, false, 48);
    assert(a.size() == 8);
    assert(a[0].ncmoe == 12 && a[0].kv_q8 && !a[0].exps_cpu);
    assert(a[1].ncmoe == 12 && !a[1].kv_q8);
    assert(a[2].ncmoe == 30 && !a[2].exps_cpu); // (12+48)/2
    assert(a[3].exps_cpu && a[3].ncmoe == 0);   // полный вынос
    for (size_t i = 4; i < a.size(); ++i)
      assert(a[i].exps_cpu && a[i].ncmoe == 0); // деградация после полного выноса
    assert(a[4].ngl == 24 && a[5].ngl == 12);
  }

  // Частичная раскладка при чётной близости к полному выносу: полу-шаг слился
  // бы с полным — вставки не будет, сразу полный.
  {
    auto a = build(true, 49, 4096, false, true, true, 47, false, 48);
    assert(a[0].ncmoe == 47);
    assert(a[2].exps_cpu && a[2].ncmoe == 0); // (47+48)/2=47 — пропущен
    assert(a.size() == 7);
  }

  // «Нечего соваться»: RAM не тянет экспертов и раскладки нет — CPU-offload
  // не предлагаем вовсе.
  {
    auto a = build(true, 48, 4096, false, true, false, 0, false, 48);
    for (auto& x : a) assert(!x.exps_cpu && x.ncmoe == 0);
  }

  // MoE, который и так влезает целиком: эксперты остаются на GPU.
  {
    auto a = build(true, 48, 4096, true, true, true, 0, false, 48);
    for (auto& x : a) assert(!x.exps_cpu && x.ncmoe == 0);
  }

  // ngl=0 (всё на CPU): дальше режется только контекст.
  {
    auto a = build(true, 0, 4096, true, false, false, 0, false, 28);
    assert(a[0].ngl == 0);
    assert(a[2].ctx == 2048);
    assert(a[3].ctx == 1024);
  }

  // Носитель: на флэшке прогрев получает любая попытка с CPU-частью.
  {
    // MoE-гибрид: первые две — эксперты на CPU, деградация — слои недогружены.
    auto a = build(true, 48, 4096, false, true, true, 0, /*slow=*/true, 48);
    for (auto& x : a) assert(x.warm);
    // Частичная раскладка — тоже CPU-часть: прогрев на первых попытках.
    auto p = build(true, 48, 4096, false, true, true, 12, /*slow=*/true, 48);
    for (auto& x : p) assert(x.warm);
    // Плотная модель целиком на GPU: пока слоёв хватает, файл прочитан целиком —
    // без прогрева (в деградации слои недогружаются — там прогрев законен).
    auto b = build(true, 32, 8192, true, false, false, 0, /*slow=*/true, 32);
    assert(!b[0].warm && !b[1].warm);
    // Плотная, не влезает: первые попытки целиком на GPU — без прогрева,
    // деградация недогружает слои — с прогревом.
    auto c = build(true, 32, 8192, false, false, false, 0, /*slow=*/true, 32);
    assert(!c[0].warm && !c[1].warm);
    assert(c[2].warm && c[3].warm);
    // Медленный носитель, MoE влезает целиком: первые попытки — всё на GPU,
    // без прогрева; но лестница деградации недогружает слои — там прогрев есть.
    auto d = build(true, 48, 4096, true, true, true, 0, /*slow=*/true, 48);
    assert(!d[0].warm && !d[1].warm);
    for (size_t i = 2; i < d.size(); ++i) assert(d[i].warm);
  }

  std::puts("plan_test: ok");
  return 0;
}
