// Лестница попыток загрузки: от самой оптимистичной конфигурации к деградации.
// Умный режим (auto_params): сначала контекст «из модели» и квантованный KV-кэш,
// для MoE, которому тесно, — частичное размещение экспертов (--n-cpu-moe N:
// первые N блоков в RAM) или полный вынос (-ot exps=CPU), если раскладки нет.
// Каждая последующая попытка откатывает одну «дорогую» фичу, добавляет вынос
// экспертов или режет ресурсы пополам. Flash Attention идёт только как носитель
// KV q8_0 (проверено строгим A/B 01.10 в одном сеансе: prefill и генерация
// одинаковы с ним и без — на CUDA движок и так включает FA по умолчанию).
// Отдельное измерение — носитель: если модель живёт на флэшке или сетевом
// диске, холодные страницы CPU-части дают рывки, поэтому попытки с оффлоадом
// помечаются warm (прогреть страничный кэш файла после старта). Чистая
// функция — есть тест.
#pragma once
#include <vector>

namespace lh::plan {

struct Attempt {
  int ngl;         // слоёв на GPU
  int ctx;         // контекст
  bool kv_q8;      // -ctk/-ctv q8_0 (требует Flash Attention)
  bool fa;         // -fa on (сопровождает kv_q8)
  bool exps_cpu;   // MoE: -ot exps=CPU — полный вынос экспертов
  int ncmoe;       // MoE частично: --n-cpu-moe N (первые N блоков в RAM)
  bool warm;       // прогреть страничный кэш файла (медленный носитель)
};

inline std::vector<Attempt> build(bool auto_params, int ngl0, int ctx0, bool fits_all,
                                  bool moe, bool ram_ok, int ncmoe0, bool slow_media,
                                  long long n_layers) {
  std::vector<Attempt> a;
  int ngl = ngl0 > 0 ? ngl0 : 0;
  int ctx = ctx0 > 0 ? ctx0 : 4096;
  // Прогрев нужен, только если при попытке часть весов остаётся на CPU:
  // MoE-эксперты или недогруженные слои. Всё на GPU — файл и так прочитан.
  auto warm_of = [&](const Attempt& x) {
    return auto_params && slow_media &&
           (x.exps_cpu || x.ncmoe > 0 || (n_layers > 0 && x.ngl < n_layers));
  };

  if (!auto_params) {
    // Ручной режим: как раньше — режем ngl пополам, затем контекст.
    for (int i = 0; i < 6; ++i) {
      Attempt x{ngl, ctx, false, false, false, 0, false};
      a.push_back(x);
      if (ngl > 0) ngl /= 2;
      else ctx /= 2;
      if (ctx < 256 && ngl <= 0) break;
    }
    return a;
  }

  // Умный: раскладка MoE для тесной модели. Частичный сплит (ncmoe0) — первые
  // ncmoe0 блоков экспертов в RAM; не влез — выносим больше (полу-шаг к полному,
  // затем полный -ot exps=CPU). Без раскладки — полный вынос сразу, как раньше
  // (MoE-эксперты на CPU только если RAM это тянет).
  const bool exps = moe && ram_ok && !fits_all && ncmoe0 <= 0;
  if (moe && !fits_all && ncmoe0 > 0) {
    // 1) всё сразу: авто-контекст + KV q8_0 (FA обязателен для q8)
    a.push_back({ngl, ctx, true, true, false, ncmoe0, false});
    // 2) откат квантования KV (если сборка/модель его не потянула) — вместе с FA
    a.push_back({ngl, ctx, false, false, false, ncmoe0, false});
    if (n_layers > 0) {
      const int mid = (ncmoe0 + (int)n_layers) / 2;
      // 3) полу-шаг к полному выносу: та же раскладка не прошла — байтов в RAM
      //    вдвое больше, GPU-часть вдвое беднее
      if (mid > ncmoe0 && mid < (int)n_layers)
        a.push_back({ngl, ctx, false, false, false, mid, false});
      // 4) полный вынос
      a.push_back({ngl, ctx, false, false, true, 0, false});
    }
  } else {
    a.push_back({ngl, ctx, true, true, exps, 0, false});
    a.push_back({ngl, ctx, false, false, exps, 0, false});
  }
  // деградация: слои, затем контекст. Размещение экспертов наследуется от
  // самой глубокой достигнутой попытки: MoE, не влезающий целиком, не
  // влезет и на половине слоёв, а сброс exps/ncmoe на середине лестницы
  // терял форму (и кэш поверх неё). Прежнее «в деградации откатываем —
  // пусть уж лучше половина слоёв на GPU» относилось к жизни до кэша:
  // 30.10 плотный сплит после OOM-попыток дал 19 ток/с против ~60-100 у
  // exps-формы с кэшем на той же модели (Next-80B).
  const bool deg_exps = a.back().exps_cpu;
  const int deg_ncmoe = a.back().ncmoe;
  for (int i = 0; i < 4; ++i) {
    if (ngl > 0) ngl /= 2;
    else ctx /= 2;
    if (ctx < 256 && ngl <= 0) break;
    a.push_back({ngl, ctx, false, false, deg_exps, deg_ncmoe, false});
  }
  for (auto& x : a) x.warm = warm_of(x);
  return a;
}

} // namespace lh::plan
