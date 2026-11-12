#include "fit.h"

#include <algorithm>
#include <cmath>

namespace lh::fit {

namespace {
// Насыщающая арифметика для произведений недоверенных скаляров Meta: переполнение
// должно означать «не влезает никогда» (UINT64_MAX против любой VRAM), а не
// замкнуться по модулю 2^64 в маленькое число и дать ложное fits.
uint64_t sat_mul(uint64_t a, uint64_t b) {
  return (a && b > ~(uint64_t)0 / a) ? ~(uint64_t)0 : a * b;
}
uint64_t sat_add(uint64_t a, uint64_t b) {
  return a > ~(uint64_t)0 - b ? ~(uint64_t)0 : a + b;
}
} // namespace

hw::Gpu pool(const std::vector<hw::Gpu>& cards, hw::Build build) {
  hw::Gpu sum;
  int n = 0;
  uint64_t free_sum = 0, total_sum = 0;
  for (const auto& c : cards) {
    if (!c.nvidia || !c.vram_total) continue; // без телеметрии планов нет
    if (n == 0) {
      sum = c; // RAM, имя и прочее — от первой
      sum.build = build;
    }
    ++n;
    free_sum += c.vram_free;
    total_sum += c.vram_total;
  }
  if (n == 0) return hw::Gpu{}; // vram_free = 0 — все оценки «мимо»
  const bool cuda = build == hw::Build::Cuda12 || build == hw::Build::Cuda13;
  // Насыщающее вычитание: на почти забитых картах (свободная сумма меньше
  // оверхедов соседей) без него беззнаковое вычитание уезжает вниз и пул
  // сообщает «бесконечную» свободную память — все оценки говорили бы
  // «влезает». Ноль — честное «мимо», семантика та же, что у пустого пула.
  const uint64_t sub = (uint64_t)(n - 1) * fixed_overhead(cuda);
  sum.vram_free = free_sum > sub ? free_sum - sub : 0;
  sum.vram_total = total_sum;
  if (n > 1) sum.name = sum.name + " +" + std::to_string(n - 1);
  return sum;
}

Estimate estimate(const gguf::Meta& m, uint64_t file_size, uint32_t ctx, const hw::Gpu& gpu,
                  bool kv_q8, const Measured* measured) {
  // Запас поверх окна SWA-слоёв: llama.cpp держит локальные слои чуть шире
  // окна (паддинг на ubatch/выравнивание кэша). Сознательно щедро: недобор
  // здесь означал бы план без запаса, а лишние ~2 ГиБ токенов у окна — шум.
  constexpr uint64_t kSwaCachePad = 1024;
  Estimate e;
  e.weights = file_size;

  const ArchProfile prof0 = arch_profile(m.arch);
  // Мета сильнее таблицы семейств: интервал токенного KV из GGUF (qwen4exp и
  // будущие гибриды, которых таблица не знает — поймано живьём: без ключа
  // qwen4exp планировался как полная attention-модель, KV вчетверо больше
  // реального) и присутствие ssm-слоёв как признак рекуррентного состояния.
  // Интервал ограничиваем числом слоёв: враждебная мета с интервалом в тысячи
  // иначе обнуляет оценку KV (kv_frac защищён так же, как остальные скаляры).
  // Калибровка 10000 байт на РЕКУРРЕНТНЫЙ слой×embd (7500 было средним по всем
  // слоям измеренной qwen35 — 48 рекуррентных из 64); живой замер профиля
  // дальше уточняет её автоматически.
  ArchProfile prof = prof0;
  uint64_t interval = 0; // слоёв на один full-attention; 0 — модель не гибрид
  if (m.full_attn_interval > 1) {
    interval = std::min<uint64_t>(m.full_attn_interval,
                                  std::max<uint64_t>(m.n_layers, 1));
    prof.kv_frac = 1.0 / (double)interval;
  } else if (prof0.kv_frac < 1.0 && prof0.kv_frac > 0.0) {
    interval = (uint64_t)std::lround(1.0 / prof0.kv_frac);
  }
  if (m.has_ssm && prof.state_per_le == 0) prof.state_per_le = 10000;

  // KV-кэш: K и V по n_embd_kv токенов на слой. GQA (n_head_kv < n_head) его сжимает.
  // Разрядность — f16 (2 байта на элемент) или q8_0 (блок 34 байта на 32
  // элемента = 17/16 байта; ровно 1 байт недосчитывал ~6% кэша). У гибридных
  // арок токенный кэш — только на части слоёв (см. ArchProfile).
  uint64_t kv_heads = m.n_head_kv ? m.n_head_kv : m.n_head;
  // Размер головы: явный attention.key_length точнее n_embd/n_head — writer'ы
  // gemma-семьи держат KV-головы шире расчётного (gemma4: 512 против 2816/16).
  uint64_t head_dim = m.n_embd_head_k ? m.n_embd_head_k
                      : (m.n_head && m.n_embd ? m.n_embd / m.n_head : 0);
  if (m.n_layers && kv_heads && head_dim) {
    // SWA (sliding-window attention): локальные слои держат лишь окно
    // (плюс паддинг кэша), глобальные — весь контекст; классы различаются
    // и размером головы, и числом KV-голов (gemma4: локальные 8×256,
    // глобальные 2×512). Без окна и паттерна в мете — слепая математика:
    // консервативно, но для gemma переоценка в гигабайты, планировщик
    // резал контекст и ломал запись профиля (надбавка уходила в минус).
    uint64_t n_global = m.n_layers, n_local = 0;
    uint64_t dim_g = head_dim, dim_l = head_dim, span_l = ctx;
    uint64_t kh_g = kv_heads, kh_l = kv_heads;
    if (m.n_swa && m.n_swa_layers && m.n_swa_layers < m.n_layers) {
      n_global = m.n_layers - m.n_swa_layers;
      n_local = m.n_swa_layers;
      dim_g = m.n_embd_head_k ? m.n_embd_head_k : head_dim;
      dim_l = m.n_embd_head_k_swa ? m.n_embd_head_k_swa : head_dim;
      kh_g = m.n_head_kv_global ? m.n_head_kv_global : kv_heads;
      kh_l = m.n_head_kv_local ? m.n_head_kv_local : kv_heads;
      span_l = std::min<uint64_t>(ctx, m.n_swa + kSwaCachePad);
    }
    const uint64_t g_part = sat_mul(sat_mul(sat_mul(n_global, (uint64_t)ctx), dim_g), kh_g);
    const uint64_t l_part = sat_mul(sat_mul(sat_mul(n_local, span_l), dim_l), kh_l);
    // Байты на элемент: f16 = 2, q8_0 = 34/32 = 17/16 (блок 34 байта на 32
    // элемента; ровно 1 байт недосчитывал ~6% кэша). Деление ДО насыщения:
    // sat_mul(17, elems/8) при переполнении даёт честный ~0, тогда как
    // насытившее произведение, поделённое после, потеряло бы семантику
    // «не влезает никогда».
    const uint64_t elems = sat_add(g_part, l_part);
    // после деления elems на 8: f16 = 32×elems/8 = 4×elems, q8 = 17×elems/8.
    const uint64_t per8 = kv_q8 ? 17ull : 32ull;
    e.kv = sat_add(sat_mul(per8, elems / 8), (elems % 8) * per8 / 8);
    if (prof.kv_frac < 1.0) e.kv = (uint64_t)((double)e.kv * prof.kv_frac);
  }
  // Рекуррентное состояние — только на рекуррентных слоях: при интервале N
  // каждый N-й слой full-attention и состояния не держит. Раньше константа
  // размазывалась по всем слоям — для гибридов с нешаблонным интервалом
  // (мета) оценка state уезжала в обе стороны.
  uint64_t rec_layers = m.n_layers;
  if (interval > 1 && m.n_layers)
    rec_layers = m.n_layers - (m.n_layers + interval - 1) / interval;
  e.state = (m.n_layers && m.n_embd && prof.state_per_le)
                ? sat_mul(prof.state_per_le, sat_mul(rec_layers, m.n_embd))
                : 0;

  // Оверхед бэкенда и вычислительные буферы. С Flash Attention (дефолт
  // CUDA-движка) compute от контекста почти не зависит: замер 02.10 —
  // 27B @65536 ест 400 МиБ compute при дефолтном ubatch (+132 МиБ draft).
  // База 512 МиБ покрывает разброс по моделям и спайки vision-кодирования,
  // ramp 16 КиБ/токен — страховка роста логитов/воркспейса. Vulkan без FA —
  // классика O(ctx): 64 КиБ на токен.
  bool cuda = gpu.build == hw::Build::Cuda12 || gpu.build == hw::Build::Cuda13;
  e.overhead = fixed_overhead(cuda);
  if (cuda)
    e.overhead = sat_add(e.overhead, sat_add(512ull << 20, (uint64_t)ctx << 14));
  else
    e.overhead = sat_add(e.overhead, (uint64_t)ctx << 16);
  // MoE ест compute заметно больше плотных арок: роутинг экспертов и
  // CUDA-графы по гигантским expert-опам. Замер 03.10 живьём (qwen3moe-30B
  // на 24 ГиБ карте, b11344): при ctx 65536 реальный compute+оверхед ~2,6 ГиБ
  // против 1,9 в оценке; при 131072 план оставлял запас 226 МиБ — ребёнок
  // ловил OOM «graph_reserve: failed to allocate compute buffers». +1 ГиБ к
  // базе уводит план на шаг контекста ниже вместо гарантированного OOM.
  // Только CUDA: на Vulkan MoE-путь не замерен, там консервативный O(ctx).
  if (cuda && m.n_experts > 0) e.overhead = sat_add(e.overhead, 1ull << 30);

  // Живой замер точнее аналитики (MoE-воркспейс, гибридные арки, NVFP4 дают
  // разброс ±1 ГиБ): подменяем оверхед измеренным, масштабируя ramp'ом на
  // дельту контекста. Вниз контекст оверхед тоже снижаем, но не ниже
  // фиксированной базы бэкенда.
  if (measured && measured->ctx > 0 && measured->overhead > 0) {
    const int64_t ramp = measured->cuda ? (1 << 14) : (1 << 16); // байт/токен
    const int64_t delta = (int64_t)ctx - (int64_t)measured->ctx;
    const int64_t floor = fixed_overhead(cuda);
    const int64_t ovh = (int64_t)measured->overhead + delta * ramp;
    e.overhead = (uint64_t)std::max(ovh < (int64_t)floor ? floor : ovh, (int64_t)0);
  }

  e.total = sat_add(sat_add(sat_add(e.weights, e.kv), e.state), e.overhead);
  e.fits = gpu.vram_free > 0 && e.total <= gpu.vram_free;
  return e;
}

uint32_t max_ctx(const gguf::Meta& m, uint64_t file_size, const hw::Gpu& gpu, uint32_t cap,
                 bool kv_q8, const Measured* measured) {
  if (!gpu.vram_free) return 0;
  uint32_t hi = cap ? cap : (m.n_ctx_train ? (m.n_ctx_train < 32768u ? (uint32_t)m.n_ctx_train : 32768u) : 4096u);
  // Оценка монотонна по контексту (KV и ramp растут, остальное постоянно),
  // поэтому максимум ищется бисекцией: прежняя лестница «пополам» возвращала
  // почти вдвое меньше реального максимума (помещалось 49k при старте с
  // 65536 — ответ 32768). Шаг кратности 1024; пол 512 — как у лестницы.
  if (estimate(m, file_size, hi, gpu, kv_q8, measured).fits) return hi;
  uint32_t lo = 0;   // заведомо влезает (0 — пока ничего не нашлось)
  uint32_t bad = hi; // заведомо не влезает
  while (bad - lo > 1024) {
    uint32_t mid = (uint32_t)(((uint64_t)lo + bad) / 2) & ~1023u;
    if (mid <= lo) break;
    if (estimate(m, file_size, mid, gpu, kv_q8, measured).fits)
      lo = mid;
    else
      bad = mid;
  }
  if (lo == 0)
    return estimate(m, file_size, 512, gpu, kv_q8, measured).fits ? 512u : 0u;
  return lo;
}

DenseSplit dense_split(const gguf::Meta& m, uint64_t file_size, const hw::Gpu& gpu,
                       uint32_t cap, bool kv_q8, uint32_t ctx_min, uint64_t extra_gpu_bytes) {
  DenseSplit s;
  if (!m.n_layers || !file_size || !gpu.vram_free) return s;
  if (ctx_min < 512) ctx_min = 512;
  const int layers = (int)m.n_layers;
  for (int ngl = layers; ngl >= dense_min_layers(layers); --ngl) {
    uint64_t w = file_size * (uint64_t)ngl / (uint64_t)layers;
    // Экстра-груз на GPU — фиксированный, не масштабируется слоями; в оценку
    // он входит как веса (estimate суммирует веса+KV+оверхед против свободной).
    uint32_t c = max_ctx(m, w + extra_gpu_bytes, gpu, cap, kv_q8);
    if (c >= ctx_min) {
      s.ok = true;
      s.ngl = ngl;
      s.ctx = c;
      s.gpu_bytes = w;
      s.cpu_bytes = file_size - w;
      break;
    }
  }
  return s;
}

MoESplit moe_split(const gguf::Meta& m, uint64_t file_size, const hw::Gpu& gpu,
                   uint32_t cap, bool kv_q8, uint32_t ctx_min, uint64_t extra_gpu_bytes) {
  MoESplit s;
  if (!m.n_layers || !file_size || !gpu.vram_free) return s;
  if (m.expert_bytes_by_block.empty()) return s; // без точных байт раскладка — догадка
  if (ctx_min < 512) ctx_min = 512;
  const int layers = (int)m.n_layers;
  // От «всё на GPU» вверх: первый проход N, дающий контекст не ниже ctx_min, —
  // минимум выноса при заданном контексте. N=0 отсеется сам (модель по гейту
  // не влезает), полный N=layers вызывающий заменяет формой exps_cpu.
  for (int n = 1; n <= layers; ++n) {
    const uint64_t gpu_w = moe_gpu_bytes(m, file_size, n);
    const uint64_t cpu_w = file_size - gpu_w;
    // Контекст считается по байтам, реально попадающим на карту: CPU-тензоры
    // (per_layer_token_embd) в файле есть, но на GPU не едут — без вычета
    // частичная раскладка гибридам вроде qwen4exp не досталась вовсе.
    const uint32_t c = max_ctx(m, less_cpu_extra(m, gpu_w) + extra_gpu_bytes, gpu,
                               cap, kv_q8);
    if (c >= ctx_min && ram_offload_ok(cpu_w, gpu)) {
      s.ok = true;
      s.ncmoe = n;
      s.ctx = c;
      s.gpu_bytes = gpu_w;
      s.cpu_bytes = cpu_w;
      break;
    }
  }
  return s;
}

} // namespace lh::fit
