// «Влезет ли модель в видеопамять»: прикидка до загрузки.
// Веса + KV-кэш + рекуррентное состояние гибридных арок + оверхед бэкенда
// (400 МиБ CUDA / 90 МиБ Vulkan) + вычислительные буферы.
#pragma once
#include "gguf.h"
#include "hw.h"

#include <cstdint>
#include <string>

namespace lh::fit {

// Профиль памяти архитектуры: не у всех арок токенный KV-кэш лежит на каждом
// слое. Гибриды (Qwen3-Next-стиль) держат полный KV лишь на части слоёв,
// остальные живут в рекуррентном состоянии ПОСТОЯННОГО размера (не растёт
// с контекстом). Калибровка по живым логам llama.cpp (02.10, b11344, FA):
// qwen35-27B @65536 q8 — KV 2176 MiB на 16 из 64 слоёв (паттерн 1:3),
// RS state 2394 MiB f32, compute 400 MiB. State отнесён к РЕКУРРЕНТНОМУ слою:
// 2394 МиБ / (64 слоя × embd) = 7500 байт — среднее по всем; на рекуррентный
// (48 из 64, 3 из 4) приходится 7500 × 4/3 = 10000 байт × embd.
// Неизвестным архам — полная консервативная математика (1.0 / 0).
struct ArchProfile {
  double kv_frac = 1.0;       // доля слоёв с токенным KV-кэшем (1/интервал)
  uint64_t state_per_le = 0;  // байт рекуррентного состояния на рекуррентный
                              // слой × n_embd
};

inline ArchProfile arch_profile(const std::string& arch) {
  // qwen35 (Qwen3-Next-стиль) и MoE-вариант qwen35moe (Qwen3.6-35B-A3B) —
  // одна гибридная семья: токенный KV на четверти слоёв + рекуррентное
  // состояние постоянного размера. qwen3next — та же архитектура под
  // именем из unsloth-GGUF (Qwen3-Next-80B-A3B: 48 слоёв, 3:1 линейные/полные).
  if (arch == "qwen35" || arch == "qwen35moe" || arch == "qwen3next")
    return {0.25, 10000};
  return {};
}

// Замеренная на живой загрузке надбавка модели (compute + рекуррентное
// состояние + прочее, всё кроме весов и токенного KV) при контексте ctx.
// Планировщик подставляет её вместо аналитического оверхеда — аналитика
// ошибается на ±1 ГиБ (MoE-воркспейс, гибридные арки, NVFP4), замер точен.
struct Measured {
  int64_t overhead = 0;  // байт VRAM сверх весов-файла и аналитического KV
  uint32_t ctx = 0;      // контекст, при котором замерено
  bool cuda = true;      // класс движка замера (ramp растёт по-разному)
};

struct Estimate {
  uint64_t weights = 0;  // весь файл GGUF — это практически веса
  uint64_t kv = 0;       // токенный KV-кэш при данном контексте и разрядности
  uint64_t state = 0;    // рекуррентное состояние гибридных арок (от ctx не зависит)
  uint64_t overhead = 0; // бэкенд + вычислительные буферы
  uint64_t total = 0;
  bool fits = false;     // целиком в свободную VRAM
};

Estimate estimate(const gguf::Meta& m, uint64_t file_size, uint32_t ctx, const hw::Gpu& gpu,
                  bool kv_q8 = false, const Measured* measured = nullptr);

// Наибольший контекст, который влезает (степени двойки вниз от cap); 0 — не влезает и минимум.
uint32_t max_ctx(const gguf::Meta& m, uint64_t file_size, const hw::Gpu& gpu, uint32_t cap,
                 bool kv_q8 = false, const Measured* measured = nullptr);

// MoE: сколько веса уедет на CPU, если вынести экспертов (-ot exps=CPU).
// Тензоры экспертов — подавляющая часть файла; берём консервативные 85%.
inline uint64_t expert_bytes_estimate(uint64_t file_size) {
  return file_size / 100 * 85;
}

// Точные байты экспертов: из тензорного списка GGUF (gguf::Meta::expert_bytes),
// без него (урезанная выборка, битый список, не MoE) — консервативная оценка
// 85%. Значение обязано СОВПАДАТЬ при планировании и при записи профиля:
// надбавка замера определена как след минус эта GPU-часть весов, и расхождение
// формул стало бы систематической ошибкой в обе стороны.
inline uint64_t expert_bytes_of(const gguf::Meta& m, uint64_t file_size) {
  if (m.expert_bytes > 0 && m.expert_bytes < file_size) return m.expert_bytes;
  return expert_bytes_estimate(file_size);
}

// GPU-часть файла при частичном сплите --n-cpu-moe N: файл минус экспертные
// тензоры первых N блоков (точные байты из expert_bytes_by_block — у
// first-k-dense семей вроде GLM/DeepSeek первые блоки плотные, равномерное
// деление на слои было бы оптимистичным). Та же формула при планировании и
// при записи профиля — расхождение стало бы систематикой замера. Без
// поблочных байт (вектор пуст) частичный сплит не планируется вовсе.
inline uint64_t moe_gpu_bytes(const gguf::Meta& m, uint64_t file_size, int ncmoe) {
  uint64_t off = 0;
  const size_t n = std::min<size_t>((size_t)std::max(ncmoe, 0),
                                    m.expert_bytes_by_block.size());
  for (size_t i = 0; i < n; ++i) off += m.expert_bytes_by_block[i];
  return file_size > off ? file_size - off : 0;
}

// Вычесть CPU-тензоры (per_layer_token_embd qwen4exp) из GPU-части: они в
// файле есть, но на карту движок их не берёт.
inline uint64_t less_cpu_extra(const gguf::Meta& m, uint64_t gpu_bytes) {
  return gpu_bytes > m.cpu_extra_bytes ? gpu_bytes - m.cpu_extra_bytes : gpu_bytes;
}

// Веса, которые при форме «эксперты в RAM» реально окажутся в видеопамяти:
// файл минус эксперты минус CPU-тензоры. Та же формула при планировании и
// при записи профиля — расхождение стало бы систематикой замера.
inline uint64_t vram_weights_of(const gguf::Meta& m, uint64_t file_size) {
  const uint64_t w = file_size > expert_bytes_of(m, file_size)
                         ? file_size - expert_bytes_of(m, file_size)
                         : 0;
  return less_cpu_extra(m, w);
}

// Пригодна ли система для оффлоада в оперативку вообще: байты CPU-части должны
// вмещаться с запасом на систему и подкачку. Нет запаса — «нечего соваться».
inline bool ram_offload_ok(uint64_t cpu_bytes, const hw::Gpu& gpu) {
  if (!gpu.ram_total) return false;
  const uint64_t os_headroom = 4ull << 30;
  if (gpu.ram_total <= os_headroom) return false;
  return cpu_bytes <= gpu.ram_total - os_headroom;
}

// Система пригодна для MoE-on-CPU: эксперты — подавляющая часть файла.
inline bool cpu_offload_ok(uint64_t file_size, const hw::Gpu& gpu) {
  return ram_offload_ok(expert_bytes_estimate(file_size), gpu);
}

// То же по точным байтам экспертов: оценка 85% иногда пропускала модель,
// чьи эксперты не лезут (у большинства MoE их доля выше 85%), и резала
// контекст тем, у кого ниже. Заголовок есть — считаем честно.
inline bool cpu_offload_ok(const gguf::Meta& m, uint64_t file_size, const hw::Gpu& gpu) {
  return ram_offload_ok(expert_bytes_of(m, file_size), gpu);
}

// Минимум слоёв плотной модели на GPU, ради которого стоит затевать частичный
// оффлоад: ниже — почти вся модель считается на CPU, честнее отказать.
inline int dense_min_layers(int n_layers) { return n_layers >= 16 ? n_layers / 4 : 4; }

// Фиксированный оверхед бэкенда на карту — та же константа, что в estimate();
// нужна наружу, чтобы пул карт резервировал её НА КАЖДУЮ.
inline uint64_t fixed_overhead(bool cuda) { return (cuda ? 400ull : 90ull) << 20; }

// Синтетическая «карта-пул» из всех планируемых устройств: сумма свободной и
// общей памяти минус оверхед всех карт, кроме первой (estimate() добавит его
// единожды — итого оверхед учтён на каждую). Участвуют карты с телеметрией
// (nvidia && vram_total > 0): по картам без неё планировать нечестно.
// Одна карта — копия с нормализованным билдом; пусто на входе — пустой Gpu
// (vram_free = 0 — всё «мимо», как и без телеметрии).
hw::Gpu pool(const std::vector<hw::Gpu>& cards, hw::Build build);

struct DenseSplit {
  bool ok = false;
  int ngl = 0;       // слоёв на GPU
  uint32_t ctx = 0;  // наибольший контекст, влезающий при этом ngl
  uint64_t gpu_bytes = 0, cpu_bytes = 0;
};

// Плотная модель крупнее VRAM: максимум слоёв на GPU при контексте не меньше
// ctx_min. Перебор от «все слои» вниз; байт на слой считается как file_size /
// n_layers — эмбеддинг и lm_head в нём растворены, и оценка выходит
// консервативной: llama.cpp при ngl ≤ n_layers держит lm_head на CPU, так что
// реальная VRAM ниже нашей прикидки. extra_gpu_bytes — груз, который едет на
// GPU целиком при любом ngl (mmproj, draft): он не скейлится слоями, но участвует
// в бюджете. gpu.build должен соответствовать запускаемому движку (оверхед CUDA
// втрое выше Vulkan-овского).
DenseSplit dense_split(const gguf::Meta& m, uint64_t file_size, const hw::Gpu& gpu,
                       uint32_t cap, bool kv_q8, uint32_t ctx_min,
                       uint64_t extra_gpu_bytes = 0);

struct MoESplit {
  bool ok = false;
  int ncmoe = 0;     // первые N блоков экспертов — в RAM (--n-cpu-moe N)
  uint32_t ctx = 0;  // наибольший контекст, влезающий при этом раскладе
  uint64_t gpu_bytes = 0, cpu_bytes = 0;
};

// MoE крупнее VRAM: МИНИМАЛЬНОЕ число блоков с экспертами в RAM (--n-cpu-moe N),
// при котором контекст не ниже ctx_min, — середина между «всё на GPU» и полным
// выносом; каждый невынесенный блок — меньше байт через PCIe на токен. N=0 не
// рассматривается как успех вызывающими (гейт — «модель не влезает»), N=n_layers
// означал бы полный вынос — вызывающие предпочитают ему форму exps_cpu (та же
// математика плюс nextn-тензоры). Байты выноса — точные, из expert_bytes_by_block
// (пуст вектор — честный отказ: без него раскладка — догадка). Без Measured:
// профиль формы подтягивается вызывающим после, как у dense_split.
MoESplit moe_split(const gguf::Meta& m, uint64_t file_size, const hw::Gpu& gpu,
                   uint32_t cap, bool kv_q8, uint32_t ctx_min,
                   uint64_t extra_gpu_bytes = 0);

} // namespace lh::fit
