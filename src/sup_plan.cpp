#include "sup_internal.h"

#include "engine.h"
#include "util.h"

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <thread>

namespace lh::sup::detail {

uint64_t gpu_base_of(const SmartPlan& p, const gguf::Meta& meta, uint64_t fsize) {
  if (p.exps_cpu) return fit::vram_weights_of(meta, fsize);
  if (p.gpu_bytes) return fit::less_cpu_extra(meta, p.gpu_bytes); // сплиты: плотный и MoE-частичный
  return fsize;
}

// ---- Профили замеров: планировщик учится на живых загрузках ----------------
// Аналитический estimate ошибается на ±1 ГиБ (MoE-воркспейс, гибридные арки,
// NVFP4), и излишний консерватизм стоил производительности: qwen36 при
// ctx 65536 теряла до 40% prefill на выселенном в CPU выходном слое. После
// каждой чистой загрузки демон меряет фактический след модели в VRAM
// (NVML-дельта «до спавна / после прогрева» на картах цели) и запоминает
// надбавку сверх весов-файла, аналитического KV и рекуррентного состояния.
// Профиль валиден только для той же формы спавна:
// класс KV, ubatch, «выходной слой на GPU», размещение экспертов (MoE) и
// точное число слоёв на GPU (частичный сплит) — всё, что меняет оверхед.
//
// У MoE с экспертами в RAM и у плотных частичных сплитов след двухчастный:
// VRAM (активная часть + KV + буферы, замер) и RAM (хранилище, детермини-
// рованное — байты файла, живущие не на GPU). RAM-часть не замеряется вовсе:
// mmap-страницы попадают в RSS лишь по мере чтения, так что дельта RSS сразу
// после загрузки занижала бы след; а плавающее при генерации (чтение активных
// экспертов в вычислительные буферы) — за скобками: планировщику нужно
// «влезет ли при загрузке и какой контекст дать», не скорость на запросе.

// Хранилище профилей (model_profiles.json) — src/profiles.{h,cpp}: ключ
// записи «id + форма спавна», у одной модели по слоту на каждую форму.
// Формы полей и семантика замера — там же (lh::profiles::ModelProfile).
using profiles::Form;
using profiles::measured_of;
using profiles::ModelProfile;


// Запас на «пустой карте» при планировании резидентов: помимо десктопа он
// покрывает init-контексты чужих детей — ggml инициализирует CUDA на КАЖДОЙ
// видимой карте, и создание первичного контекста на карте с сотнями свободных
// МиБ падает с OOM (поймано живьём: qwen36 оставляла 661 МиБ — cudaSetDevice
// на соседнюю карту ловил «out of memory»; до отключения fit-пробы туда же
// падала и она). 1,5 ГиБ: у типового резидента с mmproj реальный след на
// ~0,8 ГиБ больше оценки — и всё равно остаётся ~2 ГиБ воздуха. Фит-пробу
// llama.cpp мы выключаем (fit = off в пресете), так что запас этот держим
// сами — он и есть наша замена ихней пробе.
// План, которым модель реально заспавнена (plan_for → лестница):
// из него профиль записи берёт форму и параметры.
std::map<std::string, SmartPlan> g_plan_by_id;
std::mutex g_plan_mtx; // g_plan_by_id: форма спавна на запись профиля и демоцию

// Запись профиля по факту загрузки: след ребёнка в VRAM минус веса (GPU-часть
// файла + draft + mmproj) и аналитический KV. GPU-часть приходит по форме
// спавна той же формулой, что у планировщика (MoE-on-CPU: файл минус точные
// байты экспертов; частичный сплит: слои плана) — надбавка определена относи-
// тельно неё, и расхождение формул стало бы систематикой. RAM-часть (второе
// измерение следа) не замеряется — она детерминирована формой (см. ModelProfile).
// След — дельта «до спавна» (снимок после выселения и settle) —
// «после прогрева»: резиденты за окно загрузки свою память не меняют, так что
// дельта — ровно аллокация этого ребёнка независимо от того, кто уже резидентен.
// Для частичного сплита систематика аналитики весов (lm_head считается на GPU,
// а он на CPU) сокращается сама: планировщик складывает надбавку с той же
// аналитикой весов, что и запись.
namespace {

// Суммировать «<key> = X MiB|GiB» из хвоста журнала ребёнка: llama.cpp печатает
// разбивку памяти загрузки по статьям (model buffer size — веса по устройствам,
// KV buffer size, compute buffer size), по строке на устройство. 0 — строк нет
// (пин не печатает/кольцо уже переехало) — статья просто не логируется.
double sum_buffer_mib(const std::string& tail, const char* key) {
  double sum = 0;
  size_t pos = 0;
  const size_t klen = std::strlen(key);
  while ((pos = tail.find(key, pos)) != std::string::npos) {
    const size_t key0 = pos;
    pos += klen;
    const size_t eq = tail.find('=', pos);
    // «=» сразу после ключа (пробелы): префикс устройства («CUDA0 …»)
    // стоит ДО ключа и в окно не попадает.
    if (eq == std::string::npos || eq > key0 + klen + 4) continue;
    const char* v = tail.c_str() + eq + 1;
    char* end = nullptr;
    const double num = std::strtod(v, &end);
    if (end == v || num <= 0) continue;
    // единица — сразу после числа (MiB по умолчанию; GiB/KiB — множитель)
    const char* u = end;
    while (*u == ' ') ++u;
    if (!strncmp(u, "GiB", 3)) sum += num * 1024;
    else if (!strncmp(u, "KiB", 3)) sum += num / 1024;
    else sum += num; // MiB (и «MB» движка — тоже МИБ)
  }
  return sum;
}

// Журнал точности аналитики по статьям — ТОЛЬКО на первой загрузке формы
// (совет внешнего ревью 10.11): когда профиль уже накоплен, расхождение
// близко к нулю по построению (надбавка снимается из той же дельты) и ничего
// не показывает; по статьям, а не по сумме — компенсирующие ошибки суммы
// прячутся. Скорость падения ошибки на 2-й/3-й загрузке видна из журнала
// записи профиля (строки «замер сохранён» идут на каждую загрузку).
void log_plan_accuracy(const std::string& id, const SmartPlan& p, int layers,
                       uint64_t weights_vram, int64_t delta, const fit::Estimate& ref,
                       int64_t arena, int64_t overhead,
                       const std::string& child_tail) {
  const std::string mib = " МиБ";
  std::string child;
  if (!child_tail.empty()) {
    const double model = sum_buffer_mib(child_tail, "model buffer size");
    const double kv = sum_buffer_mib(child_tail, "KV buffer size");
    const double compute = sum_buffer_mib(child_tail, "compute buffer size");
    if (model > 0 || kv > 0 || compute > 0) {
      char b[160];
      std::snprintf(b, sizeof(b), "журнал движка: веса %.0f, KV %.0f, compute %.0f",
                    model, kv, compute);
      child = "; " + std::string(b) + mib;
    }
  }
  const std::string form =
      (p.ngl0 > 0 && p.ngl0 < layers
           ? "ngl " + std::to_string(p.ngl0) + "/" + std::to_string(layers)
           : p.ngl0 > layers ? "out_gpu" : "out_cpu") +
      (p.exps_cpu ? " exps_ram" : p.ncmoe > 0 ? " ncmoe " + std::to_string(p.ncmoe) : "") +
      (p.kv_q8 ? " q8" : " f16") + (p.moe_cache ? " кэш " + std::to_string(p.moe_cache) : "");
  logf("plan-accuracy: %s: форма [%s] ctx %d: план веса %llu, аналитика KV %llu + "
       "state %llu + арена %lld, оверхед %llu; факт NVML +%lld%s; "
       "надбавка %lld МиБ",
       id.c_str(), form.c_str(), p.ctx0,
       (unsigned long long)(weights_vram >> 20),
       (unsigned long long)(ref.kv >> 20),
       (unsigned long long)(ref.state >> 20),
       (long long)(arena >> 20),
       (unsigned long long)(ref.overhead >> 20),
       (long long)(delta >> 20), child.c_str(), (long long)(overhead >> 20));
}

} // namespace

void record_profile(const std::string& id, const SmartPlan& p, const gguf::Meta& meta,
                    uint64_t weights_vram, uint64_t ram_bytes, const hw::Gpu& before_vram,
                    int card, bool split, bool cuda, int ubatch,
                    const std::string& child_tail) {
  if (!meta.n_layers || p.ctx0 < 512) return;
  const int layers = (int)meta.n_layers;
  const bool partial = p.ngl0 > 0 && p.ngl0 < layers; // плотный частичный сплит
  const bool out_gpu = p.ngl0 > layers;
  const int prof_ngl = partial ? p.ngl0 : 0;
  // Хвост аллокаций и лаг счётчика: снимаем «после» только когда занятое
  // перестало расти (три стабильных полсекунды, потолок 8 с). Иначе замер
  // попадает в середину разгона ребёнка — у gemma-26B запись через 4 с после
  // спавна недосчитала ~3 ГиБ и честно отказалась (надбавка −2,8 ГиБ).
  hw::Gpu after = settle_view(split ? -1 : card);
  if (after.vram_total > 0) {
    uint64_t prev_used = after.vram_total - after.vram_free;
    for (int i = 0, stable = 0; i < 16 && stable < 3; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      hw::Gpu v = settle_view(split ? -1 : card);
      if (v.vram_total != after.vram_total) return; // карты изменились — замер мимо
      const uint64_t used = v.vram_total - v.vram_free;
      if (used == prev_used) {
        ++stable;
      } else {
        stable = 0;
        after = v;
        prev_used = used;
      }
    }
  }
  if (after.vram_total <= 0 || after.vram_total != before_vram.vram_total) return;
  // Честная дельта этого ребёнка: занято «после прогрева» минус занято «до
  // спавна» (снимок будилки, после выселения и settle). Резиденты за
  // окно загрузки свою память не меняют — в дельту не попадают; попадёт лишь
  // растущая аллокация ПАРАЛЛЕЛЬНОЙ загрузки — от неё страхует проверка
  // «никто не грузится» на стороне будилки.
  const int64_t delta = (int64_t)(after.vram_total - after.vram_free) -
                        (int64_t)(before_vram.vram_total - before_vram.vram_free);
  // Аналитический KV и рекуррентное состояние при фактическом контексте и
  // форме (q8/f16 — как в плане). Состояние вычитается тоже: иначе у гибридных
  // арок (qwen35/qwen35moe) оно попадало и в замер, и в аналитику — двойной
  // счёт ~2 ГиБ систематически занижал контекст.
  const fit::Estimate ref = fit::estimate(meta, 0, (uint32_t)p.ctx0, after, p.kv_q8);
  // Арена кэша MoE — плановая аллокация, а не остаточный оверхед: без
  // вычета кэш 8 ГиБ выталкивал надбавку за потолок 8 ГиБ и замер формы
  // молча выбрасывался («след не разошёлся», 29.10) — пара скоростей для
  // вывода локальности не собиралась никогда.
  const int64_t arena = (int64_t)((uint64_t)p.moe_cache << 20);
  const int64_t overhead =
      delta - (int64_t)(weights_vram + ref.kv + ref.state) - arena;
  // Форма уже писалась? После записи расхождение сходится по построению
  // (надбавка снимается из той же дельты) — честная метрика точности только
  // на первой загрузке формы (совет внешнего ревью 10.11).
  const bool first_form = !profiles::lookup(id, {out_gpu, p.kv_q8, ubatch, cuda,
                                                 p.exps_cpu, p.ncmoe, prof_ngl,
                                                 p.moe_cache, p.ot_blocks});
  if (first_form)
    log_plan_accuracy(id, p, layers, weights_vram, delta, ref, arena, overhead,
                      child_tail);
  if (delta < (int64_t)weights_vram) {
    // Ребёнок занял меньше, чем план считал весами, — аналитика переоценила
    // (qwen4exp: план 20 ГиБ против живых 13, пока не починили скан тензоров
    // и учёт гибридности). Молчаливый отказ здесь навсегда блокировал
    // самообучение: без профиля каждая загрузка повторяла переоценку.
    // Пишем профиль с надбавкой-полом: следующий план ближе к живому следу,
    // а расхождение названо числами.
    const int64_t floor =
        (int64_t)fit::fixed_overhead(cuda);
    const int64_t clamped =
        std::max(std::min(overhead, (int64_t)(2ull << 30)), floor);
    logf("profiles: %s: дельта VRAM %lld МиБ меньше плановых весов %llu МиБ — "
         "аналитика завышена на %lld МиБ; пишу профиль с надбавкой-полом %lld МиБ "
         "(проверь скан тензоров и учёт гибридных арок)",
         id.c_str(), (long long)(delta >> 20),
         (unsigned long long)(weights_vram >> 20),
         (long long)(((int64_t)weights_vram - delta) >> 20),
         (long long)(clamped >> 20));
    const int64_t safe = clamped;
    {
      profiles::ModelProfile e;
      e.out_gpu = out_gpu;
      e.cuda = cuda;
      e.kv_q8 = p.kv_q8;
      e.kv_type = p.kv_type;
      e.ubatch = ubatch;
      e.ctx = (uint32_t)p.ctx0;
      e.overhead = (uint64_t)safe;
      e.ram_bytes = ram_bytes;
      e.exps_cpu = p.exps_cpu;
      e.ncmoe = p.ncmoe;
      e.ngl = prof_ngl;
      e.moe_cache = p.moe_cache;
      e.ot_blocks = p.ot_blocks;
      profiles::put(id, {out_gpu, p.kv_q8, ubatch, cuda, p.exps_cpu, p.ncmoe, prof_ngl,
                         p.moe_cache, p.ot_blocks}, e);
    }
    return;
  }
  if (overhead < (int64_t)(64ull << 20) || overhead > (int64_t)(8ull << 30)) {
    logf("profiles: %s: след не разошёлся (надбавка %lld МиБ) — не записываю", id.c_str(),
         (long long)(overhead >> 20));
    return;
  }
  {
    // Слот — «id + форма спавна»: замеры разных форм одной модели живут
    // рядом, а не затирают друг друга. Успешный out_gpu-замер снимает флаг
    // провала амбициозной попытки своей формы (put), демоционная запись
    // (out_cpu) его не трогает — каждая загрузка не повторяет OOM-пробу.
    profiles::ModelProfile e;
    e.out_gpu = out_gpu;
    e.cuda = cuda;
    e.kv_q8 = p.kv_q8;
    e.kv_type = p.kv_type;
    e.ubatch = ubatch;
    e.ctx = (uint32_t)p.ctx0;
    e.overhead = (uint64_t)overhead;
    e.ram_bytes = ram_bytes;
    e.exps_cpu = p.exps_cpu;
    e.ncmoe = p.ncmoe;
    e.ngl = prof_ngl;
    e.moe_cache = p.moe_cache;
    e.ot_blocks = p.ot_blocks;
    profiles::put(id, {out_gpu, p.kv_q8, ubatch, cuda, p.exps_cpu, p.ncmoe, prof_ngl,
                       p.moe_cache, p.ot_blocks}, e);
  }
  std::string form;
  if (partial)
    form = "слоёв на GPU " + std::to_string(p.ngl0) + "/" + std::to_string(layers) + ", ";
  else
    form = out_gpu ? "выходной на GPU, " : "выходной на CPU, ";
  if (p.exps_cpu)
    form = "эксперты в RAM, " + form;
  else if (p.ncmoe > 0)
    form = "экспертов в RAM " + std::to_string(p.ncmoe) + "/" + std::to_string(layers) +
           ", " + form;
  const std::string ram_note =
      ram_bytes ? ", RAM " + std::to_string(ram_bytes >> 30) + " ГиБ" : "";
  logf("profiles: %s: замер сохранён — %sKV %s, ctx %d, ubatch %d: VRAM +%llu МиБ%s",
       id.c_str(), form.c_str(), p.kv_q8 ? (p.kv_type == "q4_0" ? "q4_0" : "q8_0") : "f16",
       p.ctx0, ubatch, (unsigned long long)((uint64_t)overhead >> 20), ram_note.c_str());
}

// Выходной слой (lm_head) едет на GPU только при -ngl > n_layers; оставшись
// на CPU, он стоит ~25% генерации (замер 01.10 на 4070 Ti, qwen3-0.6b при
// равных прочих: ngl 28 = n_layers → 290 ток/с, ngl 29 → 377). В этом режиме
// фактическая VRAM совпадает с estimate (он и так считает весь файл весами),
// так что «влезет» — обычный критерий. Не влезает — оставляем выход на CPU:
// медленнее, но грузится (у ребёнка роутера лестницы OOM нет). Вызывается
// повторно с полным грузом (mmproj, draft) — только понижает, идемпотентно.
void ngl_with_output(SmartPlan& p, const gguf::Meta& meta, uint64_t weights,
                     const hw::Gpu& gpu, bool kv_q8, const std::string& hint,
                     const std::optional<ModelProfile>& prof) {
  if (!meta.n_layers || p.ngl0 <= (int)meta.n_layers) return;
  p.ngl0 = (int)meta.n_layers + 1;
  uint32_t ctx = p.ctx0 > 0 ? (uint32_t)p.ctx0 : 4096;
  // Замер в форме «выходной на GPU» сильнее аналитики — он в ней и получен.
  fit::Measured m = prof ? measured_of(*prof) : fit::Measured{};
  const fit::Measured* mp =
      (prof && prof->out_gpu && !prof->out_gpu_failed && m.overhead > 0) ? &m : nullptr;
  auto est = fit::estimate(meta, weights, ctx, gpu, kv_q8, mp);
  if (!est.fits) {
    // Малый запас — цена грубости аналитики (±1 ГиБ: MoE-воркспейс, гибриды,
    // NVFP4), а не приговор: пробуем выходной на GPU живьём. Провал ловит
    // будилка, записывает в профиль и демотирует — попытка одноразовая.
    const int64_t margin = (int64_t)(est.total - gpu.vram_free);
    if (!prof && margin < (2ll << 30)) {
      p.ambitious_output = true;
      return;
    }
    p.ngl0 = (int)meta.n_layers;
    logf("load: %s: выходной слой остаётся на CPU — весь груз (%.1f ГиБ) с ним не влезает",
         hint.c_str(), weights / 1073741824.0);
  }
}

// Оценка pp-буферов для планов впритык (плотный сплит): в аналитическом
// оверхеде fit::estimate нет реальных вычислительных буферов префилла —
// логиты (ubatch × vocab × 4) и послойный воркспейс. На крупной плотной
// модели это гигабайты: Llama-3.3-70B на пуле 12+24 ГиБ ловила «failed to
// allocate compute pp buffers» (07.10) — план взял на ~3 слоя больше, чем
// влезало с буферами. Рамки 1..3 ГиБ; без vocab в мете — плоский гигабайт.
uint64_t pp_buffer_slack(const config::Config& cfg, const gguf::Meta& meta) {
  uint64_t logits = cfg.ubatch > 0 && meta.n_vocab > 0
                        ? (uint64_t)cfg.ubatch * meta.n_vocab * 4
                        : 512ull << 20;
  return std::min(std::max(logits + (512ull << 20), 1ull << 30), 3ull << 30);
}
// Тот же класс накладных, только для draft-MTP-головы: её compute-буферы
// (логиты ubatch × vocab × 4 и воркспейс MTP-контекста) планом не бюджетиро-
// вались — честно парная крупная голова ловила cudaMalloc OOM уже ПОСЛЕ
// плана, «failed to create MTP context» (a4lg: ~1,8 ГиБ; 07.10). Рамки XS:
// 1..2 ГиБ; без словаря в мете — плоский гигабайт. Для встроенного MTP
// (nextn внутри GGUF ствола) словарь берётся от ствола.
uint64_t draft_buffer_slack(const config::Config& cfg, uint64_t n_vocab) {
  uint64_t logits = cfg.ubatch > 0 && n_vocab > 0
                        ? (uint64_t)cfg.ubatch * n_vocab * 4
                        : 512ull << 20;
  return std::min(std::max(logits + (256ull << 20), 1ull << 30), 2ull << 30);
}

// Локальность a из пары замеренных скоростей форм «с кэшем / без» (обращение
// T(C), модель — sup_internal.h): при равном контексте T_аттн сокращается, и
// в разности мс/токен остаётся единственная неизвестная — покрытие h(s)
// текущего размера кэша. Составы обеих сторон (host-блоки, плоть на CPU)
// выводятся из полей записей профиля. Пары нет, пока одна из форм не
// измерена (проба его закроет) или контексты разошлись. Точка сохраняется
// как измеренная (note_moe_point) — следующие загрузки берут её сразу.
bool locality_from_speeds(const std::string& pid, const gguf::Meta& meta,
                          uint64_t fsize, double& a_out) {
  profiles::CacheProbe cp, bp;
  if (!profiles::moe_pair_of(pid, cp, bp)) return false;
  const uint64_t pool = fit::expert_bytes_of(meta, fsize);
  const double L = (double)meta.n_layers;
  if (!pool || L <= 0 || meta.n_experts == 0) return false;
  const double kE = (double)meta.n_experts_used / (double)meta.n_experts;
  auto host_blocks = [&](const profiles::CacheProbe& q) {
    return std::max(0.0, (q.exps_cpu ? L : (double)q.ncmoe) - (double)q.ot_blocks);
  };
  const double dense_total = (double)fit::vram_weights_of(meta, fsize);
  auto dense_cpu_gib = [&](const profiles::CacheProbe& q) {
    // Плоть, не доехавшая до GPU: ngl в (0, L) — плотный частичный сплит
    // (ngl слоёв на GPU из L+1 с выходным); полные формы — вся плоть на GPU.
    double gpu = dense_total;
    if (q.ngl > 0 && q.ngl < L) gpu = dense_total * ((double)q.ngl / (L + 1.0));
    return std::max(0.0, dense_total - gpu) / 1073741824.0;
  };
  auto host_gib = [&](const profiles::CacheProbe& q) {
    return (double)pool * (host_blocks(q) / L) * kE / 1073741824.0;
  };
  constexpr double kGiBtoGB = 1.0737;
  const double cpu_ms = kGiBtoGB / kMoeCpuGBs * 1000.0;  // мс за ГиБ
  const double up_ms = kGiBtoGB / kMoeUploadGBs * 1000.0;
  // Доля промахов на параллельном CPU (moe-cpu-share) — та же, что в модели
  // score(): она скрыта под GPU-работой и в критическом пути пары «с кэшем»
  // не платит PCIe. Если доля менялась между замером пары и чтением — точка
  // чуть врёт; список форм (профили) перемержится сам при следующей паре.
  const double cpu_share = std::clamp(config::current().moe_cpu_share, 0.0, 0.9);
  const double v_p = 1000.0 / cp.tps, v_b = 1000.0 / bp.tps;
  const double A_p = host_gib(cp), A_b = host_gib(bp);
  // v_p − v_b = hb_p·sync + (1−h)·A_p·up·(1−share) + (D_p − D_b)·cpu − A_b·cpu
  const double up_eff = up_ms * (1.0 - cpu_share);
  const double rhs = (v_p - v_b) - host_blocks(cp) * kMoeSyncUs / 1000.0 -
                     (dense_cpu_gib(cp) - dense_cpu_gib(bp)) * cpu_ms +
                     A_b * cpu_ms;
  const double miss = A_p > 0.0 && up_eff > 0.0 ? rhs / (A_p * up_eff) : 1.0; // = 1 − h
  const double host_pool = (double)pool * (host_blocks(cp) / L);
  const double s = host_pool > 0.0
                       ? (double)((uint64_t)cp.moe_cache << 20) / host_pool : 0.0;
  double a;
  if (miss <= 0.0 || s <= 0.0 || s >= 1.0 || miss >= 1.0) {
    // Непромахнувшаяся арифметика — тоже ответ: кэш даром съедает разницу
    // (a на потолке) либо не окупает вообще (a на полу). Пол — 1.0:
    // a < 1 — анти-локальность, физически невозможна (хуже случайного
    // доступа); точка, дающая a < 1, отравлена кривым замером (30.10:
    // бенч с уникальными промптами дал hit rate ~5% → a упиралось в пол
    // 0.5 и сетка раздавала гигантские гранты).
    a = miss <= 0.0 ? 12.0 : 1.0;
  } else {
    a = std::clamp(std::log(miss) / std::log(1.0 - s), 1.0, 12.0);
  }
  a_out = a;
  const double h = 1.0 - std::pow(1.0 - s, a);
  if (h > 0.0 && h < 1.0) profiles::note_moe_point(pid, cp.moe_cache, h);
  logf("plan: %s: локальность MoE из пары скоростей: a = %.2f (кэш %d МиБ, "
       "покрытие %.0f%%; %.1f против %.1f ток/с при ctx %u)",
       pid.c_str(), a, cp.moe_cache, h * 100.0, cp.tps, bp.tps, cp.ctx);
  return true;
}

SmartPlan plan_for(const gguf::Meta& meta, uint64_t fsize, const hw::Gpu& gpu,
                   const config::Config& cfg, const std::string& hint,
                   uint64_t extra_gpu_bytes, int kv_force,
                   int ctx_pin, bool announce, int cache_exact,
                   int ot_spare_idx, bool force_exps) {
  // Оверхед считается под ЗАПУСКАЕМЫЙ движок: карты из hw::query_all() несут
  // билд по «что может железо» (наличие Vulkan-рантайма — на этой машине
  // Vulkan), а едет CUDA. Без нормализации CUDA-модели планировались с
  // Vulkan-оверхедом и заниженным контекстом.
  hw::Gpu g = gpu;
  const bool cuda_eng = engine::launches_cuda(cfg);
  if (cuda_eng) g.build = hw::Build::Cuda13;
  SmartPlan p;
  p.ngl0 = cfg.ngl;
  p.ctx0 = cfg.ctx;
  p.moe = meta.n_experts > 0;
  // Замеренный профиль (форма «выходной на GPU» — в ней планируется весь
  // файл весами). hint у роутер-путей — id модели; у классики — полный путь,
  // sanitize его не превращает в id → профиль не найдётся, честная аналитика.
  const std::string pid = sanitize_id(hint);
  std::optional<ModelProfile> prof = profiles::lookup(pid, {/*out_gpu=*/true,
                                                            /*kv_q8=*/true, cfg.ubatch, cuda_eng});
  fit::Measured meas = prof ? measured_of(*prof) : fit::Measured{};
  const fit::Measured* mp =
      (prof && prof->out_gpu && !prof->out_gpu_failed && meas.overhead > 0) ? &meas
                                                                           : nullptr;
  if (!cfg.auto_params) {
    // Груз, едущий на GPU целиком (draft, mmproj), — в решение о выходном
    // слое и в ручном режиме: без него план держал выходной на GPU, когда
    // проекция съедала его запас (та же математика, что full_gpu ниже).
    ngl_with_output(p, meta, fsize + extra_gpu_bytes, g, /*kv_q8=*/false, hint,
                    prof);
    return p;
  }
  // Кэш горячих MoE-экспертов в VRAM ([general] moe-cache-mib → движок
  // --moe-cache-mib, пин b11963): сетка авто-режима строит каждого
  // кандидата полным plan_for на карте, срезанной его ареной (кэш живёт
  // на тех же гигабайтах, что слои и эксперты); ручной грант наоборот —
  // форма строится без арены, грант примеряется к готовой форме.
  // Режимы ключа: 0 — выкл; N — вручную (форма без арены, грант — если
  // влез рядом, инцидент 11.11); -1 — авто: планировщик строит
  // два кандидата («максимальный кэш» и «без кэша») и выбирает по прогнозу
  // резидентности: нативные эксперты на GPU весят 1, кэш — kCacheHitYield
  // (измеренный yield попаданий SLRU, 0,62–0,70 по логам 28–29.10). Модель
  // выбора сходится с замерами: 80B — смешанная раскладка (нативные блоки +
  // 12 ГиБ, 30,8→58,7), Flash-Next — отказ (нативные 5 блоков дороже
  // мелкого кэша: 15,9 против 23,3), 30B — максимум кэша (29,5→85,9).
  int cache_req = cfg.moe_cache_mib;
  if (cfg.moe_cache_mib == -1 && p.moe && g.vram_free > kPlanReserve + (2ull << 20)) {
    // АВТО: выбор по аналитической модели T(C) (вывод и константы —
    // sup_internal.h). Кандидаты: «без кэша» (лестница сама выбирает ncmoe
    // против полного выноса), сетка грантов кэша (каждый — полный plan_for,
    // раскладка честно делит карту с ареной); при expert-card = spare — ещё
    // полная форма с блоками экспертов на запасной карте. Оценка —
    // мс/токен host-экспертов и CPU-плотни (T_аттн сокращается). Покрытие
    // h(s) = 1−(1−s)^a: локальность a — из точки профиля либо выводится из
    // пары замеренных скоростей форм «с кэшем / без» (locality_from_speeds);
    // до первой пары — ПРОБА (Prior напрямую доверять нельзя: на Flash-Next
    // наводил на ложный оптимум, оценка 1,9 мс/токен против факта ~120).
    const uint64_t pool = fit::expert_bytes_of(meta, fsize);
    config::Config c2 = cfg;
    c2.moe_cache_mib = 0;
    SmartPlan base = plan_for(meta, fsize, gpu, c2, hint, extra_gpu_bytes,
                              kv_force, ctx_pin, /*announce=*/false);
    // Кандидаты строятся сеткой грантов (см. ниже); карта НЕ трогается
    // здесь — ручная ветка внутри plan_for сама согласованно уменьшит
    // total/free под арену кэша.
    const uint64_t max_grant = gpu.vram_free > kPlanReserve
                                   ? gpu.vram_free - kPlanReserve : 0;
    const uint64_t dense_need = fsize > pool ? fsize - pool : 0;
    const uint64_t pool_cap = pool * 95 / 100;
    // Запасная карта (expert-card = spare): сколько блоков экспертов взяла бы
    // полная форма — тем же правилом, что и спавн (живая свобода минус
    // 768 МиБ), расхождение ловит пост-проверка загрузки.
    int ot_blocks = 0;
    if (ot_spare_idx >= 0 && !meta.expert_bytes_by_block.empty()) {
      hw::flush_telemetry_cache();
      uint64_t spare_free = 0;
      for (const auto& c : hw::query_all())
        if (c.nvidia && c.index == ot_spare_idx) { spare_free = c.vram_free; break; }
      const uint64_t budget =
          spare_free > (768ull << 20) ? spare_free - (768ull << 20) : 0;
      uint64_t moved = 0;
      expert_share_ot(meta, ot_spare_idx, budget, &ot_blocks, &moved);
    }
    auto mk_cand = [&](uint64_t grant, bool ot) -> SmartPlan {
      if (grant < (64ull << 20)) return base;
      grant = std::min(grant, pool_cap);
      c2.moe_cache_mib = (int)(grant >> 20);
      // карту НЕ трогаем: ручная ветка внутри сама согласованно уменьшит
      // total/free под арену кэша — все ветки раскладки увидят одно и то же
      return plan_for(meta, fsize, gpu, c2, hint, extra_gpu_bytes,
                      kv_force, ctx_pin, /*announce=*/false,
                      /*cache_exact=*/(int)(grant >> 20),
                      ot ? ot_spare_idx : -1, /*force_exps=*/ot);
    };
    const double L = (double)meta.n_layers;
    const double kE = meta.n_experts > 0
                          ? (double)meta.n_experts_used / (double)meta.n_experts
                          : 0.0; // доля активных экспертов на токен
    double a = kMoeLocalityPrior;
    int pm = 0;
    double ph = 0.0;
    bool a_measured = profiles::moe_point(pid, pm, ph) && pm > 0 && pool > 0;
    if (a_measured) {
      const double s0 = (double)((uint64_t)pm << 20) / (double)pool;
      if (s0 > 0.0 && s0 < 1.0)
        // Пол 1.0: a < 1 — анти-локальность, физически невозможна; точки,
        // дающие a < 1, отравлены кривым замером (см. locality_from_speeds).
        a = std::clamp(std::log(1.0 - ph) / std::log(1.0 - s0), 1.0, 12.0);
    } else if (locality_from_speeds(pid, meta, fsize, a)) {
      a_measured = true; // точка уже записана — следующие загрузки сразу сеткой
    }
    if (!a_measured) {
      // Локальность не измерена — не отказ, а ПРОБА (как проба KV q8/f16):
      // один запуск с кэшем такого размера, что плотная часть и KV
      // остаются на GPU (скорость не рушится); пара скоростей «проба/база»
      // снимется в профили форм — со следующей загрузки локальность
      // выведется обращением T(C) и сетка найдёт оптимум.
      // размер пробы: плотные + KV остаются на GPU (иначе лестница спавна
      // рушится в dense-partial — поймано 29.10: проба 19,8 ГиБ на 80B
      // дала ngl 12 без кэша), и не больше половины свободного бюджета
      const uint64_t kv_slack = 2ull << 30;
      const uint64_t probe_floor = 2ull << 30;
      uint64_t probe = max_grant > dense_need + kv_slack
                           ? max_grant - dense_need - kv_slack : 0;
      if (probe > max_grant / 2) probe = max_grant / 2;
      if (probe > pool_cap) probe = pool_cap;
      if (probe >= probe_floor) {
        SmartPlan probe_plan = mk_cand(probe, /*ot=*/false);
        if (probe_plan.moe_cache > 0) {
          if (announce)
            logf("plan: %s: кэш MoE авто: локальность не измерена — проба "
                 "%d МиБ (плотные на GPU); скорость пары «проба/база» снимется "
                 "в профили форм, дальше сетка посчитает оптимум",
                 hint.c_str(), probe_plan.moe_cache);
          return probe_plan;
        }
      }
      if (announce)
        logf("plan: %s: кэш MoE авто: под пробу нет места — базовый план",
             hint.c_str());
      return base;
    }
    // Host-блоки (RAM-путь читает их экспертов): полная форма — все L,
    // ncmoe — первые ncmoe, блоки на запасной карте — не host.
    auto host_blocks = [&](const SmartPlan& q) -> double {
      return std::max(0.0, (q.exps_cpu ? L : (double)q.ncmoe) - (double)q.ot_blocks);
    };
    auto host_active_gib = [&](const SmartPlan& q) -> double {
      if (q.fits_all) return 0.0;
      return (double)pool * (host_blocks(q) / L) * kE / 1073741824.0;
    };
    // Плотная часть, не доехавшая на GPU: плоть = файл минус пул минус
    // CPU-тензоры (PLE-таблицы qwen4exp на карту не едут — без вычета
    // «максимум кэша» milо жертвовал их и сравнение врало). Нативные блоки
    // в gpu_bytes сверх плоти не считаются плотью на GPU.
    const uint64_t dense_total = fit::vram_weights_of(meta, fsize);
    auto dense_cpu_gib = [&](const SmartPlan& q) -> double {
      if (q.gpu_bytes >= dense_total) return 0.0;
      return (double)(dense_total - q.gpu_bytes) / 1073741824.0;
    };
    // мс/токен: ГиБ → ГБ (×1.0737) → /ГБ/с → ×1000. Без кэша — только
    // RAM-путь (модель T(C): ни синхронизаций, ни загрузок). Покрытие
    // считается по пулу HOST-слоёв: кэш держит экспертов только тех блоков,
    // что в RAM (нативные/запасные ему не принадлежат).
    // Промахи при moe-cpu-share: доля share движок (#30966) считает на CPU
    // ПАРАЛЛЕЛЬНО с GPU-работой токена — она скрыта под вниманием и
    // попаданиями и с критического пути уходит (калибровка 30.10,
    // Flash-Next, кэш 8 ГиБ, ctx 65k: share 0 → 19,7 ток/с, share 0.6 →
    // 40,3 — стена PCIe (1−h)·A/B_upload снялась почти целиком). Поэтому в
    // пути промаха остаётся только (1−share)·PCIe-часть; без share формула
    // совпадает со старой. Большие модели: у них промах доминирует
    // (десятки мс/токен), и именно завышение промаха заставляло сетку
    // отказывать форме, которая в живом замере вдвое быстрее базы.
    const double cpu_share = std::clamp(cfg.moe_cpu_share, 0.0, 0.9);
    auto score = [&](const SmartPlan& q) -> double {
      const double host_pool =
          (double)pool * (host_blocks(q) / L);
      const double sq = q.moe_cache > 0 && host_pool > 0.0
                            ? (double)((uint64_t)q.moe_cache << 20) / host_pool
                            : 0.0;
      const double hq = 1.0 - std::pow(1.0 - sq, a);
      const double ot_ms = (double)q.ot_blocks * kMoeOtBlockMs;
      if (q.moe_cache <= 0)
        return host_active_gib(q) * 1.0737 / kMoeCpuGBs * 1000.0 +
               dense_cpu_gib(q) * 1.0737 / kMoeCpuGBs * 1000.0 + ot_ms;
      return host_blocks(q) * kMoeSyncUs / 1000.0 +
             host_active_gib(q) * (1.0 - hq) * (1.0 - cpu_share) *
                 1.0737 / kMoeUploadGBs * 1000.0 +
             dense_cpu_gib(q) * 1.0737 / kMoeCpuGBs * 1000.0 + ot_ms;
    };
    // Поиск минимума: база, полная форма с блоками на запасной карте и
    // сетка грантов кэша (±запасная карта) — каждый грант полный план.
    // Две эвристические точки (из-остатка/максимум) не покрывали измеренный
    // оптимум 80B (кэш 12 + плотные на GPU + нативные блоки) — сетка покрывает.
    SmartPlan best = base;
    double best_s = score(base);
    auto consider = [&](const SmartPlan& q) {
      // Форма-пресервация: грант кэша не покупается ценой формы. Кандидат,
      // срезавший слои/контекст или уронивший плоть/выходной слой против
      // базы, мимо — под таким грантом лестница всё равно OOM'ила и
      // катилась по ngl вниз, неся кэш на борту (30.10: 48→24→12).
      if (q.ngl0 < base.ngl0 || q.ctx0 < base.ctx0 ||
          (!base.dense_partial && q.dense_partial) ||
          (!base.ambitious_output && q.ambitious_output))
        return;
      const double sq2 = score(q);
      if (sq2 < best_s - 1e-9) {
        best_s = sq2;
        best = q;
      }
    };
    if (ot_blocks > 0) {
      // Полная форма с блоками на запасной: host-часть меньше на блоки, но
      // каждый блок платит налог layer-split'а (kMoeOtBlockMs — замер
      // 29.10: 130 против 46 мс/токен) — торг решает скорость, не память.
      // c2 могла остаться с грантом пробы — кандидат обязан быть без кэша.
      c2.moe_cache_mib = 0;
      consider(plan_for(meta, fsize, gpu, c2, hint, extra_gpu_bytes, kv_force,
                        ctx_pin, /*announce=*/false, /*cache_exact=*/-2,
                        ot_spare_idx, /*force_exps=*/true));
    }
    for (int i = 1; i <= 8; ++i) {
      const uint64_t grant = max_grant * (uint64_t)(8 - i) / 8;
      if (grant < (512ull << 20)) break; // мелкие грани не строять — база уже есть
      SmartPlan q = mk_cand(grant, /*ot=*/false);
      if (q.moe_cache > 0) consider(q);  // план не смог согласовать кэш — мимо
      if (ot_blocks > 0) {
        SmartPlan q2 = mk_cand(grant, /*ot=*/true);
        if (q2.moe_cache > 0) consider(q2);
      }
    }
    if (announce)
      logf("plan: %s: кэш MoE авто: %s — оценка %.1f мс/токен против базы "
           "%.1f (C=%d МиБ%s; сетка 8 грантов, локальность a=%.2f, пул %.1f ГиБ)",
           hint.c_str(), best.moe_cache > 0 ? "включаю" : "отказываюсь",
           best_s, score(base), best.moe_cache,
           best.ot_blocks > 0 ? (" + " + std::to_string(best.ot_blocks) +
                                 " блоков на запасной карте").c_str()
                              : "",
           a, pool / 1073741824.0);
    // Возвращаем ГОТОВЫЙ план победителя: внешний проход не должен
    // пере-планировать по урезанному бюджету — кандидат строился полным
    // plan_for и уже согласован со своим размером кэша.
    return best;
  }
  if (cache_exact >= 0 && p.moe) {
    // авто-кандидат: размер известен точно; арена кэша «уменьшает карту»
    // для всех внутренних веток (total и free) — иначе раскладка строится
    // под несуществующую память и лестница спавна рушится (поймано 29.10:
    // ngl 6/12 без кэша на Flash-Next и 80B)
    p.moe_cache = cache_exact;
    g.vram_total = g.vram_total > (uint64_t)cache_exact << 20
                       ? g.vram_total - ((uint64_t)cache_exact << 20) : 0;
    g.vram_free = g.vram_free > (uint64_t)cache_exact << 20
                      ? g.vram_free - ((uint64_t)cache_exact << 20) : 0;
  } else if (cache_req > 0 && p.moe && g.vram_free > kPlanReserve + (2ull << 20)) {
    const uint64_t want = (uint64_t)cache_req << 20;
    const uint64_t grant = std::min(want, g.vram_free - kPlanReserve);
    const uint64_t pool = fit::expert_bytes_of(meta, fsize);
    if (ctx_pin >= 256) {
      // Пин контекста стоит — прежнее поведение (доказанно рабочее): арена
      // срезает карту до раскладки, но порезать контекст не может —
      // part_min ncmoe-ветки = пин, у exps-формы пин восстанавливает
      // харнесс; если KV пина не влезает рядом с ареной, гейт «пин × кэш»
      // ниже снимает кэш и возвращает карту. Инцидент 11.11 (ncmoe 46 с
      // ctx 17408) был возможен только без пина — см. ветку ниже.
      p.moe_cache = (int)(grant >> 20);
      g.vram_free -= grant;
      g.vram_total = g.vram_total > grant ? g.vram_total - grant : 0;
      if (pool > 0 && grant * 4 < pool && announce)
        logf("plan: %s: кэш MoE %llu МиБ — меньше четверти пула экспертов "
             "(%.1f ГиБ): по замерам 29.10 малая доля резидентства трэшит "
             "SLRU и снижает скорость — ключ за юзером",
             hint.c_str(), (unsigned long long)(grant >> 20),
             pool / 1073741824.0);
    } else {
      // Без пина арена больше не участвует в выборе формы: форма и
      // контекст строятся БЕЗ арены (полная карта), потом грант
      // примеряется к готовой форме — влез рядом: включён, не влез:
      // снят с честным логом. Прежняя резка карты до раскладки молча
      // продавала контекст за кэш (инцидент 11.11: тот же Flash-Next с
      // тем же грантом, но без пина — ncmoe 46 с ctx 17408 вместо
      // полных 131072; гейт «пин × кэш» проверял уже усечённый контекст
      // и срабатывал тавтологически).
      config::Config c0 = cfg;
      c0.moe_cache_mib = 0;
      SmartPlan base = plan_for(meta, fsize, gpu, c0, hint, extra_gpu_bytes,
                                kv_force, ctx_pin, /*announce=*/false,
                                /*cache_exact=*/-2, ot_spare_idx, force_exps);
      SmartPlan adopt = base;
      bool cache_on = false;
      if (grant >= (64ull << 20) && (base.exps_cpu || base.ncmoe > 0)) {
        const uint32_t eff_ctx = (uint32_t)base.ctx0;
        auto trial = [&](const SmartPlan& f) {
          if (!(f.exps_cpu || f.ncmoe > 0)) return false;
          hw::Gpu gs = gpu;
          gs.vram_total = gs.vram_total > grant ? gs.vram_total - grant : 0;
          gs.vram_free = gs.vram_free > grant ? gs.vram_free - grant : 0;
          const uint64_t w = f.exps_cpu ? fit::vram_weights_of(meta, fsize)
                            : f.ncmoe > 0 ? fit::less_cpu_extra(meta, f.gpu_bytes)
                                          : fsize;
          return fit::estimate(meta, w + extra_gpu_bytes, eff_ctx, gs, f.kv_q8,
                               nullptr).fits;
        };
        cache_on = trial(base);
        if (!cache_on) {
          // Носитель с максимальным выносом: exps-форма оставляет карте
          // максимум места под арену. Кандидат принимается только при
          // контексте не меньше базового — контекст не режется ни в
          // одном из вариантов.
          SmartPlan ex = plan_for(meta, fsize, gpu, c0, hint, extra_gpu_bytes,
                                  kv_force, ctx_pin, /*announce=*/false,
                                  /*cache_exact=*/-2, ot_spare_idx,
                                  /*force_exps=*/true);
          if (ex.exps_cpu && (uint32_t)ex.ctx0 >= eff_ctx && trial(ex)) {
            adopt = ex;
            cache_on = true;
          }
        }
        if (cache_on) {
          adopt.moe_cache = (int)(grant >> 20);
          if (announce)
            logf("plan: %s: грант кэша %llu МиБ уживается с формой (ctx %d) — включён",
                 hint.c_str(), (unsigned long long)(grant >> 20), adopt.ctx0);
        } else if (announce) {
          logf("plan: %s: грант кэша %llu МиБ не влезает рядом с формой (ctx %d) "
               "— кэш снят, форма без него",
               hint.c_str(), (unsigned long long)(grant >> 20), eff_ctx);
        }
      }
      if (cache_on && pool > 0 && grant * 4 < pool && announce)
        logf("plan: %s: кэш MoE %llu МиБ — меньше четверти пула экспертов "
             "(%.1f ГиБ): по замерам 29.10 малая доля резидентства трэшит "
             "SLRU и снижает скорость — ключ за юзером",
             hint.c_str(), (unsigned long long)(grant >> 20),
             pool / 1073741824.0);
      return adopt;
    }
  }
  // kv_force: ручной cache-type из overrides. 1 = «f16» — отключить сжатие
  // (качество), 2 = «q8_0»/«q4_0» — включить принудительно. q4 в математике
  // считается как q8: реальный кэш вдвое меньше оценки, план консервативнее
  // факта — OOM из-за этого не будет. Резолв значения — kv_force_of().
  p.ram_ok = p.moe && fsize > 0 && fit::cpu_offload_ok(meta, fsize, gpu);
  uint32_t cap = meta.n_ctx_train ? (uint32_t)meta.n_ctx_train : 8192u;
  // [general] ctx — ПОЛ плана, не пин (заморозка beta 12, сценарий 1.3):
  // планировщик не опустит контекст ниже, но модель, влезающую в свой
  // максимум, не урежет. Точный ручной контекст — per-model ctx-size
  // (ctx_pin ниже), и тот в бюджете работает как нижняя граница.
  uint32_t ctx_min = cfg.ctx >= 512 ? (uint32_t)cfg.ctx : 4096u;
  // В бюджет контекста входит ВЕСЬ груз на GPU: mmproj и draft едут туда
  // целиком и вне бюджета съедали резерв (qwen36: mmproj 861 МиБ — карта
  // после загрузки держала 661 МиБ свободных, и спавн соседей ловил OOM).
  const uint64_t full_gpu = fsize + extra_gpu_bytes;
  uint32_t auto_ctx =
      full_gpu ? fit::max_ctx(meta, full_gpu, g, cap, /*kv_q8=*/true, mp) : 0;
  p.fits_all = auto_ctx > 0 && fit::estimate(meta, full_gpu, auto_ctx, g, true, mp).fits;
  // KV q8_0 (с FA) сжимает кэш вдвое. Политика сжатия: это ДЕФОЛТ, когда
  // несжатый f16-кэш не дотягивает до полного обучающего контекста модели —
  // сжатие покупает контекст почти даром или с прибылью. Замер 02.10 на
  // qwen36/Blackwell (draft-mtp, карточка равная): короткие ответы q8 ≈ f16
  // (medium 155 против 165, long 179 против 178), а на контексте 21.4k ток
  // q8 УЖЕ БЫСТРЕЕ: декод 212–225 против 197–199 ток/с (+8–13%), префилл
  // 5.25 с против 5.96 с — внимание читает из кэша вдвое меньше байт.
  // Прежняя планка «только если f16 даёт <8k» выросла из замера мелкой модели
  // на коротком контексте (01.10, 0.6B на 4070 Ti: −8%) — нерепрезентативна
  // для рабочих контекстов. f16 остаётся, когда несжатый кэш и так даёт cap:
  // сжимать нечего. Только CUDA: quant-KV едет в связке с FA, Vulkan-путь
  // не замерен — там прежний консерватизм через порог 8k.
  // Форма «эксперты в RAM»: в VRAM остаётся файл минус эксперты (точные байты
  // из тензорного списка GGUF; без него — консервативные 85%) — контекст можно
  // куда смелее, а RAM-гейт проходит без ложных отказов у MoE с долей экспертов
  // выше 85%. Профиль формы свой: полный замер сюда не приложишь — ключ.
  uint64_t gpu_weights = fsize;
  std::optional<ModelProfile> prof_exps;
  fit::Measured meas2{};
  const fit::Measured* mp2 = nullptr;
  const bool exps_form = p.moe && !p.fits_all && p.ram_ok;
  if (exps_form) {
    gpu_weights = fit::vram_weights_of(meta, fsize);
    prof_exps = profiles::lookup(pid, {/*out_gpu=*/true, /*kv_q8=*/true, cfg.ubatch,
                                       cuda_eng, /*exps_cpu=*/true});
    meas2 = prof_exps ? measured_of(*prof_exps) : fit::Measured{};
    mp2 = (prof_exps && meas2.overhead > 0) ? &meas2 : nullptr;
  }
  const fit::Measured* mp_eff = exps_form ? mp2 : mp;
  //
  // Для формы «эксперты в RAM» политика решается по GPU-части весов и обязана
  // решиться ВСЕГДА: полный файл на карту не влезает (auto_ctx = 0), и пропуск
  // блока оставлял kv_q8 = false при контексте, посчитанном по сжатому кэшу, —
  // пресет без cache-type аллоцировал f16-кэш вдвое больше плана, ребёнок ловил
  // честный OOM на пустой карте (поймано живьём, Qwen3-30B на 12 ГиБ).
  if (auto_ctx || exps_form) {
    const uint64_t form_gpu = gpu_weights + extra_gpu_bytes; // та же логика full_gpu
    uint32_t f16_ctx = fit::max_ctx(meta, form_gpu, g, cap, /*kv_q8=*/false, mp_eff);
    p.kv_q8 = engine::launches_cuda(cfg) ? (f16_ctx < cap) : (f16_ctx < 8192);
    if (!p.kv_q8 && f16_ctx) auto_ctx = f16_ctx; // несжатым — если влезает
    if (kv_force == 1) {
      if (f16_ctx) {
        p.kv_q8 = false;
        auto_ctx = f16_ctx;
      } else if (auto_ctx) {
        logf("load: %s: cache-type f16 не влезает ни при каком контексте — остаётся сжатый",
             hint.c_str());
      }
    } else if (kv_force == 2) {
      p.kv_q8 = true; // auto_ctx уже посчитан по сжатому кэшу
      p.kv_type = "q8_0";
    } else if (kv_force == 3) {
      p.kv_q8 = true; // математика по q8: реальный q4-кэш вдвое меньше — OOM не будет
      p.kv_type = "q4_0";
    }
    // Пин контекста пользователем меняет смысл сжатия: q8 покупает контекст,
    // а здесь он уже заказан. Если f16 влезает ровно в пин — сжатие только
    // вредит: gemma-4-26B @32768 декодит на f16 на ~6% быстрее (06.10,
    // back-to-back с A/B на нашем же бинарнике: 118,9 против 112,2 ток/с —
    // вся разница с Ollama 0.35.1, кроме ~3%, была в нашем же q8-кэше);
    // при этом qwen36 @21k в замере 02.10 был быстрее на q8 — зависит от
    // архитектуры, поэтому решаем по влезанию в пин, не по скорости.
    // Ручной cache-type (kv_force) сильнее — он уже отработал выше.
    if (ctx_pin >= 256 && p.kv_q8 && kv_force == 0) {
      if (fit::estimate(meta, form_gpu, (uint32_t)ctx_pin, g, /*kv_q8=*/false,
                        mp_eff).fits) {
        p.kv_q8 = false;
        // Предполёт выселения зовёт plan_for на пустой карте без резерва и
        // сюда же попадает. Вслух говорит только план, которым спавнят.
        if (announce)
          logf("load: %s: контекст пиннут (%d) и f16-кэш в него влезает — сжатие off",
               hint.c_str(), ctx_pin);
      } else if (announce) {
        logf("load: %s: пин контекста %d на f16 не влезает — остаётся сжатый кэш",
             hint.c_str(), ctx_pin);
      }
    }
    // Обе формы уже измерены и одна быстрее хотя бы на 8% — берём её,
    // если она влезает в выбранный контекст. Ручной cache-type сюда не
    // доходит. Короткой выборки недостаточно.
    if (kv_force == 0) {
      const bool out = p.ngl0 > (int)meta.n_layers;
      const int ngl_key =
          (p.ngl0 > 0 && meta.n_layers && p.ngl0 < (int)meta.n_layers) ? p.ngl0 : 0;
      auto take = [&](bool q8, double& tps, int& n) {
        tps = 0;
        n = 0;
        for (bool og : {out, !out}) {
          profiles::Form f{og, q8, cfg.ubatch, cuda_eng, exps_form, p.ncmoe, ngl_key,
                           p.moe_cache};
          double t2 = 0;
          int n2 = 0;
          profiles::speed_of(pid, f, t2, n2);
          if (n2 > n) {
            tps = t2;
            n = n2;
          }
        }
      };
      double tps_q8 = 0, tps_f16 = 0;
      int n_q8 = 0, n_f16 = 0;
      take(true, tps_q8, n_q8);
      take(false, tps_f16, n_f16);
      const bool next = profiles::kv_q8_by_speed(p.kv_q8, tps_q8, n_q8, tps_f16, n_f16);
      if (next != p.kv_q8) {
        const uint32_t ctx =
            (uint32_t)std::max(p.ctx0, ctx_pin >= 256 ? ctx_pin : p.ctx0);
        const bool f16_fits =
            next || fit::estimate(meta, form_gpu, ctx ? ctx : 512u, g, false, mp_eff).fits;
        if (f16_fits) {
          logf("load: %s: форма KV по замеру — %s (%.0f против %.0f ток/с)",
               hint.c_str(), next ? "q8_0" : "f16", next ? tps_q8 : tps_f16,
               next ? tps_f16 : tps_q8);
          p.kv_q8 = next;
          if (!next) p.kv_type.clear();
        }
      }
    }
    if (exps_form) {
      // Контекст формы — по GPU-части с её грузом, в разрядности, которую
      // решил политикой (включая ручной cache-type): полный файл в расчёт
      // не годится.
      auto_ctx = fit::max_ctx(meta, form_gpu, g, cap, p.kv_q8, mp_eff);
    }
  }
  if (auto_ctx && ctx_pin < 256) p.ctx0 = (int)auto_ctx;
  // Пин контекста — план оценивается на НЁМ, не на авто-максимуме: иначе
  // ngl_with_output демотирует выходной слой по невлезанию f16 на 131072,
  // хотя заказанные 32768 влезают с запасом (gemma-4-26B, 06.10).
  p.exps_cpu = exps_form;
  // Плотная модель крупнее VRAM: на CUDA-движке сажаем на GPU столько слоёв,
  // сколько влезает, хвост живёт в RAM. Лестница и так умеет спускаться по
  // ngl, но бисекция стартует с двух верных OOM и садится грубо; у пресета
  // роутера лестницы нет вовсе. Vulkan — не наш эксперимент: сплит не замерен,
  // а WDDM-драйвер на тесной карте ещё и молча спиллит в shared-память.
  if (!p.moe && !p.fits_all && meta.n_layers > 0 && engine::launches_cuda(cfg)) {
    hw::Gpu eng = g;
    // Пресет пишется при произвольном состоянии карты, а загружаться модель
    // будет, когда LRU освободит место, — план считаем под «пустую карту минус
    // запас на десктоп», в той же конвенции, что guard'ы авто-замены.
    if (g.vram_total > 0)
      eng.vram_free = g.vram_total - kPlanReserve; // резерв — потолок, не нижняя граница
    // Плотный сплит планирует впритык — вычитаем оценку pp-буферов
    // (pp_buffer_slack); замеренный профиль формы потом вернёт точность
    // через надбавку (контекст поднимется внутри честного бюджета).
    eng.vram_free -= pp_buffer_slack(cfg, meta);
    // Оверхед — под запускаемый движок: hw::Gpu.build знает лишь «что может
    // железо» (на машине с Vulkan-рантаймом он Vulkan, даже когда едет CUDA).
    eng.build = hw::Build::Cuda13;
    // Ручной контекст — минимум сплита (как у частичного MoE): слоёв на GPU
    // столько, чтобы ЗАКАЗАННЫЙ контекст влезал; без пина — пол от конфига.
    // Иначе сплит посчитал бы слои под максимальный контекст, а вызывающий
    // перезаписал бы ctx0 пином — пресет обещал невлезающее.
    const uint32_t dense_min = ctx_pin >= 256 ? (uint32_t)ctx_pin : ctx_min;
    fit::DenseSplit s = fit::dense_split(meta, fsize, eng, cap, /*kv_q8=*/true, dense_min,
                                         extra_gpu_bytes);
    if (s.ok && fit::ram_offload_ok(s.cpu_bytes, g)) {
      p.ngl0 = s.ngl;
      p.ctx0 = ctx_pin >= 256 ? ctx_pin : (int)s.ctx;
      // Ради объёма: q8-кэш освобождает место слоям. Скорость здесь и так
      // ограничена CPU-частью — ценой ~8% генерации можно пренебречь.
      p.kv_q8 = true;
      p.dense_partial = true;
      p.gpu_bytes = s.gpu_bytes;
      // Профиль «ровно эта форма» (тот же ngl): замеренный оверхед точнее
      // аналитического — контекст может подняться на ступень. Только вверх и
      // только без пина: заказанный контекст — не заявка на максимум.
      std::optional<ModelProfile> pd =
          profiles::lookup(pid, {/*out_gpu=*/false, /*kv_q8=*/true, cfg.ubatch, cuda_eng,
                                 /*exps_cpu=*/false, /*ncmoe=*/0, s.ngl});
      fit::Measured measd = pd ? measured_of(*pd) : fit::Measured{};
      if (pd && measd.overhead > 0 && ctx_pin < 256) {
        if (uint32_t c3 = fit::max_ctx(meta, s.gpu_bytes + extra_gpu_bytes, eng, cap,
                                       true, &measd);
            c3 > s.ctx) {
          p.ctx0 = (int)c3;
          logf("load: %s: профиль сплита поднимает контекст: %u → %u", hint.c_str(),
               s.ctx, c3);
        }
      }
      logf("load: %s: плотная крупнее VRAM — слоёв на GPU %d из %d, контекст %d, "
           "хвост (%.1f ГиБ) в RAM",
           hint.c_str(), s.ngl, (int)meta.n_layers, p.ctx0, s.cpu_bytes / 1073741824.0);
    }
  }
  // MoE крупнее VRAM: сперва частичное размещение экспертов (--n-cpu-moe N —
  // первые N блоков в RAM, остальные на GPU), и лишь если раскладки нет или
  // она выносит всё — прежний полный вынос. Каждый оставшийся на GPU блок —
  // минус его байты в PCIe-трафике на каждый токен; у гибридов вроде
  // Qwen3-Next не-экспертная часть мала, и полный вынос оставлял большую
  // часть карты простаивать. Байты — точные, из тензорного списка; контекст —
  // минимум как у плотного сплита; на Vulkan-движок не ходим (сплит не замерен,
  // а на Windows WDDM ещё и молча спиллит в shared-память). force_exps —
  // запрос авто-ветки на полную форму (кандидат блоков на запасной карте):
  // лестница иначе предпочитает ncmoe, а -ot живёт только в полной форме.
  std::optional<ModelProfile> prof_part;
  if (p.moe && !p.fits_all && meta.n_layers > 0 && engine::launches_cuda(cfg) &&
      !force_exps) {
    hw::Gpu eng = g;
    // Та же конвенция «пустая карта минус запас», что у плотного сплита:
    // загружаться модель будет, когда LRU освободит место, — план считаем
    // под пустую карту, а запас — потолок, не нижняя граница.
    if (g.vram_total > 0) eng.vram_free = g.vram_total - kPlanReserve;
    eng.vram_free -= pp_buffer_slack(cfg, meta);
    eng.build = hw::Build::Cuda13;
    // Пин контекста — минимум расклада не ниже него (не дотягивает — раскладки
    // нет, остаётся полный вынос, у которого своя работа с пином).
    const uint32_t part_min = ctx_pin >= 256 ? (uint32_t)ctx_pin : ctx_min;
    fit::MoESplit ms =
        fit::moe_split(meta, fsize, eng, cap, /*kv_q8=*/true, part_min, extra_gpu_bytes);
    if (ms.ok && ms.ncmoe < (int)meta.n_layers) {
      p.ncmoe = ms.ncmoe;
      p.gpu_bytes = ms.gpu_bytes;
      // Выходной слой решается по GPU-части: CPU-тензоры (per_layer_token_embd)
      // на карту не едут и в частичной форме тоже.
      gpu_weights = fit::less_cpu_extra(meta, ms.gpu_bytes);
      // Ради объёма: q8-кэш освобождает место блокам экспертов на GPU —
      // генерацию здесь ограничивает PCIe-трафик CPU-блоков, а не кэш.
      p.kv_q8 = true;
      p.exps_cpu = false;
      p.ctx0 = ctx_pin >= 256 ? ctx_pin : (int)ms.ctx;
      // Профиль «ровно эта форма» (тот же ncmoe): замеренный оверхед точнее
      // аналитического — контекст может подняться внутри честного бюджета.
      prof_part = profiles::lookup(pid, {/*out_gpu=*/true, /*kv_q8=*/true, cfg.ubatch,
                                         cuda_eng, /*exps_cpu=*/false, p.ncmoe});
      fit::Measured measp = prof_part ? measured_of(*prof_part) : fit::Measured{};
      if (prof_part && measp.overhead > 0) {
        if (uint32_t c3 = fit::max_ctx(meta, ms.gpu_bytes + extra_gpu_bytes, eng, cap,
                                       true, &measp);
            c3 > (uint32_t)p.ctx0) {
          logf("load: %s: профиль сплита MoE поднимает контекст: %d → %u", hint.c_str(),
               p.ctx0, c3);
          if (ctx_pin < 256) p.ctx0 = (int)c3;
        }
      }
      logf("load: %s: MoE крупнее VRAM — блоков экспертов в RAM %d из %d, "
           "контекст %d, GPU-часть %.1f ГиБ, хвост (%.1f ГиБ) в RAM",
           hint.c_str(), ms.ncmoe, (int)meta.n_layers, p.ctx0,
           ms.gpu_bytes / 1073741824.0, ms.cpu_bytes / 1073741824.0);
    }
  }
  // Flash Attention — только носитель KV q8_0: строгий A/B (01.10, один сеанс,
  // 4x перемежающихся прогона) показал одинаковую скорость с ним и без, а
  // CUDA-движок включает FA по умолчанию и без нашего флага.
  p.fa = p.kv_q8;
  ngl_with_output(p, meta, gpu_weights + extra_gpu_bytes, g, p.kv_q8, hint,
                  p.exps_cpu ? prof_exps : (p.ncmoe > 0 ? prof_part : prof));
  // Гейт «пин контекста × арена кэша»: KV(p.ctx0) обязан влезать в карту,
  // срезанную кэшем. Не влезает — кэш роскошь, форма необходимость: снимаем
  // его и восстанавливаем карту (30.10: план выдавал форму ngl 49 /
  // ctx 131072 / кэш 18,7 ГиБ, не помещающуюся в 24 ГиБ, — лестница ловила
  // честный OOM и катилась вниз, унося модель в RAM; пер-модельный пин
  // контекста — нижняя граница и раньше не проверялся на срезанной карте).
  if (p.moe_cache > 0 && p.ctx0 >= 256 &&
      !fit::estimate(meta, gpu_weights + extra_gpu_bytes, (uint32_t)p.ctx0, g,
                     p.kv_q8, nullptr).fits) {
    if (announce)
      logf("plan: %s: пин контекста %d не влезает с кэшем %d МиБ — кэш снят, "
           "карта восстановлена",
           hint.c_str(), p.ctx0, p.moe_cache);
    p.moe_cache = 0;
    g = gpu;
  }
  // Кэш имеет смысл только при вынесенных экспертах: всё в VRAM — кэшировать
  // нечего. Флаг в спавн едет с той же условностью (try_attempt).
  if (!(p.exps_cpu || p.ncmoe > 0)) p.moe_cache = 0;
  // Блоки экспертов на запасной карте — прогноз той же полной формы: сколько
  // блоков унесло бы -ot-правило спавна при текущей живой свободе запасной.
  if (ot_spare_idx >= 0 && p.exps_cpu && !meta.expert_bytes_by_block.empty()) {
    uint64_t spare_free = 0;
    for (const auto& c : hw::query_all())
      if (c.nvidia && c.index == ot_spare_idx) { spare_free = c.vram_free; break; }
    const uint64_t budget =
        spare_free > (768ull << 20) ? spare_free - (768ull << 20) : 0;
    uint64_t moved = 0;
    expert_share_ot(meta, ot_spare_idx, budget, &p.ot_blocks, &moved);
  }
  return p;
}

// Потянет ли карта плотную модель частично (хвост слоёв в RAM) — для guard'ов
// авто-замены: они решают до выгрузки, когда карта занята, поэтому считают по
// «вся карта минус запас на десктоп» — той же конвенции, что и старый отказ
// «файл + 512 МиБ > VRAM». Гейт тот же, что в plan_for: только CUDA-движок.
// out — сам сплит (сколько слоёв и байт весов на GPU планировано); extra_gpu —
// груз, который едет на GPU целиком (draft, mmproj), как в plan_for;
// ctx_pin — ручной контекст модели: guard обязан обещать то же, что и план.
bool dense_partial_ok(const gguf::Meta& meta, uint64_t fsize, const hw::Gpu& card,
                      const config::Config& cfg, fit::DenseSplit* out,
                      uint64_t extra_gpu_bytes, int ctx_pin) {
  if (meta.n_layers == 0 || !fsize || !card.vram_total) return false;
  if (!engine::launches_cuda(cfg)) return false;
  hw::Gpu after = card;
  after.vram_free = card.vram_total - kPlanReserve; // строго с резервом
  after.build = hw::Build::Cuda13;
  // Тот же слот под pp-буферы, что у плана (pp_buffer_slack): guard решает
  // «грузится ли модель частично» по той же математике, которой plan_for
  // считает реальный сплит, — иначе предполёт обещает больше, чем план.
  after.vram_free -= pp_buffer_slack(cfg, meta);
  const uint32_t floor_ctx = cfg.ctx >= 512 ? (uint32_t)cfg.ctx : 4096u;
  fit::DenseSplit s = fit::dense_split(
      meta, fsize, after, meta.n_ctx_train ? (uint32_t)meta.n_ctx_train : 8192u,
      /*kv_q8=*/true, ctx_pin >= 256 ? (uint32_t)ctx_pin : floor_ctx, extra_gpu_bytes);
  if (out) *out = s;
  return s.ok && fit::ram_offload_ok(s.cpu_bytes, card);
}

// id попадает в INI-заголовок секции: имя файла на NTFS может содержать «]»,
// «[» и даже переводы строк — ломали бы пресет (инъекция секций). Безопасный
// ASCII, остальное — в «_».
Preflight preflight_vram(const gguf::Meta* meta, uint64_t fsize, const hw::Gpu& card,
                         const config::Config& cfg, uint64_t extra_gpu_bytes, int ctx_pin) {
  Preflight out;
  if (!meta) return out;
  out.moe_on_cpu = meta->n_experts > 0 && fit::cpu_offload_ok(*meta, fsize, card);
  out.over_vram = fsize && card.vram_total > 0 && fsize + (512ull << 20) > card.vram_total;
  out.dense_ok = out.over_vram && !out.moe_on_cpu &&
                 dense_partial_ok(*meta, fsize, card, cfg, nullptr, extra_gpu_bytes, ctx_pin);
  out.refused = out.over_vram && !out.moe_on_cpu && !out.dense_ok;
  return out;
}

std::string sanitize_id(const std::string& base) {
  std::string id;
  id.reserve(base.size());
  for (char c : base)
    id += (isalnum((unsigned char)c) || c == '.' || c == '-' || c == '_') ? c : '_';
  return id.empty() ? "model" : id;
}

} // namespace lh::sup::detail

namespace lh::sup {

// Тянет ли карта плотную модель частично (CUDA-движок) с экстра-грузом
// extra_bytes (например, MTP-модулем): планировщик подвинет слои, модуль у
// него в бюджете. Для диалога MTP — свободной VRAM «в лоб» может не быть.
// Живёт в sup_plan: это тот же предполёт, что у плотного сплита в plan_for,
// только наружу — для guard'ов и диалогов.
bool dense_partial_fits(const std::string& path, uint64_t extra_bytes) {
  config::Config cfg = config::current();
  gguf::Meta meta;
  if (!lh::modelreg::model_meta(path, meta)) return false;
  // Шард-группа — одна модель: в веса входит сумма частей, не только
  // первая, которую открывает движок.
  const uint64_t fsize = lh::modelreg::shard_group_bytes(path);
  if (!fsize) return false;
  // Пин контекста главной модели — тот же ключ переопределений, что и у
  // классического пути (санитизированное имя файла): guard MTP-головы обязан
  // обещать то, что план с пином реально сосчитает.
  int pin = 0;
  {
    auto ov = detail::load_overrides();
    std::string base = path;
    size_t slash = base.find_last_of("/\\");
    if (slash != std::string::npos) base = base.substr(slash + 1);
    auto it = ov.find(detail::sanitize_id(base));
    if (it != ov.end()) pin = it->second.ctx_size;
  }
  return detail::dense_partial_ok(meta, fsize, hw::query(), cfg, nullptr, extra_bytes, pin);
}

// Предупреждение для ручного контекста: непустая строка — «такой ctx
// исключает vision». Проекция, которая прицепится сейчас (vision-mode
// учтён), не помещается на КАРТУ ЦЕЛИ при заказанном контексте, хотя без
// неё помещается. Карта — как у пресета (gpu_target_for, «пустая карта
// минус kPlanReserve»: модель к установке ctx может спать); аналитика —
// та же, что у plan_for. MoE-формы с экспертами в RAM не проверяем:
// аналитика полного файла там заведомо врёт, раскладку решает лестница.
// Окно честно узкое (проекция стоит kv-байт на токен), но внутри него
// ctx «работает без vision и ломается с ним» — именно это и предупреждаем.
std::string vision_ctx_note(const std::string& model_path, int ctx) {
  if (ctx < 256) return {};
  gguf::Meta meta;
  if (!lh::modelreg::model_meta(model_path, meta) || !meta.n_layers) return {};
  const std::string mm = vision_mmproj_for_path(model_path);
  if (mm.empty()) return {}; // vision-mode off или проекции нет — не о чем
  const uint64_t fsize = lh::modelreg::shard_group_bytes(model_path);
  if (!fsize) return {};
  const config::Config cfg = config::current();
  auto ov = detail::load_overrides();
  std::string base = model_path;
  size_t slash = base.find_last_of("/\\");
  if (slash != std::string::npos) base = base.substr(slash + 1);
  auto it = ov.find(detail::sanitize_id(base));
  const detail::ModelOverrides* o = it == ov.end() ? nullptr : &it->second;
  const detail::GpuTarget tgt = detail::gpu_target_for(detail::sanitize_id(base), model_path,
                                                       cfg, ov, /*live=*/false);
  if (tgt.card.vram_total <= detail::kPlanReserve) return {};
  hw::Gpu g = tgt.card;
  g.vram_free = g.vram_total - detail::kPlanReserve; // пресетная конвенция «пустая карта цели»
  const uint64_t mm_size = lh::file_size(mm);
  // И без проекции тесно — контекст не про vision, раскладку ловит лестница.
  if (!fit::estimate(meta, fsize, (uint32_t)ctx, g).fits) return {};
  // Проекция дышит при этом контексте — предупреждения нет.
  if (fit::estimate(meta, fsize + mm_size, (uint32_t)ctx, g).fits) return {};
  uint32_t cap = meta.n_ctx_train ? (uint32_t)meta.n_ctx_train : 8192u;
  const uint32_t vmax = fit::max_ctx(meta, fsize + mm_size, g, cap);
  std::string note = "контекст " + std::to_string(ctx) + ": с проекцией «" +
                     mm.substr(mm.find_last_of("/\\") + 1) + "» (" +
                     std::to_string(mm_size >> 20) +
                     " МиБ) модель не помещается — vision недоступен";
  if (vmax > 0)
    note += "; для vision контекст до ~" + std::to_string(vmax) + " токенов";
  note += "; либо lhctl vision <модель> off";
  return note;
}

} // namespace lh::sup
