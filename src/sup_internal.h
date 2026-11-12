// Внутренний шов супервизора. Видят только cpp ядра демона.
// Публичный контракт — supervisor.h; proxy, control и daemond его не включают.
#pragma once

#include "supervisor.h"

#include "config.h"
#include "fit.h"
#include "gguf.h"
#include "hw.h"
#include "modelreg.h"
#include "proc.h"
#include "park.h"
#include "profiles.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <set>
#include <map>
#include <mutex>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lh::sup {

void apply_registry_change_locked();
void start_idle_watchdog();
void stop_idle_watchdog();
void start_watcher();
void stop_watcher();
void start_acceptance_guard();
void stop_acceptance_guard();

// Image-слот (sup_image.cpp): сторож простоя и останов при выходе — из
// sup::init/sup::shutdown.
void image_init();
void image_shutdown();

}

namespace lh::sup::detail {

inline void changed() {}

using clock_ = std::chrono::steady_clock;

// Общая математика умного подбора: план и лестница попыток.
struct SmartPlan {
  int ctx0 = 0, ngl0 = 0;
  int np = 1;   // слоты (роутер); контекст — общий бюджет, делится на слоты
  bool fits_all = false, moe = false, ram_ok = false, kv_q8 = false, exps_cpu = false;
  int ncmoe = 0;  // MoE частично: --n-cpu-moe N (первые N блоков экспертов в RAM)
  // Кэш горячих MoE-экспертов в VRAM (--moe-cache-mib, пин b11963): сколько
  // МиБ план реально выделил (урезано бюджетом карты; 0 — выключен: выноса
  // экспертов нет или ключ пуст). Бюджет урезается ДО планирования — кэш
  // живёт на тех же гигабайтах, что слои и эксперты.
  int moe_cache = 0;
  // Полная форма выноса + expert-card = spare: сколько блоков экспертов
  // планировано на запасную карту. Само -ot-правило спавн строит по живой
  // свободе запасной (expert_share_ot); здесь — прогноз для скоринга: эти
  // блоки не host (RAM-путь их не читает) и не сидят в карте ствола.
  int ot_blocks = 0;
  bool fa = false; // Flash Attention (на CUDA всегда, на Vulkan — под KV q8_0)
  // Разрядность сжатого кэша: «q8_0» (умная настройка) или «q4_0» (ручной
  // cache-type). Пусто — q8_0. Математика плана всегда по q8: реальный q4-кэш
  // вдвое меньше оценки, план консервативнее факта.
  std::string kv_type;
  // Выходной слой оставлен на GPU против аналитического отказа (запас мал,
  // замеренного профиля нет — пробуем живьём; неудачу демотирует будилка).
  bool ambitious_output = false;
  bool dense_partial = false; // плотная частично: ngl0 < n_layers, хвост слоёв в RAM
  uint64_t gpu_bytes = 0;     // сколько файла остаётся на GPU (сплит: плотный или
                              // MoE-частичный; пресет роутера считает по нему)
};

// Сколько файла реально живёт на GPU при плане p: полный файл, файл минус

// Запас на пустой карте при планировании резидентов. Одна константа на план,
// пресет и guard авто-замены.
constexpr uint64_t kPlanReserve = 1536ull << 20;

// Аналитическая модель кэша MoE (авто-режим moe-cache-mib = -1). Выведена
// из механики пина b11963 (llama-moe-cache.cpp: промах — загрузка в слот
// через copy-callback с хост-синхронизацией на каждый слой с кэшем;
// префилл кэш обходит) и калибрована по замерам 28–29.10:
//   T(токена) = T_аттн + L·T_sync + (1−h(s))·A/B_upload   (с кэшем)
//   T(токена) = T_аттн +            A_base/B_cpu          (без кэша)
// h(s) = 1−(1−s)^a — покрытие маршрутизации (локальность повторных
// расстояний), a — свойство модели. Движок hit-rate строку при остановке
// не печатает (проверено standalone 29.10) — a выводится обращением T(C)
// из пары замеренных скоростей форм «с кэшем / без» (равный контекст,
// T_аттн сокращается; sup_plan, авто-ветка); prior 5.0 — замер 30B-A3B
// (h≈0,9 при s=0,37) — только старт, на Flash-Next наводил на ложный
// оптимум (оценка 1,9 мс/токен, факт ~120). Константы путей:
constexpr double kMoeLocalityPrior = 5.0; // старт, пока пары скоростей нет
constexpr double kMoeUploadGBs = 25.0;    // путь промаха с кэшем: PCIe-загрузка
constexpr double kMoeCpuGBs = 35.0;       // путь без кэша: CPU-вычисление из RAM
constexpr double kMoeSyncUs = 80.0;       // хост-синхронизация на слой с кэшем
constexpr double kMoeOtBlockMs = 16.0;    // блок экспертов на запасной карте:
                                         // налог layer-split'а движка (KV-полоса и
                                         // буферы чужих слоёв на медленной карте),
                                         // Flash-Next + 3060/x4, 29.10: 130 против
                                         // 46 мс/токен при 5 блоках — против
                                         // 0,9 мс/блок экономии RAM-пути

// План, которым модель реально заспавнена. Профиль и демоция читают его под замком.
extern std::map<std::string, SmartPlan> g_plan_by_id;
extern std::mutex g_plan_mtx;

uint64_t gpu_base_of(const SmartPlan& p, const gguf::Meta& meta, uint64_t fsize);

void record_profile(const std::string& id, const SmartPlan& p, const gguf::Meta& meta,
                    uint64_t weights_vram, uint64_t ram_bytes, const hw::Gpu& before_vram,
                    int card, bool split, bool cuda, int ubatch,
                    const std::string& child_tail = {});

void ngl_with_output(SmartPlan& p, const gguf::Meta& meta, uint64_t weights,
                     const hw::Gpu& gpu, bool kv_q8, const std::string& hint,
                     const std::optional<profiles::ModelProfile>& prof = std::nullopt);

uint64_t pp_buffer_slack(const config::Config& cfg, const gguf::Meta& meta);
uint64_t draft_buffer_slack(const config::Config& cfg, uint64_t n_vocab);

SmartPlan plan_for(const gguf::Meta& meta, uint64_t fsize, const hw::Gpu& gpu,
                   const config::Config& cfg, const std::string& hint = {},
                   uint64_t extra_gpu_bytes = 0, int kv_force = 0,
                   int ctx_pin = 0, bool announce = true,
                   // точный размер кэша МиБ для авто-кандидата: ручная ветка
                   // берёт его как есть (карта уже урезана вызывающим), -2 —
                   // обычный путь через cfg.moe_cache_mib
                   int cache_exact = -2,
                   // запасная карта под блоки экспертов (-ot-кандидат сетки
                   // авто-режима): спавн строит правило по живой свободе тем
                   // же правилом (свободна − 768 МиБ), план предсказывает
                   int ot_spare_idx = -1,
                   // построить именно полную форму выноса (внутреннее — для
                   // -ot-кандидата; лестница иначе предпочитает ncmoe)
                   bool force_exps = false);

bool dense_partial_ok(const gguf::Meta& meta, uint64_t fsize, const hw::Gpu& card,
                      const config::Config& cfg, fit::DenseSplit* out = nullptr,
                      uint64_t extra_gpu_bytes = 0, int ctx_pin = 0);

// Предполёт «заведомо не влезающую даже на пустую цель не начинаем» — общая
// математика load-путей (load_harness, каталог, bench-гейты). Запасные ярусы:
// MoE с экспертами в RAM и (на CUDA) плотная с хвостом слоёв в RAM.
// meta == nullptr (меты нет) — судить нечего, отказ невозможен.
struct Preflight {
  bool moe_on_cpu = false; // эксперты влезают в RAM — запасной ярус
  bool over_vram = false;  // файл + 512 МиБ оверхеда больше карты цели
  bool dense_ok = false;   // плотная частично — запасной ярус
  bool refused = false;    // over_vram и ни одного запасного яруса
};
Preflight preflight_vram(const gguf::Meta* meta, uint64_t fsize, const hw::Gpu& card,
                         const config::Config& cfg, uint64_t extra_gpu_bytes, int ctx_pin);

// Приговор draft-модели после падения ребёнка со спекуляцией (рамки —
// 502-буря 07.10): чистый OOM ствола draft не чернит (с вернувшейся VRAM он
// должен жить), но снимается с этой загрузки (суспензия); несовместимость
// чернит до рестарта демона.
struct DraftDemotion {
  bool dead = false;    // уже был в чёрном списке
  bool blamed = false;  // падение по вине draft (looks_draft_failure)
  bool oom = false;     // OOM в логе ребёнка
};
DraftDemotion demote_draft(const std::string& draft_path, const std::string& child_log);

// Репутация draft'ов из model_profiles.json — в g_dead_drafts/g_shy_drafts.
// Зовёт sup::init() после profiles::load(): чёрные списки переживают рестарт.
void seed_draft_reputation();

std::string sanitize_id(const std::string& base);

// Определение в supervisor.cpp: сама функция ещё в анонимном namespace.
// Отсюда её зовёт record_profile, пока спавн не переехал.
hw::Gpu settle_view(int card);


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
  int moe_cache_mib = 0;    // 0 — решает [general] moe-cache-mib; N — кэш этой
                            // модели вручную (МиБ; оптимум у моделей разный:
                            // Flash-Next 8192, Next-80B 10752 — замер 30.10)
};

std::string overrides_conf_path();
std::string overrides_path();
int kv_force_of(const ModelOverrides* o, const std::string& hint);
std::map<std::string, ModelOverrides> load_overrides();
std::string override_section_of(const std::string& model_path);
// Секция по id, а промахом — по стему файла (шард-группа: id — групповое
// имя, секции в конфигах живут и по «…-00001-of-00004»).
const ModelOverrides* overrides_for(const std::map<std::string, ModelOverrides>& ov,
                                    const std::string& id,
                                    const std::string& model_path);
// Эффективный mtp-mode модели: переопределение сильнее глобального [general].
std::string mtp_mode_of(const ModelOverrides* o, const config::Config& cfg);


// Раскладка загрузки по картам: пин на одну (--device CUDA<i> — без
// контекстов на остальных) или сплит на весь пул (--device все + --main-gpu
// самая ёмкая + -ts по TOTAL-бюджетам карт; урок L11: доли по текущей свободе
// плавали между рестартами и не совпадали с планом, посчитанным против
// «пустого пула»). devices пуст при отсутствии телеметрии — спавним как
// раньше, без флагов.
struct TryTarget {
  bool split = false;
  int main_gpu = -1;    // карта пина; при сплите — позиция в --device списке
  std::string devices;  // "CUDA0" или "CUDA0,CUDA1"
  std::string ts;       // пропорции -ts (МиБ бюджетов); пусто — не передаём
  hw::Gpu card;         // карта/пул, ПО КОТОРОЙ считался план
};

// Одна попытка: спавн с данными параметрами и ожидание /health.
// ok=true — живой сервер; ключ и порт — наружу (прокси без них не достучится).
// wedge_out=true — ребёнок не умер даже по SIGKILL (клин драйвера).
// SpawnExtra: всё, что должно попасть в CLI ребёнка напрямую.
struct SpawnExtra {
  int np = 1;                    // -np (слоты); вместе с ним включаем continuous batching
  bool embedding = false;        // --embedding (эмбеддерный ребёнок)
  std::string draft_path;        // --model-draft
  std::string spec_type;         // --spec-type (с draft или без — встроенный MTP)
  int draft_ngl = 0;             // --gpu-layers-draft (слои самого черновика + выходной)
  std::string draft_device;      // --device-draft («CUDA<i>»); пусто — карта ствола, как раньше
  std::string expert_ot;         // -ot-правило ПЕРЕД «exps=CPU»: конкретные блоки экспертов —
                                 // на запасную карту; пусто — все на CPU (как раньше)
  int expert_spare_idx = -1;     // индекс запасной карты expert_ot: раскладку ствола
                                 // переводим в «обе карты + ловушка .*=ствол» (иначе
                                 // движок не знает CUDA spare и падает — проверено
                                 // 28.10: -ot =CUDA0 при --device только ствол = смерти
                                 // попыток; CUDA1,CUDA0 + .*→CUDA1 = живой сервер)
  int ncmoe = 0;                 // --n-cpu-moe N (частичное размещение экспертов MoE)
  int moe_cache_mib = 0;         // --moe-cache-mib N (пин b11963): из плана, не из
                                 // конфига напрямую — планировщик урезал по бюджету;
                                 // спавн передаёт только при вынесенных экспертах
  double moe_cpu_share = 0.0;    // --moe-cache-cpu-share ([general] moe-cpu-share):
                                 // параллельный CPU-счёт промахов (#30966); флаг
                                 // только вместе с выданным грантом кэша
  int threads = 0;               // -t ребёнку ([general] threads); 0 — дефолт
                                 // движка (hardware_concurrency = SMT)
  std::string extra_args;        // дословные доп. флаги из [general] extra-args —
                                 // универсальный обход для флагов движка, которых
                                 // llmhost ещё не знает (экспериментальные PR и
                                 // т.п.); их смысл для запускаемого бинаря — на юзере
  std::string think_mode;        // --reasoning on|off (пер-модельный thinking-mode);
                                 // пусто — флага нет, движок сам детектит по шаблону
  std::string slot_save_path;    // --slot-save-path (парковка KV слотов, harness):
                                 // каталог-белый список файлов save/restore;
                                 // пусто — ребёнку флаг не ставится
};

extern std::set<std::string> g_wedged_models;
extern std::mutex g_wedged_mtx;
extern std::atomic<int64_t> g_last_kill_ms;

int64_t steady_ms();
void warm_pages(const std::string& path);

// Парковка KV-состояния слотов на NVMe (harness; реализация — sup_harness.cpp).
// Контракт пина b11963: ребёнку ставится --slot-save-path (каталог-белый
// список, обязан существовать к спавну), демон зовёт
// POST /slots/{i}?action=save|restore с телом {"filename": "голое-имя.bin"};
// save тянет полное KV seq слота вместе с draft/MTP, restore при отказе
// очищает слот, не роняя ребёнка. Порог/потолок — [general] slot-park*.
std::string park_dir();
std::string park_filename_for(const std::string& model_path, int slot);
// Save достойных слотов перед выселением сторожем простоя (GET /slots →
// save по порогу токенов). false — модель ожила/занялась за время
// сохранения, останов отменить; ошибки HTTP — лог и true (выселение не
// блокируем никогда).
// require_idle=false — парковка при ВЫТЕСНЕНИИ (под память/лимит), а не при
// простое: idle-перепроверку пропускаем, busy/state-перепроверки остаются.
bool park_before_idle_stop(const std::string& id, bool require_idle = true);
// Restore сохранённых слотов после успешной загрузки. Зовётся ПОСЛЕ записи
// профиля (иначе восстановленная KV попала бы в дельту надбавки) и до
// будильника очереди. true — хоть один слот восстановлен (пробу соседней
// формы KV тогда не назначаем — она снесёт возвращённую KV).
bool park_restore_after_load(const std::string& id);
uint64_t parked_bytes_for(const std::string& model_path);
void park_purge_for_path(const std::string& model_path);

bool try_attempt(const std::string& exe, const std::string& path, int ngl, int ctx_req,
                 bool fa, bool kv_q8, const std::string& kv_type, bool exps_cpu, int ubatch,
                 const std::string& mmproj, const TryTarget& tgt, const SpawnExtra& extra,
                 lh::proc::Child& c, int& port_out, std::string& key_out, int& actual_ctx,
                 std::string& err, bool* wedge_out = nullptr,
                 std::string* tail_out = nullptr);


// Draft к основной модели: ручной ключ «model-draft» (любой, включая MTP)
// или авто — младшая модель той же архитектуры (draft-simple). Определение
// функции — после path_eq.
struct DraftPick {
  std::string path;
  uint64_t size = 0;
  std::string spec_type;
  bool auto_attached = false;
  int n_layers = 0;  // слои самого draft (для gpu-layers-draft); 0 — мета не читалась
  uint64_t n_vocab = 0; // словарь draft — для оценки его compute-буферов в плане
  uint64_t n_embd = 0;  // ширина draft — для оценки его KV на запасной карте
};

extern std::set<std::string> g_dead_drafts;
extern std::mutex g_dead_drafts_mtx;
extern std::set<std::string> g_shy_drafts;
extern std::mutex g_shy_drafts_mtx;
extern std::set<std::string> g_draft_suspended;
extern std::mutex g_draft_suspended_mtx;
extern std::map<std::string, std::pair<std::string, bool>> g_draft_by_id;
extern std::mutex g_draft_by_id_mtx;

bool path_eq(const std::string& a, const std::string& b);

DraftPick find_draft(const std::string& main_path, const gguf::Meta& meta, uint64_t fsize,
                     const config::Config& cfg, const ModelOverrides* o,
                     const std::vector<lh::modelreg::Model>& list);

// Оценка KV-кэша черновика (f16, вся ширина — сверху, GQA делает меньше):
// место под его кэш, когда черновик едет на запасную карту.
uint64_t draft_kv_bytes(const DraftPick& dp, int ctx);

std::string draft_hint_for(const gguf::Meta& meta, uint64_t fsize,
                           const std::vector<lh::modelreg::Model>& list);


// Целевая карта модели (мульти-GPU): куда лягут веса и по какой карте считать
// память. Общая математика раскладки и предполёта загрузки.
struct GpuTarget {
  hw::Gpu card;       // карта-цель; при сплите — синтетический пул
  std::string lines;  // "main-gpu = ...\ntensor-split = ...\n" или пусто
  int main_gpu = -1;  // индекс пина; -1 — без пина (дефолтный сплит)
  bool split = false; // веса едут на несколько карт (card — пул)
};

// Запасная карта: nvidia-карта пула, НЕ входящая в used_cards, со свободой
// не меньше need_bytes; лучшая по свободе. -1 — запасной нет. Телеметрии нет
// (пустая query_all) — -1: раскладываем только по факту, не по догадке.
int spare_card_index(const std::set<int>& used_cards, uint64_t need_bytes);

// Карты, занятые раскладкой ствола: пин — одна; сплит или без пина — весь
// пул (дефолтный делёж движка трогает все карты).
std::set<int> target_cards(const GpuTarget& gt);

// -ot-правило для КОНКРЕТНОЙ запасной карты: блоки экспертов MoE с головы,
// суммарно до budget байт («blk\.(0|1|…)\.ffn_.*_exps\.=CUDA<i>»). Правило
// едет ПЕРЕД «exps=CPU» / «--n-cpu-moe» — движок берёт первое совпавшее.
// max_blocks — брать только первые N блоков (для частичной формы ncmoe: за
// их пределом эксперты и так на GPU, переселять на запасную — только хуже;
// 0 — без ограничения, для полного выноса). Пусто — ни один блок не влезает.
std::string expert_share_ot(const gguf::Meta& meta, int card, uint64_t budget,
                            int* blocks_out = nullptr, uint64_t* bytes_out = nullptr,
                            int max_blocks = 0);

struct MState {
  std::string state; // unloaded | loading | loaded
  bool failed = false;
  std::string note; // диагноз планировщика (деградация формы) — для клиентов
};

extern std::map<std::string, std::string> g_id_map;      // id реестра → путь модели;
extern std::mutex g_apply_mtx;
extern std::set<std::string> g_emb;                       // эмбеддерные модели: чат их не трогает;
extern std::map<std::string, std::string> g_sys;          // id → system-промпт (инъекция на прокси);
extern std::mutex g_busy_mtx;
extern std::set<std::string> g_busy_set;                  // модели, где идёт генерация;
extern std::mutex g_fp_mtx;
extern std::string g_applied_fp;

extern std::mutex g_mtx;

std::string make_unique_id(const std::string& base,
                           std::map<std::string, std::string>& by_id);
bool is_embedding_arch(const std::string& arch);
// Vision-проекция рядом с моделью; diag — почему не прицеплена (пусто, если
// причин говорить нет). Вызывающий на load-пути уносит diag в лог.
std::string find_mmproj(const std::string& model_path,
                        std::string* diag = nullptr);
// Та же проекция с учётом пер-модельного vision-mode (off — без проекции);
// один гейт для всех load-путей.
std::string mmproj_for_load(const std::string& model_path,
                            const ModelOverrides* o, std::string* diag = nullptr);
GpuTarget gpu_target_for(const std::string& id, const std::string& path,
                         const config::Config& cfg,
                         const std::map<std::string, ModelOverrides>& ov,
                         bool live = false);

void note_model_used(const std::string& path);


struct HSlot {
  std::shared_ptr<lh::proc::Child> child;
  int port = 0;
  std::string api_key;
  std::string id, path;
  St st = St::Unloaded;
  std::string err;
  int ctx = 0, used_ngl = 0, used_ctx = 0, np = 1, ncmoe = 0;
  bool smart = false, kv_q8 = false, fa = false, exps_cpu = false, warm = false,
       reduced = false;
  bool parkable = false; // ребёнку выдан --slot-save-path — слоты можно парковать
  std::set<int> cards;   // карты спавна (target_cards); пусто — «пересекается
                         // со всем» (спавн без телеметрии трогает все карты)
  std::string note;      // диагноз деградации формы спавна — уезжает клиенту в чат
  clock_::time_point loaded_at{}, last_used{};
};

extern std::map<std::string, HSlot> g_slots;
// id, у которых фоновая проба соседней формы KV сейчас держит слот.
// Под g_mtx. Пока id здесь, будилка ждёт, а upstream_for порт не отдаёт.
extern std::set<std::string> g_kv_probe;
extern park::Lot g_wake_lot;

struct WarmSample {
  double tps = 0;
  int n = 0;
};
// Прогрев ребёнка (128 токенов «Hi»): собирает граф декодирования до первого
// клиентского запроса. model — для роутера: /completion без «model» не знает,
// кого греть. Общая для классики и роутера (бывший warmup_model — дубль).
WarmSample warmup(int port, const std::string& api_key, const std::string& model = {});

std::string join_ids(const std::vector<std::string>& v, const char* sep);
void save_resident_state();
void wake_notify();
std::map<std::string, std::string> harness_registry_maps(
    std::set<std::string>& emb_out, std::map<std::string, std::string>& sys_out);
std::map<std::string, MState> harness_states();
// false — слот занят генерацией, останов отменён; force=true — даже занятый
// (shutdown). Останов ребёнка — вне g_mtx; слот проходит через Stopping.
bool stop_hslot(const std::string& id, bool to_asleep, bool persist = true,
                bool force = false);
// force=true — люк в инварианте «занятое не выселяется» (только для одной
// модели, решение пользователя): stop_hslot с force, KV не паркуется.
nlohmann::json harness_unload_models_impl(const std::string& id, std::string& err,
                                          bool force = false);
bool load_harness(const std::string& path, const std::string& id, std::string& err,
                  std::vector<std::string>* busy_blockers_out);
// Будилка /v1 и админ-загрузки. allow_off_registry=true — принимать строки,
// содержащие «.gguf», как буквальный путь вне реестра (только админский вход
// sup::load: /v0/load, трей, визард). Клиентский /v1 (false) резолвит строго
// через реестр: иначе «модель» вида /home/other/private/foo.gguf заставила бы
// демона читать и грузить в VRAM любой файл, доступный его пользователю.
WakeResult harness_wake(const std::string& want, int timeout_ms,
                        long long* parked_ms_out, bool allow_off_registry = false);

} // namespace lh::sup::detail
