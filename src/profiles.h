// Хранилище обученных профилей планировщика (model_profiles.json).
// Ключ записи — «id модели + форма спавна»: замер одной формы нельзя
// приложить к другой (состав VRAM-следа иной), поэтому у одной модели
// сколько слотов, сколько форм, в которых её гоняют.
#pragma once
#include "fit.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace lh::profiles {

constexpr int kVersion = 2; // семантика замера; чужая версия отбрасывается

struct ModelProfile {
  bool out_gpu = false;
  bool out_gpu_failed = false; // амбициозная попытка (выходной на GPU) OOM-илась
  bool cuda = true;
  bool kv_q8 = true;
  std::string kv_type;
  int ubatch = 0;
  uint32_t ctx = 0;
  uint64_t overhead = 0; // байт VRAM сверх весов и аналитического KV+state
  uint64_t ram_bytes = 0;
  bool exps_cpu = false;
  int ncmoe = 0; // частичный сплит MoE: первых N блоков экспертов в RAM
  int ngl = 0;
  int moe_cache = 0;  // кэш MoE МиБ (арена SLRU): меняет и след VRAM, и скорость
  int ot_blocks = 0;  // блоки экспертов на запасной карте (-ot): host-состав другой
  double updated = 0.0;
  double gen_tps = 0; // 0 — ток/с этой формы ещё не измерен
  int gen_n = 0;       // токенов в выборке; <0 — соседнюю форму пробовали и не сняли
};

// Форма спавна — вторая половина ключа. Поля обязаны совпадать точно:
// выходной слой, класс KV, ubatch, класс движка, эксперты в RAM и число
// слоёв частичного сплита меняют сам состав следа. Контекст и kv_type
// (q8_0/q4_0) в ключе не участвуют: первое масштабируется ramp'ом,
// второй меняет объём KV, который считает аналитика, а не остаточный
// оверхед. Кэш MoE и блоки на запасной карте — участвуют: арена живёт
// в тех же гигабайтах, что веса (след), а их скорость — другая.
struct Form {
  bool out_gpu = false;
  bool kv_q8 = true;
  int ubatch = 0;
  bool cuda = true;
  bool exps_cpu = false;
  int ncmoe = 0; // частичный сплит MoE — точное число блоков в RAM
  int ngl = 0;   // частичный сплит — точное число слоёв; полные формы — 0
  int moe_cache = 0; // МиБ кэша MoE (0 — без кэша)
  int ot_blocks = 0; // блоки экспертов на запасной карте
};

// «<id>|o1k1c1e0m0n0u2048»: форма читается глазами, id санитайзится
// (символы вне [alnum._-] → «_», включая «|» — ключ не перетечёт).
std::string make_key(const std::string& id, const Form& f);

// Мост в fit::estimate: замеренная надбавка вместо аналитической.
inline fit::Measured measured_of(const ModelProfile& p) {
  return {static_cast<int64_t>(p.overhead), p.ctx, p.cuda};
}

// Читает model_profiles.json: формат v2 — {"engine": "bNNNNN",
// "profiles": {"<id>|o..u..": {...}}}; плоская карта старого формата
// читается как есть, форма записи выводится из её же полей (штампа нет —
// первая установка движка отметит и сотрёт такие замеры).
void load();
void save();

// Профиль формы: empty — замера нет ИЛИ амбициозная попытка этой формы
// уже провалилась OOM'ом (планировщик честно идёт аналитикой). Копия, а не
// указатель: note_engine() (установка движка) чистит хранилище под тем же
// замком — держатель сырого указателя поймал бы use-after-free.
std::optional<ModelProfile> lookup(const std::string& id, const Form& f);

// Сохранить/обновить замер формы. Успешный замер out_gpu-формы снимает
// флаг провала амбициозной попытки этой же формы (другие формы не трогает).
void put(const std::string& id, const Form& f, const ModelProfile& p);

// Ток/с формы. Короткая выборка не участвует в выборе. gen_n < 0 —
// проба была и цифры нет, повторять её на каждой загрузке не надо.
void note_speed(const std::string& id, const Form& f, double tps, int n);
void mark_speed_skipped(const std::string& id, const Form& f);
bool speed_settled(const std::string& id, const Form& f);
void speed_of(const std::string& id, const Form& f, double& tps, int& n);
// Оба замера не короче 128 токенов и победитель быстрее хотя бы на 8%.
// Иначе остаётся политика (policy_q8).
bool kv_q8_by_speed(bool policy_q8, double tps_q8, int n_q8,
                    double tps_f16, int n_f16);

// Амбициозная попытка формы кончилась OOM — пометить, больше не пробовать,
// пока форма не сменится или замер этой же формы не пройдёт.
void note_failed_out(const std::string& id, const Form& f);

// Установка движка другого релиза: сброс всех замеров (штамп «engine» в
// файле). Переустановка того же релиза замеры не трогает; пин манифеста
// сюда не зовётся — бинарники ещё прежние, а пин могут откатить.
void note_engine(const std::string& release);

// Репутация draft-файлов на текущем движке, переживает рестарт демона:
// «dead» — движок не смог его загрузить (ребёнок падал), «shy» — сторож
// отклонил по приёму черновиков. Хранится рядом с замерами и стирается
// вместе с ними при смене релиза: вердикт вынесен этому движку, следующий
// вправе переоткрыть вопрос.
void note_bad_draft(const std::string& path, const char* kind);
std::vector<std::string> bad_drafts(const char* kind);

// Точка кривой покрытия кэша MoE (id модели → размер кэша и hit rate из
// строки движка «llama_moe_cache: ... hit rate = X%» при остановке ребёнка
// либо выведенная из пары скоростей форм — sup_plan, обращение T(C)).
// Планировщик выводит из неё локальность маршрутизации a = ln(1−h)/ln(1−s)
// для аналитического выбора размера кэша (sup_internal.h, модель T(C)).
void note_moe_point(const std::string& id, int cache_mib, double hit_rate);
bool moe_point(const std::string& id, int& cache_mib, double& hit_rate);

// Замер скорости формы С КЭШЕМ (для вывода локальности из пары скоростей):
// последняя по времени запись этого id с moe_cache > 0 и снятым ток/с.
// Возвращает поля формы (состав host-части) и контекст замера.
struct CacheProbe {
  int moe_cache = 0;
  int ot_blocks = 0;
  bool exps_cpu = false;
  int ncmoe = 0;
  int ngl = 0;
  bool kv_q8 = true;
  uint32_t ctx = 0;
  double tps = 0;
  int n = 0;
};
// Пара «с кэшем / без» одного id: обе скорости сняты (≥128 токенов),
// контексты и класс KV совпадают (иначе T_аттн не сокращается — пары нет).
bool moe_pair_of(const std::string& id, CacheProbe& cached, CacheProbe& plain);

} // namespace lh::profiles
