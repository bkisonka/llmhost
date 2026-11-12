// Настройки: llmhost.conf (INI) в $XDG_DATA_HOME/llmhost (или $LH_DATA_DIR) —
// ЕДИНСТВЕННЫЙ источник правды. Секция [general] — глобальные настройки,
// остальные секции — per-model переопределения (их читает supervisor).
// Правится вручную или через lhctl set; демону не нужен рестарт: вотчер
// применяет [general] на лету (порты/bind/auth/cors — после
// рестарта, они не перебиндиваются живьём).
//
// Рантайм-состояние (последняя модель, ответы про MTP) — не настройки, оно в
// state.json рядом: иначе каждый ход чата переписывал бы конфиг и дёргал
// вотчер. Секреты (token.txt, api_tokens.json) и обучаемые замеры
// (model_profiles.json) тоже живут отдельными файлами.
#pragma once
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace lh::config {

struct Config {
  int openai_port = 7330;        // стабильный OpenAI-совместимый endpoint
  int control_port = 7331;       // control API
  // Адрес привязки OpenAI-порта. «127.0.0.1» — только localhost (SSH-туннель);
  // «0.0.0.0» — внешняя служба, доступная из LAN. Control API всегда
  // слушает localhost — управление не выставляется наружу никогда.
  std::string openai_bind = "127.0.0.1";
  int unload_after_min = 15;     // выгрузить модель после простоя; 0 — никогда
  int ctx = 4096;                // контекст по умолчанию
  int ngl = 999;                 // слоёв на видеокарту (999 — все)
  bool openai_auth = true;       // :7330 требует Bearer-токен (тот же, что в control)
  bool autostart = false;        // Windows; на Linux автозапуск — systemd unit
  bool debug_console = false;    // наследие Windows; на Linux не используется
  bool auto_params = true;       // умный автоподбор: контекст «из модели», KV q8_0, MoE на CPU
  // Очередь пробуждения: запрос к нерезидентной модели, которой
  // мешают ЗАНЯТЫЕ резиденты, паркуется; просыпается, когда занятость
  // блокера упала в ноль и слот выселен. TTL истёк — честный 504 с причиной
  // вместо 502-бурь. ttl 0 — парковки нет: занято → отказ сразу.
  int queue_ttl = 180;           // секунд ожидания в очереди
  int queue_depth = 16;          // максимум ждущих одновременно (все модели)
  // Парковка KV-состояния слотов на NVMe (harness): при выселении простойной
  // модели сторожем простоя демон сохраняет KV слота ребёнка
  // (POST /slots/{i}?action=save, детям ставится --slot-save-path), при
  // пробуждении возвращает (action=restore) — ход клиента попадает в кэш
  // промпта вместо полного перепрефилла. Замер 30.10: возврат 6,4 ГиБ KV
  // ~2 с (dio) против 12–20+ с перепрефилла. Отказы best-effort: не
  // сохранилось/не восстановилось — слот просто холодный, план не страдает.
  bool slot_park = true;            // выключатель (на хостинге с теским NVMe — off)
  int slot_park_min_tokens = 8192;  // глубина промпта слота, ниже которой не паркуем
  int slot_park_gib = 64;           // потолок суммарного размера файлов парковки; 0 — без предела
  int models_max = 2;            // --models-max: моделей резидентно (0 — без лимита, LRU выгрузит)
  // Слотов на модель (роутер и harness); суммарный контекст делится между
  // слотами с полом 2048/слот (np режется, если контекста не хватает).
  // Дефолт 1 — решено юзером 28.10: параллель — ручка под конкурентную
  // нагрузку (замер «N клиентов» 28.10: VRAM MoE ×1,84, оффлоуд ×1,28 на
  // N=4), одиночному интерактивному клиенту делёж контекста без надобности.
  int parallel = 1;
  // Кэш горячих MoE-экспертов в VRAM: -1 — авто (план выбирает раскладку по
  // прогнозу резидентности: нативные эксперты + кэш·yield), 0 — выкл, N —
  // вручную (МиБ; физику режет бюджет карты). Передаётся движку
  // (--moe-cache-mib, пин b11963) только при вынесенных экспертах. Замеры
  // 28–29.10: 30B-A3B ×2,9 (37% пула), Next-80B ×1,9 (27%), малые доли
  // больших пулов вредны (трэш SLRU).
  int moe_cache_mib = 0;
  // Доля промахов кэша MoE, которую движок считает на CPU параллельно с GPU
  // (--moe-cache-cpu-share, PR #30966): с критического пути токена эта доля
  // уходит — скрывается под вниманием и попаданиями. Замер 30.10 на
  // Flash-Next (кэш 8 ГиБ, ctx 65k): share 0 → 19,7 ток/с, share 0.6 →
  // 40,3. Сетка знает эту долю и при её оценке кэша, и планировщик передаёт
  // флаг ребёнку только вместе с выданным грантом. Дефолт 0 — поведение
  // апстрима; выше 0.9 не поднимать (модель промахов слепеет к загрузке).
  double moe_cpu_share = 0.0;
  bool gpu_pin = true;           // раскладывать модели по картам по очереди (мульти-GPU)
  bool chat_think = true;        // размышления модели в чате (Windows-чат; в Linux-CLI выключает клиент)
  bool auto_draft = true;        // спекуляция по умолчанию: MTP/младшая модель семьи как draft
  // Сторож acceptance спекуляции: приём черновиков считается окнами по сумме
  // счётчиков (accepted/generated из лога ребёнка, не среднее rate). Устойчиво
  // ниже floor (три плохих окна подряд, окно набрало min-tokens) — auto-draft
  // отстраняется, модель перегружается без спекуляции; ручной model-draft —
  // только предупреждение. floor 0 — сторож выключен.
  double draft_accept_floor = 0.5;
  int draft_accept_min_tokens = 1000;
  // Политика MTP-модуля вместо диалога (GUI нет): auto — включать, если модуль
  // влезает на карту по оценке планировщика; on/off — принудительно.
  std::string mtp_mode = "auto";
  int ubatch = 0;                // физический батч для llama-server; 0 — дефолт (512)
  int threads = 0;               // CPU-потоки ребёнка (-t); 0 — дефолт движка
                                 // (hardware_concurrency = SMT). Замер 31.10:
                                 // 8 физических ядер бьют SMT-16 на декоде
                                 // при exps_cpu + cpu-share > 0
  std::string models_dir;        // пусто — <app_dir>/models
  std::string llama_server;      // путь к llama-server; пусто — искать
  std::string extra_args;        // дословные доп. флаги каждого ребёнка через
                                 // пробел («--moe-cache-cpu-share 0.6»); без
                                 // валидации — обязан знать запускаемый движок
  std::string engine_build;      // сборка из manifest.json для запуска; пусто — любая
  // Целевой релиз движка («b11382»); пусто — затравка из manifest.json рядом
  // с бинарями. Пишут lhctl engine pin/update (перед записью — валидация
  // всего llmhost.conf: покорёженный юзером файл пин не получает).
  std::string engine_release;
  // Доверенные источники для CORS-эха Access-Control-Allow-Origin на OpenAI-порту
  // (кроме всегда доверенных localhost/127.0.0.1). Пусто — только localhost:
  // произвольный сайт в браузере не должен читать ответы локального API.
  std::vector<std::string> cors_origins;
  // --- генерация изображений (второй движок, sd-server; отдельный слот) ------
  // Движок sd-server из stable-diffusion.cpp: свой ребёнок на своей карте,
  // наружу только через OpenAI-порт (auth у sd-server нет — привязка 127.0.0.1).
  // Каталог моделей — раскладка sd-server: diffusion_models/, text_encoders/,
  // vae/ + recipes/*.json (рецепт — единственный источник правды о составе
  // модели и дефолтах генерации).
  std::string image_engine;      // путь к sd-server; пусто — искать <app_dir>/engine/sd/ и PATH
  std::string image_models_dir;  // пусто — models_dir (та же раскладка категорий)
  int image_gpu = 0;             // карта image-движка (индекс NVML/CUDA PCI-порядка)
  std::string image_model;       // модель автостарта при первом запросе; пусто — только ручная загрузка
  int image_idle_min = 15;       // останов image-ребёнка после простоя; 0 — никогда
  std::string image_extra_args;  // дословные доп. флаги sd-server (как extra-args)
  // «Позже» в визарде первого запуска (Windows); на Linux не используется.
  bool wizard_dismissed = false;
  // Язык интерфейса: auto — по локали машины (русская → русский, любая
  // другая → английский), ru/en — форс. Применяется живьём (тексты lhctl/
  // TUI и клиентские ответы API), рестарт не нужен. Детали — src/i18n.h.
  std::string lang = "auto";
};

// Рантайм-состояние — НЕ настройки: пишется само (последняя модель — на каждом
// ходе, ответы про MTP — один раз). В llmhost.conf ему не место.
struct State {
  std::string last_model;                      // пусто — нечего восстанавливать
  std::map<std::string, bool> mtp_choices;     // id модели → не предлагать модуль
  // harness: резидентные модели на момент записи (восстановление после
  // рестарта демона). Пути — как хранятся в конфиге (resolve_path/store_path).
  std::vector<std::string> resident;
};

// Пути к файлам данных.
std::string conf_path();    // llmhost.conf  — настройки (INI)
std::string state_path();   // state.json    — рантайм-состояние
std::string token_path();
std::string default_models_dir();

// Пути в llmhost.conf хранятся относительно папки программы; абсолютные и с
// $переменными — как есть.
std::string resolve_path(const std::string& stored);  // из конфига → рабочая
std::string store_path(const std::string& absolute);  // рабочая → в конфиг
std::string models_dir_resolved(const Config& c);     // пусто — умолчание

// Привязка только на localhost (127.0.0.1/localhost/::1)? Внешняя служба —
// всё остальное (0.0.0.0, LAN-адрес): тогда Bearer на OpenAI-порту не
// отключается (fail-closed, см. daemond main).
bool is_local_bind(const std::string& bind);

// Разовый переезд с config.json (legacy): значения — в [general] llmhost.conf
// (уже существующие там ключи НЕ перетираются), last_model/mtp_choices — в
// state.json; config.json → config.json.migrated. Идемпотентно: нет json —
// ничего не делает.
void migrate_legacy();

Config load();                          // из conf_path(), битые ключи — warn+дефолт
void save(const Config& c);
// Те же операции с явным путём — для тестов. save_to вливает [general] в
// существующий файл (комментарии и чужие секции сохраняются).
Config load_from(const std::string& path);
void save_to(const std::string& path, const Config& c);
// Единый локер всех read-modify-write писателей llmhost.conf: [general]
// (save_to), секции переопределений (sup_overrides: set_model_override_kv,
// enable_mtp) и миграция json→INI пишут ОДИН файл, и каждый до сих пор делал
// это под своим замком (enable_mtp — вообще без). Порядок вложенности:
// g_apply_mtx супервизора → conf_writers_mtx; в обратную сторону никто не
// берёт. Внутрь apply_registry_change под этим замком не заходят (иначе
// enable_mtp, держа его, уткнулся бы в g_apply_mtx).
std::mutex& conf_writers_mtx();
// Миграция с явными путями — для тестов. true — config.json был перенесён.
bool migrate_legacy_at(const std::string& conf_file, const std::string& json_file,
                       const std::string& state_file);

State load_state();                     // из state_path(); нет файла — пусто
void save_state(const State& s);

// Строгая проверка llmhost.conf перед рискованными записями (пин релиза
// движка): известные ключи — на валидность значений и принадлежность секциям,
// ключ вне секции — ошибка, неизвестные ключи игнорируются (за будущими
// версиями). allowed_builds — допустимые сборки движка из manifest.json
// (пусто — проверяется только формат engine-build). Пустой вектор — файл
// валиден (или отсутствует). Вариант _at — с явным путём (тесты).
std::vector<std::string> validate_conf(
    const std::vector<std::string>& allowed_builds = {});
std::vector<std::string> validate_conf_at(
    const std::string& path, const std::vector<std::string>& allowed_builds = {});

// Таблица [general]: все известные ключи в каноническом порядке с текущими
// значениями (значения — как в файле, пути уже через store_path). Для
// GET /v0/settings, lhctl set и diff'а вотчера.
std::vector<std::pair<std::string, std::string>> settings_table(const Config& c);

// Проверить и нормализовать значение одного ключа [general] (писатели:
// POST /v0/settings, lhctl set, validate_conf). false — err объясняет.
bool check_setting(const std::string& key, const std::string& value,
                   const std::vector<std::string>& allowed_builds,
                   std::string& norm, std::string& err);
// Присвоить нормализованное значение полю Config (false — ключ неизвестен).
bool set_field(Config& c, const std::string& key, const std::string& norm);

// Ключи, которые живьём не применяются (порты/bind/auth/cors): вотчер и
// /v0/settings про них пишут «вступит после рестарта демона».
bool needs_daemon_restart(const std::string& key);

// Актуальные настройки всего демона: main читает файл один раз и кладёт сюда,
// control API/вотчер обновляют. Копия — дёшево и без гонок.
void set_current(const Config& c);
Config current();

// Токен клиента: случайный при первом запуске, дальше живёт в файле.
// Один и тот же для control API и (по умолчанию) для OpenAI-порта.
std::string client_token();

// Проверка заголовка Authorization против токена — одна на оба сервера,
// сравнение за постоянное время. Вердикт решает вызывающий: Wrong → 401,
// NoToken → fail-closed (API закрыт, а не «пускаем всех»).
enum class BearerVerdict { Ok, Wrong, NoToken };
BearerVerdict check_bearer(const std::string& auth_header);

} // namespace lh::config
