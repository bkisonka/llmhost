// Супервизор: harness-режим — по llama-server на каждую модель в VRAM.
// Спавн (Job Object: умрёт вместе с llmhost), health, /props, прогрев,
// OOM-лестница (меньше -ngl, затем меньше -c), выгрузка по простою;
// спящую модель прокси будит сам.
#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace lh::sup {

enum class St { Unloaded, Loading, Loaded, Asleep, Error, Stopping };

St state();
St state(std::string& model_path_out);   // какая модель в слоте (и у спящей)

// Гарантированно загруженная модель: будит спящую, дожидается идущей загрузки.
// true — Loaded; по выходу можно слать запросы. Будилка — load_harness
// с выселением незанятых и парковкой в очереди пробуждения (см.
// ensure_model_ex); loading — ждём до timeout_ms.
bool ensure_model(const std::string& id, int timeout_ms, std::string& err);

// Итог будилки с деталями очереди пробуждения (harness): queued_timeout —
// TTL очереди истёк, клиенту честный 504 с причиной, а не 502-буря;
// not_found — модели нет в реестре (ошибка клиента — 404, не 503).
struct WakeResult {
  bool ok = false;
  bool queued_timeout = false;
  bool not_found = false;
  std::string err;
};
WakeResult ensure_model_ex(const std::string& id, int timeout_ms);

bool ensure_running(int timeout_ms, std::string& err);

// Адрес и ключ llama-server для прокси. true — модель Loaded.
// upstream_for: слот ЗАПРОШЕННОЙ модели (id или путь); пустой id — последний
// использованный Loaded слот.
bool upstream_for(const std::string& id, int& port, std::string& api_key);
bool upstream(int& port, std::string& api_key);

// Диагноз деградации формы спавна (harness): «загружено урезанно: … —
// варианты: …». Пусто — модель загружена по плану или не резидентна.
std::string model_note(const std::string& id);

// Хвост вывода llama-server (последние строки): диагностика ошибок запросов.
// Вариант с id — harness: хвост своего ребёнка (пусто — нет/неизвестен).
std::string server_log_tail();
std::string server_log_tail_for(const std::string& id);

// Блокирующая загрузка (из рабочего потока; прокси будит асинхронно сам).
// При нехватке видеопамяти параметры понижаются сами — см. лестницу в .cpp.
// Админский вход: путь вне реестра разрешён (в отличие от клиентского /v1).
bool load(const std::string& path, std::string& err);

// Выгрузить всё, кроме занятых (чужой стрим неубиваем). to_asleep — модели
// останутся в реестре, прокси разбудит по запросу.
void stop(bool to_asleep);

// Дождаться, пока драйвер вернёт видеопамять убитого llama-server'а: делает
// это с задержкой, и заспавненный сразу новый процесс ловит ложный OOM — или
// молча спиллит веса в shared-память (генерация падает в разы без единой
// ошибки в логе). expect_free — сколько должно стать свободно, когда память
// убитого вернётся (0 — неизвестно: ждём, пока свободная VRAM увидит скачок
// вверх и перестанет расти). card: -1 — следим за суммой всех планируемых
// карт, иначе — за указанной. Без NVML — фиксированная пауза.
void vram_settle(uint64_t expect_free, int card = -1);

// Внешняя смерть наших детей (подметание сирот на старте) — штампует момент
// kill'а: первая после этого загрузка дождётся возврата VRAM, как после
// обычной замены модели (см. vram_settle).
void note_external_kill();

// Состояние для control API и тултипа трея.
nlohmann::json status_json();

// ---- чат-фасад ----------------------------------------------------------------

// Цель чата: не-эмбеддерная модель по умолчанию (пусто — пусть подставит прокси).
std::string chat_model();

struct ChatModel {
  std::string id;     // id реестра (имя файла без .gguf)
  std::string state;  // unloaded | loading | loaded
  bool embedding;     // эмбеддер: в чате не показывать
};

// Список моделей для выбора в чате (реестр + состояния слотов).
std::vector<ChatModel> chat_models();

// Архитектура-эмбеддер? (для пометки в меню; реализация знает bert-семейство)
bool embedding_arch(const std::string& arch);

// Карта, к которой привязана модель раскладкой (мульти-GPU); -1 — без пина.
int model_gpu(const std::string& path);

// System-промпт модели из model_overrides.json (пусто — нет).
std::string system_prompt_for(const std::string& id);

// Подложить system-сообщение модели в тело запроса, если клиент своего
// не прислал (аналог Modelfile SYSTEM). true — тело переписано.
bool inject_system(const std::string& id, std::string& body);

// Эффективный «model» для запроса: известный id — как есть; пустой/«any» —
// последняя не-эмбеддерная загруженная → единственная в реестре; неизвестный —
// как есть (ребёнок честно ответит «not found»).
std::string default_chat_model(const std::string& requested);

// Разобрать тело запроса прокси и подставить «model», если нужно. Возвращает
// эффективный id (пусто — не JSON / подсказать нечем). Тело может быть
// перезаписано.
std::string route_request(std::string& body);

// Запрос начат/закончен: «последнее использование» модели и её неупятивание
// сторожем простоя посреди генерации.
void touch_model(const std::string& id);

struct BusyModel {
  explicit BusyModel(const std::string& id);
  ~BusyModel();
  BusyModel(const BusyModel&) = delete;
  BusyModel& operator=(const BusyModel&) = delete;
private:
  std::string id_;
};

// Живое состояние моделей: {id: {state, failed}} по всему реестру.
nlohmann::json models_state();

// Точная телеметрия живых детей: [{model, series{llamacpp-ключи}}] — опрос
// их /metrics на чтение; для render() метрик. Пусто — ни одного Loaded.
nlohmann::json engine_metrics();

// id реестра для пути модели (пусто — нет такой).
std::string id_for_path(const std::string& path);

// Выгрузить одну (id или путь) или все незанятые (пусто). Занятая
// модель не выселяется НИКОГДА: стрим клиента посреди слова неубиваем —
// отказ (err) на конкретную модель, «busy_kept» в ответе на «все».
// force=true — явный люк в инварианте (выключатель против зомби-генерации):
// слот останавливается даже занятым, ВСЕ текущие генерации этой модели
// обрываются для всех её клиентов; KV не паркуется (снимок посреди
// генерации — мусор), прежние park-файлы не трогаются. Только для одной
// модели (на «все» не даётся — слишком легко выстрелить себе в руки).
nlohmann::json harness_unload_models(const std::string& id, std::string& err,
                                     bool force = false);

// Есть ли у модели ручной черновик в model_overrides.json (draft уже настроен).
bool has_draft_override(const std::string& id);

// Потянет ли карта плотную модель частично (CUDA-движок) с экстра-грузом
// extra_bytes (например, MTP-модулем): планировщик подвинет слои, модуль у
// него в бюджете. Для диалога MTP — свободной VRAM «в лоб» может не быть.
bool dense_partial_fits(const std::string& path, uint64_t extra_bytes);

// Включить MTP-черновик модели: дописать «model-draft»/«spec-type» в
// переопределения и перезагрузить модель. false — ошибка в err (модель
// остаётся без черновика).
bool enable_mtp(const std::string& id, const std::string& draft_stem,
                const std::string& main_path, std::string& err);

// Реестр изменился (докачана модель): карта id→путь перестраивается сразу.
void registry_changed();

// Удалить файлы парковки KV слотов этой модели (парковка — harness, см.
// [general] slot-park*); зовётся из DELETE /v0/models. Другие режимы файлов
// не создают — безопасный no-op.
void park_purge_for_path(const std::string& model_path);

// Применить изменения реестра немедленно: перестроить карту id→путь и
// выселить слоты исчезнувших файлов (то же, что делает вотчер, только сразу).
void apply_registry_change();

// Ручной контекст модели: записать «ctx-size» в секцию модели llmhost.conf
// (ctx == 0 — убрать ключ), сохранив остальное содержимое файла. Применится
// при следующей загрузке модели (enable_mtp перегружает сразу).
// false — сбой, err — причина.
bool set_model_ctx(const std::string& model_path, int ctx, std::string& err);
// Пер-модельный размер кэша MoE: 0 — снять override ([general] moe-cache-mib
// снова решает), N — МиБ вручную. Применяется при следующей загрузке модели.
bool set_model_cache(const std::string& model_path, int mib, std::string& err);
// Пер-модельный mtp-mode: mode "auto" снимает переопределение.
bool set_model_mtp(const std::string& model_path, const std::string& mode,
                   std::string& err);
std::string override_mtp_for_path(const std::string& model_path);

// Пер-модельный vision-режим: "auto" снимает переопределение (своя проекция
// прицепляется сама, родство имён); "off" — ребёнок едет без --mmproj, что бы
// ни лежало рядом.
bool set_model_vision(const std::string& model_path, const std::string& mode,
                      std::string& err);
std::string override_vision_for_path(const std::string& model_path);
// Пер-модельный thinking-режим: "auto" снимает переопределение (движок сам
// детектит thinking по шаблону); "on"/"off" — форс --reasoning ребёнку.
bool set_model_thinking(const std::string& model_path, const std::string& mode,
                        std::string& err);
std::string override_think_for_path(const std::string& model_path);
// Проекция, которая прицепится к модели сейчас (vision-mode и родство имён
// учтены — тот же ответ, что у load-путей); пусто — никакая.
std::string vision_mmproj_for_path(const std::string& model_path);
// Предупреждение для ручного контекста: непустая строка — «такой ctx
// исключает vision» (проекция не помещается, хотя без неё помещается).
std::string vision_ctx_note(const std::string& model_path, int ctx);

// Ручной ctx-size из llmhost.conf для пути модели (0 — не задан).
int override_ctx_for_path(const std::string& model_path);

// Имя секции модели в llmhost.conf (sanitize_id от стема файла).
std::string override_section_for(const std::string& model_path);

// Убить всех детей-модели (подмена exe установщиком движка — файлы моделей
// должны освободиться).
void stop_router();

void init();     // сторож простоя
void shutdown(); // остановить модель при выходе из программы

// Фоновая работа, которая зовёт load и wake_notify (загрузка по API,
// автозагрузка, форвард в llama-server). shutdown и reap_background
// дожидаются её, пока очередь пробуждения ещё жива.
void track_background(std::function<void()> fn);
// Дождаться фоновых потоков и добить ребёнка, которого они успели поднять.
void reap_background();

// ---- генерация изображений (второй движок, sd-server; отдельный слот) --------
// Второе семейство движков параллельно llama: свой ребёнок на своей карте
// (image-gpu), свой диапазон портов, свой сторож простоя. Авторизации у
// sd-server нет — ребёнок слушает только 127.0.0.1, наружу его выпускает
// OpenAI-порт с общей auth-проверкой. Каталог моделей — image::scan().
// Смена модели = останов старого ребёнка и спавн нового (NVMe/кэш страниц
// делают это секундами; рантайм-смена через /sdcpp/v1 — в отложенном).

// Состояние слота + карточки каталога (для GET /v0/image).
nlohmann::json image_status_json();

// Блокирующая загрузка модели каталога по id (рабочий поток: control зовёт
// через track_background, прокси — прямо). Пустой id — модель автостарта
// image-model, затем последняя загруженная. false — err объясняет.
bool image_load(const std::string& id, std::string& err);

// Остановить image-ребёнка (если жив).
void image_unload();

// Удалить модель каталога с сервера: файлы слотов + недокачки + рецепт (если
// свободен) и запись из llmhost-image.json. Загруженную останавливает сам;
// занятая загрузкой — отказ. false — err; report — удалённое для ответа.
bool image_remove(const std::string& id, nlohmann::json& report, std::string& err);

// Порт image-ребёнка для прокси. Ребёнок не поднят — автостарт (см.
// image_load); не вышло — false, err с причиной (клиенту 503).
bool image_upstream(int& port, std::string& err);

// Простой считается заново (прокси зовёт на каждый запрос).
void image_touch();

// Простой считается заново (прокси зовёт на каждый запрос) + счётчик
// активных запросов: сторож не остановит ребёнка посреди генерации.
struct ImageBusy {
  ImageBusy();
  ~ImageBusy();
  ImageBusy(const ImageBusy&) = delete;
  ImageBusy& operator=(const ImageBusy&) = delete;
};

} // namespace lh::sup
