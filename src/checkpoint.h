// Именованные чекпоинты KV-состояния —
// транзакционное состояние агентского цикла: откат шага возможен, только
// если состояние слота сохраняемо и адресуемо. Два класса снимков слота:
//   park      — авто, потребляется с удалением при restore (sup_harness);
//   checkpoint — именованный, пользователя, НЕ удаляется при restore
//               (форковаться можно многократно), авто-LRU не выталкивается.
// Чекпоинт снимает ВСЕ слоты ребёнка (план: «save всех простаивающих
// слотов») — движок держит внутренние слоты и при parallel=1, нумерацию
// ведёт он; в каталоге одна запись на имя, файлы помечены «.ckpt.» и
// «.s<N>.bin», чтобы park-механика (потребление, LRU, parked_mib) их не
// трогала. Каталог llmhost-checkpoints.json лежит рядом с llmhost-image.json
// (в папке моделей) и правится только этим писателем — атомарно tmp+rename
// с сохранением чужих полей (прецедент binding_upsert). Файлы снимков живут
// в каталоге парковки (slot_parks) — движок принимает голые имена только из
// --slot-save-path.
//
// Канонизатор prefix_hash — один на проект: restore-by-hash («тот же запрос
// переживёт выселение») умирает молча без зафиксированной канонизации.
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace lh::checkpoint {

struct SlotFile {
  int slot = 0;          // слот ребёнка (нумерация ведёт движок)
  std::string file;      // голое имя файла снимка в каталоге парковки
  uint64_t bytes = 0;    // размер файла
  long long tokens = 0;  // токенов в снимке (n_saved движка)
};

struct Entry {
  std::string model;       // путь модели (первая часть шард-группы)
  std::string name;        // имя чекпоинта (sanitize_name)
  std::vector<SlotFile> files;
  std::string prefix_hash; // sha256 канонизированных messages последнего
                           // чат-хода (пусто — хэш недоступен/не снимался)
  double created = 0;      // unix
};

// Пользовательский ввод в имя файла напрямую не едет: безопасный ASCII
// ([alnum._-]), остальное — в «_», потолок 64 символа, пустое — «checkpoint».
std::string sanitize_name(const std::string& raw);

std::string catalog_path(const std::string& models_dir);

std::vector<Entry> list(const std::string& models_dir,
                        const std::string& model = {});
std::optional<Entry> find(const std::string& models_dir, const std::string& model,
                          const std::string& name);
// Запись/обновление по (model, name): свои поля обновляются, чужие (и чужие
// ключи верхнего уровня файла) сохраняются. false — err объясняет.
bool upsert(const std::string& models_dir, const Entry& e, std::string& err);
// Явное удаление — единственный путь выталкивания именованного чекпоинта
// (авто-LRU его не трогает никогда).
bool remove(const std::string& models_dir, const std::string& model,
            const std::string& name, Entry* removed, std::string& err);
// Все чекпоинты модели (удаление модели с диска — снимки её уже не
// восстановить никогда, это гигиена, а не LRU).
bool remove_model(const std::string& models_dir, const std::string& model,
                  std::string& err);

// SHA256 hex (in-process): prefix_hash зовётся на каждом чат-ходе к
// однослотной модели — цена осознанная и записана (sha256 ≈ ГиБ/с, на сотнях
// кБ — миллисекунды); системный sha256sum-ребёнок на это не годится.
std::string sha256_hex(const std::string& data);

// Канонизатор v1 (версия внутри хэша): только роль и строковый content в
// порядке следования; содержимое строк байт-в-байт как прислал клиент, без
// нормализаций. temperature/top_p, порядок ключей JSON, пробелы вне строк
// хэш не меняют; отсутствующий content (assistant с tool_calls) — пустая
// строка; массивный (мультимодальный) content — хэш не считается, пустая
// строка: restore-by-hash для такого запроса честно недоступен.
std::string prefix_hash(const nlohmann::json& messages);

} // namespace lh::checkpoint
