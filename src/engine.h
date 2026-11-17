// Движок llama-server: пин релиза в manifest.json + engine-release в
// llmhost.conf; установка = скачать (докачка, SHA256) + распаковать
// (miniz на Windows, внешний tar на POSIX) с защитой от путей наружу.
#pragma once
#include <functional>

#include "config.h"

#include <cstdint>
#include <string>
#include <vector>

namespace lh::engine {

struct BuildInfo {
  std::string id;
  bool pinned = false;      // SHA256 в manifest — установку можно доверять
  uint64_t size = 0;        // суммарный размер файлов
  bool installed = false;
  std::string exe;         // путь к llama-server.exe установленной сборки
  std::string installed_release; // факт с диска: bNNNNN из --version; пусто — не определить
};

std::vector<BuildInfo> list();
std::string exe_for(const std::string& build_id);

// Сборка по железу: NVIDIA → первая пиннутая cuda-сборка манифеста (cuda12
// на Linux, cuda13 на Windows), иначе vulkan. Дефолт для установки и выбора,
// когда engine_build в конфиге не задан.
std::string default_build();

// Откуда брать llama-server: явный путь из настроек → выбранная сборка →
// любая установленная → рядом с exe → PATH.
std::string find_server(const config::Config& cfg);

// Какая сборка реально пойдёт в запуск при этом конфиге — зеркало каскада
// find_server: явный путь («custom») → выбранная (engine_build) → по железу
// (default_build) → любая установленная. Пусто — llama-server не находится.
std::string active_build(const config::Config& cfg);

// Запустит ли find_server CUDA-сборку на этом конфиге? Зеркалит его каскад:
// явный путь (подсказка — «cuda» в пути) → выбранная сборка → сборка по
// железу → любая установленная. Не удалось определить — false: вызывающий
// живёт по-старому (для плотных моделей крупнее VRAM это «не влезает»).
bool launches_cuda(const config::Config& cfg);

// Скачивает и распаковывает сборку (блокирующе — из рабочего потока).
// Пин из llmhost.conf обязан сходиться с манифестом (его разрешает pin()).
bool install(const std::string& build_id, std::string& err);

// Пин релиза движка: источник правды — engine-release в llmhost.conf;
// пусто/битое — затравка из manifest.json рядом с бинарями.
std::string pinned_release();

// Идентификаторы сборок manifest.json — допустимые значения engine-build
// для валидации llmhost.conf (config::validate_conf, lhctl set).
std::vector<std::string> manifest_builds();

// Свежий b-релиз llama.cpp с GitHub (releases?per_page=15, первый bNNNNN).
// Кэш 10 минут (ошибки — минута): меню движка открывают чаще, чем апстрим
// выпускает релизы. Пусто — ошибка в err.
std::string latest_release(std::string& err);

// Обновить пин движка на релиз llama.cpp. ПЕРЕД записью — валидация всего
// llmhost.conf (юзер мог покорёжить его руками: битый конфиг пин не получает).
// Имена ассетов выводятся из текущего пина (замена старого тега на новый),
// SHA256 и размеры — из digest'ов GitHub API (api.github.com/repos/ggml-org/
// llama.cpp), сами тарболлы не скачиваются. Итог: manifest.json рядом с
// бинарями (кэш url/хэшей, .bak обязателен) + engine-release в llmhost.conf
// (источник правды). Бинарники установленных сборок не трогаются — установка
// после пина по-прежнему через install() (и по-прежнему требует выгрузки).
struct PinResult {
  std::string release; // новый тег (bNNNNN)
  int files = 0;       // перепинено файлов
  std::string path;    // какой манифест перепинен (рядом с бинарником демона)
  std::string added;   // сборки, достроенные из ассетов релиза (через запятую)
};
bool pin(const std::string& tag, PinResult& out, std::string& err);

struct State {
  std::string state = "idle"; // idle | downloading | extracting | error
  std::string build;
  uint64_t done = 0, total = 0;
  double speed = 0;
  std::string error;
};
std::string status_json();
bool busy();

// Гейт «загрузка/установка не совпадают»: подмена файлов движка (install) и
// запуск детей llama-server (загрузки моделей, bench-замер) не пересекаются.
// Брони атомарны под одним замком: либо загрузка успела взять билет — тогда
// установка честно откажется, либо установка — и до её конца (любой исход)
// билет загрузки не возьмётся. Обычная проверка состояния окно гонки не
// закрывает в принципе: между «проверил» и «начал» состояние меняется
// (поймано живьём 10.10: установка стартовала под загружающейся моделью,
// хук Quiesce видел Loading лишь иногда). Билет RAII — освобождается при
// любом исходе носителя.
class GateTicket {
public:
  enum class Kind { Load, Install };
  GateTicket() = default;
  ~GateTicket();
  GateTicket(GateTicket&& o) noexcept : kind_(o.kind_), held_(o.held_) { o.held_ = false; }
  GateTicket& operator=(GateTicket&& o) noexcept;
  GateTicket(const GateTicket&) = delete;
  GateTicket& operator=(const GateTicket&) = delete;
  // true — бронь взята (held()); false — отказ, err с причиной, бронь не взята.
  bool acquire(Kind kind, std::string& err);
  bool held() const noexcept { return held_; }
  void release() noexcept;
private:
  Kind kind_ = Kind::Load;
  bool held_ = false;
};

// Установка в полёте? Чтение без брони — для мягких отказов вне труб
// загрузки (спавн роутера, пред-проверка /v0/load): авторитетная проверка —
// acquire(), не она.
bool install_in_progress();

// Предусловия и глушение перед подменой файлов движка: «слот свободен, бенч
// тихий, роутер с детьми остановлены». Хук ставит демон на старте (daemond,
// main), который знает sup и bench — сам engine супервизора не зовёт (ребро
// вверх в графе зависимостей перевёрнуто: раньше install дёргал
// sup::state/sup::stop_router/bench::busy напрямую). false — err с причиной.
// От гонки «начали под замену файлов, а модель загрузилась» защищает не хук
// (он — проверка в один момент времени), а гейт GateTicket выше.
using Quiesce = std::function<bool(std::string& err)>;
void set_quiesce(Quiesce fn);

} // namespace lh::engine
