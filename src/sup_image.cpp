// Image-слот: второй движок (sd-server из stable-diffusion.cpp) параллельно
// llama. Своё состояние, свой спавн, свой сторож простоя — с llama-слотом
// (слоты harness) не пересекается ни данными, ни портами (8200–8249 против
// 8100–8199).
// Конвейер загрузки короче llama-лестницы и делит с ней только общий низ:
// GateTicket «загрузка/установка», free_tcp_port, wait_health, vram_settle.
// Каталог моделей — image::scan() (рецепты sd-server + привязки
// llmhost-image.json). Смена модели = останов старого ребёнка и спавн нового.
#include "supervisor.h"
#include "sup_internal.h"

#include "config.h"
#include "engine.h"
#include "image.h"
#include "image_catalog.h"
#include "proc.h"
#include "util.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;
#include <memory>
#include <mutex>
#include <system_error>
#include <thread>
#ifndef _WIN32
#include <unistd.h>
#endif

#include <httplib.h>
#include <nlohmann/json.hpp>

namespace lh::sup {

namespace {

using clock_ = std::chrono::steady_clock;

#ifdef _WIN32
constexpr const char* kSdServerExe = "sd-server.exe";
#else
constexpr const char* kSdServerExe = "sd-server";
#endif

struct ImageSlot {
  St st = St::Unloaded;
  std::unique_ptr<proc::Child> child;
  int port = 0;
  std::string model_id;   // загруженная/загружаемая модель каталога
  std::string err;        // последняя ошибка (для статуса и 503)
  uint64_t bytes = 0; // вес резидента — сколько VRAM вернётся после останова
  clock_::time_point last_used;
};

std::mutex g_img_mtx;            // слот + состояние
std::condition_variable g_img_cv; // ждущие Loading (прокси/ensure)
ImageSlot g_img;
std::string g_last_image;        // последняя удачно загруженная (автостарт)

std::atomic<int> g_img_busy{0};  // активные генерации: сторож ждёт нуля
std::atomic<bool> g_img_stop{false};
std::thread g_img_watch;

// Каталог image-моделей — image::models_dir_resolved: единая точка с pull
// image-моделей и удалением (иначе привязки и файлы разъезжаются по каталогам).
std::string image_models_dir_resolved() {
  return image::models_dir_resolved(config::current());
}

// Путь к sd-server: явный image-engine → <app_dir>/engine/sd/ → PATH.
// Пусто — движок не найден (статус честно скажет «не найден»).
std::string image_engine_path() {
  const config::Config c = config::current();
  if (!c.image_engine.empty()) {
    std::string p = expand_env(config::resolve_path(c.image_engine));
    if (file_exists(p)) return p;
  }
  std::string local = app_dir() + "/engine/sd/" + kSdServerExe;
  if (file_exists(local)) return local;
#ifndef _WIN32
  // PATH — как в find_llama_server: руками, exec-доступность на элементе.
  // Пустые сегменты PATH не исполняем (cwd может быть world-writable).
  const char* path = ::getenv("PATH");
  if (path && *path) {
    size_t start = 0;
    const size_t len = strlen(path);
    while (start <= len) {
      size_t end = start;
      while (end < len && path[end] != ':') ++end;
      std::string dir(path + start, end - start);
      if (!dir.empty()) {
        std::string cand = dir + "/" + kSdServerExe;
        if (::access(cand.c_str(), X_OK) == 0) return cand;
      }
      if (end >= len) break;
      start = end + 1;
    }
  }
#endif
  return {};
}

// Флаги дефолтов генерации из рецепта: ребёнок получит их CLI-дефолтами,
// запрос клиента всё ещё может переопределить.
void append_recipe_defaults(std::vector<std::string>& args, const image::Model& m) {
  if (m.steps > 0) {
    args.push_back("--steps");
    args.push_back(std::to_string(m.steps));
  }
  if (m.cfg >= 0) {
    args.push_back("--cfg-scale");
    args.push_back([&] {
      char b[32];
      std::snprintf(b, sizeof(b), "%.2f", m.cfg);
      return std::string(b);
    }());
  }
  if (!m.sampler.empty()) {
    args.push_back("--sampling-method");
    args.push_back(m.sampler);
  }
  if (!m.scheduler.empty()) {
    args.push_back("--scheduler");
    args.push_back(m.scheduler);
  }
}

const image::Model* find_model(std::vector<image::Model>& v, const std::string& id) {
  for (const auto& m : v)
    if (m.id == id) return &m;
  return nullptr;
}

// Жив ли ребёнок на самом деле; мёртвый переводится в Error (внешняя смерть).
// Вызывается под g_img_mtx.
void reap_dead_locked() {
  if (g_img.st != St::Loaded || !g_img.child) return;
  int code = 0;
  if (g_img.child->exited(code)) {
    const std::string tail = g_img.child->log_tail();
    logf("image: ребёнок sd-server умер (код %d): %s", code,
         lh::proc::last_meaningful_line(tail).c_str());
    // Ненормальная смерть — сыпем хвост вывода целиком (лимит строк): одна
    // «значимая» строка из abort-трейса libc диагноз не даёт (инцидент 09.11:
    // код 134 без причины в журнале, весь ggml-трейс остался в кольце).
    if (code != 0) {
      int lines = 0;
      size_t start = 0;
      while (lines < 60 && start < tail.size()) {
        size_t nl = tail.find('\n', start);
        if (nl == std::string::npos) nl = tail.size();
        std::string l = tail.substr(start, nl - start);
        if (!l.empty())
          logf("image:   | %.*s", (int)std::min<size_t>(l.size(), 300), l.c_str());
        start = nl + 1;
        ++lines;
      }
    }
    g_img.st = St::Error;
    g_img.err = "sd-server завершился (код " + std::to_string(code) + ")";
    g_img.child.reset();
    g_img.port = 0;
  }
}

// Останов ребёнка под замком. expect_bytes — сколько VRAM должно вернуться
// (0 — неизвестно): следующий спавн дождётся возврата через vram_settle.
void stop_locked() {
  if (g_img.child) {
    g_img.child->stop();
    g_img.child.reset();
  }
  g_img.port = 0;
  if (g_img.st != St::Error) g_img.st = St::Unloaded;
  g_img_cv.notify_all();
}

// Спавн ребёнка с моделью m и ожидание готовности. Замок НЕ держится
// (wait_health до трёх минут); вызывающий держит st = Loading.
bool spawn_locked_release(const image::Model& m, const std::string& exe,
                          std::string& err, bool had_child, uint64_t prev_bytes) {
  const config::Config cfg = config::current();
  const std::string dir = image_models_dir_resolved();

  // VRAM к моменту спавна: если остановили резидента — дождаться возврата
  // его памяти драйверу, как после смены llama-модели. Холодный спавн
  // (резидента не было) ждать нечему — settle сжёг бы 24 с впустую.
  if (had_child) vram_settle(prev_bytes, cfg.image_gpu);
  hw::flush_telemetry_cache();
  auto cards = hw::query_all();
  if (cfg.image_gpu < 0 || cfg.image_gpu >= (int)cards.size() ||
      !cards[cfg.image_gpu].nvidia || !cards[cfg.image_gpu].vram_total) {
    err = "карта image-gpu=" + std::to_string(cfg.image_gpu) +
          " недоступна (нет NVML-телеметрии или не NVIDIA)";
    return false;
  }
  const hw::Gpu& card = cards[cfg.image_gpu];
  // Веса + запас на активации/буферы: без оффлоада всё резидентно на карте
  // (замер 08.11: активации 1024² у 6B/4B DiT — сотни МиБ; 0,75 ГиБ покрывает
  // с толком, больший запас гонит пограничные формы в ложный отказ).
  const uint64_t need = m.bytes + m.bytes / 20 + ((uint64_t)768 << 20);
  if (card.vram_free < need) {
    char free_gb[32], need_gb[32];
    std::snprintf(free_gb, sizeof(free_gb), "%.1f", card.vram_free / 1073741824.0);
    std::snprintf(need_gb, sizeof(need_gb), "%.1f", need / 1073741824.0);
    err = "на карте image-gpu=" + std::to_string(cfg.image_gpu) + " свободно " +
          free_gb + " ГиБ, веса с запасом требуют " + need_gb +
          " ГиБ — занято llama-моделью? разгрузите её (lhctl unload) или смените image-gpu";
    return false;
  }

  int port = lh::free_tcp_port(8200, 8249);
  if (!port) {
    err = "нет свободного порта 8200–8249 для sd-server";
    return false;
  }

  std::vector<std::string> args = {
      "--model-dir", dir,
      "--diffusion-model", m.diffusion,
      "--vae", m.vae,
      "--llm", m.llm,
      "--listen-ip", "127.0.0.1",
      "--listen-port", std::to_string(port),
      "--backend", "cuda" + std::to_string(cfg.image_gpu),
  };
  append_recipe_defaults(args, m);
  // Значения с пробелами не поддерживаются — флаги движка их не требуют.
  {
    size_t pos = 0;
    const std::string& ea = cfg.image_extra_args;
    while (pos <= ea.size()) {
      size_t sp = ea.find(' ', pos);
      std::string tok = ea.substr(pos, sp == std::string::npos ? std::string::npos
                                                               : sp - pos);
      if (!tok.empty()) args.push_back(tok);
      if (sp == std::string::npos) break;
      pos = sp + 1;
    }
  }

  auto c = std::make_unique<proc::Child>();
  // CUDA_VISIBLE_DEVICES: внутри ребёнка cuda<N> — единственная карта;
  // без этого ggml-CUDA инициализирует все устройства (~200 МиБ контекста
  // на чужих картах — недопустимо рядом с планировкой llama-моделей).
  if (!c->start(exe, args,
                {{"CUDA_VISIBLE_DEVICES", std::to_string(cfg.image_gpu)}})) {
    err = std::string("не удалось запустить sd-server (fork/exec): ") + strerror(errno);
    return false;
  }
  auto w = proc::wait_health(*c, port, 180000, "/");
  if (w != proc::Wait::Healthy) {
    std::string tail = lh::proc::last_meaningful_line(c->log_tail());
    if (w == proc::Wait::Oom)
      err = "sd-server упал по памяти: " + (tail.empty() ? "нехватка VRAM" : tail);
    else if (w == proc::Wait::Exited)
      err = "sd-server не поднялся: " + (tail.empty() ? "процесс завершился" : tail);
    else
      err = "sd-server не ответил за 180 с: " +
            (tail.empty() ? "нет готовности на GET /" : tail);
    c->stop();
    vram_settle(m.bytes, cfg.image_gpu);
    return false;
  }

  {
    std::lock_guard l(g_img_mtx);
    // Спавн шёл без замка: image_unload мог остановить «резидента» в статусе
    // Loading, пока ребёнок поднимался. Тогда не коммитим Loaded поверх
    // явного Unloaded — свежего ребёнка глушим (следующая загрузка честная).
    if (g_img.st != St::Loading) {
      logf("image: ребёнок «%s» поднят после отмены загрузки — останавливаю",
           m.id.c_str());
      c->stop();
      vram_settle(m.bytes, cfg.image_gpu);
      return false;
    }
    g_img.child = std::move(c);
    g_img.port = port;
    g_img.model_id = m.id;
    g_img.bytes = m.bytes;
    g_img.st = St::Loaded;
    g_img.err.clear();
    g_img.last_used = clock_::now();
    g_last_image = m.id;
  }
  g_img_cv.notify_all();
  logf("image: модель «%s» загружена на карту %d, порт %d (%.1f ГиБ весов)",
       m.id.c_str(), cfg.image_gpu, port, m.bytes / 1073741824.0);
  return true;
}

// Рантайм-смена модели живому ребёнку (POST /sdcpp/v1/models/load): ребёнок
// сам проверяет выборку, выгружает резидента и грузит новую — без респавна
// (секунды вместо полного подъёма). Ребёнок должен быть жив и Loaded.
// false — err объясняет; ребёнок после неудачи может остаться без модели
// (422) — вызывающий переводит слот в Error, следующий load идёт респавном.
bool runtime_switch(const image::Model& m, std::string& err) {
  int port = 0;
  uint64_t resident_bytes = 0;
  {
    std::lock_guard l(g_img_mtx);
    if (g_img.st != St::Loaded || !g_img.child || !g_img.port) return false;
    port = g_img.port;
    resident_bytes = g_img.bytes;
  }
  const config::Config cfg = config::current();
  const std::string dir = image_models_dir_resolved();

  // Идентификаторы файлов каталога ребёнка (--model-dir): относительные пути.
  std::error_code ec;
  const std::string rel_d = fs::relative(m.diffusion, dir, ec).string();
  const std::string rel_l = fs::relative(m.llm, dir, ec).string();
  const std::string rel_v = fs::relative(m.vae, dir, ec).string();
  if (ec || rel_d.empty() || rel_l.empty() || rel_v.empty() ||
      rel_d[0] == '/' || rel_l[0] == '/' || rel_v[0] == '/') {
    // Канонический путь вне --model-dir (симлинк-компонент) — редкий
    // разворот: смена falls back к респавну, который путям не доверяет.
    err = "пути модели вне каталога ребёнка";
    return false;
  }

  // Предполёт: новый влезает в свободу ПЛЮС возврат весов резидента (ребёнок
  // выгружает старую модель до загрузки новой). Резидента не трогаем —
  // при отказе он продолжает работать.
  hw::flush_telemetry_cache();
  auto cards = hw::query_all();
  if (cfg.image_gpu >= 0 && cfg.image_gpu < (int)cards.size() &&
      cards[cfg.image_gpu].nvidia && cards[cfg.image_gpu].vram_total) {
    const uint64_t need = m.bytes + m.bytes / 20 + ((uint64_t)768 << 20);
    const uint64_t avail = cards[cfg.image_gpu].vram_free + resident_bytes;
    if (avail < need) {
      char free_gb[32], need_gb[32];
      std::snprintf(free_gb, sizeof(free_gb), "%.1f", avail / 1073741824.0);
      std::snprintf(need_gb, sizeof(need_gb), "%.1f", need / 1073741824.0);
      err = "смена модели: на карте image-gpu=" + std::to_string(cfg.image_gpu) +
            " доступно " + free_gb + " ГиБ (с возвратом резидента), веса требуют " +
            need_gb + " ГиБ";
      return false;
    }
  }

  httplib::Client cli("127.0.0.1", port);
  cli.set_connection_timeout(std::chrono::seconds(3));
  cli.set_read_timeout(std::chrono::seconds(120));
  const std::string body = nlohmann::json{
      {"recipe", m.recipe},
      {"models",
       {{"diffusion_model", rel_d}, {"llm", rel_l}, {"vae", rel_v}}}}
                  .dump();
  auto res = cli.Post("/sdcpp/v1/models/load", body, "application/json");
  if (!res) {
    err = "sd-server не ответил на смену модели";
    return false;
  }
  if (res->status == 409) {
    err = "sd-server занят генерацией — смена модели отложена до тишины";
    return false;
  }
  if (res->status != 200) {
    // 422 — модель не загрузилась: ребёнок жив, слот пуст. Error уводит
    // следующую загрузку на чистый респавн.
    std::string why = res->body.substr(0, res->body.find('\n'));
    if (why.size() > 200) why.resize(200);
    err = "sd-server отклонил смену модели (HTTP " +
          std::to_string(res->status) + "): " +
          (why.empty() ? "без подробностей" : why);
    std::lock_guard l(g_img_mtx);
    if (g_img.st == St::Loaded) {
      g_img.st = St::Error;
      g_img.err = err;
      g_img.child.reset();
      g_img.port = 0;
    }
    return false;
  }
  {
    std::lock_guard l(g_img_mtx);
    // Смена шла без замка: выгрузка/смерть ребёнка успели — не коммитим.
    if (g_img.st != St::Loaded || !g_img.child || g_img.port != port) {
      err = "слот сменился во время смены модели";
      return false;
    }
    g_img.model_id = m.id;
    g_img.bytes = m.bytes;
    g_img.last_used = clock_::now();
    g_last_image = m.id;
  }
  logf("image: рантайм-смена на «%s» (%.1f ГиБ весов) — без респавна",
       m.id.c_str(), m.bytes / 1073741824.0);
  return true;
}

// Полный конвейер загрузки: гейт → останов старого → settle → спавн.
// Блокирующая; состояние Loading видно остальным потокам весь путь.
bool load_pipeline(const std::string& id, std::string& err) {
  const std::string exe = image_engine_path();
  if (exe.empty()) {
    err = "sd-server не найден: установите его в <app_dir>/engine/sd/ или задайте image-engine";
    return false;
  }
  const std::string dir = image_models_dir_resolved();
  auto catalog = image::scan(dir);
  if (catalog.empty()) {
    err = "каталог image-моделей пуст (" + dir + "): нужен llmhost-image.json + рецепты";
    return false;
  }
  const image::Model* m = find_model(catalog, id);
  if (!m) {
    err = "в каталоге нет модели «" + id + "» (есть: ";
    for (size_t i = 0; i < catalog.size(); ++i)
      err += (i ? ", " : "") + catalog[i].id;
    err += ")";
    return false;
  }

  lh::engine::GateTicket ticket;
  if (!ticket.acquire(lh::engine::GateTicket::Kind::Load, err)) return false;

  std::pair<bool, uint64_t> load_args{false, 0};
  bool resident = false;

  {
    // Сериализация загрузок: гейт Load допускает конкурентность (harness
    // грузит модели параллельно), здесь конкурирующий загрузчик дождётся
    // чужого Loading; та же модель — идемпотентный успех.
    std::unique_lock l(g_img_mtx);
    g_img_cv.wait_for(l, std::chrono::seconds(240), [] { return g_img.st != St::Loading; });
    reap_dead_locked();
    if (g_img.st == St::Loaded && g_img.model_id == id) {
      g_img.last_used = clock_::now();
      return true;
    }
    resident = g_img.st == St::Loaded && g_img.child && g_img.port;
  }
  // Резидент жив — смена без респавна: ребёнок сам выгружает старую модель
  // и грузит новую. Респавн остаётся фолбэком (пути вне каталога, смерть
  // ребёнка, опустевший слот после 422). Замок в runtime_switch берётся сам.
  if (resident) {
    std::string switch_err;
    if (runtime_switch(*m, switch_err)) return true;
    // Итог свитча: живой слот — отказ наружу (409/отказ ребёнка), иначе —
    // фолбэк к респавну. Замки последовательные, без вложенности.
    bool raced_loading = false;
    {
      std::lock_guard l(g_img_mtx);
      if (g_img.st == St::Loaded) {
        err = switch_err;
        return false;
      }
      raced_loading = g_img.st == St::Loading;
    }
    if (raced_loading) {
      // Конкурентная загрузка пришла во время свитча — дождаться её и уйти
      // в идемпотентный успех, если она грузила ту же модель.
      std::unique_lock l2(g_img_mtx);
      g_img_cv.wait_for(l2, std::chrono::seconds(240),
                        [] { return g_img.st != St::Loading; });
      if (g_img.st == St::Loaded && g_img.model_id == id) {
        g_img.last_used = clock_::now();
        return true;
      }
    }
  }
  {
    // Смена респавном = останов старого ребёнка: его VRAM нужна новой.
    std::unique_lock l(g_img_mtx);
    bool had_child = g_img.child != nullptr;
    uint64_t prev_bytes = g_img.bytes;
    stop_locked();
    g_img.st = St::Loading;
    g_img.model_id = id;
    g_img.err.clear();
    g_img_cv.notify_all();
    load_args = {had_child, prev_bytes};
  }

  bool ok = spawn_locked_release(*m, exe, err, load_args.first, load_args.second);
  if (!ok) {
    // Причина обязана доехать до журнала: control грузит в фоне
    // (track_background) — возвращаемое err там теряется.
    logf("image: загрузка «%s» не удалась: %s", id.c_str(), err.c_str());
    std::lock_guard l(g_img_mtx);
    // Error — только для честного провала: если во время спавна успела
    // отработать явная выгрузка (st == Unloaded), её приговор не трогаем.
    if (g_img.st == St::Loading) {
      g_img.st = St::Error;
      g_img.model_id = id;
      g_img.err = err;
    }
    g_img_cv.notify_all();
  }
  return ok;
}

// Сторож простоя: Loaded && нет активных запросов && image-idle-min истёк →
// останов (модель остаётся автостартом). Дольки по 500 мс — shutdown не ждёт.
void watch_loop() {
  while (!g_img_stop.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    if (g_img_stop.load()) return;
    const config::Config cfg = config::current();
    std::lock_guard l(g_img_mtx);
    reap_dead_locked();
    if (cfg.image_idle_min <= 0 || g_img.st != St::Loaded || g_img_busy.load() > 0)
      continue;
    auto idle = std::chrono::duration_cast<std::chrono::minutes>(
                    clock_::now() - g_img.last_used)
                    .count();
    if (idle >= cfg.image_idle_min) {
      logf("image: простой %lld мин — останавливаю sd-server (модель «%s» останется автостартом)",
           (long long)idle, g_img.model_id.c_str());
      stop_locked();
    }
  }
}

} // namespace

ImageBusy::ImageBusy() { g_img_busy.fetch_add(1); }
ImageBusy::~ImageBusy() { g_img_busy.fetch_sub(1); }

nlohmann::json image_status_json() {
  const config::Config cfg = config::current();
  const std::string dir = image_models_dir_resolved();
  nlohmann::json j;
  j["gpu"] = cfg.image_gpu;
  j["idle_min"] = cfg.image_idle_min;
  j["models_dir"] = dir;
  std::string exe = image_engine_path();
  j["engine"] = nlohmann::json{{"path", exe}, {"found", !exe.empty()}};
  std::vector<image::Model> catalog = image::scan(dir);
  j["configured"] = image::catalog_exists(dir);
  nlohmann::json models = nlohmann::json::array();
  for (const auto& m : catalog) {
    nlohmann::json features = nlohmann::json::array();
    for (const auto& f : m.features) features.push_back(f);
    models.push_back({
        {"id", m.id},
        {"name", m.name},
        {"recipe", m.recipe},
        {"bytes", m.bytes},
        {"diffusion", m.diffusion},
        {"steps", m.steps},
        {"cfg", m.cfg},
        {"sampler", m.sampler},
        {"features", features},
    });
  }
  j["models"] = models;
  // Каталог художников для pull: имена для «lhctl image pull <имя>» и раздела
  // TUI — клиентам не нужен второй маршрут, чтобы узнать, что можно скачать.
  nlohmann::json aliases = nlohmann::json::array();
  for (const auto& e : image_catalog::all()) aliases.push_back(e.id);
  j["aliases"] = aliases;

  std::lock_guard l(g_img_mtx);
  reap_dead_locked();
  const char* st = "unloaded";
  switch (g_img.st) {
    case St::Unloaded: st = "unloaded"; break;
    case St::Loading: st = "loading"; break;
    case St::Loaded: st = "loaded"; break;
    case St::Asleep: st = "unloaded"; break;
    case St::Error: st = "error"; break;
  }
  j["state"] = st;
  j["model"] = g_img.model_id;
  j["port"] = g_img.port;
  j["in_flight"] = g_img_busy.load();
  j["error"] = g_img.err.empty() ? nlohmann::json(nullptr) : nlohmann::json(g_img.err);
  // Кандидат автостарта — тот же выбор, что делает image_load(""): гейты
  // прокси решают по целевой модели до будки ребёнка (edits-гейт).
  std::string auto_model = cfg.image_model;
  if (auto_model.empty()) auto_model = g_last_image;
  if (auto_model.empty() && catalog.size() == 1) auto_model = catalog[0].id;
  j["autostart_model"] = auto_model;
  return j;
}

bool image_load(const std::string& id, std::string& err) {
  // Пустой id — автостарт: image-model, затем последняя удачная, затем
  // единственная модель каталога.
  std::string want = id;
  if (want.empty()) want = config::current().image_model;
  if (want.empty()) {
    std::lock_guard l(g_img_mtx);
    want = g_last_image;
  }
  if (want.empty()) {
    auto catalog = image::scan(image_models_dir_resolved());
    if (catalog.size() == 1) want = catalog[0].id;
  }
  if (want.empty()) {
    err = "модель не выбрана: задайте image-model или передайте id (POST /v0/image/load)";
    return false;
  }
  return load_pipeline(want, err);
}

void image_unload() {
  std::lock_guard l(g_img_mtx);
  if (g_img.st == St::Unloaded && !g_img.child) return;
  logf("image: останов sd-server по запросу (модель «%s»)", g_img.model_id.c_str());
  stop_locked();
  g_img.err.clear();
}

// Удаление модели с сервера: файлы слотов + их недокачки + рецепт (если после
// удаления он больше никому не нужен — VAE и рецепты общие) и запись привязки.
// Прецедент — DELETE /v0/models: загруженную демон выгружает сам (sd-server
// держит веса замапленными — удалять файлы из-под живого ребёнка нельзя);
// занятую загрузкой — отказ: гонку со спавном не вскрываем.
bool image_remove(const std::string& id, nlohmann::json& report, std::string& err) {
  {
    std::lock_guard l(g_img_mtx);
    if (g_img.st == St::Loading) {
      err = "image-движок загружается — дождитесь или выгрузите (lhctl image unload)";
      return false;
    }
    if (g_img.st == St::Loaded && g_img.model_id == id) {
      logf("image: удаление «%s» — останавливаю загруженную модель", id.c_str());
      stop_locked();
    }
  }
  const std::string dir = image::models_dir_resolved(config::current());
  image::Binding removed;
  if (!image::binding_remove(dir, id, &removed, err)) return false;
  if (removed.recipe.empty() || removed.diffusion_model.empty()) {
    err = "запись «" + id + "» без recipe/diffusion_model — правьте llmhost-image.json руками";
    return false;
  }

  struct Item { std::string path, kind; uint64_t size; };
  std::vector<Item> items;
  auto add = [&](const std::string& rel, const char* kind) {
    // Тот же containment, что у скана (записи могли править руками после
    // последнего скана): путь записи обязан остаться внутри каталога.
    if (!lh::safe_rel_path(rel) || !lh::safe_local_filename(rel)) {
      logf("image: «%s»: небезопасный путь слота «%s» — пропущен", id.c_str(), rel.c_str());
      return;
    }
    fs::path p = fs::path(dir) / rel;
    std::error_code ec;
    fs::path canon = fs::weakly_canonical(p, ec);
    fs::path root = fs::weakly_canonical(fs::path(dir), ec);
    const std::string c = canon.string(), r = root.string();
    const bool inside = !ec && !c.empty() &&
                        (c == r || (c.size() > r.size() && c.compare(0, r.size(), r) == 0 &&
                                    c[r.size()] == '/'));
    if (!inside) {
      logf("image: «%s»: путь «%s» уходит из каталога — пропущен", id.c_str(), rel.c_str());
      return;
    }
    if (lh::file_exists(c)) items.push_back({c, kind, lh::file_size(c)});
    else logf("image: «%s»: файла уже нет: %s", id.c_str(), c.c_str());
    // Недокачка того же файла — вместе с ним.
    if (lh::file_exists(c + ".part"))
      items.push_back({c + ".part", "part", lh::file_size(c + ".part")});
  };
  add(removed.diffusion_model, "diffusion");
  add(removed.llm, "llm");
  add(removed.vae, "vae");

  // Рецепт удаляем, только если после стирания записи он никому не нужен.
  bool recipe_used = false;
  for (const auto& m : image::scan(dir))
    if (m.recipe == removed.recipe) recipe_used = true;
  const std::string recipe_path =
      (fs::path(dir) / "recipes" / (removed.recipe + ".json")).string();
  bool recipe_removed = false;
  if (!recipe_used && lh::file_exists(recipe_path)) {
    std::error_code ec;
    recipe_removed = fs::remove(recipe_path, ec) && !ec;
    if (!recipe_removed)
      logf("image: рецепт не удалился: %s (%s)", recipe_path.c_str(),
           ec ? ec.message().c_str() : "нет прав?");
  }

  uint64_t bytes = 0;
  nlohmann::json files = nlohmann::json::array();
  std::string fail;
  for (auto& it : items) {
    std::error_code ec;
    bool removed_ok = fs::remove(it.path, ec);
    if (ec) {
      fail = it.path + ": " + ec.message();
      break;
    }
    if (removed_ok) {
      bytes += it.size;
      files.push_back({{"path", it.path}, {"kind", it.kind}, {"size", it.size}});
    }
  }
  report = {{"deleted", id},
            {"recipe", recipe_removed ? nlohmann::json(removed.recipe) : nlohmann::json(nullptr)},
            {"files", files},
            {"bytes", bytes}};
  if (!fail.empty()) {
    // Запись привязки уже стерта: модель из каталога исчезла, остатки файлов —
    // честный отчёт клиенту (докачать поверх не выйдет, pull перепишет).
    err = "удалено " + std::to_string(files.size()) + " из " + std::to_string(items.size()) +
          ", останов на: " + fail;
    return false;
  }
  logf("image: «%s» удалена (%llu байт%s)", id.c_str(), (unsigned long long)bytes,
       recipe_removed ? ", рецепт удалён" : ", рецепт оставлен");
  return true;
}

bool image_upstream(int& port, std::string& err) {
  {
    std::unique_lock l(g_img_mtx);
    // Ждать чужую загрузку; свой автостарт — ниже под тем же замком.
    g_img_cv.wait_for(l, std::chrono::seconds(180), [] { return g_img.st != St::Loading; });
    reap_dead_locked();
    if (g_img.st == St::Loaded && g_img.port) {
      port = g_img.port;
      g_img.last_used = clock_::now();
      return true;
    }
    if (g_img.st == St::Loading) { // так и не дождались
      err = "sd-server всё ещё загружается";
      return false;
    }
  }
  // Не поднят: автостарт (image_load сам выберет модель и возьмёт гейт).
  if (!image_load("", err)) return false;
  std::lock_guard l(g_img_mtx);
  if (g_img.st == St::Loaded && g_img.port) {
    port = g_img.port;
    return true;
  }
  if (err.empty()) err = "image-слот не готов";
  return false;
}

void image_touch() {
  std::lock_guard l(g_img_mtx);
  g_img.last_used = clock_::now();
}

void image_init() {
  g_img_stop.store(false);
  g_img_watch = std::thread(watch_loop);
}

void image_shutdown() {
  g_img_stop.store(true);
  if (g_img_watch.joinable()) g_img_watch.join();
  std::lock_guard l(g_img_mtx);
  stop_locked();
}

} // namespace lh::sup
