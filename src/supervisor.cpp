#include "supervisor.h"
#include "sup_internal.h"

#include "config.h"
#include "profiles.h"
#include "util.h"

#include <strings.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

namespace lh::sup {

using detail::g_busy_mtx;
using detail::g_busy_set;
using detail::g_emb;
using detail::g_id_map;
using detail::g_mtx;
using detail::g_slots;
using detail::g_sys;
using detail::HSlot;
using detail::harness_registry_maps;
using detail::harness_unload_models_impl;
using detail::harness_wake;
using detail::join_ids;
using detail::load_overrides;
using detail::gpu_target_for;
using detail::note_model_used;
using detail::is_embedding_arch;
using detail::path_eq;
using detail::sanitize_id;
using detail::save_resident_state;
using detail::stop_hslot;
using detail::seed_draft_reputation;
using detail::wake_notify;

namespace {

// Потоки, которые нельзя detach: на выходе они будят g_wake_lot.
struct BgJob {
  std::thread thr;
  std::shared_ptr<std::atomic<bool>> done;
};
std::mutex g_bg_mtx;
std::vector<BgJob> g_bg;

void join_background() {
  // Вложенный spawn (ensure поднимает load) подбираем следующим кругом.
  for (;;) {
    std::vector<BgJob> all;
    {
      std::lock_guard l(g_bg_mtx);
      if (g_bg.empty()) return;
      all.swap(g_bg);
    }
    for (auto& w : all)
      if (w.thr.joinable()) w.thr.join();
  }
}

// Смена состояния. GUI-уведомлений нет (headless), все значимые переходы
// логируются вызывающими; актуальный статус — lhctl status / /v0/status.
void changed() {}

} // namespace

void track_background(std::function<void()> fn) {
  auto done = std::make_shared<std::atomic<bool>>(false);
  std::thread thr([fn = std::move(fn), done]() mutable {
    fn();
    done->store(true, std::memory_order_release);
  });
  std::lock_guard l(g_bg_mtx);
  std::vector<BgJob> keep;
  keep.reserve(g_bg.size() + 1);
  for (auto& w : g_bg) {
    if (w.done->load(std::memory_order_acquire)) {
      if (w.thr.joinable()) w.thr.join();
    } else {
      keep.push_back(std::move(w));
    }
  }
  keep.push_back(BgJob{std::move(thr), std::move(done)});
  g_bg.swap(keep);
}


nlohmann::json harness_unload_models(const std::string& id, std::string& err,
                                     bool force) {
  return harness_unload_models_impl(id, err, force);
}

void park_purge_for_path(const std::string& model_path) {
  detail::park_purge_for_path(model_path);
}

// Нота деградации — прокси вешает её на ответ чата (заголовок/поле/SSE-коммент).
// Только у живого Loaded-слота: спящему «урезанность» не приписывают.
std::string model_note(const std::string& id) {
  std::lock_guard l(detail::g_mtx);
  auto it = detail::g_slots.find(id);
  if (it == detail::g_slots.end() || it->second.st != St::Loaded) return {};
  return it->second.note;
}

bool upstream_for(const std::string& id, int& port, std::string& api_key) {
  std::lock_guard l(g_mtx);
  if (id.empty()) {
    // Модель не названа: последний использованный живой Loaded-слот.
    const HSlot* best = nullptr;
    for (auto& [sid, s] : g_slots)
      if (s.st == St::Loaded && s.child && s.child->alive() &&
          !detail::g_kv_probe.count(sid) &&
          (!best || s.last_used > best->last_used))
        best = &s;
    if (!best) return false;
    port = best->port;
    api_key = best->api_key;
    return true;
  }
  if (detail::g_kv_probe.count(id)) return false; // проба сейчас меняет ребёнка
  auto by_id = g_slots.find(id);
  if (by_id != g_slots.end() && by_id->second.st == St::Loaded && by_id->second.child &&
      by_id->second.child->alive()) {
    port = by_id->second.port;
    api_key = by_id->second.api_key;
    return true;
  }
  for (auto& [sid, s] : g_slots)
    if (path_eq(s.path, id) && !detail::g_kv_probe.count(sid) && s.st == St::Loaded &&
        s.child && s.child->alive()) {
      port = s.port;
      api_key = s.api_key;
      return true;
    }
  return false;
}

bool upstream(int& port, std::string& api_key) {
  return upstream_for({}, port, api_key);
}

bool ensure_model(const std::string& id, int timeout_ms, std::string& err) {
  WakeResult w = ensure_model_ex(id, timeout_ms);
  err = w.err;
  return w.ok;
}

WakeResult ensure_model_ex(const std::string& id, int timeout_ms) {
  if (!id.empty()) return harness_wake(id, timeout_ms, nullptr);
  // id пуст (вызов без конкретной модели): «хоть один живой Loaded-слот» —
  // поднимать конкретную модель здесь нечего (выбор за запросом).
  std::string err;
  WakeResult w;
  w.ok = ensure_running(timeout_ms, err);
  w.err = err;
  return w;
}

// Загрузка через общую будилку: выселение незанятых внутри, при занятых
// блокерах /v0/load тоже честно подождёт в очереди (вызов асинхронный —
// статус смотрят через /v0/status). Это админский вход: путь вне реестра
// (недокачанный/положенный только что) здесь разрешён, /v1-клиентам — нет
// (см. harness_wake, allow_off_registry).
bool load(const std::string& path, std::string& err) {
  WakeResult w = harness_wake(path, 600000, nullptr, /*allow_off_registry=*/true);
  err = w.err;
  return w.ok;
}

void stop(bool to_asleep) {
  // «Выгрузить всё»: занятые слоты НЕ трогаем (чужой стрим неубиваем) —
  // они уходят по своему простою; список оставленных — в журнал.
  std::string uerr;
  nlohmann::json r = harness_unload_models(std::string{}, uerr);
  if (r.contains("busy_kept"))
    logf("stop: занятые модели не выгружены: %s",
         join_ids(r["busy_kept"].get<std::vector<std::string>>(), ", ").c_str());
  if (to_asleep)
    logf("stop: незанятые модели выгружены — видеопамять освобождена. "
         "Запрос к API разбудит нужную автоматически.");
}

bool ensure_running(int timeout_ms, std::string& err) {
  // Хоть один живой Loaded-слот — «запущено». Автозагрузка резидентов — в init().
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(timeout_ms);
  for (;;) {
    bool any = false;
    {
      std::lock_guard l(g_mtx);
      for (auto& [id, s] : g_slots)
        any = any || (s.st == St::Loaded && s.child && s.child->alive());
    }
    if (any) return true;
    if (std::chrono::steady_clock::now() > deadline) {
      err = "ни одна модель не загружена";
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
}

std::string server_log_tail() {
  // Хвост последнего использованного живого слота (однослотовые читатели:
  // чат-Windows на ошибке запроса).
  std::lock_guard l(g_mtx);
  const HSlot* best = nullptr;
  for (auto& [id, s] : g_slots)
    if (s.st == St::Loaded && s.child && (!best || s.last_used > best->last_used))
      best = &s;
  if (!best || !best->child) return {};
  return best->child->log_tail(); // Child::log_tail сам под своим замком
}

std::string server_log_tail_for(const std::string& id) {
  if (!id.empty()) {
    std::lock_guard l(g_mtx);
    auto it = g_slots.find(id);
    if (it == g_slots.end())
      for (auto& [sid, s] : g_slots)
        if (path_eq(s.path, id)) { it = g_slots.find(sid); break; }
    if (it == g_slots.end() || !it->second.child) return {};
    return it->second.child->log_tail();
  }
  return server_log_tail();
}

void init() {
  profiles::load(); // замеры прошлых загрузок: планировщик сразу без слепого пятна
  seed_draft_reputation(); // чёрные списки draft из профиля: рестарт не переоткрывает вердикты
  start_watcher(); // реестр моделей: добавили/удалили файл — карта следует за ним
  start_acceptance_guard(); // спекуляция не окупается — отключается сама
  image_init(); // image-слот: сторож простоя sd-server
  // Карта реестра (id → путь, эмбеддеры, system-промпты) — сразу при старте:
  // /v1/models и подстановка модели нужны до первого запроса. Вотчер дальше
  // только догоняет изменения.
  {
    std::set<std::string> emb;
    std::map<std::string, std::string> sys;
    auto by_id = harness_registry_maps(emb, sys);
    std::lock_guard l(g_mtx);
    g_id_map = by_id;
    g_emb = emb;
    g_sys = std::move(sys);
    logf("harness: карта реестра на старте: %zu моделей", by_id.size());
  }
  // Восстановление резидентории — state.json хранит список моделей, живых на
  // момент прошлого останова демона (последняя использованная включена).
  // Грузим по одной, мелкие вперёд: планировщик с выселением разрулит, что
  // влезло; не влезание — честный лог, демон работает.
  std::vector<std::string> resident;
  {
    auto st = config::load_state();
    for (auto& p : st.resident)
      resident.push_back(config::resolve_path(p));
  }
  std::sort(resident.begin(), resident.end(), [&](const std::string& a, const std::string& b) {
    return lh::file_size(a) < lh::file_size(b);
  });
  if (resident.size() > 1)
    logf("harness: восстанавливаю резиденторию (%zu моделей)", resident.size());
  track_background([resident] {
    for (auto& p : resident) {
      if (!lh::file_exists(p)) continue;
      std::string e;
      if (!load(p, e))
        logf("harness: автозагрузка «%s» не удалась: %s", p.c_str(), e.c_str());
    }
  });
  start_idle_watchdog();
}

void reap_background() {
  auto stop_models = [] {
    // Резидентория уже в state.json — после рестарта восстановится. Дети
    // обязаны умереть: mmap-файлы моделей и карты памяти за ними.
    // Loading тоже: поток /v0/load мог ещё не дописать Loaded.
    std::vector<std::string> ids;
    {
      std::lock_guard l(g_mtx);
      for (auto& [id, s] : g_slots)
        if (s.st == St::Loaded || s.st == St::Asleep || s.st == St::Loading)
          ids.push_back(id);
    }
    for (auto& id : ids) stop_hslot(id, false, /*persist=*/false, /*force=*/true);
  };
  stop_models();
  // Потоки load зовут wake_notify. Дождаться их, пока g_wake_lot жива,
  // и добить ребёнка, которого они успели поднять уже после первого прохода.
  join_background();
  stop_models();
}

void shutdown() {
  // Вотчер первым: joinable-поток не должен переживать статические
  // деструкторы (гонка на g_applied_fp при выходе — поймал TSan).
  stop_watcher();
  stop_acceptance_guard();
  stop_idle_watchdog();
  image_shutdown(); // image-слот: останов сторожа и ребёнка sd-server
  save_resident_state(); // состав зафиксирован ДО остановок
  reap_background();
}

void touch_model(const std::string& id) {
  if (id.empty()) return;
  std::lock_guard l(g_mtx);
  // id приходит из прокси после будилки, но «неизвестная» строка (клиентское
  // «model», прошедшее default_chat_model как есть) не должна расширять карты:
  // штамп — только моделям, известным реестру или слотам. Без guard'а
  // служебные карты росли бы с каждого уникального имени (тот же класс, что
  // метрики).
  if (!g_id_map.count(id) && !g_slots.count(id)) return;
  auto it = g_slots.find(id);
  if (it != g_slots.end()) it->second.last_used = std::chrono::steady_clock::now();
}

BusyModel::BusyModel(const std::string& id) : id_(id) {
  if (id_.empty()) return;
  std::lock_guard l(g_busy_mtx);
  g_busy_set.insert(id_);
}
BusyModel::~BusyModel() {
  if (id_.empty()) return;
  bool now_idle;
  {
    std::lock_guard l(g_busy_mtx);
    g_busy_set.erase(id_);
    now_idle = g_busy_set.count(id_) == 0;
  }
  // Занятость блокера упала в ноль — ждущая в очереди пробуждения модель
  // может выселять освободившегося резидента и грузиться.
  if (now_idle) wake_notify();
}

std::string chat_model() { return default_chat_model(""); }

int model_gpu(const std::string& path) {
  std::string base = path;
  size_t slash = base.find_last_of("/\\");
  if (slash != std::string::npos) base = base.substr(slash + 1);
  if (base.size() > 5 && strcasecmp(base.c_str() + base.size() - 5, ".gguf") == 0)
    base.erase(base.size() - 5);
  config::Config cfg = config::current();
  return gpu_target_for(sanitize_id(base), path, cfg, load_overrides()).main_gpu;
}

bool embedding_arch(const std::string& arch) { return is_embedding_arch(arch); }

// Наш аналог Modelfile SYSTEM: подложить system-сообщение из переопределений
// модели, если клиент своего не прислал. true — тело переписано.
bool inject_system(const std::string& id, std::string& body) {
  std::string sys = system_prompt_for(id);
  if (sys.empty()) return false;
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(body);
  } catch (...) {
    return false;
  }
  if (!j.is_object() || !j.contains("messages") || !j["messages"].is_array()) return false;
  auto& msgs = j["messages"];
  if (!msgs.empty() && msgs[0].is_object() && msgs[0].value("role", "") == "system")
    return false; // клиент прислал свой system — он знает лучше
  msgs.insert(msgs.begin(), nlohmann::json{{"role", "system"}, {"content", sys}});
  body = j.dump();
  return true;
}

std::string route_request(std::string& body) {
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(body);
  } catch (...) {
    return {};
  }
  if (!j.is_object()) return {};
  // Подсказывать модель имеет смысл только «разговорным» запросам — чат
  // (messages) и legacy completions (prompt). Прочим (embeddings, responses)
  // молчаливая подстановка чат-модели давала бы вводящий в заблуждение
  // 501 «This server does not support embeddings» вместо честного
  // «модель не выбрана» — их ребёнок отвергнет сам.
  if (!j.contains("messages") && !j.contains("prompt")) return {};
  std::string req;
  if (j.contains("model") && j["model"].is_string()) req = j["model"].get<std::string>();
  std::string id = default_chat_model(req);
  // Обслуженная модель — «последняя использованная»: при следующем старте
  // приложения она загрузится сама. id нет в карте (неизвестный) — запоминать
  // нечего.
  if (!id.empty()) {
    std::string path;
    {
      std::lock_guard l(g_mtx);
      auto it = g_id_map.find(id);
      if (it != g_id_map.end()) path = it->second;
    }
    if (!path.empty()) note_model_used(path);
  }
  // Пустой id — подсказать нечем (неоднозначно); ребёнок сам честно ответит.
  if (id.empty() || id == req) return id;
  j["model"] = id;
  body = j.dump();
  return id;
}

void registry_changed() {
  // Состав моделей изменился (докачана/удалена): карта реестра перестраивается
  // сразу — вотчер догнал бы её через пару секунд, но новый файл хочется
  // видеть в /v1/models немедленно (вотчер сам погасит повтор по отпечатку).
  apply_registry_change();
}

// Подмена exe установщиком движка: глушим своих детей (файлы моделей должны
// освободиться).
void stop_router() {
  std::lock_guard apply_l(detail::g_apply_mtx);
  std::vector<std::string> ids;
  {
    std::lock_guard l(g_mtx);
    for (auto& [id, s] : g_slots)
      if (s.st == St::Loaded || s.st == St::Asleep) ids.push_back(id);
  }
  for (auto& id : ids) stop_hslot(id, true);
  changed();
}

} // namespace lh::sup
