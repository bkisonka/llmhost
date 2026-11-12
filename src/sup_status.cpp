// Агрегаты состояния супервизора: St state(), status_json для control API и
// трея, списки моделей для чата (chat_models / default_chat_model /
// models_state), system-промпты, id по пути. Загрузки и будилки —
// sup_harness; диспетчеры и фон — supervisor.cpp.
#include "supervisor.h"
#include "sup_internal.h"

#include "engine.h"
#include "metrics.h"
#include "version.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace lh::sup {

using detail::HSlot;
using detail::g_busy_mtx;
using detail::g_busy_set;
using detail::g_emb;
using detail::g_id_map;
using detail::g_mtx;
using detail::g_slots;
using detail::g_sys;
using detail::g_wake_lot;
using detail::harness_states;
using detail::path_eq;

St state() {
  std::lock_guard l(g_mtx);
  bool any_loading = false, any_loaded = false;
  for (auto& [id, s] : g_slots) {
    any_loading = any_loading || s.st == St::Loading;
    any_loaded = any_loaded || s.st == St::Loaded;
  }
  if (any_loading) return St::Loading;
  return any_loaded ? St::Loaded : St::Unloaded;
}

St state(std::string& model_path_out) {
  // «Модель» для однослотовых читателей (трей, классический UI): последняя
  // использованная среди Loaded. Агрегат — здесь же, под тем же замком
  // (state() без параметра берёт g_mtx сам — вложенный лок = дедлок).
  std::lock_guard l(g_mtx);
  const HSlot* best = nullptr;
  bool any_loading = false, any_loaded = false;
  for (auto& [id, s] : g_slots) {
    any_loading = any_loading || s.st == St::Loading;
    any_loaded = any_loaded || s.st == St::Loaded;
    if (s.st == St::Loaded && (!best || s.last_used > best->last_used)) best = &s;
  }
  model_path_out = best ? best->path : std::string();
  if (any_loading) return St::Loading;
  return any_loaded ? St::Loaded : St::Unloaded;
}

// Общие секции status_json (gpu/engine/gpus).
// Системная RAM: used = total − available (семантика free: page cache в «used»
// не считается — десятки ГиБ mmap-экспертов MoE живут в кэше и ядром
// вытесняемы; честная цифра «сколько занято»).
static nlohmann::json gpu_json(const hw::Gpu& gpu) {
  return {
    {"nvidia", gpu.nvidia},
    {"name", gpu.name},
    {"cc", gpu.cc_major >= 0 ? std::to_string(gpu.cc_major) + "." + std::to_string(gpu.cc_minor)
                             : ""},
    {"vram_free", gpu.vram_free},
    {"vram_total", gpu.vram_total},
    {"ram_total", gpu.ram_total},
    {"ram_avail", gpu.ram_avail},
    {"vulkan", gpu.vulkan},
    {"build", lh::hw::build_name(gpu.build)},
  };
}
// Что реально пойдёт в запуск: gpu.build — «на что способно железо»,
// дети могут ехать на другой сборке (engine_build, явный путь).
static nlohmann::json engine_json(const config::Config& cfg) {
  return {{"build", engine::active_build(cfg)},
          {"release", engine::pinned_release()},
          {"selected", cfg.engine_build}};
}
static nlohmann::json gpus_json() {
  nlohmann::json gpus = nlohmann::json::array();
  for (auto& c : hw::query_all())
    gpus.push_back({{"index", c.index},
                    {"name", c.name},
                    {"vram_total", c.vram_total},
                    {"vram_free", c.vram_free},
                    {"ram_total", c.ram_total},
                    {"ram_avail", c.ram_avail}});
  return gpus;
}

nlohmann::json status_json() {
  config::Config cfg = config::current();
  hw::Gpu gpu = hw::query();
  nlohmann::json j;
  std::map<std::string, HSlot> slots;
  {
    std::lock_guard l(g_mtx);
    slots = g_slots;
  }
  std::set<std::string> emb;
  {
    std::lock_guard l(g_mtx);
    emb = g_emb;
  }
  bool any_loading = false, any_loaded = false;
  for (auto& [id, s] : slots) {
    any_loading = any_loading || s.st == St::Loading;
    any_loaded = any_loaded || s.st == St::Loaded;
  }
  j["state"] = any_loading ? "loading" : any_loaded ? "loaded" : "unloaded";
  j["pid"] = 0; // единого ребёнка нет: по процессу на модель (в models[].pid)
  j["port"] = 0;
  j["harness_mode"] = true;
  j["models_max"] = cfg.models_max;
  std::map<std::string, std::string> id_map;
  {
    std::lock_guard l(g_mtx);
    id_map = g_id_map;
  }
  nlohmann::json arr = nlohmann::json::array();
  // Весь реестр с живыми состояниями слотов поверх (контракт /v0/status.models
  // — все известные модели, не только резиденты).
  for (auto& [id, path] : id_map) {
    auto it = slots.find(id);
    const HSlot* s = it == slots.end() ? nullptr : &it->second;
    nlohmann::json m = {{"id", id},
                        {"state", !s ? "unloaded"
                                  : s->st == St::Loading  ? "loading"
                                  : s->st == St::Loaded   ? "loaded"
                                                          : "unloaded"},
                        {"failed", s && s->st == St::Error},
                        {"embedding", emb.count(id) > 0}};
    if (s) {
      m["pid"] = s->child ? (int)s->child->pid() : 0;
      m["port"] = s->port;
      m["ctx"] = s->ctx;
      m["np"] = s->np;
      // Аргументы спавна — диагностика: «что едет ребёнку» видно только так.
      if (s->child)
        for (auto& a : s->child->args()) m["args"].push_back(a);
      if (!s->err.empty()) m["error"] = s->err;
      if (s->st == St::Loaded && !s->note.empty()) m["note"] = s->note;
      // Парковка KV: файлы слотов модели (если есть) — наблюдаемость парка.
      if (s->parkable && !s->path.empty()) {
        const uint64_t b = detail::parked_bytes_for(s->path);
        if (b > 0) m["parked_mib"] = b >> 20;
      }
    }
    arr.push_back(std::move(m));
  }
  j["models"] = arr;
  // Последняя использованная.
  std::string prim;
  {
    std::lock_guard l(g_mtx);
    const HSlot* best = nullptr;
    for (auto& [id, s] : g_slots)
      if (s.st == St::Loaded && (!best || s.last_used > best->last_used)) best = &s;
    if (best) prim = best->path;
  }
  size_t slash = prim.find_last_of("/\\");
  j["model"] = {{"path", prim},
                {"file", slash == std::string::npos ? prim : prim.substr(slash + 1)}};
  j["busy"] = false;
  {
    std::lock_guard lb(g_busy_mtx);
    j["busy"] = !g_busy_set.empty();
  }
  j["queue_depth"] = (int)g_wake_lot.depth_total();
  j["unload_after_min"] = cfg.unload_after_min;
  j["version"] = LH_VERSION;
  j["gpu"] = gpu_json(gpu);
  j["engine"] = engine_json(cfg);
  j["gpus"] = gpus_json();
  return j;
}


std::string system_prompt_for(const std::string& id) {
  if (id.empty()) return {};
  std::lock_guard l(g_mtx);
  auto it = g_sys.find(id);
  return it == g_sys.end() ? std::string() : it->second;
}

std::vector<ChatModel> chat_models() {
  std::vector<ChatModel> out;
  std::set<std::string> emb;
  {
    std::lock_guard l(g_mtx);
    emb = g_emb;
  }
  // Состояния — из слотов; остальной реестр — «unloaded» (выбор запомнится,
  // первый запрос поднимет модель).
  auto st = harness_states();
  std::lock_guard l(g_mtx);
  for (auto& [id, p] : g_id_map) {
    auto it = st.find(id);
    out.push_back({id, it == st.end() ? "unloaded" : it->second.state,
                   emb.count(id) > 0});
  }
  return out;
}

std::string default_chat_model(const std::string& requested) {
  {
    std::lock_guard l(g_mtx);
    if (!requested.empty() && requested != "any") return requested;
  }
  std::string single;
  std::set<std::string> emb;
  {
    std::lock_guard l(g_mtx);
    emb = g_emb;
    if (g_id_map.size() == 1) single = g_id_map.begin()->first;
  }
  if (!single.empty() && !emb.count(single)) return single;
  // Последний шанс: последняя использованная не-эмбеддерная Loaded-модель.
  std::lock_guard l(g_mtx);
  const HSlot* best = nullptr;
  for (auto& [id, s] : g_slots)
    if (s.st == St::Loaded && !emb.count(id) && (!best || s.last_used > best->last_used))
      best = &s;
  return best ? best->id : std::string();
}

nlohmann::json models_state() {
  auto st = harness_states();
  std::set<std::string> emb;
  std::map<std::string, std::string> park_paths; // id → путь модели с парковкой
  {
    std::lock_guard l(g_mtx);
    emb = g_emb;
    for (auto& [sid, hs] : g_slots)
      if (hs.parkable && !hs.path.empty()) park_paths[sid] = hs.path;
  }
  nlohmann::json o = nlohmann::json::object();
  for (auto& [id, s] : st) {
    nlohmann::json m = {{"state", s.state},
                        {"failed", s.failed},
                        {"embedding", emb.count(id) > 0}};
    if (!s.note.empty()) m["note"] = s.note;
    auto pit = park_paths.find(id);
    if (pit != park_paths.end()) {
      const uint64_t b = detail::parked_bytes_for(pit->second);
      if (b > 0) m["parked_mib"] = b >> 20;
    }
    o[id] = m;
  }
  return o;
}

std::string id_for_path(const std::string& path) {
  std::lock_guard l(g_mtx);
  for (auto& [i, p] : g_id_map)
    if (path_eq(p, path)) return i;
  return {};
}

nlohmann::json engine_metrics() {
  // Точная телеметрия живых детей (llamacpp:* из их /metrics): опрос строго
  // на чтение, слоты только под g_mtx, HTTP — вне замка (ребёнок может
  // молчать — таймаут короткий, рендер /v0/metrics не обязан виснуть).
  struct Child {
    std::string id;
    int port;
    std::string key;
  };
  std::vector<Child> kids;
  {
    std::lock_guard l(g_mtx);
    for (auto& [id, hs] : g_slots)
      if (hs.st == St::Loaded && hs.port > 0 && !hs.api_key.empty())
        kids.push_back({hs.id, hs.port, hs.api_key});
  }
  nlohmann::json arr = nlohmann::json::array();
  for (auto& c : kids) {
    httplib::Client cli("127.0.0.1", c.port);
    cli.set_connection_timeout(std::chrono::milliseconds(300));
    cli.set_read_timeout(std::chrono::milliseconds(500));
    auto res = cli.Get("/metrics", httplib::Headers{{"Authorization", "Bearer " + c.key}});
    if (!res || res->status != 200 || res->body.empty()) continue;
    arr.push_back(nlohmann::json{{"model", c.id},
                                 {"series", metrics::parse_engine(res->body)}});
  }
  return arr;
}

} // namespace lh::sup
