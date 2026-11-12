#include "bench.h"

#include "config.h"
#include "engine.h"
#include "fit.h"
#include "hw.h"
#include "modelreg.h"
#include "proc.h"
#include "supervisor.h"
#include "util.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace lh::bench {

namespace {

using nlohmann::json;

struct BuildResult {
  std::string build;   // id сборки или «vulkan (-ub 256)» для варианта ubatch
  bool ub_variant = false;
  double gen_tps = 0, pp_tps = 0;
  std::string error;
};

struct State {
  std::string state = "idle"; // idle | running | error
  std::string current;        // что замеряем сейчас
  std::vector<BuildResult> results;
  std::string winner;
  double winner_tps = 0;
  std::string error;
};
std::mutex g_mtx;
State g_state;

void set(State s) {
  std::lock_guard l(g_mtx);
  g_state = std::move(s);
  // GUI-уведомлений нет (headless): результат — lhctl status / журнал.
}

// Промпт ~150 токенов: одинаковый для всех сборок, температура 0 — детерминизм.
const char* BENCH_PROMPT =
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor "
    "jugs. How vexingly quick daft zebras jump! Sphinx of black quartz, judge my vow. "
    "Bright vixens jump; dozy fowl quack. Quick zephyrs blow, vexing daft Jim. Two "
    "driven jocks help fax my big quiz. Five quacking zephyrs jolt my wax bed. The "
    "job requires extra pluck and zeal from every young wage earner.";

// Одна сборка: спавн с одинаковыми «честными» параметрами, генерация, timings.
bool measure(const std::string& exe, const std::string& model, int ub, BuildResult& out,
             std::string& err) {
  int port = lh::free_tcp_port(8300, 8399);
  if (!port) { err = "нет свободного порта"; return false; }
  std::string key = lh::random_hex(16);
  if (key.empty()) { err = "не удалось сгенерировать api-key (getrandom)"; return false; }
  // Замер — на одной карте (самой свободной): кросс-карточный обмен портил бы
  // сравнение сборок, а дефолтный делёж «поровну» на картах разного размера
  // ловил бы OOM на меньшей.
  hw::Gpu gpu0 = hw::query(); // база: сколько было свободно до спавна
  for (const auto& c : hw::query_all())
    if (c.nvidia && c.vram_total && c.vram_free > gpu0.vram_free) gpu0 = c;
  std::vector<std::string> args = {
      "--host", "127.0.0.1", "--port", std::to_string(port),
      "--api-key", key, "-m", model, "-c", "2048", "-ngl", "999", "--no-webui"};
  if (gpu0.vram_total > 0) {
    args.push_back("--device");
    args.push_back("CUDA" + std::to_string(gpu0.index));
  }
  if (ub > 0) { args.push_back("-ub"); args.push_back(std::to_string(ub)); }

  lh::proc::Child c;
  if (!c.start(exe, args, lh::engine_child_env(exe))) {
    err = "llama-server не запустился";
    return false;
  }
  auto w = lh::proc::wait_health(c, port, 180000);
  if (w != lh::proc::Wait::Healthy) {
    std::string tail = c.log_tail();
    c.stop();
    // Процесс умер — драйвер вернёт его VRAM с задержкой, а следующий замер
    // спавнится сразу: не подождав, следующая сборка ловит ложный OOM.
    sup::vram_settle(gpu0.vram_free + lh::file_size(model),
                     gpu0.vram_total > 0 ? gpu0.index : -1);
    err = w == lh::proc::Wait::Oom
              ? "модель не влезла при -ngl 999"
              : "нет /health: " + lh::proc::last_meaningful_line(tail);
    return false;
  }

  httplib::Client cli("127.0.0.1", port);
  cli.set_connection_timeout(std::chrono::seconds(3));
  cli.set_read_timeout(std::chrono::seconds(240));
  httplib::Headers hdrs = {{"Authorization", "Bearer " + key}};
  // Прогрев: свежий Vulkan-процесс компилирует шейдеры на первом запросе —
  // без него первая сборка всегда казалась бы медленной.
  cli.Post("/completion", hdrs,
           R"({"prompt":"Hi","n_predict":1,"cache_prompt":false})", "application/json");
  auto res = cli.Post("/completion", hdrs,
                      std::string(R"({"prompt":")") + BENCH_PROMPT +
                          R"(","n_predict":128,"cache_prompt":false,"temperature":0})",
                      "application/json");
  c.stop();
  sup::vram_settle(gpu0.vram_free + lh::file_size(model), // VRAM вернётся не мгновенно
                   gpu0.vram_total > 0 ? gpu0.index : -1);
  if (!res || res->status != 200) {
    err = res ? "запрос генерации: код " + std::to_string(res->status)
              : "запрос генерации не прошёл";
    return false;
  }
  try {
    auto j = json::parse(res->body);
    auto t = j.at("timings");
    out.gen_tps = t.value("predicted_per_second", 0.0);
    out.pp_tps = t.value("prompt_per_second", 0.0);
  } catch (...) {
    err = "timings в ответе не нашлись";
    return false;
  }
  if (out.gen_tps <= 0) {
    err = "нулевой скорость генерации";
    return false;
  }
  return true;
}

// Автовыбор модели, когда слот пуст: языковая (не mmproj/MTP/эмбеддер), которая
// целиком влезает в свободную VRAM при контексте замера (2048) — из влезающих
// самая крупная (показательнее); не влезает ни одна — самая мелкая (получит
// честный OOM от measure вместо молчаливого выбора).
std::string pick_model() {
  auto list = modelreg::scan(config::models_dir_resolved(config::current()));
  std::vector<const modelreg::Model*> llms;
  for (auto& m : list)
    if (m.meta.n_layers > 0 && !sup::embedding_arch(m.arch) &&
        !modelreg::is_mtp_module(m.path) && !modelreg::is_mmproj_module(m.path) &&
        !modelreg::is_shard_continuation(m.path))
      llms.push_back(&m);
  if (llms.empty()) return {};
  hw::Gpu gpu = hw::query(); // та же карта, на которую measure пиннит ребёнка
  for (const auto& c : hw::query_all())
    if (c.nvidia && c.vram_total && c.vram_free > gpu.vram_free) gpu = c;
  const modelreg::Model* best = nullptr;
  const modelreg::Model* fallback = nullptr;
  auto weights = [](const modelreg::Model* m) {
    int i = 0, c = 1;
    modelreg::shard_info(m->path, i, c);
    return c > 1 ? modelreg::shard_group_bytes(m->path) : m->size;
  };
  for (auto* m : llms) {
    uint64_t sz = weights(m);
    if (!fallback || sz < weights(fallback)) fallback = m;
    if (gpu.vram_free && fit::estimate(m->meta, sz, 2048, gpu, false).fits &&
        (!best || sz > weights(best)))
      best = m;
  }
  return (best ? best : fallback)->path;
}

} // namespace

bool busy() {
  std::lock_guard l(g_mtx);
  return g_state.state == "running";
}

std::string status_json() {
  std::lock_guard l(g_mtx);
  json j = {{"state", g_state.state}, {"current", g_state.current},
            {"winner", g_state.winner}, {"winner_tps", g_state.winner_tps}};
  json arr = json::array();
  for (auto& r : g_state.results) {
    json e = {{"build", r.build}, {"gen_tps", r.gen_tps}, {"pp_tps", r.pp_tps}};
    if (!r.error.empty()) e["error"] = r.error;
    arr.push_back(e);
  }
  j["results"] = arr;
  if (!g_state.error.empty()) j["error"] = g_state.error;
  return j.dump(2);
}

bool start(const std::string& model_in, std::string& err) {
  {
    std::lock_guard l(g_mtx);
    if (g_state.state == "running") {
      err = "замер уже идёт";
      return false;
    }
    // Помечаем сразу, под замком: второй быстрый POST не проскочит, пока
    // рабочий поток ещё не начал (раньше state ставился в потоке — гонка).
    g_state.state = "running";
    g_state.current.clear();
  }
  auto fail_fast = [&](const std::string& msg) {
    { std::lock_guard l(g_mtx); g_state = State{}; }
    err = msg;
    return false;
  };
  // Загрузка в процессе — выгружать её чужими руками нельзя, это транзит.
  if (sup::state() == sup::St::Loading)
    return fail_fast("идёт загрузка модели — дождитесь её окончания и повторите замер");
  // Установка движка глушит замер (гейт «загрузка/установка»): дети замера
  // спавнятся из подменяемых бинарников. Чтение для честного 400 сразу;
  // авторитетная бронь — в рабочем потоке ниже.
  if (engine::install_in_progress())
    return fail_fast("идёт установка движка — замер недоступен до её конца");
  // Явный и заранее битый путь — честный синхронный 400: детерминированный
  // провал не прячем за 202 (тот же принцип, что у /v0/load → 404 сразу).
  // Автоподбор (путь пуст) остаётся в рабочем потоке — там честный OOM.
  if (!model_in.empty() && !lh::file_exists(model_in))
    return fail_fast("файл модели не найден: " + model_in);
  std::vector<engine::BuildInfo> builds;
  for (auto& b : engine::list())
    if (b.installed) builds.push_back(b);
  if (builds.empty()) return fail_fast("нет установленных сборок llama-server");

  std::thread([model_in, builds] {
    // Гейт «загрузка/установка не совпадают»: замер спавнит llama-server —
    // берёт билет загрузки на всю свою работу (см. engine.h). Установка,
    // начавшаяся между 400/202 и этим местом, отменит замер честной ошибкой.
    engine::GateTicket gate;
    std::string gate_err;
    if (!gate.acquire(engine::GateTicket::Kind::Load, gate_err)) {
      State st;
      st.state = "error";
      st.error = gate_err;
      set(st);
      return;
    }
    // Замеру нужен файл модели на диске и свободная видеопамять. Загруженная
    // мешает только второй части — выгружаем её сами (в слот: на ней и мерим),
    // а требовать «сначала загрузите» при пустом слоте — ловушка-22.
    if (sup::state() == sup::St::Loaded) {
      logf("bench: выгружаю модель — замеру нужна свободная видеопамять");
      std::string slot;
      sup::state(slot);
      const hw::Gpu g0 = hw::query();
      sup::stop(true);
      sup::vram_settle(g0.vram_free + (slot.empty() ? 0 : lh::file_size(slot)));
    }

    std::string model = model_in; // явный путь проверен синхронно в start()
    if (model.empty()) {
      sup::state(model); // спящая модель остаётся в слоте — на ней и замерим
      if (model.empty() || !lh::file_exists(model)) model = pick_model();
    }
    if (model.empty() || !lh::file_exists(model)) {
      State st;
      st.state = "error";
      st.error = "нечего замерять: в папке моделей нет ни одной языковой модели "
                 "(GGUF) — скачайте модель: lhctl pull <репозиторий>";
      set(st);
      return;
    }

    State st;
    logf("bench: замер %zu сборок на %s", builds.size(), model.c_str());

    std::vector<BuildResult> results;
    std::string winner;
    double winner_tps = 0;
    for (auto& b : builds) {
      {
        std::lock_guard l(g_mtx);
        g_state.current = b.id;
      }
      BuildResult r;
      r.build = b.id;
      std::string e;
      if (measure(engine::exe_for(b.id), model, 0, r, e)) {
        results.push_back(r);
        if (r.gen_tps > winner_tps) {
          winner_tps = r.gen_tps;
          winner = b.id;
        }
      } else {
        r.error = e;
        results.push_back(r);
        logf("bench: %s — не вышло: %s", b.id.c_str(), e.c_str());
      }
    }

    // Тонкая подстройка ubatch для победителя: -ub 256 иногда быстрее на Vulkan.
    int best_ub = 0;
    if (!winner.empty()) {
      {
        std::lock_guard l(g_mtx);
        g_state.current = winner + " (-ub 256)";
      }
      BuildResult r;
      r.build = winner + " (-ub 256)";
      r.ub_variant = true;
      std::string e;
      if (measure(engine::exe_for(winner), model, 256, r, e)) {
        results.push_back(r);
        if (r.gen_tps > winner_tps * 1.05) best_ub = 256;
      }
    }

    // Победитель живёт в конфиге: следующие загрузки идут на нём.
    // set_current ДО save: инвариант «конфиг на диске == current()», иначе
    // вотчер примет собственную запись bench за внешнюю правку.
    config::Config cfg = config::current();
    if (!winner.empty()) cfg.engine_build = winner;
    cfg.ubatch = best_ub;
    config::set_current(cfg);
    config::save(cfg);

    st.results = results;
    st.winner = winner;
    st.winner_tps = winner_tps;
    if (winner.empty()) {
      st.state = "error";
      st.error = "ни одна сборка не смогла запуститься на этой модели";
    } else {
      st.state = "idle";
      logf("bench: замер готов: быстрее %s — %.1f ток/с. Настройка сохранена.",
           winner.c_str(), winner_tps);
      logf("bench: победил %s (%.1f ток/с), ubatch=%d", winner.c_str(), winner_tps, best_ub);
    }
    set(st);
  }).detach();
  return true;
}

} // namespace lh::bench
