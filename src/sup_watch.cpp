#include "supervisor.h"
#include "sup_internal.h"

#include <httplib.h>

#include "metrics.h"
#include "profiles.h"
#include "util.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace lh::sup {

using clock_ = std::chrono::steady_clock;

using detail::changed;
using detail::HSlot;
using detail::g_mtx;
using detail::g_fp_mtx;
using detail::g_applied_fp;
using detail::g_id_map;
using detail::g_emb;
using detail::g_sys;
using detail::g_slots;
using detail::g_busy_set;
using detail::g_busy_mtx;
using detail::park_before_idle_stop;
using detail::g_apply_mtx;
using detail::g_wake_lot;
using detail::g_shy_drafts;
using detail::g_shy_drafts_mtx;
using detail::g_draft_by_id;
using detail::g_draft_by_id_mtx;
using detail::path_eq;
using detail::steady_ms;
using detail::overrides_conf_path;
using detail::overrides_path;
using detail::harness_registry_maps;
using detail::stop_hslot;
using detail::fetch_child_slot_tokens;
using detail::schedule_form_probe;
using detail::ProbeCtx;
using detail::g_kv_probe;
using detail::save_resident_state;
using detail::wake_notify;

static std::thread g_idle;
static std::atomic<bool> g_idle_stop{false};

void start_idle_watchdog() {
  // Как вотчер реестра: joinable, иначе поток переживает деструктор
  // g_wake_lot. Сон кусочками — shutdown не ждёт целых 5 с.
  if (g_idle.joinable()) return;
  g_idle = std::thread([] {
    while (!g_idle_stop) {
      for (int i = 0; i < 50 && !g_idle_stop; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      if (g_idle_stop) break;
      config::Config cfg = config::current();
      // Сторож простоя per-слот: занятый слот не выселяется никогда.
      if (cfg.unload_after_min > 0) {
        // Окно простоя per-модельное: precision-first даёт удвоенное окно —
        // «право на неспешность» агента, который думает между шагами. Свойство
        // ФЛАГА, не чекпоинтов: динамика «окно меняется, когда кто-то создал
        // чекпоинт» — худший вид неожиданности.
        // Чекпоинты есть без флага, флаг есть без чекпоинтов — поведение
        // одинаково предсказуемо.
        const auto ov = detail::load_overrides();
        std::set<std::string> busy_snap;
        {
          std::lock_guard lb(g_busy_mtx);
          busy_snap = g_busy_set;
        }
        std::vector<std::string> to_sleep, dead;
        {
          std::lock_guard l(g_mtx);
          for (auto& [id, s] : g_slots) {
            if (s.st != St::Loaded) continue;
            if (!s.child || !s.child->alive()) {
              // Ребёнок умер сам (краш, OOM-killer) — слот в сон, будилка поднимет.
              dead.push_back(id);
              continue;
            }
            if (busy_snap.count(id)) {
              s.last_used = clock_::now(); // идёт генерация — не выгружаем посреди ответа
              continue;
            }
            int win = cfg.unload_after_min;
            const detail::ModelOverrides* o = detail::overrides_for(ov, id, s.path);
            if (o && o->precision_first) win = std::max(win, win * 2);
            auto idle = std::chrono::duration_cast<std::chrono::minutes>(
                            clock_::now() - s.last_used).count();
            if (idle >= win) to_sleep.push_back(id);
          }
        }
        for (auto& id : dead) {
          std::lock_guard l(g_mtx);
          auto it = g_slots.find(id);
          if (it == g_slots.end() || it->second.st != St::Loaded) continue;
          logf("harness: llama-server «%s» умер сам — слот в сон, следующий запрос разбудит",
               id.c_str());
          it->second.child.reset();
          it->second.port = 0;
          it->second.st = St::Asleep;
          changed();
        }
        for (auto& id : to_sleep) {
          // Перепроверка перед выстрелом: запрос мог начаться между снимком
          // и этим местом — выгружать модель посреди генерации нельзя.
          {
            std::lock_guard l2(g_busy_mtx);
            if (g_busy_set.count(id) > 0) continue;
          }
          // Окно для журнала — то же, по которому сработали: у
          // precision-first-моделей оно удвоено.
          int win = cfg.unload_after_min;
          {
            std::lock_guard l(g_mtx);
            auto it = g_slots.find(id);
            if (it != g_slots.end()) {
              const detail::ModelOverrides* o =
                  detail::overrides_for(ov, id, it->second.path);
              if (o && o->precision_first) win = std::max(win, win * 2);
            }
          }
          // Парковка KV слотов перед остановом (best-effort, [general]
          // slot-park): внутри — сохранение достойных слотов и повторная
          // перепроверка занятости/простоя после секундного HTTP. false —
          // модель ожила, останов отменить. Выселение по плану (load_harness)
          // и models_max паркую́т тоже (require_idle=false — вытеснение под
          // память/лимит).
          if (!park_before_idle_stop(id)) continue;
          logf("harness: простой %d+ мин — выгружаю «%s» (запрос к API разбудит автоматически)",
               win, id.c_str());
          // false — запрос начался между перепроверкой и остановом: атомарная
          // проверка занятости внутри stop_hslot отменила останов — слот
          // дорабатывает, сторож вернётся к нему в следующий тик.
          if (!stop_hslot(id, true)) continue;
        }
      }
      // Периодический тычок очереди: на случай потерянного будильника
      // (например, блокер выгружен внешним путём мимо наших нотификаторов).
      if (g_wake_lot.depth_total() > 0) wake_notify();
    }
  });
}

void stop_idle_watchdog() {
  g_idle_stop = true;
  if (g_idle.joinable()) g_idle.join();
}

// ---- вотчер реестра ----------------------------------------------------------
// Пользователь волен добавлять файлы в папку моделей и удалять оттуда — карта
// реестра обязана за этим следовать. Папка сканируется раз в две секунды
// (разбор заголовков кэшируется по размеру и времени — дёшево); изменение
// применяется, когда снимок устоялся: два одинаковых скана подряд означают,
// что файл докопирован, — растущий файл трогать нельзя. При удалениях слоты
// пропавших моделей выселяются (занятые — дорабатывают до выгрузки).

std::string registry_fingerprint() {
  config::Config cfg = config::current();
  auto list = lh::modelreg::scan(config::models_dir_resolved(cfg));
  std::string fp;
  for (auto& m : list) {
    size_t slash = m.path.find_last_of("/\\");
    std::string f = slash == std::string::npos ? m.path : m.path.substr(slash + 1);
    // mtime важен: файл могли заменить другим того же размера.
    fp += f + "|" + std::to_string(m.size) + "|" + m.arch + "|" +
          m.quant + "|" + std::to_string(m.mtime) + "\n";
  }
  // Правки переопределений тоже меняют карту — вотчер должен их замечать
  // (llmhost.conf — основной источник, model_overrides.json — legacy).
  // Хэш КОНТЕНТА, не mtime/размер: записи самого демона (пин, bench, /v0
  // settings) без семантических изменений не должны трогать слоты.
  std::error_code ec;
  for (const std::string& f : {overrides_conf_path(), overrides_path()}) {
    if (!fs::exists(f, ec)) continue;
    fp += "#" + f + "|" +
          std::to_string(std::hash<std::string>{}(lh::read_small_file(f))) + "\n";
  }
  return fp;
}

// [general] из llmhost.conf — глобальные настройки, тот же файл, что и
// per-model переопределения (решение юзера: один источник правды). Вотчер и
// все писатели идут через apply_registry_change, поэтому применяются здесь:
// diff с current() → set_current. Порты/bind/auth/cors живьём не перебиндить
// — их изменение логируется подсказкой, а current() не трогается (он честно
// описывает живой процесс до рестарта). Лупа нет: демон пишет conf только
// после set_current — diff пуст.
static void apply_general_from_conf() {
  config::Config parsed = config::load_from(config::conf_path());
  config::Config cur = config::current();
  auto a = config::settings_table(parsed), b = config::settings_table(cur);
  if (a == b) return;
  std::string hot, cold;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].second == b[i].second) continue;
    std::string& dst = config::needs_daemon_restart(a[i].first) ? cold : hot;
    dst += (dst.empty() ? "" : ", ") + a[i].first;
  }
  if (hot.empty()) {
    logf("config: llmhost.conf [general] изменён (%s) — вступит после рестарта: "
         "systemctl --user restart llmhostd",
         cold.c_str());
    return;
  }
  config::set_current(parsed);
  logf("config: llmhost.conf [general] изменён (%s) — применяю на лету%s",
       hot.c_str(),
       cold.empty() ? "" : ("; «" + cold + "» тоже изменены — проверь после рестарта демона")
                              .c_str());
}

// Тело apply_registry_change без замка: enable_mtp зовёт его под уже взятым
// g_apply_mtx, чтобы догнать свежие overrides ДО собственной загрузки.
void apply_registry_change_locked() {
  std::string fp_now = registry_fingerprint();
  {
    std::lock_guard l(g_fp_mtx);
    if (fp_now == g_applied_fp) return; // уже применено — не перестраиваем зря
  }
  // Сначала глобальные настройки ([general]) — загрузки ниже идут уже с
  // новыми значениями.
  apply_general_from_conf();
  // Догоняем карту id→путь и выселяем слоты моделей, файлы которых исчезли.
  // Занятый слот удалённого файла не убиваем: пусть доработает (загрузить
  // его заново всё равно не выйдет — честный отказ).
  std::map<std::string, std::string> old_map;
  {
    std::lock_guard l(g_mtx);
    old_map = g_id_map;
  }
  std::set<std::string> emb;
  std::map<std::string, std::string> sys;
  auto by_id = harness_registry_maps(emb, sys);
  std::vector<std::string> removed;
  for (auto& [id, p] : old_map) {
    bool still = false;
    for (auto& [i2, p2] : by_id) still = still || path_eq(p2, p);
    if (!still) removed.push_back(id);
  }
  // Снимок до g_mtx: g_busy_set мутируют прокси под g_busy_mtx.
  // Держать оба замка сразу нельзя — сторож берёт их в порядке
  // busy, затем g_mtx.
  std::set<std::string> busy_snap;
  {
    std::lock_guard lb(g_busy_mtx);
    busy_snap = g_busy_set;
  }
  {
    std::lock_guard l(g_mtx);
    g_id_map = by_id;
    g_emb = emb;
    g_sys = std::move(sys);
    for (auto& id : removed) {
      auto it = g_slots.find(id);
      if (it == g_slots.end()) continue;
      if (it->second.st == St::Loaded && busy_snap.count(id) > 0) {
        logf("harness: файл «%s» удалён, но модель занята — работаем до выгрузки",
             id.c_str());
        continue;
      }
      logf("harness: файл модели «%s» удалён — слот выгружен", id.c_str());
    }
  }
  for (auto& id : removed) {
    bool busy;
    {
      std::lock_guard lb(g_busy_mtx);
      busy = g_busy_set.count(id) > 0;
    }
    if (!busy) stop_hslot(id, false);
    std::lock_guard l(g_mtx);
    g_slots.erase(id); // слот без файла не resurrection-кандидат
  }
  logf("harness: реестр изменился — карта обновлена (%zu моделей%s)", by_id.size(),
       removed.empty() ? "" : ", есть удаления");
  changed();
  save_resident_state();
  {
    std::lock_guard l(g_fp_mtx);
    g_applied_fp = registry_fingerprint(); // вотчеру видно: изменения уже применены
  }
}

void apply_registry_change() {
  std::lock_guard apply_l(g_apply_mtx);
  apply_registry_change_locked();
}

static std::thread g_watcher;                 // вотчер реестра — joinable
static std::atomic<bool> g_watch_stop{false}; // флаг останова для shutdown

void start_watcher() {
  {
    std::lock_guard l(g_fp_mtx);
    g_applied_fp = registry_fingerprint();
  }
  // Откреплённый поток пережил бы статические деструкторы при выходе (TSan:
  // гонка деструктора g_applied_fp с живым вотчером) — держим joinable и
  // гасим в stop_watcher() из shutdown().
  if (g_watcher.joinable()) return;
  g_watcher = std::thread([] {
    std::string pending;
    int stable = 0;
    while (!g_watch_stop) {
      // Сон кусочками: shutdown не должен ждать полных 2 с.
      for (int i = 0; i < 20 && !g_watch_stop; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      if (g_watch_stop) break;
      std::string fp = registry_fingerprint();
      std::string applied;
      {
        std::lock_guard l(g_fp_mtx);
        applied = g_applied_fp;
      }
      if (fp == applied) {
        pending.clear();
        stable = 0;
        continue;
      }
      if (fp == pending) {
        if (++stable >= 2) { // снимок устоялся — файл докопирован
          apply_registry_change(); // сам обновит g_applied_fp
          pending.clear();
          stable = 0;
        }
      } else {
        pending = std::move(fp); // копируется — ждём устойствия
        stable = 0;
      }
    }
  });
}

void stop_watcher() {
  g_watch_stop = true;
  if (g_watcher.joinable()) g_watcher.join();
}

// ---- Сторож acceptance спекуляции -----------------------------------------
//
// llama-server печатает по завершении запроса «draft acceptance = X
// ( A accepted / B generated)» (формат плавает между пинами — парсер в
// util.cpp понимает несколько вариантов). Сторож раз в 30 с собирает
// последние отчёты в окно ПО СУММЕ СЧЁТЧИКОВ (не среднее rate — среднее
// врёт) и при устойчиво низком приёме отстраняет авто-draft: путь уходит
// в g_shy_drafts, слот в сон — будилка поднимет модель уже без спекуляции.
// До рестарта демона draft не вернётся — как dead-draft.
//
// Предохранители: окно набрало draft-accept-min-tokens черновиков; три
// плохих окна ПОДРЯД с новыми данными между ними (watermark по task-номеру
// — без новых отчётов счётчик не растёт); срабатывание не чаще раза в
// 30 минут. Ручной model-draft не отстраняется никогда (переопределения
// сильнее умной настройки) — одно предупреждение на сеанс ребёнка.

static std::thread g_guard;                    // сторож acceptance — joinable
static std::atomic<bool> g_guard_stop{false};  // флаг останова для shutdown

// ---- Сторож acceptance спекуляции, per-слот ----------------------------------
// Каждый ребёнок — отдельный процесс со своим кольцом stderr: атрибуция
// отчётов точная по слоту. Логика окна — сумма шести отчётов, минимум
// токенов, три плохих окна подряд, не чаще раза в 30 минут. Приговор: путь
// draft'а в g_shy_drafts, слот в сон — будилка поднимет модель уже без
// отстранённого черновика (find_draft shy-пути пропускает), перегружать
// немедленно не нужно.

// Разобрать хвост лога ребёнка: отчёты acceptance (a/b, b>0) и task-номера.
static void parse_acceptance_reports(
    const std::string& tail, std::vector<std::pair<long long, long long>>& reports,
    std::vector<long long>& tasks) {
  size_t pos = 0;
  while (pos <= tail.size()) {
    size_t eol = tail.find('\n', pos);
    std::string line = tail.substr(pos, eol == std::string::npos
                                            ? std::string::npos
                                            : eol - pos);
    pos = eol == std::string::npos ? tail.size() + 1 : eol + 1;
    double r;
    long long a, b;
    if (!lh::parse_draft_acceptance(line, r, a, b) || b <= 0) continue;
    long long task = -1;
    size_t t = line.find("task ");
    if (t != std::string::npos) task = std::atoll(line.c_str() + t + 5);
    reports.push_back({a, b});
    tasks.push_back(task);
  }
}

struct AcceptanceVerdict {
  bool triggered = false; // отстранить draft
  long long A = 0, B = 0; // окно (для логов)
  double acc = 0;
  int taken = 0;          // отчётов в окне
};

// Политика acceptance одна (раньше жила в двух экземплярах — тик по слотам
// и inline-ветка роутера — и молча расходилась).
struct AcceptanceWatch {
  long long pid = -1;            // pid ребёнка: смена — кольцо свежее, с нуля
  long long watermark = -1;      // task последнего разобранного отчёта
  int bad_streak = 0;            // подряд плохих окон с новыми данными
  long long last_trigger_ms = 0; // троттлинг приговора (30 мин)
  std::string manual_warned;     // кому уже предупредили про ручной draft

  AcceptanceVerdict feed(const std::string& who, long long child_pid,
                         const std::string& tail, bool auto_attached,
                         const config::Config& cfg) {
    AcceptanceVerdict v;
    if (child_pid != pid) {
      pid = child_pid;
      watermark = -1;
      bad_streak = 0;
      manual_warned.clear();
    }
    std::vector<std::pair<long long, long long>> reports;
    std::vector<long long> tasks;
    parse_acceptance_reports(tail, reports, tasks);
    if (reports.empty() || tasks.back() == watermark) return v; // нет нового
    watermark = tasks.back();
    long long A = 0, B = 0;
    int taken = 0;
    for (int i = (int)reports.size() - 1; i >= 0 && taken < 6; --i, ++taken) {
      A += reports[i].first;
      B += reports[i].second;
    }
    v.A = A;
    v.B = B;
    v.taken = taken;
    if (B < (long long)cfg.draft_accept_min_tokens) return v; // статистики мало
    v.acc = (double)A / (double)B;
    if (v.acc >= cfg.draft_accept_floor) {
      if (bad_streak)
        logf("acceptance: %s: окно оздоровилось — %d/%d (%.2f)", who.c_str(), (int)A,
             (int)B, v.acc);
      bad_streak = 0;
      return v;
    }
    if (!auto_attached) {
      // Ручной model-draft — юзерское переопределение: предупреждаем раз на
      // сеанс ребёнка, действие не совершаем.
      if (manual_warned != who) {
        manual_warned = who;
        logf("acceptance: %s: приём черновиков %d/%d (%.2f) ниже порога %.2f — "
             "draft задан вручную (model-draft в llmhost.conf), НЕ отстраняю; "
             "уберите ключ, если спекуляция не нужна",
             who.c_str(), (int)A, (int)B, v.acc, cfg.draft_accept_floor);
      }
      return v;
    }
    if (++bad_streak < 3) {
      logf("acceptance: %s: окно %d/%d (%.2f) ниже порога %.2f (%d/3)", who.c_str(),
           (int)A, (int)B, v.acc, cfg.draft_accept_floor, bad_streak);
      return v;
    }
    long long now_ms = steady_ms();
    if (last_trigger_ms && now_ms - last_trigger_ms < 30LL * 60 * 1000) {
      bad_streak = 0; // уже отстраняли недавно — спамить не надо
      return v;
    }
    last_trigger_ms = now_ms;
    bad_streak = 0;
    v.triggered = true;
    return v;
  }
};
std::map<std::string, AcceptanceWatch> g_hguard;
std::mutex g_hguard_mtx;

void harness_acceptance_tick(const config::Config& cfg) {
  std::map<std::string, HSlot> slots;
  {
    std::lock_guard l(g_mtx);
    slots = g_slots;
  }
  for (auto& [id, s] : slots) {
    if (s.st != St::Loaded || !s.child || !s.child->alive()) continue;
    // Точка кривой покрытия кэша MoE: движок печатает статистику при
    // остановке («llama_moe_cache: ... hit rate = X%»), кольцо ребёнка
    // держит хвост — ловим и учим локальность маршрутизации модели
    // (планировщик выводит a для аналитического выбора размера кэша).
    {
      int cache_mib = 0;
      {
        std::lock_guard pl(lh::sup::detail::g_plan_mtx);
        auto it = lh::sup::detail::g_plan_by_id.find(id);
        if (it != lh::sup::detail::g_plan_by_id.end()) cache_mib = it->second.moe_cache;
      }
      if (cache_mib > 0) {
        const std::string tail = s.child->log_tail();
        size_t pos = tail.find("llama_moe_cache:");
        if (pos != std::string::npos) {
          const size_t hr = tail.find("hit rate = ", pos);
          if (hr != std::string::npos) {
            const double hit = std::atof(tail.c_str() + hr + 11) / 100.0;
            if (hit > 0.0 && hit < 1.0) {
              lh::profiles::note_moe_point(id, cache_mib, hit);
              logf("moe-cache: %s: точка покрытия снята — кэш %d МиБ, "
                   "hit rate %.0f%%", id.c_str(), cache_mib, hit * 100.0);
            }
          }
        }
      }
    }
    std::string draft_path;
    bool auto_attached = false;
    {
      std::lock_guard l(g_draft_by_id_mtx);
      auto it = g_draft_by_id.find(id);
      if (it == g_draft_by_id.end()) continue;
      draft_path = it->second.first;
      auto_attached = it->second.second;
    }
    {
      std::lock_guard l(g_shy_drafts_mtx);
      if (g_shy_drafts.count(draft_path)) continue; // уже отстранён
    }
    std::lock_guard gl(g_hguard_mtx);
    AcceptanceWatch& g = g_hguard[id];
    const auto v =
        g.feed(id, (long long)s.child->pid(), s.child->log_tail(), auto_attached, cfg);
    if (!v.triggered) continue;
    {
      std::lock_guard l(g_shy_drafts_mtx);
      g_shy_drafts.insert(draft_path);
    }
    // Переживёт рестарт: вердикт «приёма нет» вынесен этому движку, но
    // переоткрывать его каждой первой загрузкой после рестарта — зряшный
    // круг slow-сессий и приговоров (так было с Flash-Next).
    lh::profiles::note_bad_draft(draft_path, "shy");
    logf("acceptance: %s: приём черновиков устойчиво низкий (%d/%d = %.2f < %.2f) — "
         "draft «%s» отстранён до рестарта демона; слот в сон, первый запрос "
         "поднимет модель без спекуляции",
         id.c_str(), (int)v.A, (int)v.B, v.acc, cfg.draft_accept_floor,
         draft_path.c_str());
    stop_hslot(id, true);
  }
  // Чистка состояний ушедших слотов, чтобы карта не росла вечно.
  std::lock_guard gl(g_hguard_mtx);
  for (auto it = g_hguard.begin(); it != g_hguard.end();) {
    if (slots.count(it->first)) ++it;
    else it = g_hguard.erase(it);
  }
}

void start_acceptance_guard() {
  if (g_guard.joinable()) return;
  g_guard = std::thread([] {
    while (!g_guard_stop) {
      for (int i = 0; i < 300 && !g_guard_stop; ++i) // 30 с кусочками
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      if (g_guard_stop) break;
      config::Config cfg = config::current();
      if (cfg.draft_accept_floor <= 0.0) continue;
      harness_acceptance_tick(cfg);
    }
  });
}

void stop_acceptance_guard() {
  g_guard_stop = true;
  if (g_guard.joinable()) g_guard.join();
}

// ---- Сборщик sustained-скоростей --------------------------------------------
//
// Живой трафик — бесплатный бенч на реальном контексте: warmup-замеры слепы
// к нему (128 токенов «Hi» на пустом контексте), а пара «кэш/база» из них
// даёт верное НАПРАВЛЕНИЕ выбора и неверный абсолютный прогноз (замер
// 13.11: a=4.23 сходится на 4k, хвост 260k невидим). Раз в ~10 с опрашиваем
// /metrics живых Loaded-детей (строго на чтение, HTTP вне g_mtx — как
// engine_metrics) и дельта-сэмплним декодирование: прирост ≥128 токенов
// окна даёт sustained-скорость формы фактического спавна — она перезаписывает
// warmup-запись: живой замер важнее,
// warmup остаётся бутстрапом первой загрузки).
//
// Гейты (план): только Loaded (Loading/Stopping не пишем); только слоты без
// draft (форма не несёт draft-состояния — пара честна лишь при одинаковом;
// консервативно не смешиваем). Контекст сэмпла — максимум n_past сессии
// слота на момент сэмпла (/slots ребёнка): едет в
// speed_ctx записи, пару сверяющую, и не трогает калибровочный ctx надбавки.

struct SustainState {
  metrics::SustainDelta delta;
  long long last_log_ms = 0; // троттлинг журнала: длинная сессия не топит log.txt
};
std::map<std::string, SustainState> g_sustain;
std::mutex g_sustain_mtx;

static std::thread g_sustain_thread;          // сборщик sustained — joinable
static std::atomic<bool> g_sustain_stop{false};

void start_sustained_collector() {
  if (g_sustain_thread.joinable()) return;
  g_sustain_thread = std::thread([] {
    while (!g_sustain_stop) {
      for (int i = 0; i < 100 && !g_sustain_stop; ++i) // ~10 с кусочками
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      if (g_sustain_stop) break;
      // Снимок слотов под g_mtx; HTTP и профили — вне замка.
      struct Kid {
        std::string id;
        int port = 0;
        std::string key;
        long long pid = 0;
        profiles::Form form{};
      };
      // draft-слоты отпадают до g_mtx: вложенный лок (g_mtx → draft) не нужен.
      std::set<std::string> drafted;
      {
        std::lock_guard ld(g_draft_by_id_mtx);
        for (auto& [id, _] : g_draft_by_id) drafted.insert(id);
      }
      std::vector<Kid> kids;
      {
        std::lock_guard l(g_mtx);
        for (auto& [id, s] : g_slots) {
          // Гейт состояний: пишем только в Loaded; формы нет (слот поднят
          // старым демоном) — замеру некуда ехать честно.
          if (s.st != St::Loaded || !s.child || !s.child->alive() || s.port <= 0 ||
              s.api_key.empty() || !s.spawn_form)
            continue;
          if (drafted.count(id)) continue; // draft не в форме — не смешиваем
          kids.push_back({id, s.port, s.api_key, (long long)s.child->pid(), *s.spawn_form});
        }
      }
      for (auto& k : kids) {
        httplib::Client cli("127.0.0.1", k.port);
        cli.set_connection_timeout(std::chrono::milliseconds(300));
        cli.set_read_timeout(std::chrono::milliseconds(500));
        auto res = cli.Get("/metrics", httplib::Headers{{"Authorization", "Bearer " + k.key}});
        if (!res || res->status != 200 || res->body.empty()) continue;
        const nlohmann::json series = metrics::parse_engine(res->body);
        if (!series.contains("tokens_predicted_total") ||
            !series.contains("tokens_predicted_seconds_total"))
          continue; // эмбеддеры и дети без телеметрии — мимо
        metrics::SustainDelta::Sample sm;
        {
          std::lock_guard ls(g_sustain_mtx);
          sm = g_sustain[k.id].delta.feed(k.pid, series["tokens_predicted_total"].get<double>(),
                                    series["tokens_predicted_seconds_total"].get<double>());
        }
        if (sm.n < metrics::kSustainMinTokens || sm.tps <= 0) continue;
        // Контекст сэмпла: максимум n_past сессии слота.
        uint32_t speed_ctx = 0;
        for (const auto& [sidx, n] : fetch_child_slot_tokens(k.port, k.key))
          if (n > 0 && (uint64_t)n > speed_ctx) speed_ctx = (uint32_t)n;
        profiles::note_speed(k.id, k.form, sm.tps, sm.n, speed_ctx);
        {
          std::lock_guard ls(g_sustain_mtx);
          SustainState& st = g_sustain[k.id];
          const long long now_ms = steady_ms();
          if (!st.last_log_ms || now_ms - st.last_log_ms >= 60 * 1000) {
            st.last_log_ms = now_ms;
            logf("sustain: «%s»: %d токенов окна с %.1f ток/с (ctx %u)%s",
                 k.id.c_str(), sm.n, sm.tps, speed_ctx,
                 speed_ctx ? "" : ", контекст не снялся");
          }
        }
      }
      // Чистка состояний ушедших слотов.
      std::lock_guard ls(g_sustain_mtx);
      for (auto it = g_sustain.begin(); it != g_sustain.end();) {
        bool gone = true;
        for (auto& k : kids)
          if (k.id == it->first) { gone = false; break; }
        if (gone) it = g_sustain.erase(it);
        else ++it;
      }
    }
  });
}

void stop_sustained_collector() {
  g_sustain_stop = true;
  if (g_sustain_thread.joinable()) g_sustain_thread.join();
}

// ---- Исследователь форм -----------------------------------------------------
//
// Проба KV меряет только соседний класс KV. Остальные формы (соседний грант
// кэша, соседний ncmoe) замеряются здесь — обобщение того же цикла
// (schedule_form_probe, параметризация, не копия лестницы). Одно окно
// простоя — один прогон: окно = половина unload-after-min, и отсчёт
// перезапускается пробой (она возвращает слоту свежий last_used), так что
// темп исследования сам следует за простоем.
//
// Гейты (условия AcceptanceWatch + план): резидент ровно один (проба
// спиливает его веса — второй резидент на общих картах получил бы чужую
// загрузку посреди окна), очередь пробуждения пуста, не busy, без живой
// KV-пробы; только MoE-вынос без блоков -ot (правило запасной карты живое —
// не пересобирается). Ручной moe-cache-mib (0/N) — решение юзера: исследо-
// ватель работает только в авто-режиме.
//
// Стоп (план): пара собрана И сетка сошлась (телеметрия потока 1), либо
// шесть форм на модель, либо два окна подряд без улучшения медианы.

struct ExplorerState {
  std::set<std::string> tried;   // кандидаты сеанса: неудача не вечна, но и не повтор
  size_t last_forms = 0;         // осевших форм на прошлом окне (для столлов)
  int stalls = 0;                // окон подряд без улучшения медианы
  long long last_launch_ms = 0;  // темп окна: отменённый прогон тоже потратил окно
};
std::map<std::string, ExplorerState> g_explorer;
std::mutex g_explorer_mtx;

static std::thread g_explorer_thread;         // исследователь форм — joinable
static std::atomic<bool> g_explorer_stop{false};

void start_form_explorer() {
  if (g_explorer_thread.joinable()) return;
  g_explorer_thread = std::thread([] {
    while (!g_explorer_stop) {
      for (int i = 0; i < 300 && !g_explorer_stop; ++i) // 30 с кусочками
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      if (g_explorer_stop) break;
      config::Config cfg = config::current();
      if (cfg.moe_cache_mib != -1 || cfg.unload_after_min <= 0) continue;
      if (g_wake_lot.depth_total() > 0) continue; // ждущих будить посреди пересадки нельзя
      const long long window_ms =
          60LL * 1000 * std::max(1, cfg.unload_after_min / 2);
      const auto window = std::chrono::minutes(std::max(1, cfg.unload_after_min / 2));
      // Снимок единственного резидента под g_mtx: busy — под своим замком
      // (порядок busy → g_mtx, как у сторожей).
      std::set<std::string> busy_snap;
      {
        std::lock_guard lb(g_busy_mtx);
        busy_snap = g_busy_set;
      }
      std::string id, key;
      int port = 0, warm_n = 0;
      std::shared_ptr<ProbeCtx> pc;
      profiles::Form f;
      // Гейт единственного резидента с допуском верификатора: проба
      // спиливает веса worker'а — второй резидент на
      // общих картах получил бы чужую загрузку посреди окна; назначенный
      // quality-verifier — штатный второй резидент, проба его не трогает.
      const std::string vmodel = cfg.quality_verifier;
      size_t residents = 0;
      bool verifier_live = false;
      {
        std::lock_guard l(g_mtx);
        for (auto& [sid, s] : g_slots) {
          if (s.st != St::Loaded) continue;
          ++residents;
          if (!vmodel.empty() && sid == vmodel) verifier_live = true;
        }
        if (residents != 1 && !(residents == 2 && verifier_live)) continue;
        for (auto& [sid, s] : g_slots) {
          if (s.st != St::Loaded || !s.child || !s.child->alive()) continue;
          if (busy_snap.count(sid) || g_kv_probe.count(sid)) continue; // занято/проба
          if (!vmodel.empty() && sid == vmodel) continue; // верификатор не пробуем
          if (!s.probe || !s.spawn_form) continue;
          if (clock_::now() - s.last_used < window) continue; // не отгулял окно простоя
          id = sid;
          port = s.port;
          key = s.api_key;
          warm_n = s.warm_n;
          pc = s.probe;
          f = *s.spawn_form;
        }
      }
      if (id.empty() || !pc) continue;
      // Только MoE-вынос без -ot: кандидаты меняют кэш/ncmoe, правило
      // запасной карты живое (свобода карты) — не пересобирается.
      if (!(f.moe_cache > 0 || f.ncmoe > 0 || f.exps_cpu) || f.ot_blocks > 0) continue;
      {
        std::lock_guard le(g_explorer_mtx);
        ExplorerState& st = g_explorer[id];
        const long long now_ms = steady_ms();
        if (st.last_launch_ms && now_ms - st.last_launch_ms < window_ms) continue;
        st.last_launch_ms = now_ms; // окно потрачено, чем бы прогон ни кончился
      }
      // Гейт холодного промпта — тот же, что внутри пробы (n > хвоста
      // прогрева): тёплый слот проба всё равно отменит; не жечь окно на
      // заведомую отмену (завершённый ход диалога держит n_past до выгрузки).
      {
        bool warm = false;
        for (const auto& [sidx, n] : fetch_child_slot_tokens(port, key))
          if (n > warm_n) {
            warm = true;
            break;
          }
        if (warm) continue;
      }

      profiles::LoopState ls = profiles::loop_of(id);
      {
        std::lock_guard le(g_explorer_mtx);
        ExplorerState& st = g_explorer[id];
        // Стоп: петля сошлась (пара есть, сетка вынесла вердикт) или форм
        // достаточно — план исследователя исполнен.
        // Стоп: петля сошлась (пара есть, сетка вынесла вердикт), вердикт
        // «качество» (форма отставлена гейтом — до смены резидента/движка,
        // вердикт сносится note_engine) или форм достаточно — план
        // исследователя исполнен.
        if ((ls.pair && ls.verdict == "grid") || ls.verdict == "quality" ||
            ls.forms >= 6) {
          if (ls.verdict == "quality")
            logf("explore: «%s»: сетка отставила кандидата гейтом качества — "
                 "петля остановлена (эскалация человеку — карточка модели)",
                 id.c_str());
          g_explorer.erase(id);
          continue;
        }
        // Новая осевшая форма с прошлого окна — улучшила медиану остальных?
        if (ls.speed_pts.size() > st.last_forms && st.last_forms > 0) {
          std::vector<double> others;
          for (size_t i = 0; i + 1 < ls.speed_pts.size(); ++i)
            others.push_back(ls.speed_pts[i].second); // свежайшая — последняя
          std::sort(others.begin(), others.end());
          const double med = others.empty()
                                 ? 0.0
                                 : (others.size() % 2 ? others[others.size() / 2]
                                                      : (others[others.size() / 2 - 1] +
                                                         others[others.size() / 2]) / 2.0);
          const double newest = ls.speed_pts.back().second;
          if (newest <= med) {
            if (++st.stalls >= 2) {
              logf("explore: «%s»: два окна без улучшения медианы — исследователя "
                   "форм остановил (%zu форм осело)", id.c_str(), (size_t)ls.forms);
              g_explorer.erase(id);
              continue;
            }
          } else {
            st.stalls = 0;
          }
        }
        st.last_forms = ls.speed_pts.size();

        // Кандидаты: соседний грант кэша (±шаг сетки), соседний ncmoe (±4).
        // Базовую форму не плодим — для моделей без пары её меряет
        // чередующаяся проба авто-ветки plan_for.
        const int layers = (int)pc->meta.n_layers;
        struct Cand {
          profiles::Form form;
          int ncmoe, cache;
          std::string what;
        };
        std::vector<Cand> cands;
        auto push = [&](profiles::Form c, int ncmoe, int cache, std::string what) {
          if (profiles::make_key(id, c) == profiles::make_key(id, f)) return;
          if (profiles::speed_settled(id, c)) return; // форма уже осела
          if (st.tried.count(profiles::make_key(id, c))) return;
          cands.push_back({c, ncmoe, cache, std::move(what)});
        };
        if (f.moe_cache > 0) {
          const int step = std::max(512, f.moe_cache / 4);
          if (f.moe_cache - step >= 512)
            push(profiles::Form{f.out_gpu, f.kv_q8, f.ubatch, f.cuda, f.exps_cpu, f.ncmoe,
                                f.ngl, f.moe_cache - step, f.ot_blocks},
                 -1, f.moe_cache - step, "кэш " + std::to_string(f.moe_cache - step) + " МиБ");
          push(profiles::Form{f.out_gpu, f.kv_q8, f.ubatch, f.cuda, f.exps_cpu, f.ncmoe,
                              f.ngl, f.moe_cache + step, f.ot_blocks},
               -1, f.moe_cache + step, "кэш " + std::to_string(f.moe_cache + step) + " МиБ");
        }
        if (f.ncmoe > 0) {
          if (f.ncmoe - 4 >= 0)
            push(profiles::Form{f.out_gpu, f.kv_q8, f.ubatch, f.cuda, f.exps_cpu, f.ncmoe - 4,
                                f.ngl, f.moe_cache, f.ot_blocks},
                 f.ncmoe - 4, -1, "ncmoe " + std::to_string(f.ncmoe - 4));
          if (f.ncmoe + 4 < layers)
            push(profiles::Form{f.out_gpu, f.kv_q8, f.ubatch, f.cuda, f.exps_cpu, f.ncmoe + 4,
                                f.ngl, f.moe_cache, f.ot_blocks},
                 f.ncmoe + 4, -1, "ncmoe " + std::to_string(f.ncmoe + 4));
        }
        if (cands.empty()) {
          g_explorer.erase(id); // соседей нет (или все осели) — делать нечего
          continue;
        }
        const Cand& c = cands.front();
        st.tried.insert(profiles::make_key(id, c.form));
        logf("explore: «%s»: окно простоя — замеряю соседнюю форму (%s)",
             id.c_str(), c.what.c_str());
        schedule_form_probe(id, port, pc,
                            {/*cand=*/c.form, /*ncmoe=*/c.ncmoe, /*cache=*/c.cache,
                             /*kv_q8=*/f.kv_q8, /*keep_on_win=*/false});
      }
    }
  });
}

void stop_form_explorer() {
  g_explorer_stop = true;
  if (g_explorer_thread.joinable()) g_explorer_thread.join();
}

} // namespace lh::sup
