#include "profiles.h"

#include "util.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <numeric>
#include <ctime>
#include <filesystem>
#include <map>
#include <mutex>
#include <set>
#include <system_error>

namespace lh::profiles {

namespace fs = std::filesystem;

namespace {

std::mutex g_mtx;
std::map<std::string, ModelProfile> g_store; // ключ make_key(id, form)
std::string g_engine;                        // релиз движка, на котором сняты замеры
// Точка кривой покрытия кэша MoE: id → (размер кэша МиБ, hit rate 0..1,
// локальность a — выводится из пары скоростей, 0 — снята из строки движка
// без вывода). Локальность маршрутизации — свойство модели, а не формы
// спавна, поэтому живёт отдельной картой, а не в ModelProfile.
struct MoePoint {
  int mib = 0;
  double hit = 0;
  double a = 0;
};
std::map<std::string, MoePoint> g_moe_point;
// Вердикт авто-режима кэша MoE (sup_plan): где петля сейчас — сетка сошлась,
// назначена проба или идёт замер базы. Телеметрия петли читает её для
// карточки; сносится сменой релиза движка вместе с замерами.
struct MoeGrid {
  std::string kind;
  int mib = 0;
  double est = 0;
  double ts = 0;
};
std::map<std::string, MoeGrid> g_moe_grid;
// Репутация draft'ов: kind («dead», «shy») → пути. Живёт и стирается вместе
// с замерами — вердикт вынесен текущему движку.
std::map<std::string, std::set<std::string>> g_bad_drafts;
// Контур качества: история прогонов суита
// per «id+форма» (потолок kQualityRuns последних прогонов на метрику) — σ
// для ε гейта считается по накопленным повторным прогонам. Качество —
// свойство (модель, движок, пин, форма): сносится note_engine вместе с
// замерами (пин входит в идентичность через установку пиннутого релиза —
// install() зовёт note_engine; см. profiles.h).
struct QualityRec {
  std::vector<double> needle, tools, code, agree;
  double ts = 0;
};
constexpr size_t kQualityRuns = 8;
std::map<std::string, QualityRec> g_quality;

void append_runs(std::vector<double>& v, double score) {
  if (score < 0.0) return; // «нет» — не провал, в историю не едет
  v.push_back(score);
  if (v.size() > kQualityRuns) v.erase(v.begin()); // хвост — свежие прогоны
}

bool known_draft_kind(const std::string& kind) {
  return kind == "dead" || kind == "shy";
}

std::string path() { return lh::app_dir() + "/model_profiles.json"; }

// Санитайзер id-части ключа — та же таблица, что у супервизорного
// sanitize_id (id приходят уже санитайзенными; это защита от ключей
// на диске): символы вне [alnum._-] → «_», «|» не переживёт — составной
// ключ не перетечёт, пустой id — «model».
std::string sanitize(const std::string& base) {
  std::string id;
  id.reserve(base.size());
  for (char c : base)
    id += (std::isalnum((unsigned char)c) || c == '.' || c == '-' || c == '_') ? c : '_';
  return id.empty() ? "model" : id;
}

// Поля записи — источник истины о форме (ключ в файле — производная).
Form form_of(const ModelProfile& p) {
  return {p.out_gpu, p.kv_q8, p.ubatch, p.cuda, p.exps_cpu, p.ncmoe, p.ngl,
          p.moe_cache, p.ot_blocks};
}

void read_record(const nlohmann::json& v, ModelProfile& p) {
  p.out_gpu = v.value("out_gpu", false);
  p.out_gpu_failed = v.value("out_gpu_failed", false);
  p.cuda = v.value("cuda", true);
  p.kv_q8 = v.value("kv_q8", true);
  p.kv_type = v.value("kv_type", std::string());
  p.ubatch = v.value("ubatch", 0);
  p.ctx = v.value("ctx", 0);
  p.overhead = v.value("overhead", (uint64_t)0);
  p.ram_bytes = v.value("ram_bytes", (uint64_t)0);
  p.exps_cpu = v.value("exps_cpu", false);
  p.ncmoe = v.value("ncmoe", 0);
  p.ngl = v.value("ngl", 0);
  p.moe_cache = v.value("moe_cache", 0);
  p.ot_blocks = v.value("ot_blocks", 0);
  p.updated = v.value("updated", 0.0);
  p.gen_tps = v.value("gen_tps", 0.0);
  p.gen_n = v.value("gen_n", 0);
  p.speed_ctx = v.value("speed_ctx", (uint32_t)0);
}

nlohmann::json write_record(const ModelProfile& p) {
  nlohmann::json j = {{"v", kVersion},
                      {"out_gpu", p.out_gpu},
                      {"out_gpu_failed", p.out_gpu_failed},
                      {"cuda", p.cuda},
                      {"kv_q8", p.kv_q8},
                      {"kv_type", p.kv_type},
                      {"ubatch", p.ubatch},
                      {"ctx", p.ctx},
                      {"overhead", p.overhead},
                      {"ram_bytes", p.ram_bytes},
                      {"exps_cpu", p.exps_cpu},
                      {"ncmoe", p.ncmoe},
                      {"ngl", p.ngl},
                      {"moe_cache", p.moe_cache},
                      {"ot_blocks", p.ot_blocks},
                      {"gen_tps", p.gen_tps},
                      {"gen_n", p.gen_n},
                      {"updated", p.updated}};
  if (p.speed_ctx) j["speed_ctx"] = p.speed_ctx; // 0 не пишем — ключ с умолчанием
  return j;
}

} // namespace

std::string make_key(const std::string& id, const Form& f) {
  // Форма в ключе в фиксированном порядке с односимвольными префиксами —
  // чтение глазами и никаких сюрпризов при сортировке.
  std::string out = sanitize(id) + "|o";
  out += f.out_gpu ? '1' : '0';
  out += "k";
  out += f.kv_q8 ? '1' : '0';
  out += "c";
  out += f.cuda ? '1' : '0';
  out += "e";
  out += f.exps_cpu ? '1' : '0';
  out += "m" + std::to_string(f.ncmoe);
  out += "n" + std::to_string(f.ngl);
  out += "u" + std::to_string(f.ubatch);
  out += "h" + std::to_string(f.moe_cache);
  out += "t" + std::to_string(f.ot_blocks);
  return out;
}

void load() {
  std::string raw = lh::read_small_file(path());
  if (raw.empty()) return;
  try {
    auto j = nlohmann::json::parse(raw);
    if (!j.is_object()) return;
    std::lock_guard l(g_mtx);
    // Формат v2: {"engine": ..., "profiles": {...}}. Плоская карта старого
    // формата (ключ = id) читается как есть; штампа нет — первая установка
    // другого релиза отметит и сотрёт такие замеры.
    nlohmann::json profs = (j.contains("profiles") && j["profiles"].is_object())
                               ? j["profiles"]
                               : j;
    if (j.contains("engine") && j["engine"].is_string())
      g_engine = j["engine"].get<std::string>();
    for (const char* kind : {"dead", "shy"})
      if (j.contains(kind) && j[kind].is_array())
        for (auto& p : j[kind])
          if (p.is_string()) g_bad_drafts[kind].insert(p.get<std::string>());
    if (j.contains("moe_cache_point") && j["moe_cache_point"].is_object())
      for (auto& [id, v] : j["moe_cache_point"].items())
        if (v.is_object() && v.contains("mib") && v.contains("hit"))
          g_moe_point[id] = {v["mib"].get<int>(), v["hit"].get<double>(),
                             v.value("a", 0.0)};
    if (j.contains("moe_grid") && j["moe_grid"].is_object())
      for (auto& [id, v] : j["moe_grid"].items()) {
        if (!v.is_object()) continue;
        MoeGrid g;
        g.kind = v.value("kind", std::string());
        g.mib = v.value("mib", 0);
        g.est = v.value("est", 0.0);
        g.ts = v.value("ts", 0.0);
        if (g.kind == "grid" || g.kind == "probe" || g.kind == "base" ||
            g.kind == "quality")
          g_moe_grid[id] = g;
      }
    if (j.contains("quality") && j["quality"].is_object())
      for (auto& [key, v] : j["quality"].items()) {
        if (!v.is_object()) continue;
        QualityRec q;
        auto arr = [&](const char* k, std::vector<double>& dst) {
          if (v.contains(k) && v[k].is_array())
            for (auto& x : v[k])
              if (x.is_number()) dst.push_back(x.get<double>());
        };
        arr("needle", q.needle);
        arr("tools", q.tools);
        arr("code", q.code);
        arr("agree", q.agree);
        q.ts = v.value("ts", 0.0);
        if (!q.needle.empty() || !q.tools.empty() || !q.code.empty() || !q.agree.empty())
          g_quality[key] = q;
      }
    for (auto& [key, v] : profs.items()) {
      if (!v.is_object()) continue;
      ModelProfile p;
      read_record(v, p);
      if (v.value("v", 1) < kVersion) continue; // чужая семантика замера
      // Оболочка только с ток/с (проба соседней формы без следа VRAM)
      // должна пережить перезапуск. Пустая запись — нет.
      if ((p.ctx == 0 || p.overhead == 0) && p.gen_n == 0) continue;
      // id-часть — до первого «|» (новый формат); форма выводится из полей
      // записи, а не из ключа: поля — источник истины, ключ — производная.
      const size_t bar = key.find('|');
      const std::string id = bar == std::string::npos ? key : key.substr(0, bar);
      g_store[make_key(id, form_of(p))] = p;
    }
  } catch (const std::exception& e) {
    logf("profiles: model_profiles.json не читается — начинаем без профилей (%s)",
         e.what());
  }
}

void save() {
  nlohmann::json profs = nlohmann::json::object();
  nlohmann::json dead = nlohmann::json::array();
  nlohmann::json shy = nlohmann::json::array();
  std::string engine;
  {
    std::lock_guard l(g_mtx);
    engine = g_engine;
    for (auto& [key, p] : g_store) profs[key] = write_record(p);
    auto dump = [&](const char* kind) {
      nlohmann::json arr = nlohmann::json::array();
      auto it = g_bad_drafts.find(kind);
      if (it != g_bad_drafts.end())
        for (auto& p : it->second) arr.push_back(p);
      return arr;
    };
    dead = dump("dead");
    shy = dump("shy");
  }
  nlohmann::json j = {{"engine", engine},
                      {"profiles", std::move(profs)},
                      {"dead_drafts", std::move(dead)},
                      {"shy_drafts", std::move(shy)}};
  {
    nlohmann::json pts = nlohmann::json::object();
    nlohmann::json grids = nlohmann::json::object();
    std::lock_guard l(g_mtx);
    for (auto& [id, pt] : g_moe_point) {
      nlohmann::json v = {{"mib", pt.mib}, {"hit", pt.hit}};
      if (pt.a > 0.0) v["a"] = pt.a;
      pts[id] = std::move(v);
    }
    for (auto& [id, g] : g_moe_grid)
      grids[id] = {{"kind", g.kind}, {"mib", g.mib}, {"est", g.est}, {"ts", g.ts}};
    nlohmann::json quals = nlohmann::json::object();
    for (auto& [key, q] : g_quality) {
      nlohmann::json v = {{"ts", q.ts}};
      auto put_arr = [&](const char* k, const std::vector<double>& src) {
        if (src.empty()) return;
        nlohmann::json a = nlohmann::json::array();
        for (double x : src) a.push_back(x);
        v[k] = std::move(a);
      };
      put_arr("needle", q.needle);
      put_arr("tools", q.tools);
      put_arr("code", q.code);
      put_arr("agree", q.agree);
      quals[key] = std::move(v);
    }
    if (!pts.empty()) j["moe_cache_point"] = std::move(pts);
    if (!grids.empty()) j["moe_grid"] = std::move(grids);
    if (!quals.empty()) j["quality"] = std::move(quals);
  }
  const std::string out = j.dump(2) + "\n";
  // tmp уникален на запись: два конкурентных save() делили один путь и
  // переименовывали друг у друга файл из-под носа.
  const std::string tmp = path() + ".tmp." + lh::random_hex(6);
  std::error_code ec;
  if (lh::write_small_file(tmp, out)) fs::rename(tmp, path(), ec);
}

std::optional<ModelProfile> lookup(const std::string& id, const Form& f) {
  std::lock_guard l(g_mtx);
  auto it = g_store.find(make_key(id, f));
  if (it == g_store.end()) return std::nullopt;
  // Амбициозная попытка (выходной на GPU) уже OOM-илась на этой форме —
  // замера нет и живьём не пробуем: планировщик идёт аналитикой.
  if (it->second.out_gpu_failed) return std::nullopt;
  // Оболочка пробы скорости — не след VRAM.
  if (it->second.ctx == 0 || it->second.overhead == 0) return std::nullopt;
  return it->second; // копия: хранилище может быть очищено сразу после
}

void put(const std::string& id, const Form& f, const ModelProfile& p) {
  {
    std::lock_guard l(g_mtx);
    ModelProfile& e = g_store[make_key(id, f)];
    const double keep_tps = e.gen_tps;
    const int keep_n = e.gen_n;
    e = p;
    // Поля формы — производная ключа: запись обязана им соответствовать,
    // иначе save/load пере-ключует её в чужой слот. Кэш и блоки -ot — новые
    // поля, вызывающие их ещё не заполняют все — коэрсируем из формы.
    e.moe_cache = f.moe_cache;
    e.ot_blocks = f.ot_blocks;
    if (p.gen_n == 0 && keep_n != 0) {
      e.gen_tps = keep_tps;
      e.gen_n = keep_n;
    }
    if (f.out_gpu) e.out_gpu_failed = false; // замер прошёл — попытка оправдалась
    e.updated = (double)time(nullptr);
  }
  save();
}

namespace {

void write_speed(const std::string& id, const Form& f, double tps, int n,
                 uint32_t speed_ctx = 0) {
  ModelProfile e;
  bool fresh = true;
  {
    std::lock_guard l(g_mtx);
    auto it = g_store.find(make_key(id, f));
    if (it != g_store.end()) {
      e = it->second;
      fresh = false;
    }
  }
  if (fresh) {
    e.out_gpu = f.out_gpu;
    e.kv_q8 = f.kv_q8;
    e.ubatch = f.ubatch;
    e.cuda = f.cuda;
    e.exps_cpu = f.exps_cpu;
    e.ncmoe = f.ncmoe;
    e.ngl = f.ngl;
    e.moe_cache = f.moe_cache;
    e.ot_blocks = f.ot_blocks;
  }
  e.gen_tps = tps;
  e.gen_n = n;
  {
    std::lock_guard l(g_mtx);
    ModelProfile& slot = g_store[make_key(id, f)];
    if (slot.ctx == 0 && slot.overhead == 0 && slot.updated == 0) {
      slot.out_gpu = f.out_gpu;
      slot.kv_q8 = f.kv_q8;
      slot.ubatch = f.ubatch;
      slot.cuda = f.cuda;
      slot.exps_cpu = f.exps_cpu;
      slot.ncmoe = f.ncmoe;
      slot.ngl = f.ngl;
      slot.moe_cache = f.moe_cache;
      slot.ot_blocks = f.ot_blocks;
    } else {
      e.out_gpu = slot.out_gpu;
      e.kv_q8 = slot.kv_q8;
      e.cuda = slot.cuda;
      e.ubatch = slot.ubatch;
      e.exps_cpu = slot.exps_cpu;
      e.ncmoe = slot.ncmoe;
      e.ngl = slot.ngl;
      e.ctx = slot.ctx;
      e.overhead = slot.overhead;
      e.kv_type = slot.kv_type;
      e.ram_bytes = slot.ram_bytes;
      e.out_gpu_failed = slot.out_gpu_failed;
    }
    e.gen_tps = tps;
    e.gen_n = n;
    // Контекст замера: sustained-сэмпл несёт фактический n_past сессии —
    // он важнее прежнего (живой замер честнее); warmup-замеры зовут с 0 —
    // полная запись хранит калибровочный ctx, пустая остаётся без контекста.
    e.speed_ctx = speed_ctx > 0 ? speed_ctx : e.speed_ctx;
    slot = e;
    slot.updated = (double)time(nullptr);
  }
  save();
}

} // namespace

void note_speed(const std::string& id, const Form& f, double tps, int n,
                uint32_t speed_ctx) {
  if (n <= 0) return;
  write_speed(id, f, tps, n, speed_ctx);
}

void mark_speed_skipped(const std::string& id, const Form& f) {
  write_speed(id, f, 0, -1);
}

void speed_of(const std::string& id, const Form& f, double& tps, int& n) {
  tps = 0;
  n = 0;
  std::lock_guard l(g_mtx);
  auto it = g_store.find(make_key(id, f));
  if (it == g_store.end()) return;
  tps = it->second.gen_tps;
  n = it->second.gen_n;
}

bool speed_settled(const std::string& id, const Form& f) {
  std::lock_guard l(g_mtx);
  auto it = g_store.find(make_key(id, f));
  if (it == g_store.end()) return false;
  return it->second.gen_n >= 128 || it->second.gen_n < 0;
}

bool kv_q8_by_speed(bool policy_q8, double tps_q8, int n_q8,
                    double tps_f16, int n_f16) {
  if (n_q8 < 128 || n_f16 < 128 || tps_q8 <= 0 || tps_f16 <= 0) return policy_q8;
  if (tps_q8 >= tps_f16 * 1.08) return true;
  if (tps_f16 >= tps_q8 * 1.08) return false;
  return policy_q8;
}

void note_failed_out(const std::string& id, const Form& f) {
  {
    std::lock_guard l(g_mtx);
    ModelProfile& e = g_store[make_key(id, f)];
    if (e.out_gpu_failed) return;
    e.out_gpu_failed = true;
    e.out_gpu = f.out_gpu; // форма попытки всегда с выходным на GPU
  }
  logf("profiles: %s: амбициозная попытка выходного слоя OOM-илась — больше не пробую",
       id.c_str());
  save();
}

void note_engine(const std::string& release) {
  if (release.empty()) return;
  {
    std::lock_guard l(g_mtx);
    if (release == g_engine) return;
    size_t verdicts = 0;
    for (auto& [kind, paths] : g_bad_drafts) verdicts += paths.size();
    logf("profiles: релиз движка %s → %s: замеров сброшено %zu%s%s",
         g_engine.empty() ? "—" : g_engine.c_str(), release.c_str(), g_store.size(),
         verdicts ? ", вердиктов по draft тоже" : "",
         g_moe_grid.empty() ? "" : ", вердиктов сетки тоже");
    g_store.clear();
    g_bad_drafts.clear();
    g_moe_grid.clear(); // вердикт вынесен замерами этого движка — следующему
                        // не переносится (точка покрытия остаётся: свойство модели)
    g_quality.clear(); // качество — свойство (модель, движок, пин, форма): пин
                       // входит в идентичность через установку пиннутого релиза
    g_engine = release;
  }
  save();
}

void note_bad_draft(const std::string& path, const char* kind) {
  if (!path.empty() && known_draft_kind(kind)) {
    std::lock_guard l(g_mtx);
    g_bad_drafts[kind].insert(path);
  }
  save();
}

void note_moe_point(const std::string& id, int cache_mib, double hit_rate,
                    double a) {
  if (id.empty() || cache_mib <= 0 || hit_rate <= 0.0 || hit_rate >= 1.0)
    return;
  {
    std::lock_guard l(g_mtx);
    g_moe_point[id] = {cache_mib, hit_rate, a};
  }
  save();
}

bool moe_point(const std::string& id, int& cache_mib, double& hit_rate) {
  std::lock_guard l(g_mtx);
  auto it = g_moe_point.find(id);
  if (it == g_moe_point.end()) return false;
  cache_mib = it->second.mib;
  hit_rate = it->second.hit;
  return true;
}

void note_moe_grid(const std::string& id, const char* kind, int cache_mib,
                   double est_ms) {
  const std::string k = kind ? kind : "";
  if (id.empty() || (k != "grid" && k != "probe" && k != "base" && k != "quality"))
    return;
  {
    std::lock_guard l(g_mtx);
    g_moe_grid[id] = {k, cache_mib, est_ms, (double)time(nullptr)};
  }
  save();
}

namespace {

// Последняя по времени пригодная запись скорости этого id с кэшем или без.
bool latest_form_speed(const std::string& id, bool want_cache, CacheProbe& out) {
  const std::string want = sanitize(id) + "|";
  std::lock_guard l(g_mtx);
  bool found = false;
  double best_updated = -1.0;
  for (const auto& [key, p] : g_store) {
    if (key.rfind(want, 0) != 0) continue;
    if ((p.moe_cache > 0) != want_cache) continue;
    if (p.gen_n < 128 || p.gen_tps <= 0) continue;
    // Запись видна паре, если у замера есть контекст: полный — калибровочный,
    // sustained-shell — фактический n_past сессии (speed_ctx).
    if (!(p.speed_ctx ? p.speed_ctx : p.ctx)) continue;
    if (p.updated <= best_updated) continue;
    best_updated = p.updated;
    out.moe_cache = p.moe_cache;
    out.ot_blocks = p.ot_blocks;
    out.exps_cpu = p.exps_cpu;
    out.ncmoe = p.ncmoe;
    out.ngl = p.ngl;
    out.kv_q8 = p.kv_q8;
    // Контекст ЗАМЕРА скорости (sustained — фактический n_past сессии;
    // warmup — калибровочный ctx записи): пара «кэш/база» сверяет его,
    // T_аттн сокращается только при равных контекстах замера.
    out.ctx = p.speed_ctx ? p.speed_ctx : p.ctx;
    out.tps = p.gen_tps;
    out.n = p.gen_n;
    out.updated = p.updated;
    found = true;
  }
  return found;
}

} // namespace

void note_quality(const std::string& id, const Form& f, const QualityRun& r) {
  if (id.empty()) return;
  {
    std::lock_guard l(g_mtx);
    QualityRec& q = g_quality[make_key(id, f)];
    append_runs(q.needle, r.needle);
    append_runs(q.tools, r.tools);
    append_runs(q.code, r.code);
    append_runs(q.agree, r.agree);
    q.ts = (double)time(nullptr);
  }
  save();
}

bool quality_of(const std::string& id, const Form& f, QualityState& out) {
  if (id.empty()) return false;
  std::lock_guard l(g_mtx);
  auto it = g_quality.find(make_key(id, f));
  if (it == g_quality.end()) return false;
  out.needle = it->second.needle;
  out.tools = it->second.tools;
  out.code = it->second.code;
  out.agree = it->second.agree;
  out.ts = it->second.ts;
  return true;
}

// σ выборки (n ≥ 2); 0 при вырожденной (все значения равны — детерминированный
// суит сходится, ε = 0 честен).
static double stddev_of(const std::vector<double>& v) {
  if (v.size() < 2) return -1.0; // «нет» — прогона мало, ε не вычислить
  const double m = std::accumulate(v.begin(), v.end(), 0.0) / (double)v.size();
  double s = 0;
  for (double x : v) s += (x - m) * (x - m);
  return std::sqrt(s / (double)(v.size() - 1));
}

bool quality_gate(const std::string& id, const Form& resident, const Form& cand,
                  std::string* why) {
  if (why) why->clear();
  QualityState qr, qc;
  const bool has_r = quality_of(id, resident, qr);
  const bool has_c = quality_of(id, cand, qc);
  // Метрика «нет» (не прогонялась) гейт не закрывает: пустота ≠ провал —
  // иначе петля умирает от отсутствия данных.
  if (!has_r || !has_c) return true;
  struct Metric {
    const char* name;
    const std::vector<double> &r, &c;
  };
  const Metric ms[] = {{"needle", qr.needle, qc.needle},
                       {"tools", qr.tools, qc.tools},
                       {"code", qr.code, qc.code},
                       {"agree", qr.agree, qc.agree}};
  for (const auto& m : ms) {
    if (m.r.empty() || m.c.empty()) continue; // метрика «нет» — гейт открыт
    double mr = std::accumulate(m.r.begin(), m.r.end(), 0.0) / (double)m.r.size();
    double mc = std::accumulate(m.c.begin(), m.c.end(), 0.0) / (double)m.c.size();
    const double sr = stddev_of(m.r), sc = stddev_of(m.c);
    // ε = 2σ по истории каждой формы (консервативно максимум); σ ещё не
    // калибрована (< 2 прогонов) — метрика не закрывает гейт: капризничающий
    // гейт отключают через месяц.
    if (sr < 0 || sc < 0) continue;
    const double eps = 2.0 * std::max(sr, sc);
    if (mr - mc > eps) {
      if (why)
        *why = std::string(m.name) + ": " +
               std::to_string((int)std::lround(mc * 100)) + "% против " +
               std::to_string((int)std::lround(mr * 100)) + "% резидента (ε " +
               std::to_string(eps) + ")";
      return false; // регресс по любой метрике — форма не принимается
    }
  }
  return true;
}

bool moe_pair_of(const std::string& id, CacheProbe& cached, CacheProbe& plain) {
  if (!latest_form_speed(id, /*want_cache=*/true, cached)) return false;
  if (!latest_form_speed(id, /*want_cache=*/false, plain)) return false;
  // Равный контекст и один класс KV: T_аттн сокращается только при них.
  const uint32_t tol = std::max<uint32_t>(512u, plain.ctx / 50);
  if (cached.ctx + tol < plain.ctx || plain.ctx + tol < cached.ctx) return false;
  if (cached.kv_q8 != plain.kv_q8) return false;
  return true;
}

bool has_cached_form_speed(const std::string& id) {
  CacheProbe tmp;
  return latest_form_speed(id, /*want_cache=*/true, tmp);
}

LoopState loop_of(const std::string& id) {
  LoopState out;
  if (id.empty()) return out;
  {
    const std::string want = sanitize(id) + "|";
    std::lock_guard l(g_mtx);
    for (const auto& [key, p] : g_store) {
      if (key.rfind(want, 0) != 0) continue;
      out.moe = out.moe || p.moe_cache > 0 || p.ncmoe > 0 || p.exps_cpu;
      if (p.gen_n >= 128 && p.gen_tps > 0) {
        ++out.forms;
        out.speed_pts.push_back({p.updated, p.gen_tps});
      }
    }
    if (auto it = g_moe_point.find(sanitize(id)); it != g_moe_point.end()) {
      out.point_mib = it->second.mib;
      out.point_hit = it->second.hit;
      out.a = it->second.a;
    }
    if (auto it = g_moe_grid.find(sanitize(id)); it != g_moe_grid.end()) {
      out.verdict = it->second.kind;
      out.verdict_mib = it->second.mib;
      out.verdict_ms = it->second.est;
      out.verdict_ts = it->second.ts;
    }
  }
  // Пара — теми же правилами, что планировщик; время берём у свежайшей из
  // двух записей (возраст пары снаружи меряет «петля мертва»).
  CacheProbe cp, bp;
  if (moe_pair_of(id, cp, bp)) {
    out.pair = true;
    out.pair_ctx = cp.ctx;
    out.pair_ts = std::max(cp.updated, bp.updated);
  }
  std::sort(out.speed_pts.begin(), out.speed_pts.end()); // по возрасту замера
  return out;
}

std::vector<std::string> bad_drafts(const char* kind) {
  std::vector<std::string> out;
  if (!known_draft_kind(kind)) return out;
  std::lock_guard l(g_mtx);
  auto it = g_bad_drafts.find(kind);
  if (it != g_bad_drafts.end())
    out.assign(it->second.begin(), it->second.end());
  return out;
}

} // namespace lh::profiles
