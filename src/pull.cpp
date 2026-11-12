#include "pull.h"

#include "catalog.h"
#include "catalog_est.h"
#include "catalog_alias.h"
#include "config.h"
#include "download.h"
#include "image.h"
#include "image_catalog.h"
#include "modelreg.h"
#include "supervisor.h"
#include "util.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <set>
#include <strings.h>
#include <thread>
#include <vector>

namespace lh::pull {

namespace fs = std::filesystem;

namespace {

struct State {
  std::string state = "idle"; // idle | resolving | downloading | done | error
  std::string repo;
  std::string file;           // какой файл качается сейчас
  uint64_t done = 0, total = 0;
  double speed = 0;
  std::string error;
  // Скачано без SHA256 (только мелочь ниже порога LFS — для неё API хеша
  // не даёт). Крупное без хеша не качается вовсе: fail-closed.
  bool unverified = false;
  std::string kind = "llm";   // llm | image — какой pull в статусе
};
std::mutex g_mtx;
State g_state;
// Запрос на остановку: ставится cancel() (DELETE /v0/pull, CtrlX TUI,
// lhctl pull stop), читается тиком прогресса download_one — false глушит
// curl; рабочие потоки поглощают флаг checkpoint'ами и в финале.
std::atomic<bool> g_cancel{false};

void set(State s) {
  std::lock_guard l(g_mtx);
  g_state = std::move(s);
  // GUI-уведомлений нет (headless): актуальный статус — lhctl pull / /v0/pull.
}

// Поглотить флаг отмены (checkpoint рабочей фазы между сетевыми вызовами).
// true — остановиться: флаг мог прийти, пока фаза шла.
bool take_cancel() { return g_cancel.exchange(false); }

// Конечное состояние по остановке: .part остаётся на диске, повторный pull
// докачивает; error несёт человеческое «остановлено» — его же покажут lhctl/TUI.
void set_stopped(const std::string& repo, const std::string& file, uint64_t done,
                 uint64_t total, const std::string& kind) {
  State s;
  s.state = "stopped";
  s.repo = repo;
  s.file = file;
  s.done = done;
  s.total = total;
  s.error = "остановлено";
  s.kind = kind;
  set(s);
  logf("pull: %s — остановлено по запросу", repo.c_str());
}

std::string human(uint64_t b) {
  char buf[32];
  if (b >= 1ull << 30) snprintf(buf, sizeof(buf), "%.1f ГБ", b / 1073741824.0);
  else if (b >= 1ull << 20) snprintf(buf, sizeof(buf), "%.0f МБ", b / 1048576.0);
  else snprintf(buf, sizeof(buf), "%llu КБ", (unsigned long long)(b >> 10));
  return buf;
}

// Фильтр может быть списком альтернатив через «|»: «Base-*|mmproj.gguf».
// «|» в путях HF невозможен (safe_rel_path режет), разделитель безопасен.
bool filter_match(const std::string& filter, const std::string& path) {
  size_t start = 0;
  for (;;) {
    size_t bar = filter.find('|', start);
    std::string part =
        filter.substr(start, bar == std::string::npos ? std::string::npos : bar - start);
    if (!part.empty() && lh::wildcard_match(part.c_str(), path.c_str())) return true;
    if (bar == std::string::npos) return false;
    start = bar + 1;
  }
}

// Фильтр указывает на сам модуль, а не на квант модели.
bool module_filter(const std::string& filter) {
  std::string low = filter;
  for (char& c : low) c = (char)std::tolower((unsigned char)c);
  size_t slash = low.find_last_of("/\\");
  std::string base = slash == std::string::npos ? low : low.substr(slash + 1);
  if (base.rfind("mtp-", 0) == 0 || base.rfind("mmproj", 0) == 0) return true;
  return low.find("/mtp/") != std::string::npos || low.find("\\mtp\\") != std::string::npos;
}

// К фильтру кванта дописать один подходящий MTP и одну проекцию.
// fill_recommended: короткое имя без файла — квант берёт каталог.
std::string with_companions(const std::string& repo, std::string filter, bool fill_recommended) {
  if (!filter.empty() && module_filter(filter)) return filter;
  if (filter.empty() && !fill_recommended) return filter;
  std::string e2d;
  std::string dj = lh::catalog::detail_json(repo, e2d);
  if (dj.empty()) return filter;
  nlohmann::json d;
  try {
    d = nlohmann::json::parse(dj);
  } catch (...) {
    return filter;
  }
  if (filter.empty()) filter = d.value("recommended", "");
  if (filter.empty() || module_filter(filter)) return filter;

  std::string model_quant;
  std::vector<lh::catalog::est::CompanionFile> sides;
  for (auto& f : d.value("files", nlohmann::json::array())) {
    std::string kind = f.value("kind", "quant");
    std::string path = f.value("path", "");
    std::string ff = f.value("filter", path);
    auto covered = [&](const std::string& s) {
      return !s.empty() && (s == filter || lh::wildcard_match(filter.c_str(), s.c_str()));
    };
    if (kind == "quant") {
      if (model_quant.empty() && (covered(ff) || covered(path)))
        model_quant = lh::catalog::est::quant_from_name(path);
      continue;
    }
    if (kind != "mtp" && kind != "mmproj") continue;
    if (ff.empty() || covered(ff) || covered(path)) continue;
    sides.push_back({kind, ff, lh::catalog::est::quant_from_name(path),
                     f.value("size", (uint64_t)0)});
  }
  if (model_quant.empty()) model_quant = lh::catalog::est::quant_from_name(filter);
  auto picks = lh::catalog::est::pick_companion_filters(model_quant, sides);
  std::string extra;
  for (auto& p : picks) extra += (extra.empty() ? "" : "|") + p;
  if (!extra.empty()) {
    logf("pull: к кванту добавляю %s", extra.c_str());
    filter += "|" + extra;
  }
  return filter;
}

enum class Kind { Llm, Image };

// Файл-кандидат. У llama репо одно на pull и dest = path; у image — у каждого
// слота свой репо и назначение (категория/имя в image-каталоге).
struct Cand {
  std::string path;     // путь файла в репо
  uint64_t size = 0;
  std::string sha;      // SHA256 из LFS-метаданных (пусто — мелочь без хеша)
  std::string repo;     // репо файла
  std::string dest;     // путь назначения относительно базового каталога
};

// lfs.oid из элемента дерева: голый 64-hex (принимаем и «sha256:»-префикс —
// в других версиях API бывает). Пусто — хеша нет.
std::string oid_of(const nlohmann::json& f) {
  if (!f.contains("lfs") || !f["lfs"].is_object()) return "";
  std::string oid = f["lfs"].value("oid", "");
  if (oid.rfind("sha256:", 0) == 0) oid = oid.substr(7);
  return oid.size() == 64 ? oid : std::string();
}

// Репо открыт (не gated)? gated бывает bool или строкой ("manual"/"auto"/
// "false") — учитываем оба вида. err — готовый текст для статуса.
bool repo_open(const std::string& repo, std::string& err) {
  std::string out, e2;
  if (!lh::dl::get_string("https://huggingface.co/api/models/" + repo, out, e2)) {
    err = "репозиторий не читается: " + e2;
    return false;
  }
  nlohmann::json info;
  try {
    info = nlohmann::json::parse(out);
  } catch (...) {
    err = "ответ HuggingFace не разобрался";
    return false;
  }
  nlohmann::json gated = info.contains("gated") ? info["gated"] : nlohmann::json(false);
  bool locked = gated.is_boolean() ? gated.get<bool>()
                                   : (gated.is_string() && gated.get<std::string>() != "false");
  if (locked) {
    err = "репозиторий закрыт (gated) — нужен токен, скачивание недоступно";
    return false;
  }
  return true;
}

// Дерево файлов репо: path/size. blobs=true — дерево отдаёт и lfs.oid: второй
// источник SHA256, если paths-info не читается (падало на сетевом сбое — и
// pull молча качал неподтверждённые байты).
bool fetch_tree(const std::string& repo, nlohmann::json& tree, std::string& err) {
  std::string raw, e2;
  if (!lh::dl::get_string("https://huggingface.co/api/models/" + repo +
                              "/tree/main?recursive=true&blobs=true",
                          raw, e2)) {
    err = "дерево файлов не читается: " + e2;
    return false;
  }
  try {
    tree = nlohmann::json::parse(raw);
  } catch (...) {
    err = "дерево файлов " + repo + " не разобралось";
    return false;
  }
  return true;
}

// SHA256 и точные размеры из LFS-метаданных; paths-info принимает
// {"paths": [...]}, не массив. Обрабатывает кандидатов своего репо — у
// image-ветки репо у слотов разные. Не читается — остаются oid'ы из дерева.
void enrich_sha(const std::string& repo, std::vector<Cand>& cands) {
  std::vector<Cand*> group;
  for (auto& c : cands)
    if (c.repo == repo) group.push_back(&c);
  if (group.empty()) return;
  nlohmann::json paths_req = {{"paths", nlohmann::json::array()}};
  for (auto& c : group) paths_req["paths"].push_back(c->path);
  std::string raw, e2;
  if (!lh::dl::post_json("https://huggingface.co/api/models/" + repo + "/paths-info/main",
                         paths_req.dump(), raw, e2)) {
    logf("pull: LFS-метаданные %s не читаются (%s) — полагаюсь на дерево", repo.c_str(),
         e2.c_str());
    return;
  }
  try {
    auto arr = nlohmann::json::parse(raw);
    for (auto& c : group)
      for (auto& p : arr)
        if (p.value("path", "") == c->path) {
          c->size = p.value("size", c->size);
          if (p.contains("lfs") && p["lfs"].is_object()) {
            c->size = p["lfs"].value("size", c->size);
            std::string oid = p["lfs"].value("oid", "");
            if (oid.rfind("sha256:", 0) == 0) oid = oid.substr(7);
            if (oid.size() == 64) c->sha = oid;
          }
        }
  } catch (...) { /* остаются oid'ы из дерева (blobs) — см. проверку ниже */ }
}

// Целостность — fail-closed: крупный файл (порог LFS у HF — 10 МиБ,
// с запасом 16) без SHA256 — это сбой обоих источников, а «качаем без
// проверки» превращал подмену тела ответа (CDN/зеркало) в тихий канал
// битого GGUF парсеру и детям. Мелочь, для которой API хеша не даёт,
// качаем, но с громкой записью в журнал и флагом unverified в статусе.
// st — текущее состояние (repo/kind): ошибка пишется от него.
bool sha_failclosed(std::vector<Cand>& cands, State& st) {
  for (auto& c : cands) {
    if (c.sha.size() != 64) c.sha.clear(); // мусор (не hex) — что нет хеша
    if (!c.sha.empty()) continue;
    if (c.size >= (16ull << 20)) {
      State e = st;
      e.state = "error";
      e.file = c.path;
      e.error = "не удалось получить SHA256 для «" + c.path +
                "» — крупный файл без проверки целостности не качаем";
      set(e);
      return false;
    }
    st.unverified = true;
    logf("pull: «%s» (%s) — SHA256 недоступен, качаем БЕЗ проверки целостности",
         c.path.c_str(), human(c.size).c_str());
  }
  return true;
}

// Скачивание одного кандидата в base/dest: уже скачанное и целое — пропустить
// (размер перепроверяется хешем, пока это дёшево — до 2 ГиБ; хешить стом-ГиБ
// стек на каждый pull накладно, там остаётся сверка размера; расхождение —
// перекачиваем), иначе докачка .part curl'ом с прогрессом в общий стейт.
bool download_one(const std::string& base, const Cand& c, uint64_t done_before,
                  uint64_t total, std::string& err) {
  const std::string dest = base + c.dest;
  lh::ensure_dir(dest.substr(0, dest.find_last_of('/')));
  if (lh::file_exists(dest)) {
    uint64_t sz = lh::file_size(dest);
    if (sz == c.size) { // уже скачан
      bool intact = c.sha.empty() || sz > (2ull << 30) || lh::dl::sha256_file(dest) == c.sha;
      if (intact) return true;
      logf("pull: «%s» на диске не совпал с SHA256 — перекачиваем", c.dest.c_str());
    }
  }
  { std::lock_guard l(g_mtx); g_state.file = c.dest; }
  logf("pull: %s (%s)", c.dest.c_str(), human(c.size).c_str());
  // Путь в URL — с процент-кодированием: пробелы и не-ASCII в именах
  // файлов HF иначе ломают строку HTTP-запроса.
  std::string url = "https://huggingface.co/" + c.repo + "/resolve/main/" +
                    lh::percent_encode_path(c.path);
  return lh::dl::download(url, dest, c.sha,
                          [&](const lh::dl::Progress& p) {
                            std::lock_guard l(g_mtx);
                            g_state.done = done_before + p.done;
                            g_state.total = total;
                            g_state.speed = p.speed;
                            return !g_cancel.load(std::memory_order_relaxed);
                          },
                          err);
}

// ---- image-ветка: художники ----------------------------------------------------
//
// Источники два. Встроенный каталог (image_catalog): слоты из разных репо
// (диффузия и энкодеры у квантизаторов, VAE в comfy-репо) + встроенный рецепт.
// Конвенция «org/repo»: в дереве репо recipes/*.json (ровно один) и по одному
// файлу в diffusion_models/ (.gguf), text_encoders/ (.gguf), vae/. Всё
// проверяется до первого байта — полуфабрикат в каталог не попадает; по
// завершении привязка дописывается в llmhost-image.json, и модель сразу видна
// в /v0/image и селекторе прокси (скан ходит в диск на каждый статус).

// Категория sd-server по слоту рецепта: имена полей привязки совпадают со
// слотами, имена каталогов — категориями sd-server.
const char* category_of_slot(const std::string& slot) {
  if (slot == "diffusion_model") return "diffusion_models";
  if (slot == "llm") return "text_encoders";
  if (slot == "vae") return "vae";
  return nullptr;
}

void image_worker(const std::string& repo_or_alias) {
  const std::string dir = lh::image::models_dir_resolved(config::current());
  auto fail = [&](const std::string& msg, const std::string& file = "",
                  uint64_t done = 0, uint64_t total = 0) {
    State s;
    s.state = "error";
    s.repo = repo_or_alias;
    s.file = file;
    s.done = done;
    s.total = total;
    s.error = msg;
    s.kind = "image";
    set(s);
  };

  std::vector<Cand> plan;
  std::string recipe_id, recipe_body; // тело пусто — рецепт едет из репо-конвенции

  if (const lh::image_catalog::Entry* a = lh::image_catalog::find(repo_or_alias)) {
    recipe_id = a->id;
    recipe_body = a->recipe;
    // Дерево каждого уникального репо слота; файл — точным именем (пути VAE —
    // с подпапками comfy-репо).
    std::map<std::string, nlohmann::json> trees;
    for (const auto& s : a->slots) {
      const char* cat = category_of_slot(s.slot);
      if (!cat) {
        fail(std::string("неизвестный слот «") + s.slot + "» во встроенном каталоге");
        return;
      }
      if (!trees.count(s.repo)) {
        std::string e2;
        if (!repo_open(s.repo, e2) || !fetch_tree(s.repo, trees[s.repo], e2)) {
          fail(e2);
          return;
        }
      }
      bool hit = false;
      for (const auto& f : trees[s.repo]) {
        if (!f.is_object() || f.value("type", "") != "file") continue;
        std::string path = f.value("path", "");
        if (path != s.file) continue;
        Cand c;
        c.repo = s.repo;
        c.path = path;
        c.dest = std::string(cat) + "/" + s.dest;
        c.size = f.value("size", (uint64_t)0);
        c.sha = oid_of(f);
        plan.push_back(std::move(c));
        hit = true;
        break;
      }
      if (!hit) {
        fail("в " + std::string(s.repo) + " нет файла «" + s.file +
             "» — встроенный каталог устарел, источник переименовали");
        return;
      }
    }
  } else {
    nlohmann::json tree;
    std::string e2;
    if (!repo_open(repo_or_alias, e2) || !fetch_tree(repo_or_alias, tree, e2)) {
      fail(e2);
      return;
    }
    std::vector<std::string> recipes;
    for (const auto& f : tree) {
      if (!f.is_object() || f.value("type", "") != "file") continue;
      std::string path = f.value("path", "");
      if (!lh::safe_rel_path(path)) {
        fail("в репозитории небезопасное имя файла: " + path);
        return;
      }
      size_t slash = path.find('/');
      std::string top = slash == std::string::npos ? "" : path.substr(0, slash);
      if (top == "recipes" && path.size() > 13 &&
          strcasecmp(path.c_str() + path.size() - 5, ".json") == 0) {
        recipes.push_back(path);
        continue;
      }
      if (!lh::image::is_catalog_dir_name(top)) continue; // README и прочее — мимо
      // Диффузия и энкодер — только GGUF (ребёнок ест только их), VAE — любая
      // раскладка (safetensors).
      if ((top == "diffusion_models" || top == "text_encoders") &&
          (path.size() <= 5 || strcasecmp(path.c_str() + path.size() - 5, ".gguf") != 0)) {
        fail(top + ": только .gguf, а не «" + path + "»");
        return;
      }
      Cand c;
      c.repo = repo_or_alias;
      c.path = path;
      c.dest = path; // категории уже в пути репо
      c.size = f.value("size", (uint64_t)0);
      c.sha = oid_of(f);
      plan.push_back(std::move(c));
    }
    if (recipes.size() != 1) {
      fail(recipes.empty()
               ? "в репозитории нет recipes/*.json — конвенция: рецепт sd-server + "
                 "diffusion_models/ + text_encoders/ + vae/; или имя из каталога "
                 "художников (lhctl image pull без имени)"
               : "в репозитории несколько рецептов — нужен ровно один recipes/*.json");
      return;
    }
    recipe_id = recipes[0].substr(8, recipes[0].size() - 13); // без «recipes/» и «.json»
    if (!lh::image::id_ok(recipe_id)) {
      fail("id рецепта «" + recipe_id + "» — только [A-Za-z0-9._-]: переименуйте "
           "recipes/<id>.json");
      return;
    }
    for (const char* cat : {"diffusion_models", "text_encoders", "vae"}) {
      int n = 0;
      std::string example;
      for (auto& c : plan)
        if (c.dest.rfind(std::string(cat) + "/", 0) == 0) {
          ++n;
          example = c.dest;
        }
      if (n != 1) {
        fail(std::string("в ") + cat + "/ должен быть ровно один файл, а их " +
             std::to_string(n) + (n ? " («" + example + "»…)" : ""));
        return;
      }
    }
  }

  // Сборка плана — несколько сетевых вызовов (деревья репо слотов):
  // отмена, пришедшая сюда, поглощается до медленной SHA-фазы.
  if (take_cancel()) { set_stopped(repo_or_alias, "", 0, 0, "image"); return; }

  // SHA256 по группам репо; крупное без хеша — отказ (тот же fail-closed).
  State st;
  st.state = "resolving";
  st.repo = repo_or_alias;
  st.kind = "image";
  {
    std::set<std::string> repos;
    for (auto& c : plan) repos.insert(c.repo);
    for (const auto& r : repos) enrich_sha(r, plan);
    if (!sha_failclosed(plan, st)) return;
  }

  st.state = "downloading";
  uint64_t total = 0;
  for (auto& c : plan) total += c.size;
  st.total = total;
  set(st);
  if (take_cancel()) { set_stopped(repo_or_alias, "", 0, total, "image"); return; }

  uint64_t done_before = 0;
  for (auto& c : plan) {
    std::string de;
    if (!download_one(dir + "/", c, done_before, total, de)) {
      if (take_cancel()) { set_stopped(repo_or_alias, c.dest, done_before, total, "image"); return; }
      fail(de, c.dest, done_before, total);
      return;
    }
    done_before += c.size;
  }

  // Рецепт из встроенного каталога: отсутствующий — положить, существующий —
  // не трогать (правки пользователя ценнее дословного повтора).
  if (!recipe_body.empty()) {
    std::string rp = (fs::path(dir) / "recipes" / (recipe_id + ".json")).string();
    if (lh::read_small_file(rp).empty() && !lh::write_small_file(rp, recipe_body)) {
      fail("рецепт не записался: " + rp);
      return;
    }
  }

  // Привязка: модель сразу видна в /v0/image и селекторе прокси.
  lh::image::Binding b;
  b.id = recipe_id;
  b.recipe = recipe_id;
  for (const auto& c : plan) {
    if (c.dest.rfind("diffusion_models/", 0) == 0) b.diffusion_model = c.dest;
    else if (c.dest.rfind("text_encoders/", 0) == 0) b.llm = c.dest;
    else if (c.dest.rfind("vae/", 0) == 0) b.vae = c.dest;
  }
  std::string berr;
  if (!lh::image::binding_upsert(dir, b, berr)) {
    fail("файлы скачаны, но привязка не записалась: " + berr);
    return;
  }

  logf("pull: image-модель «%s» готова (%s)", recipe_id.c_str(), dir.c_str());
  g_cancel = false; // отмена, пришедшая на последнем файле, не должна висеть
  {
    State fin;
    fin.state = "done";
    fin.repo = repo_or_alias;
    fin.done = fin.total = total;
    fin.unverified = st.unverified;
    fin.kind = "image";
    set(std::move(fin));
  }
}

} // namespace

bool busy() {
  std::lock_guard l(g_mtx);
  return g_state.state == "resolving" || g_state.state == "downloading";
}

bool cancel(std::string& err) {
  std::lock_guard l(g_mtx);
  if (g_state.state != "resolving" && g_state.state != "downloading") {
    err = "скачивание не идёт";
    return false;
  }
  g_cancel = true;
  return true;
}

std::string status_json() {
  std::lock_guard l(g_mtx);
  nlohmann::json j = {{"state", g_state.state},
                      {"kind", g_state.kind},
                      {"repo", g_state.repo},
                      {"file", g_state.file},
                      {"done", g_state.done},
                      {"total", g_state.total},
                      {"speed", g_state.speed}};
  if (!g_state.error.empty()) j["error"] = g_state.error;
  if (g_state.unverified) j["unverified"] = true;
  if (g_cancel.load(std::memory_order_relaxed)) j["stopping"] = true;
  return j.dump(2);
}

bool start_kind(Kind kind, const std::string& repo_in, const std::string& file_filter,
                std::string& err) {
  {
    std::lock_guard l(g_mtx);
    // Проверка и бронирование — под одним замком: два одновременных POST
    // /v0/pull иначе оба проходили проверку и качали в один и тот же .part
    // (государство «resolving» раньше ставилось только в рабочем потоке).
    if (g_state.state == "resolving" || g_state.state == "downloading") {
      err = "скачивание уже идёт";
      return false;
    }
    // Инвариант жизни флага: к моменту, когда состояние покинуло
    // resolving/downloading, рабочий поток его уже поглотил (checkpoint/
    // финал) или сбросил; сброс здесь — страховка от ранних выходов
    // (fail_idle и сбор image-плана), чтобы новый pull не унаследовал
    // чужую отмену и не умер мгновенным «stopped».
    g_cancel = false;
    g_state.state = "resolving";
    g_state.repo = repo_in;
    g_state.kind = kind == Kind::Image ? "image" : "llm";
    g_state.error.clear();
  }
  auto fail_idle = [&](const std::string& msg) {
    g_cancel = false; // отмена из окна «забронировано — не стартовало»
    std::lock_guard l(g_mtx);
    std::string k = g_state.kind;
    g_state = State{};
    g_state.kind = k;
    err = msg;
    return false;
  };
  std::string repo = repo_in;
  bool aliased = false;
  if (kind == Kind::Image) {
    // Имя каталога художников или «org/repo» с раскладкой-конвенцией. Что это —
    // решает дерево репо в рабочем потоке; здесь только честный отказ для
    // строки, которая не может быть ни тем, ни другим.
    if (!lh::image_catalog::find(repo_in)) {
      std::string norm_err;
      if (!lh::normalize_hf_repo(repo, norm_err)) { // нормализует repo на месте
        err = "«" + repo_in + "» нет в каталоге художников, и это не «org/repo» — " +
              norm_err + " (каталог: lhctl image pull без имени)";
        return fail_idle(err);
      }
    }
  } else {
    std::string norm_err;
    if (!lh::normalize_hf_repo(repo, norm_err)) {
      // Не «org/repo» — может, короткое имя (qwen3, gemma-4…): репозиторий
      // известен, квант под карту выберет каталог.
      if (const lh::alias::Entry* a = lh::alias::find(repo_in.c_str())) {
        repo = a->repo;
        aliased = true;
        if (!lh::normalize_hf_repo(repo, err)) return fail_idle(err); // битая строка таблицы
      } else {
        err = norm_err;
        if (repo_in.find('/') == std::string::npos)
          err += " — или короткое имя (список: lhctl pull без аргументов)";
        return fail_idle(err);
      }
    }
  }

  std::thread([kind, repo, repo_in, file_filter, aliased] {
    if (kind == Kind::Image) {
      image_worker(repo_in);
      return;
    }
    std::string filter = file_filter;
    if (filter.empty() && aliased) {
      // Короткое имя без файла: квант под карту и подходящие MTP с проекцией.
      filter = with_companions(repo, "", true);
      if (!filter.empty())
        logf("pull: «%s» → %s, выбор каталога: «%s»", repo_in.c_str(), repo.c_str(),
             filter.c_str());
    } else if (!filter.empty()) {
      // Явный квант из каталога или вторым аргументом — те же спутники.
      // Сам модуль (строка MTP или mmproj) качается как есть.
      filter = with_companions(repo, filter, false);
    }
    State st;
    st.state = "resolving";
    st.repo = repo;
    set(st);
    logf("pull: %s", repo.c_str());

    std::string e2;
    if (!repo_open(repo, e2)) {
      State s = st;
      s.state = "error";
      s.error = e2;
      set(s);
      return;
    }

    nlohmann::json tree;
    if (!fetch_tree(repo, tree, e2)) {
      State s = st;
      s.state = "error";
      s.error = e2;
      set(s);
      return;
    }
    if (take_cancel()) { set_stopped(repo, "", 0, 0, "llm"); return; }
    std::vector<Cand> cands;
    for (auto& f : tree) {
      if (!f.is_object() || f.value("type", "") != "file") continue;
      std::string path = f.value("path", "");
      if (path.size() < 5 || strcasecmp(path.c_str() + path.size() - 5, ".gguf") != 0) continue;
      if (!lh::safe_rel_path(path)) {
        // Имя из сети клеится к папке моделей: «..», «:», кавычки и прочее
        // могут вывести запись за её пределы — такой репозиторий не качаем.
        State s = st;
        s.state = "error";
        s.error = "в репозитории небезопасное имя файла: " + path;
        set(s);
        return;
      }
      if (!filter.empty() && !filter_match(filter, path)) continue;
      Cand c;
      c.repo = repo;
      c.path = path;
      c.dest = path;
      c.size = f.value("size", (uint64_t)0);
      c.sha = oid_of(f);
      cands.push_back(std::move(c));
    }
    if (cands.empty()) {
      State s = st;
      s.state = "error";
      s.error = filter.empty() ? "в репозитории нет .gguf файлов"
                               : "под фильтр «" + filter + "» файлов нет";
      set(s);
      return;
    }
    if (cands.size() > 1 && filter.empty()) {
      std::string list;
      for (size_t i = 0; i < cands.size() && i < 8; ++i)
        list += (list.empty() ? "" : ", ") + cands[i].path + " — " + human(cands[i].size);
      State s = st;
      s.state = "error";
      s.error = "в репозитории несколько .gguf — уточните «file»: " + list;
      set(s);
      return;
    }

    enrich_sha(repo, cands);
    if (!sha_failclosed(cands, st)) return;

    // Куда: <models_dir>/<org>/<repo>/<путь из репо>.
    config::Config cfg = config::current();
    std::string dir = config::models_dir_resolved(cfg);
    std::string repo_dir = dir + "/" + repo;
    lh::ensure_dir(repo_dir);

    st.state = "downloading";
    uint64_t total = 0;
    for (auto& c : cands) total += c.size;
    st.total = total;
    set(st);
    if (take_cancel()) { set_stopped(repo, "", 0, total, "llm"); return; }

    uint64_t done_before = 0;
    for (auto& c : cands) {
      std::string de;
      if (!download_one(repo_dir + "/", c, done_before, total, de)) {
        if (take_cancel()) { set_stopped(repo, c.dest, done_before, total, "llm"); return; }
        State s = st;
        s.state = "error";
        s.file = c.path;
        s.done = done_before;
        s.total = total;
        s.error = de;
        set(s);
        return;
      }
      done_before += c.size;
    }
    logf("pull: %s готово", repo.c_str());
    g_cancel = false; // отмена, пришедшая на последнем файле, не должна висеть
    lh::sup::registry_changed(); // пресет роутера: новая модель — не ждать вотчер
    {
      State fin;
      fin.state = "done";
      fin.repo = repo;
      fin.done = fin.total = total;
      fin.unverified = st.unverified; // флаг «качалось без хеша» доживает до статуса
      set(std::move(fin));
    }

    // Обратная подсказка: скачали маленькую модель той же семьи, что уже
    // стоит большой? Она — кандидат в draft (ключ «model-draft»).
    std::vector<std::string> done_files;
    for (auto& c : cands) done_files.push_back(repo_dir + "/" + c.path);
    auto locals = lh::modelreg::scan(dir);
    for (auto& f : done_files) {
      lh::gguf::Meta meta;
      std::string perr;
      uint64_t sz = lh::file_size(f);
      if (!lh::gguf::parse_header(f, meta, perr) || !meta.n_layers) continue;
      for (auto& big : locals) {
        if (big.arch != meta.arch || !big.meta.n_layers || big.size < sz * 4) continue;
        size_t s = f.find_last_of('/');
        std::string small = s == std::string::npos ? f : f.substr(s + 1);
        if (small.size() > 5) small.erase(small.size() - 5);
        logf("pull: «%s» подходит как draft для «%s»: добавьте в model_overrides.json "
             "ключ \"model-draft\": \"%s\".",
             small.c_str(), big.name.c_str(), small.c_str());
      }
    }
  }).detach();
  return true;
}

bool start(const std::string& repo_in, const std::string& file_filter, std::string& err) {
  return start_kind(Kind::Llm, repo_in, file_filter, err);
}

bool start_image(const std::string& repo_or_alias, std::string& err) {
  return start_kind(Kind::Image, repo_or_alias, "", err);
}

// ---- недокачанные загрузки ----------------------------------------------------

std::vector<PartDownload> scan_parts(const std::string& models_dir) {
  std::vector<PartDownload> out;
  std::vector<std::string> dirs{models_dir};
  for (int depth = 0; depth < 3 && !dirs.empty(); ++depth) {
    std::vector<std::string> next;
    for (auto& d : dirs) {
      std::error_code ec;
      fs::directory_iterator it(d, ec), end;
      if (ec) continue;
      for (; it != end; it.increment(ec)) {
        if (ec) break;
        const fs::directory_entry& e = *it;
        std::string full = e.path().string();
        std::string name = e.path().filename().string();
        std::error_code ec2;
        if (e.is_directory(ec2)) {
          // Каталоги image-моделей — другая раскладка: .part там докачивается
          // повторным image-pull, а «repo» из двух компонентов бессмыслен.
          if (depth > 0 || !lh::image::is_catalog_dir_name(name)) next.push_back(full);
          continue;
        }
        size_t n = name.size();
        if (n <= 5 || strcasecmp(name.c_str() + n - 5, ".part") != 0) continue;
        PartDownload p;
        p.path = full;
        p.size = lh::file_size(full);
        name.erase(name.size() - 5); // без ".part"
        p.file = name;
        // repo для докачки: первые два уровня каталогов относительно models_dir.
        std::string rel = full.substr(models_dir.size() + 1);
        std::vector<std::string> comps;
        size_t pos = 0;
        while ((pos = rel.find('/')) != std::string::npos) {
          comps.push_back(rel.substr(0, pos));
          rel.erase(0, pos + 1);
        }
        if (comps.size() >= 2) p.repo = comps[0] + "/" + comps[1];
        out.push_back(std::move(p));
        if (out.size() >= 20) return out;
      }
    }
    dirs = std::move(next);
  }
  return out;
}

} // namespace lh::pull
