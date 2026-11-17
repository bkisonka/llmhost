#include "image.h"
#include "util.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <map>
#include <mutex>
#include <set>
#include <system_error>

#include <nlohmann/json.hpp>

namespace lh::image {

namespace fs = std::filesystem;

namespace {

constexpr const char* kBindingFile = "llmhost-image.json";

const char* kCatalogDirs[] = {"diffusion_models", "text_encoders", "vae", "recipes"};

// Битые записи (нет рецепта/файла, битый json) жалуемся один раз на состояние
// файла — вотчер сканирует каталог каждые 2 с, спамить нельзя.
std::mutex g_complained_mtx;
std::set<std::string> g_complained;

void complain_once(const std::string& key, const std::string& msg) {
  std::lock_guard l(g_complained_mtx);
  if (g_complained.insert(key).second) logf("image catalog: %s", msg.c_str());
}

std::string lower(std::string s) {
  for (char& c : s) c = (char)tolower((unsigned char)c);
  return s;
}

std::string error_of(const nlohmann::json::exception& e) { return e.what(); }

// Писатель привязок — один на процесс: read-modify-write под замком, иначе
// pull и удаление затирали бы записи друг друга (ср. conf_writers_mtx).
std::mutex g_bind_mtx;

std::string binding_path(const std::string& dir) {
  return (fs::path(dir) / kBindingFile).string();
}

// Привязки как объект; нет файла — пустой каталог. Битый файл — err:
// вслепую дописывать в него запись значит хоронить остальные.
bool read_bindings(const std::string& dir, nlohmann::json& j, std::string& err) {
  std::string raw = read_small_file(binding_path(dir));
  if (raw.empty()) {
    j = nlohmann::json{{"models", nlohmann::json::array()}};
    return true;
  }
  try {
    j = nlohmann::json::parse(raw);
  } catch (const nlohmann::json::exception& e) {
    err = std::string("битый ") + kBindingFile + ": " + e.what();
    return false;
  }
  if (!j.is_object() || !j.contains("models") || !j["models"].is_array()) {
    err = std::string(kBindingFile) + ": ожидается {\"models\": [...]}";
    return false;
  }
  return true;
}

bool slot_rel_ok(const std::string& rel, const char* what, std::string& err) {
  if (!rel.empty() && lh::safe_rel_path(rel) && lh::safe_local_filename(rel))
    return true;
  err = std::string(what) + ": небезопасный путь «" + rel + "»";
  return false;
}

// Атомарная запись (ср. save_to в config.cpp, но tmp с случайным хвостом:
// «конф.tmp» предсказуем и гонит писателей в один файл).
bool save_bindings(const std::string& dir, const nlohmann::json& j, std::string& err) {
  const std::string path = binding_path(dir);
  const std::string tmp = path + ".tmp." + lh::random_hex(6);
  std::error_code ec;
  bool ok = lh::write_small_file(tmp, j.dump(2)) && (fs::rename(tmp, path, ec), !ec);
  if (!ok) {
    err = "запись " + path + " не удалась" + (ec ? ": " + ec.message() : "");
    fs::remove(tmp, ec);
    return false;
  }
  return true;
}

} // namespace

bool id_ok(const std::string& s) {
  if (s.empty() || s.size() > 128) return false;
  for (unsigned char c : s)
    if (!isalnum(c) && c != '-' && c != '_' && c != '.') return false;
  return true;
}

std::string models_dir_resolved(const config::Config& cfg) {
  return cfg.image_models_dir.empty() ? config::models_dir_resolved(cfg)
                                      : config::resolve_path(cfg.image_models_dir);
}

bool binding_upsert(const std::string& dir, const Binding& b, std::string& err) {
  std::lock_guard l(g_bind_mtx);
  if (!id_ok(b.id)) {
    err = "плохой id модели «" + b.id + "» — [A-Za-z0-9._-]";
    return false;
  }
  if (b.recipe.empty() || !id_ok(b.recipe)) {
    err = "«" + b.id + "» — обязателен корректный recipe";
    return false;
  }
  if (!slot_rel_ok(b.diffusion_model, "diffusion_model", err) ||
      !slot_rel_ok(b.llm, "llm", err) || !slot_rel_ok(b.vae, "vae", err))
    return false;
  nlohmann::json j;
  if (!read_bindings(dir, j, err)) return false;
  // Существующая запись с тем же id — обновить пять наших ключей, сохранив
  // чужие (скан их игнорирует; затирать пользовательские пометки нельзя).
  nlohmann::json* entry = nullptr;
  for (auto& e : j["models"])
    if (e.is_object() && e.value("id", "") == b.id) entry = &e;
  nlohmann::json fields = {{"recipe", b.recipe},
                           {"diffusion_model", b.diffusion_model},
                           {"llm", b.llm},
                           {"vae", b.vae}};
  if (entry) entry->update(fields);
  else {
    nlohmann::json ne{{"id", b.id}};
    ne.update(fields);
    j["models"].push_back(std::move(ne));
  }
  return save_bindings(dir, j, err);
}

bool binding_remove(const std::string& dir, const std::string& id, Binding* removed,
                    std::string& err) {
  std::lock_guard l(g_bind_mtx);
  nlohmann::json j;
  if (!read_bindings(dir, j, err)) return false;
  nlohmann::json found;
  for (size_t i = 0; i < j["models"].size(); ++i) {
    nlohmann::json& e = j["models"][i];
    if (e.is_object() && e.value("id", "") == id) {
      found = std::move(e);
      j["models"].erase(i);
      break;
    }
  }
  if (found.is_null()) {
    err = "в каталоге image-моделей нет «" + id + "»";
    return false;
  }
  if (removed) {
    removed->id = id;
    removed->recipe = found.value("recipe", "");
    removed->diffusion_model = found.value("diffusion_model", "");
    removed->llm = found.value("llm", "");
    removed->vae = found.value("vae", "");
  }
  return save_bindings(dir, j, err);
}

bool is_catalog_dir_name(const std::string& name) {
  const std::string n = lower(name);
  for (const char* d : kCatalogDirs)
    if (n == d) return true;
  return false;
}

bool catalog_exists(const std::string& dir) {
  std::error_code ec;
  return fs::is_regular_file(fs::path(dir) / kBindingFile, ec) && !ec;
}

namespace {

// Дефолты генерации из рецепта: recipes/<recipe>.json, поля name и defaults.
// false — рецепта нет/битый (запись пропускается).
bool read_recipe(const std::string& dir, const std::string& recipe_id, Model& m,
                 std::string& err) {
  std::string path = (fs::path(dir) / "recipes" / (recipe_id + ".json")).string();
  std::string raw = read_small_file(path);
  if (raw.empty()) {
    err = "рецепт не найден: " + path;
    return false;
  }
  try {
    auto j = nlohmann::json::parse(raw);
    if (!j.is_object()) {
      err = "рецепт «" + recipe_id + "»: ожидается объект";
      return false;
    }
    m.name = j.value("name", recipe_id);
    if (j.contains("defaults") && j["defaults"].is_object()) {
      const auto& d = j["defaults"];
      if (d.contains("steps") && d["steps"].is_number_integer())
        m.steps = d["steps"].get<int>();
      if (d.contains("cfg_scale") && d["cfg_scale"].is_number())
        m.cfg = d["cfg_scale"].get<double>();
      if (d.contains("sampling_method") && d["sampling_method"].is_string())
        m.sampler = d["sampling_method"].get<std::string>();
      if (d.contains("scheduler") && d["scheduler"].is_string())
        m.scheduler = d["scheduler"].get<std::string>();
    }
    // Фичи рецепта — массив строк; мусорное значение фичу не убивает (скан
    // пропускает элемент, а не запись целиком).
    if (j.contains("features") && j["features"].is_array()) {
      for (const auto& f : j["features"])
        if (f.is_string() && !f.get<std::string>().empty())
          m.features.push_back(f.get<std::string>());
    }
    return true;
  } catch (const nlohmann::json::exception& e) {
    err = "рецепт «" + recipe_id + "»: " + error_of(e);
    return false;
  }
}

// Слот: относительный путь, существующий файл, размер. false — отказ записи.
bool slot_file(const std::string& dir, const std::string& rel, uint64_t& size,
               std::string& abs, std::string& err, const char* what) {
  if (rel.empty() || !lh::safe_rel_path(rel) || !lh::safe_local_filename(rel)) {
    err = std::string(what) + ": небезопасный путь «" + rel + "»";
    return false;
  }
  fs::path p = fs::path(dir) / rel;
  std::error_code ec;
  if (!fs::is_regular_file(p, ec) || ec) {
    err = std::string(what) + ": файл не найден: " + rel;
    return false;
  }
  // Симлинк-побег: легальный rel может указывать на ссылку, ведущую за
  // пределы каталога — CLI-аргумент ребёнка уехал бы к чужому файлу.
  // Проверка границы точная: «/data/models-evil» не префикс «/data/models».
  fs::path canon = fs::weakly_canonical(p, ec);
  fs::path root = fs::weakly_canonical(fs::path(dir), ec);
  const std::string c = canon.string(), r = root.string();
  const bool inside = !ec && (c == r || (c.size() > r.size() &&
                                         c.compare(0, r.size(), r) == 0 &&
                                         c[r.size()] == '/'));
  if (!inside) {
    err = std::string(what) + ": путь уходит за пределы каталога: " + rel;
    return false;
  }
  size = fs::file_size(p, ec);
  if (ec) {
    err = std::string(what) + ": размер недоступен: " + rel;
    return false;
  }
  abs = canon.string();
  return true;
}

} // namespace

std::vector<Model> scan(const std::string& dir) {
  std::vector<Model> out;
  std::string bind_path = (fs::path(dir) / kBindingFile).string();
  std::string raw = read_small_file(bind_path);
  if (raw.empty()) return out; // каталог не настроен — не ошибка
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(raw);
  } catch (const nlohmann::json::exception& e) {
    complain_once(bind_path + ":parse", bind_path + ": " + error_of(e));
    return out;
  }
  if (!j.is_object() || !j.contains("models") || !j["models"].is_array()) {
    complain_once(bind_path + ":shape", bind_path + ": ожидается {\"models\": [...]}");
    return out;
  }
  for (const auto& e : j["models"]) {
    if (!e.is_object()) continue;
    std::string id, recipe, diffusion, llm, vae;
    try {
      // value() бросает type_error на значении не того типа («id»: 123) —
      // без ловли битая запись ломала бы весь скан, а с ней и /v0/image.
      id = e.value("id", "");
      recipe = e.value("recipe", "");
      diffusion = e.value("diffusion_model", "");
      llm = e.value("llm", "");
      vae = e.value("vae", "");
    } catch (const nlohmann::json::exception& ex) {
      complain_once(bind_path + ":entry", bind_path + ": запись модели: " + ex.what());
      continue;
    }
    std::string err;
    if (!id_ok(id)) {
      complain_once(bind_path + ":id:" + id, bind_path + ": плохой id «" + id + "»");
      continue;
    }
    if (recipe.empty() || !id_ok(recipe)) {
      complain_once(bind_path + ":recipe:" + id,
                    bind_path + ": «" + id + "» — обязателен корректный recipe");
      continue;
    }
    Model m;
    m.id = id;
    m.recipe = recipe;
    if (!read_recipe(dir, recipe, m, err)) {
      complain_once(bind_path + ":recipefile:" + id, bind_path + ": «" + id + "»: " + err);
      continue;
    }
    uint64_t sz = 0, bytes = 0;
    std::string d_abs, l_abs, v_abs;
    if (!slot_file(dir, diffusion, sz, d_abs, err, "diffusion_model")) {
      complain_once(bind_path + ":slots:" + id, bind_path + ": «" + id + "»: " + err);
      continue;
    }
    bytes += sz;
    if (!slot_file(dir, llm, sz, l_abs, err, "llm")) {
      complain_once(bind_path + ":slots:" + id, bind_path + ": «" + id + "»: " + err);
      continue;
    }
    bytes += sz;
    if (!slot_file(dir, vae, sz, v_abs, err, "vae")) {
      complain_once(bind_path + ":slots:" + id, bind_path + ": «" + id + "»: " + err);
      continue;
    }
    bytes += sz;
    // Диффузионная модель и энкодер — GGUF (ребёнок съест только их), VAE —
    // что отдаёт рецептурная раскладка (safetensors).
    if (d_abs.size() <= 5 || d_abs.compare(d_abs.size() - 5, 5, ".gguf") != 0 ||
        l_abs.size() <= 5 || l_abs.compare(l_abs.size() - 5, 5, ".gguf") != 0) {
      complain_once(bind_path + ":gguf:" + id,
                    bind_path + ": «" + id + "»: diffusion_model и llm — только .gguf");
      continue;
    }
    m.diffusion = std::move(d_abs);
    m.llm = std::move(l_abs);
    m.vae = std::move(v_abs);
    m.bytes = bytes;
    if (m.name.empty()) m.name = id;
    out.push_back(std::move(m));
  }
  // Дубликат id — двусмысленность для API/прокси: скрываем ВСЕ записи с ним.
  std::map<std::string, int> count;
  for (const auto& m : out) count[m.id]++;
  std::erase_if(out, [&](const Model& m) { return count[m.id] > 1; });
  for (const auto& [id, n] : count)
    if (n > 1)
      complain_once(bind_path + ":dup:" + id,
                    bind_path + ": дубликат id «" + id + "» — записи скрыты");
  return out;
}

} // namespace lh::image
