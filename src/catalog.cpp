#include "catalog.h"
#include "i18n.h"

#include "catalog_est.h"
#include "config.h"
#include "download.h"
#include "engine.h"
#include "fit.h"
#include "gguf.h"
#include "hw.h"
#include "modelreg.h"
#include "util.h"

#include <nlohmann/json.hpp>

#include <strings.h>
#include <algorithm>
#include <cctype>
#include <ctime>
#include <map>
#include <mutex>

using lh::i18n::tr;

namespace lh::catalog {

namespace {

using nlohmann::json;
using lh::modelreg::shard_base;
using lh::modelreg::shard_info;

std::string url_encode(const std::string& s) {
  static const char* hex = "0123456789ABCDEF";
  std::string out;
  for (unsigned char c : s) {
    if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
      out.push_back((char)c);
    } else {
      out.push_back('%');
      out.push_back(hex[c >> 4]);
      out.push_back(hex[c & 15]);
    }
  }
  return out;
}

// HF отвечает 401 и на приватные, и на несуществующие репозитории —
// сообщаем об этом по-человечески.
bool hf_get(const std::string& url, std::string& out, std::string& err) {
  if (!lh::dl::get_string(url, out, err)) return false;
  return true;
}

void hf_err(const std::string& raw, std::string& err) {
  if (raw.find("401") != std::string::npos)
    err = tr("репозиторий не найден, приватный или нужен токен HF_TOKEN (401)");
  else if (raw.find("429") != std::string::npos)
    err = "HuggingFace просит не спешить (429) — попробуйте позже";
  else
    err = raw;
}

// Кэш поиска: HF-ответ пересобирается с свежей VRAM, сам ответ — 5 минут.
struct SearchCache {
  time_t fetched = 0;
  json raw = json::array();
};
std::mutex g_search_mtx;
std::map<std::string, SearchCache> g_search_cache;

// Кэши без мусора: при вставке выкидываем просроченные, а сверх лимита —
// самые старые. Без этого поток уникальных запросов поиска раздувал память
// бесконечно (записи никогда не удалялись).
template <typename Map>
void cache_sweep(Map& m, time_t ttl, size_t cap) {
  time_t now = time(nullptr);
  for (auto it = m.begin(); it != m.end();)
    if (now - it->second.fetched >= ttl) it = m.erase(it);
    else ++it;
  while (m.size() > cap) {
    auto oldest = m.begin();
    for (auto it = m.begin(); it != m.end(); ++it)
      if (it->second.fetched < oldest->second.fetched) oldest = it;
    m.erase(oldest);
  }
}

// Кэш деталей: сеть (дерево файлов, заголовок GGUF) дорогая — 10 минут.
struct DetailCache {
  time_t fetched = 0;
  bool gated = false;
  gguf::Meta meta;
  bool precise = false;          // заголовок дочитан из сети
  double params = 0;             // из имени — фолбэк для KV
  struct File {
    std::string path;
    uint64_t size = 0;
  };
  std::vector<File> files;       // все .gguf репозитория
};
std::mutex g_detail_mtx;
std::map<std::string, DetailCache> g_detail_cache;


DetailCache fetch_detail(const std::string& repo, std::string& err) {
  DetailCache d;
  const std::string api = "https://huggingface.co/api/models/" + repo;

  std::string out;
  if (!hf_get(api, out, err)) {
    hf_err(err, err);
    return d;
  }
  json info;
  try {
    info = json::parse(out);
  } catch (...) {
    err = tr("ответ HuggingFace не разобрался");
    return d;
  }
  // gated бывает bool или строкой ("manual"/"auto") — см. грабли P3.
  json gated = info.contains("gated") ? info["gated"] : json(false);
  d.gated = gated.is_boolean() ? gated.get<bool>()
                               : (gated.is_string() && gated.get<std::string>() != "false");

  std::string tree_raw;
  if (!hf_get(api + "/tree/main?recursive=true", tree_raw, err)) {
    hf_err(err, err);
    return d;
  }
  json tree;
  try {
    tree = json::parse(tree_raw);
  } catch (...) {
    err = tr("дерево файлов не разобралось");
    return d;
  }
  for (auto& f : tree) {
    if (!f.is_object() || f.value("type", "") != "file") continue;
    std::string path = f.value("path", "");
    if (path.size() < 5 || strcasecmp(path.c_str() + path.size() - 5, ".gguf") != 0) continue;
    d.files.push_back({path, f.value("size", (uint64_t)0)});
  }
  if (d.files.empty()) {
    err = tr("в репозитории нет .gguf файлов");
    return d;
  }

  // Точные размеры — из LFS-метаданных (грабли P3: тело paths-info — {"paths": [...]}).
  json paths_req = {{"paths", json::array()}};
  for (auto& f : d.files) paths_req["paths"].push_back(f.path);
  std::string sha_raw;
  if (lh::dl::post_json(api + "/paths-info/main", paths_req.dump(), sha_raw,
                        err)) {
    try {
      auto arr = json::parse(sha_raw);
      for (auto& f : d.files)
        for (auto& p : arr)
          if (p.value("path", "") == f.path && p.contains("lfs") && p["lfs"].is_object())
            f.size = p["lfs"].value("size", f.size);
    } catch (...) { /* останутся размеры из дерева */ }
  } else {
    logf("catalog: paths-info не читается — размеры из дерева");
  }

  // Заголовок GGUF — из сети, первые 512 КБ предпочитаемого файла. У шардов
  // полные метаданные лежат в первом («00001-of-…»).
  const char* preferred[] = {"Q4_K_M", "Q5_K_M", "Q4_K_S", "Q4_0", "Q6_K", "Q8_0", "F16"};
  std::string header_path;
  for (const char* q : preferred) {
    bool found = false;
    for (auto& f : d.files) {
      int idx, count;
      shard_info(f.path, idx, count);
      if (count > 1 && idx != 1) continue;
      if (est::quant_from_name(f.path) == q) {
        header_path = f.path;
        found = true;
        break;
      }
    }
    if (found) break;
  }
  if (header_path.empty()) {
    // Единственный файл — или наибольший первый шард.
    auto biggest = std::max_element(d.files.begin(), d.files.end(),
                                    [](auto& a, auto& b) { return a.size < b.size; });
    header_path = biggest->path;
    int idx, count;
    shard_info(header_path, idx, count);
    if (count > 1 && idx != 1) {
      std::string base = shard_base(header_path);
      for (auto& f : d.files) {
        int i2, c2;
        shard_info(f.path, i2, c2);
        if (i2 == 1 && shard_base(f.path) == base) { header_path = f.path; break; }
      }
    }
  }
  std::string blob;
  std::string url = "https://huggingface.co/" + repo + "/resolve/main/" +
                    lh::percent_encode_path(header_path);
  if (lh::dl::get_partial(url, 512 * 1024, blob, err) &&
      lh::gguf::parse_header_mem(blob.data(), blob.size(), d.meta, err)) {
    d.precise = true;
    logf("catalog: %s — заголовок %s (%s), %s", repo.c_str(), header_path.c_str(),
         d.meta.arch.c_str(), d.meta.name.c_str());
  } else {
    d.params = est::params_from_name(repo);
    logf("catalog: %s — заголовок не дочитан (%s), оценка по имени", repo.c_str(), err.c_str());
  }
  err.clear();
  return d;
}

// Светофор по группе файлов: точный по заголовку или грубый по размерам/имени.
// Учитывает умную настройку: KV q8_0 (вдвое меньше кэша) и MoE-экспертов на CPU.
// cuda — какой бэкенд реально поедет (engine::launches_cuda): от него зависит
// и оверхед, и плотный частичный сплит.
json fit_json(const DetailCache& d, uint64_t group_size, uint32_t ctx, const hw::Gpu& gpu,
              uint32_t& max_ctx_out, bool cuda) {
  config::Config cfg = config::current();
  bool kv_q8 = cfg.auto_params;
  uint32_t cap = d.meta.n_ctx_train ? (uint32_t)d.meta.n_ctx_train : 8192u;
  uint64_t vram_needed = 0;
  uint64_t overhead = ((cuda ? 400ull : 90ull) << 20) + (uint64_t)ctx * 65536;
  bool moe = d.meta.n_experts > 0;
  bool moe_cpu = false;
  bool dense_partial = false;
  int d_ngl = 0, d_layers = 0;
  if (d.precise) {
    // Если модель тесна, но RAM тянет экспертов — веса на GPU считаем как файл
    // минус эксперты (точные байты из тензорного списка, без него — 85%).
    uint64_t weights = group_size;
    if (moe && gpu.vram_free > 0 &&
        !fit::estimate(d.meta, group_size, ctx, gpu, kv_q8).fits &&
        fit::cpu_offload_ok(d.meta, group_size, gpu)) {
      weights = group_size - fit::expert_bytes_of(d.meta, group_size);
      moe_cpu = true;
    }
    auto e = fit::estimate(d.meta, weights, ctx, gpu, kv_q8);
    vram_needed = e.total;
    max_ctx_out = fit::max_ctx(d.meta, weights, gpu, cap, kv_q8);
    // Плотная и не влезает: на CUDA-движке llama.cpp режет слои по -ngl так же,
    // как у MoE, — вердикт «частично, часть слоёв в RAM» вместо «мимо».
    if (!moe && d.meta.n_layers > 0 && gpu.vram_free > 0 && vram_needed > gpu.vram_free) {
      hw::Gpu eng = gpu;
      eng.build = hw::Build::Cuda13; // оверхед — под запускаемый движок
      auto s = fit::dense_split(d.meta, group_size, eng, cap, /*kv_q8=*/true,
                                cfg.ctx >= 512 ? (uint32_t)cfg.ctx : 4096u);
      if (s.ok && fit::ram_offload_ok(s.cpu_bytes, gpu)) {
        auto e2 = fit::estimate(d.meta, s.gpu_bytes, s.ctx, eng, /*kv_q8=*/true);
        vram_needed = e2.total;
        max_ctx_out = s.ctx;
        dense_partial = true;
        d_ngl = s.ngl;
        d_layers = (int)d.meta.n_layers;
      }
    }
  } else {
    uint64_t kv_bytes_per_tok = est::kv_per_token(d.params) / (kv_q8 ? 2 : 1);
    vram_needed = group_size + kv_bytes_per_tok * ctx + overhead;
    // Лестница вниз по контексту — как в fit::max_ctx, но с грубым KV.
    max_ctx_out = 0;
    for (uint32_t c = cap;; c >>= 1) {
      if (gpu.vram_free > 0 &&
          group_size + kv_bytes_per_tok * c + ((cuda ? 400ull : 90ull) << 20) +
              (uint64_t)c * 65536 <= gpu.vram_free) {
        max_ctx_out = c;
        break;
      }
      if (c <= 512) break;
    }
    if (moe && gpu.vram_free > 0 && vram_needed > gpu.vram_free &&
        fit::cpu_offload_ok(d.meta, group_size, gpu))
      moe_cpu = true;
    // Без заголовка слои неизвестны — грубая проверка по параметрам; веса берём
    // настоящие (размер группы), а не догадку params × байт/параметр.
    if (!moe && gpu.vram_free > 0 && vram_needed > gpu.vram_free) {
      int ngl, layers;
      uint32_t sctx;
      uint64_t need;
      if (est::dense_partial_coarse(d.params, group_size, cuda, gpu.vram_free, kv_q8,
                                    cfg.ctx >= 512 ? (uint32_t)cfg.ctx : 4096u, ngl,
                                    layers, sctx, need)) {
        vram_needed = need;
        max_ctx_out = sctx;
        dense_partial = true;
        d_ngl = ngl;
        d_layers = layers;
      }
    }
  }
  bool gpu_known = gpu.vram_total > 0;
  bool fits = gpu_known && vram_needed <= gpu.vram_free;
  // partial: веса с оверхедом влезают — модель запустится с уменьшенным контекстом.
  bool partial = gpu_known && !fits && !dense_partial && group_size + overhead <= gpu.vram_free;
  json j = {
    {"ctx", ctx},
    {"vram_needed", vram_needed},
    {"fits", fits},
    {"partial", partial},
    {"gpu_known", gpu_known},
    {"max_ctx", max_ctx_out},
  };
  if (moe_cpu) j["moe_cpu"] = true;
  if (dense_partial) {
    j["dense_partial"] = true;
    j["ngl"] = d_ngl;
    j["n_layers"] = d_layers;
  }
  return j;
}

} // namespace

std::string search_json(const std::string& q_in, int limit, std::string& err) {
  std::string q = q_in;
  while (!q.empty() && std::isspace((unsigned char)q.front())) q.erase(q.begin());
  while (!q.empty() && std::isspace((unsigned char)q.back())) q.pop_back();
  if (q.empty()) { err = tr("пустой запрос"); return {}; }
  limit = std::clamp(limit, 1, 50);

  std::string key = q + "|" + std::to_string(limit);
  bool cached = false;
  json raw = json::array();
  {
    std::lock_guard l(g_search_mtx);
    auto it = g_search_cache.find(key);
    if (it != g_search_cache.end() && time(nullptr) - it->second.fetched < 300) {
      raw = it->second.raw;
      cached = true;
    }
  }
  if (!cached) {
    std::string url = "https://huggingface.co/api/models?search=" + url_encode(q) +
                      "&filter=gguf&sort=downloads&direction=-1&limit=" + std::to_string(limit);
    std::string out;
    if (!hf_get(url, out, err)) {
      hf_err(err, err);
      return {};
    }
    try {
      raw = json::parse(out);
    } catch (...) {
      err = tr("ответ HuggingFace не разобрался");
      return {};
    }
    if (!raw.is_array()) {
      err = tr("неожиданный ответ поиска");
      return {};
    }
    std::lock_guard l(g_search_mtx);
    cache_sweep(g_search_cache, 300, 64);
    g_search_cache[key] = {time(nullptr), raw};
  }

  config::Config cfg = config::current();
  const bool cuda = engine::launches_cuda(cfg); // под реальный движок, не «что может железо»
  // Светофор — по всему пулу карт: вопрос юзера «влезет ли у МЕНЯ», а у него
  // все карты разом (светофор каждой отдельной карте врёт про сплит).
  hw::Gpu gpu = fit::pool(hw::query_all(), cuda ? hw::Build::Cuda13 : hw::Build::Vulkan);
  json results = json::array();
  for (auto& m : raw) {
    if (!m.is_object()) continue;
    std::string repo = m.value("id", "");
    if (repo.empty()) continue;
    auto c = est::coarse(repo, (uint32_t)cfg.ctx, cuda, gpu.vram_free,
                         /*kv_q8=*/cfg.auto_params);
    size_t slash = repo.find('/');
    results.push_back({
      {"repo", repo},
      {"author", slash == std::string::npos ? "" : repo.substr(0, slash)},
      {"downloads", m.value("downloads", (uint64_t)0)},
      {"likes", m.value("likes", 0)},
      {"last_modified", m.value("lastModified", "")},
      {"fit", {
        {"known", c.known},
        {"params", c.params},
        {"size_est", c.total},
        {"vram_needed", c.gpu_total ? c.gpu_total : c.total},
        {"fits", c.fits},
        {"partial", c.partial},
        {"dense_partial", c.dense_partial},
        {"ngl", c.ngl},
        {"n_layers", c.layers},
        {"gpu_known", gpu.vram_total > 0},
      }},
    });
  }
  return json{{"results", results}, {"ctx", cfg.ctx}, {"gpu", {
    {"name", gpu.name},
    {"vram_free", gpu.vram_free},
    {"vram_total", gpu.vram_total},
  }}}.dump(2);
}

// Путь в репозитории — MTP-модуль (родной ускоритель спекулятивного
// декодирования этой модели): подпапка MTP/ или имя файла mtp-*.
static bool mtp_repo_path(const std::string& path) {
  size_t slash = path.find_last_of('/');
  std::string fname = slash == std::string::npos ? path : path.substr(slash + 1);
  if (strncasecmp(fname.c_str(), "mtp-", 4) == 0) return true;
  std::string dir = slash == std::string::npos ? "" : path.substr(0, slash);
  return strcasecmp(dir.c_str(), "MTP") == 0;
}

std::string detail_json(const std::string& repo_in, std::string& err) {
  std::string repo = repo_in;
  if (!lh::normalize_hf_repo(repo, err)) return {};

  DetailCache d;
  bool cached = false;
  {
    std::lock_guard l(g_detail_mtx);
    auto it = g_detail_cache.find(repo);
    if (it != g_detail_cache.end() && time(nullptr) - it->second.fetched < 600) {
      d = it->second;
      cached = true;
    }
  }
  if (!cached) {
    d = fetch_detail(repo, err);
    if (!err.empty()) return {};
    std::lock_guard l(g_detail_mtx);
    cache_sweep(g_detail_cache, 600, 128);
    g_detail_cache[repo] = d;
  }

  config::Config cfg = config::current();
  // Gpu.build знает лишь «что может железо» (на машине с Vulkan-рантаймом он
  // Vulkan, даже когда выбрана cuda13) — светофор считается под реальный движок.
  const bool cuda = engine::launches_cuda(cfg);
  hw::Gpu gpu = fit::pool(hw::query_all(), cuda ? hw::Build::Cuda13 : hw::Build::Vulkan);

  // Группы шардов: базовое имя → сумма размеров, число частей, первый файл.
  // mmproj-* (vision-проекции) — НЕ кванты модели: в списке они выглядели
  // как «F16/BF16/F32 на 876 МБ» с влезающими вердиктами, хотя как модель
  // такой файл не запускается. Считаем их отдельно — для пометки в статусе.
  struct Group {
    std::string base, path, quant;
    uint64_t size = 0;
    int parts = 0;
  };
  std::map<std::string, Group> groups;
  int mmproj_files = 0;
  uint64_t mmproj_bytes = 0;
  std::vector<std::pair<std::string, uint64_t>> mmproj_list; // (путь, размер)
  int mtp_files = 0;
  uint64_t mtp_bytes = 0;
  std::vector<std::pair<std::string, uint64_t>> mtp_list;    // (путь, размер)
  for (auto& f : d.files) {
    std::string fname = f.path;
    size_t fs = fname.find_last_of('/');
    if (fs != std::string::npos) fname = fname.substr(fs + 1);
    if (mtp_repo_path(f.path)) {
      ++mtp_files;
      mtp_bytes += f.size;
      mtp_list.push_back({f.path, f.size});
      continue;
    }
    if (strncasecmp(fname.c_str(), "mmproj", 6) == 0) {
      ++mmproj_files;
      mmproj_bytes += f.size;
      mmproj_list.push_back({f.path, f.size});
      continue;
    }
    if (strncasecmp(fname.c_str(), "imatrix", 7) == 0) continue; // служебные веса калибровки
    std::string base = shard_base(f.path);
    Group& g = groups[base];
    if (g.parts == 0) {
      g.path = f.path; // первый встреченный — у шардов это «00001»
      g.base = base;
      g.quant = est::quant_from_name(f.path);
    }
    g.size += f.size;
    int idx, count;
    shard_info(f.path, idx, count);
    if (count > g.parts) g.parts = count;
  }

  std::vector<Group*> order;
  for (auto& [_, g] : groups) order.push_back(&g);
  std::sort(order.begin(), order.end(), [](auto* a, auto* b) { return a->size < b->size; });

  // Рекомендация. Целиком в VRAM — наибольший из влезающих (скорость и качество
  // вместе). Если запуск возможен только гибридом (эксперты на CPU), гнаться за
  // наибольшим квантом бессмысленно: Q8_0 в RAM не даёт заметного прироста, а
  // качается и грузится втрое дольше — берём по лестнице разумного качества,
  // от классического Q4_K_M вниз.
  static const char* const moe_pref[] = {
      "Q4_K_M", "Q4_K_S", "IQ4_XS", "Q4_K", "IQ4_NL", "Q4_0", "Q3_K_XL", "Q3_K_M",
      "Q3_K_S", "IQ3_XXS", "IQ2_M", "IQ2_XXS", "IQ1_M", "IQ1_S",
  };
  bool any_moe_cpu = false;
  for (auto* g : order) {
    uint32_t mc;
    json fj = fit_json(d, g->size, (uint32_t)cfg.ctx, gpu, mc, cuda);
    if (fj.value("moe_cpu", false)) { any_moe_cpu = true; break; }
  }
  Group* rec = nullptr;
  if (any_moe_cpu) {
    auto find_moe = [&](const char* want) -> Group* {
      for (auto* g : order) {
        uint32_t mc;
        json fj = fit_json(d, g->size, (uint32_t)cfg.ctx, gpu, mc, cuda);
        if (fj.value("moe_cpu", false) && g->quant == want) return g;
      }
      return nullptr;
    };
    for (const char* want : moe_pref)
      if ((rec = find_moe(want))) break; // первый приоритет — и стоп
  }
  if (!rec) {
    // Наибольший (самый качественный) квант, который влезает целиком; ничего
    // не влезло — самый маленький, запустится с урезанным ctx.
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
      uint32_t mc;
      json fj = fit_json(d, (*it)->size, (uint32_t)cfg.ctx, gpu, mc, cuda);
      if (fj.value("fits", false)) {
        rec = *it;
        break;
      }
    }
    if (!rec && !order.empty()) rec = order.front();
  }

  // Что уже скачано: <models_dir>/<org>/<repo>/<файл>.
  std::string dir = config::models_dir_resolved(cfg);
  std::string repo_dir = dir + "/" + repo;

  json files = json::array();
  for (auto* g : order) {
    uint32_t mc = 0;
    json fj = fit_json(d, g->size, (uint32_t)cfg.ctx, gpu, mc, cuda);
    std::string local = repo_dir + "/" + g->path;
    // shard_base уже кончается на «-» перед номером части («…XL-»).
    // «-*» сверху удваивает тире: «…XL--*» не совпадает с
    // «…XL-00001-of-00004.gguf», и pull отказывается качать.
    std::string filter = g->parts > 1 ? g->base + "*" : g->path;
    files.push_back({
      {"kind", "quant"},
      {"quant", g->quant.empty() ? "?" : g->quant},
      {"path", g->path},
      {"filter", filter},
      {"parts", g->parts},
      {"size", g->size},
      {"fit", fj},
      {"downloaded", lh::file_exists(local)},
    });
  }
  // Vision-проекции — в конец списка, с явной ролью: к модели для картинок.
  for (auto& mp : mmproj_list) {
    std::string local = repo_dir + "/" + mp.first;
    files.push_back({
      {"kind", "mmproj"},
      {"quant", "mmproj " + est::quant_from_name(mp.first)},
      {"path", mp.first},
      {"filter", mp.first},
      {"parts", 1},
      {"size", mp.second},
      {"downloaded", lh::file_exists(local)},
    });
  }
  // MTP-модули — в самом конце списка. Скачивание кванта забирает один
  // подходящий по классу кванта (Q4 к Q4), не всю папку. Отдельной строкой
  // модуль по-прежнему можно скачать самому.
  for (auto& mp : mtp_list) {
    std::string local = repo_dir + "/" + mp.first;
    files.push_back({
      {"kind", "mtp"},
      {"quant", "MTP-модуль " + est::quant_from_name(mp.first)},
      {"path", mp.first},
      {"filter", mp.first},
      {"parts", 1},
      {"size", mp.second},
      {"downloaded", lh::file_exists(local)},
    });
  }

  json meta = {
    {"precise", d.precise},
    {"arch", d.meta.arch},
    {"name", d.meta.name},
    {"n_layers", d.meta.n_layers},
    {"n_ctx_train", d.meta.n_ctx_train},
    {"n_experts", d.meta.n_experts},
    {"params_from_name", d.precise ? 0.0 : d.params},
  };
  return json{
    {"repo", repo},
    {"gated", d.gated},
    {"meta", meta},
    {"mmproj_files", mmproj_files},
    {"mmproj_bytes", mmproj_bytes},
    {"mtp_files", mtp_files},
    {"mtp_bytes", mtp_bytes},
    {"files", files},
    {"recommended", rec ? (rec->parts > 1 ? rec->base + "*" : rec->path) : ""},
    {"ctx", cfg.ctx},
    {"gpu", {
      {"name", gpu.name},
      {"vram_free", gpu.vram_free},
      {"vram_total", gpu.vram_total},
    }},
  }.dump(2);
}

} // namespace lh::catalog
