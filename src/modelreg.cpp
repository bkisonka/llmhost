#include "modelreg.h"
#include "image.h"
#include "util.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <map>
#include <mutex>
#include <set>
#include <system_error>

namespace lh::modelreg {

namespace fs = std::filesystem;

namespace {

constexpr size_t MAX_FILES = 300;
constexpr int MAX_DEPTH = 4;

struct CacheKey { uint64_t size; uint64_t mtime; bool operator==(const CacheKey&) const = default; };

std::mutex g_cache_mtx;
std::map<std::string, std::pair<CacheKey, Model>> g_cache;

// Небезопасные имена: вотчер сканирует каждые 2 с — жалуемся один раз на путь.
std::mutex g_badname_mtx;
std::set<std::string> g_badname;

// Размер и время изменения для отпечатка кэша. mtime — как есть из
// file_time_type (time_since_epoch): любое стабильное представление годится.
bool stat_file(const std::string& path, uint64_t& size, uint64_t& mtime) {
  std::error_code ec;
  uint64_t sz = fs::file_size(path, ec);
  if (ec) return false;
  auto t = fs::last_write_time(path, ec);
  if (ec) return false;
  size = sz;
  mtime = (uint64_t)t.time_since_epoch().count();
  return true;
}

Model describe(const std::string& path) {
  Model m;
  m.path = path;
  uint64_t size = 0, mtime = 0;
  if (!stat_file(path, size, mtime)) return m;
  {
    std::lock_guard lock(g_cache_mtx);
    auto it = g_cache.find(path);
    if (it != g_cache.end() && it->second.first == CacheKey{size, mtime}) return it->second.second;
  }
  m.size = size;
  m.mtime = mtime;
  gguf::Meta meta;
  std::string err;
  if (model_meta(path, meta)) {
    m.meta = meta;
    m.name = meta.name;
    m.arch = meta.arch;
    m.quant = meta.quant;
  } else {
    m.quant = "не gguf";
    // Растущий файл (копируется/скачивается) не разбирается много раз подряд —
    // шумим в журнале только первой неудачей для каждого пути.
    static std::mutex complained_mtx;
    static std::set<std::string> complained;
    bool first = false;
    {
      std::lock_guard l2(complained_mtx);
      first = complained.insert(path).second;
    }
    if (first) logf("registry: %s — %s (возможно, файл ещё копируется)", path.c_str(),
                    err.c_str());
  }
  if (m.name.empty()) {
    // Имя файла без расширения: «Qwen3-8B-Q4_K_M.gguf» → «Qwen3-8B-Q4_K_M».
    std::string file = path;
    size_t slash = file.find_last_of('/');
    if (slash != std::string::npos) file = file.substr(slash + 1);
    size_t dot = file.find_last_of('.');
    if (dot != std::string::npos) file = file.substr(0, dot);
    m.name = file;
  }
  std::lock_guard lock(g_cache_mtx);
  g_cache[path] = {{size, mtime}, m};
  return m;
}

// Рекурсивный обход: файлов ≤ MAX_FILES, глубина ≤ MAX_DEPTH (считается по
// относительной глубине от dir). Симлинки на каталоги итератор по умолчанию
// не разворачивает — зацикливания не будет.
void walk(const std::string& dir, std::vector<Model>& out) {
  std::error_code ec;
  for (fs::recursive_directory_iterator it(dir, ec), end;
       !ec && it != end && out.size() < MAX_FILES; it.increment(ec)) {
    std::error_code ec2;
    if (it->is_directory(ec2)) {
      // Верхнеуровневые категории sd-server (diffusion_models/ и пр.) —
      // территория image-движка: энкодеры — легитимные GGUF (мусорные
      // «чат-модели» в реестре), VAE — «не gguf». В llama-реестр не идём.
      if (it.depth() == 0 && lh::image::is_catalog_dir_name(it->path().filename().string())) {
        it.disable_recursion_pending();
        continue;
      }
      // Спускаться в подкаталог, только если его файлы ещё в пределах глубины.
      if (ec2 || it.depth() + 1 > MAX_DEPTH) it.disable_recursion_pending();
      continue;
    }
    std::string full = it->path().string();
    std::string name = it->path().filename().string();
    bool gguf = name.size() > 5 && name.compare(name.size() - 5, 5, ".gguf") == 0;
    // Файловые системы допускают control-символы в именах (кроме NUL и «/»):
    // «\n[секция]» в имени .gguf — готовая инъекция в INI-пресет роутера.
    // Такие файлы в реестр не берём вовсе — не попадут ни в пресет, ни в API,
    // ни в UI.
    if (gguf && !lh::safe_local_filename(full)) {
      std::lock_guard l(g_badname_mtx);
      if (g_badname.insert(full).second)
        logf("registry: небезопасное имя файла (control-символы) — пропущено: %s",
             full.c_str());
      continue;
    }
    if (gguf && out.size() < MAX_FILES) out.push_back(describe(full));
  }
}

} // namespace

std::vector<Model> scan(const std::string& dir) {
  std::vector<Model> out;
  if (!dir.empty()) walk(dir, out);
  // Вытеснение кэша describe(): он пополнялся вечно, записи удалённых и
  // переименованных моделей копились до рестарта демона. describe() зовётся
  // только из walk, так что всё, что этот скан не видел, — протухло.
  {
    std::set<std::string> seen;
    for (auto& m : out) seen.insert(m.path);
    std::lock_guard lock(g_cache_mtx);
    for (auto it = g_cache.begin(); it != g_cache.end();) {
      if (seen.count(it->first)) ++it;
      else it = g_cache.erase(it);
    }
  }
  std::sort(out.begin(), out.end(), [](const Model& a, const Model& b) {
    return a.path < b.path;
  });
  return out;
}

bool is_mtp_module(const std::string& path) {
  std::string file = path;
  size_t s = file.find_last_of('/');
  std::string dir = s == std::string::npos ? "" : file.substr(0, s);
  if (s != std::string::npos) file = file.substr(s + 1);
  // Подпапка репозитория MTP/ — основной признак (так кладёт unsloth).
  size_t ds = dir.find_last_of('/');
  std::string dname = ds == std::string::npos ? dir : dir.substr(ds + 1);
  if (dname == "MTP") return true;
  // Иное именование: mtp-*.gguf.
  size_t d = file.find_last_of('.');
  std::string stem = d == std::string::npos ? file : file.substr(0, d);
  return stem.rfind("mtp-", 0) == 0;
}

namespace {

bool path_under(const std::string& root, const std::string& path) {
  if (root.empty() || path.size() <= root.size()) return false;
  if (path.compare(0, root.size(), root) != 0) return false;
  const char c = path[root.size()];
  return c == '/' || c == '\\';
}

} // namespace

int mtp_place(const std::string& model_path, const std::string& mtp_path) {
  if (!is_mtp_module(mtp_path)) return 0;
  const size_t slash = model_path.find_last_of("/\\");
  if (slash == std::string::npos) return 0;
  const std::string dir = model_path.substr(0, slash);
  if (path_under(dir, mtp_path)) return 1;
  const size_t up = dir.find_last_of("/\\");
  if (up == std::string::npos) return 0;
  const std::string parent = dir.substr(0, up);
  const bool win = dir.find('\\') != std::string::npos && dir.find('/') == std::string::npos;
  const char sep = win ? '\\' : '/';
  if (path_under(parent + sep + "MTP", mtp_path)) return 2;
  return 0;
}

namespace {

std::string lower_copy(std::string s) {
  for (char& c : s) c = (char)std::tolower((unsigned char)c);
  return s;
}

// Начало квантового суффикса: q4, q8_0, iq4, tq1 (хвост — K/M/XS/0).
bool quant_start(const std::string& t) {
  size_t b = 0;
  if (!t.empty() && (t[0] == 'i' || t[0] == 't')) b = 1;
  return t.size() > b + 1 && t[b] == 'q' && t[b + 1] >= '0' && t[b + 1] <= '9';
}

bool quant_token(const std::string& t) {
  if (t == "f16" || t == "bf16" || t == "fp16" || t == "f32") return true;
  return quant_start(t);
}

// Обломок квантового суффикса после самого кванта: K, M, S, XL, XS.
bool quant_tail_frag(const std::string& t) {
  return t.size() <= 2;
}

std::vector<std::string> split_tokens(const std::string& s) {
  std::vector<std::string> out;
  size_t pos = 0;
  while (pos <= s.size()) {
    size_t dash = s.find('-', pos);
    if (dash == std::string::npos) {
      if (pos < s.size()) out.push_back(s.substr(pos));
      break;
    }
    out.push_back(s.substr(pos, dash - pos));
    pos = dash + 1;
  }
  return out;
}

} // namespace

bool mtp_matches_trunk(const std::string& model_path, const std::string& mtp_path) {
  std::string stem = registry_stem(mtp_path);
  if (stem.size() > 4 && lower_copy(stem.substr(0, 4)) == "mtp-") stem.erase(0, 4);
  std::vector<std::string> mtok = split_tokens(stem);
  // Хвост имени — квант с обломками («Q4_K_M», «Q8_0», «IQ4_XS»): ищем
  // квантовый токен с конца, до него — только обломки; не нашли — имя
  // целиком годится в базу.
  long q = -1;
  for (long i = (long)mtok.size() - 1; i >= 0; --i) {
    const std::string t = lower_copy(mtok[(size_t)i]);
    if (quant_token(t)) {
      q = i;
      break;
    }
    if (!quant_tail_frag(t)) break; // обычное слово имени — кванта в хвосте нет
  }
  if (q <= 0) return false; // пустая база или имя без «mtp-» — не на что вязать
  const std::vector<std::string> ttok = split_tokens(registry_stem(model_path));
  if ((size_t)q > ttok.size()) return false;
  for (long i = 0; i < q; ++i)
    if (lower_copy(mtok[(size_t)i]) != lower_copy(ttok[(size_t)i])) return false;
  return true;
}

bool is_mmproj_module(const std::string& path) {
  std::string file = path;
  size_t s = file.find_last_of('/');
  if (s != std::string::npos) file = file.substr(s + 1);
  size_t d = file.find_last_of('.');
  std::string stem = d == std::string::npos ? file : file.substr(0, d);
  // «mmproj» в любом месте стема: и апстримное «mmproj-F16.gguf», и ручное
  // «qwen36-mmproj-BF16.gguf». Прежний «только префикс» заводил рукименным
  // проекциям слот в реестре, как обычной модели (урок 29.10).
  return lower_copy(stem).find("mmproj") != std::string::npos;
}

void shard_info(const std::string& path, int& index, int& count) {
  index = 0;
  count = 1;
  if (path.size() < 6) return;
  std::string stem = path.substr(0, path.size() - 5); // без «.gguf»
  size_t of = stem.rfind("-of-");
  if (of == std::string::npos || of < 5) return;
  auto digits = [&](size_t from, size_t to) {
    if (from >= to || to - from > 9) return false; // stoi не должен переполняться
    for (size_t i = from; i < to; ++i)
      if (!std::isdigit((unsigned char)stem[i])) return false;
    return true;
  };
  // «base-00001-of-00003»: номер части — цифры перед «-of-», всего частей —
  // сразу после. Число после «-of-» должно идти до конца стема.
  size_t idx_end = of; // «-» перед номером части
  size_t idx_begin = stem.find_last_not_of("0123456789", idx_end - 1);
  if (idx_begin == std::string::npos) return;
  if (!digits(idx_begin + 1, idx_end)) return;
  size_t total_begin = of + 4;
  if (!digits(total_begin, stem.size())) return;
  index = std::stoi(stem.substr(idx_begin + 1, idx_end - idx_begin - 1));
  count = std::stoi(stem.substr(total_begin));
  if (index < 1 || count < 1) { index = 0; count = 1; }
}

std::string shard_base(const std::string& path) {
  int idx, count;
  shard_info(path, idx, count);
  std::string stem = path.substr(0, path.size() - 5);
  if (count > 1) {
    size_t of = stem.rfind("-of-");
    size_t idx_begin = of == std::string::npos
                           ? std::string::npos
                           : stem.find_last_not_of("0123456789", of - 1);
    if (idx_begin != std::string::npos) stem = stem.substr(0, idx_begin + 1);
  }
  return stem;
}

bool is_shard_continuation(const std::string& path) {
  int idx = 0, count = 1;
  shard_info(path, idx, count);
  return count > 1 && idx > 1;
}

std::string registry_stem(const std::string& path) {
  int idx = 0, count = 1;
  shard_info(path, idx, count);
  if (count > 1) {
    std::string b = shard_base(path);
    size_t sl = b.find_last_of("/\\");
    if (sl != std::string::npos) b = b.substr(sl + 1);
    if (!b.empty() && b.back() == '-') b.pop_back();
    if (!b.empty()) return b;
  }
  std::string base = path;
  size_t slash = base.find_last_of("/\\");
  if (slash != std::string::npos) base = base.substr(slash + 1);
  if (base.size() > 5) {
    const char* e = base.c_str() + base.size() - 5;
    auto eq = [](char a, char b) {
      return std::tolower((unsigned char)a) == std::tolower((unsigned char)b);
    };
    if (eq(e[0], '.') && eq(e[1], 'g') && eq(e[2], 'g') && eq(e[3], 'u') && eq(e[4], 'f'))
      base.erase(base.size() - 5);
  }
  return base;
}

// Ядро сигнатуры проекции — из её собственного имени, без пути и «.gguf».
// Сплит по «-» — как у mtp_matches_trunk: «Q8_0» остаётся одним токеном и
// уходит целиком как квант; не-алфанумерика внутри токена («qwen3.6») сняется
// посимвольно, чтобы ядро сравнивалось с нормализованным именем ствола.
std::string mmproj_core_of(const std::string& mmproj_path) {
  const std::string stem = registry_stem(mmproj_path);
  std::string core;
  for (const std::string& t : split_tokens(stem)) {
    const std::string lt = lower_copy(t);
    // «model» — стоп-слово апстримного «mmproj-model-f16.gguf» (Qwen2-VL):
    // сигнатуры в имени нет, файл остаётся легаси-кандидатом.
    if (lt == "mmproj" || lt == "model") continue;
    if (quant_token(lt) || lt == "f64") continue;
    for (char c : lt)
      if (std::isalnum((unsigned char)c)) core += c;
  }
  return core;
}

bool mmproj_matches_trunk(const std::string& trunk_path,
                          const std::string& mmproj_path) {
  const std::string core = mmproj_core_of(mmproj_path);
  // Ниже 4 символов ядро ловит случайные подпоследовательности («q3b»
  // в «Qwen3-8B») — такая сигнатура родства не доказывает.
  if (core.size() < 4) return false;
  std::string tn;
  for (char c : registry_stem(trunk_path))
    if (std::isalnum((unsigned char)c)) tn += (char)std::tolower((unsigned char)c);
  size_t pos = 0;
  for (char c : core) {
    const size_t hit = tn.find(c, pos);
    if (hit == std::string::npos) return false;
    pos = hit + 1;
  }
  return true;
}

uint64_t shard_group_bytes(const std::string& path) {
  int idx = 0, count = 1;
  shard_info(path, idx, count);
  std::error_code ec;
  uint64_t self = fs::file_size(path, ec);
  if (ec) self = 0;
  if (count <= 1) return self;
  const std::string base = shard_base(path);
  fs::path file(path);
  if (!file.has_parent_path()) return self;
  uint64_t sum = 0;
  int seen = 0;
  for (fs::directory_iterator it(file.parent_path(), ec), end; !ec && it != end;
       it.increment(ec)) {
    if (ec) break;
    std::error_code fec;
    if (!it->is_regular_file(fec) || fec) continue;
    std::string fp = it->path().string();
    int i2 = 0, c2 = 1;
    shard_info(fp, i2, c2);
    if (c2 > 1 && shard_base(fp) == base) {
      uint64_t sz = it->file_size(fec);
      if (!fec) {
        sum += sz;
        ++seen;
      }
    }
  }
  return seen ? sum : self;
}

bool model_meta(const std::string& path, gguf::Meta& out) {
  std::string err;
  if (!gguf::parse_header(path, out, err)) return false;
  int idx = 0, count = 1;
  shard_info(path, idx, count);
  if (count <= 1) return true;
  // Тензорные списки живут в частях 2..N (у заголовочной tensor_count = 0):
  // суммируем экспертные байты, CPU-тензоры и поблочную раскладку.
  const std::string base = shard_base(path);
  fs::path file(path);
  if (!file.has_parent_path()) return true;
  std::error_code ec;
  for (fs::directory_iterator it(file.parent_path(), ec), end; !ec && it != end;
       it.increment(ec)) {
    if (ec) break;
    std::error_code fec;
    if (!it->is_regular_file(fec) || fec) continue;
    const std::string fp = it->path().string();
    int i2 = 0, c2 = 1;
    shard_info(fp, i2, c2);
    if (c2 <= 1 || i2 <= idx || shard_base(fp) != base) continue;
    gguf::Meta part;
    if (!gguf::parse_header(fp, part, err)) continue; // растущая/битая часть
    out.expert_bytes += part.expert_bytes;
    out.cpu_extra_bytes += part.cpu_extra_bytes;
    if (part.expert_bytes_by_block.size() > out.expert_bytes_by_block.size())
      out.expert_bytes_by_block.resize(part.expert_bytes_by_block.size());
    for (size_t i = 0; i < part.expert_bytes_by_block.size(); ++i)
      out.expert_bytes_by_block[i] += part.expert_bytes_by_block[i];
  }
  return true;
}

std::vector<Companion> companions_of(const std::string& model_path,
                                     const std::vector<Model>& list) {
  std::vector<Companion> out;
  size_t slash = model_path.find_last_of("/\\");
  if (slash == std::string::npos) return out;
  const std::string dir = model_path.substr(0, slash);
  int idx0, cnt0;
  shard_info(model_path, idx0, cnt0);
  const std::string my_base = cnt0 > 1 ? shard_base(model_path) : std::string();

  // Оставшаяся основная модель (не модуль, не часть собственной шард-группы
  // удаляемой — те уйдут вместе с ней).
  auto main_left = [&](const std::string& p) {
    if (p == model_path) return false;
    if (is_mtp_module(p) || is_mmproj_module(p)) return false;
    if (!my_base.empty()) {
      int i2, c2;
      shard_info(p, i2, c2);
      if (c2 > 1 && shard_base(p) == my_base) return false;
    }
    return true;
  };

  for (auto& m : list) {
    if (m.path == model_path) continue;
    size_t s = m.path.find_last_of("/\\");
    if (s == std::string::npos) continue;
    std::string pdir = m.path.substr(0, s);
    if (is_mmproj_module(m.path)) {
      // find_mmproj ищет проекцию только рядом с моделью.
      if (pdir != dir) continue;
      bool shared = false;
      for (auto& o : list)
        if (main_left(o.path)) {
          size_t os = o.path.find_last_of("/\\");
          if (os != std::string::npos && o.path.substr(0, os) == dir) {
            shared = true;
            break;
          }
        }
      if (!shared) out.push_back({m.path, "mmproj", m.size});
      continue;
    }
    if (is_mtp_module(m.path)) {
      // Тот же охват, что у find_draft: поддерево папки файла и MTP/ уровнем выше.
      if (mtp_place(model_path, m.path) == 0) continue;
      bool shared = false;
      for (auto& o : list) {
        if (!main_left(o.path)) continue;
        if (mtp_place(o.path, m.path) != 0) {
          shared = true;
          break;
        }
      }
      if (!shared) out.push_back({m.path, "mtp", m.size});
      continue;
    }
    // Часть собственной шард-группы — та же модель, уходит вместе с ней.
    if (!my_base.empty() && pdir == dir) {
      int i2, c2;
      shard_info(m.path, i2, c2);
      if (c2 > 1 && shard_base(m.path) == my_base)
        out.push_back({m.path, "shard", m.size});
    }
  }
  return out;
}

} // namespace lh::modelreg
