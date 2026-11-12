// Движок llama-server: источник правды о пине — [general] engine-release в
// llmhost.conf, manifest.json рядом с бинарями — каталог сборок и кэш
// разрешённых url/sha256 (его переписывает pin()). Установка = скачать
// (докачка, SHA256) + распаковать (Linux — tar.gz через /usr/bin/tar, Windows —
// zip через miniz) с проверкой содержимого: лимиты от архивных бомб и пути наружу.
#include "engine.h"


#include "config.h"
#include "download.h"
#include "engine_assets.h"
#include "hw.h"
#include "proc.h"
#include "profiles.h"

#include "util.h"

#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <miniz.h>
#include <cctype>
#include <cstdio>
#else
#include <sys/stat.h>
#include <unistd.h>
#include <set>
#endif

#include <chrono>
#include <filesystem>
#include <map>
#include <mutex>
#include <system_error>
#include <thread>
#include <vector>

namespace lh::engine {

namespace fs = std::filesystem;

namespace {

#ifdef _WIN32
constexpr const char* SERVER_EXE = "llama-server.exe";
#else
constexpr const char* SERVER_EXE = "llama-server";
#endif

struct Manifest {
  struct File { std::string url; std::string sha256; uint64_t size = 0; };
  struct Build { std::string id; std::vector<File> files; };
  std::string release;
  std::vector<Build> builds;
};

std::string engine_dir() { return lh::app_dir() + "/engine"; }
std::string build_dir(const std::string& id) { return engine_dir() + "/" + id; }

Manifest manifest() {
  Manifest m;
  std::string raw = lh::read_small_file(lh::exe_dir() + "/manifest.json");
  if (raw.empty()) {
    logf("engine: manifest.json не найден рядом с программой");
    return m;
  }
  try {
    auto j = nlohmann::json::parse(raw);
    m.release = j.value("release", "");
    for (auto& b : j.at("builds")) {
      Manifest::Build nb;
      nb.id = b.at("id").get<std::string>();
      if (b.contains("files"))
        for (auto& f : b.at("files"))
          nb.files.push_back({f.at("url").get<std::string>(),
                              f.value("sha256", ""), f.value("size", (uint64_t)0)});
      m.builds.push_back(std::move(nb));
    }
  } catch (const std::exception& e) {
    logf("engine: manifest не читается: %s", e.what());
  }
  return m;
}

State g_state;

std::mutex g_state_mtx;

void set_state(const State& s) { std::lock_guard l(g_state_mtx); g_state = s; }

#ifdef _WIN32

// fwrite-колбэк: miniz зовёт его для записи распакованного файла. Счёт
// ФАКТИЧЕСКИ выписанных байтов на весь архив: сумма m_uncomp_size из
// заголовков — лишь заявка злоумышленника, честные мелкие заголовки при
// настоящем раздутии проходят проверку по заголовкам и заполняют диск без
// лимита в колбэке. Превышение — 0, miniz прерывает распаковку.
struct UnpackCtx {
  FILE* f;
  uint64_t* total; // накоплено за весь архив
  uint64_t cap;    // MAX_TOTAL
};
size_t mz_write_cb(void* opaque, mz_uint64, const void* buf, size_t n) {
  UnpackCtx* c = (UnpackCtx*)opaque;
  if (*c->total + n > c->cap) return 0;
  size_t put = fwrite(buf, 1, n, c->f);
  *c->total += put;
  return put;
}

// Зарезервированные Windows-имена (базовый сегмент без расширения): fopen
// пустит запись в консольное устройство вместо файла.
bool win_reserved_name(const std::string& name) {
  size_t slash = name.find_last_of("/\\");
  std::string base = slash == std::string::npos ? name : name.substr(slash + 1);
  size_t dot = base.find('.');
  std::string stem = dot == std::string::npos ? base : base.substr(0, dot);
  for (auto& ch : stem) ch = (char)toupper((unsigned char)ch);
  if (stem == "CON" || stem == "PRN" || stem == "AUX" || stem == "NUL") return true;
  if (stem.size() == 4 && (stem.rfind("COM", 0) == 0 || stem.rfind("LPT", 0) == 0) &&
      stem[3] >= '1' && stem[3] <= '9')
    return true;
  return false;
}

// Распаковка zip в папку: только безопасные относительные пути. Лимиты —
// от zip-бомбы: файлов и суммарного объёма.
bool unpack(const std::string& zip_path, const std::string& into, std::string& err) {
  constexpr int MAX_FILES = 4096;
  constexpr uint64_t MAX_TOTAL = 16ull << 30; // 16 ГиБ
  FILE* in = fopen(zip_path.c_str(), "rb");
  if (!in) {
    err = "zip не открывается";
    return false;
  }
  mz_zip_archive z{};
  mz_zip_zero_struct(&z);
  if (!mz_zip_reader_init_cfile(&z, in, 0, 0)) {
    fclose(in);
    err = "zip не читается (повреждён?)";
    return false;
  }
  int n = mz_zip_reader_get_num_files(&z);
  if (n > MAX_FILES) {
    mz_zip_reader_end(&z);
    fclose(in);
    err = "в архиве слишком много файлов";
    return false;
  }
  uint64_t total = 0;
  uint64_t written = 0; // фактические байты колбэка на весь архив
  for (int i = 0; i < n; ++i) {
    mz_zip_archive_file_stat st;
    if (!mz_zip_reader_file_stat(&z, i, &st)) continue;
    total += st.m_uncomp_size;
    if (total > MAX_TOTAL) {
      mz_zip_reader_end(&z);
      fclose(in);
      err = "распакованный объём сверх лимита (16 ГиБ)";
      return false;
    }
    std::string name = st.m_filename;
    // Защита: без "..", ":" и абсолютных путей (архив от пинованного сервера,
    // но проверка стоит копейки).
    bool safe = !name.empty() && name[0] != '/' && name[0] != '\\' && name.find(':') == std::string::npos;
    if (safe) {
      size_t pos = 0;
      while (safe) {
        size_t next = name.find_first_of("/\\", pos);
        std::string part = name.substr(pos, next == std::string::npos ? next : next - pos);
        if (part == ".." || part.empty() && next != std::string::npos) safe = false;
        if (next == std::string::npos) break;
        pos = next + 1;
      }
    }
    if (safe && !st.m_is_directory && win_reserved_name(name)) safe = false;
    if (!safe) {
      mz_zip_reader_end(&z);
      fclose(in);
      err = "в архиве небезопасный путь: " + name;
      return false;
    }
    if (st.m_is_directory) continue;
    std::string dest = into + "/" + name;
    lh::ensure_dir(dest.substr(0, dest.find_last_of('/')));
    FILE* out = fopen(dest.c_str(), "wb");
    if (!out) {
      mz_zip_reader_end(&z);
      fclose(in);
      err = "файл не создаётся: " + name;
      return false;
    }
    UnpackCtx uctx{out, &written, MAX_TOTAL};
    mz_bool ok = mz_zip_reader_extract_to_callback(&z, i, mz_write_cb, &uctx, 0);
    fclose(out);
    if (!ok) {
      mz_zip_reader_end(&z);
      fclose(in);
      fs::remove(dest);
      err = written >= MAX_TOTAL ? "распакованный объём сверх лимита (16 ГиБ) — архив остановлен"
                                 : "распаковка не удалась: " + name;
      return false;
    }
  }
  mz_zip_reader_end(&z);
  fclose(in);
  return true;
}

#else

// Распаковка tar.gz в стадийную папку: /usr/bin/tar как ребёнок. Тарболл —
// с пинованного сервера, но tar сам не ограничивает ни объём, ни число файлов,
// ни пути — потому после распаковки обязательна validate_stage() (ниже).
bool unpack_tar(const std::string& pkg, const std::string& into, std::string& err) {
  // Превентивная страховка от «бомбы»: validate_stage считает объём ПОСЛЕ
  // распаковки, а до неё тар пишет всё, что влезет, — архив из
  // скомпрометированного релиза заполнил бы диск до отказа раньше отказа.
  // Запас: типичное раздутие tar.gz llama.cpp ~2-3x, берём 4x плюс гига.
  {
    std::error_code sec;
    const uint64_t asz = lh::file_size(pkg);
    const auto sp = fs::space(into, sec);
    if (!sec && asz && sp.available < asz * 4 + (1ull << 30)) {
      char buf[64];
      snprintf(buf, sizeof(buf), "%.1f/%.1f ГиБ",
               (double)(asz * 4 + (1ull << 30)) / 1073741824.0,
               (double)sp.available / 1073741824.0);
      err = "распаковка требует свободного места (нужно/есть: " +
            std::string(buf) + ") — освободите диск";
      return false;
    }
  }
  lh::ensure_dir(into);
  proc::Child tar;
  if (!tar.start("/usr/bin/tar", {"-xzf", pkg, "-C", into, "--no-same-owner"})) {
    err = "tar не запускается (/usr/bin/tar недоступен?)";
    return false;
  }
  int code = 0;
  while (!tar.exited(code)) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  if (code != 0) {
    std::string line = proc::last_meaningful_line(tar.log_tail());
    err = "tar: код " + std::to_string(code) + (line.empty() ? "" : " — " + line);
    return false;
  }
  // Тарболлы llama.cpp упакованы в один корневой каталог (llama-bNNNNN/,
  // cudart-…/), причём у движка и cudart он РАЗНЫЙ — а RUNPATH библиотек
  // `$ORIGIN`: оставь мы два подкаталога, libggml-cuda.so не нашла бы
  // libcudart.so.12 из соседней папки. Windows-zip'ы тех же сборок плоские —
  // выравниваемся: единственный каталог в корне стадии поднимаем на уровень.
  std::error_code ec;
  fs::directory_iterator it(into, ec), end;
  if (ec) {
    err = "чтение стадии: " + ec.message();
    return false;
  }
  std::string only;
  int entries = 0;
  for (; it != end; it.increment(ec)) {
    if (ec) {
      err = "чтение стадии: " + ec.message();
      return false;
    }
    if (++entries > 1) break;
    only = it->path().string();
  }
  std::error_code ec2;
  if (entries == 1 && fs::is_directory(only, ec2) && !ec2) {
    // Пути собираем до переименований: rename во время итерации по каталогу
    // в POSIX может молча пропускать ещё не прочитанные записи.
    std::vector<fs::path> kids;
    for (fs::directory_iterator c(only, ec2), cend; !ec2 && c != cend; c.increment(ec2))
      kids.push_back(c->path());
    if (ec2) {
      err = "чтение " + only + ": " + ec2.message();
      return false;
    }
    for (const fs::path& k : kids)
      fs::rename(k, into + "/" + k.filename().string(), ec2);
    if (ec2) {
      err = "подъём содержимого " + only + ": " + ec2.message();
      return false;
    }
    fs::remove(only, ec2);
    if (ec2) {
      err = "удаление пустого корня: " + ec2.message();
      return false;
    }
  }
  return true;
}

// Пост-валидация распакованного (замена проверки записей до распаковки у zip):
// лимиты те же, что у zip-ветки — файлов и суммарный объём. Симлинки разрешены
// (официальные тарболлы разводят версии .so цепочками симлинков), но цель
// обязана быть относительной и лексически оставаться внутри стадии; «..» в
// относительных путях записей запрещён. Неразборчивая ошибка файловой
// системы — тоже отказ.
bool validate_stage(const std::string& dir, std::string& err) {
  constexpr int MAX_FILES = 4096;
  constexpr uint64_t MAX_TOTAL = 16ull << 30; // 16 ГиБ
  std::error_code ec;
  uint64_t total = 0;
  int files = 0;
  for (fs::recursive_directory_iterator it(dir, ec), end; !ec && it != end;
       it.increment(ec)) {
    std::error_code ec2;
    fs::file_status st = it->symlink_status(ec2);
    if (ec2) {
      err = "статус " + it->path().string() + ": " + ec2.message();
      return false;
    }
    for (const fs::path& part : it->path().lexically_relative(dir))
      if (part == "..") {
        err = "в пакете путь наружу: " + it->path().string();
        return false;
      }
    if (fs::is_symlink(st)) {
      std::error_code ec3;
      fs::path target = fs::read_symlink(it->path(), ec3);
      if (ec3 || target.empty()) {
        err = "симлинк " + it->path().string() + ": цель не читается";
        return false;
      }
      if (target.is_absolute()) {
        err = "абсолютный симлинк в пакете: " + it->path().string() + " → " +
              target.string();
        return false;
      }
      fs::path rel = (it->path().parent_path() / target).lexically_normal().lexically_relative(dir);
      if (rel.empty() || *rel.begin() == "..") {
        err = "симлинк наружу: " + it->path().string() + " → " + target.string();
        return false;
      }
      if (++files > MAX_FILES) {
        err = "в пакете слишком много файлов";
        return false;
      }
      continue;
    }
    if (st.type() != fs::file_type::regular) continue;
    uint64_t sz = it->file_size(ec2);
    if (ec2) {
      err = "размер " + it->path().string() + ": " + ec2.message();
      return false;
    }
    total += sz;
    if (++files > MAX_FILES) {
      err = "в пакете слишком много файлов";
      return false;
    }
    if (total > MAX_TOTAL) {
      err = "распакованный объём сверх лимита (16 ГиБ)";
      return false;
    }
  }
  if (ec) {
    err = "обход распакованной папки: " + ec.message();
    return false;
  }
  return true;
}

#endif // _WIN32

// Ищет SERVER_EXE в папке сборки: рекурсивно — тарболл раскладывает файлы
// по подкаталогам (bin/, lib/ и т.п.).
std::string locate_exe(const std::string& dir) {
  std::string direct = dir + "/" + SERVER_EXE;
  if (lh::file_exists(direct)) return direct;
  std::error_code ec;
  for (fs::recursive_directory_iterator it(dir, ec), end; !ec && it != end;
       it.increment(ec)) {
    std::error_code ec2;
    if (!it->is_regular_file(ec2) || ec2) continue;
    if (it->path().filename() == SERVER_EXE) return it->path().string();
  }
  return {};
}

// Перенос распакованного из стадийной папки на место: файлы по одному,
// поверх старых — fs::rename заменяет атомарно. Отказ — честная ошибка,
// сборка не остаётся наполовину заменённой. Каталог/файл — только через
// lstat (symlink_status): is_directory следует по симлинку, а у нас ссылки
// переименовываются РАНЬШЕ или ПОЗЖЕ своих целей (порядок выдачи каталога
// произвольный) — цель к моменту переноса ссылки уже может уехать, и stat
// даёт ложное ENOENT. Симлинк переносится как есть, цель не трогаем.
bool promote(const std::string& from, const std::string& to, std::string& err) {
  std::error_code ec;
  std::vector<fs::path> entries;
  for (fs::recursive_directory_iterator it(from, ec), end; !ec && it != end;
       it.increment(ec))
    entries.push_back(it->path());
  if (ec) {
    err = "обход распакованной папки: " + ec.message();
    return false;
  }
  for (const fs::path& p : entries) {
    std::error_code ec2;
    fs::file_status st = fs::symlink_status(p, ec2);
    if (ec2) {
      err = "статус " + p.string() + ": " + ec2.message();
      return false;
    }
    if (st.type() == fs::file_type::directory) continue;
    std::string rel = p.lexically_relative(from).string();
    std::string dest = to + "/" + rel;
    lh::ensure_dir(dest.substr(0, dest.find_last_of('/')));
    fs::rename(p, dest, ec2);
    if (ec2) {
      err = "не удалось заменить " + rel + ": " + ec2.message();
      return false;
    }
  }
  return true;
}

#ifndef _WIN32
// tar мог привезти бинарник без права выполнения — добавляем x поверх текущих.
bool add_exec_bit(const std::string& path, std::string& err) {
  std::error_code ec;
  fs::permissions(path, fs::perms::owner_exec | fs::perms::group_exec | fs::perms::others_exec,
                  fs::perm_options::add, ec);
  if (ec) {
    err = "chmod +x " + path + ": " + ec.message();
    return false;
  }
  return true;
}
#endif

// ---- факт на диске: релиз установленного бинарника --------------------------
//
// Пин манифеста — намерение, на диске может лежать другое: пересборка способна
// затереть обновлённый пин затравкой (случилось 04.10: b11382 перепинился
// обратно в b11344, а установленным остался b11382), бинарник могут обновить,
// а пин откатить. Истина — сам llama-server: «--version» печатает «… (build
// NNNNN, …)». Спавн на каждый запрос недопустим — кэш с ключом (размер,
// mtime): файл тронули — перепроверим.

// Спавн exe с --version и разбор «build NNNNN». Пусто — не запустился или не
// разобрался; ребёнок в своей группе, выходит мгновенно, потолок ~4.5 с.
std::string binary_release(const std::string& exe) {
  proc::Child c;
  if (!c.start(exe, {"--version"})) return {};
  int code = 0;
  bool done = false;
  for (int i = 0; i < 150; ++i) {
    if (c.exited(code)) {
      done = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
  }
  if (!done) {
    c.stop();
    return {};
  }
  return lh::release_from_version_output(c.log_tail());
}

struct RelEntry { uint64_t size = 0; fs::file_time_type mtime{}; std::string rel; };
std::mutex g_rel_mtx;
std::map<std::string, RelEntry> g_rel_cache;

#ifndef _WIN32
// Бинарник, который мы собираемся спрашивать --version (проживёт ли пин),
// не должен быть доступен на запись тем, кто его не создавал: иначе подмена
// файла в расшаренном LH_DATA_DIR заставляет демона исполнить чужое от своего
// имени. Граница доверия: писаемость всеми (S_IWOTH) запрещена всегда;
// писаемость группой — только когда владелец НЕ мы (у одиночных пользователей
// группа своя, 0775 из-за umask — не дыра, а реальная раскладка живых установок,
// ср. engine-директория 0775 живого демона). Владелец файла — мы или root,
// каталог — тот же владелец. Аномалию логируем один раз на путь (опрос идёт
// на каждом /v0/engine).
bool binary_trusted(const std::string& exe, std::string& why) {
  const uid_t me = ::geteuid();
  struct stat st;
  if (::stat(exe.c_str(), &st) != 0) { why = "stat не читается"; return false; }
  if (!S_ISREG(st.st_mode)) { why = "не обычный файл"; return false; }
  if ((st.st_mode & S_IWOTH) != 0) { why = "файл писаем всеми"; return false; }
  if ((st.st_mode & S_IWGRP) != 0 && st.st_uid != me) { why = "файл писаем группой чужого владельца"; return false; }
  if (st.st_uid != me && st.st_uid != 0) { why = "владелец файла не наш (и не root)"; return false; }
  std::string dir = exe.substr(0, exe.find_last_of('/'));
  struct stat dst;
  if (!dir.empty() && ::stat(dir.c_str(), &dst) == 0) {
    if ((dst.st_mode & S_IWOTH) != 0) { why = "каталог писаем всеми"; return false; }
    if ((dst.st_mode & S_IWGRP) != 0 && dst.st_uid != me) { why = "каталог писаем группой чужого владельца"; return false; }
    if (dst.st_uid != me && dst.st_uid != 0) { why = "владелец каталога не наш (и не root)"; return false; }
    // Замену записи в каталоге диктует право записи на каталог — оно уже
    // проверено двумя строками выше (all-writable отказ, group-writable
    // чужого владельца отказ). Каталог root без прав записи для остальных
    // безопасен и при нашем файле внутри.
  }
  return true;
}
#endif

std::string installed_release_of(const std::string& exe) {
  std::error_code ec;
  uint64_t sz = fs::file_size(exe, ec);
  auto mt = ec ? fs::file_time_type{} : fs::last_write_time(exe, ec);
  if (ec) return {};
#ifndef _WIN32
  {
    std::string why;
    if (!binary_trusted(exe, why)) {
      static std::mutex seen_mtx;
      static std::set<std::string> seen;
      std::lock_guard l(seen_mtx);
      if (seen.insert(exe).second)
        logf("engine: не опрашиваю «%s» (--version): %s — проверка пина против диска невозможна",
             exe.c_str(), why.c_str());
      return {};
    }
  }
#endif
  {
    std::lock_guard l(g_rel_mtx);
    auto it = g_rel_cache.find(exe);
    if (it != g_rel_cache.end() && it->second.size == sz && it->second.mtime == mt)
      return it->second.rel;
  }
  std::string rel = binary_release(exe);
  std::lock_guard l(g_rel_mtx);
  g_rel_cache[exe] = {sz, mt, std::move(rel)};
  return g_rel_cache[exe].rel;
}

// Пин обязывает: бинарник из распакованной стадии обязан отчитаться тем же
// релизом, что записан в манифесте, и только затем едет в promote — чужая
// сборка не доезжает до движкового каталога. Архивы без llama-server внутри
// (cudart) — мимо проверки.
bool stage_matches_pin(const std::string& stage, const std::string& release,
                       std::string& err) {
  std::string exe = locate_exe(stage);
  if (exe.empty()) return true;
#ifndef _WIN32
  std::string xerr;
  if (!add_exec_bit(exe, xerr)) {
    err = xerr;
    return false;
  }
#endif
  std::string got = binary_release(exe);
  if (got.empty()) {
    err = std::string("не удалось определить версию ") + SERVER_EXE + " из пакета";
    return false;
  }
  if (got != release) {
    err = "бинарник в пакете — " + got + ", пин манифеста — " + release +
          ": SHA256 сходится, ассет чужой";
    return false;
  }
  return true;
}

} // namespace

Quiesce g_quiesce; // ставит демон; пусто — install без предусловий (не бывает)
void set_quiesce(Quiesce fn) { g_quiesce = std::move(fn); }

// ---- гейт «загрузка/установка не совпадают» ---------------------------------
// Замок короткий и листовой: наружу из него не зовётся ничего, вложенных
// замков нет — порядок с g_mtx/g_apply_mtx супервизора не важен.

namespace {
std::mutex g_gate_mtx;
int g_gate_loads = 0;        // билеты загрузок «в полёте»
bool g_gate_install = false; // билет установки
} // namespace

bool gate_acquire(GateTicket::Kind kind, std::string& err) {
  std::lock_guard l(g_gate_mtx);
  if (kind == GateTicket::Kind::Load) {
    if (g_gate_install) {
      err = "идёт установка движка — загрузка моделей недоступна до её конца";
      return false;
    }
    ++g_gate_loads;
    return true;
  }
  if (g_gate_install) { err = "установка уже идёт"; return false; }
  if (g_gate_loads > 0) {
    err = "идёт загрузка модели — установке нужна тихая система";
    return false;
  }
  g_gate_install = true;
  return true;
}

void gate_release(GateTicket::Kind kind) {
  std::lock_guard l(g_gate_mtx);
  if (kind == GateTicket::Kind::Load)
    --g_gate_loads;
  else
    g_gate_install = false;
}

bool install_in_progress() {
  std::lock_guard l(g_gate_mtx);
  return g_gate_install;
}

GateTicket::~GateTicket() { release(); }

GateTicket& GateTicket::operator=(GateTicket&& o) noexcept {
  if (this != &o) {
    release();
    kind_ = o.kind_;
    held_ = o.held_;
    o.held_ = false;
  }
  return *this;
}

bool GateTicket::acquire(Kind kind, std::string& err) {
  release();
  kind_ = kind;
  held_ = gate_acquire(kind, err);
  return held_;
}

void GateTicket::release() noexcept {
  if (!held_) return;
  held_ = false;
  gate_release(kind_);
}

std::string exe_for(const std::string& build_id) {
  if (build_id.empty()) return {};
  return locate_exe(build_dir(build_id));
}

// Дефолтная сборка по железу: NVIDIA — первая пиннутая cuda-сборка манифеста
// («cuda12» на Linux — 12.8 с родным SASS Blackwell; «cuda13» на Windows;
// CUDA быстрее Vulkan на мелких моделях: замер 01.10 на 4070 Ti, qwen3-0.6b —
// счёт бэкенду), остальным (AMD/Intel/CPU) — Vulkan. Возвращаем только
// пиннутую в manifest: если CUDA для карты недоступна — Vulkan.
std::string default_build() {
  std::string cuda;
  for (auto& b : manifest().builds)
    if (b.id.rfind("cuda", 0) == 0 && !b.files.empty()) {
      cuda = b.id;
      break;
    }
  return (hw::query().nvidia && !cuda.empty()) ? cuda : "vulkan";
}

std::vector<BuildInfo> list() {
  Manifest m = manifest();
  std::vector<BuildInfo> out;
  for (auto& b : m.builds) {
    BuildInfo bi;
    bi.id = b.id;
    bi.pinned = !b.files.empty();
    for (auto& f : b.files) bi.size += f.size;
    bi.exe = exe_for(b.id);
    bi.installed = !bi.exe.empty();
    if (bi.installed) bi.installed_release = installed_release_of(bi.exe);
    out.push_back(bi);
  }
  return out;
}

// Итог каскада: бинарник, как он назван в каталоге/конфиге и похож ли на CUDA.
struct EnginePick {
  std::string exe;   // путь llama-server; «ничего не нашлось» — пуст
  std::string id;    // "custom" | id сборки; вне каталога — пуст
  bool cuda = false; // engine-build объявлен CUDA, пин/имя сборки CUDA или путь содержит "cuda"
};

// Один каскад выбора движка для всех потребителей: раньше он был отрукописан
// трижды (find_server / active_build / launches_cuda) и расползался бы при
// любой правке порядка. Порядок: явный путь → выбранная сборка → по железу
// (NVIDIA → CUDA) → что установлено → рядом с exe и PATH.
EnginePick pick_engine(const config::Config& cfg) {
  auto id_cuda = [](const std::string& id) { return id.rfind("cuda", 0) == 0; };
  if (!cfg.llama_server.empty()) {
    // Относительный путь — от папки программы (portable); $переменные — тоже.
    std::string p = lh::expand_env(config::resolve_path(cfg.llama_server));
    if (lh::file_exists(p)) {
      // Какая сборка у стороннего бинарника — не знаем; сначала явный
      // engine-build (объявленный вид сборки), подстрока пути — фолбэк.
      // Без этого пин на путь без «cuda» в имени (engine/pr30966) читался
      // как Vulkan: планировщик молча брал f16 вместо q8 и вёл себя так,
      // будто профиль формы «cuda» не существует (31.10: плотная 27B с
      // пином 64000 поймала OOM и уехала в ngl 32 — 8 ток/с против 39).
      bool cuda = false;
      if (!cfg.engine_build.empty()) {
        cuda = id_cuda(cfg.engine_build);
      } else {
        std::string low = p;
        for (char& c : low)
          if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
        cuda = low.find("cuda") != std::string::npos;
      }
      return {p, "custom", cuda};
    }
  }
  if (!cfg.engine_build.empty()) {
    std::string p = exe_for(cfg.engine_build);
    if (!p.empty()) return {p, cfg.engine_build, id_cuda(cfg.engine_build)};
  }
  const std::string d = default_build();
  if (std::string p = exe_for(d); !p.empty()) return {p, d, id_cuda(d)};
  for (auto& b : list())
    if (b.installed) return {b.exe, b.id, id_cuda(b.id)};
  return {lh::find_llama_server(""), {}, false}; // рядом с exe и PATH
}

std::string find_server(const config::Config& cfg) { return pick_engine(cfg).exe; }

// Что реально пойдёт в запуск (для показа в UI).
std::string active_build(const config::Config& cfg) { return pick_engine(cfg).id; }

bool launches_cuda(const config::Config& cfg) { return pick_engine(cfg).cuda; }

bool busy() {
  std::lock_guard l(g_state_mtx);
  return g_state.state == "downloading" || g_state.state == "extracting";
}

std::string status_json() {
  // list() при первом касании спавнит --version (кэш по (size, mtime)) —
  // делаем это ДО замка состояния, чтобы не тормозить set_state установщика.
  Manifest m = manifest();
  std::vector<BuildInfo> builds = list();
  config::Config cfg = config::current();
  std::string active = active_build(cfg); // тоже может спавнить --version
  std::lock_guard l(g_state_mtx);
  nlohmann::json j = {{"state", g_state.state},
                      {"build", g_state.build},
                      {"active", active},
                      {"selected", cfg.engine_build},
                      {"release", m.release},
                      {"done", g_state.done},
                      {"total", g_state.total},
                      {"speed", g_state.speed}};
  if (!g_state.error.empty()) j["error"] = g_state.error;
  bool mismatch = false;
  nlohmann::json arr = nlohmann::json::array();
  for (auto& b : builds) {
    if (b.installed && !b.installed_release.empty() && b.installed_release != m.release)
      mismatch = true;
    arr.push_back({{"id", b.id},
                   {"pinned", b.pinned},
                   {"size", b.size},
                   {"installed", b.installed},
                   {"exe", b.exe},
                   {"installed_release", b.installed_release}});
  }
  j["builds"] = arr;
  // Пин и диск могут разойтись молча — наверх ложится и намерение (release),
  // и факт (installed_release); расходятся — mismatch, а не молчаливая ложь.
  j["mismatch"] = mismatch;
  return j.dump(2);
}

bool install(const std::string& build_id, std::string& err) {
  {
    std::lock_guard l(g_state_mtx);
    // Проверка и бронирование — под одним замком: два одновременных POST
    // /v0/engine иначе оба проходили проверку «не занято» и качали в одни и те
    // же _pkg/_new (поймано аудитом: state ставился только после выселения).
    if (g_state.state == "downloading" || g_state.state == "extracting") {
      err = "установка уже идёт";
      return false;
    }
    g_state.state = "downloading";
    g_state.build = build_id;
    g_state.error.clear();
  }
  // Бронь гейта против загрузок — сразу, до всяких проверок: у 202-ответа
  // control не должно быть окна, когда загрузка ещё может проскочить. RAII
  // снимет бронь на любом исходе, включая ранние отказы ниже.
  std::string gate_err;
  GateTicket gate;
  if (!gate.acquire(GateTicket::Kind::Install, gate_err)) {
    set_state({"error", build_id, 0, 0, 0, gate_err});
    err = gate_err;
    return false;
  }
  Manifest m = manifest();
  // Пин из llmhost.conf обязан быть разрешён в манифесте (pin() держит их
  // синхронно): расхождение = ручная правка conf или манифест свежего
  // приложения после апдейта. Качаем лишь то, что сходится с хэшами манифеста.
  if (m.release != pinned_release()) {
    std::string e2 = "пин llmhost.conf (" + pinned_release() +
                     ") не разрешён в manifest.json (" + m.release +
                     ") — выполни: lhctl engine pin " + pinned_release();
    set_state({"error", build_id, 0, 0, 0, e2});
    err = e2;
    return false;
  }
  Manifest::Build* b = nullptr;
  for (auto& x : m.builds)
    if (x.id == build_id) b = &x;
  // Четыре ранних отказа ниже обязаны снять бронь «downloading», иначе busy()
  // залипает до перезапуска демона и блокирует и install, и pin (см. 07.10).
  auto refuse = [&](const std::string& why) {
    set_state({"error", build_id, 0, 0, 0, why});
    err = why;
    return false;
  };
  if (!b)
    return refuse("в manifest.json нет сборки «" + build_id + "»");
  if (b->files.empty())
    return refuse("сборка «" + build_id +
                  "» не пинована в manifest.json (нет url и sha256)");
  // Пустой sha256 в записи манифеста — не «качаем без проверки», а отказ:
  // fail-open позволял подложенному/отредактированному manifest.json вести
  // установку непроверенным байтам. Хост зафиксирован на официальный
  // релизный путь — правка манифеста рядом с бинарником не должна уметь
  // послать скачивание на чужую машину (редирект на objects.githubusercontent.com
  // внутри https — curl -L, схема тоже закрыта в dl::download).
  static const char* const kAssetPrefix =
      "https://github.com/ggml-org/llama.cpp/releases/";
  for (auto& f : b->files) {
    if (f.sha256.size() != 64 ||
        f.sha256.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos)
      return refuse("запись манифеста для «" + build_id +
                    "» без корректного sha256 — переустановите приложение");
    if (f.url.rfind(kAssetPrefix, 0) != 0)
      return refuse("манифест ссылается на не-официальный хост: " + f.url);
  }
  // Поверх живого не ставим и подмену делаем на тихой системе: проверки и
  // глушение (слот свободен, бенч тихий, роутер с детьми остановлены) —
  // обязанность владельца движка, не файлового менеджера (см. engine.h).
  // Новые загрузки гейт уже отсёк бронью установки; хук добирает резидентов,
  // бенч и глушение роутера.
  if (g_quiesce && !g_quiesce(err)) return refuse(err); // бронь «downloading» снять!

  std::string dir = build_dir(build_id);
  lh::ensure_dir(dir + "/_pkg");
  std::string stage = dir + "/_new";
  std::error_code rm_ec;
  fs::remove_all(stage, rm_ec); // хвост прошлой неудачной попытки

  State st;
  st.state = "downloading";
  st.build = build_id;
  for (auto& f : b->files) st.total += f.size;
  set_state(st);
  logf("engine: ставлю «%s» (%llu МБ)", build_id.c_str(),
       (unsigned long long)(st.total >> 20));

  uint64_t done_before = 0;
  for (auto& f : b->files) {
    std::string name = f.url.substr(f.url.find_last_of('/') + 1);
    std::string pkg = dir + "/_pkg/" + name;
    std::string e2;
    if (!lh::dl::download(
              f.url, pkg, f.sha256,
              [&](const lh::dl::Progress& p) {
                std::lock_guard l(g_state_mtx);
                g_state.done = done_before + p.done;
                g_state.total = st.total;
                g_state.speed = p.speed;
                return true;
              },
              e2)) {
      if (e2 == "остановлено") { set_state({"idle"}); return false; }
      set_state({"error", build_id, 0, 0, 0, e2});
      err = e2;
      return false;
    }
    done_before += f.size;

    // Распаковка может занять секунды — не держим мьютекс, чтобы тултип и
    // control API не замерзали. Схема: распаковать в _new, проверить и
    // перенести поверх — сборка не оказывается наполовину заменённой.
    {
      std::lock_guard l2(g_state_mtx);
      g_state.state = "extracting";
    }
    std::string uerr;
#ifdef _WIN32
    bool ok = unpack(pkg, stage, uerr);
#else
    bool ok = unpack_tar(pkg, stage, uerr) && validate_stage(stage, uerr);
#endif
    // Пин обязывает: бинарник стадии отчитывается тем же релизом, что в
    // манифесте, до всякого promote — прежняя сборка при отказе не тронута.
    if (ok) ok = stage_matches_pin(stage, m.release, uerr);
    if (ok) {
      std::string perr;
      ok = promote(stage, dir, perr);
      if (!ok) uerr = perr;
    }
    fs::remove_all(stage, rm_ec);
    if (ok) fs::remove(pkg, rm_ec);
    {
      std::lock_guard l2(g_state_mtx);
      g_state.state = ok ? "downloading" : "error";
      if (!ok) g_state.error = uerr;
    }
    if (!ok) {
      err = uerr;
      return false;
    }
  }

  std::string exe = locate_exe(dir);
  fs::remove(dir + "/_pkg", rm_ec); // пустой кэш архивов можно убрать
  if (exe.empty()) {
    set_state({"error", build_id, 0, 0, 0, std::string("в пакете нет ") + SERVER_EXE});
    err = std::string("в пакете нет ") + SERVER_EXE;
    return false;
  }
#ifndef _WIN32
  // Права исполнения ставим на итоговом пути: tar мог привезти файл без x.
  std::string xerr;
  if (!add_exec_bit(exe, xerr)) {
    set_state({"error", build_id, 0, 0, 0, xerr});
    err = xerr;
    return false;
  }
#endif
  // Путь в конфиг не пишем: find_server() сам находит установленную сборку, а
  // записанный путь навсегда перебивал бы смену engine_build.
  // Профили замерялись на прежнем движке: другой релиз — другие замеры
  // (переустановка того же релиза их сохраняет). Пин НЕ сбрасывает —
  // бинарники ещё прежние, а пин могут откатить.
  profiles::note_engine(m.release);
  set_state({"idle"});
  logf("engine: «%s» готов: %s", build_id.c_str(), exe.c_str());
  return true;
}

// ---- пин релиза llama.cpp (обновление движка) -------------------------------

// Один релиз из GitHub API: tag_name + ассеты. message в ответе — ошибка API
// (rate limit, нет такого тега) — наверх как есть.

// «latest» — свежий b-релиз. У llama.cpp бинары живут в b-релизах, а
// версионирующие (v0.5.0) — указатели с одним nightly-tag.txt внутри, и тот
// бывает просрочен (b11146 при актуальном b11379): GitHub /releases/latest
// для наших целей бесполезен. Берём список последних релизов и выбираем
// первый тег вида bNNNNN.
bool latest_b_tag(std::string& out, std::string& err) {
  std::string raw, e2;
  if (!lh::dl::get_string(
          "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15",
          raw, e2)) {
    err = "GitHub API недоступен: " + e2;
    return false;
  }
  nlohmann::json arr;
  try {
    arr = nlohmann::json::parse(raw);
  } catch (...) {
    err = "GitHub API: ответ не JSON";
    return false;
  }
  if (!arr.is_array()) {
    err = "GitHub API: ожидался список релизов";
    return false;
  }
  for (auto& r : arr) {
    std::string t = r.value("tag_name", std::string());
    bool is_b = t.size() > 1 && t[0] == 'b';
    for (size_t i = 1; i < t.size() && is_b; ++i) is_b = t[i] >= '0' && t[i] <= '9';
    if (is_b) {
      out = t;
      return true;
    }
  }
  err = "GitHub API: среди последних 15 релизов нет ни одного bNNNNN";
  return false;
}

bool fetch_release(const std::string& tag, nlohmann::json& rel, std::string& err) {
  std::string resolved = tag;
  if (resolved.empty() || resolved == "latest") {
    if (!latest_b_tag(resolved, err)) return false;
  }
  // Тег идёт в URL path конкатенацией: «../other/repo/releases/...» читал бы
  // чужой релиз тем же доверенным запросом (path-traversal внутри
  // api.github.com). Форма — ровно та же, что у pinned_release: b+цифры.
  if (resolved.size() < 2 || resolved.size() > 20 || resolved[0] != 'b') {
    err = "тег релиза «" + resolved + "» не похож на bNNNNN";
    return false;
  }
  for (size_t i = 1; i < resolved.size(); ++i)
    if (resolved[i] < '0' || resolved[i] > '9') {
      err = "тег релиза «" + resolved + "» не похож на bNNNNN";
      return false;
    }
  std::string url =
      "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/" + resolved;
  std::string raw, e2;
  if (!lh::dl::get_string(url, raw, e2)) {
    err = "GitHub API недоступен: " + e2;
    return false;
  }
  try {
    rel = nlohmann::json::parse(raw);
  } catch (...) {
    err = "GitHub API: ответ не JSON";
    return false;
  }
  if (rel.contains("message")) {
    err = "GitHub API: " + rel.value("message", std::string("?"));
    return false;
  }
  if (!rel.contains("tag_name") || !rel.contains("assets") || !rel["assets"].is_array()) {
    err = "GitHub API: в ответе нет тега или ассетов";
    return false;
  }
  return true;
}

// Идентификаторы сборок manifest.json — допустимые значения engine-build
// для валидации llmhost.conf.
std::vector<std::string> manifest_builds() {
  std::vector<std::string> out;
  for (auto& b : manifest().builds) out.push_back(b.id);
  return out;
}

// Пин релиза: источник правды — engine-release в llmhost.conf; пусто/битое →
// затравка из manifest.json рядом с бинарями (она обновляется вместе с
// приложением, ручной пин в conf переживает апдейт приложения).
std::string pinned_release() {
  std::string conf_pin = config::current().engine_release;
  std::string baked = manifest().release;
  if (!conf_pin.empty()) {
    bool ok = conf_pin.size() >= 2 && conf_pin[0] == 'b';
    for (size_t i = 1; ok && i < conf_pin.size(); ++i)
      ok = isdigit((unsigned char)conf_pin[i]);
    if (ok) return conf_pin;
    static std::once_flag bad_once;
    std::call_once(bad_once, [&] {
      logf("engine: engine-release «%s» в llmhost.conf не похож на тег bNNNNN — "
           "беру затравку манифеста (%s)",
           conf_pin.c_str(), baked.c_str());
    });
  }
  return baked;
}

std::string latest_release(std::string& err) {
  static std::mutex mtx;
  static std::string cached, cached_err;
  static std::chrono::steady_clock::time_point ok_until, err_until;
  std::lock_guard l(mtx);
  auto now = std::chrono::steady_clock::now();
  if (!cached.empty() && now < ok_until) return cached;
  if (!cached_err.empty() && now < err_until) {
    err = cached_err;
    return "";
  }
  std::string tag;
  if (latest_b_tag(tag, err)) {
    cached = tag;
    ok_until = now + std::chrono::minutes(10);
    return tag;
  }
  cached_err = err;
  err_until = now + std::chrono::minutes(1);
  return "";
}

bool pin(const std::string& tag, PinResult& out, std::string& err) {
  if (busy()) {
    err = "установка уже идёт";
    return false;
  }
  // Два одновременных пина читали и писали манифест вперемешку: .bak ловил
  // уже новый файл, апдейт терялся. Пины редки — мьютекс дешевле атомики.
  static std::mutex g_pin_mtx;
  std::lock_guard pin_l(g_pin_mtx);
  if (busy()) {
    err = "установка уже идёт";
    return false;
  }
  const std::string path = lh::exe_dir() + "/manifest.json";
  const std::string raw = lh::read_small_file(path);
  if (raw.empty()) {
    err = "manifest.json не найден рядом с программой";
    return false;
  }
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(raw);
  } catch (const std::exception& e) {
    err = std::string("manifest не читается: ") + e.what();
    return false;
  }
  const std::string old_tag = j.value("release", "");
  if (old_tag.empty() || !j.contains("builds") || !j["builds"].is_array()) {
    err = "в manifest.json нет release или массива builds";
    return false;
  }

  nlohmann::json rel;
  if (!fetch_release(tag, rel, err)) return false;
  const std::string new_tag = rel.value("tag_name", "");

  // Ассеты по имени: имя нового выводится из текущего пина заменой тега —
  // оно сидит и в имени файла, и в URL (собирается заново). Схему имён
  // апстрим меняет редко, но меняет: нет ассета — честный отказ, ручной
  // пин манифеста остаётся запасным путём. Ассет без digest не пинуем —
  // установка верифицирует SHA256, пинить «на честном слове» нельзя.
  std::map<std::string, std::pair<uint64_t, std::string>> assets;
  for (auto& a : rel["assets"]) {
    std::string dig = a.value("digest", std::string());
    if (dig.rfind("sha256:", 0) != 0) continue;
    assets[a.value("name", std::string())] = {a.value("size", (uint64_t)0), dig.substr(7)};
  }
  int files = 0;
  for (auto& b : j["builds"]) {
    if (!b.is_object() || !b.contains("files") || !b["files"].is_array()) continue;
    for (auto& f : b["files"]) {
      std::string url = f.value("url", std::string());
      size_t slash = url.find_last_of('/');
      std::string name = slash == std::string::npos ? url : url.substr(slash + 1);
      std::string want = lh::replace_all(name, old_tag, new_tag);
      auto it = assets.find(want);
      if (it == assets.end()) {
        err = "в релизе " + new_tag + " нет ассета «" + want +
              "» — обновите manifest.json вручную";
        return false;
      }
      if (it->second.second.empty()) {
        err = "у ассета «" + want + "» нет sha256-digest — обновите manifest.json вручную";
        return false;
      }
      f["url"] = "https://github.com/ggml-org/llama.cpp/releases/download/" +
                 new_tag + "/" + want;
      f["sha256"] = it->second.second;
      f["size"] = it->second.first;
      ++files;
    }
  }
  // Достройка списка сборок из самих ассетов релиза. Подстановка выше
  // обновляет только то, что в манифесте уже есть, — новая вариата апстрима
  // (вторая cuda-ветка: 12.8 и 13.4 живут в релизе одновременно) без неё
  // вечно оставалась бы сборкой-сиротой: «есть в релизе, в манифест не
  // попадёт». Добавляются только новые id (engine_assets.h — linux-паттерны,
  // win-сид достраивает вин-репо своим кодом).
  {
    AddedBuilds ab = builds_from_assets(assets, j["builds"], new_tag);
    if (!ab.added.empty()) {
      j["builds"] = ab.builds;
      out.added = ab.added;
      logf("engine: из релиза %s достроены сборки: %s", new_tag.c_str(),
           ab.added.c_str());
    }
  }
  j["release"] = new_tag;
  // Гейт валидации — после достройки: engine-build может указывать на
  // сборку-сироту, которой в сиде не было (это и есть повод для пина), а
  // достройка добавляет только новые id. Проверяем по тому, чем манифест
  // СТАНЕТ, иначе пин, лечащий сироту, ею же и блокируется навсегда. Пин
  // пишет engine-release в llmhost.conf, а юзер мог покорёжить его руками:
  // на битом конфиге движок не обновляем — невалидируемые значения (ctx,
  // engine-build…) молча поедут в детей.
  {
    std::vector<std::string> ids;
    for (auto& b : j["builds"])
      if (b.is_object() && b.contains("id") && b["id"].is_string())
        ids.push_back(b["id"].get<std::string>());
    std::vector<std::string> errs = config::validate_conf(ids);
    if (!errs.empty()) {
      err = "llmhost.conf невалиден — пин не выполняю, исправь:";
      for (auto& e : errs) err += "\n  " + e;
      return false;
    }
  }
  // Бэкап до первой записи: откат пина — положить .bak обратно или
  // перепинить на старый тег тем же путём.
  if (!lh::write_small_file(path + ".bak", raw)) {
    err = "не удалось сохранить manifest.json.bak — манифест не меняю";
    return false;
  }
  const std::string tmp = path + ".tmp";
  if (!lh::write_small_file(tmp, j.dump(2) + "\n")) {
    err = "не удалось записать " + tmp;
    return false;
  }
  std::error_code ec;
  fs::rename(tmp, path, ec);
  if (ec) {
    err = "замена манифеста: " + ec.message();
    return false;
  }
  // Источник правды — llmhost.conf; манифест выше — кэш разрешённых url/хэшей.
  {
    config::Config cfg = config::current();
    cfg.engine_release = new_tag;
    config::set_current(cfg);
    config::save(cfg);
  }
  out.release = new_tag;
  out.files = files;
  out.path = path;
  logf("engine: пин %s → %s (%d файлов): manifest.json + llmhost.conf engine-release",
       old_tag.c_str(), new_tag.c_str(), files);
  return true;
}

} // namespace lh::engine
