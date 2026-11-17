#include "mtp.h"

#include "catalog.h"
#include "config.h"
#include "hw.h"
#include "modelreg.h"
#include "pull.h"
#include "supervisor.h"
#include "util.h"

#include <nlohmann/json.hpp>

#include <cctype>
#include <thread>

using nlohmann::json;

namespace lh::mtp {

namespace {

// Запас поверх весов модуля: KV-кэш черновика и рабочие буферы ребёнка.
constexpr uint64_t kDraftOverhead = 512ull << 20;

std::string stem_of(std::string p) {
  size_t s = p.find_last_of("/\\");
  if (s != std::string::npos) p = p.substr(s + 1);
  size_t d = p.find_last_of('.');
  if (d != std::string::npos) p = p.substr(0, d);
  return p;
}

// "<models_dir>/org/repo/..." → "org/repo"; пусто — путь не внутри папки моделей.
std::string repo_of(const std::string& models_dir, const std::string& path) {
  if (models_dir.empty() || path.compare(0, models_dir.size(), models_dir) != 0) return {};
  std::string rel = path.substr(models_dir.size());
  while (!rel.empty() && rel[0] == '/') rel = rel.substr(1);
  size_t a = rel.find('/');
  if (a == std::string::npos) return {};
  size_t b = rel.find('/', a + 1);
  if (b == std::string::npos) return {};
  return rel.substr(0, a) + "/" + rel.substr(a + 1, b - a - 1);
}

// Путь файла внутри репозитория (модуль лежит в подпапке MTP): диск → "MTP/…".
std::string file_in_repo(const std::string& repo_dir, const std::string& path) {
  if (path.compare(0, repo_dir.size(), repo_dir) != 0) return {};
  std::string rel = path.substr(repo_dir.size());
  while (!rel.empty() && rel[0] == '/') rel = rel.substr(1);
  return rel;
}

} // namespace

bool find_module(const std::string& main_path, Module& out, std::string& err) {
  if (lh::modelreg::is_mtp_module(main_path)) return false; // сам модуль — не кандидат
  config::Config cfg = config::current();
  std::string models_dir = config::models_dir_resolved(cfg);
  std::string repo_dir = main_path.substr(0, main_path.find_last_of("/\\"));

  // 1) Уже скачан в тот же репозиторий (обычно в подпапке MTP/).
  std::string fdir = repo_dir + "/";
  for (auto& m : lh::modelreg::scan(models_dir)) {
    if (!lh::modelreg::is_mtp_module(m.path)) continue;
    if (m.path.compare(0, fdir.size(), fdir) != 0) continue;
    out.path = m.path;
    out.repo = repo_of(models_dir, m.path);
    out.file = file_in_repo(repo_dir, m.path);
    out.stem = stem_of(m.path);
    out.size = m.size;
    out.downloaded = true;
    return true;
  }

  // 2) Есть в каталоге, но не скачан: сетевой запрос с кэшем (10 мин);
  //    оффлайн или приватный репозиторий — просто не предлагаем.
  std::string repo = repo_of(models_dir, main_path);
  if (repo.empty()) return false;
  std::string dj = lh::catalog::detail_json(repo, err);
  if (dj.empty()) return false;
  try {
    auto d = json::parse(dj);
    for (auto& f : d.value("files", json::array())) {
      if (f.value("kind", "") != "mtp") continue;
      std::string file = f.value("path", "");
      if (file.empty()) continue;
      std::string local = models_dir + "/" + repo + "/" + file;
      out.path = local;
      out.repo = repo;
      out.file = file;
      out.stem = stem_of(file);
      out.size = f.value("size", (uint64_t)0);
      out.downloaded = lh::file_exists(local);
      return true;
    }
  } catch (...) {
    err = "ответ каталога не читается";
  }
  return false;
}

void after_load(const std::string& main_path) {
  std::string id = sup::id_for_path(main_path);
  if (id.empty()) return;
  if (sup::has_draft_override(id)) return; // черновик уже настроен (MTP или ручной)
  config::State st = config::load_state();
  if (st.mtp_choices.count(id)) return;    // уже решали — не решаем дважды

  // Диалога нет (headless) — политика из конфига: off — никогда, on — всегда,
  // auto — по оценке планировщика (влезет ли модуль).
  const std::string& mode = config::current().mtp_mode;

  Module mod;
  std::string err;
  if (!find_module(main_path, mod, err)) {
    if (!err.empty())
      logf("mtp: каталог не ответил для %s (%s) — предложение пропущено", id.c_str(),
           err.c_str());
    return;
  }

  // Влезает ли модуль рядом с моделью. Места сейчас может не хватать из-за
  // других резидентов. Исключение — плотная частичная (CUDA): планировщик сам
  // подвинет слои, модуль у него в бюджете, так что «свободно мало» её не
  // останавливает.
  lh::hw::Gpu gpu = lh::hw::query();
  uint64_t need = mod.size + kDraftOverhead;
  bool fits = !gpu.vram_total || gpu.vram_free >= need;
  if (!fits) fits = sup::dense_partial_fits(main_path, mod.size);
  if (mode == "off") {
    st.mtp_choices[id] = false;
    config::save_state(st);
    logf("mtp: %s: режим off — модуль «%s» не подключаю", id.c_str(), mod.stem.c_str());
    return;
  }
  if (mode == "auto" && !fits) {
    st.mtp_choices[id] = false;
    config::save_state(st);
    logf("mtp: %s: режим auto — модуль «%s» не влезает: свободно %llu МиБ, нужно %llu МиБ",
         id.c_str(), mod.stem.c_str(), (unsigned long long)(gpu.vram_free >> 20),
         (unsigned long long)(need >> 20));
    return;
  }
  // mode == "on" или ("auto" и влезает).
  logf("mtp: %s: режим %s — включаю модуль «%s» (%llu МиБ%s)", id.c_str(), mode.c_str(),
       mod.stem.c_str(), (unsigned long long)(mod.size >> 20),
       mod.downloaded ? ", уже скачан" : "");

  if (!mod.downloaded) {
    logf("mtp: скачиваю MTP-модуль — прогресс: lhctl pull");
    std::string e2;
    if (!lh::pull::start(mod.repo, mod.file, e2)) {
      logf("mtp: скачивание MTP-модуля не началось: %s", e2.c_str());
      return;
    }
    while (lh::pull::busy()) std::this_thread::sleep_for(std::chrono::milliseconds(500));
    auto st = json::parse(lh::pull::status_json(), nullptr, false);
    if (st.is_null() || st.value("state", "") != "done") {
      logf("mtp: MTP-модуль не скачался: %s",
           st.is_null() ? "нет ответа" : st.value("error", "ошибка").c_str());
      return;
    }
  }

  logf("mtp: включаю MTP: модель перезагрузится");
  std::string e3;
  if (sup::enable_mtp(id, mod.stem, main_path, e3)) {
    logf("mtp: MTP включён (спекулятивное декодирование, draft-mtp)");
  } else {
    logf("mtp: MTP включить не удалось: %s", e3.c_str());
  }
}

} // namespace lh::mtp
