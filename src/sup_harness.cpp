#include "supervisor.h"
#include "i18n.h"
#include "sup_internal.h"

#include "engine.h"
#include "fit.h"
#include "gguf.h"
#include "hw.h"
#include "metrics.h"
#include "modelreg.h"
#include "plan.h"
#include "util.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <filesystem>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <strings.h>
#include <utility>
#include <vector>

using lh::i18n::tr;

namespace fs = std::filesystem;

namespace lh::sup::detail {

using lh::proc::Child;
using profiles::ModelProfile;

// ---- harness-режим: по llama-server на каждую модель, загруженную в VRAM ----
// Оркестрирует демон: сам спавнит ребёнка на модель (classic-путь спавна с
// умным планом и лестницей), сам ведёт карту «модель → порт», сам роутит /v1
// по полю model и сам решает, кого выселять. Инвариант один и главный:
// выселению подлежат только НЕзанятые слоты — чужой стрим неубиваем.
// Запрос к модели, которой мешают занятые резиденты, паркуется в очереди
// пробуждения и доезжает, когда занятость блокера упала в ноль и слот
// выселен; TTL истёк — честный отказ с причиной.

std::map<std::string, HSlot> g_slots; // id → слот (под g_mtx)

// Пересекаются ли карты слота и цели раскладки. Пустое множество слота
// (спавн без телеметрии — движок делит карты по умолчанию, трогает все) —
// «пересекается со всем»: выселение консервативно считает такой слот соседом.
bool cards_overlap(const std::set<int>& slot_cards, const std::set<int>& target) {
  if (slot_cards.empty() || target.empty()) return true;
  for (int c : slot_cards)
    if (target.count(c)) return true;
  return false;
}

// harness: перечисление id через разделитель — для сообщений об ошибках.
std::string join_ids(const std::vector<std::string>& v, const char* sep) {
  std::string out;
  for (auto& s : v) {
    if (!out.empty()) out += sep;
    out += s;
  }
  return out;
}

// id модели для внешнего мира (реестр id → путь): ASCII-подмножество имени
// файла с уникализацией «-2». Единственное именование моделей наружу: /v1/models,
// секции переопределений, карточки клиентов.
std::string make_unique_id(const std::string& base,
                           std::map<std::string, std::string>& by_id) {
  std::string id;
  id.reserve(base.size());
  // id попадает в заголовки секций переопределений ([id] в llmhost.conf) и в
  // JSON-ответы: имя файла на NTFS может содержать «]», «[» и даже переводы
  // строк — ломали бы разбор секций. Оставляем безопасный ASCII, остальное —
  // в «_».
  for (char c : base)
    id += (isalnum((unsigned char)c) || c == '.' || c == '-' || c == '_') ? c : '_';
  if (id.empty()) id = "model";
  for (int n = 2; by_id.count(id); ++n) id = id + "-" + std::to_string(n);
  return id;
}


// Архитектуры-эмбеддеры: bert-семейство покрывает nomic/bge/e5/gte.
// Им включается --embedding, чат их не выбирает.
bool is_embedding_arch(const std::string& arch) {
  static const char* const k[] = {"bert",       "nomic-bert",    "jina-bert-v2",
                                  "jina-bert-v3", "qwen2-embed", "gemma-embedding"};
  for (auto* a : k)
    if (arch == a) return true;
  return false;
}

// Vision-проекция рядом с моделью — прикладывается сама (b11223: mmproj-auto
// подхватывает файл только рядом с -hf; локальным моделям ключ прикладываем
// сами, спавн пробрасывает её ребёнку как --mmproj — проверено
// живьём 01.10: gemma-4-12b описала картинку).
// Родство (29.10, урок «чужой mmproj»): проекция с сигнатурным ядром имени
// (mmproj_core_of) вяжется к стволу только по mmproj_matches_trunk. Прежний
// поиск «mmproj*.gguf» молча прицеплял первую попавшуюся: gemma-4-26B утащила
// qwen36-проекцию и прожгла лестницу пятью OOM-попытками подряд, а рукименные
// «ствол-mmproj-формат» не находились вовсе — vision молча не работал. Легаси-
// договор pull сохранён: файл с «mmproj» в начале имени и без сигнатуры —
// проекция репозитория (родство даёт каталог), из нескольких — F16/BF16,
// иначе наименьший. Отказы не молчат: diag объясняет, вызывающий на load-пути
// уносит её в лог.
std::string find_mmproj(const std::string& model_path, std::string* diag) {
  auto say = [diag](std::string s) {
    if (diag) *diag = std::move(s);
  };
  std::string dir = model_path.substr(0, model_path.find_last_of("/\\"));
  if (dir.empty()) return {};
  struct Cand {
    std::string core;
    bool f16 = false;
    uint64_t size = 0;
    std::string path;
  };
  std::vector<Cand> kin, legacy;
  std::vector<std::string> alien; // сигнатура есть, к этому стволу не вяжется
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(dir, ec)) {
    if (ec) break;
    if (e.is_directory(ec)) continue;
    const std::string name = e.path().filename().string();
    std::string low = name;
    for (char& c : low) c = (char)std::tolower((unsigned char)c);
    if (low.find("mmproj") == std::string::npos) continue;
    Cand c;
    c.path = e.path().string();
    c.size = lh::file_size(c.path);
    size_t n = strlen("F16.gguf");
    c.f16 = name.size() >= n &&
            strcasecmp(name.c_str() + name.size() - n, "F16.gguf") == 0;
    c.core = lh::modelreg::mmproj_core_of(c.path);
    if (c.core.size() >= 4) {
      if (!lh::modelreg::mmproj_matches_trunk(model_path, c.path)) {
        alien.push_back(c.core + " («" + name + "»)");
        continue;
      }
      kin.push_back(std::move(c));
    } else {
      legacy.push_back(std::move(c));
    }
  }
  auto pick = [](std::vector<Cand>& v) {
    for (auto& c : v)
      if (c.f16) return c.path;
    return std::min_element(v.begin(), v.end(),
                            [](const Cand& a, const Cand& b) {
                              return a.size < b.size;
                            })
        ->path;
  };
  if (!kin.empty()) {
    size_t best = 0;
    for (auto& k : kin) best = std::max(best, k.core.size());
    std::string best_core;
    std::vector<Cand> finals;
    for (auto& k : kin) {
      if (k.core.size() != best) continue;
      if (best_core.empty()) best_core = k.core;
      if (k.core != best_core) {
        say("несколько проекций вяжутся к стволу одинаково (ядро «" +
            best_core + "» и «" + k.core + "») — не прицепляю ни одну, " +
            "переименуйте по стволу однозначно");
        return {};
      }
      finals.push_back(std::move(k));
    }
    return pick(finals);
  }
  if (!legacy.empty()) return pick(legacy);
  if (!alien.empty()) {
    std::string list = alien.front();
    for (size_t i = 1; i < alien.size(); ++i) list += ", " + alien[i];
    say("рядом есть проекция(и) " + list +
        " — ни одна не вяжется к имени ствола, vision не прицеплен "
        "(переименуйте: семья-mmproj-формат.gguf)");
  }
  return {};
}

// Проекция для загрузки с учётом пер-модельного vision-mode: off — ребёнок
// едет без --mmproj, что бы ни лежало рядом (diag объяснит); auto — родство.
// Один гейт на все load-пути.
std::string mmproj_for_load(const std::string& model_path, const ModelOverrides* o,
                            std::string* diag) {
  if (o && o->vision_mode == "off") {
    if (diag) *diag = "vision отключён переопределением (vision-mode = off)";
    return {};
  }
  return find_mmproj(model_path, diag);
}

// «Последняя использованная модель» (state.json): запоминается при явной
// загрузке и на каждом обслуженном запросе — при следующем старте демона
// грузится сразу, чтобы lhctl chat был готов без ожидания. На диск пишем
// только при смене модели: запросы к той же — не трогают файл. Рантайм-состояние
// живёт в state.json, не в llmhost.conf: писать его в конфиг значило бы
// перезаписывать настройки на каждом ходе чата.
void note_model_used(const std::string& path) {
  if (path.empty()) return;
  config::State st = config::load_state();
  if (!st.last_model.empty() && path_eq(config::resolve_path(st.last_model), path)) return;
  st.last_model = path;
  config::save_state(st);
}

// Целевая карта модели. При 2+ картах модели пинятся по очереди, но
// НЕ вслепую: карты сортируются по скорости (cc, затем VRAM — тягловая
// эвристика «новее и больше — быстрее», без замера), модели — по размеру
// файла, и ранг модели выбирает карту из этого порядка. Крупные модели
// получают быстрые карты первыми, мелкие — оставшиеся; на гетерогенном
// пуле мелочь больше не молчит на медленной карте (06.10: 0.6B на 3060
// — 268 ток/с, на Blackwell — 459; бенч-матрица vs Ollama). mmproj и
// MTP-модули в ранжировании не участвуют — они едут с родителем, и их
// наличие больше не сдвигает пины остальных (радар «пин плавает между
// рестартами»). Детерминировано: сортировка стабильна по пути.
// Ручная привязка — main-gpu/tensor-split в llmhost.conf. ПИН —
// ПРЕДПОЧТЕНИЕ, а не приговор: режим исполнения выбирает алгоритм по лестнице
// «пин целиком → другая карта целиком → сплит по пулу → гибрид: слои по
// всему пулу + хвост в RAM (плотная крупнее пула, L11) → RAM-ярус на пине
// (MoE)».
// Целиком = max_ctx > 0 при kv_q8, по «пустая карта − резерв плана». Сплит по
// пулу и другая карта предпочтительнее RAM-компромиссов (exps=CPU, частичный
// оффлад) для плотных И MoE: все веса на GPU несравнимо быстрее любого
// оффлоада в RAM. Явные пропорции tensor-split — юзер решил раскладку сам,
// пишем как есть. Без пина (выключен) — сплит по пулу с долями по памяти карт:
// делёж движка «поровну» меньшую карту теребит зря и ловит OOM.
GpuTarget gpu_target_for(const std::string& id, const std::string& path,
                         const config::Config& cfg,
                         const std::map<std::string, ModelOverrides>& ov,
                         bool live) {
  GpuTarget t;
  auto cards = hw::query_all();
  const int ngpu = (int)cards.size();
  if (!cards.empty()) t.card = cards[0];
  if (ngpu < 2) return t;
  std::vector<int> speed_order(ngpu);
  for (int i = 0; i < ngpu; ++i) speed_order[i] = i;
  std::sort(speed_order.begin(), speed_order.end(), [&](int a, int b) {
    const hw::Gpu& A = cards[a];
    const hw::Gpu& B = cards[b];
    const std::pair<int, int> ca = {A.cc_major, A.cc_minor};
    const std::pair<int, int> cb = {B.cc_major, B.cc_minor};
    if (ca != cb) return ca > cb;
    if (A.vram_total != B.vram_total) return A.vram_total > B.vram_total;
    return a < b;
  });
  {
    static std::mutex order_mtx;
    static std::string order_last;
    std::string order_desc;
    for (int c : speed_order) {
      if (!order_desc.empty()) order_desc += ", ";
      order_desc += "#" + std::to_string(c) + " " + cards[c].name;
    }
    std::lock_guard l(order_mtx);
    if (order_last != order_desc) {
      order_last = order_desc;
      logf("harness: пин-порядок карт по скорости: %s", order_desc.c_str());
    }
  }

  const bool cuda = engine::launches_cuda(cfg);
  const hw::Build bld = cuda ? hw::Build::Cuda13 : hw::Build::Vulkan;

  // Сплит по пулу: main-gpu — карта с максимумом памяти, -ts — бюджеты
  // (total минус оверхед) в МиБ. Доли — по TOTAL, не по текущей свободе:
  // план (fits_whole/dense_split) считается против «пустого пула», и сплит,
  // написанный при занятых картах, после выселения residents не совпадал бы
  // с планом — ребёнок ловил OOM на карте с заниженной долей; заодно
  // раскладка перестаёт плавать между рестартами (урок «пин плавает»).
  // Лог — только когда раскладка модели изменилась: gpu_target_for зовётся
  // на каждый /v0/models (model_gpu), и без гроссбуха плотные модели
  // спамили журнал каждые две секунды.
  auto pool_target = [&](const char* why) {
    hw::Gpu p = fit::pool(cards, bld);
    if (p.vram_total <= 0) return false;
    int main_idx = 0;
    for (int c = 1; c < ngpu; ++c)
      if (cards[c].vram_total > cards[main_idx].vram_total) main_idx = c;
    const uint64_t ovh = fit::fixed_overhead(cuda);
    std::string ts;
    for (int c = 0; c < ngpu; ++c) {
      if (c) ts += ",";
      const uint64_t budget =
          cards[c].vram_total > ovh ? cards[c].vram_total - ovh : 1;
      ts += std::to_string(budget >> 20);
    }
    t.card = p;
    t.split = true;
    t.main_gpu = main_idx;
    t.lines = "main-gpu = " + std::to_string(main_idx) + "\n" + "tensor-split = " + ts + "\n";
    if (why) {
      static std::mutex log_mtx;
      static std::map<std::string, std::string> last;
      std::lock_guard l(log_mtx);
      std::string key = id + "|" + t.lines;
      if (last[id] != key) {
        last[id] = key;
        logf("harness: %s: %s — сплит по пулу (%d карт), main-gpu %d, -ts %s", id.c_str(),
             why, ngpu, main_idx, ts.c_str());
      }
    }
    return true;
  };

  // Гроссбух смены раскладки: gpu_target_for зовётся на каждый /v0/models —
  // перестановку модели между картами логируем один раз, не на каждый вызов.
  auto note = [&](const std::string& what) {
    static std::mutex log_mtx;
    static std::map<std::string, std::string> last;
    std::lock_guard l(log_mtx);
    if (last[id] != what) {
      last[id] = what;
      logf("harness: %s: %s", id.c_str(), what.c_str());
    }
  };

  int main_gpu = -1;
  bool manual_pin = false;
  std::string ts;
  auto o = overrides_for(ov, id, path);
  if (o) {
    main_gpu = o->main_gpu;
    ts = o->tensor_split;
    manual_pin = main_gpu >= 0;
  }
  if (main_gpu < 0 && cfg.gpu_pin) {
    // Дом — самая быстрая карта, куда файл влезает с резервом.
    // Круг по рангу размера сажал мелкую модель на медленную карту,
    // хотя быстрая была пуста (0.6B: 268 ток/с на 3060 против 459 на Blackwell).
    // Шард-группа — вся группа, не первая часть, которую открывает движок.
    const uint64_t sz = lh::modelreg::shard_group_bytes(path);
    for (int c : speed_order) {
      if ((int64_t)cards[c].vram_total - (int64_t)kPlanReserve >= (int64_t)sz) {
        main_gpu = c;
        break;
      }
    }
    if (main_gpu < 0 && !speed_order.empty()) main_gpu = speed_order[0];
  }
  if (main_gpu < 0 || main_gpu >= ngpu) {
    pool_target(nullptr);
    return t;
  }

  // Явные пропорции сплита — юзер решил раскладку сам: пишем как есть.
  if (!ts.empty()) {
    t.card = cards[main_gpu];
    t.main_gpu = main_gpu;
    t.lines = "main-gpu = " + std::to_string(main_gpu) + "\n" +
              "tensor-split = " + ts + "\n";
    return t;
  }

  // Пин — предпочтение. Мета нужна для теста «влезает ли целиком»: берём из
  // реестра (describe кэширует разбор по пути), иначе читаем заголовок сами.
  // Шард-группа — одна модель: в веса входит сумма частей, не только
  // первая, которую открывает движок.
  const uint64_t fsize = lh::modelreg::shard_group_bytes(path);
  gguf::Meta meta;
  bool has_meta = false;
  {
    auto list = lh::modelreg::scan(config::models_dir_resolved(cfg));
    for (auto& m : list) {
      if (!path_eq(m.path, path)) continue;
      meta = m.meta;
      has_meta = meta.n_layers > 0;
      break;
    }
  }
  if (!has_meta) {
    has_meta = lh::modelreg::model_meta(path, meta) && meta.n_layers > 0;
  }
  if (has_meta && fsize > 0) {
    const uint32_t cap = meta.n_ctx_train ? (uint32_t)meta.n_ctx_train : 8192u;
    auto fits_whole = [&](const hw::Gpu& c) {
      hw::Gpu g = c;
      g.build = bld;
      if (g.vram_total > 0) g.vram_free = g.vram_total - kPlanReserve; // строго
      return fit::max_ctx(meta, fsize, g, cap, true) > 0;
    };
    auto set_pin = [&](int mg) {
      std::string all;
      for (int c = 0; c < ngpu; ++c) {
        if (c) all += ",";
        all += (c == mg) ? "1" : "0";
      }
      t.card = cards[mg];
      t.main_gpu = mg;
      t.lines = "main-gpu = " + std::to_string(mg) + "\n" +
                "tensor-split = " + all + "\n";
    };
    if (fits_whole(cards[main_gpu])) {
      // При загрузке занятая быстрая карта уступает следующей, куда модель
      // влезает целиком и где сейчас хватает свободной памяти.
      // /v0/models (live=false) дом не двигает.
      if (live && !manual_pin &&
          cards[main_gpu].vram_free < fsize + kPlanReserve) {
        int alt = -1;
        for (int c : speed_order) {
          if (c == main_gpu || !fits_whole(cards[c])) continue;
          if (cards[c].vram_free < fsize + kPlanReserve) continue;
          alt = c;
          break;
        }
        if (alt >= 0) {
          char buf[200];
          snprintf(buf, sizeof(buf),
                   "карта %d занята (%llu МиБ свободно) — сажаю на карту %d",
                   main_gpu, (unsigned long long)(cards[main_gpu].vram_free >> 20), alt);
          note(buf);
          set_pin(alt);
          return t;
        }
      }
      set_pin(main_gpu);
      return t;
    }
    // Пин не вмещает целиком — ищем карту, которая вмещает (самая ёмкая):
    // целиком на одной карте быстрее любого сплита и любого RAM-компромисса.
    // Юзерский пин тут тоже уступает — с громким журналом.
    int best = -1;
    for (int c = 0; c < ngpu; ++c) {
      if (c == main_gpu || !fits_whole(cards[c])) continue;
      if (best < 0 || cards[c].vram_total > cards[best].vram_total) best = c;
    }
    if (best >= 0) {
      char buf[160];
      snprintf(buf, sizeof(buf),
               "пин карты %d не вмещает модель целиком — целиком на карту %d",
               main_gpu, best);
      note(buf);
      set_pin(best);
      return t;
    }
    // Ни одна карта не вмещает целиком — сплит по пулу, если пул вмещает:
    // все веса на GPU (плотная И MoE) лучше exps=CPU/частичного оффлоада.
    hw::Gpu p = fit::pool(cards, bld);
    if (p.vram_total > 0) {
      p.vram_free = p.vram_total;
      if (fit::max_ctx(meta, fsize, p, cap, true) > 0) {
        if (pool_target("не влезает целиком ни в одну карту — все веса на GPU"))
          return t;
      }
    }
    // Пул тоже не вмещает целиком — плотная едет ГИБРИДОМ (L11): слои
    // распределяются по ВСЕМУ пулу пропорционально памяти карт, хвост —
    // в RAM. На одной пиннутой карте слоёв осталось бы в разы меньше
    // (модель 39,6 ГиБ на пуле 12+24: ~34,5 ГиБ слоёв на двух картах
    // против ~10 на одной). MoE не трогаем: у него свой ярус (эксперты
    // в RAM), не-экспертная часть обычно влезает в пиннутую карту.
    if (meta.n_experts == 0 &&
        pool_target("не влезает целиком даже в пул — слои по всем картам, хвост в RAM"))
      return t;
  }

  if (ts.empty()) {
    for (int c = 0; c < ngpu; ++c) {
      ts += c ? "," : "";
      ts += (c == main_gpu) ? "1" : "0";
    }
  }
  t.card = cards[main_gpu];
  t.main_gpu = main_gpu;
  t.lines = "main-gpu = " + std::to_string(main_gpu) + "\n" +
            "tensor-split = " + ts + "\n";
  return t;
}

// Очередь пробуждения. Глубина из конфига проверяется на входе (queue-depth —
// горячий ключ), потолок самого Lot — просто страховка от бессмысленно
// большого числа.
park::Lot g_wake_lot(4096);

// Резидентория (harness) в state.json: список Loaded-слотов на момент записи.
// Читается init'ом при старте демона. Пишем только при изменении состава.
void save_resident_state() {
  config::State st = config::load_state();
  std::vector<std::string> now;
  {
    std::lock_guard l(g_mtx);
    for (auto& [id, s] : g_slots)
      if (s.st == St::Loaded) now.push_back(config::store_path(s.path));
  }
  std::sort(now.begin(), now.end());
  if (now == st.resident) return;
  st.resident = std::move(now);
  config::save_state(st);
}

void wake_notify() { g_wake_lot.notify_all(); }

// Карта реестра harness: id → путь (те же id видят клиенты в /v1/models и в
// секциях переопределений), плюс эмбеддеры и system-промпты.
std::map<std::string, std::string> harness_registry_maps(
    std::set<std::string>& emb_out, std::map<std::string, std::string>& sys_out) {
  config::Config cfg = config::current();
  auto list = lh::modelreg::scan(config::models_dir_resolved(cfg));
  auto ov = load_overrides();
  std::map<std::string, std::string> by_id;
  for (auto& m : list) {
    if (!m.meta.n_layers) continue; // не GGUF
    if (lh::modelreg::is_mtp_module(m.path) || lh::modelreg::is_mmproj_module(m.path))
      continue;
    if (lh::modelreg::is_shard_continuation(m.path)) continue; // часть 2..N — та же модель
    if (!lh::safe_local_filename(m.path)) continue;
    // У шард-группы id — имя модели, не «…-00001-of-00004». Путь остаётся
    // первой частью: движок открывает её и находит остальные по имени.
    std::string base = lh::modelreg::registry_stem(m.path);
    std::string id = make_unique_id(base, by_id);
    by_id[id] = m.path;
    if (embedding_arch(m.arch)) emb_out.insert(id);
    const ModelOverrides* o = overrides_for(ov, id, m.path);
    if (o && !o->system.empty()) sys_out[id] = o->system;
  }
  return by_id;
}

// GpuTarget (решение gpu_target_for, формат пресета) → TryTarget (флаги
// спавна). main-gpu/tensor-split читаем из lines — там они в финальном виде.
// Пин одной карты: показываем ребёнку ТОЛЬКО её (инициализация контекстов на
// соседних картах — та рулетка спавна, что уже ловили живьём); сплит: все
// карты списком с бюджетами.
TryTarget try_target_of(const GpuTarget& gt) {
  TryTarget t;
  t.split = gt.split;
  t.card = gt.card;
  t.main_gpu = gt.main_gpu;
  if (gt.lines.empty()) return t; // без пина: раскладку ведёт движок
  std::string mg, ts;
  for (size_t pos = 0; pos < gt.lines.size();) {
    size_t eol = gt.lines.find('\n', pos);
    std::string line = gt.lines.substr(pos, eol == std::string::npos
                                                ? std::string::npos
                                                : eol - pos);
    pos = eol == std::string::npos ? gt.lines.size() + 1 : eol + 1;
    size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    std::string key = line.substr(0, eq);
    while (!key.empty() && (key.back() == ' ' || key.back() == '\t')) key.pop_back();
    std::string val = line.substr(eq + 1);
    size_t b = val.find_first_not_of(" \t");
    val = b == std::string::npos ? std::string() : val.substr(b);
    if (key == "main-gpu") mg = val;
    else if (key == "tensor-split") ts = val;
  }
  if (gt.split) {
    int n = 1;
    for (char c : ts)
      if (c == ',') ++n;
    t.ts = ts;
    for (int i = 0; i < n; ++i) {
      if (i) t.devices += ",";
      t.devices += "CUDA" + std::to_string(i);
    }
  } else if (!mg.empty()) {
    t.devices = "CUDA" + mg; // только пиннутая карта
  }
  return t;
}

// Живые состояния всех известных моделей (слоты поверх реестра; словарь
// состояний — роутерный: unloaded/loading/loaded + failed).
std::map<std::string, MState> harness_states() {
  std::map<std::string, MState> out;
  std::lock_guard l(g_mtx);
  for (auto& [id, p] : g_id_map) out[id] = MState{"unloaded", false, {}};
  for (auto& [id, s] : g_slots) {
    MState m;
    switch (s.st) {
      case St::Loading: m.state = "loading"; break;
      case St::Loaded:
        m.state = "loaded";
        m.note = s.note; // нота — про живую форму спавна; спящей не место
        break;
      case St::Error: m.state = "unloaded"; m.failed = true; break;
      default: m.state = "unloaded"; break; // Stopping и прочие — тоже «не готов»
    }
    out[id] = m;
  }
  return out;
}

// Остановить один слот (harness). to_asleep — помнить модель, будилка поднимет.
// persist=false — не обновлять резиденторию в state.json (shutdown: список
// должен остаться тем, что было загружено, иначе рестарт ничего не вернёт).
// force=true — остановить даже занятый слот (только shutdown: дети обязаны
// умереть, mmap-файлы и карты памяти за ними).
// false — слот занят генерацией, останов отменён (главный инвариант режима);
// проверка занятости атомарна с переводом слота в Stopping: запрос, начавшийся
// после снимка занятости вызывающим, уже не получит порт и не будет убит
// посреди стрима (ребёнок останавливается ВНЕ g_mtx — stop() ждёт SIGTERM до
// 5 с и SIGKILL, под замком это вставало бы на harness_states и маршрутизацию).
bool stop_hslot(const std::string& id, bool to_asleep, bool persist, bool force) {
  std::shared_ptr<Child> child;
  {
    // Порядок документированный: busy, затем g_mtx (см. sup_watch).
    std::lock_guard lb(g_busy_mtx);
    const bool busy = g_busy_set.count(id) > 0;
    std::lock_guard l(g_mtx);
    auto it = g_slots.find(id);
    if (it == g_slots.end()) return true; // слота нет — делать нечего
    if (busy && !force) return false;
    child = it->second.child;
    if (child) {
      logf("harness: останавливаю «%s»", id.c_str());
      // Слот в промежуточное состояние ДО отпуска замка: маршрутизация видит
      // «не готов» сразу, вторая будилка сюда не заходит.
      it->second.st = St::Stopping;
      it->second.child.reset();
      it->second.port = 0;
    } else {
      it->second.st = to_asleep && !it->second.path.empty() ? St::Asleep : St::Unloaded;
    }
  }
  if (child) {
    g_last_kill_ms = steady_ms(); // будившие модель подождут возврата VRAM
    // SIGTERM: деструктор движка печатает статистику кэша — точку кривой
    // покрытия снимаем сразу после stop(), пока кольцо ребёнка живо (тик
    // сторожа к мёртвому слоту не успевает).
    child->stop();
    int cache_mib = 0;
    {
      std::lock_guard pl(g_plan_mtx);
      auto pit = g_plan_by_id.find(id);
      if (pit != g_plan_by_id.end()) cache_mib = pit->second.moe_cache;
    }
    if (cache_mib > 0) {
      const std::string tail = child->log_tail();
      size_t pos = tail.find("llama_moe_cache:");
      size_t hr = pos == std::string::npos ? std::string::npos
                                           : tail.find("hit rate = ", pos);
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
  {
    std::lock_guard l(g_mtx);
    auto it = g_slots.find(id);
    if (it != g_slots.end() && it->second.st == St::Stopping)
      it->second.st = to_asleep && !it->second.path.empty() ? St::Asleep : St::Unloaded;
    // Иначе слот уже перезанят (load_harness/проба взяли его) — состояние не трогаем.
  }
  changed();
  if (persist) save_resident_state();
  wake_notify(); // освободившаяся память — шанс для ждущих в очереди
  return true;
}

// ---- парковка KV-состояния слотов на NVMe -----------------------------------

std::string park_dir() {
  // ensure_dir на каждом вызове: каталог могут удалить под живым демоном —
  // движок отказывает на старте, если --slot-save-path не существует.
  static const std::string d = lh::app_dir() + "/slot_parks";
  lh::ensure_dir(d);
  return d;
}

// Сбросить страницы снимка из страничного кэша (POSIX): многогигабайтный
// файл после save/restore иначе вытесняет mmap-страницы экспертов других
// резидентов — «тихий спил весов», скорость падает кратно (см. dio-замеры).
// fdatasync дожимает данные на NVMe (restore-файл может быть удалён сразу
// после — данные уже на диске на случай краша), DONTNEED выкидывает кэш.
void park_drop_cache(const std::string& path) {
#ifndef _WIN32
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) return;
  ::fdatasync(fd);
  ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
  ::close(fd);
#else
  (void)path;
#endif
}

// Стем модели, ужатый до безопасного имени файла: движок (fs_validate_filename)
// отвергает пути и служебные символы в filename, файл всегда лежит в каталоге
// --slot-save-path. Граница имени проверяется суффиксом «.slotN.bin», поэтому
// стем-префикс чужой модели не совпадёт («abc» не заденет «abc-def.slot0.bin»).
static std::string park_stem_of(const std::string& model_path) {
  std::string stem = fs::path(model_path).stem().string();
  std::string out;
  for (unsigned char ch : stem)
    out += (std::isalnum(ch) || ch == '.' || ch == '_' || ch == '-') ? (char)ch
                                                                     : '_';
  if (out.empty()) out = "model";
  if (out.size() > 200) out.resize(200); // запас на суффикс, потолок движка 255
  return out;
}

std::string park_filename_for(const std::string& model_path, int slot) {
  return park_stem_of(model_path) + ".slot" + std::to_string(slot) + ".bin";
}

// Индекс слота из имени «stem.slotN.bin» (-1 — не наш формат).
static int park_slot_of(const std::string& name) {
  const size_t p = name.rfind(".slot");
  if (p == std::string::npos || name.size() < p + 10) return -1;
  const std::string tail = name.substr(p + 5); // «N.bin»
  const size_t dot = tail.rfind(".bin");
  if (dot == std::string::npos || dot == 0 ||
      tail.substr(dot) != ".bin" ||
      tail.find_first_not_of("0123456789", 0) != dot)
    return -1;
  return std::atoi(tail.substr(0, dot).c_str());
}

// Наш ли это файл парковки (стем + ровно «.slotN.bin»), без прихватывания
// чужих моделей с общим префиксом стема.
static bool is_park_file_of(const std::string& name, const std::string& stem) {
  return name.size() > stem.size() + 5 && name.compare(0, stem.size(), stem) == 0 &&
         name.compare(stem.size(), 5, ".slot") == 0 &&
         name.rfind(".bin") == name.size() - 4;
}

static uint64_t parked_bytes_dir() {
  uint64_t total = 0;
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(park_dir(), ec))
    if (e.is_regular_file(ec)) total += (uint64_t)e.file_size(ec);
  return total;
}

// POST /slots/{i}?action=save|restore (контракт пина b11963; тело — одно поле
// filename, голое имя). 200 — тело наружу для n_written/n_restored.
static bool child_post_slots(int port, const std::string& key, int slot,
                             const char* action, const std::string& filename,
                             nlohmann::json* out) {
  httplib::Client cli("127.0.0.1", port);
  cli.set_connection_timeout(3);
  cli.set_read_timeout(180); // 6+ ГиБ KV — секунды; запас на тесный диск
  httplib::Headers hdrs = {{"Authorization", "Bearer " + key}};
  const std::string url = "/slots/" + std::to_string(slot) + "?action=" + action;
  auto res = cli.Post(url.c_str(), hdrs,
                      nlohmann::json{{"filename", filename}}.dump(),
                      "application/json");
  if (!res || res->status != 200) return false;
  if (out) {
    try {
      *out = nlohmann::json::parse(res->body);
    } catch (...) {
      *out = nlohmann::json::object();
    }
  }
  return true;
}

// GET /slots: глубина промпта каждого слота. Свежий слот без задач поля
// n_prompt_tokens не отдаёт — считается пустым (0). Ошибка — пусто.
static std::map<int, long> fetch_child_slot_tokens(int port,
                                                   const std::string& key) {
  std::map<int, long> out;
  httplib::Client cli("127.0.0.1", port);
  cli.set_connection_timeout(3);
  cli.set_read_timeout(10);
  httplib::Headers hdrs = {{"Authorization", "Bearer " + key}};
  auto res = cli.Get("/slots", hdrs);
  if (!res || res->status != 200) return out;
  try {
    auto arr = nlohmann::json::parse(res->body);
    if (!arr.is_array()) return out;
    for (const auto& s : arr) {
      if (!s.is_object() || !s.contains("id")) continue;
      out[s["id"].get<int>()] = s.value("n_prompt_tokens", 0L);
    }
  } catch (...) {
  }
  return out;
}

bool park_before_idle_stop(const std::string& id, bool require_idle) {
  const config::Config cfg = config::current();
  int port = 0, np = 1;
  std::string key, path;
  {
    std::lock_guard l(g_mtx);
    auto it = g_slots.find(id);
    if (it == g_slots.end() || !it->second.parkable || !it->second.child ||
        !it->second.child->alive() || it->second.st != St::Loaded)
      return true; // парковать нечего/некому — сторож останавливает как раньше
    port = it->second.port;
    key = it->second.api_key;
    path = it->second.path;
    np = std::max(1, it->second.np);
  }
  (void)np; // GET /slots перечисляет все слоты сам
  const auto tokens = fetch_child_slot_tokens(port, key);
  // Потолок каталога: сначала LRU-выталкивание старейших снимков (по mtime,
  // чужие тоже — устаревший снимок при несовпадении формы всё равно
  // восстанавливается холодным); раньше потолок просто забивался давними
  // файлами навсегда, и свежие парковки молча пропускались.
  uint64_t total = parked_bytes_dir();
  const uint64_t cap =
      cfg.slot_park_gib > 0 ? (uint64_t)cfg.slot_park_gib << 30 : UINT64_MAX;
  // Выталкивать старейшие, пока tracked-сумма не опустится до limit; keep —
  // файл, который сейчас перезаписываем (его старый размер в total уже
  // учтён, а выталкивать его бессмысленно).
  auto purge_oldest_until = [&](uint64_t limit, const std::string& keep_full) {
    std::vector<std::pair<fs::file_time_type, fs::path>> files;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(park_dir(), ec))
      if (e.is_regular_file(ec) && e.path().string() != keep_full)
        files.push_back({e.last_write_time(ec), e.path()});
    std::sort(files.begin(), files.end());
    for (const auto& [mt, p] : files) {
      if (total <= limit) break;
      const uint64_t sz = (uint64_t)fs::file_size(p, ec);
      if (!fs::remove(p, ec) || ec) break;
      total = total > sz ? total - sz : 0;
      logf("park: потолок каталога — выталкиваю старейший снимок %s (%llu МиБ)",
           p.filename().string().c_str(), (unsigned long long)(sz >> 20));
    }
  };
  for (const auto& [slot, n_tok] : tokens) {
    if (n_tok < cfg.slot_park_min_tokens) continue;
    const std::string fn = park_filename_for(path, slot);
    const std::string full = park_dir() + "/" + fn;
    // Перезапись того же слота: старый размер вычитается, иначе total
    // разрастался бы на каждый простой (старый и новый считались разом).
    std::error_code ec;
    const uint64_t old_sz = fs::file_size(full, ec);
    const uint64_t base = total - (ec ? 0 : std::min(total, old_sz));
    if (base >= cap) {
      logf("park: «%s»: каталог парковки у потолка (%llu МиБ) — слот %d "
           "не сохранён",
           id.c_str(), (unsigned long long)(total >> 20), slot);
      break;
    }
    purge_oldest_until(cap, full);
    nlohmann::json resp;
    if (child_post_slots(port, key, slot, "save", fn, &resp)) {
      const long long written = resp.value("n_written", 0LL);
      const double ms = resp.value("timings", nlohmann::json::object())
                            .value("save_ms", 0.0);
      logf("park: «%s» слот %d: сохранено %lld токенов (%lld МиБ) за %.1f с",
           id.c_str(), slot, resp.value("n_saved", 0LL), written >> 20,
           ms / 1000.0);
      park_drop_cache(full); // снимок не должен теснить страницы экспертов
      total = parked_bytes_dir(); // факт: перезапись + выталкивания
    } else {
      logf("park: «%s» слот %d: сохранение не удалось — слот уйдёт холодным",
           id.c_str(), slot);
    }
  }
  // Перепроверка после секундного HTTP: за время сохранения модель могли
  // разбудить или занять — останавливать её посреди чужого хода нельзя.
  {
    std::lock_guard l(g_mtx);
    auto it = g_slots.find(id);
    if (it == g_slots.end() || !it->second.child || !it->second.child->alive() ||
        it->second.st != St::Loaded)
      return false;
    const auto idle = std::chrono::duration_cast<std::chrono::minutes>(
                          clock_::now() - it->second.last_used).count();
    // require_idle=false — вытеснение под память/лимит, а не простой: слот
    // могли использовать только что, парковать всё равно честно.
    if (require_idle && cfg.unload_after_min > 0 && idle < cfg.unload_after_min)
      return false;
  }
  {
    std::lock_guard l2(g_busy_mtx);
    if (g_busy_set.count(id) > 0) return false;
  }
  return true;
}

bool park_restore_after_load(const std::string& id) {
  const config::Config cfg = config::current();
  if (!cfg.slot_park) return false;
  int port = 0;
  std::string key, path;
  {
    std::lock_guard l(g_mtx);
    auto it = g_slots.find(id);
    if (it == g_slots.end() || !it->second.parkable) return false;
    port = it->second.port;
    key = it->second.api_key;
    path = it->second.path;
  }
  bool any = false;
  // Файлы по стему модели, а не диапазон -np: нумерацию слотов ведёт ребёнок
  // (GET /slots), и она не обязана совпадать с np демона — при parallel=1
  // ребёнок с двумя внутренними слотами паркует «slot1.bin», который диапазон
  // np=1 никогда не нашёл бы (restore молча не работал).
  const std::string stem = park_stem_of(path);
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(park_dir(), ec)) {
    if (ec) break;
    if (!e.is_regular_file(ec)) continue;
    const std::string fn = e.path().filename().string();
    if (!is_park_file_of(fn, stem)) continue;
    const int slot = park_slot_of(fn);
    if (slot < 0) continue;
    nlohmann::json resp;
    const std::string full = e.path().string();
    if (child_post_slots(port, key, slot, "restore", fn, &resp)) {
      const double ms = resp.value("timings", nlohmann::json::object())
                            .value("restore_ms", 0.0);
      logf("park: «%s» слот %d: восстановлено %lld токенов за %.1f с — ход "
           "клиента попадёт в кэш промпта",
           id.c_str(), slot, resp.value("n_restored", 0LL), ms / 1000.0);
      park_drop_cache(full); // страницы снимка в кэше больше не нужны
      // Файл потреблён: KV теперь в VRAM. Без удаления старый снимок лежал
      // бы вечно (до потолка каталога), тесня свежие парковки.
      std::error_code ec;
      fs::remove(full, ec);
      any = true;
    } else {
      // Штатный отказ (другой ctx/модель/движок после апгрейда): движок сам
      // очистил слот, ребёнок жив и работает холодным.
      logf("park: «%s» слот %d: возврат не состоялся — слот холодный",
           id.c_str(), slot);
    }
  }
  return any;
}

uint64_t parked_bytes_for(const std::string& model_path) {
  const std::string stem = park_stem_of(model_path);
  uint64_t total = 0;
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(park_dir(), ec)) {
    if (!e.is_regular_file(ec)) continue;
    const std::string name = e.path().filename().string();
    if (is_park_file_of(name, stem)) total += (uint64_t)e.file_size(ec);
  }
  return total;
}

void park_purge_for_path(const std::string& model_path) {
  const std::string stem = park_stem_of(model_path);
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(park_dir(), ec)) {
    if (!e.is_regular_file(ec)) continue;
    const std::string name = e.path().filename().string();
    if (is_park_file_of(name, stem)) {
      logf("park: «%s»: удаляю файл парковки %s", model_path.c_str(),
           name.c_str());
      fs::remove(e.path(), ec);
    }
  }
}

// Выгрузить одну (id/путь) или все НЕзанятые (пусто). Занятая модель не
// выселяется никогда — главный инвариант harness-режима; force=true (только
// для одной модели, явное решение пользователя) — люк в инварианте: останов
// даже занятого слота, KV не паркуется (снимок посреди генерации — мусор).
// (impl — под именем _impl внутри анонимного ns; публичная обёртка — сразу
// после его закрытия.)
nlohmann::json harness_unload_models_impl(const std::string& id, std::string& err,
                                          bool force) {
  nlohmann::json out{{"state", "unloaded"}};
  std::set<std::string> busy_snap;
  {
    std::lock_guard lb(g_busy_mtx);
    busy_snap = g_busy_set;
  }
  std::vector<std::string> targets, busy_kept;
  {
    std::lock_guard l(g_mtx);
    if (!id.empty()) {
      // id или путь — приводим к id слота.
      std::string sid;
      if (g_slots.count(id)) sid = id;
      if (sid.empty())
        for (auto& [i, p] : g_id_map)
          if (path_eq(p, id)) { sid = i; break; }
      if (sid.empty() && id.find(".gguf") == std::string::npos) sid = id;
      auto it = g_slots.find(sid);
      if (it == g_slots.end()) {
        err = tr("модель «") + id + tr("» не загружена");
        return {};
      }
      targets.push_back(sid);
    } else {
      for (auto& [sid, s] : g_slots)
        if (s.st == St::Loaded || s.st == St::Asleep || s.st == St::Error)
          targets.push_back(sid);
    }
  }
  for (auto& sid : targets) {
    // Снимок busy — быстрый путь; атомарный отказ stop_hslot ловит запрос,
    // начавший между снимком и остановом (гонка «снимок → stop» закрыта).
    if (!force && busy_snap.count(sid)) {
      if (!id.empty()) {
        err = tr("модель «") + sid + tr("» занята — идёт генерация, выгрузка отклонена");
        return {};
      }
      busy_kept.push_back(sid);
      continue;
    }
    if (!stop_hslot(sid, true, /*persist=*/true, force)) {
      if (!id.empty()) {
        err = tr("модель «") + sid + tr("» занята — идёт генерация, выгрузка отклонена");
        return {};
      }
      busy_kept.push_back(sid);
    }
  }
  if (!busy_kept.empty()) out["busy_kept"] = busy_kept;
  return out;
}

std::set<std::string> g_kv_probe;

// Соседняя форма KV снимается не в той загрузке, которой ждёт клиент.
// Поток ждёт восемь секунд простоя (генерация держит g_busy_set) и только
// тогда останавливает ребёнка. Запрос, пришедший в это окно, видит id в
// g_kv_probe и ждёт, пока слот снова не станет Loaded.
void schedule_kv_probe(std::string id, std::string path, std::string exe, std::string mmproj,
                       std::string kv_type, gguf::Meta meta, uint64_t fsize, GpuTarget tgt,
                       SpawnExtra ex, int ngl, int ctx, int ubatch, int port, int ncmoe,
                       bool fa, bool kv_q8, bool exps_cpu, bool cuda) {
  lh::sup::track_background([=]() mutable {
    const auto deadline = clock_::now() + std::chrono::minutes(30);
    bool quiet = false;
    auto quiet_since = clock_::now();
    while (clock_::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      bool busy = false;
      {
        std::lock_guard lb(g_busy_mtx);
        busy = g_busy_set.count(id) > 0;
      }
      bool same = false;
      {
        std::lock_guard l(g_mtx);
        auto it = g_slots.find(id);
        same = it != g_slots.end() && it->second.st == St::Loaded && it->second.port == port &&
               it->second.child && it->second.child->alive();
      }
      if (!same) return;
      if (busy) {
        quiet = false;
        continue;
      }
      if (!quiet) {
        quiet = true;
        quiet_since = clock_::now();
        continue;
      }
      if (clock_::now() - quiet_since >= std::chrono::seconds(8)) break;
    }
    if (!quiet || clock_::now() - quiet_since < std::chrono::seconds(8)) return;

    const bool sib_q8 = !kv_q8;
    const int layers = (int)meta.n_layers;
    const bool out_gpu = ngl > layers;
    const int ngl_key = (ngl > 0 && ngl < layers) ? ngl : 0;
    profiles::Form sib{out_gpu, sib_q8, ubatch, cuda, exps_cpu, ncmoe, ngl_key};
    if (profiles::speed_settled(id, sib)) return;

    std::shared_ptr<Child> old;
    {
      std::lock_guard lb(g_busy_mtx);
      if (g_busy_set.count(id)) return;
    }
    // Проба не должна убивать тёплый контекст: у дочерних слотов должен быть
    // пустой промпт (n_prompt_tokens == 0). Один завершившийся ход диалога с
    // паузой больше окна тишины оставляет живой кэш — без гейта проба
    // снесла бы его, и следующий ход клиента уехал бы в полный перепрефилл.
    {
      std::string pkey;
      {
        std::lock_guard l(g_mtx);
        auto it = g_slots.find(id);
        if (it != g_slots.end()) pkey = it->second.api_key;
      }
      const auto tokens = fetch_child_slot_tokens(port, pkey);
      bool warm = false;
      for (const auto& [sidx, n] : tokens)
        if (n > 0) {
          logf("harness: «%s»: проба соседней формы KV отменена — в слоте %d "
               "живой промпт (%ld токенов)", id.c_str(), sidx, n);
          warm = true;
          break;
        }
      if (warm) return;
    }
    // Гейт «загрузка/установка не совпадают»: проба спавнит ребёнка — берёт
    // билет загрузки до перевода слота в Loading; установка в полёте — проба
    // отменяется, слот не трогаем (см. engine.h).
    engine::GateTicket gate;
    std::string gate_err;
    if (!gate.acquire(engine::GateTicket::Kind::Load, gate_err)) {
      logf("harness: «%s»: проба соседней формы KV отменена (%s)", id.c_str(),
           gate_err.c_str());
      return;
    }
    {
      std::lock_guard l(g_mtx);
      auto it = g_slots.find(id);
      if (it == g_slots.end() || it->second.st != St::Loaded || it->second.port != port ||
          !it->second.child || !it->second.child->alive())
        return;
      g_kv_probe.insert(id);
      it->second.st = St::Loading;
      old = it->second.child;
    }
    auto drop_probe = [&](St restore) {
      std::lock_guard l(g_mtx);
      g_kv_probe.erase(id);
      auto it = g_slots.find(id);
      if (it == g_slots.end()) return;
      if (it->second.st == St::Loading) it->second.st = restore;
    };
    {
      std::lock_guard lb(g_busy_mtx);
      if (g_busy_set.count(id)) {
        drop_probe(St::Loaded);
        return;
      }
    }

    logf("harness: «%s»: один замер соседней формы KV (%s)", id.c_str(), sib_q8 ? "q8_0" : "f16");
    // Тихий спил весов: без ожидания возврата VRAM замер соседней формы едет
    // на недосвобожденной карте — скорость лжи в профиль (note_speed) и
    // приговор kv_q8_by_speed по кривому замеру (история dio: ×7 падение).
    const uint64_t free_before_kill = settle_view(tgt.split ? -1 : tgt.main_gpu).vram_free;
    g_last_kill_ms = steady_ms();
    old->stop();
    vram_settle(free_before_kill + fsize, tgt.split ? -1 : tgt.main_gpu);
    auto probe = std::make_shared<Child>();
    int pport = 0, pctx = 0;
    std::string pkey, perr, ptail;
    bool pwedge = false;
    SpawnExtra pex = ex;
    pex.ncmoe = ncmoe;
    const bool pok = try_attempt(exe, path, ngl, ctx, fa, sib_q8, kv_type, exps_cpu, ubatch, mmproj,
                                 try_target_of(tgt), pex, *probe, pport, pkey, pctx, perr, &pwedge,
                                 &ptail);
    WarmSample ps;
    if (pok) ps = warmup(pport, pkey);

    auto abandoned = [&]() {
      std::lock_guard l(g_mtx);
      auto it = g_slots.find(id);
      return it == g_slots.end() || it->second.st != St::Loading || !g_kv_probe.count(id);
    };
    if (abandoned()) {
      if (pok) probe->stop();
      {
        std::lock_guard l(g_mtx);
        g_kv_probe.erase(id);
      }
      wake_notify();
      return;
    }

    bool keep = false;
    if (pok && ps.n >= 128) {
      profiles::note_speed(id, sib, ps.tps, ps.n);
      // Выборка победителя уже записана при загрузке. Здесь сравниваем с ней.
      double win_tps = 0;
      int win_n = 0;
      profiles::Form win{out_gpu, kv_q8, ubatch, cuda, exps_cpu, ncmoe, ngl_key};
      profiles::speed_of(id, win, win_tps, win_n);
      const double tps_q8 = sib_q8 ? ps.tps : win_tps;
      const double tps_f16 = sib_q8 ? win_tps : ps.tps;
      const int n_q8 = sib_q8 ? ps.n : win_n;
      const int n_f16 = sib_q8 ? win_n : ps.n;
      if (profiles::kv_q8_by_speed(kv_q8, tps_q8, n_q8, tps_f16, n_f16) == sib_q8) {
        keep = true;
        logf("harness: «%s»: соседняя форма KV быстрее — оставляю её", id.c_str());
      }
    } else {
      profiles::mark_speed_skipped(id, sib);
      logf("harness: «%s»: соседняя форма KV не снялась — остаётся текущая", id.c_str());
    }

    std::shared_ptr<Child> kept = probe;
    int kport = pport, kctx = pctx;
    std::string kkey = pkey;
    bool kq8 = sib_q8;
    if (!keep) {
      if (pok) probe->stop();
      kept = std::make_shared<Child>();
      std::string rerr, rtail;
      bool rwedge = false;
      const bool rok = try_attempt(exe, path, ngl, ctx, fa, kv_q8, kv_type, exps_cpu, ubatch,
                                   mmproj, try_target_of(tgt), ex, *kept, kport, kkey, kctx, rerr,
                                   &rwedge, &rtail);
      if (!rok) {
        logf("harness: «%s»: не удалось вернуть форму KV после пробы", id.c_str());
        drop_probe(St::Unloaded);
        wake_notify();
        return;
      }
      warmup(kport, kkey);
      kq8 = kv_q8;
    }
    if (abandoned()) {
      kept->stop();
      {
        std::lock_guard l(g_mtx);
        g_kv_probe.erase(id);
      }
      wake_notify();
      return;
    }
    bool slot_took = false;
    {
      std::lock_guard l(g_mtx);
      auto it = g_slots.find(id);
      if (it != g_slots.end() && it->second.st == St::Loading) {
        it->second.child = kept;
        it->second.port = kport;
        it->second.api_key = kkey;
        it->second.ctx = kctx;
        it->second.kv_q8 = kq8;
        it->second.st = St::Loaded;
        it->second.last_used = clock_::now();
        slot_took = true;
      }
      g_kv_probe.erase(id);
    }
    if (!slot_took) kept->stop(); // вне g_mtx: stop() ждёт смерти процесса
    wake_notify();
  });
}


// Хвост отказа с вариантами действий: тексты ошибок загрузки едут клиенту
// дословно (WakeResult.err → 503 error.message чата) — там должно быть
// «что делать», а не только «что случилось». Релевантное: vision-проекция
// съедает карту — отдать её можно выключением; пин контекста — снятием.
std::string fail_advice(const std::string& id, bool vision, bool pinned_ctx) {
  std::string adv;
  auto add = [&](const std::string& s) {
    if (!adv.empty()) adv += " · ";
    adv += s;
  };
  if (vision) add("lhctl vision " + id + " off");
  if (pinned_ctx) add("lhctl ctx " + id + " auto");
  add(tr("закройте резидентов (свободите VRAM)"));
  return "; варианты: " + adv;
}

// Диагноз деградации формы спавна: лестница поймала OOM и осела ниже плана —
// клиент в чате видит ноту с фактом и вариантами, а не гадает по скорости
// (31.10: плотная 27B молча уехала в ngl 32 — 8 ток/с против 39). Форма
// сравнивается с ПЛАНОМ (что обещал plan_for до лестницы), не с «идеалом»:
// вынос экспертов, заложенный планом, — не деградация. plan_* — снимок плана
// до раундов (раунд 2 демотирует выходной слой в самом плане). Пусто —
// загрузка состоялась ровно по плану.
std::string degrade_note(const std::string& id, int plan_ngl, int plan_ctx,
                         bool plan_exps, int plan_ncmoe, bool plan_q8, int plan_cache,
                         const plan::Attempt& used, int actual_cache, long long n_layers,
                         bool vision, bool pinned_ctx) {
  std::string what;
  auto add = [&](const std::string& s) {
    if (!what.empty()) what += ", ";
    what += s;
  };
  if (used.ngl < plan_ngl) {
    if (plan_ngl > n_layers && used.ngl <= n_layers && n_layers > 0)
      add(tr("выходной слой на CPU"));
    if (n_layers > 0 && used.ngl > 0 && used.ngl < n_layers)
      add(tr("слоёв на GPU ") + std::to_string(used.ngl) + tr(" из ") + std::to_string(n_layers) +
          tr(" (хвост в RAM)"));
    else if (used.ngl <= 0)
      add(tr("веса целиком в RAM"));
  }
  if (used.ctx > 0 && used.ctx < plan_ctx)
    add(tr("контекст ") + std::to_string(used.ctx) + tr(" вместо ") + std::to_string(plan_ctx));
  if (used.exps_cpu && !plan_exps) add(tr("эксперты целиком в RAM"));
  if (used.ncmoe > plan_ncmoe && plan_ncmoe == 0)
    add(tr("экспертов в RAM ") + std::to_string(used.ncmoe));
  if (actual_cache < plan_cache) add(tr("кэш MoE снят"));
  if (plan_q8 && !used.kv_q8) add("KV f16 вместо q8_0");
  if (what.empty()) return {};
  return tr("загружено урезанно: ") + what + tr(" — не влезла в память") +
         fail_advice(id, vision, pinned_ctx);
}


// Загрузка модели в свой слот: предполёт → выселение незанятых по плану →
// лестница спавна (classic-механика с draft/-np) → прогрев → профиль.
// busy_blockers_out непуст на выходе = грузиться не пытались, мешают занятые
// (решение о парковке — за вызывающим).
bool load_harness(const std::string& path, const std::string& id, std::string& err,
                  std::vector<std::string>* busy_blockers_out) {
  if (busy_blockers_out) busy_blockers_out->clear();
  config::Config cfg = config::current();
  {
    std::lock_guard l(g_wedged_mtx);
    if (g_wedged_models.count(path)) {
      err = tr("модель в стоп-листе: прошлую загрузку пришлось убивать, а ребёнок не умер "
            "(клин драйвера) — перезапустите демон, чтобы повторить");
      logf("harness: отказ: %s", err.c_str());
      return false;
    }
  }

  gguf::Meta meta;
  bool have_meta = lh::modelreg::model_meta(path, meta);
  // Шард-группа — одна модель: в веса входит сумма частей, не только
  // первая, которую открывает движок.
  const uint64_t fsize = lh::modelreg::shard_group_bytes(path);

  // Слот переводим в Loading сразу — статус виден и вторая будилка той же
  // модели не пойдёт спавнить параллельно. Полная сериализация будилок —
  // g_wake_serial_mtx в harness_wake (см. ниже).
  {
    std::lock_guard l(g_mtx);
    auto& slot = g_slots[id];
    if (slot.st == St::Loading && slot.path == path) {
      err = tr("уже идёт загрузка");
      return false;
    }
    if (slot.st == St::Loaded && slot.path == path && slot.child && slot.child->alive()) {
      slot.last_used = clock_::now();
      return true;
    }
    slot.id = id;
    slot.path = path;
    slot.st = St::Loading;
    slot.err.clear();
    slot.note.clear(); // нота прошлого спавна недействительна — форма новая
    slot.child.reset();
  }
  changed();
  // Путь — в журнал: id санитизирован (небезопасные имена в «_»), по нему
  // файл в логе не найти; на диагностике «что грузилось» нужен literal путь.
  logf("harness: «%s»: загрузка %s", id.c_str(), path.c_str());
  // Провал загрузки — слот в Error с причиной; занятость/парковка сюда не
  // доходят (это не провал, «слот» просто не начинают).
  auto fail = [&](const std::string& why) {
    std::shared_ptr<Child> dead;
    {
      std::lock_guard l(g_mtx);
      // Id вне реестра (админский путь sup::load мимо карты): не заводить
      // фантомный Error-слот — причину уже получил вызывающий (err), а сторож
      // простоя и вотчер реестра такого id не знают и не удалят никогда.
      if (!g_id_map.count(id) && !g_slots.count(id)) {
        err = why;
        return;
      }
      auto& slot = g_slots[id];
      if (slot.child) {
        dead = slot.child;
        slot.child.reset();
      }
      slot.port = 0;
      slot.st = St::Error;
      slot.err = why;
      err = why;
    }
    if (dead) dead->stop(); // вне g_mtx: stop() ждёт смерти процесса до 5 с
  };

  auto ov = load_overrides();
  const GpuTarget tgt = gpu_target_for(id, path, cfg, ov, true);
  auto list = lh::modelreg::scan(config::models_dir_resolved(cfg));
  const ModelOverrides* o = overrides_for(ov, id, path);
  // Пер-модельный размер кэша MoE ([model] moe-cache-mib): у моделей разный
  // оптимум (Flash-Next 8192, Next-80B 10752 — замер 30.10), глобальному
  // ключу такое не поднять. 0 — решает [general].
  if (o && o->moe_cache_mib != 0) cfg.moe_cache_mib = o->moe_cache_mib;
  const bool emb = embedding_arch(meta.arch);
  DraftPick dp = find_draft(path, meta, fsize, cfg, o, list);
  // Встроенный nextn перекрывает авто-голову внутри find_draft — те же
  // рамки, что у встроенного nextn в find_draft (см. комментарий там).
  if (!dp.path.empty())
    logf("harness: speculative decoding%s: +draft %llu МиБ",
         dp.auto_attached ? " (авто)" : "",
         (unsigned long long)(dp.size >> 20));
  std::string mm_diag;
  std::string mmpath = mmproj_for_load(path, o, &mm_diag);
  if (!mmpath.empty())
    logf("harness: vision: mmproj %s (%llu МиБ)", mmpath.c_str(),
         (unsigned long long)(lh::file_size(mmpath) >> 20));
  else if (!mm_diag.empty())
    logf("harness: «%s»: vision: %s", id.c_str(), mm_diag.c_str());
  const uint64_t mm_size = mmpath.empty() ? 0 : lh::file_size(mmpath);
  const uint64_t draft_slack =
      !dp.path.empty() && dp.spec_type == "draft-mtp"
          ? draft_buffer_slack(cfg, dp.n_vocab ? dp.n_vocab : meta.n_vocab)
          : 0;
  const uint64_t fsize_eff = fsize + dp.size + mm_size + draft_slack;

  // Предполёт: заведомо не влезающую даже на пустой пул не начинаем — общая
  // математика preflight_vram (запасные ярусы: MoE с экспертами в RAM,
  // плотная с хвостом слоёв в RAM).
  const auto pf = preflight_vram(have_meta ? &meta : nullptr, fsize_eff, tgt.card, cfg,
                                 dp.size + mm_size + draft_slack, o ? o->ctx_size : 0);
  if (pf.refused) {
      err = tr("модели нужно ") + std::to_string((fsize_eff + (512ull << 20)) >> 30) +
            tr(" ГиБ видеопамяти") +
            (tgt.main_gpu >= 0 ? tr(" на её карте") : tr(" (с учётом раскладки по картам)")) +
            ", а доступно " + std::to_string(tgt.card.vram_total >> 30) +
            " ГиБ — не поместится даже после выгрузки остальных" +
            fail_advice(id, !mmpath.empty(), o && o->ctx_size >= 256);
      logf("harness: отказ: %s", err.c_str());
      fail(err);
      changed();
      return false;
  }

  // models-max в harness: резиденторию диктует не только VRAM — лимит
  // количества из конфига. Три маленькие модели спокойно жили при
  // models-max = 2 (поймано заморозкой beta 12, сценарий 2.1). ЛИМИТ считает
  // всех загруженных по картам цели — занятые тоже; раньше считались одни
  // незанятые, и при лимите 2 (занятый + незанятый) грузилась третья модель.
  // Выселение — только незанятых, LRU, с парковкой KV (вытеснение под лимит —
  // не простой: художник вытесняет LLM, ход клиента вернётся из кэша промпта).
  // Художник на своей карте — не сосед: лимит по пересечению карт цели, его
  // загрузка не гонится за лишними выселениями и не стоит за ним в очереди.
  if (cfg.models_max > 0) {
    const std::set<int> my_cards = target_cards(tgt);
    for (;;) {
      std::vector<std::pair<clock_::time_point, std::string>> loaded, evictable;
      {
        std::set<std::string> busy_snap;
        {
          std::lock_guard lb(g_busy_mtx);
          busy_snap = g_busy_set;
        }
        std::lock_guard l(g_mtx);
        for (auto& [sid, s] : g_slots) {
          if (sid == id || s.st != St::Loaded) continue;
          if (!cards_overlap(s.cards, my_cards)) continue;
          loaded.push_back({s.last_used, sid});
          if (!busy_snap.count(sid)) evictable.push_back({s.last_used, sid});
        }
      }
      if ((int)loaded.size() < cfg.models_max) break;
      if (evictable.empty()) break; // все заняты — конфликт решит VRAM-ветка и очередь
      std::sort(evictable.begin(), evictable.end());
      const std::string& victim = evictable.front().second;
      logf("harness: models-max = %d: выселяю самого старого незанятого "
           "«%s» для «%s»",
           cfg.models_max, victim.c_str(), id.c_str());
      park_before_idle_stop(victim, /*require_idle=*/false); // best-effort
      if (!stop_hslot(victim, true)) break; // жертва занялась — конфликт решит VRAM-ветка
    }
  }

  // ВЫСЕЛЕНИЕ по плану (конвенция роутера: need против пустой карты цели).
  // Незанятые резиденты выселяются, занятые — нет: если после выселения
  // незанятых всё равно тесно и виноваты занятые, загрузка не начинается,
  // а список блокеров уезжает в очередь пробуждения.
  if (have_meta && tgt.card.vram_total > 0) {
    hw::Gpu empty = tgt.card;
    empty.vram_free = std::max(empty.vram_free, empty.vram_total);
    SmartPlan pp = plan_for(meta, fsize, empty, cfg, id,
                            dp.size + mm_size + draft_slack, kv_force_of(o, id),
                            o ? o->ctx_size : 0, /*announce=*/false);
    const uint64_t gpu_base = gpu_base_of(pp, meta, fsize);
    hw::Gpu need_view = tgt.card;
    need_view.build = hw::Build::Cuda13;
    const uint64_t need = gpu_base + dp.size + mm_size + draft_slack +
                          fit::estimate(meta, 0, (uint32_t)std::max(pp.ctx0, 512),
                                        need_view, true).total;
    // Жертвы — по картам цели: художник на своей карте цели не мешает, его
    // загрузка не должна стоять за ним в очереди и не выселяет его.
    const std::set<int> evict_cards = target_cards(tgt);
    std::vector<std::pair<clock_::time_point, std::string>> idle_r; // LRU: старые первыми
    std::vector<std::string> busy_r;
    {
      std::set<std::string> busy_snap;
      {
        std::lock_guard lb(g_busy_mtx);
        busy_snap = g_busy_set;
      }
      std::lock_guard l(g_mtx);
      for (auto& [sid, s] : g_slots) {
        if (sid == id || s.st != St::Loaded || !s.child || !s.child->alive()) continue;
        if (!cards_overlap(s.cards, evict_cards)) continue;
        if (busy_snap.count(sid))
          busy_r.push_back(sid);
        else
          idle_r.push_back({s.last_used, sid});
      }
    }
    hw::Gpu view = settle_view(tgt.split ? -1 : tgt.main_gpu);
    if (view.vram_free < need) {
      // LRU до достаточного места, не «всё разом»: маленький эмбеддер не
      // уходит вместе с большой моделью, если хватило первой жертвы. Парковка
      // KV перед остановом (без требования простоя — вытеснение под память):
      // пути простоя тут нет, а ход клиента выселенной модели обязан вернуться
      // в кэш промпта (сценарий «художник вытесняет LLM» — основной).
      std::sort(idle_r.begin(), idle_r.end());
      for (auto& [ts, sid] : idle_r) {
        if (view.vram_free >= need) break;
        logf("harness: выселяю незанятого резидента «%s» — освобождаю память для «%s»",
             sid.c_str(), id.c_str());
        park_before_idle_stop(sid, /*require_idle=*/false); // best-effort
        if (!stop_hslot(sid, true)) continue; // занялся между снимком и остановом
        vram_settle(need, tgt.split ? -1 : tgt.main_gpu);
        view = settle_view(tgt.split ? -1 : tgt.main_gpu);
      }
      if (view.vram_free < need && !busy_r.empty()) {
        // Мешают занятые: выселение запрещено, грузиться «так» — рулетка OOM
        // рядом с чужим стримом. Парковка решает вызывающий.
        if (busy_blockers_out) *busy_blockers_out = busy_r;
        err = tr("мешают занятые резиденты: ") + join_ids(busy_r, ", ") +
              " — выселение запрещено, ждём освобождения";
        logf("harness: «%s»: %s", id.c_str(), err.c_str());
        std::lock_guard l(g_mtx);
        auto& slot = g_slots[id];
        if (slot.st == St::Loading && !slot.child) slot.st = St::Unloaded;
        changed();
        return false;
      }
      // Иначе грузимся оптимистично, как роутер: аналитика ошибается на
      // ±1 ГиБ (TURBO «не влезал по плану» и спокойно живёт на карте), а
      // честный OOM ловит лестница, понижая параметры.
    }
  }

  std::string exe = lh::engine::find_server(cfg);
  if (exe.empty()) {
    fail("llama-server не найден — скачайте (POST /v0/engine) или укажите путь в настройках");
    changed();
    return false;
  }
  if (!lh::file_exists(path)) {
    fail(tr("файл модели не найден"));
    changed();
    return false;
  }

  // План — та же математика умного подбора: пустая карта цели с резервом,
  // выходной слой с полным грузом, параллельные слоты, draft в бюджете.
  // Запасная карта под блоки экспертов (expert-card = spare): план строит
  // -ot-кандидата сетки авто-режима по той же живой свободе, что и спавн —
  // если юзер включил, но запасной нет, кандидат просто не строится.
  const bool slow = lh::slow_media(path);
  const bool smart = have_meta && cfg.auto_params;
  hw::Gpu plan_card = tgt.card;
  if (plan_card.vram_total > 0) plan_card.vram_free = plan_card.vram_total - kPlanReserve;
  int ot_spare = -1;
  if (o && o->expert_card == "spare" && !tgt.split && tgt.main_gpu >= 0)
    ot_spare = spare_card_index(target_cards(tgt), 768ull << 20);
  SmartPlan p = plan_for(meta, fsize, plan_card, cfg, id,
                         dp.size + mm_size + draft_slack, kv_force_of(o, id),
                         o ? o->ctx_size : 0, /*announce=*/true, /*cache_exact=*/-2,
                         ot_spare);
  if (emb) { p.kv_q8 = false; p.fa = false; } // эмбеддеру FA и q8 не нужны
  if (o) {
    if (o->ngl > 0) p.ngl0 = o->ngl;
    if (o->ctx_size >= 256) p.ctx0 = o->ctx_size;
  }
  uint64_t gpu_base = gpu_base_of(p, meta, fsize);
  uint64_t full_load = gpu_base + (dp.path.empty() ? 0 : dp.size) + mm_size + draft_slack;
  if (o && o->ngl > 0) {
    p.ngl0 = std::min(o->ngl, (int)meta.n_layers + 1);
  } else {
    hw::Gpu out_card = tgt.card;
    if (engine::launches_cuda(cfg)) out_card.build = hw::Build::Cuda13;
    std::optional<ModelProfile> pf =
        profiles::lookup(id, {/*out_gpu=*/true, p.kv_q8, cfg.ubatch,
                              engine::launches_cuda(cfg), p.exps_cpu, p.ncmoe});
    ngl_with_output(p, meta, full_load, out_card, p.kv_q8, id, pf);
  }
  if (cfg.parallel > 1 && !emb) {
    int np = cfg.parallel;
    if (p.ctx0 / np < 2048) np = std::max(1, p.ctx0 / 2048);
    int per_slot = p.ctx0 / np;
    if (meta.n_ctx_train && per_slot > (int)meta.n_ctx_train)
      per_slot = (int)meta.n_ctx_train;
    p.ctx0 = per_slot * np;
    p.np = np;
  }
  SpawnExtra ex;
  ex.np = p.np;
  ex.embedding = emb;
  ex.moe_cache_mib = p.moe_cache; // план погасит без выноса; try_attempt условен так же
  ex.moe_cpu_share = cfg.moe_cpu_share;
  ex.threads = cfg.threads;
  ex.extra_args = cfg.extra_args;
  // Пер-модельный thinking-mode (auto/on/off): auto — флага нет (движок сам
  // детектит по шаблону); on/off — --reasoning. Эмбеддеру не нужен.
  ex.think_mode = (!emb && o) ? o->think_mode : std::string();
  // Парковка KV слотов: каталог создаётся при первом вызове park_dir() —
  // движок отказывает на старте, если --slot-save-path не существует.
  ex.slot_save_path = cfg.slot_park && !emb ? park_dir() : "";
  // Честность раскладки на запасную карту: факт замеряется ПОСЛЕ загрузки
  // (пин b11934, проверено живьём 28.10, может молча не применять
  // --device-draft — тогда предупреждение, не немой no-op).
  int draft_spare_card = -1, expert_spare_card = -1;
  uint64_t draft_spare_before = 0, expert_spare_before = 0, expert_expect = 0;
  if (!dp.path.empty()) {
    ex.draft_path = dp.path;
    ex.spec_type = dp.spec_type;
    ex.draft_ngl = dp.n_layers > 0 ? dp.n_layers + 1 : p.ngl0;
    // «Черновик — на запасную карту» (draft-card = spare): ствол живёт на
    // своём пине, черновик с весами, compute-буферами и своим KV — на карте
    // вне раскладки ствола. Вторая карта перестаёт греть воздух, у ствола
    // остаётся плановый запас. Метим по ЖИВОЙ свободе (грузимся сейчас), в
    // скобках — KV черновика при p.ctx0 (по слотам суммарно это и есть ctx0).
    // Работает только для draft-simple: MTP-черновик (файловая голова или
    // встроенный nextn) читает KV и активации ствола через общий граф — на
    // другом устройстве он не раскладывается (проверено живьём 28.10 на
    // b11934: --device-draft + draft-mtp = abort «pre-allocated tensor
    // (cache_k_l46) in a buffer (CUDA1) that cannot run the operation
    // (NONE)», ребёнок мрёт на загрузке).
    if (o && o->draft_card == "spare") {
      if (dp.spec_type != "draft-simple") {
        logf("harness: «%s»: draft-card = spare проигнорирован для «%s»: "
             "MTP-черновик обязан жить на карте ствола (общий с ним граф KV), "
             "на запасную переселяется только draft-simple", id.c_str(),
             dp.spec_type.c_str());
      } else {
        const int spare = spare_card_index(
            target_cards(tgt),
            dp.size + draft_kv_bytes(dp, p.ctx0) + draft_slack + (768ull << 20));
        if (spare >= 0) {
          ex.draft_device = "CUDA" + std::to_string(spare);
          draft_spare_card = spare;
          const hw::Gpu sc = settle_view(spare);
          draft_spare_before = sc.vram_total - sc.vram_free;
          logf("harness: «%s»: draft-card = spare — черновик (%llu МиБ + KV %llu МиБ) "
               "на CUDA%d, ствол не трогаем", id.c_str(),
               (unsigned long long)(dp.size >> 20),
               (unsigned long long)(draft_kv_bytes(dp, p.ctx0) >> 20), spare);
        } else {
          logf("harness: «%s»: draft-card = spare, но запасной карты с местом "
               "под черновик нет — едет на карту ствола, как раньше", id.c_str());
        }
      }
    }
  } else if (!dp.spec_type.empty()) {
    ex.spec_type = dp.spec_type; // встроенный MTP: тип без draft-файла
  }

  // «До спавна» для профиля: после выселения и settle — дельта «до/после»
  // отделит аллокацию этого ребёнка от памяти резидентов.
  hw::flush_telemetry_cache();
  const hw::Gpu vram_before = settle_view(tgt.split ? -1 : tgt.main_gpu);

  WarmSample speed;
  bool ok = false;
  std::string last_err;
  plan::Attempt used{}; // форма удавшейся попытки — для записи профиля
  int used_ot = 0;      // блоков экспертов реально на запасной карте (форма)
  std::string success_tail; // кольцо удачного ребёнка: разбивка буферов загрузки
  // Снимок плана ДО раундов: раунд 2 демотирует выходной слой в самом плане
  // (p.ngl0 = n_layers), а нота деградации сравнивает факт с тем, что
  // планировщик обещал сначала.
  const int plan_ngl0 = p.ngl0, plan_ctx0 = p.ctx0, plan_ncmoe = p.ncmoe,
            plan_cache = p.moe_cache;
  const bool plan_exps = p.exps_cpu, plan_q8 = p.kv_q8;
  const bool vision_note = !mmpath.empty();
  const bool pinned_ctx = o && o->ctx_size >= 256;
  for (int round = 0; round < 2 && !ok; ++round) {
    if (round > 0)
      logf("harness: «%s»: раунд 2 (демоция: %s)", id.c_str(),
           ex.draft_path.empty() && !dp.path.empty()
               ? tr("без draft")
               : "выходной слой на CPU");
    auto attempts =
        plan::build(smart, p.ngl0, p.ctx0, p.fits_all, p.moe, p.ram_ok, p.ncmoe, slow,
                    have_meta ? (long long)meta.n_layers : -1);
    // Лестница plan::build начинает безусловно с KV q8_0 + FA, но политика
    // сжатия (plan_for, как в пресете роутера) умеет выбирать f16 — «сжимать
    // нечего». Ребёнок с q8, который просто влез, ехал медленнее (0.6B:
    // 464,9 против 510,4 ток/с на f16, 09.10): первой попыткой делаем
    // несжатый вариант (тот же ngl/ctx без q8 — attempts[1]).
    if (smart && !p.kv_q8 && attempts.size() >= 2)
      attempts.erase(attempts.begin());
    // Шаг «та же форма без кэша»: грант — роскошь, форма — необходимость.
    // Если первая попытка (с кэшем) ловит OOM, повторяем её же без кэша
    // раньше, чем plan::build порежет слои (30.10: сброс гранта прыгал к
    // следующей ступени ngl 24 и уносил модель в RAM целиком — наблюдение
    // юзера: 4,7 ГиБ VRAM, 20 ток/с на форме, которая влезает целиком).
    if (p.moe_cache > 0 && !attempts.empty())
      attempts.insert(attempts.begin() + 1, attempts.front());
    for (size_t attempt = 0; attempt < attempts.size() && !ok; ++attempt) {
      const plan::Attempt& at = attempts[attempt];
      if (attempt > 0) {
        // Дисциплина logf: в varargs только const char* (жил потерянный %s
        // у «прогрев кэша»).
        const std::string ncmoe_note =
            at.ncmoe > 0 ? " экспертов в RAM " + std::to_string(at.ncmoe) : "";
        logf("harness: попытка %zu: ngl=%d ctx=%d%s%s%s%s%s", attempt + 1, at.ngl,
             at.ctx, at.fa ? " FA" : "", at.kv_q8 ? " KV q8_0" : "",
             at.exps_cpu ? " эксперты→CPU" : "", ncmoe_note.c_str(),
             at.warm ? " + прогрев кэша" : "");
      }
      auto c = std::make_shared<Child>();
      int port = 0, actual_ctx = 0;
      std::string key, aerr, tail;
      bool wedged = false;
      ex.ncmoe = at.ncmoe; // раскладка экспертов — на каждую попытку своя
      // Грант кэша — верхняя роскошь лестницы: первая попытка раунда 0 едет
      // с ним, деградации сбрасывают его раньше, чем резать слои — кэш без
      // формы бессмыслен, форма без кэша жива (30.10: лестница катилась
      // 48→24→12, неся грант 18,7 ГиБ на борту).
      ex.moe_cache_mib = round == 0 && attempt == 0 ? p.moe_cache : 0;
      // Доля экспертов на запасную карту (expert-card = spare) — полная форма
      // выноса (exps=CPU); частичную (ncmoe) не трогаем: её дележ — задача
      // флага, а раскладка «обе карты + ловушка» её смысл ломает.
      ex.expert_ot.clear();
      ex.expert_spare_idx = -1;
      int blocks = 0; // блоков экспертов на запасной карте в ЭТОЙ попытке
      if (at.exps_cpu && o && o->expert_card == "spare" && !tgt.split &&
          tgt.main_gpu >= 0) {
        const int spare = spare_card_index(target_cards(tgt), 768ull << 20);
        if (spare >= 0) {
          uint64_t free_b = 0;
          for (const auto& c : hw::query_all())
            if (c.nvidia && c.index == spare) { free_b = c.vram_free; break; }
          const uint64_t budget = free_b > (768ull << 20) ? free_b - (768ull << 20) : 0;
          uint64_t moved = 0;
          ex.expert_ot = expert_share_ot(meta, spare, budget, &blocks, &moved);
          if (blocks > 0) {
            ex.expert_spare_idx = spare;
            expert_spare_card = spare;
            expert_expect = moved;
            const hw::Gpu sc = settle_view(spare);
            expert_spare_before = sc.vram_total - sc.vram_free;
            logf("harness: «%s»: expert-card = spare — %d блоков экспертов "
                 "(%llu МиБ) на CUDA%d, остальные в RAM",
                 id.c_str(), blocks, (unsigned long long)(moved >> 20), spare);
          }
        }
      }
      ok = try_attempt(exe, path, at.ngl, at.ctx, at.fa, at.kv_q8, p.kv_type, at.exps_cpu,
                       cfg.ubatch, mmpath, try_target_of(tgt), ex, *c, port, key,
                       actual_ctx, aerr, &wedged, &tail);
      if (ok) {
        used = at;
        success_tail = c->log_tail(); // до warmup: строки разбивки загрузки ещё в кольце
        used_ot = expert_spare_card >= 0 ? blocks : 0; // landed blocks in this attempt
        if (at.warm) {
          warm_pages(path);
          bool cancelled = false;
          {
            std::lock_guard l(g_mtx);
            cancelled = g_slots[id].st != St::Loading;
          }
          if (cancelled) { c->stop(); ok = false; err = tr("загрузку отменили"); }
          if (!ok) break;
        }
        speed = warmup(port, key);
        bool cancelled = false;
        {
          std::lock_guard l(g_mtx);
          auto& slot = g_slots[id];
          cancelled = slot.st != St::Loading;
        }
        if (cancelled) { c->stop(); ok = false; err = tr("загрузку отменили"); }
        if (!ok) break;
        std::lock_guard l(g_mtx);
        auto& slot = g_slots[id];
        slot.child = c;
        slot.port = port;
        slot.api_key = key;
        // st остаётся Loading: Loaded слот получит только после записи профиля
        // и возврата KV (ниже) — иначе параллельный запрос попадает в замер
        // надбавки, а park_restore встречает занятый слот (движок откладывает
        // restore и мог бы перезаписать свежий кэш старым снимком).
        slot.ctx = actual_ctx;
        slot.used_ngl = at.ngl;
        slot.used_ctx = at.ctx;
        slot.np = ex.np;
        slot.smart = smart;
        slot.kv_q8 = at.kv_q8;
        slot.fa = at.fa;
        slot.exps_cpu = at.exps_cpu;
        slot.ncmoe = at.ncmoe;
        slot.warm = at.warm;
        slot.parkable = !ex.slot_save_path.empty();
        slot.cards = target_cards(tgt); // карты спавна — для по-карточного выселения
        slot.note = degrade_note(id, plan_ngl0, plan_ctx0, plan_exps, plan_ncmoe,
                                 plan_q8, plan_cache, at, ex.moe_cache_mib,
                                 (long long)meta.n_layers, vision_note, pinned_ctx);
        slot.loaded_at = slot.last_used = clock_::now();
        slot.err.clear();
        if (!slot.note.empty())
          logf("harness: «%s»: %s", id.c_str(), slot.note.c_str());
        break;
      }
      last_err = aerr;
      if (wedged) {
        // Клин драйвера: лестницу не продолжаем (07.10 это стоило ребута).
        fail(aerr);
        changed();
        return false;
      }
      if (aerr.find("не запустил") != std::string::npos ||
          aerr.find("порт") != std::string::npos) {
        fail(aerr); // системные причины, не память — лестница не поможет
        changed();
        return false;
      }
    }
    if (ok) break;
    // Раунд 2 — одна демоция за загрузку: сначала draft, потом выходной слой.
    if (round == 0 && !dp.path.empty()) {
      // Вина draft: падение загрузки головы или MTP-контекста часто выглядит
      // как OOM, но OOM там — буферы головы. Чистый OOM ствола draft не
      // чернит — но и вторым кругом с ним не едем (суспензия до конца
      // загрузки), чёрный список — до рестарта демона. Рамки — как в
      // все load-пути (502-буря 07.10).
      const auto dd = demote_draft(dp.path, last_err);
      if (!dd.dead)
        logf("harness: «%s»: ребёнок упал с draft «%s» — повторяю без спекуляций",
             id.c_str(), dp.path.c_str());
      ex.draft_path.clear();
      ex.draft_ngl = 0;
      ex.draft_device.clear();
      // Встроенный nextn — не ускорение, а условие загрузки: сняв файловую
      // голову, оставляем --spec-type draft-mtp, иначе движок не прочитает
      // nextn-тензоры («wrong number of tensors», b11476, 09.10).
      ex.spec_type = meta.nextn_predict_layers > 0 ? "draft-mtp" : std::string();
      continue;
    }
    if (round == 0 && have_meta && p.ngl0 > (int)meta.n_layers) {
      // Амбициозная попытка выходного слоя кончилась провалом всей лестницы —
      // тихо демотируем (профиль запомнит), раунд 2 с выходным на CPU.
      profiles::note_failed_out(id, {/*out_gpu=*/true, p.kv_q8, cfg.ubatch,
                                     engine::launches_cuda(cfg), p.exps_cpu, p.ncmoe});
      logf("harness: «%s»: выходной слой не влез — повторяю с выходным на CPU",
           id.c_str());
      p.ngl0 = (int)meta.n_layers;
      continue;
    }
    break;
  }

  if (ok) {
    // Факт раскладки на запасную карту: пин мог молча не применить
    // --device-draft или -ot-правило (b11934 с --device-draft так и делает,
    // проверено живьём 28.10). Замер памяти запасной карты против снимка
    // «до» — и честное предупреждение вместо немого no-op.
    if (draft_spare_card >= 0 && !ex.draft_device.empty()) {
      const hw::Gpu sc = settle_view(draft_spare_card);
      const uint64_t used = sc.vram_total - sc.vram_free;
      if (used + (dp.size / 2) < draft_spare_before + dp.size)
        logf("harness: «%s»: ПРЕДУПРЕЖДЕНИЕ: черновик на CUDA%d не доехал "
             "(%llu → %llu МиБ) — пин не применяет --device-draft, спекуляция "
             "живёт на карте ствола", id.c_str(), draft_spare_card,
             (unsigned long long)(draft_spare_before >> 20),
             (unsigned long long)(used >> 20));
    }
    if (expert_spare_card >= 0 && !ex.expert_ot.empty()) {
      const hw::Gpu sc = settle_view(expert_spare_card);
      const uint64_t used = sc.vram_total - sc.vram_free;
      if (used + (expert_expect / 2) < expert_spare_before + expert_expect)
        logf("harness: «%s»: ПРЕДУПРЕЖДЕНИЕ: блоки экспертов на CUDA%d не "
             "доехали (%llu → %llu МиБ) — пин не применил -ot-правило, все "
             "эксперты в RAM", id.c_str(), expert_spare_card,
             (unsigned long long)(expert_spare_before >> 20),
             (unsigned long long)(used >> 20));
    }
    {
      std::lock_guard l(g_draft_by_id_mtx);
      if (!dp.path.empty())
        g_draft_by_id[id] = {dp.path, dp.auto_attached};
      else
        g_draft_by_id.erase(id);
    }
    // Профиль по живой загрузке — тот же замер, что у роутера. Форма — та,
    // которой ребёнок реально заспавнен (лестница могла деградировать план:
    // недогруз слоёв, полный вынос экспертов), иначе замер ляжет не в свой
    // слот и следующий план поверит чужому следу.
    {
      SmartPlan rec = p;
      rec.ngl0 = used.ngl;
      rec.ctx0 = used.ctx;
      rec.kv_q8 = used.kv_q8;
      rec.exps_cpu = used.exps_cpu;
      rec.ncmoe = used.ncmoe;
      // Кэш/блоки — фактической формы: лестница могла деградировать план
      // (без выноса экспертов кэш не ехал вовсе), а блоки на запасной карте
      // считает правило попытки, не прогноз плана.
      rec.moe_cache = (used.exps_cpu || used.ncmoe > 0) ? ex.moe_cache_mib : 0;
      rec.ot_blocks = used_ot;
      const uint64_t gpu_w = gpu_base_of(rec, meta, fsize);
      record_profile(id, rec, meta, gpu_w + dp.size + mm_size, fsize - gpu_w, vram_before,
                     tgt.main_gpu, tgt.split, engine::launches_cuda(cfg), cfg.ubatch,
                     success_tail);
      const int layers = (int)meta.n_layers;
      const bool partial = rec.ngl0 > 0 && rec.ngl0 < layers;
      profiles::Form spd{rec.ngl0 > layers, rec.kv_q8, cfg.ubatch, engine::launches_cuda(cfg),
                         rec.exps_cpu, rec.ncmoe, partial ? rec.ngl0 : 0,
                         rec.moe_cache, used_ot};
      profiles::note_speed(id, spd, speed.tps, speed.n);
    }
    note_model_used(path);
    logf("harness: готово: «%s» на 127.0.0.1:%d", id.c_str(), [&] {
      std::lock_guard l(g_mtx);
      return g_slots[id].port;
    }());
    // Возврат KV припаркованных слотов: после записи профиля (иначе
    // восстановленная KV попала бы в дельту надбавки). Слот ещё Loading —
    // будилка порт не раздаёт, restore не встретит занятый слот. Отказы
    // движка штатны: слот очищен, работает холодным.
    const bool parked_any = park_restore_after_load(id);
    {
      // Loaded — самый последний шаг: профиль записан, KV возвращена.
      // wake_notify() в конце load_harness — будильник очереди пробуждения.
      std::lock_guard l(g_mtx);
      auto it = g_slots.find(id);
      if (it != g_slots.end() && it->second.st == St::Loading) {
        it->second.st = St::Loaded;
        it->second.last_used = clock_::now();
      }
    }
    save_resident_state(); // Loaded-слот теперь в резидентории
    // Проба соседней формы KV останавливает и пересаживает ребёнка —
    // вместе с только что возвращённой KV; не назначаем её после парковки.
    if (!parked_any && kv_force_of(o, id) == 0 && speed.n >= 128 && have_meta) {
      const bool sib_q8 = !used.kv_q8;
      const int layers = (int)meta.n_layers;
      const bool out_gpu = used.ngl > layers;
      const int ngl_key = (used.ngl > 0 && used.ngl < layers) ? used.ngl : 0;
      profiles::Form sib{out_gpu, sib_q8, cfg.ubatch, engine::launches_cuda(cfg),
                         used.exps_cpu, used.ncmoe, ngl_key,
                         (used.exps_cpu || used.ncmoe > 0) ? ex.moe_cache_mib : 0,
                         used_ot};
      hw::Gpu fitg = tgt.card;
      if (fitg.vram_total > 0) fitg.vram_free = fitg.vram_total - kPlanReserve;
      const bool sib_fits =
          fit::estimate(meta, fsize, (uint32_t)std::max(used.ctx, 512), fitg, sib_q8).fits;
      int live_port = 0;
      {
        std::lock_guard l(g_mtx);
        auto it = g_slots.find(id);
        if (it != g_slots.end()) live_port = it->second.port;
      }
      if (!profiles::speed_settled(id, sib) && sib_fits && live_port > 0) {
        logf("harness: «%s»: проба соседней формы KV отложена до простоя", id.c_str());
        schedule_kv_probe(id, path, exe, mmpath, p.kv_type, meta, fsize, tgt, ex, used.ngl,
                          used.ctx, cfg.ubatch, live_port, used.ncmoe, used.fa, used.kv_q8,
                          used.exps_cpu, engine::launches_cuda(cfg));
      }
    }
  } else {
    fail(tr("не удалось загрузить модель: ") + last_err +
         fail_advice(id, vision_note, pinned_ctx));
  }
  changed();
  wake_notify(); // раскладка памяти изменилась — ждущие пересчитаются
  return ok;
}

// Будилка harness: резидентную — сразу; спящую/отсутствующую — load_harness
// (выселение незанятых внутри); упёрлись в занятых — парковка в очереди
// пробуждения с TTL. parked_ms_out — сколько запрос просидел в очереди.
WakeResult harness_wake(const std::string& want, int timeout_ms,
                        long long* parked_ms_out, bool allow_off_registry) {
  if (parked_ms_out) *parked_ms_out = 0;
  WakeResult out;
  const auto deadline = clock_::now() + std::chrono::milliseconds(timeout_ms);
  auto resolve = [&]() -> std::pair<std::string, std::string> {
    if (want.find(".gguf") != std::string::npos) {
      std::lock_guard l(g_mtx);
      for (auto& [i, p] : g_id_map)
        if (path_eq(p, want)) return {i, want};
      // Путь части 2..N — та же модель. Грузим первую часть, её id уже в реестре.
      if (lh::modelreg::is_shard_continuation(want)) {
        std::string base = lh::modelreg::shard_base(want);
        for (auto& [i, p] : g_id_map) {
          int idx = 0, cnt = 1;
          lh::modelreg::shard_info(p, idx, cnt);
          if (idx == 1 && cnt > 1 && lh::modelreg::shard_base(p) == base)
            return {i, p};
        }
      }
      // Путь вне реестра (недокачан/положен только что) — только с админского
      // входа (sup::load): /v1-клиент с именованным токеном не должен
      // превращать поле «model» в произвольный путь, который демон прочитает
      // и положит в VRAM. Сюда же не попадают фантомные Error-слоты: id из
      // имени файла жил бы в g_slots вечно.
      if (!allow_off_registry) return {{}, {}};
      std::string base = want;
      size_t slash = base.find_last_of("/\\");
      if (slash != std::string::npos) base = base.substr(slash + 1);
      if (base.size() > 5 && strcasecmp(base.c_str() + base.size() - 5, ".gguf") == 0)
        base.erase(base.size() - 5);
      std::map<std::string, std::string> taken;
      for (auto& [i, p] : g_slots) taken[i] = p.path;
      return {make_unique_id(sanitize_id(base), taken), want};
    }
    std::lock_guard l(g_mtx);
    auto it = g_id_map.find(want);
    if (it != g_id_map.end()) return {want, it->second};
    // Точного id нет: принимаем и стем файла, и групповое имя шард-группы —
    // те же ключи, что в /v0/load и в секциях переопределений.
    for (auto& [i, p] : g_id_map) {
      std::string stem = lh::modelreg::registry_stem(p);
      std::string file = p;
      size_t slash = file.find_last_of("/\\");
      if (slash != std::string::npos) file = file.substr(slash + 1);
      if (file.size() > 5 && strcasecmp(file.c_str() + file.size() - 5, ".gguf") == 0)
        file.erase(file.size() - 5);
      if (strcasecmp(stem.c_str(), want.c_str()) == 0 ||
          strcasecmp(file.c_str(), want.c_str()) == 0)
        return {i, p};
    }
    return {{}, {}};
  };
  // Сериализация будилок: параллельные чат-запросы к одной модели иначе
  // спавнили бы детей вдвоём (у роутера тот же приём — g_wake_mtx).
  static std::mutex g_wake_serial_mtx;
  const void* ticket = nullptr;
  uint64_t gen = 0;
  clock_::time_point parked_at{}, ttl_deadline{};
  std::string first_blockers;
  std::string lot_key; // ключ, под которым занято место в очереди
  auto unpark = [&](const std::string& id) {
    if (!ticket) return;
    g_wake_lot.leave(ticket);
    ticket = nullptr;
    metrics::queue_depth(id, (int)g_wake_lot.depth(id));
  };
  // Гарантированное освобождение места: выходы, идущие мимо unpark
  // (модель исчезла из реестра, пока запрос стоял в очереди; истёк дедлайн
  // ожидания чужой загрузки), без гарда оставляли тикет живым навсегда —
  // per_key_ не обнулялся («очередь полна» для всех будущих парковок этой
  // модели), а waiters_ рос и замедлял leave/wait сканом O(n).
  struct LeaveTicket {
    const void*& t;
    std::string& key;
    ~LeaveTicket() {
      if (!t) return;
      g_wake_lot.leave(t);
      t = nullptr;
      if (!key.empty()) metrics::queue_depth(key, (int)g_wake_lot.depth(key));
    }
  } leave_ticket{ticket, lot_key};
  for (;;) {
    auto [id, path] = resolve();
    if (path.empty()) {
      out.not_found = true;
      out.err = tr("модель «") + want + tr("» не найдена в реестре");
      return out;
    }
    bool neighbor_loading = false;
    {
      std::lock_guard l(g_mtx);
      auto it = g_slots.find(id);
      if (it != g_slots.end()) {
        if (it->second.st == St::Loaded && it->second.child && it->second.child->alive()) {
          it->second.last_used = clock_::now();
          out.ok = true;
          unpark(id);
          return out;
        }
        neighbor_loading =
            it->second.st == St::Loading || it->second.st == St::Stopping;
      }
    }
    if (neighbor_loading) {
      if (clock_::now() > deadline) {
        out.err = tr("истекло ожидание загрузки модели");
        return out;
      }
      unpark(id);
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      continue;
    }

    enum class Step { Fail, Park, Done };
    Step step = Step::Fail;
    {
      std::lock_guard wake_l(g_wake_serial_mtx);
      // Перепроверка под замком: сосед мог загрузить, пока мы ждали очереди.
      bool loaded_now = false;
      {
        std::lock_guard l(g_mtx);
        auto it = g_slots.find(id);
        loaded_now = it != g_slots.end() && it->second.st == St::Loaded &&
                     it->second.child && it->second.child->alive();
        if (loaded_now) it->second.last_used = clock_::now();
        if (!loaded_now && it != g_slots.end() &&
            (it->second.st == St::Loading || it->second.st == St::Stopping))
          neighbor_loading = true;
      }
      if (loaded_now) {
        out.ok = true;
        unpark(id);
        return out;
      }
      if (neighbor_loading) {
        step = Step::Park; // чужая загрузка: подождём снаружи, без паркинга
      } else {
        std::vector<std::string> blocked;
        std::string lerr;
        // Гейт «загрузка/установка не совпадают»: билет на саму попытку
        // загрузки; парковка в очереди установку не держит (см. engine.h).
        engine::GateTicket gate;
        if (!gate.acquire(engine::GateTicket::Kind::Load, lerr)) {
          out.err = lerr;
          unpark(id);
          return out;
        }
        if (load_harness(path, id, lerr, &blocked)) {
          out.ok = true;
          unpark(id);
          return out;
        }
        if (blocked.empty()) {
          out.err = lerr;
          unpark(id);
          return out;
        }
        // Занятые резиденты — парковка (TTL/глубина из конфига, оба горячие).
        config::Config cfg = config::current();
        std::string bl = join_ids(blocked, ", ");
        if (cfg.queue_ttl <= 0) {
          out.err = tr("память занята резидентами (") + bl + tr("), очередь пробуждения выключена");
          unpark(id);
          return out;
        }
        if (!ticket && g_wake_lot.depth(id) >= (size_t)cfg.queue_depth) {
          out.err = tr("очередь пробуждения модели «") + id + "» полна (" +
                    std::to_string(cfg.queue_depth) + " ждущих) — повторите позже";
          unpark(id);
          return out;
        }
        if (!ticket) {
          ticket = g_wake_lot.enter(id);
          if (!ticket) {
            out.err = tr("очередь пробуждения полна — повторите позже");
            return out;
          }
          lot_key = id;
          gen = g_wake_lot.generation_snapshot();
          parked_at = clock_::now();
          ttl_deadline = parked_at + std::chrono::seconds(cfg.queue_ttl);
          first_blockers = bl;
          metrics::queue_depth(id, (int)g_wake_lot.depth(id));
          logf("harness: «%s» паркуется: ждёт освобождения занятых (%s), TTL %d с",
               id.c_str(), bl.c_str(), cfg.queue_ttl);
        }
        if (clock_::now() >= ttl_deadline) {
          metrics::queue_timeout(id);
          out.queued_timeout = true;
          out.err = tr("очередь пробуждения: истекло ожидание освобождения памяти (") +
                    first_blockers + ")";
          unpark(id);
          return out;
        }
        step = Step::Park;
      }
    }
    if (step == Step::Park) {
      // Спим вне сериализующего мьютекса: будильник — изменение занятости
      // (деструктор BusyModel), выселение слота или истечение TTL.
      g_wake_lot.wait(ticket, ttl_deadline, gen);
      if (parked_ms_out)
        *parked_ms_out = std::chrono::duration_cast<std::chrono::milliseconds>(
                             clock_::now() - parked_at).count();
      continue;
    }
    return out;
  }
}


} // namespace lh::sup::detail
