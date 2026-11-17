#include "sup_internal.h"

#include "modelreg.h"
#include "profiles.h"
#include "util.h"

#include <strings.h>

#include <algorithm>

#include <string>
#include <vector>

namespace lh::sup::detail {

// Draft-файлы, которые движок не смог загрузить (например, MTP-модуль свежей
// архитектуры, не переваренный пином): с ними модель негрузима в принципе.
// Спекуляция по ним отключается до рестарта приложения — загрузка важнее.
// Читается find_draft'ом из нескольких потоков (вотчер, параллельные загрузки)
// и пополняется будилкой (load_harness) — под своим мьютексом.
std::set<std::string> g_dead_drafts;
std::mutex g_dead_drafts_mtx;
// Draft'ы, отстранённые сторожем acceptance: приём черновиков устойчиво ниже
// floor (llmhost.conf [general] draft-accept-floor) — спекуляция не окупается,
// грузим без неё до рестарта демона. Ручной model-draft сторож не трогает
// (переопределения сильнее умной настройки) — только предупреждает в журнал.
std::set<std::string> g_shy_drafts;
std::mutex g_shy_drafts_mtx;
// Draft, снятый с ОДНОЙ текущей загрузки (чистый OOM ствола: draft не виноват
// и чёрный список ему не писан — «с вернувшейся VRAM должен жить», — но и
// повторять его вторым кругом бессмысленно, footprint меньше без него).
// Стирается в конце каждой загрузки (RAII в load_harness).
std::set<std::string> g_draft_suspended;
std::mutex g_draft_suspended_mtx;
// Черновик, прицепленный к модели пресетом: id → {путь, auto}. Источник истины
// для сторожа; пишется планировщиком (там же и стирается при отсутствии).
std::map<std::string, std::pair<std::string, bool>> g_draft_by_id;
std::mutex g_draft_by_id_mtx;

// Пути сравниваем после нормализации слэшей. На Linux ФС чувствительна к
// регистру — сравнение байтовое; реестр выдаёт канонические пути.
bool path_eq(const std::string& a, const std::string& b) {
  auto fold = [](std::string s) {
    for (char& c : s)
      if (c == '\\') c = '/';
    return s;
  };
  return fold(a) == fold(b);
}

// Draft-модель по имени: general.name, стем файла или путь — как в /v0/load.
// Пусто в out — не нашли/сам на себя (спекуляция сама с собой бессмысленна).
bool resolve_draft(const std::string& want, const std::vector<lh::modelreg::Model>& list,
                   const std::string& main_path, std::string& out, uint64_t& out_size,
                   int& out_layers) {
  if (want.empty()) return false;
  auto stem_of = [](const std::string& p) {
    std::string f = p;
    size_t s = f.find_last_of("/\\");
    if (s != std::string::npos) f = f.substr(s + 1);
    size_t d = f.find_last_of('.');
    if (d != std::string::npos) f = f.substr(0, d);
    return f;
  };
  for (auto& m : list) {
    bool by_name = m.name == want;
    bool by_stem = strcasecmp(stem_of(m.path).c_str(), want.c_str()) == 0;
    bool by_path = path_eq(m.path, want);
    if (!by_name && !by_stem && !by_path) continue;
    if (path_eq(m.path, main_path)) return false; // сам на себя
    out = m.path;
    out_size = m.size;
    out_layers = (int)m.meta.n_layers;
    return true;
  }
  return false;
}

// Найти draft к основной модели. Автоматом — только младшая модель той же
// архитектуры (≥64 МиБ, основная ≥4× крупнее, тип draft-simple): механизм
// проверенный. MTP-модули репозиториев — только вручную («model-draft» +
// «spec-type»: «draft-mtp» в overrides): авто-правило по arch их не видит
// (arch у модуля чужой), хотя сама технология на b11223 работает отлично —
// замер 30.09: gemma-4-12b + её mtp-модуль НА GPU = 45,0 против 35,4 ток/с
// (+27%, через пресет 44,9). Прежний замер «1,1 ток/с» был артефактом
// раскладки: прямой спавн без явного gpu-layers-draft — авто-фит черновика
// молча клал модуль на CPU. llmhost всегда пишет gpu-layers-draft, через
// overrides модуль ложится на карту.
// Замечание: роутер b11223 ключи из пресета пробрасывает в ребёнка целиком
// (проверено живьём 30.09: model-draft, spec-type, gpu-layers-draft,
// spec-draft-n-max видны в args ребёнка), и спекуляция сквозь пресет
// работает: draft-simple пара Qwen3-30B-A3B + Qwen3-0.6B (qwen3moe + qwen3 —
// токенайзер общий, авто-правило по arch её не находит) = 17,8 против
// 13,9 ток/с.
DraftPick find_draft(const std::string& main_path, const lh::gguf::Meta& meta,
                     uint64_t fsize, const config::Config& cfg, const ModelOverrides* o,
                     const std::vector<lh::modelreg::Model>& list) {
  // Структурная парность головы со стволом по GGUF-мете — условие того самого
  // GGML_ASSERT «MTP input row width must match the target h_nextn width»:
  // ширина эмбеддинга головы обязана совпадать со стволом, словарь — тем более
  // (логиты головы едут в токены ствола). Проверяются только когда оба значения
  // известны; каталоговая эвристика «mtp-* рядом» — лишь поверх. До неё в
  // плоской папке моделей (модели и чужие головы вперемешку) эвристика цепляла
  // первую попавшуюся голову — ребёнок умирал на загрузке, failover закольцовы-
  // вался, клиент ловил 502-бурю (07.10, 14 эпизодов).
  auto vocab_mismatch = [&](const lh::gguf::Meta& dm) {
    return dm.n_vocab && meta.n_vocab && dm.n_vocab != meta.n_vocab;
  };
  DraftPick out;
  // Пер-модельный mtp-mode (переопределение) сильнее глобального [general] и
  // сильнее ручного model-draft: off — детерминированное «спекуляции нет».
  const std::string mtp_mode = o && !o->mtp_mode.empty() ? o->mtp_mode : cfg.mtp_mode;
  if (mtp_mode == "off") {
    logf("find_draft: %s: mtp-mode = off — без спекуляций",
         main_path.substr(main_path.find_last_of("/\\") + 1).c_str());
    return out;
  }
  if (o && !o->model_draft.empty()) {
    std::string p;
    uint64_t s = 0;
    int layers = 0;
    if (resolve_draft(o->model_draft, list, main_path, p, s, layers)) {
      bool dead;
      {
        std::lock_guard l(g_dead_drafts_mtx);
        dead = g_dead_drafts.count(p) > 0;
      }
      if (dead) {
        logf("find_draft: «%s» в чёрном списке (движок не смог его загрузить) — без спекуляций",
             p.c_str());
      } else {
        for (auto& m : list)
          if (path_eq(m.path, p)) {
            const bool by_embd = m.meta.n_embd && meta.n_embd && m.meta.n_embd != meta.n_embd;
            const uint64_t got = by_embd ? m.meta.n_embd : m.meta.n_vocab;
            const uint64_t want = by_embd ? meta.n_embd : meta.n_vocab;
            // Ручной ключ не трогаем (переопределения сильнее умной настройки),
            // но факт несовместимости называем честно: движок отвергнет голову
            // на загрузке, и приговор переживёт рестарт (профиль).
            if (by_embd)
              logf("find_draft: ручной draft «%s» несовместим со стволом по ширине "
                   "эмбеддинга (%llu против %llu) — движок его отвергнет, ключ "
                   "model-draft не трогаю", p.c_str(),
                   (unsigned long long)got, (unsigned long long)want);
            else if (m.meta.n_vocab && meta.n_vocab && m.meta.n_vocab != meta.n_vocab)
              logf("find_draft: ручной draft «%s» несовместим со стволом по словарю "
                   "(%llu против %llu) — спекуляция будет мусорной, ключ "
                   "model-draft не трогаю", p.c_str(),
                   (unsigned long long)got, (unsigned long long)want);
            out.n_vocab = m.meta.n_vocab;
            out.n_embd = m.meta.n_embd;
            break;
          }
        out.path = p;
        out.size = s;
        out.n_layers = layers;
        out.spec_type = o->spec_type.empty() ? "draft-simple" : o->spec_type;
      }
    }
    return out; // не разрешился — авто не подменяем, предупредит вызывающий
  }
  // Встроенный MTP: nextn-тензоры лежат в самом GGUF (Qwen3.6/3.5-MTP, GLM-4.5),
  // отдельного draft-файла нет — ручной ключ spec-type без model-draft.
  if (o && o->model_draft.empty() && o->spec_type == "draft-mtp") {
    out.spec_type = "draft-mtp";
    return out;
  }
  // Встроенный nextn сильнее любой авто-головы: обучен вместе с этим стволом,
  // а внешнему файлу уступает ещё и потому, что для nextn-модели spec-type
  // draft-mtp обязателен (без него движок не читает nextn-тензоры и падает
  // «wrong number of tensors», b11476). Решение — здесь, а не в каждом
  // вызове: все load-пути получают одно поведение
  // (раньше третий вызов перекрытия не имел). Ручной model-draft выше уже
  // вернулся — переопределения сильнее умной настройки.
  if (cfg.auto_draft && meta.nextn_predict_layers > 0) {
    logf("find_draft: %s: встроенный MTP (nextn_predict_layers=%u) — спекуляции "
         "без отдельного draft-файла",
         main_path.substr(main_path.find_last_of("/\\") + 1).c_str(),
         (unsigned)meta.nextn_predict_layers);
    out.spec_type = "draft-mtp";
    return out;
  }
  // Родной MTP-модуль. Два места: поддерево папки файла и папка MTP/ на
  // уровень выше. Второе — раскладка скачивания: квант лежит в своей
  // подпапке репозитория, голова — в MTP/ рядом с ней. Обученный вместе
  // с моделью драфтер сильнее «младшей модели семьи» — прицепляем
  // автоматически; нет модуля — модель работает без спекуляций. Сеть не
  // трогаем: пресет пишет вотчер. Провал по месту ловит лестница загрузки
  // (повтор без спекуляций).
  if (cfg.mtp_mode != "off") {
    // Сначала ширина ствола, потом несовпадающая (родные головы gemma несут
    // свою проекцию: 1024 против 3840 у ствола, живьём работает 09.10).
    // Внутри каждого прохода ближняя папка важнее MTP/ уровнем выше. Модуль
    // вяжется к своему стволу по имени (mtp_matches_trunk): в плоской папке
    // близость — не родство, чужая голова с общим словарём проходит мета-
    // фильтры и роняет ребёнка. Словарь обязателен всегда: логиты головы
    // едут в токены ствола.
    std::string rejected; // несовместимые головы — одной строкой на модель
    bool width_logged = false;
    for (int pass = 0; pass < 2 && out.path.empty(); ++pass) {
      for (int loc = 0; loc < 2 && out.path.empty(); ++loc) {
      for (auto& m : list) {
        if (!lh::modelreg::is_mtp_module(m.path)) continue;
        if (lh::modelreg::mtp_place(main_path, m.path) != loc + 1) continue;
        if (!lh::modelreg::mtp_matches_trunk(main_path, m.path)) continue;
        if (vocab_mismatch(m.meta)) {
          if (pass == 0) {
            size_t b = m.path.find_last_of("/\\");
            if (!rejected.empty()) rejected += ", ";
            rejected += b == std::string::npos ? m.path : m.path.substr(b + 1);
          }
          continue;
        }
        const bool width_ok =
            !m.meta.n_embd || !meta.n_embd || m.meta.n_embd == meta.n_embd;
        if (pass == 0 && !width_ok) continue; // несовпадающие — во втором проходе
        if (!width_ok && !width_logged) {
          width_logged = true;
          logf("find_draft: %s: ширина головы %llu против ствола %llu — у MTP-головы "
               "своя проекция, прицепляю (совпадающих по ширине голов рядом нет)",
               m.path.c_str(), (unsigned long long)m.meta.n_embd,
               (unsigned long long)meta.n_embd);
        }
        {
          std::lock_guard l(g_dead_drafts_mtx);
          if (g_dead_drafts.count(m.path)) continue;
        }
        {
          std::lock_guard l(g_shy_drafts_mtx);
          if (g_shy_drafts.count(m.path)) continue; // сторож отстранил: приёма нет
        }
        {
          std::lock_guard l(g_draft_suspended_mtx);
          if (g_draft_suspended.count(m.path)) continue; // снят с текущей загрузки
        }
        out.path = m.path;
        out.size = m.size;
        out.n_layers = (int)m.meta.n_layers;
        out.n_vocab = m.meta.n_vocab;
        out.n_embd = m.meta.n_embd;
        out.spec_type = "draft-mtp";
        out.auto_attached = true;
        break;
      }
      }
    }
    if (out.path.empty() && !rejected.empty())
      logf("find_draft: рядом есть MTP-модули (%s), но ни один не совместим со "
           "стволом по мете — без спекуляций", rejected.c_str());
  }
  if (!cfg.auto_draft || !meta.n_layers) return out;

  // Младшая модель той же архитектуры (Qwen3-0.6B при Qwen3-8B и т.п.).
  for (auto& m : list) {
    if (path_eq(m.path, main_path) || !m.meta.n_layers || m.arch != meta.arch) continue;
    // MTP-модуль — не standalone-модель: как draft-simple он не грузится
    // (llama_model_load падает), свой путь у него — draft-mtp через overrides.
    if (lh::modelreg::is_mtp_module(m.path)) continue;
    // Хвост шард-группы — не младшая модель. Размер кандидата — вся группа.
    if (lh::modelreg::is_shard_continuation(m.path)) continue;
    int ci = 0, cc = 1;
    lh::modelreg::shard_info(m.path, ci, cc);
    const uint64_t cand =
        cc > 1 ? lh::modelreg::shard_group_bytes(m.path) : m.size;
    {
      std::lock_guard l(g_shy_drafts_mtx);
      if (g_shy_drafts.count(m.path)) continue; // сторож отстранил: приёма нет
    }
    {
      std::lock_guard l(g_draft_suspended_mtx);
      if (g_draft_suspended.count(m.path)) continue; // снят с текущей загрузки
    }
    if (cand < 64ull << 20 || fsize < cand * 4) continue;
    if (out.path.empty() || cand < out.size) {
      out.path = m.path;
      out.size = cand;
      out.n_layers = (int)m.meta.n_layers;
      out.n_vocab = m.meta.n_vocab;
      out.n_embd = m.meta.n_embd;
      out.spec_type = "draft-simple";
      out.auto_attached = true;
    }
  }
  return out;
}

// Подсказка draft-кандидата: та же архитектура, заметно меньше — классика
// «0.6B тянет за собой всю семью». Идентификатор (стем файла) годится для
// llmhost.conf как есть.
// Оценка KV-кэша черновика при размещении на запасной карте: 2 (K и V) ×
// слои × ширина × 2 байта f16 × контекст. Ширина — сверху (GQA даёт меньше),
// мета не читалась — консервативный минимум, чтобы не сажать черновик на
// карту «наугад».
uint64_t draft_kv_bytes(const DraftPick& dp, int ctx) {
  const uint64_t layers = dp.n_layers ? (uint64_t)dp.n_layers : 8;
  const uint64_t embd = dp.n_embd ? dp.n_embd : 1024;
  return 2ull * layers * embd * 2ull * (uint64_t)std::max(ctx, 512);
}

DraftDemotion demote_draft(const std::string& draft_path, const std::string& child_log) {
  DraftDemotion d;
  d.oom = lh::proc::looks_oom(child_log);
  d.blamed = lh::proc::looks_draft_failure(child_log);
  bool newly_dead = false;
  {
    std::lock_guard l(g_dead_drafts_mtx);
    d.dead = g_dead_drafts.count(draft_path) > 0;
    newly_dead = !d.dead && (!d.oom || d.blamed);
    if (newly_dead) g_dead_drafts.insert(draft_path);
  }
  if (newly_dead)
    // Переживёт рестарт демона: иначе «движок не смог его загрузить»
    // переоткрывается каждой первой загрузкой после рестарта.
    lh::profiles::note_bad_draft(draft_path, "dead");
  if (!d.dead && d.oom && !d.blamed) {
    std::lock_guard l(g_draft_suspended_mtx);
    g_draft_suspended.insert(draft_path);
  }
  return d;
}

// Подсказка draft-кандидата: та же архитектура, заметно меньше — классика
// «0.6B тянет за собой всю семью». Идентификатор (стем файла) годится для
// llmhost.conf как есть.
std::string draft_hint_for(const lh::gguf::Meta& meta, uint64_t fsize,
                           const std::vector<lh::modelreg::Model>& list) {
  if (!meta.n_layers || !fsize) return {};
  std::string best;
  uint64_t best_size = 0;
  for (auto& m : list) {
    if (!m.meta.n_layers || m.arch != meta.arch) continue;
    if (lh::modelreg::is_shard_continuation(m.path)) continue;
    int ci = 0, cc = 1;
    lh::modelreg::shard_info(m.path, ci, cc);
    const uint64_t cand =
        cc > 1 ? lh::modelreg::shard_group_bytes(m.path) : m.size;
    if (cand >= fsize || cand < 64ull << 20) continue;
    if (!best_size || cand < best_size) { best_size = cand; best = m.name; }
  }
  return best;
}

void seed_draft_reputation() {
  std::vector<std::string> bad = lh::profiles::bad_drafts("dead");
  {
    std::lock_guard l(g_dead_drafts_mtx);
    g_dead_drafts.insert(bad.begin(), bad.end());
  }
  bad = lh::profiles::bad_drafts("shy");
  {
    std::lock_guard l(g_shy_drafts_mtx);
    g_shy_drafts.insert(bad.begin(), bad.end());
  }
}

} // namespace lh::sup::detail
