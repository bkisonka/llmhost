#include "control.h"
#include "i18n.h"
#include "webui.h"

#include "bench.h"
#include "catalog.h"
#include "catalog_alias.h"
#include "config.h"
#include "engine.h"
#include "fit.h"
#include "gguf.h"
#include "hw.h"
#include "image.h"
#include "metrics.h"
#include "modelreg.h"
#include "pull.h"
#include "quit.h"
#include "supervisor.h"
#include "tokens.h"
#include "util.h"

#include <future>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <set>
#include <strings.h>
#include <thread>

using lh::i18n::tr;

namespace lh::control {

namespace {

void json(httplib::Response& res, const nlohmann::json& j, int status = 200) {
  res.status = status;
  res.set_content(j.dump(2), "application/json");
}

void err(httplib::Response& res, int status, const std::string& msg) {
  // Тот же стабильный конверт, что у OpenAI-прокси: {"error":{message,type,code}}.
  const char* type = status == 401 ? "authentication_error"
                   : status < 500 ? "invalid_request_error"
                                  : "api_error";
  json(res, {{"error", {{"message", msg}, {"type", type}, {"code", status}}}}, status);
}

// Строковое поле тела: чужой тип — не повод для 500 nlohmann-исключением,
// это ошибка клиента (helper + non_string ниже ловят её до value()).
std::string sfield(const nlohmann::json& b, const char* k) {
  if (!b.is_object()) return {};
  auto it = b.find(k);
  if (it == b.end() || !it->is_string()) return {};
  return it->get<std::string>();
}

// true — поле есть, но не строка (и не null): ответ 400 уже записан.
bool non_string(const nlohmann::json& b, const char* k, httplib::Response& res) {
  if (!b.is_object()) return false;
  auto it = b.find(k);
  if (it == b.end() || it->is_string() || it->is_null()) return false;
  err(res, 400, std::string("\"") + k + "\" должен быть строкой");
  return true;
}

// Обёртка: проверка токена + перехват исключений на каждом маршруте.
// Control API всегда за токеном; токен не создался — fail-closed (503),
// а не «пускаем всех».
httplib::Server::Handler auth_wrap(httplib::Server::Handler h) {
  return [h = std::move(h)](const httplib::Request& req, httplib::Response& res) {
    switch (config::check_bearer(req.get_header_value("Authorization"))) {
      case config::BearerVerdict::Ok:
        break;
      case config::BearerVerdict::Wrong:
        err(res, 401, tr("нужен Bearer-токен"));
        return;
      case config::BearerVerdict::NoToken:
        err(res, 503, "Bearer-токен не создался — API закрыт (подробности в журнале)");
        return;
    }
    try {
      h(req, res);
    } catch (const std::exception& e) {
      // e.what() — в журнал (для админа путь к истине), клиенту — общая фраза:
      // text исключений nlohmann/fs содержит фрагменты путей и разбора.
      lh::logf("control: %s %s — внутренняя ошибка: %s", req.method.c_str(),
               req.target.c_str(), e.what());
      err(res, 500, tr("внутренняя ошибка — подробности в журнале демона"));
    }
  };
}

// Тело запроса → JSON: пустое тело читается как «{}»; strict — не-JSON даёт
// 400 и nullptr (вызывающий выходит), иначе — пустой объект. Раньше этот
// бойлерплейт был отрукописан 14 раз с расхождениями: часть эндпоинтов
// отвечала 400 на пустое тело, часть молча терпела.
nlohmann::json body_json(const httplib::Request& req, httplib::Response& res,
                         bool strict) {
  nlohmann::json body;
  try {
    body = nlohmann::json::parse(req.body.empty() ? "{}" : req.body);
  } catch (...) {
    if (strict) {
      err(res, 400, tr("тело запроса — не JSON"));
      return nlohmann::json();
    }
    body = nlohmann::json::object();
  }
  return body;
}

// Резолв «path» или «name» из тела запроса в путь модели (как в /v0/load).
// При сбое ответ уже записан (400/404) — вызывающему достаточно выйти.
std::string resolve_model_arg(const nlohmann::json& body, httplib::Response& res) {
  std::string path = sfield(body, "path");
  if (path.empty() && body.contains("name")) {
    std::string want = sfield(body, "name");
    auto list = modelreg::scan(config::models_dir_resolved(config::current()));
    const modelreg::Model* tail = nullptr;
    for (auto& m : list) {
      std::string file = m.path;
      size_t slash = file.find_last_of("/\\");
      if (slash != std::string::npos) file = file.substr(slash + 1);
      size_t dot = file.find_last_of('.');
      if (dot != std::string::npos) file = file.substr(0, dot);
      std::string stem = modelreg::registry_stem(m.path);
      bool hit = m.name == want || strcasecmp(file.c_str(), want.c_str()) == 0 ||
                 strcasecmp(stem.c_str(), want.c_str()) == 0;
      if (!hit) continue;
      // Имя части 2..N — всё равно эта модель; грузить надо первую часть.
      if (modelreg::is_shard_continuation(m.path)) {
        if (!tail) tail = &m;
        continue;
      }
      path = m.path;
      break;
    }
    if (path.empty() && tail) {
      std::string base = modelreg::shard_base(tail->path);
      for (auto& m : list) {
        int i = 0, c = 1;
        modelreg::shard_info(m.path, i, c);
        if (i == 1 && c > 1 && modelreg::shard_base(m.path) == base) {
          path = m.path;
          break;
        }
      }
      if (path.empty()) path = tail->path;
    }
    if (path.empty()) {
      err(res, 404, tr("модель «") + want + tr("» не найдена в реестре"));
      return {};
    }
  }
  if (path.empty()) err(res, 400, tr("укажите \"path\" или \"name\""));
  // Явный путь части 2..N — всё равно эта модель: движку нужна первая часть.
  if (modelreg::is_shard_continuation(path)) {
    std::string base = modelreg::shard_base(path);
    for (auto& m : modelreg::scan(config::models_dir_resolved(config::current()))) {
      int i = 0, c = 1;
      modelreg::shard_info(m.path, i, c);
      if (i == 1 && c > 1 && modelreg::shard_base(m.path) == base) {
        path = m.path;
        break;
      }
    }
  }
  return path;
}

nlohmann::json models_json() {
  config::Config cfg = config::current();
  hw::Gpu gpu = hw::query();
  auto list = modelreg::scan(config::models_dir_resolved(cfg));
  // Живое состояние моделей знает супервизор; один раз.
  nlohmann::json live = nlohmann::json();
  live = sup::models_state();
  nlohmann::json arr = nlohmann::json::array();
  std::string loaded_path;
  sup::state(loaded_path);
  for (auto& m : list) {
    // MTP-модуль и vision-проекция — не самостоятельные модели: первый
    // прицепляется черновиком к своей модели, вторая прикладывается через
    // --mmproj. В списки моделей они не попадают.
    if (modelreg::is_mtp_module(m.path)) continue;
    if (modelreg::is_mmproj_module(m.path)) continue;
    // Части 2..N — та же модель. В списке одна запись, путь — первая часть.
    if (modelreg::is_shard_continuation(m.path)) continue;
    int shard_idx = 0, shard_cnt = 1;
    modelreg::shard_info(m.path, shard_idx, shard_cnt);
    const uint64_t weights =
        shard_cnt > 1 ? modelreg::shard_group_bytes(m.path) : m.size;
    nlohmann::json e = {
      {"path", m.path},
      {"name", m.name},
      {"arch", m.arch},
      {"quant", m.quant},
      {"size", weights},
      {"n_layers", m.meta.n_layers},
      {"n_ctx_train", m.meta.n_ctx_train},
      {"loaded", m.path == loaded_path},
      {"embedding", sup::embedding_arch(m.arch)},
    };
    if (!live.is_null()) {
      std::string id = sup::id_for_path(m.path);
      e["id"] = id;
      if (!id.empty() && live.contains(id) && live[id].is_object()) {
        e["state"] = live[id].value("state", "unloaded");
        e["loaded"] = e["state"].get<std::string>() == "loaded";
      } else {
        e["state"] = "unloaded";
        e["loaded"] = false;
      }
      // Карта по раскладке: −1 — модель размазана по всем (без пина).
      e["gpu"] = sup::model_gpu(m.path);
    }
    // «Светофор»: влезет ли в свободную VRAM при текущем контексте и какой максимум.
    if (gpu.vram_total > 0) {
      auto est = fit::estimate(m.meta, weights, (uint32_t)cfg.ctx, gpu);
      uint32_t cap = m.meta.n_ctx_train ? (uint32_t)m.meta.n_ctx_train : 8192u;
      e["fit"] = {
        {"ctx", cfg.ctx},
        {"vram_needed", est.total},
        {"fits", est.fits},
        {"max_ctx", fit::max_ctx(m.meta, weights, gpu, cap)},
      };
    }
    // Спутники: что уйдёт с диска вместе с моделью при удалении — показывают
    // подтверждения rm и TUI. У одиночной модели без модулей пусто.
    uint64_t comp_bytes = 0;
    nlohmann::json comps = nlohmann::json::array();
    for (auto& c : modelreg::companions_of(m.path, list)) {
      comps.push_back({{"path", c.path}, {"kind", c.kind}, {"size", c.size}});
      comp_bytes += c.size;
    }
    e["companions"] = comps;
    e["companions_bytes"] = comp_bytes;
    // Ручной контекст и пер-модельный mtp-mode из llmhost.conf — карточка
    // и подсказки к планировщику. mtp_mode — эффективный (override или [general]).
    e["ctx_override"] = sup::override_ctx_for_path(m.path);
    {
      const std::string ovr = sup::override_mtp_for_path(m.path);
      e["mtp_override"] = ovr;
      e["mtp_mode"] = ovr.empty() ? cfg.mtp_mode : ovr;
    }
    // Thinking-режим карточки: переопределение или auto (движок сам детектит
    // по шаблону).
    {
      const std::string t = sup::override_think_for_path(m.path);
      e["think_override"] = t;
      e["think_mode"] = t.empty() ? "auto" : t;
    }
    // Vision-режим карточки: переопределение или auto; mmproj — проекция,
    // которая прицепится (пусто — выключено или не найдено). Один ответ
    // с load-путём (mmproj_for_load), чтобы карточка не обещала лишнего.
    {
      const std::string vovr = sup::override_vision_for_path(m.path);
      e["vision_override"] = vovr;
      e["vision_mode"] = vovr.empty() ? "auto" : vovr;
      e["mmproj"] = sup::vision_mmproj_for_path(m.path);
    }
    arr.push_back(e);
  }
  return arr;
}

void setup_routes(httplib::Server& svr) {
  // Веб-дашборд: один статический файл (webui.h) без секретов внутри — все
  // данные ходят через /v0/* под auth. Раздаётся без auth, чтобы браузер
  // открывал страницу обычной навигацией; без токена дашборд видит только
  // свои 401. Токен UI хранит в localStorage и прикладывает к fetch сам.
  svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
    // Клиентские заголовки: страница живёт на localhost с токеном в
    // localStorage — встраивание в чужой iframe (кликджекинг кнопки
    // «удалить модель») запрещаем, кэширование — нет (одностраничный
    // дашборд всегда должен быть свежим; секретов в HTML нет).
    // Только frame-ancestors: полноценный default-src ломает кросс-портовый
    // чат дашборда (OpenAI-порт — другой origin, отсюда база из
    // location.hostname), а JS у нас инлайновый. XSS-поверхности это не
    // касается — экранирование в webui.h на месте.
    res.set_header("Content-Security-Policy", "frame-ancestors 'none'");
    res.set_header("X-Frame-Options", "DENY");
    res.set_header("Cache-Control", "no-store");
    res.set_content(lh::WEBUI_HTML, "text/html; charset=utf-8");
  });
  svr.Get("/v0/status", auth_wrap([](const httplib::Request&, httplib::Response& res) {
           json(res, sup::status_json());
         }));

  // Prometheus-текст: счётчики прокси (запросы/ошибки/токены/длительности)
  // + карты и состояния моделей из живых источников. За токеном, как весь
  // control API — Prometheus умеет bearer_token в scrape-конфиге.
  svr.Get("/v0/metrics", auth_wrap([](const httplib::Request&, httplib::Response& res) {
            res.set_content(metrics::render(sup::status_json(), sup::models_state(),
                                            sup::engine_metrics()),
                            "text/plain; version=0.0.4; charset=utf-8");
          }));

  svr.Get("/v0/hardware", auth_wrap([](const httplib::Request&, httplib::Response& res) {
            json(res, {{"gpu", sup::status_json()["gpu"]}});
          }));

  svr.Get("/v0/models", auth_wrap([](const httplib::Request&, httplib::Response& res) {
           json(res, {{"models", models_json()}});
         }));

  // Удаление модели с диска — чистое: вместе со спутниками (шард-части,
  // mmproj, MTP-модули из её репозитория), безвозвратно — корзин на headless-
  // сервере нет; загруженную демон выгрузит, пересобрав пресет роутера.
  // «path» — файл, «name» — ищется в реестре, как в /v0/load.
  svr.Delete("/v0/models", auth_wrap([](const httplib::Request& req,
                                        httplib::Response& res) {
             auto body = body_json(req, res, true);
             if (body.is_null()) return;
             if (non_string(body, "path", res)) return;
             if (non_string(body, "name", res)) return;
             std::string path = resolve_model_arg(body, res);
             if (path.empty()) return;
             // Путь обязан лежать внутри папки моделей: эндпоинт — про удаление
             // моделей, а не «удали что угодно, до чего дотянется демон».
             // Проверка каноническая (симлинки раскрываются) — «models/../secrets»
             // и ссылки наружу отсекаются.
             const std::string root = config::models_dir_resolved(config::current());
             std::string canon;
             {
               std::error_code cep;
               canon = std::filesystem::weakly_canonical(path, cep).string();
               if (cep || canon.empty()) { err(res, 404, tr("файл не найден: ") + path); return; }
               if (canon != root && canon.rfind(root + "/", 0) != 0) {
                 err(res, 400, tr("путь вне папки моделей: ") + path);
                 return;
               }
             }
             if (!lh::file_exists(path)) {
               err(res, 404, tr("файл не найден: ") + path);
               return;
             }
             // Чистое удаление: модель уходит вместе с тем, что скачалось вместе
             // с ней, — частями шард-группы и модулями (mmproj, MTP), лежащими в
             // её репозитории. Модуль, нужный ещё живой основной модели, не
             // трогаем; переданный явно модуль удаляется поодиночке.
             struct Item { std::string path, kind; uint64_t size; };
             std::vector<Item> items{{path, "model", lh::file_size(path)}};
             std::set<std::string> seen{path};
             auto add = [&](const std::string& p, const char* kind, uint64_t sz) {
               if (p.empty() || !seen.insert(p).second) return;
               std::error_code cep;
               std::string c = std::filesystem::weakly_canonical(p, cep).string();
               if (cep || c.empty() || (c != root && c.rfind(root + "/", 0) != 0))
                 return; // та же страховка, что и для основного пути
               items.push_back({p, kind, sz});
             };
             if (!modelreg::is_mtp_module(path) && !modelreg::is_mmproj_module(path)) {
               auto list = modelreg::scan(root);
               for (auto& c : modelreg::companions_of(path, list))
                 add(c.path, c.kind.c_str(), c.size);
               // Защитный проход по папке: части шарда, до которых реестр не
               // дошёл (растущий файл, предел скана).
               std::string dir = path.substr(0, path.find_last_of("/\\"));
               std::error_code dep;
               for (const auto& e : std::filesystem::directory_iterator(dir, dep)) {
                 if (dep) break;
                 std::string fp = e.path().string();
                 int i2, c2;
                 modelreg::shard_info(fp, i2, c2);
                 if (c2 > 1 && modelreg::shard_base(fp) == modelreg::shard_base(path))
                   add(fp, "shard", lh::file_size(fp));
               }
             }
             for (size_t k = 0; k < items.size(); ++k) // недокачки .part тех же файлов
               if (lh::file_exists(items[k].path + ".part"))
                 add(items[k].path + ".part", "part", lh::file_size(items[k].path + ".part"));

             uint64_t bytes = 0;
             nlohmann::json files = nlohmann::json::array();
             std::string fail;
             for (auto& it : items) {
               std::error_code ec;
               bool removed = std::filesystem::remove(it.path, ec);
               if (ec) { fail = it.path + ": " + ec.message(); break; }
               if (removed) {
                 bytes += it.size;
                 files.push_back(
                     {{"path", it.path}, {"kind", it.kind}, {"size", it.size}});
               }
             }
             // Пустые папки репозитория (включая MTP/) не оставляем — выметаем
             // вверх до корня папки моделей; не пустая папка остановит цепочку.
             std::vector<std::string> parents;
             for (auto& it : items) {
               size_t s = it.path.find_last_of("/\\");
               if (s == std::string::npos) continue;
               std::string d = it.path.substr(0, s);
               if (std::find(parents.begin(), parents.end(), d) == parents.end())
                 parents.push_back(d);
             }
             std::sort(parents.begin(), parents.end(),
                       [](const std::string& a, const std::string& b) {
                         return a.size() > b.size();
                       });
             for (auto& d : parents)
               for (std::string p = d; p.size() > root.size() &&
                                       p.compare(0, root.size() + 1, root + "/") == 0;) {
                 std::error_code ec;
                 if (!std::filesystem::remove(p, ec) || ec) break;
                 size_t s = p.find_last_of("/\\");
                 if (s == std::string::npos) break;
                 p = p.substr(0, s);
               }
             // Файлы парковки KV слотов удалённых моделей — вместе с моделями.
             for (auto& it : items) sup::park_purge_for_path(it.path);
             sup::registry_changed(); // выгрузить удалённое и обновить карту сразу
             if (!fail.empty()) {
               err(res, 500, tr("удалено ") + std::to_string(files.size()) + tr(" из ") +
                                 std::to_string(items.size()) + tr(", останов на: ") + fail);
               return;
             }
             json(res, {{"deleted", path}, {"recycle", false},
                        {"files", files}, {"bytes", bytes}});
           }));

  // Ручной контекст модели: {"path"|"name", "ctx": N} — пишет «ctx-size» в
  // секцию модели llmhost.conf (0 — убрать override) и сразу пересобирает
  // пресет: дети перезапускаются плавно с новым контекстом.
  svr.Post("/v0/models/ctx",
           auth_wrap([](const httplib::Request& req, httplib::Response& res) {
             auto body = body_json(req, res, true);
             if (body.is_null()) return;
             if (non_string(body, "path", res)) return;
             if (non_string(body, "name", res)) return;
             if (!body.contains("ctx") || !body["ctx"].is_number_integer()) {
               err(res, 400, "\"ctx\" — целое число токенов (0 — убрать ручной контекст)");
               return;
             }
             long long ctx = body["ctx"].get<long long>();
             if (ctx < 0 || (ctx > 0 && ctx < 256) || ctx > 2097152) {
               err(res, 400, "\"ctx\" — 0 (авто) или от 256 до 2097152 токенов");
               return;
             }
             std::string path = resolve_model_arg(body, res);
             if (path.empty()) return;
             std::string e2;
             if (!sup::set_model_ctx(path, (int)ctx, e2)) { err(res, 500, e2); return; }
             json(res, {{"model", path},
                        {"section", sup::override_section_for(path)},
                        {"ctx", ctx},
                        {"applied", true},
                        // Ручной контекст может исключить vision: проекция
                        // перестаёт помещаться. Предупреждение — клиенту.
                        {"note", ctx > 0 ? sup::vision_ctx_note(path, (int)ctx)
                                         : std::string()}});
           }));

  // Пер-модельный mtp-mode: "auto" снимает переопределение (глобальный
  // [general] снова решает), "on"/"off" — детерминированно включить/выключить
  // спекуляцию независимо от сторожа и наличия голов. Слот уходит в сон —
  // первый запрос поднимет ребёнка с новым режимом.
  svr.Post("/v0/models/mtp",
           auth_wrap([](const httplib::Request& req, httplib::Response& res) {
             auto body = body_json(req, res, true);
             if (body.is_null()) return;
             if (non_string(body, "path", res)) return;
             if (non_string(body, "name", res)) return;
             if (!body.contains("mode") || !body["mode"].is_string()) {
               err(res, 400, "\"mode\" — строка: auto, on или off");
               return;
             }
             std::string mode = body["mode"].get<std::string>();
             for (char& ch : mode)
               if (ch >= 'A' && ch <= 'Z') ch = (char)(ch - 'A' + 'a');
             if (mode != "auto" && mode != "on" && mode != "off") {
               err(res, 400, "\"mode\" — auto (умолчание), on или off");
               return;
             }
             std::string path = resolve_model_arg(body, res);
             if (path.empty()) return;
             std::string e2;
             if (!sup::set_model_mtp(path, mode, e2)) { err(res, 500, e2); return; }
             json(res, {{"model", path},
                        {"section", sup::override_section_for(path)},
                        {"mode", mode},
                        {"applied", true}});
           }));

  // Пер-модельный размер кэша MoE: {"path"|"name", "mib": N}. 0 — снять
  // override ([general] moe-cache-mib снова решает), N — МиБ вручную
  // (оптимум у моделей разный: Flash-Next 8192, Next-80B 10752 — 30.10).
  // Применение — при следующей загрузке модели.
  svr.Post("/v0/models/cache",
           auth_wrap([](const httplib::Request& req, httplib::Response& res) {
             auto body = body_json(req, res, true);
             if (body.is_null()) return;
             if (non_string(body, "path", res)) return;
             if (non_string(body, "name", res)) return;
             if (!body.contains("mib") || !body["mib"].is_number_integer()) {
               err(res, 400, "\"mib\" — целое число МиБ (0 — снять override)");
               return;
             }
             long long mib = body["mib"].get<long long>();
             if (mib < 0 || mib > 65536) {
               err(res, 400, "\"mib\" — 0 (снять override) или 1…65536 МиБ");
               return;
             }
             std::string path = resolve_model_arg(body, res);
             if (path.empty()) return;
             std::string e2;
             if (!sup::set_model_cache(path, (int)mib, e2)) {
               err(res, 500, e2);
               return;
             }
             json(res, {{"model", path},
                        {"section", sup::override_section_for(path)},
                        {"mib", mib},
                        {"applied", true}});
           }));

  // Пер-модельный vision-режим: "auto" снимает переопределение (своя
  // проекция прицепляется сама — родство имён, mmproj_for_load), "off" —
  // ребёнок едет без --mmproj. Слот уходит в сон — первый запрос поднимет
  // ребёнка с новым режимом.
  svr.Post("/v0/models/vision",
           auth_wrap([](const httplib::Request& req, httplib::Response& res) {
             auto body = body_json(req, res, true);
             if (body.is_null()) return;
             if (non_string(body, "path", res)) return;
             if (non_string(body, "name", res)) return;
             if (!body.contains("mode") || !body["mode"].is_string()) {
               err(res, 400, "\"mode\" — строка: auto или off");
               return;
             }
             std::string mode = body["mode"].get<std::string>();
             for (char& ch : mode)
               if (ch >= 'A' && ch <= 'Z') ch = (char)(ch - 'A' + 'a');
             if (mode != "auto" && mode != "off") {
               err(res, 400, "\"mode\" — auto (умолчание) или off");
               return;
             }
             std::string path = resolve_model_arg(body, res);
             if (path.empty()) return;
             std::string e2;
             if (!sup::set_model_vision(path, mode, e2)) { err(res, 500, e2); return; }
             json(res, {{"model", path},
                        {"section", sup::override_section_for(path)},
                        {"mode", mode},
                        {"applied", true}});
           }));

  // Пер-модельный thinking-режим: "auto" снимает переопределение (движок
  // сам детектит thinking по шаблону), "on"/"off" — форс --reasoning
  // ребёнку. Слот уходит в сон — первый запрос поднимет ребёнка с новым
  // режимом.
  svr.Post("/v0/models/thinking",
           auth_wrap([](const httplib::Request& req, httplib::Response& res) {
             auto body = body_json(req, res, true);
             if (body.is_null()) return;
             if (non_string(body, "path", res)) return;
             if (non_string(body, "name", res)) return;
             if (!body.contains("mode") || !body["mode"].is_string()) {
               err(res, 400, "\"mode\" — строка: auto, on или off");
               return;
             }
             std::string mode = body["mode"].get<std::string>();
             for (char& ch : mode)
               if (ch >= 'A' && ch <= 'Z') ch = (char)(ch - 'A' + 'a');
             if (mode != "auto" && mode != "on" && mode != "off") {
               err(res, 400, "\"mode\" — auto (умолчание), on или off");
               return;
             }
             std::string path = resolve_model_arg(body, res);
             if (path.empty()) return;
             std::string e2;
             if (!sup::set_model_thinking(path, mode, e2)) { err(res, 500, e2); return; }
             json(res, {{"model", path},
                        {"section", sup::override_section_for(path)},
                        {"mode", mode},
                        {"applied", true}});
           }));

  svr.Post("/v0/load", auth_wrap([](const httplib::Request& req, httplib::Response& res) {
             auto body = body_json(req, res, true);
             if (body.is_null()) return;

             if (non_string(body, "path", res)) return;
             if (non_string(body, "name", res)) return;
             // Установка движка глушит загрузки (гейт в engine.h): честный 409
             // сразу, а не 202 с тихим провалом в фоне. Чтение для UX;
             // авторитетная бронь — в трубах загрузки (sup::load и др.).
             if (engine::install_in_progress()) {
               err(res, 409, tr("идёт установка движка — загрузка моделей недоступна до её конца"));
               return;
             }
             // Единый резолв с rm/ctx/mtp: name/файл/стем + redirect шард-части
             // 2..N на первую (раньше здесь был inline-поиск без redirect —
             // load по имени части вёл себя иначе, чем rm той же модели).
             std::string path = resolve_model_arg(body, res);
             if (path.empty()) return; // ответ уже записан (400/404)
             // Несуществующий файл — честный 404 сразу, а не 202 «грузимся»
             // с тихим провалом загрузки в фоне.
             if (!lh::file_exists(path)) {
               err(res, 404, tr("файл не найден: ") + path);
               return;
             }
             // Битый/обрезанный GGUF — честный 400 сразу: заголовок читается
             // за миллисекунды, а 202 с провалом в фоне выглядит как
             // зависание (заморозка beta 12, сценарий 5.3). Части шард-группы
             // сюда не доезжают: резолв уводит на первую.
             {
               gguf::Meta meta0;
               if (!lh::modelreg::model_meta(path, meta0)) {
                 err(res, 400, tr("файл не читается как GGUF: ") + path);
                 return;
               }
             }

            nlohmann::json ack = {{"state", "loading"}};
            // harness: каждая модель — свой слот. «Уже загружена» —
            // идемпотентно; та же модель в Loading — честный 409 (спавны
            // сериализованы, вторая нога всё равно ждала бы).
            std::string id = sup::id_for_path(path);
            if (!id.empty()) {
              auto ms = sup::models_state();
              if (ms.contains(id)) {
                const std::string st = ms[id].value("state", "unloaded");
                if (st == "loading") {
                  err(res, 409, tr("уже идёт загрузка модели «") + id + "»");
                  return;
                }
                if (st == "loaded") {
                  err(res, 409, tr("модель уже загружена"));
                  return;
                }
              }
              ack["model"] = id;
            }

             sup::track_background([p = path] {
               std::string e;
               sup::load(p, e);
             });
             json(res, ack, 202);
           }));

  svr.Post("/v0/unload", auth_wrap([](const httplib::Request& req, httplib::Response& res) {
             auto body = body_json(req, res, false);
             // Занятая модель не выселяется никогда — идёт генерация.
             // Конкретная занятая → 409 с причиной; «все» → занятые остаются
             // (busy_kept), их приберёт сторож простоя. force=true — явный
             // люк в инварианте для конкретной модели: зомби-генерация
             // (клиент оборвал, а стрим жил) гасится принудительно, ВСЕ
             // генерации модели обрываются; на «все» не даётся.
             if (non_string(body, "model", res)) return;
             std::string want = sfield(body, "model");
             bool force = false;
             if (body.contains("force") && !body["force"].is_null()) {
               if (!body["force"].is_boolean()) {
                 err(res, 400, "\"force\" должен быть булевым");
                 return;
               }
               force = body["force"].get<bool>();
             }
             if (force && want.empty()) {
               err(res, 400,
                   "\"force\" выгружает одну занятую модель — назовите её "
                   "в \"model\" (принудительная выгрузка «всех» не даётся)");
               return;
             }
             if (!want.empty()) {
               std::string id = sup::id_for_path(want);
               if (id.empty() && !sup::models_state().contains(want) &&
                   want.find(".gguf") == std::string::npos) {
                 err(res, 404, "модель «" + want + tr("» не известна"));
                 return;
               }
             }
             std::string uerr;
             nlohmann::json r = sup::harness_unload_models(want, uerr, force);
             if (!uerr.empty()) {
               err(res, 409, uerr);
               return;
             }
             json(res, r);
           }));

  // ---- генерация изображений (второй движок, sd-server) --------------------
  // Слот и каталог моделей — sup_image.cpp/image.cpp; наружу генерация идёт
  // через OpenAI-порт (/v1/images/*), control отвечает за состояние и загрузку.

  svr.Get("/v0/image", auth_wrap([](const httplib::Request&, httplib::Response& res) {
            json(res, sup::image_status_json());
          }));

  svr.Post("/v0/image/load", auth_wrap([](const httplib::Request& req, httplib::Response& res) {
             auto body = body_json(req, res, true);
             if (body.is_null()) return;
             if (non_string(body, "model", res)) return;
             std::string id = sfield(body, "model");
             // Каталог и модель проверяем сразу: 400/404 до 202 — «грузимся»
             // с тихим провалом в фоне выглядит зависанием (ср. /v0/load).
             nlohmann::json st = sup::image_status_json();
             if (!st.value("configured", false)) {
               err(res, 400, tr("каталог image-моделей не настроен: ") +
                                  st.value("models_dir", std::string()) +
                                  tr(" — нужен llmhost-image.json"));
               return;
             }
             if (!id.empty()) {
               // id едет в журнал и сообщения ошибок: charset как в каталоге
               // (image.cpp id_ok) — иначе «\n» рисует фальшивые строки лога.
               for (unsigned char ch : id) {
                 if (!isalnum(ch) && ch != '-' && ch != '_' && ch != '.') {
                   err(res, 400, tr("id модели — [A-Za-z0-9._-], а не «") + id + "»");
                   return;
                 }
               }
               bool known = false;
               for (const auto& m : st["models"])
                 if (m.value("id", "") == id) { known = true; break; }
               if (!known) {
                 err(res, 404, tr("в каталоге image-моделей нет «") + id + tr("»"));
                 return;
               }
             }
             if (engine::install_in_progress()) {
               err(res, 409, tr("идёт установка движка — загрузка моделей недоступна до её конца"));
               return;
             }
             nlohmann::json ack = {{"state", "loading"}};
             if (!id.empty()) ack["model"] = id;
             sup::track_background([id] {
               std::string e;
               sup::image_load(id, e);
             });
             json(res, ack, 202);
           }));

  svr.Post("/v0/image/unload",
           auth_wrap([](const httplib::Request&, httplib::Response& res) {
             sup::image_unload();
             json(res, {{"state", "unloaded"}});
           }));

  // Удаление художника с сервера — файлы слотов + недокачки + свободный рецепт
  // и запись привязки; загруженную демон выгружает сам (как DELETE /v0/models).
  svr.Delete("/v0/image", auth_wrap([](const httplib::Request& req,
                                       httplib::Response& res) {
               auto body = body_json(req, res, true);
               if (body.is_null()) return;
               if (non_string(body, "model", res)) return;
               std::string id = sfield(body, "model");
               // Charset сразу: id едет в журнал и сообщения ошибок.
               if (!image::id_ok(id)) {
                 err(res, 400, tr("id модели — [A-Za-z0-9._-], а не «") + id + "»");
                 return;
               }
               // Именованный локальный: подписка временного из range-for —
               // висячая ссылка (временный умирает до тела цикла).
               nlohmann::json st = sup::image_status_json();
               bool known = false;
               for (const auto& m : st["models"])
                 if (m.value("id", "") == id) { known = true; break; }
               if (!known) {
                 err(res, 404, tr("в каталоге image-моделей нет «") + id + tr("»"));
                 return;
               }
               nlohmann::json report;
               std::string e2;
               if (!sup::image_remove(id, report, e2)) {
                 // Занятая загрузкой — гонка со спавном не вскрывается.
                 err(res, e2.find("загружается") != std::string::npos ? 409 : 500, e2);
                 return;
               }
               json(res, report);
             }));

  // ---- движок llama-server -------------------------------------------------

  svr.Get("/v0/engine", auth_wrap([](const httplib::Request&, httplib::Response& res) {
            // active/selected считает сам движок (active_build — зеркало
            // каскада find_server: с учётом установки и явного пути).
            json(res, nlohmann::json::parse(engine::status_json()));
          }));

  // Свежий b-релиз llama.cpp (GitHub, кэш внутри движка) — меню движка в
  // TUI показывает «установлен / актуальный» без собственного сетевого кода.
  svr.Get("/v0/engine/latest",
          auth_wrap([](const httplib::Request&, httplib::Response& res) {
            std::string e2;
            std::string rel = engine::latest_release(e2);
            if (rel.empty()) json(res, {{"error", e2}});
            else json(res, {{"release", rel}});
          }));

  // Выбрать сборку для запуска: {"build": "cuda12"} — явно, {"build": ""} или
  // «auto» — снова по железу. Выбор живёт в llmhost.conf [general]
  // (engine-build), дети подхватывают её при плавном перезапуске роутера.
  // Сборка должна быть пиннута и установлена (ставить: lhctl engine install
  // <сборка>).
  svr.Post("/v0/engine/build",
           auth_wrap([](const httplib::Request& req, httplib::Response& res) {
             auto body = body_json(req, res, true);
             if (body.is_null()) return;
             if (non_string(body, "build", res)) return;
             std::string build = sfield(body, "build");
             if (build == "auto") build.clear();
             if (!build.empty()) {
               bool known = false, installed = false;
               for (auto& b : engine::list())
                 if (b.id == build) {
                   known = b.pinned;
                   installed = b.installed;
                   break;
                 }
               if (!known)
                 err(res, 404, tr("в manifest.json нет пиннутой сборки «") + build + "»");
               else if (!installed)
                 err(res, 409, tr("сборка «") + build + tr("» не установлена — lhctl engine install ") + build);
               else {
                 config::Config cfg = config::current();
                 cfg.engine_build = build;
                 config::set_current(cfg);
                 config::save(cfg);
                 sup::apply_registry_change(); // пресет + плавный перезапуск детей
                 json(res, {{"selected", build},
                            {"active", engine::active_build(cfg)},
                            {"restarted", true}});
               }
               return;
             }
             config::Config cfg = config::current();
             cfg.engine_build.clear();
             config::set_current(cfg);
             config::save(cfg);
             sup::apply_registry_change();
             json(res, {{"selected", "auto"},
                        {"active", engine::active_build(cfg)},
                        {"restarted", true}});
           }));

  // Настройки [general] llmhost.conf — один источник правды (плюс per-model
  // секции, которые правит lhctl ctx). GET — вся таблица ключей; POST —
  // {"key": "...", "value": "..."}: валидация → запись в conf → применение
  // на лету (порты/bind/auth/cors/router-mode живьём не применяются — в
  // ответе restart_required: true и подсказка).
  svr.Get("/v0/settings", auth_wrap([](const httplib::Request&, httplib::Response& res) {
            nlohmann::json arr = nlohmann::json::array();
            for (auto& [k, v] : config::settings_table(config::current()))
              arr.push_back({{"key", k},
                             {"value", v},
                             {"restart_required", config::needs_daemon_restart(k)}});
            json(res, {{"conf", config::conf_path()}, {"settings", arr}});
          }));

  svr.Post("/v0/settings",
           auth_wrap([](const httplib::Request& req, httplib::Response& res) {
             auto body = body_json(req, res, true);
             if (body.is_null()) return;
             if (non_string(body, "key", res)) return;
             if (non_string(body, "value", res)) return;
             const std::string key = body["key"].get<std::string>();
             const std::string value = body["value"].get<std::string>();
             std::string norm, e2;
             if (!config::check_setting(key, value, engine::manifest_builds(), norm, e2)) {
               err(res, 400, e2);
               return;
             }
             config::Config cfg = config::current();
             config::set_field(cfg, key, norm);
             config::set_current(cfg);
             config::save(cfg);
             const bool restart_required = config::needs_daemon_restart(key);
             if (!restart_required)
               sup::apply_registry_change(); // пресет + плавный перезапуск детей
             else
               logf("config: %s изменён через API — вступит после рестарта демона",
                    key.c_str());
             json(res, {{"key", key},
                        {"value", norm},
                        {"applied", !restart_required},
                        {"restart_required", restart_required}});
           }));

  // Обновить пин манифеста на релиз llama.cpp: {"release": "bNNNNN"|"latest"}.
  // Синхронно (GitHub API + запись файла — секунды). Бинарники не трогаются,
  // поэтому модель в памяти не мешает; установка после пина — POST /v0/engine
  // и по-прежнему требует выгрузки.
  svr.Post("/v0/engine/pin",
           auth_wrap([](const httplib::Request& req, httplib::Response& res) {
             auto body = body_json(req, res, true);
             if (body.is_null()) return;
             if (non_string(body, "release", res)) return;
             std::string tag = body.value("release", std::string("latest"));
             if (engine::busy()) { err(res, 409, tr("установка уже идёт")); return; }
             engine::PinResult out;
             std::string e2;
             if (!engine::pin(tag, out, e2)) { err(res, 502, e2); return; }
             json(res, {{"release", out.release},
                        {"files", out.files},
                        {"path", out.path},
                        {"added", out.added}});
           }));

  svr.Post("/v0/engine", auth_wrap([](const httplib::Request& req, httplib::Response& res) {
             auto body = body_json(req, res, true);
             if (body.is_null()) return;
             if (non_string(body, "build", res)) return;
             std::string build = body.contains("build") && body["build"].is_string()
                                     ? body["build"].get<std::string>()
                                     : "vulkan";
             // Список сборок проверяем ДО занятости: неизвестное имя — всегда
             // 404, даже когда роутер грузится или модель резидентна (иначе
             // мусорный запрос получал 409 «занято» и зависел от состояния).
             bool known = false;
             for (auto& b : engine::list())
               if (b.id == build) known = true;
             if (!known) { err(res, 404, tr("в manifest.json нет сборки «") + build + "»"); return; }
             if (engine::busy()) { err(res, 409, tr("установка уже идёт")); return; }
             // Установка глушит роутер: пока модель в памяти — честный 409
             // сразу, а не молчаливый отказ потока после 202.
             if (sup::state() == sup::St::Loaded || sup::state() == sup::St::Loading) {
               err(res, 409, tr("сначала выгрузите модель — установке нужен свободный llama-server"));
               return;
             }
             std::thread([build] {
               std::string e;
               engine::install(build, e);
             }).detach();
             json(res, {{"state", "downloading"}, {"build", build}}, 202);
           }));

  // ---- скачивание моделей ---------------------------------------------------

  svr.Get("/v0/pull", auth_wrap([](const httplib::Request&, httplib::Response& res) {
            json(res, nlohmann::json::parse(pull::status_json()));
          }));

  // Короткие имена для /v0/pull (qwen3, gemma-4…): репозиторий известен,
  // квант под карту выбирает каталог. Список курируемый — см. catalog_alias.h.
  svr.Get("/v0/aliases", auth_wrap([](const httplib::Request&, httplib::Response& res) {
            nlohmann::json arr = nlohmann::json::array();
            for (const lh::alias::Entry& e : lh::alias::kAll)
              arr.push_back({{"name", e.name}, {"repo", e.repo}, {"note", e.note}});
            json(res, {{"aliases", arr}});
          }));

  svr.Post("/v0/pull", auth_wrap([](const httplib::Request& req, httplib::Response& res) {
             auto body = body_json(req, res, true);
             if (body.is_null()) return;
             if (non_string(body, "repo", res)) return;
             if (non_string(body, "file", res)) return;
             if (non_string(body, "kind", res)) return;
             std::string repo = sfield(body, "repo");
             std::string file = sfield(body, "file");
             const std::string kind = sfield(body, "kind").empty()
                                          ? "llm"
                                          : sfield(body, "kind");
             if (kind != "llm" && kind != "image") {
               err(res, 400, tr("kind — «llm» или «image», а не «") + kind + "»");
               return;
             }
             if (repo.empty()) { err(res, 400, tr("укажите \"repo\": \"org/repo\"")); return; }
             std::string e;
             if (kind == "image") {
               // Художник: имя каталога или репо-конвенция; раскладку задаёт
               // рецепт — файловый фильтр тут не имеет смысла.
               if (!file.empty()) {
                 err(res, 400, tr("для kind=image «file» не нужен — раскладку задаёт рецепт"));
                 return;
               }
               if (!pull::start_image(repo, e)) { err(res, pull::busy() ? 409 : 400, e); return; }
             } else if (!pull::start(repo, file, e)) {
               err(res, pull::busy() ? 409 : 400, e);
               return;
             }
             json(res, {{"state", "resolving"}, {"kind", kind}, {"repo", repo}}, 202);
           }));

  // Штатная остановка идущего скачивания: curl глушится, .part остаётся
  // для докачки (повторный pull продолжает с места останова).
  svr.Delete("/v0/pull", auth_wrap([](const httplib::Request&, httplib::Response& res) {
             std::string e;
             if (!pull::cancel(e)) { err(res, 400, tr(e.c_str())); return; }
             json(res, {{"state", "stopping"}}, 200);
           }));

  // ---- каталог HuggingFace ----------------------------------------------------

  svr.Get("/v0/catalog/search", auth_wrap([](const httplib::Request& req, httplib::Response& res) {
            std::string e;
            int limit = 24;
            try {
              limit = std::stoi(req.get_param_value("limit"));
            } catch (...) {}
            std::string out = catalog::search_json(req.get_param_value("q"), limit, e);
            if (!e.empty()) { err(res, 502, e); return; }
            res.set_content(out, "application/json");
          }));

  svr.Get("/v0/catalog/model", auth_wrap([](const httplib::Request& req, httplib::Response& res) {
            std::string e;
            std::string repo = req.get_param_value("repo");
            if (repo.empty()) { err(res, 400, tr("укажите \"repo\": org/repo")); return; }
            std::string out = catalog::detail_json(repo, e);
            if (!e.empty()) { err(res, e.find("401") != std::string::npos ? 404 : 502, e); return; }
            res.set_content(out, "application/json");
          }));

  // ---- именованные токены /v1 (SQLite-БД) -------------------------------------

  svr.Get("/v0/tokens", auth_wrap([](const httplib::Request&, httplib::Response& res) {
            json(res, {{"tokens", lh::tokens::list()}});
          }));

  svr.Post("/v0/tokens", auth_wrap([](const httplib::Request& req, httplib::Response& res) {
              auto body = body_json(req, res, true);
              if (body.is_null()) return;
              if (non_string(body, "name", res)) return;
              std::string name = sfield(body, "name");
              std::string token, e;
              if (!lh::tokens::add(name, token, e)) { err(res, 400, e); return; }
              json(res, {{"name", name}, {"token", token}});
            }));

  svr.Delete("/v0/tokens", auth_wrap([](const httplib::Request& req, httplib::Response& res) {
               auto body = body_json(req, res, true);
               if (body.is_null()) return;
               if (non_string(body, "name", res)) return;
               std::string e;
               if (!lh::tokens::remove(sfield(body, "name"), e)) { err(res, 404, e); return; }
               json(res, {{"revoked", sfield(body, "name")}});
             }));

  // ---- замер движков ----------------------------------------------------------

  svr.Get("/v0/bench", auth_wrap([](const httplib::Request&, httplib::Response& res) {
           json(res, nlohmann::json::parse(bench::status_json()));
         }));

  svr.Post("/v0/bench", auth_wrap([](const httplib::Request& req, httplib::Response& res) {
             auto body = body_json(req, res, true);
             if (body.is_null()) return;
             if (non_string(body, "path", res)) return;
             std::string path = sfield(body, "path");
             std::string e;
             if (!bench::start(path, e)) { err(res, bench::busy() ? 409 : 400, e); return; }
             json(res, {{"state", "running"}}, 202);
           }));

  svr.Get("/v0/credentials", auth_wrap([](const httplib::Request&, httplib::Response& res) {
            config::Config cfg = config::current();
            json(res, {
              {"openai_base_url", "http://127.0.0.1:" + std::to_string(cfg.openai_port) + "/v1"},
              // Порт отдельно: дашборд строит базу от хоста адресной строки
              // (127.0.0.1 в base_url — взгляд сервера, чужому браузеру он
              // показывает его собственный localhost).
              {"openai_port", cfg.openai_port},
              {"api_key", config::client_token()},
              {"auth_required", cfg.openai_auth},
            });
          }));

  svr.Post("/v0/quit", auth_wrap([](const httplib::Request&, httplib::Response& res) {
             json(res, {{"bye", true}});
             lh::quit::request();
           }));
}

} // namespace

namespace {
std::shared_ptr<httplib::Server> g_svr;
std::thread g_thr;
} // namespace

bool start(int port) {
  g_svr = std::make_shared<httplib::Server>();
  auto svr = g_svr;
  // Пул воркеров по умолчанию = hardware_concurrency−1: одно длинное
  // (задержанное read-таймаутом) соединение снимает с обслуживания весь
  // control API локальным хулиганом. Явный пул шире дефолта.
  svr->new_task_queue = [] { return new httplib::ThreadPool(64); };
  svr->set_payload_max_length(1024 * 1024);
  // Контрольный API — короткие запросы/ответы, никакого стрима: лимиты
  // строгие, локальный хулиган не вешает потоки вечным чтением.
  svr->set_read_timeout(30, 0);
  svr->set_write_timeout(60, 0);
  setup_routes(*svr);

  // Результат bind — наружу, синхронно: занятый порт не должен оставлять
  // полуживой демон (процесс жив, control мёртв) — старт обязан отказать.
  std::promise<bool> bound;
  g_thr = std::thread([svr, port, &bound]() {
    if (!svr->bind_to_port("127.0.0.1", port)) {
      logf("control: порт %d занят — control API не поднят", port);
      bound.set_value(false);
      return;
    }
    logf("control: слушаю 127.0.0.1:%d", port);
    bound.set_value(true);
    svr->listen_after_bind();
  });
  return bound.get_future().get();
}

void stop() {
  if (g_svr) g_svr->stop();
  if (g_thr.joinable()) g_thr.join();
  g_svr.reset();
}

} // namespace lh::control
