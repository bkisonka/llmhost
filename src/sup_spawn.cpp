#include "supervisor.h"
#include "sup_internal.h"

#include "engine.h"
#include "proc.h"
#include "util.h"

#include <httplib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace lh::sup::detail {

using clock_ = std::chrono::steady_clock;
using lh::proc::Child;

// Последний kill дочернего процесса (steady_ms): драйвер возвращает VRAM
// убитого процесса с задержкой (на Windows это поведение WDDM, на Linux NVML
// показывает возврат не мгновенно), и спавн нового сервера раньше времени
// видит занятую карту.
std::atomic<int64_t> g_last_kill_ms{0};

int64_t steady_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             clock_::now().time_since_epoch())
      .count();
}

// Свободная/общая память «вида» для ожидания возврата VRAM: карта пина или
// сумма всех планируемых. Сумма — сырая (без планировочных вычетов оверхеда):
// settle следит за физикой возврата памяти, не за бюджетом плана.
// (Переехало из supervisor.cpp — settle нужен и sup_plan.record_profile.)
hw::Gpu settle_view(int card) {
  auto cards = hw::query_all();
  if (card >= 0 && card < (int)cards.size() && cards[card].vram_total) return cards[card];
  hw::Gpu sum;
  bool any = false;
  for (const auto& c : cards) {
    if (!c.nvidia || !c.vram_total) continue;
    if (!any) { sum = c; any = true; continue; }
    sum.vram_free += c.vram_free;
    sum.vram_total += c.vram_total;
  }
  return any ? sum : hw::query();
}

// Запасная карта для второстепенного груза (черновик, доля экспертов MoE):
// nvidia-карта пула вне used_cards со свободой не меньше need_bytes, лучшая
// по свободе. -1 — запасной нет: телеметрии нет или всё занято. Нумерация
// CUDA<i> зеркалит hw-индексы — демон ставит CUDA_DEVICE_ORDER=PCI_BUS_ID
// (daemond), как nvidia-smi.
int spare_card_index(const std::set<int>& used_cards, uint64_t need_bytes) {
  int best = -1;
  uint64_t best_free = 0;
  for (const auto& c : hw::query_all()) {
    if (!c.nvidia || !c.vram_total) continue;
    if (used_cards.count(c.index)) continue;
    if (c.vram_free < need_bytes) continue;
    if (c.vram_free > best_free) {
      best_free = c.vram_free;
      best = c.index;
    }
  }
  return best;
}

// Карты, занятые раскладкой ствола: пин — одна; сплит или без пина — весь
// пул (дефолтный делёж движка трогает все карты).
std::set<int> target_cards(const GpuTarget& gt) {
  std::set<int> used;
  if (!gt.split && gt.main_gpu >= 0) {
    used.insert(gt.main_gpu);
    return used;
  }
  for (const auto& c : hw::query_all())
    if (c.nvidia && c.vram_total) used.insert(c.index);
  return used;
}

// -ot-правило для конкретной запасной карты: блоки экспертов MoE с головы,
// суммарно до budget байт. Имена блоков — точным перечислением (alternation),
// не диапазоном: expert_bytes_by_block несёт байты каждого блока, а плотные
// блоки first-k-dense-семей в нём нулевые — перечисление их просто не берёт.
// nextn-тензоры (blk.N.nextn.ffn_*) под «blk\.N\.ffn_…» не подходят и едут
// общим exps=CPU — как при полном выносе.
std::string expert_share_ot(const gguf::Meta& meta, int card, uint64_t budget,
                            int* blocks_out, uint64_t* bytes_out, int max_blocks) {
  if (card < 0 || meta.expert_bytes_by_block.empty() || !budget) return {};
  uint64_t acc = 0;
  int blocks = 0;
  std::string alt;
  for (size_t b = 0; b < meta.expert_bytes_by_block.size(); ++b) {
    if (max_blocks > 0 && (int)b >= max_blocks) break; // за пределом выноса — уже GPU
    const uint64_t eb = meta.expert_bytes_by_block[b];
    if (!eb) continue; // плотный блок — не эксперты
    if (acc + eb > budget) break;
    acc += eb;
    ++blocks;
    if (!alt.empty()) alt += "|";
    alt += std::to_string(b);
  }
  if (!blocks) return {};
  if (blocks_out) *blocks_out = blocks;
  if (bytes_out) *bytes_out = acc;
  return "blk\\.(" + alt + ")\\.ffn_.*_exps\\.=CUDA" + std::to_string(card);
}

// Прогрев: первый запрос в Vulkan-бэкенде включает компиляцию шейдеров
// (~1,4 с). Делаем его сами сразу после загрузки — клиент этого ждать не
// должен. model — для роутера: /completion без «model» не знает, кого греть.
// (Переехало из supervisor.cpp вместе с дублем warmup_model.)
WarmSample warmup(int port, const std::string& api_key, const std::string& model) {
  WarmSample out;
  httplib::Client cli("127.0.0.1", port);
  cli.set_connection_timeout(std::chrono::seconds(3));
  cli.set_read_timeout(std::chrono::seconds(120));
  httplib::Headers hdrs = {{"Authorization", "Bearer " + api_key}};
  // 128 токенов — чтобы собрался граф декодирования и спекуляции,
  // а не только предзаполнение одного токена. Без шаблона чата.
  nlohmann::json body = {
      {"prompt", "Hi"},
      {"n_predict", 128},
      {"temperature", 0},
      {"cache_prompt", false},
      // «Hi» на temperature 0 часто упирается в EOS за десятки токенов
      // (gemma-26B — 83). Короткий ответ не собирает граф декодирования
      // и не годится как выборка скорости.
      {"ignore_eos", true},
  };
  if (!model.empty()) body["model"] = model;
  auto res = cli.Post("/completion", hdrs, body.dump(), "application/json");
  if (!res || res->status != 200) {
    logf("warmup: прогрев не прошёл (не страшно): %d", res ? res->status : -1);
    return out;
  }
  try {
    auto r = nlohmann::json::parse(res->body);
    auto tm = r.value("timings", nlohmann::json::object());
    out.tps = tm.value("predicted_per_second", 0.0);
    out.n = tm.value("predicted_n", 0);
  } catch (...) {
    return out;
  }
  if (out.n > 0)
    logf("warmup: %.0f ток/с за %d токенов", out.tps, out.n);
  return out;
}

// Прогрев страничного кэша под CPU-часть модели: на сетевой ФС холодные
// mmap-страницы экспертов читаются рывками; одна последовательная загрузка
// файла в кэш — и роутер всегда попадает по горячим страницам. Страницы
// кэша ядра не закоммичены, при давлении памяти вытесняются сами.
void warm_pages(const std::string& path) {
#ifdef _WIN32
  // FILE_FLAG_SEQUENTIAL_SCAN — тот же удвоенный readahead, что и
  // POSIX_FADV_SEQUENTIAL (см. POSIX-ветку ниже).
  std::wstring wpath = utf8_to_wide(path);
  HANDLE h = CreateFileW(wpath.c_str(), GENERIC_READ,
                         FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                         nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN,
                         nullptr);
  if (h == INVALID_HANDLE_VALUE) {
    logf("warm: файл не открылся — работаем без прогрева (GetLastError=%lu)",
         GetLastError());
    return;
  }
  LARGE_INTEGER sz{};
  if (!GetFileSizeEx(h, &sz) || sz.QuadPart <= 0) { CloseHandle(h); return; }
  const uint64_t total = (uint64_t)sz.QuadPart;
  std::vector<char> buf(8u << 20);
  auto t0 = clock_::now();
  uint64_t done = 0;
  int last_report = -10;
  while (done < total) {
    DWORD want = (DWORD)std::min<uint64_t>(buf.size(), total - done);
    DWORD got = 0;
    if (!ReadFile(h, buf.data(), want, &got, nullptr) || got == 0) break;
    done += (uint64_t)got;
    int pct = (int)(done * 10 / total); // десятки процентов
    if (pct > last_report) {
      logf("warm: %d0%% (%.1f/%.1f ГиБ)", pct, done / 1073741824.0, total / 1073741824.0);
      last_report = pct;
    }
  }
  CloseHandle(h);
  logf("warm: %llu/%llu байт за %.1f с", (unsigned long long)done,
       (unsigned long long)total,
       std::chrono::duration<double>(clock_::now() - t0).count());
#else
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    logf("warm: файл не открылся — работаем без прогрева (%s)", strerror(errno));
    return;
  }
  struct stat st{};
  if (::fstat(fd, &st) != 0 || st.st_size <= 0) { ::close(fd); return; }
  const uint64_t total = (uint64_t)st.st_size;
  // POSIX_FADV_SEQUENTIAL — удвоенный readahead, аналог FILE_FLAG_SEQUENTIAL_SCAN.
  ::posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
  std::vector<char> buf(8u << 20);
  auto t0 = clock_::now();
  uint64_t done = 0;
  int last_report = -10;
  while (done < total) {
    size_t want = (size_t)std::min<uint64_t>(buf.size(), total - done);
    ssize_t got = ::read(fd, buf.data(), want);
    if (got < 0 && errno == EINTR) continue; // сигнал демону — не повод обрывать прогрев
    if (got <= 0) break;
    done += (uint64_t)got;
    int pct = (int)(done * 10 / total); // десятки процентов
    if (pct > last_report) {
      logf("warm: %d0%% (%.1f/%.1f ГиБ)", pct, done / 1073741824.0, total / 1073741824.0);
      last_report = pct;
    }
  }
  ::close(fd);
  logf("warm: %llu/%llu байт за %.1f с", (unsigned long long)done,
       (unsigned long long)total,
       std::chrono::duration<double>(clock_::now() - t0).count());
#endif
}


// Модели, чья загрузка кончилась «вечным» ребёнком: SIGKILL не берёт процесс в
// непрерываемом сне (D) внутри nvidia-драйвера. Повторная загрузка до рестарта
// демона запрещена — каждый такой ребёнок держит драйвер и вешает весь сервер
// (поймано 07.10: два подхода к одно-карточному 70B = load 146, лёг sshd,
// SysRq-ребут). Чистится только рестартом демона.
std::set<std::string> g_wedged_models;
std::mutex g_wedged_mtx;


bool try_attempt(const std::string& exe, const std::string& path, int ngl, int ctx_req,
                 bool fa, bool kv_q8, const std::string& kv_type, bool exps_cpu, int ubatch,
                 const std::string& mmproj,
                 const TryTarget& tgt, const SpawnExtra& extra,
                 Child& c, int& port_out, std::string& key_out, int& actual_ctx,
                 std::string& err, bool* wedge_out,
                 std::string* tail_out) {
  const hw::Gpu gpu0 = settle_view(tgt.split ? -1 : tgt.main_gpu); // свободно до спавна
  port_out = lh::free_tcp_port(8100, 8199);
  if (!port_out) { err = "нет свободного порта для llama-server"; return false; }
  key_out = lh::random_hex(32);
  if (key_out.empty()) {
    err = "не удалось сгенерировать api-key (getrandom) — без ключа сервер не поднимаем";
    return false;
  }
  std::vector<std::string> args = {
      "--host", "127.0.0.1", "--port", std::to_string(port_out),
      "--api-key", key_out, "-m", path,
      "-c", std::to_string(ctx_req), "-ngl", std::to_string(ngl), "--no-webui",
      // Прометей-счётчики слота (префилл/декод/токены) — анализ ходов
      // постфактум: без них /metrics ребёнка отвечает 501 (урок 29.10:
      // разбор агентского хода с инструментами упёрся в отсутствие цифр).
      // Эндпоинт за тем же api-key и на 127.0.0.1, наружу не светится.
      "--metrics"};
  // Доля экспертов на запасную карту живёт по проверенному 28.10 рецепту:
  // движок обязан ВИДЕТЬ обе карты (--device ствол,запасная; -ot "=CUDA<i>"
  // на неперечисленное устройство роняет загрузку), а весь неэкспертный груз
  // прикрывается ловушкой .*=ствол — иначе дефолтный слой-сплит размажет
  // плотные части по двум картам. main-gpu — позиция ствола в списке.
  const bool spare_experts =
      exps_cpu && !extra.expert_ot.empty() && extra.expert_spare_idx >= 0 &&
      !tgt.split && tgt.main_gpu >= 0;
  std::string devices = tgt.devices;
  if (spare_experts) {
    const std::string spare_name = "CUDA" + std::to_string(extra.expert_spare_idx);
    if (devices.find(spare_name) == std::string::npos) devices += "," + spare_name;
  }
  if (!devices.empty()) { args.push_back("--device"); args.push_back(devices); }
  if (spare_experts) { args.push_back("--main-gpu"); args.push_back("0"); }
  if (!tgt.ts.empty()) { args.push_back("-ts"); args.push_back(tgt.ts); }
  if (tgt.split && tgt.main_gpu >= 0) {
    args.push_back("--main-gpu");
    args.push_back(std::to_string(tgt.main_gpu));
  }
  if (!mmproj.empty()) { args.push_back("--mmproj"); args.push_back(mmproj); }
  if (fa) { args.push_back("-fa"); args.push_back("on"); }
  if (kv_q8) {
    const std::string ct = kv_type.empty() ? "q8_0" : kv_type;
    args.push_back("-ctk"); args.push_back(ct);
    args.push_back("-ctv"); args.push_back(ct);
  }
  if (exps_cpu) {
    // Доля экспертов — на запасную карту (правило конкретных блоков), остальное
    // — на CPU, и ловушка .*=ствол закрывает плотные части (см. spare_experts
    // выше). Порядок правил значим: движок берёт первое совпавшее, поэтому
    // точное правило блоков едет ПЕРЕД общим exps=CPU.
    if (spare_experts) {
      args.push_back("-ot");
      args.push_back(extra.expert_ot);
    }
    args.push_back("-ot");
    args.push_back("exps=CPU");
    if (spare_experts) {
      args.push_back("-ot");
      args.push_back(".*=CUDA" + std::to_string(tgt.main_gpu));
    }
  } else if (extra.ncmoe > 0) {
    // Частичное размещение экспертов MoE: первые N блоков — в RAM, остальные
    // на GPU. Имя флага сверено с пином b11476 (--n-cpu-moe, arg.cpp).
    // expert-card = spare к этой форме НЕ применяется: её блоки — задача
    // ncmoe, а переселение части в CUDA требует другой раскладки устройств
    // (ловушка .* перекроет смысл ncmoe). Полная форма — где фича живёт.
    args.push_back("--n-cpu-moe");
    args.push_back(std::to_string(extra.ncmoe));
  }
  if (extra.moe_cache_mib > 0 && (exps_cpu || extra.ncmoe > 0)) {
    // Кэш горячих MoE-экспертов в VRAM (пин b11963, --moe-cache-mib; у
    // апстрима выключен по умолчанию — как и у нас). Смысл есть только при
    // вынесенных экспертах, без выноса кэшировать нечего; бюджет под него
    // план уже урезал (plan_for — до построения раскладки).
    args.push_back("--moe-cache-mib");
    args.push_back(std::to_string(extra.moe_cache_mib));
    if (extra.moe_cpu_share > 0.0) {
      // Параллельный CPU-счёт промахов (#30966): без гранта кэша флаг
      // бессмыслен — передаём только вместе с ним.
      char b[32];
      std::snprintf(b, sizeof(b), "%.2f", extra.moe_cpu_share);
      args.push_back("--moe-cache-cpu-share");
      args.push_back(b);
    }
  }
  if (ubatch > 0) { args.push_back("-ub"); args.push_back(std::to_string(ubatch)); }
  // CPU-потоки ребёнка: дефолт движка = hardware_concurrency (SMT) — на
  // exps_cpu-формах с cpu-share > 0 физические ядра бьют SMT (замер 31.10,
  // bench/results/2026-10-31-flash-next-bar).
  if (extra.threads > 0) { args.push_back("-t"); args.push_back(std::to_string(extra.threads)); }
  // Спекуляция: имена флагов сверены с пином b11476 (--model-draft/--spec-type/
  // --gpu-layers-draft). spec-type без draft-файла — встроенный MTP (nextn).
  if (!extra.draft_path.empty()) {
    args.push_back("--model-draft"); args.push_back(extra.draft_path);
    args.push_back("--gpu-layers-draft"); args.push_back(std::to_string(extra.draft_ngl));
    // Черновик на запасную карту: --device-draft сверено с пином b11934
    // (--spec-draft-device/-devd/--device-draft). Пусто — карта ствола.
    if (!extra.draft_device.empty()) {
      args.push_back("--device-draft"); args.push_back(extra.draft_device);
    }
  }
  if (!extra.spec_type.empty()) {
    args.push_back("--spec-type"); args.push_back(extra.spec_type);
  }
  if (extra.np > 1) {
    // Параллельные слоты: контекст делится движком между слотами (план уже
    // посчитан с этим делением), continuous batching включаем явно — не
    // полагаемся на дефолт пина.
    args.push_back("-np"); args.push_back(std::to_string(extra.np));
    args.push_back("-cb");
  }
  if (extra.embedding) args.push_back("--embedding");
  if (!extra.think_mode.empty()) {
    // Пер-модельный thinking-mode: on/off — --reasoning ребёнку (шаблонам
    // с enable_thinking движок отдаёт kwarg); auto — флага нет вовсе.
    args.push_back("--reasoning");
    args.push_back(extra.think_mode);
  }
  // Фит-проба llama.cpp выключена: на почти полной соседней карте она сама
  // падала с OOM (рулетка спавна), а её «умное урезание» дублирует наш план —
  // контекст всегда задан явно, недолезает — лестница понизит по честному OOM.
  args.push_back("-fit");
  args.push_back("off");
  if (!extra.slot_save_path.empty()) {
    // Парковка KV слотов (harness): включает POST /slots/{i}?action=save|restore
    // и задаёт каталог-белый список имён файлов (наш park_dir; каталог уже
    // существует — движок отказывает на старте, если его нет).
    args.push_back("--slot-save-path");
    args.push_back(extra.slot_save_path);
  }
  // [general] extra-args: дословный хвост флагов, разбитый по пробелам.
  // Значения с пробелами не поддерживаются — флаги движка их не требуют.
  {
    size_t pos = 0;
    const std::string &ea = extra.extra_args;
    while (pos <= ea.size()) {
      size_t sp = ea.find(' ', pos);
      std::string tok = ea.substr(pos, sp == std::string::npos ? std::string::npos
                                                               : sp - pos);
      if (!tok.empty()) args.push_back(tok);
      if (sp == std::string::npos) break;
      pos = sp + 1;
    }
  }
  if (!c.start(exe, args, lh::engine_child_env(exe))) {
    err = std::string("не удалось запустить llama-server (fork/exec): ") + strerror(errno);
    return false;
  }

  auto w = lh::proc::wait_health(c, port_out, 240000);
  if (w == lh::proc::Wait::Healthy) {
    // Фактический контекст — llama-server мог подрезать по памяти.
    actual_ctx = ctx_req;
    httplib::Client cli("127.0.0.1", port_out);
    cli.set_connection_timeout(std::chrono::seconds(3));
    auto res = cli.Get("/props");
    if (res && res->status == 200) {
      try {
        actual_ctx = nlohmann::json::parse(res->body)
                         .at("default_generation_settings")
                         .at("n_ctx")
                         .get<int>();
      } catch (...) { /* нет поля — оставим запрошенный */ }
    }
    return true;
  }

  std::string tail = c.log_tail();
  if (tail_out) *tail_out = tail;
  const pid_t spawned = c.pid(); // до stop(): stop() забывает pid
  c.stop();
  // Клин драйвера: SIGKILL не берёт процесс в непрерываемом сне (D) внутри
  // nvidia-модуля. Обычная смерть — /proc исчез или зомби; если через 3 с
  // после SIGKILL ребёнок жив — это он, и лестнице здесь конец (следующая
  // попытка плодит нового вечного ребёнка поверх старого: 07.10 два подхода
  // подряд положили весь сервер). Модель — в стоп-лист до рестарта демона.
  if (spawned > 0) {
    bool eternal = true;
    char state = '?';
    for (int i = 0; i < 6; ++i) {
      std::ifstream st("/proc/" + std::to_string(spawned) + "/stat");
      if (!st) { eternal = false; break; } // исчез — умер
      std::string all((std::istreambuf_iterator<char>(st)), {});
      const size_t rp = all.rfind(')');
      if (rp == std::string::npos || rp + 2 >= all.size()) { eternal = false; break; }
      state = all[rp + 2];
      if (state == 'Z') { eternal = false; break; } // зомби — умер, ждёт reap
      if (i < 5) std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    if (eternal) {
      logf("load: КЛИН ДРАЙВЕРА: ребёнок %d не умирает после SIGKILL (состояние %c) — "
           "лестница остановлена, модель в стоп-лист до рестарта демона",
           (int)spawned, state);
      { std::lock_guard l(g_wedged_mtx); g_wedged_models.insert(path); }
      if (wedge_out) *wedge_out = true;
      err = "клин драйвера: ребёнок не умирает после SIGKILL — серверу нужен рестарт, "
            "серверу целиком может понадобиться ребут";
      return false; // мимо vram_settle: память «вечного» не вернётся, ждать её — ложь
    }
  }
  // Процесс умер, но драйвер вернёт его VRAM с задержкой, а следующая попытка
  // лестницы спавнится немедленно: не подождав, получим ложный OOM и
  // спустимся по лестнице зря (или уедем в медленную «расшитую» память).
  g_last_kill_ms = steady_ms();
  vram_settle(gpu0.vram_free + lh::file_size(path), tgt.split ? -1 : tgt.main_gpu);
  if (w == lh::proc::Wait::Oom) {
    err = "не хватило видеопамяти при -ngl " + std::to_string(ngl) + ", -c " +
          std::to_string(ctx_req);
    logf("load: %s — понижаю параметры", err.c_str());
    return false;
  }
  int code = 0;
  c.exited(code);
  std::string line = lh::proc::last_meaningful_line(tail);
  err = (w == lh::proc::Wait::Exited
             ? "llama-server завершился (код " + std::to_string(code) + ")"
             : "llama-server не отвечает на /health") +
        (line.empty() ? "" : ": " + line);
  return false;
}

} // namespace lh::sup::detail

namespace lh::sup {

using detail::g_last_kill_ms;
using detail::settle_view;
using detail::steady_ms;

// Драйвер отдаёт память убитого процесса с задержкой (на Windows это гигиена
// уровня WDDM из scripts/bench_vs_ollama.py, где без ожидания однажды словили
// тихий спилл: gemma 69 → 10 ток/с без единой ошибки в логе; на Linux NVML
// тоже показывает возврат не мгновенно). Спавн на глазах у задержки даёт
// ложный OOM на лестнице — или веса молча уезжают в медленную память.
// Ждём возврата: expect_free — сколько должно стать свободно, когда память
// убитого вернётся (0 — неизвестно); выходим и по «рост прекратился», но
// только увидев хотя бы один скачок: без него «стабильно мало» неотличимо
// от ещё не начавшегося возврата. Максимум 12 с; телеметрии нет совсем —
// фиксированная пауза, лучшее, что можем.
// Внешняя смерть наших детей (свип сирот, жёсткая смерть прошлого демона):
// штампуем как обычный kill — первая загрузка дождётся возврата VRAM.
void note_external_kill() { g_last_kill_ms = steady_ms(); }

void vram_settle(uint64_t expect_free, int card) {
  hw::Gpu g = settle_view(card);
  if (!g.vram_free) { // телеметрии нет
    std::this_thread::sleep_for(std::chrono::seconds(3));
    return;
  }
  if (expect_free && g.vram_free + (256ull << 20) >= expect_free) return; // уже вернулась
  // expect_free — это «снимок до возврата + размер убитой модели». Если такая
  // сумма переваливает за всю карту, свободно больше, чем осталось бы при ещё
  // висящей модели, — память убитого вернулась, и ждать нечего (иначе цель
  // недостижима и сработает только таймаут).
  // Недостижимая цель (план с запасами просит больше карты) — не повод
  // возвращаться МГНОВЕННО: после выселения память ещё возвращается драйвером,
  // а снимок «до» для профиля делается сразу за нами. Ждём стабильности наравне
  // со всеми (поймано живьём: qwen36 после выселения 27B писала дельту −160 МиБ
  // — снимок «до» наследовал секунду кэша с резидентом).
  const bool reachable = expect_free <= g.vram_total;
  logf("vram: жду возврата видеопамяти (свободно %llu из %llu МиБ, ждём%s%llu МиБ)",
       (unsigned long long)(g.vram_free >> 20), (unsigned long long)(g.vram_total >> 20),
       expect_free ? " до " : " роста до ",
       (unsigned long long)(expect_free >> 20));
  uint64_t prev = g.vram_free;
  bool grew = false;
  int stable = 0;
  for (int i = 0; i < 48; ++i) { // до 24 с: роутер выгружает асинхронно
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    uint64_t v = settle_view(card).vram_free; // вид цели, не карта 0
    if (v > prev + (32ull << 20)) { grew = true; stable = 0; } else ++stable;
    prev = std::max(prev, v);
    if (expect_free && prev + (256ull << 20) >= expect_free) return; // вернулось
    if (grew && stable >= 3) return; // рост был и 1,5 с прекратился — всё, что ушло, вернулось
    if (!reachable && stable >= 6) return; // цель недостижима и 3 с тишины — ждём лишь стабилизации
  }
  logf("vram: память убитого процесса не вернулась за 24 с (свободно %llu из %llu МиБ) — грузимся так",
       (unsigned long long)(prev >> 20), (unsigned long long)(g.vram_total >> 20));
}

} // namespace lh::sup
