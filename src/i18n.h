// Выбор языка интерфейса и перевод строк — ЕДИНСТВЕННАЯ таблица проекта.
// Msgid-стиль: русский литерал — ключ tr(), английский берётся из kTable;
// промах возвращает русский (прогрессивное покрытие: обёртывание строк и
// наполнение таблицы могут расходиться по коммитам — дыра видна глазами,
// а не молча ломает вывод). Формат-плейсхолдеры (%s/%d) в английских
// строках совпадают с ключом дословно: расхождение ловит -Werror=format
// на месте вызова, а не в рантайме.
//
// Язык процесса (правило проекта: русская машина говорит по-русски, любая
// другая — по-английски) решается по приоритету: LH_LANG (ru|en, форс) →
// хинт конфига демона (set_lang_hint, его зовёт config::set_current на
// старте и при hot-правке lang) → локаль машины (LC_ALL → LC_MESSAGES →
// LANG, префикс «ru»; «ru_RU.UTF-8» обрезается до префиса кодировкой
// после «ru»). Ни setlocale, ни системные каталоги переводов: таблица
// вкомпилирована — статический бинарник ни от чего не зависит.
//
// Заголовок самодостаточен (только std): lhctl не линкует config.cpp,
// поэтому хинт демона заходит сеттером, а не чтением конфига.
#pragma once
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>

namespace lh::i18n {

// Индексы стабильны: 0 — русский (канон, он же ключ таблицы), 1 — английский.
inline constexpr int kRu = 0;
inline constexpr int kEn = 1;

struct Row {
  const char* ru;  // ключ — дословный русский литерал из кода
  const char* en;  // перевод; плейсхолдеры совпадают с ключом
};

// Единственная таблица переводов проекта. Наполняется по мере обёртывания
// строк; незавёрнутые литералы остаются русскими.
inline constexpr std::array kTable{
    // --- fmt.h: величины и общие карточки -----------------------------------
    Row{"%.1f ГБ", "%.1f GB"},
    Row{"%.0f МБ", "%.0f MB"},
    Row{"%llu КБ", "%llu KB"},
    Row{"%lluК", "%lluK"},
    Row{"%.1fМ", "%.1fM"},
    Row{"%.0fК", "%.0fK"},
    Row{"(не загрузилась)", "(failed to load)"},
    Row{"(в памяти)", "(in memory)"},
    Row{"(грузится…)", "(loading…)"},
    Row{"(ошибка)", "(error)"},
    Row{"контекст", "context"},
    Row{"слоёв на GPU", "GPU layers"},
    Row{"блоков экспертов в RAM", "expert blocks in RAM"},
    Row{"тензоры вне GPU", "tensors off GPU"},
    Row{"черновик", "draft"},
    Row{"тип спекуляции", "speculation type"},
    Row{"слоёв черновика на GPU", "draft GPU layers"},
    Row{"черновик на карте", "draft on GPU"},
    Row{"устройство", "device"},
    Row{"сплит карт", "card split"},
    Row{"vision-проекция", "vision projection"},
    Row{"слотов parallel", "parallel slots"},
    Row{"дом-карта", "home GPU"},
    Row{"Модель: ", "Model: "},
    Row{"Имя", "Name"},
    Row{"Файл", "File"},
    Row{"Размер", "Size"},
    Row{"Архитектура", "Architecture"},
    Row{"  ·  квант ", "  ·  quant "},
    Row{"Слоёв", "Layers"},
    Row{"Контекст", "Context"},
    Row{"до ", "up to "},
    Row{" токенов", " tokens"},
    Row{"  ·  карта ", "  ·  GPU "},
    Row{"Состояние", "State"},
    Row{" (потолок ", " (max "},
    Row{" — не влезает", " — doesn't fit"},
    Row{"План VRAM", "VRAM plan"},
    Row{"Ручной ctx", "Manual ctx"},
    Row{"  ·  снять: lhctl ctx <модель> auto", "  ·  clear: lhctl ctx <model> auto"},
    Row{"  ·  глобальный [general]", "  ·  global [general]"},
    Row{"  ·  переопределение секции", "  ·  section override"},
    Row{"  ·  сменить: M в карточке", "  ·  change: M in the card"},
    Row{"  ·  без --mmproj  ·  включить: V (lhctl vision <модель> auto)",
        "  ·  no --mmproj  ·  enable: V (lhctl vision <model> auto)"},
    Row{"»  ·  выключить: V (lhctl vision <модель> off)",
        "»  ·  disable: V (lhctl vision <model> off)"},
    Row{"  ·  проекций рядом нет", "  ·  no projections nearby"},
    Row{"Эмбеддер", "Embedder"},
    Row{"да (/v1/embeddings)", "yes (/v1/embeddings)"},
    Row{"Движок llama.cpp", "llama.cpp engine"},
    Row{"Пин", "Pin"},
    Row{"Активная", "Active"},
    Row{"Выбрана", "Selected"},
    Row{"  ·  сбросить: lhctl engine use auto", "  ·  reset: lhctl engine use auto"},
    Row{"Установка", "Installing"},
    Row{"Прогресс", "Progress"},
    Row{" из ", " of "},
    Row{"Ошибка", "Error"},
    Row{"⚠ пин и установленный движок расходятся — «установлено» ниже показывает факт",
        "⚠ the pin and the installed engine diverge — «installed» below shows the fact"},
    Row{"СБОРКА", "BUILD"},
    Row{"РАЗМЕР", "SIZE"},
    Row{"СТАТУС", "STATUS"},
    Row{"✓ установлена", "✓ installed"},
    Row{" ≠ пин ", " ≠ pin "},
    Row{"  ·  пиннута", "  ·  pinned"},
    Row{"пиннута", "pinned"},
    Row{"  ·  не установлена", "  ·  not installed"},
    Row{"БЕЗ ПИНА", "UNPINNED"},
    Row{"  ← работает", "  ← running"},
    Row{"установить: lhctl engine install <сборка>  ·  выбрать: lhctl engine use <сборка>",
        "install: lhctl engine install <build>  ·  select: lhctl engine use <build>"},
    Row{"обновить пин: lhctl engine update", "update pin: lhctl engine update"},
    Row{"%.1f МБ/с", "%.1f MB/s"},
    Row{"  ·  ~%lu мин %lu с", "  ·  ~%lu min %lu s"},
    Row{"  ·  ~%lu с", "  ·  ~%lu s"},
    // --- lhctl.cpp: панели и команды ----------------------------------------
    Row{"модели загружены, готов к запросам", "models loaded, ready for requests"},
    Row{"идёт загрузка модели…", "loading a model…"},
    Row{"ошибка: ", "error: "},
    Row{"модели выгружены", "models unloaded"},
    Row{"  ·  идёт генерация", "  ·  generating"},
    Row{"Роутер", "Router"},
    Row{"  ·  одновременно до ", "  ·  up to "},
    Row{" моделей", " models"},
    Row{"Режим", "Mode"},
    Row{"классический (одна модель)", "classic (single model)"},
    Row{"выгрузка после ", "unloads after "},
    Row{" мин простоя", " min idle"},
    Row{"выгрузка по простою выключена", "idle unload disabled"},
    Row{"  ·  авто-замена при загрузке", "  ·  auto-swap on load"},
    Row{"Сон", "Sleep"},
    Row{"Железо:", "Hardware:"},
    Row{" · по железу", " · by hardware"},
    Row{" · выбрана: ", " · selected: "},
    Row{"Движок:", "Engine:"},
    Row{"Основная", "Primary"},
    Row{"МОДЕЛЬ", "MODEL"},
    Row{"СОСТОЯНИЕ", "STATE"},
    Row{"РОЛЬ", "ROLE"},
    Row{"эмбеддер", "embedder"},
    Row{"Слот ", "Slot "},
    Row{"КАРТА", "GPU"},
    Row{"КВАНТ", "QUANT"},
    Row{"ПЛАН CTX", "CTX PLAN"},
    Row{"VRAM-ОЦЕНКА", "VRAM EST"},
    Row{"ИМЯ", "NAME"},
    Row{"РЕПОЗИТОРИЙ", "REPOSITORY"},
    Row{"ПРИМЕЧАНИЕ", "NOTE"},
    Row{"Поиск: «", "Search: «"},
    Row{"Железо", "Hardware"},
    Row{"  ·  свободно ", "  ·  free "},
    Row{"ПОМЕЩАЕТСЯ", "FITS"},
    Row{"СКАЧИВАНИЙ", "DOWNLOADS"},
    Row{"✓ целиком", "✓ whole"},
    Row{"частично", "partial"},
    Row{"частично (плотная)", "partial (dense)"},
    Row{"неизвестно", "unknown"},
    Row{"✗ не влезает", "✗ doesn't fit"},
    Row{"нет оценки", "no estimate"},
    Row{"установить: lhctl pull <org/repo>", "install: lhctl pull <org/repo>"},
    Row{"\r\x1b[2K%s загружаю «%s»  ·  %lu с", "\r\x1b[2K%s loading «%s»  ·  %lu s"},
    Row{"\nзагрузка идёт дольше 15 минут — сдаюсь", "\nloading for over 15 minutes — giving up"},
    Row{" загружена: «", " loaded: «"},
    Row{" с", " s"},
    Row{" готово", " done"},
    Row{"\r\x1b[2K\nмодель не загрузилась — lhctl status покажет причину",
        "\r\x1b[2K\nmodel failed to load — lhctl status will show the reason"},
    Row{"модель не загрузилась", "model failed to load"},
    Row{" загружена  ·  ", " loaded  ·  "},
    Row{"llmhost недоступен на 127.0.0.1:%d\n", "llmhost unreachable at 127.0.0.1:%d\n"},
    Row{"lhctl: неверный токен (401) — см. data/token.txt\n",
        "lhctl: wrong token (401) — see data/token.txt\n"},
    Row{"модель «%s» не найдена — список: lhctl models\n",
        "model «%s» not found — list: lhctl models\n"},
    Row{"загрузка не началась: ", "load didn't start: "},
    Row{"нет ответа", "no response"},
    Row{"(думаю…)\n", "(thinking…)\n"},
    Row{"ошибка запроса: ", "request error: "},
    Row{"пустой ответ модели", "empty model response"},
    Row{"не удалось узнать OpenAI-порт у llmhost", "couldn't get the OpenAI port from llmhost"},
    Row{"Чат с llmhost — модель по умолчанию (последняя используемая)",
        "Chat with llmhost — default model (last used)"},
    Row{"Чат с llmhost — модель: ", "Chat with llmhost — model: "},
    Row{"/bye — выход, /clear — новая сессия, Ctrl+D — выход.\n"
         "Многострочная вставка уходит одним сообщением.\n",
         "/bye — quit, /clear — new session, Ctrl+D — quit.\n"
         "A multi-line paste is sent as one message.\n"},
    Row{"\nВы> ", "\nYou> "},
    Row{"(сессия сброшена)\n", "(session cleared)\n"},
    Row{"llmhost недоступен на 127.0.0.1:%d (%s)\n", "llmhost unreachable at 127.0.0.1:%d (%s)\n"},
    Row{"демон не отвечает на 127.0.0.1:", "the daemon is not responding on 127.0.0.1:"},
    Row{"lhctl: сервер недоступен на 127.0.0.1:%d (%s)\n",
        "lhctl: server unreachable at 127.0.0.1:%d (%s)\n"},
    Row{"lhctl: сервер ответил %d\n", "lhctl: server replied %d\n"},
    Row{"  [карта ", "  [GPU "},
    Row{"completion: укажите bash или powershell\n", "completion: specify bash or powershell\n"},
    Row{"show: укажите модель (lhctl models — список)\n", "show: specify a model (lhctl models — list)\n"},
    Row{"модель «%s» не найдена — lhctl models покажет список\n",
        "model «%s» not found — lhctl models will list them\n"},
    Row{"модель:    ", "model:     "},
    Row{"файл:      ", "file:      "},
    Row{"размер:    %.1f ГБ\n", "size:      %.1f GB\n"},
    Row{"архитектура: %s, квант %s\n", "architecture: %s, quant %s\n"},
    Row{"слоёв:     %d\n", "layers:    %d\n"},
    Row{"контекст обучения: %llu токенов\n", "train context: %llu tokens\n"},
    Row{"состояние: ", "state: "},
    Row{" (карта %d)", " (GPU %d)"},
    Row{"состояние: загружена", "state: loaded"},
    Row{"эмбеддер:  да (/v1/embeddings)", "embedder:  yes (/v1/embeddings)"},
    Row{"vision:     off (переопределение) — без --mmproj", "vision:     off (override) — no --mmproj"},
    Row{"vision:     auto — «", "vision:     auto — «"},
    Row{"» прицепится сама", "» will attach itself"},
    Row{"vision:     auto — проекций рядом нет", "vision:     auto — no projections nearby"},
    Row{"thinking:   auto — решает движок по шаблону",
        "thinking:   auto — the engine decides by template"},
    Row{"  ·  без рассуждений  ·  вернуть: T (lhctl thinking <модель> auto)",
        "  ·  no reasoning  ·  restore: T (lhctl thinking <model> auto)"},
    Row{"  ·  форс --reasoning on  ·  выключить: T (lhctl thinking <модель> off)",
        "  ·  force --reasoning on  ·  disable: T (lhctl thinking <model> off)"},
    Row{"  ·  решает движок по шаблону  ·  выключить: T (lhctl thinking <модель> off)",
        "  ·  the engine decides by template  ·  disable: T (lhctl thinking <model> off)"},
    Row{"thinking:   %s (переопределение)\n", "thinking:   %s (override)\n"},
    Row{"thinking: укажите модель (lhctl models — список)\n",
        "thinking: specify a model (lhctl models — list)\n"},
    Row{"«%s»: thinking auto — решает движок по шаблону\n",
        "«%s»: thinking auto — the engine decides by template\n"},
    Row{"«%s»: thinking = %s (переопределение)\n",
        "«%s»: thinking = %s (override)\n"},
    Row{"установить: lhctl thinking %s auto|on|off\n",
        "set: lhctl thinking %s auto|on|off\n"},
    Row{"thinking снят: режим решает движок по шаблону\n",
        "thinking cleared: the engine decides by template\n"},
    Row{"thinking-mode = %s → llmhost.conf [%s] — ребёнок с --reasoning %s\n",
        "thinking-mode = %s → llmhost.conf [%s] — child with --reasoning %s\n"},
    Row{"thinking-mode записан", "thinking-mode saved"},
    Row{"llmhost — метрики движка", "llmhost — engine metrics"},
    Row{"  ·  обновление живьём (Ctrl-C — выход)",
        "  ·  live refresh (Ctrl-C to exit)"},
    Row{"источник: /metrics детей; декод/префилл — из таймингов движка, промпт — из usage прокси (с кэшированными)",
        "source: child /metrics; decode/prefill — engine timings, prompt — proxy usage (cached included)"},
    Row{"модель                                   декод тк/с  префилл  приём  декод ткн  промпт ткн   запр.",
        "model                                     dec tok/s    prefill    acc   dec tokns  prompt tkns   reqs"},
    Row{"мгновенно ", "instant "},
    Row{"(нет моделей с трафиком — счётчики пусты)",
        "(no models with traffic — counters are empty)"},
    Row{" llmhost — метрики движка ", " llmhost — engine metrics "},
    Row{"источник: /metrics детей через демон; обновление каждые 2 с; промпт прокси — с кэшированными, движок — без",
        "source: child /metrics via the daemon; refresh every 2 s; proxy prompt — cached included, engine — excluded"},
    Row{"модель", "model"},
    Row{"декод тк/с", "dec tok/s"},
    Row{"префилл", "prefill"},
    Row{"приём", "accept"},
    Row{"декод ткн", "dec tokns"},
    Row{"промпт", "prompt"},
    Row{"запр.", "reqs"},
    Row{"сейчас ", "now "},
    Row{"обновить", "refresh"},
    Row{"rm: укажите модель (lhctl models — список)\n", "rm: specify a model (lhctl models — list)\n"},
    Row{"MTP-модуль", "MTP module"},
    Row{"шард", "shard"},
    Row{"недокачка", "partial download"},
    Row{"rm: ввод не интерактивен — подтвердите флагом --yes\n",
        "rm: input is not interactive — confirm with the --yes flag\n"},
    Row{"Удалить «%s» (%s", "Delete «%s» (%s"},
    Row{" + спутники %s", " + satellites %s"},
    Row{") безвозвратно? [y/N]\n", ") permanently? [y/N]\n"},
    Row{"отменено", "cancelled"},
    Row{"удалено безвозвратно: %s — файлов %zu, освобождено %s\n",
        "deleted permanently: %s — %zu files, freed %s\n"},
    Row{"удалено безвозвратно: %s\n", "deleted permanently: %s\n"},
    Row{"llmhost недоступен", "llmhost unreachable"},
    Row{"ctx: укажите модель (lhctl models — список)\n", "ctx: specify a model (lhctl models — list)\n"},
    Row{"«%s»: ручной ctx-size %d (план умной настройки %s)\n",
        "«%s»: manual ctx-size %d (smart plan %s)\n"},
    Row{"«%s»: ручной ctx нет — умная настройка решает сама (план %s)\n",
        "«%s»: no manual ctx — smart tuning decides (plan %s)\n"},
    Row{"установить: lhctl ctx %s <токены>  ·  убрать: lhctl ctx %s auto\n",
        "set: lhctl ctx %s <tokens>  ·  clear: lhctl ctx %s auto\n"},
    Row{"ctx: значение — число токенов или auto (получено «%s»)\n",
        "ctx: value is a token count or auto (got «%s»)\n"},
    Row{"ctx: разумный диапазон 256…2097152 токенов", "ctx: sensible range is 256…2097152 tokens"},
    Row{" (план ", " (plan "},
    Row{"ручной ctx снят: «%s» — умная настройка решает сама%s\n",
        "manual ctx cleared: «%s» — smart tuning decides%s\n"},
    Row{"пресет пересобирается — дети перезапустятся плавно", "preset rebuilt — children restart smoothly"},
    Row{"предупреждение: ", "warning: "},
    Row{"ручной ctx снят", "manual ctx cleared"},
    Row{"ctx-size записан", "ctx-size saved"},
    Row{"mtp: укажите модель (lhctl models — список)\n", "mtp: specify a model (lhctl models — list)\n"},
    Row{"«%s»: mtp-mode не задан — глобальный «%s»\n", "«%s»: mtp-mode not set — global «%s»\n"},
    Row{"«%s»: mtp-mode = %s (переопределение)\n", "«%s»: mtp-mode = %s (override)\n"},
    Row{"установить: lhctl mtp %s auto|on|off\n", "set: lhctl mtp %s auto|on|off\n"},
    Row{"mtp: значение — auto, on или off (получено «%s»)\n", "mtp: value is auto, on or off (got «%s»)\n"},
    Row{"mtp-mode снят: «%s» — решает глобальный [general]\n", "mtp-mode cleared: «%s» — global [general] decides\n"},
    Row{"слот в сон — первый запрос поднимет ребёнка с новым режимом",
        "slot to sleep — the first request will wake the child with the new mode"},
    Row{"mtp-mode записан", "mtp-mode saved"},
    Row{"vision: укажите модель (lhctl models — список)\n", "vision: specify a model (lhctl models — list)\n"},
    Row{"«%s»: vision auto — проекция прицепится сама%s\n", "«%s»: vision auto — a projection will attach itself%s\n"},
    Row{" (проекций рядом нет)", " (no projections nearby)"},
    Row{" (проекция «", " (projection «"},
    Row{"«%s»: vision = %s (переопределение) — без --mmproj\n", "«%s»: vision = %s (override) — no --mmproj\n"},
    Row{"установить: lhctl vision %s auto|off\n", "set: lhctl vision %s auto|off\n"},
    Row{"vision: значение — auto или off (получено «%s»)\n", "vision: value is auto or off (got «%s»)\n"},
    Row{"vision снят: «%s» — своя проекция прицепится сама\n", "vision cleared: «%s» — its own projection will attach itself\n"},
    Row{"vision-mode = off → llmhost.conf [%s] — ребёнок без --mmproj\n", "vision-mode = off → llmhost.conf [%s] — child without --mmproj\n"},
    Row{"vision-mode записан", "vision-mode saved"},
    Row{"[general] в %s:\n", "[general] in %s:\n"},
    Row{"· после рестарта демона", "· after daemon restart"},
    Row{"set: ответ демона не читается", "set: daemon response unreadable"},
    Row{"set: укажите значение (lhctl set — список ключей)", "set: specify a value (lhctl set — key list)"},
    Row{"вступит после рестарта: systemctl --user restart llmhostd", "takes effect after restart: systemctl --user restart llmhostd"},
    Row{"применено — пресет пересобран, дети перезапустятся плавно", "applied — preset rebuilt, children restart smoothly"},
    Row{"записано", "saved"},
    Row{"aliases: сервер ответил %d\n", "aliases: server replied %d\n"},
    Row{"Короткие имена для «lhctl pull <имя>» — квант под вашу карту выбирается сам:\n",
        "Short names for «lhctl pull <name>» — the quant for your GPU is picked automatically:\n"},
    Row{"\nПолные репозитории (org/repo) тоже принимаются; файл — вторым аргументом или wildcard-фильтром.",
        "\nFull repositories (org/repo) work too; the file is a second argument or a wildcard filter."},
    Row{"Короткие имена для «lhctl pull <имя>» (квант под вашу карту выбирается сам):\n",
        "Short names for «lhctl pull <name>» (the quant for your GPU is picked automatically):\n"},
    Row{"aliases: ответ не разобрался", "aliases: response unreadable"},
    Row{"резолв репозитория…", "resolving repository…"},
    Row{"\r  ↓ %-40s %2llu%%   %6.1f МБ/с   ", "\r  ↓ %-40s %2llu%%   %6.1f MB/s   "},
    Row{"\r  ↓ %-40s %6.1f МБ/с   ", "\r  ↓ %-40s %6.1f MB/s   "},
    Row{"✓ скачано: ", "✓ downloaded: "},
    Row{"\rскачано: %s\n", "\rdownloaded: %s\n"},
    Row{"ошибка скачивания", "download error"},
    Row{"скачивание не идёт", "no download in progress"},
    Row{"остановлено: недокачанный .part остался, повторный pull докачит",
        "stopped: the .part stays on disk, a repeated pull resumes it"},
    Row{"engine: сервер ответил %d\n", "engine: server replied %d\n"},
    Row{"пин манифеста:   %s\n", "manifest pin:   %s\n"},
    Row{"активная сборка: %s\n", "active build: %s\n"},
    Row{"выбрана:         %s (снять: lhctl engine use auto)\n", "selected:        %s (clear: lhctl engine use auto)\n"},
    Row{"установка: %s «%s»\n", "installing: %s «%s»\n"},
    Row{"установлена", "installed"},
    Row{"РАСХОЖДЕНИЕ: пин и установленный движок расходятся — скобки выше показывают факт",
        "MISMATCH: the pin and the installed engine diverge — the parentheses above show the fact"},
    Row{"engine: ответ не разобрался", "engine: response unreadable"},
    Row{"engine use: укажите сборку (lhctl engine — список) или auto",
        "engine use: specify a build (lhctl engine — list) or auto"},
    Row{"выбор снят — сборка по железу: %s\n", "selection cleared — build by hardware: %s\n"},
    Row{"выбрана сборка %s (активная: %s) — дети перезапускаются\n",
        "build %s selected (active: %s) — children restarting\n"},
    Row{"сменить обратно: lhctl engine use auto", "change back: lhctl engine use auto"},
    Row{"готово", "done"},
    Row{"engine: неизвестная подкоманда «%s» (status|use|install|update)\n",
        "engine: unknown subcommand «%s» (status|use|install|update)\n"},
    Row{"engine install: укажите сборку (lhctl engine — список)", "engine install: specify a build (lhctl engine — list)"},
    Row{"engine install: флаги --release/--pin-only/--all не применяются",
        "engine install: the --release/--pin-only/--all flags do not apply"},
    Row{"engine update: указана сборка и --all одновременно", "engine update: a build and --all given at once"},
    Row{"engine: демон не определил активную сборку — укажите явно: lhctl engine update <сборка>",
        "engine: the daemon could not determine the active build — specify explicitly: lhctl engine update <build>"},
    Row{"пин уже %s — обновление не требуется\n(переустановить текущий пин: lhctl engine install <сборка>)\n",
        "pin already %s — no update needed\n(to reinstall the current pin: lhctl engine install <build>)\n"},
    Row{"пин манифеста: %s → %s\n", "manifest pin: %s → %s\n"},
    Row{"файл: %s\n", "file: %s\n"},
    Row{"достроены из ассетов релиза: %s\n", "built up from release assets: %s\n"},
    Row{"engine: нет пиннутых сборок для установки", "engine: no pinned builds to install"},
    Row{"ставлю «%s»\n", "installing «%s»\n"},
    Row{"lhctl: неверный токен (401)\n", "lhctl: wrong token (401)\n"},
    Row{"распаковка tar.gz…", "extracting tar.gz…"},
    Row{"\r  ↓ %-10s %2llu%%   %6.1f МБ/с   ", "\r  ↓ %-10s %2llu%%   %6.1f MB/s   "},
    Row{"\r  ↓ %-10s %6.1f МБ/с   ", "\r  ↓ %-10s %6.1f MB/s   "},
    Row{"\r  ↓ %-10s распаковка…        ", "\r  ↓ %-10s extracting…        "},
    Row{"✓ готово: ", "✓ done: "},
    Row{"\rготово: «%s»%70s\n", "\rdone: «%s»%70s\n"},
    Row{"\nустановка не началась за 30 с — смотрите lhctl engine", "\nthe install did not start within 30 s — see lhctl engine"},
    Row{"ТОКЕН", "TOKEN"},
    Row{"СОЗДАН", "CREATED"},
    Row{"ПОСЛЕДНИЙ ИСПОЛЬЗОВАН", "LAST USED"},
    Row{"токенов нет — lhctl tokens add <имя>", "no tokens — lhctl tokens add <name>"},
    Row{"tokens add: укажите имя (lhctl tokens add rag2)", "tokens add: specify a name (lhctl tokens add rag2)"},
    Row{"Токен создан: ", "Token created: "},
    Row{"Значение", "Value"},
    Row{"клиенты передают его как Bearer-ключ OpenAI-порта;", "clients pass it as the Bearer key of the OpenAI port;"},
    Row{"значение хранится и показывается снова: lhctl tokens", "the value is stored and shown again: lhctl tokens"},
    Row{"tokens rm: укажите имя (lhctl tokens — список)", "tokens rm: specify a name (lhctl tokens — list)"},
    Row{"отозван: %s\n", "revoked: %s\n"},
    Row{"tokens: неизвестная подкоманда «%s» (list|add|rm)\n", "tokens: unknown subcommand «%s» (list|add|rm)\n"},
    Row{"load: укажите .gguf-файл или имя модели", "load: specify a .gguf file or a model name"},
    Row{"search: укажите запрос (lhctl search qwen3)", "search: specify a query (lhctl search qwen3)"},
    Row{"лишний аргумент: %s\n", "extra argument: %s\n"},
    Row{"неизвестная команда: %s\n", "unknown command: %s\n"},
    // usage() — вся справка одним ключом (склеенный литерал).
    Row{
        "lhctl — управление llmhost из консоли\n"
        "  lhctl                    полноэкранный интерфейс управления (в духе mc:\n"
        "                           реестр с курсором, загрузка/выгрузка, поиск,\n"
        "                           прогресс; F1 — помощь) — только в терминале\n"
        "  lhctl status [-w сек]    приборная панель (загруженные модели, VRAM-бары);\n"
        "                           -w — обновлять живьём (resize окна не ломает)\n"
        "  lhctl models [--raw]     список моделей реестра (--raw — id построчно)\n"
        "  lhctl show <модель>      карточка модели (архитектура, квант, контекст)\n"
        "  lhctl ctx <модель> [N|auto]\n"
        "                           ручной контекст модели: пишет ctx-size в\n"
        "                           llmhost.conf и пересобирает пресет сразу;\n"
        "                           без значения — показать текущее\n"
        "  lhctl mtp <модель> [auto|on|off]\n"
        "                           MTP-спекуляция модели (дополнение токенов\n"
        "                           черновиком): auto — как решает умная настройка,\n"
        "                           on/off — форс; без значения — показать текущее\n"
        "  lhctl vision <модель> [auto|off]\n"
        "                           vision-режим модели (картинки через mmproj):\n"
        "                           auto — своя проекция прицепится сама, off —\n"
        "                           грузить без неё; без значения — показать текущее\n"
        "  lhctl thinking <модель> [auto|on|off]\n"
        "                           thinking-режим модели (рассуждения перед\n"
        "                           ответом): auto — решает движок по шаблону,\n"
        "                           on/off — форс; без значения — показать текущее\n"
        "  lhctl image [load [модель] | pull [имя] | rm <модель> | unload]\n"
        "                           генерация изображений (движок sd-server):\n"
        "                           без аргумента — статус и каталог моделей,\n"
        "                           load — загрузить (к модели обращается\n"
        "                           /v1/images/* OpenAI-порта), unload — остановить,\n"
        "                           pull — скачать художника (без имени — каталог\n"
        "                           встроенных; или org/repo с раскладкой\n"
        "                           recipes/ + категории), rm — удалить с диска\n"
        "  lhctl set [ключ [значение]]\n"
        "                           настройки [general] llmhost.conf (единый\n"
        "                           конфиг): без аргументов — вся таблица,\n"
        "                           с аргументами — записать; применяется на\n"
        "                           лету (порты — после рестарта демона)\n"
        "  lhctl search <запрос>    поиск по каталогу HuggingFace: размер, оценка\n"
        "                           VRAM, влезет ли на ваши карты\n"
        "  lhctl tokens [list]      именованные токены /v1 (значение — в колонке)\n"
        "  lhctl tokens add <имя>   выдать токен (значение генерируется и\n"
        "                           сохраняется; пока токены есть, /v1 — только по ним)\n"
        "  lhctl tokens rm <имя>    отозвать токен\n"
        "  lhctl rm <модель>        удалить модель С ДИСКА безвозвратно: вместе\n"
        "                           с mmproj/MTP и шардами, скачанными с ней\n"
        "                           (спросит подтверждение, --yes — без вопроса)\n"
        "  lhctl chat [модель] [вопрос]\n"
        "                           чат: без вопроса — диалог со стримом (/bye —\n"
        "                           выход); с вопросом — один ответ; без модели —\n"
        "                           последняя используемая\n"
        "  lhctl load <файл|имя>    загрузить модель (.gguf или имя из реестра)\n"
        "  lhctl pull [имя] [файл]  скачать модель: короткое имя (без аргумента —\n"
        "                           список; квант подберётся под вашу карту) или\n"
        "                           org/repo; файл — имя или wildcard-фильтр\n"
        "  lhctl engine             движок: пин манифеста, сборки, что ставится\n"
        "  lhctl engine install <сборка>\n"
        "                           установить сборку из текущего пина (скачивание\n"
        "                           с проверкой SHA256; модели должны быть\n"
        "                           выгружены)\n"
        "  lhctl engine update [сборка]\n"
        "                           обновить движок: пин последнего релиза\n"
        "                           llama.cpp и установка активной сборки\n"
        "                           (--release bNNNNN — конкретный релиз,\n"
        "                           --pin-only — только манифест без установки,\n"
        "                           --all — все пиннутые сборки)\n"
        "  lhctl unload [id]        выгрузить одну (id) или все\n"
        "  lhctl quit               завершить llmhost\n"
        "  lhctl version            версия\n"
        "  lhctl completion bash|powershell\n"
        "                           скрипт автодополнения (TAB подсказывает модели)\n"
        "Опции: --port N (7331), --token XXX (иначе data/token.txt рядом с lhctl\n"
        "или $XDG_DATA_HOME/llmhost/token.txt), --json — машинный вывод даже в\n"
        "терминале.\n"
        "Окружение LHCTL_PORT / LHCTL_TOKEN задаёт значения по умолчанию.\n"
        "Коды выхода: 0 — успех; 1 — сервер недоступен; 2 — неверный токен (401);\n"
        "3 — ошибка запроса (404/500); 4 — ошибка аргументов.",
        "lhctl — control llmhost from the console\n"
        "  lhctl                    full-screen control interface (mc-style:\n"
        "                           registry with a cursor, load/unload, search,\n"
        "                           progress; F1 — help) — terminal only\n"
        "  lhctl status [-w sec]    dashboard (loaded models, VRAM bars);\n"
        "                           -w — refresh live (window resize is safe)\n"
        "  lhctl models [--raw]     registry list (--raw — ids line by line)\n"
        "  lhctl show <model>       model card (architecture, quant, context)\n"
        "  lhctl ctx <model> [N|auto]\n"
        "                           manual model context: writes ctx-size to\n"
        "                           llmhost.conf and rebuilds the preset at once;\n"
        "                           without a value — show the current one\n"
        "  lhctl mtp <model> [auto|on|off]\n"
        "                           MTP speculation of the model (token completion\n"
        "                           by a draft): auto — as smart tuning decides,\n"
        "                           on/off — force; without a value — show current\n"
        "  lhctl vision <model> [auto|off]\n"
        "                           vision mode of the model (images via mmproj):\n"
        "                           auto — its own projection attaches itself, off —\n"
        "                           load without it; without a value — show current\n"
        "  lhctl thinking <model> [auto|on|off]\n"
        "                           model thinking mode (reasoning before the\n"
        "                           answer): auto — the engine decides by template,\n"
        "                           on/off — force; without a value — show current\n"
        "  lhctl metrics [-w sec]\n"
        "                           exact engine metrics per model (decode and\n"
        "                           prefill throughput, speculative acceptance,\n"
        "                           tokens): without -w — one snapshot, with -w —\n"
        "                           live table with an instantaneous tps sparkline\n"
        "  lhctl image [load [model] | pull [name] | rm <model> | unload]\n"
        "                           image generation (sd-server engine): no\n"
        "                           argument — status and the model catalog,\n"
        "                           load — load (generation requests come via\n"
        "                           the /v1/images/* OpenAI port), unload — stop,\n"
        "                           pull — fetch an artist (no name — the built-in\n"
        "                           catalog; or org/repo with the recipes/ +\n"
        "                           categories layout), rm — delete from disk\n"
        "  lhctl set [key [value]]\n"
        "                           [general] settings of llmhost.conf (the single\n"
        "                           config): no arguments — the whole table,\n"
        "                           with arguments — write; applies on the fly\n"
        "                           (ports — after a daemon restart)\n"
        "  lhctl search <query>     search the HuggingFace catalog: size, VRAM\n"
        "                           estimate, will it fit your cards\n"
        "  lhctl tokens [list]      named /v1 tokens (the value is in a column)\n"
        "  lhctl tokens add <name>  issue a token (the value is generated and\n"
        "                           stored; while tokens exist, /v1 accepts only them)\n"
        "  lhctl tokens rm <name>   revoke a token\n"
        "  lhctl rm <model>        delete a model FROM DISK permanently: together\n"
        "                           with its mmproj/MTP and downloaded shards\n"
        "                           (asks for confirmation, --yes — no question)\n"
        "  lhctl chat [model] [question]\n"
        "                           chat: without a question — a streamed dialog\n"
        "                           (/bye — quit); with a question — one answer;\n"
        "                           without a model — the last used one\n"
        "  lhctl load <file|name>   load a model (.gguf or a registry name)\n"
        "  lhctl pull [name] [file]  download a model: a short name (no argument —\n"
        "                           the list; the quant is picked for your GPU) or\n"
        "                           org/repo; file — a name or a wildcard filter\n"
        "  lhctl engine             engine: manifest pin, builds, what installs\n"
        "  lhctl engine install <build>\n"
        "                           install a build from the current pin (download\n"
        "                           with SHA256 verification; models must be\n"
        "                           unloaded)\n"
        "  lhctl engine update [build]\n"
        "                           update the engine: pin the latest llama.cpp\n"
        "                           release and install the active build\n"
        "                           (--release bNNNNN — a specific release,\n"
        "                           --pin-only — the manifest only, no install,\n"
        "                           --all — all pinned builds)\n"
        "  lhctl unload [id]        unload one (id) or all\n"
        "  lhctl quit               stop llmhost\n"
        "  lhctl version            version\n"
        "  lhctl completion bash|powershell\n"
        "                           completion script (TAB suggests models)\n"
        "Options: --port N (7331), --token XXX (otherwise data/token.txt next to lhctl\n"
        "or $XDG_DATA_HOME/llmhost/token.txt), --json — machine output even in a\n"
        "terminal.\n"
        "The LHCTL_PORT / LHCTL_TOKEN environment sets the defaults.\n"
        "Exit codes: 0 — success; 1 — server unreachable; 2 — wrong token (401);\n"
        "3 — request error (404/500); 4 — argument error."},

    // image-слот: статус/загрузка/выгрузка + pull/rm художников
    Row{"image: unload не удался\n", "image: unload failed\n"},
    Row{"image-движок остановлен\n", "image engine stopped\n"},
    Row{"image: загрузка…\n", "image: loading…\n"},
    Row{"image: «%s» загружена (порт %d)\n", "image: «%s» loaded (port %d)\n"},
    Row{"image: ошибка — %s\n", "image: error — %s\n"},
    Row{"image: загрузка не завершилась за 240 с\n",
        "image: loading did not finish within 240 s\n"},
    Row{"image: статус — без аргумента; load [модель], pull [имя], rm <модель>, unload\n",
        "image: status — no argument; load [model], pull [name], rm <model>, unload\n"},
    Row{"Каталог художников для «lhctl image pull <имя>»:\n",
        "Artist catalog for «lhctl image pull <name>»:\n"},
    Row{"\nСвой источник — репозиторий org/repo с раскладкой: recipes/*.json + "
        "diffusion_models/ + text_encoders/ + vae/ (по одному файлу на слот).",
        "\nOwn source — an org/repo repository with the layout: recipes/*.json + "
        "diffusion_models/ + text_encoders/ + vae/ (one file per slot)."},
    Row{"image rm: укажите модель (lhctl image — список)\n",
        "image rm: specify a model (lhctl image — the list)\n"},
    Row{"в каталоге image-моделей нет «%s» — lhctl image покажет список\n",
        "no «%s» in the image-model catalog — lhctl image shows the list\n"},
    Row{"image rm: ввод не интерактивен — подтвердите флагом --yes\n",
        "image rm: input is not interactive — confirm with --yes\n"},
    Row{"Удалить «%s» (%s) с диска безвозвратно? [y/N]\n",
        "Delete «%s» (%s) from disk permanently? [y/N]\n"},
    Row{"удалено безвозвратно: %s — файлов %zu, освобождено %s%s\n",
        "deleted permanently: %s — %zu file(s), freed %s%s\n"},
    Row{", рецепт ", ", recipe "},
    Row{" удалён", " deleted"},

    // --- tui.cpp: полноэкранный интерфейс ------------------------------------
    Row{"? VRAM неизвестна", "? VRAM unknown"},
    Row{"✓ влезает, ", "✓ fits, "},
    Row{"~ частично, ", "~ partial, "},
    Row{"✗ не влезает, ", "✗ doesn't fit, "},
    Row{"реестр пуст — нечего загружать", "registry is empty — nothing to load"},
    Row{"» уже в памяти", "» is already in memory"},
    Row{"загружаю «", "loading «"},
    Row{"демон не отвечает", "the daemon is not responding"},
    Row{"реестр пуст", "registry is empty"},
    Row{"выгружаю все модели…", "unloading all models…"},
    Row{"выгрузка не прошла", "unload failed"},
    Row{"» не в памяти", "» is not in memory"},
    Row{"выгружаю «", "unloading «"},
    Row{"нужны токены числом или auto", "a token count or auto expected"},
    Row{"диапазон 256…2097152 токенов", "range 256…2097152 tokens"},
    Row{"ручной ctx снят — умная настройка решает сама", "manual ctx cleared — smart tuning decides"},
    Row{" — пресет пересобирается", " — preset rebuilding"},
    Row{"… и ещё ", "… and "},
    Row{"удалено: ", "deleted: "},
    Row{" файлов", " files"},
    Row{" — освобождено ", " — freed "},
    Row{"введите запрос и нажмите Enter", "type a query and press Enter"},
    Row{"ищу…", "searching…"},
    Row{"поиск не удался: ", "search failed: "},
    Row{"ничего не найдено", "nothing found"},
    Row{"найдено: ", "found: "},
    Row{" — вердикты по состоянию на момент поиска", " — verdicts as of the search time"},
    Row{" (свободно ", " (free "},
    Row{"; обновите поиск после выгрузки моделей", "; refresh the search after unloading models"},
    Row{"читаю ", "reading "},
    Row{"не читается: ", "can't read: "},
    Row{"не читается: ответ не JSON", "can't read: the response is not JSON"},
    Row{" · ★ Рекомендую: ", " · ★ Recommended: "},
    Row{" — наибольший квант, влезающий целиком", " — the largest quant that fits whole"},
    Row{"оценка по реальному заголовку GGUF", "estimate from the real GGUF header"},
    Row{"заголовок GGUF не дочитан — оценка грубая, по имени", "GGUF header not fully read — a rough estimate by name"},
    Row{"уже качается другое — прогресс внизу", "another download is running — progress at the bottom"},
    Row{"скачивание началось — прогресс внизу", "download started — progress at the bottom"},
    Row{"MTP-модуль отдельно не грузят — он предложится при загрузке основной модели",
        "an MTP module is not loaded on its own — it is offered when the main model loads"},
    Row{"файл ещё не скачан — сначала «Скачать выбранное»", "the file is not downloaded yet — «Download selected» first"},
    Row{"загрузка модели началась — статус на главном экране", "the model load started — status on the main screen"},
    Row{"… (старая часть ленты обрезана)", "… (older tape trimmed)"},
    Row{"» не в памяти — чат с «", "» is not in memory — chatting with «"},
    Row{"Диалог с «", "Dialog with «"},
    Row{"Enter — отправить, Esc — стоп генерации / выход, /clear — новая сессия, /bye — закрыть чат.",
        "Enter — send, Esc — stop generation / exit, /clear — new session, /bye — close the chat."},
    Row{"Модель загружается — подождите…", "The model is loading — please wait…"},
    Row{"Сессия сброшена.", "Session cleared."},
    Row{"⏹ генерация остановлена", "⏹ generation stopped"},
    Row{"токен создан — значение в списке", "token created — the value is in the list"},
    Row{"отозван: ", "revoked: "},
    Row{"грузится «", "loading «"},
    Row{"резолв…", "resolving…"},
    Row{"остановка…", "stopping…"},
    Row{"остановка скачивания…", "stopping the download…"},
    Row{"скачивание остановлено", "download stopped"},
    Row{"  Ctrl-X            остановить идущее скачивание (любой экран)",
        "  Ctrl-X            stop the running download (any screen)"},
    Row{"работает", "running"},
    Row{"загрузка модели…", "model loading…"},
    Row{"простой", "idle"},
    Row{"  ·  в памяти: ", "  ·  in memory: "},
    Row{"  ·  данные от ", "  ·  data from "},
    Row{" мин назад", " min ago"},
    Row{" с назад", " s ago"},
    Row{"реестр пуст — F7 открывает каталог моделей", "registry is empty — F7 opens the model catalog"},
    Row{"  эмбеддер", "  embedder"},
    Row{"   ↓ ещё ", "   ↓ more "},
    Row{"  ·  движок ", "  ·  engine "},
    Row{"помощь", "help"},
    Row{"статус", "status"},
    Row{"карточка", "card"},
    Row{"чат", "chat"},
    Row{"загрузить", "load"},
    Row{"токены", "tokens"},
    Row{"каталог", "catalog"},
    Row{"выгрузить", "unload"},
    Row{"удалить", "delete"},
    Row{"все", "all"},
    Row{"выход", "quit"},
    Row{" llmhost — каталог моделей ", " llmhost — model catalog "},
    Row{"Поиск по HuggingFace (GGUF): ", "HuggingFace search (GGUF): "},
    Row{"РЕЗУЛЬТАТЫ", "RESULTS"},
    Row{"(↑↓, Enter/Tab — к квантам)", "(↑↓, Enter/Tab — to quants)"},
    Row{"? размер неизвестен", "? size unknown"},
    Row{"(пока пусто)", "(empty so far)"},
    Row{"КВАНТЫ «", "QUANTS «"},
    Row{"КВАНТЫ ", "QUANTS "},
    Row{"(выберите результат)", "(pick a result)"},
    Row{"(F5 — скачать выбранное, F6 — загрузить скачанное)", "(F5 — download selected, F6 — load downloaded)"},
    Row{"скачано ✓ — нужна для картинок (вместе с моделью)", "downloaded ✓ — needed for images (together with the model)"},
    Row{"vision-проекция — нужна для картинок (качается вместе с моделью)", "vision projection — needed for images (downloads with the model)"},
    Row{"скачано ✓ — включится при загрузке модели", "downloaded ✓ — engages when the model loads"},
    Row{"MTP-модуль — ускоряет генерацию этой модели (~+25%)", "MTP module — speeds up this model's generation (~+25%)"},
    Row{"скачано ✓", "downloaded ✓"},
    Row{"влезет (MoE: эксперты на CPU)", "fits (MoE: experts on CPU)"},
    Row{"влезет частично — ", "fits partially — "},
    Row{" слоёв на GPU, хвост в RAM (медленнее)", " layers on GPU, tail in RAM (slower)"},
    Row{"можно качать", "ready to download"},
    Row{"частично — меньше контекст", "partial — less context"},
    Row{"мимо", "no fit"},
    Row{" ток.", " tok."},
    Row{"(выберите результат — Enter или Tab)", "(pick a result — Enter or Tab)"},
    Row{"(читаю…)", "(reading…)"},
    Row{"статус: ", "status: "},
    Row{"искать", "search"},
    Row{"кванты", "quants"},
    Row{"скачать", "download"},
    Row{"списки", "lists"},
    Row{"назад", "back"},
    Row{"Вы>", "You>"},
    Row{" llmhost — чат: ", " llmhost — chat: "},
    Row{"модель по умолчанию", "default model"},
    Row{"  ·  генерирую… (Esc — стоп)", "  ·  generating… (Esc — stop)"},
    Row{" — ↑ или PgUp", " — ↑ or PgUp"},
    Row{"  …Esc — стоп", "  …Esc — stop"},
    Row{"отправить", "send"},
    Row{"стоп", "stop"},
    Row{"новая сессия", "new session"},
    Row{"закрыть чат", "close the chat"},
    Row{"лента", "tape"},
    Row{" llmhost — токены доступа к /v1 ", " llmhost — /v1 access tokens "},
    Row{"Именованные ключи OpenAI-порта (api_tokens.json); пока они есть, /v1 отвечает только по ним.",
        "Named keys of the OpenAI port (api_tokens.json); while they exist, /v1 answers only to them."},
    Row{"создан ", "created "},
    Row{"  ·  использован ", "  ·  used "},
    Row{"(токенов нет — Enter создаёт первый)", "(no tokens — Enter creates the first)"},
    Row{"создать", "create"},
    Row{" llmhost — новый токен /v1 ", " llmhost — new /v1 token "},
    Row{" Имя (буквы/цифры/-_. ): ", " Name (letters/digits/-_. ): "},
    Row{"Значение сгенерируется само и показывается в списке —", "The value is generated automatically and shown in the list —"},
    Row{"клиенты передают его как Bearer-ключ OpenAI-порта.", "clients pass it as the Bearer key of the OpenAI port."},
    Row{"отмена", "cancel"},
    Row{"загруженных моделей нет — F5 загрузит выбранную", "no loaded models — F5 will load the selected one"},
    Row{"модель курсора: ", "cursor model: "},
    Row{" llmhost — статус модели ", " llmhost — model status "},
    Row{"прокрутка", "scroll"},
    Row{" llmhost — карточка модели ", " llmhost — model card "},
    Row{"mtp-режим", "mtp mode"},
    Row{" llmhost — движок llama.cpp ", " llmhost — llama.cpp engine "},
    Row{"команды", "commands"},
    Row{" llmhost — команды движка ", " llmhost — engine commands "},
    Row{"выбор", "pick"},
    Row{"выполнить", "run"},
    Row{"Обновить на конкретный релиз", "Update to a specific release"},
    Row{"Тег релиза llama.cpp (bNNNNN), пусто — последний:", "llama.cpp release tag (bNNNNN), empty — latest:"},
    Row{"пин переписывает manifest.json (SHA256 из GitHub API);", "the pin rewrites manifest.json (SHA256 from the GitHub API);"},
    Row{"установка — следом, активной сборкой", "the install follows, on the active build"},
    Row{"пин + установить", "pin + install"},
    Row{"Ручной контекст", "Manual context"},
    Row{"  сейчас: ", "  now: "},
    Row{" токенов (вручную)", " tokens (manual)"},
    Row{"авто — умная настройка", "auto — smart tuning"},
    Row{"  ·  план ", "  ·  plan "},
    Row{"Новые токены (число) или auto — снять ручной ctx:", "New tokens (a number) or auto — clear manual ctx:"},
    Row{"применить", "apply"},
    Row{"Подтверждение", "Confirmation"},
    Row{"Удалить «", "Delete «"},
    Row{") с диска", ") from disk"},
    Row{"  вместе с ней: ", "  along with it: "},
    Row{"БЕЗВОЗВРАТНО", "IRREVERSIBLE"},
    Row{"? Файл .gguf нельзя будет восстановить.", "? The .gguf file cannot be restored."},
    Row{" llmhost — управление (клавиши) ", " llmhost — controls (keys) "},
    Row{"ГЛАВНЫЙ ЭКРАН", "MAIN SCREEN"},
    Row{"  ↑↓, PgUp/PgDn     выбор модели", "  ↑↓, PgUp/PgDn     pick a model"},
    Row{"  Enter, F3         карточка модели (C — ручной ctx: число или auto;\n M — mtp-режим, V — vision, T — thinking)",
        "  Enter, F3         model card (C — manual ctx: a number or auto;\n M — mtp mode, V — vision, T — thinking)"},
    Row{"  F4                чат с выбранной моделью (лента; Enter — отправить,",
        "  F4                chat with the selected model (tape; Enter — send,"},
    Row{"                    Esc во время генерации — стоп; ↑↓/PgUp — лента;",
        "                    Esc during generation — stop; ↑↓/PgUp — the tape;"},
    Row{"  F5                загрузить выбранную", "  F5                load the selected one"},
    Row{"  F6                токены доступа к /v1 (создать/отозвать)", "  F6                /v1 access tokens (create/revoke)"},
    Row{"  F7                каталог моделей (поиск, кванты, скачать/загрузить)", "  F7                model catalog (search, quants, download/load)"},
    Row{"  F8, u             выгрузить выбранную", "  F8, u             unload the selected one"},
    Row{"  F9 → F5           команды движка: обновить пин/установить/переключить", "  F9 → F5           engine commands: re-pin/install/switch"},
    Row{"  Ctrl-D            удалить модель с диска (mmproj/MTP и шарды — с ней)",
        "  Ctrl-D            delete the model from disk (mmproj/MTP and shards go with it)"},
    Row{"  Ctrl-U            выгрузить все", "  Ctrl-U            unload all"},
    Row{"  F10, q, Ctrl-C    выход (демон продолжает работать)", "  F10, q, Ctrl-C    quit (the daemon keeps running)"},
    Row{"КАТАЛОГ (как окно каталога Windows-версии)", "CATALOG (like the Windows-version catalog window)"},
    Row{"  Enter             у результата — показать кванты; у кванта — скачать", "  Enter             on a result — show quants; on a quant — download"},
    Row{"  F5 / F6           скачать выбранное / загрузить скачанное", "  F5 / F6           download selected / load downloaded"},
    Row{"Слова — те же, что в Windows-версии: «в памяти», «грузится…»,",
        "The words are the same as in the Windows version: «in memory»,"},
    Row{"«не загрузилась». Данные обновляются сами раз в секунду.",
        "«loading…», «failed to load». The data refreshes itself once a second."},
    Row{"запись не найдена", "entry not found"},
    Row{"Слот: живые аргументы ребёнка", "Slot: live child arguments"},
    Row{"модель не выбрана", "no model selected"},
    Row{" — слот в сон", " — slot to sleep"},
    Row{"Свежий b-релиз: перепинить манифест + установить активную", "Fresh b-release: re-pin the manifest + install the active"},
    Row{"Только перепинить манифест (бинарники не качаются)", "Re-pin the manifest only (no binaries downloaded)"},
    Row{"Свежий b-релиз на ВСЕ пиннутые сборки", "Fresh b-release to ALL pinned builds"},
    Row{"Конкретный релиз… (ввод тега bNNNNN)", "A specific release… (enter a bNNNNN tag)"},
    Row{"Ветка «", "Branch «"},
    Row{"»: переустановить из текущего пина", "»: reinstall from the current pin"},
    Row{"»: переключиться", "»: switch to"},
    Row{"»: пиннута, не установлена — установить", "»: pinned, not installed — install"},
    Row{"Сборка по железу (снять ручной выбор)", "Build by hardware (clear the manual choice)"},
    Row{"GitHub недоступен", "GitHub unreachable"},
    Row{"ответ не разобрался", "response unreadable"},
    Row{"Установка «", "Installing «"},
    Row{" — остальные команды недоступны", " — other commands unavailable"},
    Row{"Установлен: ", "Installed: "},
    Row{" — активная", " — active"},
    Row{" — установлена", " — installed"},
    Row{" — пиннута, не установлена", " — pinned, not installed"},
    Row{"  ·  актуальный: проверяю…", "  ·  latest: checking…"},
    Row{"Ветки: ", "Branches: "},
    Row{"  ·  актуальный: ", "  ·  latest: "},
    Row{"не узнать", "unknown"},
    Row{"      установленный релиз не определён", "      the installed release is not determined"},
    Row{"      доступно обновление — первый пункт меню", "      an update is available — the first menu item"},
    Row{"      актуален", "      up to date"},
    Row{"ставлю «", "installing «"},
    Row{"» — прогресс на экране движка", "» — progress on the engine screen"},
    Row{"пин манифеста: ", "manifest pin: "},
    Row{" — установка отдельно: F5 → Установить", " — install separately: F5 → Install"},
    Row{"выбор снят — дети перезапускаются на сборке по железу", "selection cleared — children restart on the by-hardware build"},
    Row{"выбрана «", "selected «"},
    Row{"» — дети перезапускаются", "» — children restarting"},
    Row{"ошибка установки", "install error"},
    Row{"✓ сборка «", "✓ build «"},
    Row{"» установлена", "» installed"},
    Row{" (в очереди ещё ", " (in queue: "},

    // раздел «Изображения» (художники)
    Row{" llmhost — изображения (художники) ", " llmhost — images (artists) "},
    Row{" Имя каталога или org/repo: ", " Catalog name or org/repo: "},
    Row{"Конвенция своего репо: recipes/*.json + diffusion_models/ + ",
        "Own-repo convention: recipes/*.json + diffusion_models/ + "},
    Row{"text_encoders/ + vae/ — по одному файлу на слот.",
        "text_encoders/ + vae/ — one file per slot."},
    Row{"движок: загружена «", "engine: loaded «"},
    Row{"», порт ", "», port "},
    Row{"движок: загрузка…", "engine: loading…"},
    Row{"движок: ошибка — ", "engine: error — "},
    Row{"движок: не загружен (автостарт при первом запросе)",
        "engine: not loaded (autostart on the first request)"},
    Row{" ГиБ", " GiB"},
    Row{"шаги ", "steps "},
    Row{" · в памяти", " · in memory"},
    Row{"(каталог пуст — скачайте: p)", "(catalog is empty — download with p)"},
    Row{"каталог image-моделей пуст — скачайте: p",
        "image-model catalog is empty — download with p"},
    Row{"останавливаю image-движок…", "stopping the image engine…"},
    Row{"каталог пуст", "catalog is empty"},
    Row{"скачивание «", "downloading «"},
    Row{"» началось — прогресс внизу", "» started — progress below"},
    Row{"  i                 художники (изображения): каталог, загрузка/удаление",
        "  i                 artists (images): catalog, load/delete"},
    Row{"ИЗОБРАЖЕНИЯ (художники)", "IMAGES (artists)"},
    Row{"  Enter/F5          загрузить выбранного (к нему идут запросы /v1/images)",
        "  Enter/F5          load the selected one (/v1/images requests go to it)"},
    Row{"  F8, u             остановить движок (выгрузить)",
        "  F8, u             stop the engine (unload)"},
    Row{"  p                 скачать: имя каталога или org/repo с раскладкой",
        "  p                 download: catalog name or org/repo with the layout"},
    Row{"                    recipes/ + diffusion_models/ + text_encoders/ + vae/",
        "                    recipes/ + diffusion_models/ + text_encoders/ + vae/"},
    Row{"  Ctrl-D            удалить с диска (загруженную демон выгрузит сам)",
        "  Ctrl-D            delete from disk (a loaded one is unloaded first)"},

    // --- демон: клиентские строки (control/proxy/sup_harness/catalog/daemond)
    Row{"нужен Bearer-токен", "a Bearer token is required"},
    Row{"внутренняя ошибка — подробности в журнале демона", "internal error — details are in the daemon journal"},
    Row{"тело запроса — не JSON", "the request body is not JSON"},
    Row{"модель «", "model «"},
    Row{"» не найдена в реестре", "» not found in the registry"},
    Row{"файл не найден: ", "file not found: "},
    Row{"путь вне папки моделей: ", "path outside the models dir: "},
    Row{"удалено ", "deleted "},
    Row{", останов на: ", ", stopped at: "},
    Row{"идёт установка движка — загрузка моделей недоступна до её конца",
        "an engine install is running — model loads are unavailable until it finishes"},
    Row{"файл не читается как GGUF: ", "the file does not read as GGUF: "},
    Row{"уже идёт загрузка модели «", "a load is already running for model «"},
    Row{"модель уже загружена", "the model is already loaded"},
    Row{"уже идёт загрузка", "a load is already running"},
    Row{"» не известна роутеру", "» is not known to the router"},
    Row{"» не известна", "» is not known"},
    Row{"в manifest.json нет пиннутой сборки «", "manifest.json has no pinned build «"},
    Row{"сборка «", "build «"},
    Row{"» не установлена — lhctl engine install ", "» is not installed — lhctl engine install "},
    Row{"в manifest.json нет сборки «", "manifest.json has no build «"},
    Row{"установка уже идёт", "an install is already running"},
    Row{"сначала выгрузите модель — установке нужен свободный llama-server",
        "unload the model first — the install needs a free llama-server"},
    Row{"дождитесь конца замера движков", "wait for the engine benchmark to finish"},
    Row{"тело запроса — не корректный JSON", "the request body is not valid JSON"},
    Row{"тело запроса — не JSON-объект", "the request body is not a JSON object"},
    Row{"элементы \"prompt\" — строки или целые токен-идентификаторы",
        "\"prompt\" elements are strings or integer token ids"},
    Row{"элементы \"input\" — строки или массивы токенов", "\"input\" elements are strings or arrays of tokens"},
    Row{"нужен Bearer-токен (lhctl tokens — как выдать)", "a Bearer token is required (lhctl tokens — how to issue one)"},
    Row{"модель не выбрана: укажите \"model\": <id> в запросе — список id: GET /v1/models",
        "no model chosen: specify \"model\": <id> in the request — the id list: GET /v1/models"},
    Row{"модель недоступна", "the model is unavailable"},
    Row{"на OpenAI-порту доступны только /v1/* (GET/POST) и /health",
        "only /v1/* (GET/POST) and /health are available on the OpenAI port"},
    Row{"репозиторий не найден, приватный или нужен токен HF_TOKEN (401)",
        "repository not found, private, or an HF_TOKEN is required (401)"},
    Row{"ответ HuggingFace не разобрался", "the HuggingFace response is unreadable"},
    Row{"дерево файлов не разобралось", "the file tree failed to parse"},
    Row{"в репозитории нет .gguf файлов", "the repository has no .gguf files"},
    Row{"пустой запрос", "empty query"},
    Row{"неожиданный ответ поиска", "unexpected search response"},
    Row{"» не загружена", "» is not loaded"},
    Row{"» занята — идёт генерация, выгрузка отклонена", "» is busy — generation is running, unload refused"},
    Row{"закройте резидентов (свободите VRAM)", "close the residents (free up VRAM)"},
    Row{"выходной слой на CPU", "output layer on CPU"},
    Row{"слоёв на GPU ", "GPU layers "},
    Row{" (хвост в RAM)", " (tail in RAM)"},
    Row{"веса целиком в RAM", "weights whole in RAM"},
    Row{"контекст ", "context "},
    Row{" вместо ", " instead of "},
    Row{"эксперты целиком в RAM", "experts whole in RAM"},
    Row{"экспертов в RAM ", "experts in RAM "},
    Row{"кэш MoE снят", "MoE cache dropped"},
    Row{"загружено урезанно: ", "loaded degraded: "},
    Row{" — не влезла в память", " — did not fit in memory"},
    Row{"модель в стоп-листе: прошлую загрузку пришлось убивать, а ребёнок не умер (клин драйвера) — перезапустите демон, чтобы повторить",
        "the model is on the stop list: the previous load had to be killed and the child did not die (a driver wedge) — restart the daemon to retry"},
    Row{"модели нужно ", "the model needs "},
    Row{" ГиБ видеопамяти", " GiB of GPU memory"},
    Row{" на её карте", " on its card"},
    Row{" (с учётом раскладки по картам)", " (accounting for the card layout)"},
    Row{"мешают занятые резиденты: ", "busy residents are in the way: "},
    Row{"файл модели не найден", "the model file was not found"},
    Row{"без draft", "no draft"},
    Row{"загрузку отменили", "the load was cancelled"},
    Row{"не удалось загрузить модель: ", "failed to load the model: "},
    Row{"истекло ожидание загрузки модели", "timed out waiting for the model to load"},
    Row{"память занята резидентами (", "memory is held by residents ("},
    Row{"очередь пробуждения модели «", "the wake queue for model «"},
    Row{"очередь пробуждения полна — повторите позже", "the wake queue is full — try again later"},
    Row{"очередь пробуждения: истекло ожидание освобождения памяти (",
        "wake queue: timed out waiting for memory to free ("},

    // image-маршруты control + pull kind=image
    Row{"каталог image-моделей не настроен: ",
        "the image-model catalog is not configured: "},
    Row{" — нужен llmhost-image.json", " — llmhost-image.json is required"},
    Row{"id модели — [A-Za-z0-9._-], а не «", "model id is [A-Za-z0-9._-], not «"},
    Row{"в каталоге image-моделей нет «", "no «"},
    Row{"» в каталоге image-моделей", "» in the image-model catalog"},
    Row{"kind — «llm» или «image», а не «", "kind — «llm» or «image», not «"},
    Row{"для kind=image «file» не нужен — раскладку задаёт рецепт",
        "«file» is not needed for kind=image — the layout is set by the recipe"},
    Row{"тело запроса не разбирается как JSON", "request body does not parse as JSON"},    Row{"» не поддерживает редактирование по изображению (в рецепте нет ref_images) — "
        "текст-в-картинку: /v1/images/generations",
        "» does not support image editing (the recipe has no ref_images) — "
        "text-to-image: /v1/images/generations"},
    // Дыры покрытия клиентского контура, закрытые аудитом tr() vs kTable
    // (на не-русской машине ранее оставались русскими: чипы keybar, диалоги
    // подтверждений, статусы image-слота, страницы lhctl image).
    Row{"движок", "engine"},
    Row{"изо", "img"},
    Row{"метрики", "metrics"},
    Row{"отозвать", "revoke"},
    Row{"снять", "stop"},
    Row{"ошибка", "error"},
    Row{"найден", "found"},
    Row{"загрузка…", "loading…"},
    Row{"генерация остановлена", "generation stopped"},
    Row{"× не влезает, ", "× won't fit, "},
    Row{"«", "\""},
    Row{"»", "\""},
    Row{"»  ·  ", "\"  ·  "},
    Row{"                    /clear, /bye)", "                    /clear, /bye)"},
    Row{"  (в памяти)", "  (in memory)"},
    Row{"  (грузится…)", "  (loading…)"},
    Row{"  (не загрузилась)", "  (not loaded)"},
    Row{"  ·  изо: ", "  ·  img: "},
    Row{"  ·  изо: загрузка…", "  ·  img: loading…"},
    Row{"  ·  изо: ошибка", "  ·  img: error"},
    Row{"  ·  обновление каждые N с (Ctrl-C — выход)",
        "  ·  refresh every N s (Ctrl-C — exit)"},
    Row{"  ·  резидентно до ", "  ·  resident until "},
    Row{"  ↑ ещё ", "  ↑ more "},
    Row{"  файлы слотов, рецепт (если свободен) и запись привязки",
        "  slot files, the recipe (if free) and the binding record"},
    Row{" (порт ", " (port "},
    Row{" порт ", " port "},
    Row{"порт ", "port "},
    Row{" — для всех клиентов.", " — for all clients."},
    Row{"), очередь пробуждения выключена", "), the wake-up queue is off"},
    Row{"? Загруженную модель демон сначала остановит.",
        "? A loaded model is stopped by the daemon first."},
    Row{"ctx: разумный диапазон 256…2097152 токенов\n",
        "ctx: sane range 256…2097152 tokens\n"},
    Row{"harness: по llama-server на каждую модель",
        "harness: one llama-server per model"},
    Row{"image engine unavailable: ", "image engine unavailable: "},
    Row{"image-движок остановлен", "image engine stopped"},
    Row{"image: %s\n", "image: %s\n"},
    Row{"image: load не удался\n", "image: load failed\n"},
    Row{"image: битый ответ демона\n", "image: daemon response unreadable\n"},
    Row{"lhctl: неверный токен (401)", "lhctl: wrong token (401)"},
    Row{"llmhost недоступен\n", "llmhost unreachable\n"},
    Row{"mtp-mode = %s → llmhost.conf [%s]\n", "mtp-mode = %s → llmhost.conf [%s]\n"},
    Row{"thinking снят: режим решает движок по шаблону",
        "thinking cleared: the engine decides by the template"},
    Row{"· в памяти", "· in memory"},
    Row{"» занята генерацией — снять принудительно?",
        "\" is busy generating — force stop?"},
    Row{"» с диска", "\" from disk"},
    Row{"» снята принудительно — генерации модели обрываются",
        "\" force-stopped — running generations are cut"},
    Row{"» — список: lhctl image", "\" — list: lhctl image"},
    Row{"ВСЕ генерации этой модели обрываются",
        "ALL generations of this model are cut"},
    Row{"Конвенция своего репо: recipes/*.json + diffusion_models/ +",
        "Own-repo convention: recipes/*.json + diffusion_models/ +"},
    Row{"НЕ найден (image-engine или engine/sd/)",
        "NOT found (image-engine or engine/sd/)"},
    Row{"Удалить художника «", "Delete the artist \""},
    Row{"движок:  sd-server %s\n", "engine:  sd-server %s\n"},
    Row{"изо sd-server: ", "img sd-server: "},
    Row{"карта:   image-gpu=%d, простой ≤ %d мин\n",
        "card:   image-gpu=%d, idle ≤ %d min\n"},
    Row{"модель:  %s — %s, %s ГиБ, шаги %d, CFG %.2f %s\n",
        "model:  %s — %s, %s GiB, steps %d, CFG %.2f %s\n"},
    Row{"простой (автостарт: карта ", "idle (autostart: GPU "},
    Row{"снимаю принудительно «", "force-stopping \""},
    Row{"состояние: загружена «%s» (порт %d)\n", "state: loaded \"%s\" (port %d)\n"},
    Row{"состояние: идёт загрузка…\n", "state: loading in progress…\n"},
    Row{"состояние: не загружена (автостарт при первом запросе)\n",
        "state: not loaded (autostart on first request)\n"},
    Row{"состояние: ошибка — %s\n", "state: error — %s\n"},
    Row{"укажите \"path\" или \"name\"", "specify \"path\" or \"name\""},
    Row{"укажите \"repo\": \"org/repo\"", "specify \"repo\": \"org/repo\""},
    Row{"укажите \"repo\": org/repo", "specify \"repo\": org/repo"},
    Row{"установка «", "installing \""},
};

// Решённый язык (-1 — «ещё не решён», решается лениво при первом tr()) и
// хинт из [general] lang (-1 — нет, 0 — auto, 1 — ru, 2 — en; валидность
// гарантирует config::check_setting, сюда доезжает только верное).
inline std::atomic<int> g_resolved{-1};
inline std::atomic<int> g_hint{-1};

// Локаль машины: первый непустой из LC_ALL → LC_MESSAGES → LANG; «ru»-префикс
// → русский, остальное (включая отсутствие) → английский.
inline int detect_locale() {
  for (const char* v : {"LC_ALL", "LC_MESSAGES", "LANG"}) {
    const char* s = ::getenv(v);
    if (s && s[0] != '\0') return (s[0] == 'r' && s[1] == 'u') ? kRu : kEn;
  }
  return kEn;
}

// LH_LANG: ручной форс (ru|en, регистр и суффикс «ru_RU» не важны);
// unset/пусто/мусор — -1 (нет форса).
inline int env_lang() {
  const char* s = ::getenv("LH_LANG");
  if (!s || s[0] == '\0') return -1;
  auto eq = [&](const char* two) {
    return (s[0] == two[0] || s[0] == two[0] - 32) &&
           (s[1] == two[1] || s[1] == two[1] - 32) &&
           (s[2] == '\0' || s[2] == '_');
};
  if (eq("ru")) return kRu;
  if (eq("en")) return kEn;
  return -1;
}

// Хинт из конфига демона: «auto»/пусто — детект локали, «ru»/«en» — форс.
// Сбрасывает решённое: вотчер применяет hot-правку lang живьём.
inline void set_lang_hint(const char* cfg) {
  if (cfg && cfg[0] != '\0' && std::strcmp(cfg, "auto") != 0)
    g_hint.store(std::strcmp(cfg, "ru") == 0 ? 1 : 2, std::memory_order_relaxed);
  else
    g_hint.store(0, std::memory_order_relaxed);
  g_resolved.store(-1, std::memory_order_relaxed);
}

// Явный сеттер — тесты.
inline void set_lang(int lang) {
  g_hint.store(lang == kRu ? 1 : 2, std::memory_order_relaxed);
  g_resolved.store(lang, std::memory_order_relaxed);
}

inline int lang() {
  int e = env_lang();  // форс окружением — сильнее конфига
  if (e >= 0) return e;
  int h = g_hint.load(std::memory_order_relaxed);
  if (h == 1) return kRu;
  if (h == 2) return kEn;
  int r = g_resolved.load(std::memory_order_relaxed);
  if (r >= 0) return r;
  r = detect_locale();
  g_resolved.store(r, std::memory_order_relaxed);
  return r;
}

// Перевод: русский литерал — ключ. En: поиск по таблице, промах — ключ как
// есть. Ru: ключ и есть русский — ноль поиска.
inline const char* tr(const char* ru) {
  if (lang() != kEn) return ru;
  for (const auto& row : kTable)
    if (std::strcmp(row.ru, ru) == 0) return row.en;
  return ru;
}

} // namespace lh::i18n
