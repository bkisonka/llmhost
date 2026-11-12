// Мелкая обвязка: журнал, пути, случайности, свободный порт, маленькие файлы.
// Linux-версия: всё в UTF-8 std::string, разделитель путей '/'.
#pragma once
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace lh {

#ifdef _WIN32
// Конвертация путей/строк: WinAPI живёт в UTF-16, демон и конфиги — в UTF-8.
std::string wide_to_utf8(const std::wstring& w);
std::wstring utf8_to_wide(const std::string& utf8);
#endif

// Полный путь к бинарнику (readlink /proc/self/exe) / его папка (без слэша).
std::string exe_path();
std::string exe_dir();

// Изменяемые данные: $LH_DATA_DIR, иначе $XDG_DATA_HOME/llmhost, иначе
// ~/.local/share/llmhost — конфиг, токен, журнал, движок, модели.
std::string app_dir();
std::string ensure_dir(const std::string& path);

// Журнал: <app_dir>/log.txt (хвост до 1 МБ). LH_LOG_STDERR=1 — эхо в stderr.
// format(printf) — типизированная защита: GCC проверяет каждый вызов против
// строки формата (std::string в varargs и потерянные/лишние аргументы
// становятся ошибками сборки с -Werror=format; поймано на заморозке beta 12
// — два живых UB см. git log). Динамические строки формата не запрещены:
// -Wformat-nonliteral не входит в plain -Wformat.
void logf(const char* fmt, ...)
#ifdef __GNUC__
    __attribute__((format(printf, 1, 2)))
#endif
    ;
std::string log_path();

// Отпечаток фатального сигнала (SIGFPE/SIGILL/SIGBUS/SIGSEGV): одна строка в
// log.txt и stderr асинхронно-безопасными вызовами (open/write), затем выход
// дефолтным обработчиком — core и строка trap в dmesg сохраняются. Смысл:
// «приложение молча умерло» в поле должно само рассказывать, чем именно
// (бывал и SIGFPE между «control: слушаю» и следующей строкой — журнал
// обрывался без единого следа). POSIX; на Windows — no-op (win-репо
// делает SetUnhandledExceptionFilter). Вызывается один раз в начале main().
void install_crash_telemetry();

// Криптостойкие случайные байты → hex (getrandom(2)) — для api-ключей и токена.
// Пусто — система не дала (fail-closed: без ключа API не открывается).
std::string random_hex(size_t n_bytes);

// Base64 (стандартный алфавит с паддингом) — для картинок в data-URI чата.
std::string base64_encode(const std::string& raw);

// Свободный TCP-порт в диапазоне [lo; hi], 0 — не нашли.
int free_tcp_port(int lo, int hi);

std::string read_small_file(const std::string& path);       // пусто — не удалось
bool write_small_file(const std::string& path, const std::string& data);
// Как write_small_file, но файл создаётся сразу с правами 0600 (open, не
// ofstream+chmod): для секретов (api_tokens.json) — вариант «записать, потом
// урезать» оставлял окно чтения по umask и следовал симлинкам.
bool write_private_file(const std::string& path, const std::string& data);
bool file_exists(const std::string& path);
uint64_t file_size(const std::string& path);                // 0 — не удалось

// Поиск llama-server: явный путь из конфига (с $переменными) → рядом с exe → PATH.
std::string find_llama_server(const std::string& configured);

// Окружение для ребёнка-движка: LD_LIBRARY_PATH с каталогом бинарника поверх
// унаследованного. Сборки движка не обязаны иметь rpath $ORIGIN (локальные
// пины без него: llama-server не находит свой libllama-server-impl.so рядом),
// а от окружения запустившего демона (systemd без переменных против ручного
// старта из оболочки) работа детей зависеть не должна.
std::vector<std::pair<std::string, std::string>> engine_child_env(const std::string& exe);

// Wildcards * и ?, регистр не важен: «*Q4_K_M*» матчит «Llama-Q4_K_M.gguf».
bool wildcard_match(const char* pat, const char* str);

// «Медленный» носитель под файлом: сетевая ФС (NFS/SMB/9p/fuse). На ней холодные
// страницы mmap-экспертов читаются рывками — умная настройка прогревает кэш.
bool slow_media(const std::string& path);

// «org/repo» ← «org/repo», URL вида huggingface.co/org/repo/resolve/main/… и т.п.
// false — err с объяснением. Компоненты — только [A-Za-z0-9._-]: путь из сети
// клеится к папке моделей, «..» и прочее туда попадать не должны.
// Экспортировано для теста.
bool normalize_hf_repo(std::string& repo, std::string& err);

// $переменные и ${переменные} → развёрнутый путь; нет в окружении — как есть.
std::string expand_env(const std::string& s);

// Безопасен ли ОТНОСИТЕЛЬНЫЙ путь, пришедший из сети (HF API), для склейки
// с базовой папкой: без «..», «:», кавычек, control-символов, абсолютных
// начал («/»). true — можно клеить.
bool safe_rel_path(const std::string& utf8_path);

// Безопасно ли ИМЯ/ПУТЬ ФАЙЛА С ДИСКА для записи в INI-пресет и журналы:
// файловые системы допускают control-символы в именах (кроме NUL и «/»),
// а «\n[секция]» в имени .gguf — готовая инъекция в router_preset.ini.
// Запрещены только control-символы; кириллица, пробелы и прочая печать — законны.
bool safe_local_filename(const std::string& path);

// Сузить права файла до текущего пользователя (chmod 0600): token.txt не
// должен читаться другими пользователями машины.
bool restrict_file_permissions(const std::string& path);

// Путь для URL: [A-Za-z0-9._~/-] остаются, остальное (включая пробелы и
// не-ASCII байты UTF-8) — %XX.
std::string percent_encode_path(const std::string& s);

// Заменить ВСЕ вхождения from на to. from пуст — строка без изменений.
// Экспортировано для теста (engine::pin: тег релиза в имени ассета).
std::string replace_all(std::string s, const std::string& from, const std::string& to);

// «version: 0.6.0-dev (build 11382, commit 11fe02151)» → «b11382». Не
// разобралось (нет «build NNNNN») — пусто. Экспортировано для теста.
std::string release_from_version_output(const std::string& out);

// Строка лога ребёнка со статистикой черновика → приём и счётчики.
// Формат плавает между пинами llama.cpp, пробуем варианты:
//   «draft acceptance = 0.98 ( 374 accepted / 387 generated), mean len = 3.9»
//   «draft accept rate = 0.98 ( 374/ 387)» · «spec_draft: accept=0.98 n=374/387»
// Распознано — true, rate ≥ 0; accepted/generated = 0, если в строке только
// rate (окно по токенам такие строки не учитывает). Экспортировано для теста.
bool parse_draft_acceptance(const std::string& line, double& rate,
                            long long& accepted, long long& generated);

// ---- единый обход INI (llmhost.conf, пер-модельные секции, пресеты) ----------
// Раньше грамматика «[секция] / ключ = значение / комментарий строкой» была
// переписана заново в пяти местах и расходилась в мелочах. Все потребители
// ходят через for_each_ini_line; сырые строки (save_to с сохранением
// комментариев) не проходят через него сознательно.
struct IniLine {
  bool section = false;    // заголовок секции
  bool closed = true;      // «]» найден («[x» — нет: потребители сами решают)
  std::string name;        // имя секции КАК НАПИСАНО (без trim'а — trim у потребителя)
  bool has_eq = false;     // в строке был «=»
  std::string key, value;  // trim'нутые; без «=» — пустые
  int lineno = 0;          // 1-based, считая все строки файла
};

// Построчный обход: пустые строки и комментарии (#/; строкой целиком)
// не отдаются. Инлайн-комментарии не поддерживаются сознательно — значение
// «system» может их содержать.
inline void for_each_ini_line(const std::string& raw, auto&& f) {
  size_t pos = 0;
  int lineno = 0;
  while (pos <= raw.size()) {
    size_t eol = raw.find('\n', pos);
    std::string line = raw.substr(pos, eol == std::string::npos
                                            ? std::string::npos
                                            : eol - pos);
    pos = eol == std::string::npos ? raw.size() + 1 : eol + 1;
    ++lineno;
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ' ||
                             line.back() == '\t'))
      line.pop_back();
    size_t n0 = line.find_first_not_of(" \t");
    if (n0 == std::string::npos) continue;
    line = line.substr(n0);
    if (line[0] == '#' || line[0] == ';') continue;
    IniLine l;
    l.lineno = lineno;
    if (line[0] == '[') {
      l.section = true;
      size_t close = line.find(']');
      l.closed = close != std::string::npos;
      l.name = close == std::string::npos ? line.substr(1)
                                          : line.substr(1, close - 1);
      f(l);
      continue;
    }
    size_t eq = line.find('=');
    if (eq == std::string::npos) { f(l); continue; }
    l.has_eq = true;
    l.key = line.substr(0, eq);
    while (!l.key.empty() && (l.key.back() == ' ' || l.key.back() == '\t'))
      l.key.pop_back();
    l.value = line.substr(eq + 1);
    size_t v0 = l.value.find_first_not_of(" \t");
    l.value = v0 == std::string::npos ? "" : l.value.substr(v0);
    f(l);
  }
}

} // namespace lh
