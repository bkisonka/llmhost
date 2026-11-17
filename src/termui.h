// termui — панели, таблицы, бары и прогресс для lhctl.
//
// Принципы:
//   * рисуем только в интерактивном терминале (pretty()); пайпы и скрипты
//     получают прежний машинный вывод — это рендер-надстройка, не контракт;
//   * ширина терминала перечитывается на КАЖДОЙ отрисовке (width()) —
//     изменение размера окна не ломает ни таблицу, ни строку прогресса;
//   * подсчёт ширины понимает UTF-8 (кириллица, рамочные символы = 1 ячейка)
//     и пропускает ANSI-последовательности — покрашенные строки можно
//     спокойно передавать в pad/trunc/таблицы;
//   * никакого ncurses: голый ANSI + box-drawing, работает и через ssh.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <termios.h>
#endif

namespace lh::ui {

// Раз в main(): определяет tty/цвет/ширина. force_json — «--json/--raw»:
// панели отключаются, вывод остаётся машинным даже в tty.
void init(bool force_json = false);

bool pretty();   // можно рисовать панели (tty и не --json)
bool color();    // разрешены ANSI-цвета (tty, без NO_COLOR, TERM != dumb)

// Текущая ширина терминала в ячейках; перечитывается при каждом вызове
// (ioctl TIOCGWINSZ / GetConsoleScreenBufferInfo → $COLUMNS → 100).
int width();
// Высота терминала в строках (fallback 24).
int height();
// «─» × n (рамочные литералы многобайтовые — char-литерал не годится).
std::string horiz(size_t n);

// Ширина строки в ячейках терминала (UTF-8, ANSI-последовательности = 0).
size_t dwidth(const std::string& s);
// Обрезка до w ячеек (хвост — «…»); «…» тоже занимает ячейку.
std::string trunc(const std::string& s, size_t w);
// Выравнивание: добить пробелами до w (учитывая ANSI и UTF-8).
std::string pad(const std::string& s, size_t w);   // влево
std::string padl(const std::string& s, size_t w);  // вправо
// Снять ANSI-краски (инверсия поверх покрашенной строки обрывается первым
// же внутренним ESC[0m — курсорные строки красятся только после plain()).
std::string plain(const std::string& s);

// Цвета: без color() возвращают строку как есть. Кодируются ДО передачи в
// таблицы/панели — те корректно посчитают ширину.
namespace c {
std::string dim(const std::string& s);     // серый — подписи, второстепенное
std::string accent(const std::string& s);  // сине-зелёный — акценты, заголовки
std::string ok(const std::string& s);      // зелёный
std::string warn(const std::string& s);    // жёлтый
std::string err(const std::string& s);     // красный, жирный
std::string bold(const std::string& s);
// Инверсия (видеообратный) — курсорные строки, верхняя/нижняя панели.
std::string hl(const std::string& s);
// Цвет по доле заполнения: <60% зелёный, <85% жёлтый, дальше красный.
std::string by_frac(double frac, const std::string& s);
}  // namespace c

// ---- Полноэкранный режим (интерфейс управления, в духе mc) ------------------

// Клавиши read_key(); Char — печатный/UTF-8 байт в chr.
enum class Key {
  None, Up, Down, Left, Right, Home, End, PgUp, PgDn, Enter, Esc, Backspace,
  Del, F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, Tab, CtrlC, CtrlD, CtrlU, CtrlX, Char
};

// Читает клавишу; ждёт до timeout_ms, None — ничего не нажато. Разбирает
// escape-последовательности (стрелки, F1–F10 в обоих кодированиях, PgUp/PgDn,
// Home/End, Del); Ctrl-C/Ctrl-D отдаёт как клавиши, не убивая процесс.
Key read_key(int timeout_ms, char& chr);

// Командная клавиша как UTF-8 код ("m", "с", "д"…): латиница — один байт,
// кириллица ЙЦУКЕН (RU/UA/BG — общие коды букв) — два; prev — предыдущий
// байт Key::Char (окно ведёт вызывающий цикл). Фонетические раскладки и
// многослайсовые коды (CJK, эмодзи) не покрываются сознательно: командных
// клавиш среди них нет; паттерн расширяется длиной последовательности.
// Однобайтовые команды (латиница) от prev не зависят: байты-продолжения
// UTF-8 всегда >= 0x80 и с ASCII-командой не сталкиваются.
inline bool key_seq(unsigned char prev, char chr, const char* utf8) {
  const auto u = (const unsigned char*)utf8;
  if (u[1] == 0) return (unsigned char)chr == u[0];
  return u[2] == 0 && prev == u[0] && (unsigned char)chr == u[1];
}

// RAII-сессия полного экрана: cbreak-ввод без эха (ISIG тоже выключен —
// Ctrl-C приходит клавишей), альтернативный экран, скрытый курсор.
// Деструктор восстанавливает терминал даже при исключении.
class RawTerminal {
public:
  RawTerminal();
  ~RawTerminal();
  RawTerminal(const RawTerminal&) = delete;
  RawTerminal& operator=(const RawTerminal&) = delete;

private:
#ifdef _WIN32
  DWORD in_mode_ = 0, out_mode_ = 0;
#else
  termios saved_{};
  int orig_rows_ = 0, orig_cols_ = 0; // исходный размер окна (если меняли)
#endif
  bool active_ = false;
};

// Панель со скруглённой рамкой и заголовком в верхней кромке:
//   ╭─ llmhost 0.6.0 ─────────────╮
//   │  Состояние   работает       │
//   ╰─────────────────────────────╯
class Panel {
public:
  explicit Panel(std::string title) : title_(std::move(title)) {}
  Panel& line(const std::string& s);             // готовая строка содержимого
  Panel& kv(const std::string& k, const std::string& v);  // «ключ  значение»
  Panel& hr();                                   // тонкий разделитель внутри
  std::string render(int w) const;               // рамка по ширине w
  std::string render() const;                    // по текущей ширине терминала

private:
  std::string title_;
  std::vector<std::string> lines_;
  std::vector<bool> is_hr_;
};

// Таблица: колонки с авторазмером, сжимаются самые широкие, содержимое
// обрезается с «…» — таблица всегда точно вписывается в ширину терминала.
class Table {
public:
  enum Align { Left, Right };
  // max_w: потолок ширины колонки в ячейках (0 — не ограничивать).
  Table& col(const std::string& title, Align a = Left, size_t max_w = 0);
  Table& row(std::vector<std::string> cells);    // значения можно красить
  size_t rows() const { return rows_.size(); }
  std::string render(int w) const;
  std::string render() const;

private:
  struct Col {
    std::string title;
    Align align = Left;
    size_t max_w = 0;
    size_t w = 0;
  };
  std::vector<Col> cols_;
  std::vector<std::vector<std::string>> rows_;
  size_t natural_width(size_t i) const;
};

// Бар заполнения «████░░░░»: frac 0..1, ширина w ячеек. Цвет — снаружи
// (c::by_frac); сам бар нейтральный, чтобы красить целиком.
std::string bar(double frac, int w);

// Кадр спиннера (внутренний счётчик, брайлевские точки).
const char* spinner();

// Перезарисовываемая строка прогресса: \r + затирание до конца строки +
// line. Вызывать раз в тик — ширина и затирание подстраиваются на каждом
// кадре, так что resize окна посреди скачивания не оставляет мусора.
void progress_update(const std::string& line);
// Завершение: затереть прогресс, напечатать final_line и перевод строки.
void progress_end(const std::string& final_line);

// Стирание экрана с курсором в угол (режим watch).
void clear_screen();

// Полный экран: альтернативный буфер, очистка, скрыть/показать курсор.
void enter_alt();
void leave_alt();
void hide_cursor();
void show_cursor();

// Разбить текст на строки терминала. Перевод строки — новая строка,
// прочие управляющие выкидываются, длиннее w ячеек режется. В куске нет
// \\n: иначе полноэкранный кадр прокручивает альтернативный экран и
// следующая строка ложится поверх предыдущей.
std::vector<std::string> hard_wrap(const std::string& s, size_t w);

// Нарисовать экран: строки дополняются до ширины, хвост очищается —
// без мерцания и без очистки всего экрана на каждом кадре. Строк не больше
// высоты окна: лишняя прокрутила бы альтернативный экран без истории.
void draw_lines(const std::vector<std::string>& lines);

}  // namespace lh::ui
