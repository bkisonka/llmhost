#include "termui.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>
#endif

namespace lh::ui {

namespace {

bool g_tty = false;     // stdout — терминал с ANSI
bool g_force_json = false;
bool g_color = false;

// ANSI-последовательности — нулевой ширины: ESC … букву пропускаем.
// Возвращает true, если байт i начал последовательность, i сдвигается за неё.
bool skip_ansi(const std::string& s, size_t& i) {
  if (s[i] != '\x1b') return false;
  size_t j = i + 1;
  if (j < s.size() && s[j] == '[') {
    for (++j; j < s.size(); ++j) {
      unsigned char c = (unsigned char)s[j];
      if (c >= 0x40 && c <= 0x7e) { ++j; break; }  // терминальный байт CSI
    }
  } else {
    ++j;  // прочие ESC-последовательности — один код
  }
  i = j;
  return true;
}

int terminal_width_now() {
#ifdef _WIN32
  CONSOLE_SCREEN_BUFFER_INFO info;
  HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
  if (h != INVALID_HANDLE_VALUE && GetConsoleScreenBufferInfo(h, &info))
    return info.srWindow.Right - info.srWindow.Left + 1;
#else
  winsize ws;
  if (::ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0)
    return ws.ws_col;
#endif
  if (const char* cols = ::getenv("COLUMNS")) {
    int n = ::atoi(cols);
    if (n > 0) return n;
  }
  return 100;
}

int spin_pos = 0;
const char* const kSpinFrames[] = {"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"};

}  // namespace

void init(bool force_json) {
  g_force_json = force_json;
#ifdef _WIN32
  HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
  DWORD mode = 0;
  g_tty = GetConsoleMode(h, &mode) != 0;
  if (g_tty) {
#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif
    g_tty = SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0;
  }
#else
  g_tty = ::isatty(STDOUT_FILENO) != 0;
#endif
  const char* no_color = ::getenv("NO_COLOR");
  const char* term = ::getenv("TERM");
  g_color = g_tty && !g_force_json && (!no_color || !*no_color) &&
            (!term || std::strcmp(term, "dumb") != 0);
}

bool pretty() { return g_tty && !g_force_json; }
bool color() { return g_color; }

int width() {
  int w = g_tty ? terminal_width_now() : 100;
  return w < 40 ? 40 : (w > 250 ? 250 : w);
}

size_t dwidth(const std::string& s) {
  size_t w = 0;
  for (size_t i = 0; i < s.size();) {
    if (skip_ansi(s, i)) continue;
    // Управляющие байты (переводы строк внутри ячеек не используем) — ноль.
    unsigned char b = (unsigned char)s[i];
    if (b < 0x20) { ++i; continue; }
    if ((b & 0xC0) == 0x80) { ++i; continue; }  // байты продолжения UTF-8
    ++w;                                        // первый байт символа
    // Один кодовый пункт = один терминальный символ: кириллица и рамочные
    // символы в западных терминалах одинарные, эмодзи не используем.
    ++i;
    while (i < s.size() && ((unsigned char)s[i] & 0xC0) == 0x80) ++i;
  }
  return w;
}

std::vector<std::string> hard_wrap(const std::string& s, size_t w) {
  if (w < 1) w = 1;
  std::vector<std::string> out;
  std::string line;
  size_t cells = 0;
  auto push = [&]() {
    out.push_back(line);
    line.clear();
    cells = 0;
  };
  for (size_t i = 0; i < s.size();) {
    unsigned char b = (unsigned char)s[i];
    if (b == '\n' || b == '\r') {
      if (b == '\r' && i + 1 < s.size() && s[i + 1] == '\n') ++i;
      push();
      ++i;
      continue;
    }
    if (b == '\t') {
      if (cells >= w) push();
      line.push_back(' ');
      ++cells;
      ++i;
      continue;
    }
    if (b < 0x20 || b == 0x7f) { ++i; continue; }
    size_t start = i;
    ++i;
    while (i < s.size() && ((unsigned char)s[i] & 0xC0) == 0x80) ++i;
    if (cells >= w) push();
    line.append(s, start, i - start);
    ++cells;
  }
  if (!line.empty() || out.empty()) out.push_back(std::move(line));
  return out;
}

std::string trunc(const std::string& s, size_t w) {
  if (dwidth(s) <= w) return s;
  if (w == 0) return "";
  size_t used = 0, cut = 0;
  for (size_t i = 0; i < s.size();) {
    size_t start = i;
    if (skip_ansi(s, i)) { cut = i; continue; }  // ANSI не тратит ширину
    unsigned char b = (unsigned char)s[i];
    size_t len = 1;
    if (b >= 0x20 && (b & 0xC0) != 0x80) {
      ++i;
      while (i < s.size() && ((unsigned char)s[i] & 0xC0) == 0x80) ++i;
      len = i - start;
    } else {
      ++i;
      continue;
    }
    if (used + 1 > w - 1) { cut = start; break; }  // место для «…»
    used += 1;
    cut = i;
  }
  std::string out = s.substr(0, cut) + "…";
  // Обрезка могла отрезать закрывающий \x1b[0m — вернуть, иначе цвет
  // «протечёт» на остаток строки таблицы.
  if (out != s && s.find("\x1b[") != std::string::npos) out += "\x1b[0m";
  return out;
}

std::string pad(const std::string& s, size_t w) {
  size_t dw = dwidth(s);
  return dw >= w ? trunc(s, w) : s + std::string(w - dw, ' ');
}

std::string padl(const std::string& s, size_t w) {
  size_t dw = dwidth(s);
  return dw >= w ? trunc(s, w) : std::string(w - dw, ' ') + s;
}

std::string plain(const std::string& s) {
  if (s.find("\x1b") == std::string::npos) return s;
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size();) {
    if (s[i] == '\x1b') {
      if (i + 1 < s.size() && s[i + 1] == '[') {
        // CSI: параметры/промежуточные до финального байта 0x40–0x7E.
        i += 2;
        while (i < s.size() && !(s[i] >= '@' && s[i] <= '~')) ++i;
        if (i < s.size()) ++i;
      } else {
        ++i;  // не-CSI ESC наружу не едет — терминал иначе истолкует хвост
      }
      continue;
    }
    out += s[i++];
  }
  return out;
}

namespace c {
namespace {
std::string wrap(const char* code, const std::string& s) {
  return color() ? std::string(code) + s + "\x1b[0m" : s;
}
}  // namespace

std::string dim(const std::string& s) { return wrap("\x1b[38;5;245m", s); }
std::string accent(const std::string& s) { return wrap("\x1b[38;5;39m", s); }
std::string ok(const std::string& s) { return wrap("\x1b[38;5;114m", s); }
std::string warn(const std::string& s) { return wrap("\x1b[38;5;179m", s); }
std::string err(const std::string& s) { return wrap("\x1b[38;5;203m", s); }
std::string bold(const std::string& s) { return wrap("\x1b[1m", s); }
std::string hl(const std::string& s) { return wrap("\x1b[7m", s); }

std::string by_frac(double frac, const std::string& s) {
  if (frac >= 0.85) return err(s);
  if (frac >= 0.60) return warn(s);
  return ok(s);
}
}  // namespace c

Panel& Panel::line(const std::string& s) {
  lines_.push_back(s);
  is_hr_.push_back(false);
  return *this;
}

Panel& Panel::kv(const std::string& k, const std::string& v) {
  return line("  " + c::dim(pad(k + ":", 12)) + " " + v);
}

Panel& Panel::hr() {
  lines_.push_back("");
  is_hr_.push_back(true);
  return *this;
}

std::string Panel::render(int w) const {
  const size_t inner = (size_t)w - 4;  // «│ » + « │»
  std::string out;
  // Верхняя кромка: заголовок вшит в линию.
  size_t tdw = dwidth(title_);
  std::string top = "╭─ " + c::accent(title_) + " ";
  size_t fill = w - 3 - tdw - 2;
  if (title_.empty()) { top = "╭"; fill = w - 2; }
  out += top + horiz(fill > w ? 0 : fill) + "╮\n";
  for (size_t i = 0; i < lines_.size(); ++i) {
    if (is_hr_[i]) {
      out += "├" + horiz((size_t)w - 2) + "┤\n";
    } else {
      out += "│ " + pad(trunc(lines_[i], inner), inner) + " │\n";
    }
  }
  out += "╰" + horiz((size_t)w - 2) + "╯\n";
  return out;
}

std::string Panel::render() const { return render(width()); }

size_t Table::natural_width(size_t i) const {
  size_t w = dwidth(cols_[i].title);
  for (auto& r : rows_)
    if (i < r.size()) w = std::max(w, dwidth(r[i]));
  if (cols_[i].max_w) w = std::min(w, cols_[i].max_w);
  return w;
}

Table& Table::col(const std::string& title, Align a, size_t max_w) {
  cols_.push_back({title, a, max_w, 0});
  return *this;
}

Table& Table::row(std::vector<std::string> cells) {
  rows_.push_back(std::move(cells));
  return *this;
}

std::string Table::render(int w) const {
  if (cols_.empty()) return "";
  const size_t overhead = 3 * cols_.size() + 1;  // «│ x » на колонку + финальный «│»
  // Авторазмер: от натуральной ширины; если не влезает — ступенчато сжимаем
  // самую широкую колонку, пока таблица не войдёт в w (пол — 6 ячеек).
  std::vector<size_t> cw(cols_.size());
  for (size_t i = 0; i < cols_.size(); ++i) cw[i] = natural_width(i);
  for (;;) {
    size_t total = overhead;
    for (size_t x : cw) total += x;
    if (total <= (size_t)w) break;
    size_t biggest = cols_.size(), bw = 0;
    for (size_t i = 0; i < cw.size(); ++i)
      if (cw[i] > bw && cw[i] > 6) { biggest = i; bw = cw[i]; }
    if (bw == 0) break;  // дальше некуда — пусть будет шире терминала
    size_t shrink = std::min<size_t>(bw - 6, (total - (size_t)w + 1) / 2);
    if (shrink == 0) shrink = 1;
    cw[biggest] = bw - shrink;
  }
  auto cell = [&](size_t r, size_t i) {
    const std::string& s = r < rows_.size() && i < rows_[r].size() ? rows_[r][i] : std::string();
    return cols_[i].align == Right ? padl(trunc(s, cw[i]), cw[i]) : pad(trunc(s, cw[i]), cw[i]);
  };
  std::string out;
  // Шапка: «│ АББРЕВИАТУРА │ …» + линия под ней.
  out += "│";
  for (size_t i = 0; i < cols_.size(); ++i)
    out += " " + c::bold(c::dim(pad(trunc(cols_[i].title, cw[i]), cw[i]))) + " │";
  out += "\n├";
  for (size_t i = 0; i < cols_.size(); ++i)
    out += horiz(cw[i] + 2) + (i + 1 == cols_.size() ? "┤" : "┼");
  out += "\n";
  for (size_t r = 0; r < rows_.size(); ++r) {
    out += "│";
    for (size_t i = 0; i < cols_.size(); ++i) out += " " + cell(r, i) + " │";
    out += "\n";
  }
  return out;
}

std::string Table::render() const { return render(width()); }

std::string bar(double frac, int w) {
  if (w <= 0) return "";
  if (frac < 0) frac = 0;
  if (frac > 1) frac = 1;
  int filled = (int)(frac * w + 0.5);
  std::string out;
  out.reserve((size_t)w * 3);
  for (int i = 0; i < filled; ++i) out += "█";
  for (int i = 0; i < w - filled; ++i) out += "░";
  return out;
}

const char* spinner() {
  spin_pos = (spin_pos + 1) % (int)(sizeof(kSpinFrames) / sizeof(kSpinFrames[0]));
  return kSpinFrames[spin_pos];
}

void progress_update(const std::string& line) {
  std::string clipped = trunc(line, width() - 1);
  std::fprintf(stdout, "\r\x1b[2K%s", clipped.c_str());
  std::fflush(stdout);
}

void progress_end(const std::string& final_line) {
  std::fprintf(stdout, "\r\x1b[2K%s\n", final_line.c_str());
  std::fflush(stdout);
}

void clear_screen() { std::fputs("\x1b[H\x1b[2J", stdout); }

std::string horiz(size_t n) {
  std::string out;
  out.reserve(n * 3);
  for (size_t i = 0; i < n; ++i) out += "─";
  return out;
}

int height() {
  int h = 24;
#ifdef _WIN32
  CONSOLE_SCREEN_BUFFER_INFO info;
  HANDLE hd = GetStdHandle(STD_OUTPUT_HANDLE);
  if (hd != INVALID_HANDLE_VALUE && GetConsoleScreenBufferInfo(hd, &info))
    h = info.srWindow.Bottom - info.srWindow.Top + 1;
#else
  winsize ws;
  if (::ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 0) h = ws.ws_row;
#endif
  return h < 10 ? 10 : (h > 200 ? 200 : h);
}

void enter_alt() { std::fputs("\x1b[?1049h\x1b[2J\x1b[H", stdout); std::fflush(stdout); }
void leave_alt() { std::fputs("\x1b[?1049l", stdout); std::fflush(stdout); }
void hide_cursor() { std::fputs("\x1b[?25l", stdout); std::fflush(stdout); }
void show_cursor() { std::fputs("\x1b[?25h", stdout); std::fflush(stdout); }

void draw_lines(const std::vector<std::string>& lines) {
  // Управляющие внутри строки — это лишние ряды терминала. ANSI оставляем.
  auto flatten = [](const std::string& s) {
    std::string o;
    o.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
      if (s[i] == '\x1b') {
        size_t k = i;
        if (skip_ansi(s, k)) {
          o.append(s, i, k - i);
          i = k;
          continue;
        }
      }
      unsigned char b = (unsigned char)s[i];
      if (b == '\n' || b == '\r' || b == '\t' || b < 0x20 || b == 0x7f) {
        if (b == '\t') o.push_back(' ');
        ++i;
        continue;
      }
      o.push_back(s[i]);
      ++i;
    }
    return o;
  };
  const int H = height();
  const int W = width();
  const size_t n = std::min(lines.size(), (size_t)std::max(H, 1));
  // Автоперенос выключен на кадр: ровно W ячеек в последнем ряду иначе
  // прокручивают альтернативный экран, и следующий кадр ложится со сдвигом.
  std::string out = "\x1b[?7l\x1b[H";
  for (size_t i = 0; i < n; ++i) {
    out += pad(trunc(flatten(lines[i]), (size_t)W), (size_t)W);
    if (i + 1 < n) out += "\n";
  }
  out += "\x1b[J\x1b[?7h";
  std::fputs(out.c_str(), stdout);
  std::fflush(stdout);
}

// ---- Клавиатура -------------------------------------------------------------

RawTerminal::RawTerminal() {
#ifdef _WIN32
  HANDLE hin = GetStdHandle(STD_INPUT_HANDLE);
  DWORD mode = 0;
  if (hin != INVALID_HANDLE_VALUE && GetConsoleMode(hin, &mode)) {
    in_mode_ = mode;
    // VT-ввод: стрелки/функциональные приходят теми же escape-строками,
    // что и на POSIX — один парсер на обе платформы.
    if (SetConsoleMode(hin, mode | 0x0200 /*ENABLE_VIRTUAL_TERMINAL_INPUT*/ &
                             ~(DWORD)(0x0002 /*ECHO*/ | 0x0004 /*LINE*/)))
      active_ = true;
  }
#else
  if (::tcgetattr(0, &saved_) == 0) {
    termios t = saved_;
    t.c_lflag &= ~(tcflag_t)(ICANON | ECHO | ISIG);
    t.c_iflag &= ~(tcflag_t)(IXON | ICRNL);
    t.c_cc[VMIN] = 0;
    t.c_cc[VTIME] = 0;
    if (::tcsetattr(0, TCSANOW, &t) == 0) active_ = true;
  }
  // Полноэкранному интерфейсу нужны ~35 строк: в мелком окне кадры не
  // влезают и уезжают за верх. XTWINOPS (CSI 8;t) просят терминал
  // развернуться на время работы — исходный размер вернём на выходе.
  // Терминал, который последовательность не знает, её молча игнорирует.
  {
    const char* no_resize = ::getenv("LH_TUI_NO_RESIZE");
    winsize ws{};
    if ((!no_resize || !*no_resize) &&
        ::ioctl(1, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 0) {
      if (ws.ws_row < 35 || ws.ws_col < 100) {
        orig_rows_ = ws.ws_row;
        orig_cols_ = ws.ws_col;
        std::string req = "\x1b[8;35;" +
                          std::to_string(ws.ws_col < 100 ? 100 : ws.ws_col) + "t";
        ::fwrite(req.data(), 1, req.size(), stdout);
        ::fflush(stdout);
        // Окно могло не поддержать запрос — работаем с тем, что есть.
        winsize ws2{};
        if (::ioctl(1, TIOCGWINSZ, &ws2) == 0 && ws2.ws_row > 0) {
          orig_rows_ = ws2.ws_row;
          orig_cols_ = ws2.ws_col;
        }
      }
    }
  }
  enter_alt();
  hide_cursor();
#endif
}

RawTerminal::~RawTerminal() {
#ifdef _WIN32
  if (active_) SetConsoleMode(GetStdHandle(STD_INPUT_HANDLE), in_mode_);
#else
  if (orig_rows_ > 0) {
    std::string restore =
        "\x1b[8;" + std::to_string(orig_rows_) + ";" + std::to_string(orig_cols_) + "t";
    ::fwrite(restore.data(), 1, restore.size(), stdout);
    ::fflush(stdout);
  }
  leave_alt();
  show_cursor();
  if (active_) ::tcsetattr(0, TCSANOW, &saved_);
#endif
}

namespace {
// Очередь байтов: read() приносит сразу до 32 байт — escape-последовательность
// разбирается из очереди, остаток ждёт следующего вызова.
unsigned char g_buf[32];
ssize_t g_len = 0, g_pos = 0;

bool refill(int ms) {
  if (g_pos < g_len) return true;
#ifdef _WIN32
  (void)ms;
  unsigned char b = 0;
  DWORD got = 0;
  if (!ReadFile(GetStdHandle(STD_INPUT_HANDLE), &b, 1, &got, nullptr) || got == 0)
    return false;
  g_buf[0] = b;
  g_len = 1;
  g_pos = 0;
  return true;
#else
  fd_set rf;
  FD_ZERO(&rf);
  FD_SET(0, &rf);
  timeval tv{ms / 1000, (ms % 1000) * 1000};
  if (::select(1, &rf, nullptr, nullptr, &tv) <= 0) return false;
  g_len = ::read(0, g_buf, sizeof g_buf);
  g_pos = 0;
  return g_len > 0;
#endif
}

unsigned char next_byte(int ms) {
  // Нет данных (таймаут, EOF, EINTR): курсор не двигаем — иначе g_pos уходил
  // за g_len и за размер g_buf. 0xFF — вне терминаторного диапазона CSI
  // (0x40–0x7e): разбор закроется по лимиту длины seq, а не примет мусор.
  if (!refill(ms)) return 0xFF;
  return g_buf[g_pos++];
}
}  // namespace

Key read_key(int timeout_ms, char& chr) {
  chr = 0;
  if (!refill(timeout_ms)) return Key::None;
  unsigned char b = next_byte(0);
  chr = (char)b;
  if (b == '\r' || b == '\n') return Key::Enter;
  if (b == 0x7f || b == 0x08) return Key::Backspace;
  if (b == '\t') return Key::Tab;
  if (b == 0x04) return Key::CtrlD;
  if (b == 0x15) return Key::CtrlU;
  if (b == 0x18) return Key::CtrlX;
  if (b == 0x03) return Key::CtrlC;  // ISIG выключен: Ctrl-C — обычная клавиша
  if (b == 0x1b) {
    if (!refill(30)) return Key::Esc;  // одиночный Esc
    unsigned char b2 = next_byte(30);
    if (b2 == '[') {
      // CSI-последовательность: параметры до терминального байта (0x40–0x7e).
      unsigned char term = 0;
      std::string seq;
      for (;;) {
        unsigned char c3 = next_byte(30);
        if (c3 >= 0x40 && c3 <= 0x7e) { term = c3; break; }
        seq += (char)c3;
        if (seq.size() > 16) return Key::Esc;
      }
      switch (term) {
        case 'A': return Key::Up;    // модификаторы («1;5A») игнорируем
        case 'B': return Key::Down;
        case 'C': return Key::Right;
        case 'D': return Key::Left;
        case 'H': return Key::Home;
        case 'F': return Key::End;
        case '~':
          if (seq.empty() || seq[0] < '0' || seq[0] > '9') return Key::Esc;
          switch (::atoi(seq.c_str())) {
            case 1: case 7: return Key::Home;
            case 4: case 8: return Key::End;
            case 3: return Key::Del;
            case 5: return Key::PgUp;
            case 6: return Key::PgDn;
            case 11: return Key::F1;
            case 12: return Key::F2;
            case 13: return Key::F3;
            case 14: return Key::F4;
            case 15: return Key::F5;
            case 17: return Key::F6;
            case 18: return Key::F7;
            case 19: return Key::F8;
            case 20: return Key::F9;
            case 21: return Key::F10;
          }
          return Key::Esc;
      }
      return Key::Esc;
    }
    if (b2 == 'O') {  // «ESC O P» — F1–F4 в application-режиме
      unsigned char b3 = next_byte(30);
      switch (b3) {
        case 'P': return Key::F1;
        case 'Q': return Key::F2;
        case 'R': return Key::F3;
        case 'S': return Key::F4;
      }
      return Key::Esc;
    }
    return Key::Esc;
  }
  return Key::Char;  // печатный байт или кусок UTF-8 — собирается наверху
}

}  // namespace lh::ui
