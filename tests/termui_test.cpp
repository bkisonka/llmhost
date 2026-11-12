// Тесты termui: ширина строк (UTF-8 + ANSI), обрезка, сжатие таблиц,
// рамки панелей, бары и спиннер. Запускается вне tty: цвета выключены,
// рендер детерминирован.
#include "termui.h"

#include <cassert>
#include <cstdio>

using namespace lh::ui;

// Каждая строка многострочного рендера ровно w ячеек (рамки/таблицы).
static void assert_lines_fit(const std::string& s, size_t w) {
  for (size_t pos = 0; pos < s.size();) {
    size_t nl = s.find('\n', pos);
    if (nl == std::string::npos) nl = s.size();
    std::string line = s.substr(pos, nl - pos);
    if (line.empty()) continue;
    assert(dwidth(line) <= w);
    pos = nl + 1;
  }
}

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

#ifndef _WIN32
// Регрессия OOB-read (next_byte/refill): read() приносит ровно 32 байта —
// ёмкость g_buf, последние два байта — ESC '['. Первый же next_byte(30)
// CSI-цикла зовётся при пустом буфере: select таймаутит (пайп открыт, но
// молчит), и раньше refill(false) игнорировался — читался g_buf[32].
// Под ASAN старый код падает global-buffer-overflow, новый закрывает
// разбор сентинелом и лимитом длины seq.
static void csi_boundary_oob_test() {
  int fds[2];
  assert(pipe(fds) == 0);
  fflush(nullptr);
  pid_t pid = fork();
  assert(pid >= 0);
  if (pid == 0) {
    dup2(fds[0], 0);
    close(fds[0]);
    close(fds[1]);
    alarm(10); // страховка: зависание — тоже провал, падаем громко
    for (;;) {
      char chr = 0;
      Key k = read_key(2000, chr);
      if (k == Key::Esc) _exit(0);  // разбор закрылся, за границу не читали
      if (k != Key::Char) _exit(1); // None/клавиша — не договаривались
    }
  }
  close(fds[0]);
  std::string burst(30, 'a');
  burst += "\x1b[";
  assert(burst.size() == 32);
  ssize_t wr = ::write(fds[1], burst.data(), burst.size());
  assert(wr == (ssize_t)burst.size());
  // Пайп не закрываем: EOF сделал бы select читаемым, refill сбросил бы
  // g_pos до чтения за границей — сценарий размазался бы.
  int st = 0;
  waitpid(pid, &st, 0);
  close(fds[1]);
  assert(WIFEXITED(st) && WEXITSTATUS(st) == 0);
}
#endif

int main() {
  // Вне терминала панели не считаются «pretty», цвета выключены.
  init(false);
  assert(!pretty());
  assert(!color());

  // dwidth: кириллица и рамочные символы — по ячейке на символ; ANSI — ноль.
  assert(dwidth("") == 0);
  assert(dwidth("abc") == 3);
  assert(dwidth("привет") == 6);
  assert(dwidth("─│╭╮╰╯") == 6);
  assert(dwidth("\x1b[38;5;39mtext\x1b[0m") == 4);

  // trunc: влезает целиком; иначе «…» занимает ячейку; ANSI не теряется.
  assert(trunc("abcdef", 10) == "abcdef");
  assert(trunc("abcdef", 4) == "abc…");
  assert(dwidth(trunc("привет мир", 6)) == 6);
  {
    std::string colored = "\x1b[38;5;114mзагружена\x1b[0m";
    std::string t = trunc(colored, 6);
    assert(dwidth(t) == 6);
    assert(t.find("\x1b[0m") != std::string::npos);  // цвет закрыт после среза
  }

  // pad/padl: выравнивание по ширине отображения.
  assert(dwidth(pad("abc", 5)) == 5);
  assert(pad("abcdef", 3) == "ab…");  // 2 символа + «…» = 3 ячейки
  assert(padl("abc", 5) == "  abc");

  // Таблица: на широком терминале всё на месте; на узком — вписывается и
  // обрезается с «…», сетка не разваливается.
  {
    Table t;
    t.col("МОДЕЛЬ")
        .col("СОСТОЯНИЕ")
        .col("КАРТА", Table::Right)
        .col("ПРИМЕЧАНИЕ", Table::Left, 20);
    t.row({"Qwen3.6-35B-A3B-NVFP4-MTP-TURBO", "загружена", "1",
           "основная рабочая модель"});
    t.row({"Qwen3-0.6B-Q8_0", "выгружена", "0", ""});
    std::string wide = t.render(200);
    assert(wide.find("Qwen3.6-35B-A3B-NVFP4-MTP-TURBO") != std::string::npos);
    // Потолок max_w=20 у ПРИМЕЧАНИЙ: 23 ячейки обрезаются с «…».
    assert(wide.find("основная рабочая") != std::string::npos);
    assert(wide.find("основная рабочая модель") == std::string::npos);

    std::string narrow = t.render(50);
    assert_lines_fit(narrow, 50);
    assert(narrow.find("…") != std::string::npos);
    // Заголовок и рамочные символы на месте.
    assert(narrow.find("МОДЕЛЬ") != std::string::npos);
    assert(narrow.find("├") != std::string::npos && narrow.find("┤") != std::string::npos);
  }

  // Панель: рамка ровно в ширину, заголовок в кромке, содержимое обрезано.
  {
    Panel p("Заголовок");
    p.kv("Ключ", "значение");
    p.hr();
    p.line("длинная строка, которая гарантированно не влезет в узкую панель целиком");
    std::string s = p.render(40);
    for (size_t pos = 0; pos < s.size();) {
      size_t nl = s.find('\n', pos);
      if (nl == std::string::npos) nl = s.size();
      std::string line = s.substr(pos, nl - pos);
      if (!line.empty()) assert(dwidth(line) == 40);
      pos = nl + 1;
    }
    assert(s.find("Заголовок") != std::string::npos);
    assert(s.find("Ключ:") != std::string::npos);
    assert(s.find("целиком") == std::string::npos);  // не влезло — срезано
  }

  // Бар: границы и ширина.
  assert(bar(0.0, 10) == "░░░░░░░░░░");
  assert(bar(1.0, 10) == "██████████");
  assert(bar(0.5, 10) == "█████░░░░░");
  assert(bar(0.5, 0).empty());
  assert(dwidth(bar(0.3, 7)) == 7);

  // Переносы стрима: перевод строки — новая строка ленты, не символ внутри
  // ряда. Иначе кадр выше окна, верх уезжает из альтернативного экрана.
  {
    auto lines = hard_wrap("абв\nгде", 2);
    assert(lines.size() == 4);
    assert(lines[0] == "аб" && lines[1] == "в");
    assert(lines[2] == "гд" && lines[3] == "е");
    for (auto& l : lines) {
      assert(l.find('\n') == std::string::npos);
      assert(dwidth(l) <= 2);
    }
    auto crlf = hard_wrap("a\r\nb\nc", 10);
    assert(crlf.size() == 3 && crlf[0] == "a" && crlf[1] == "b" && crlf[2] == "c");
    auto blank = hard_wrap("hello\n\nworld", 80);
    assert(blank.size() == 3 && blank[0] == "hello" && blank[1].empty() && blank[2] == "world");
    auto empty = hard_wrap("", 10);
    assert(empty.size() == 1 && empty[0].empty());
    auto ctl = hard_wrap("a\tb\x01" "c", 10); // \x01c съел бы «c» как hex
    assert(ctl.size() == 1 && ctl[0] == "a bc");
    auto exact = hard_wrap("abcd", 4);
    assert(exact.size() == 1 && exact[0] == "abcd");
  }

  // Спиннер: кадры разные, по одной ячейке.
  const char* f1 = spinner();
  const char* f2 = spinner();
  assert(f1 != f2 && dwidth(f1) == 1);

  // CSI-разбор на границе 32-байтной порции: чтения за границей g_buf нет.
#ifndef _WIN32
  csi_boundary_oob_test();
#endif

  // key_seq: командная клавиша как UTF-8 код. Кириллица — два байта
  // (ь = D1 8C, Д = D0 94); окно — предыдущий байт Key::Char.
  {
    // Латиница — один байт, prev не участвует (продолжения UTF-8 всегда
    // >= 0x80 и с ASCII-командой не сталкиваются — гарантия формата).
    assert(key_seq(0, 'm', "m"));
    assert(key_seq(0x81, 'm', "m"));
    assert(!key_seq(0, 'M', "m"));           // регистр — разные клавиши
    // Кириллица: точная пара.
    assert(key_seq(0xD1, '\x8C', "ь"));
    assert(key_seq(0xD0, '\x94', "Д"));
    // Не та ведущая доля, не тот хвост, обратный порядок — мимо.
    assert(!key_seq(0xD0, '\x8C', "ь"));
    assert(!key_seq(0x8C, '\xD1', "ь"));
    assert(!key_seq(0xD1, '\x8F', "ь"));     // я вместо ь — та же ведущая доля
    // Латиница мимо кириллической команды и наоборот.
    assert(!key_seq(0xD1, 'm', "ь"));
    assert(!key_seq(0, '\x8C', "m"));
    // Последовательности из 3+ байт не матчатся (командных клавиш среди них
    // нет — сознательное ограничение, см. комментарий в termui.h).
    assert(!key_seq(0xE2, '\x82', "\xE2\x82\xAC"));  // € — трёхбайтовый
  }

  std::puts("termui: ok");
  return 0;
}
