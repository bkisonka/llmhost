// Детерминированный тест рендера Markdown: блоки/runs и RTF, который уйдёт
// в RichEdit (таблицы — \trowd/\cellx/\cell/\row). Печать — глазами видно,
// что получит контрол.
#include "mdrender.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

using namespace lh::md;
using std::string;
using std::wstring;
using std::vector;

static void put_ws(const wstring& s) {
  for (wchar_t c : s)
    if (c == L'\n') std::printf("\\n"); else std::putwchar(c);
}

static void dump(const char* title, const vector<Block>& blocks) {
  std::printf("---- %s ----\n", title);
  for (auto& b : blocks) {
    if (b.kind == Block::kTable) {
      std::printf("[таблица %zux%zu шапка=%zu align=", b.table.rows.size(),
                  b.table.rows.empty() ? 0 : b.table.rows[0].size(), b.table.header_rows);
      for (int a : b.table.align) std::printf("%d,", a);
      std::printf("]\n");
      for (auto& row : b.table.rows) {
        std::printf("  |");
        for (auto& cell : row) {
          for (auto& r : cell) { std::printf("%d:", r.style); put_ws(r.text); }
          std::printf("|");
        }
        std::printf("\n");
      }
    } else {
      std::printf("[строка]");
      for (auto& r : b.runs) { std::printf(" %d:", r.style); put_ws(r.text); }
      std::printf("\n");
    }
  }
}

static vector<Run> text_runs(const vector<Block>& blocks) {
  vector<Run> out;
  for (auto& b : blocks)
    if (b.kind == Block::kText) out.insert(out.end(), b.runs.begin(), b.runs.end());
  return out;
}

static bool has_run(const vector<Run>& runs, const wstring& text, int style) {
  for (auto& r : runs)
    if (r.text == text && r.style == style) return true;
  return false;
}

static bool has_cell(const Table& t, size_t row, size_t col, const wstring& text, int style) {
  if (row >= t.rows.size() || col >= t.rows[row].size()) return false;
  for (auto& r : t.rows[row][col])
    if (r.text == text && r.style == style) return true;
  return false;
}

static size_t count(const string& s, const string& needle) {
  size_t n = 0, pos = 0;
  while ((pos = s.find(needle, pos)) != string::npos) { ++n; pos += needle.size(); }
  return n;
}

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0); // abort() не должен терять дамп

  // 1. Таблица с сепаратором и выравниваниями: шапка жирная + заливка,
  //    «:---:» — центр, «---:» — право, ячейки — инлайн-runs.
  {
    auto blocks = render(L"| Color | Hex |\n| :---: | ---: |\n| Red | #FF0000 |\n| Blue | #0000FF |\n");
    dump("таблица с сепаратором", blocks);
    assert(blocks.size() == 1 && blocks[0].kind == Block::kTable);
    const Table& t = blocks[0].table;
    assert(t.rows.size() == 3);
    assert(t.header_rows == 1);
    assert(t.align.size() == 2 && t.align[0] == 1 && t.align[1] == 2);
    assert(has_cell(t, 0, 0, L"Color", MD_BOLD));
    assert(has_cell(t, 0, 1, L"Hex", MD_BOLD));
    assert(has_cell(t, 1, 0, L"Red", MD_NORMAL));
    assert(has_cell(t, 1, 1, L"#FF0000", MD_NORMAL));
    assert(has_cell(t, 2, 0, L"Blue", MD_NORMAL));
    // сепараторная строка |:---:|---:| не попала в таблицу
    for (auto& row : t.rows)
      for (auto& cell : row)
        for (auto& r : cell) assert(r.text.find(L"---") == wstring::npos);

    string rtf = to_rtf(blocks, 9000);
    std::printf("---- rtf таблицы ----\n%s\n\n", rtf.c_str());
    assert(rtf.find("{\\rtf1") == 0);
    assert(rtf.find("Segoe UI") != string::npos && rtf.find("Consolas") != string::npos);
    // \colortbl обязан быть группой в скобках — иначе MSFTEDIT не узнаёт его
    // и вываливает «;» текстом.
    assert(rtf.find("{\\colortbl;") != string::npos);
    // 3 строки × 2 ячейки
    assert(count(rtf, "\\trowd") == 3);
    assert(count(rtf, "\\row") == 3);
    assert(count(rtf, "\\cellx") == 6);
    assert(rtf.find("\\qc") != string::npos);      // :---: → центр
    assert(rtf.find("\\qr") != string::npos);      // ---: → право
    assert(rtf.find("\\clcbpat2") != string::npos); // заливка шапки
    // \clshdng не должно быть: у MSFTEDIT оно заливку ячейки убивает
    assert(rtf.find("\\clshdng") == string::npos);
    assert(rtf.find("\\brdrcf3") != string::npos);  // рамки серым
    // ширины: контент колонок 5 (Color) и 7 (#FF0000) из 12 → 3750 + 5250
    assert(rtf.find("\\cellx3750") != string::npos);
    assert(rtf.find("\\cellx9000") != string::npos);
    assert(rtf.find("\\cellx3750") < rtf.find("\\cellx9000"));
  }

  // 2. Модель забыла сепаратор: всё равно таблица, первая строка — шапка,
  //    выравнивание — по левому краю.
  {
    auto blocks = render(L"| Имя | Год |\n| Qwen | 2025 |\n| Llama | 2024 |\n");
    dump("таблица без сепаратора", blocks);
    assert(blocks.size() == 1 && blocks[0].kind == Block::kTable);
    const Table& t = blocks[0].table;
    assert(t.header_rows == 1);
    assert(t.align.size() == 2 && t.align[0] == 0 && t.align[1] == 0);
    assert(has_cell(t, 0, 0, L"Имя", MD_BOLD));
    assert(has_cell(t, 1, 0, L"Qwen", MD_NORMAL));
    assert(has_cell(t, 2, 0, L"Llama", MD_NORMAL));
    assert(to_rtf(blocks, 9000).find("\\ql") != string::npos);
  }

  // 3. Таблица в начале ответа (первая строка) — та, что терялась раньше.
  {
    auto blocks = render(L"| A | B |\n| --- | --- |\n| 1 | 2 |\n");
    assert(blocks[0].kind == Block::kTable && blocks[0].table.rows.size() == 2);
    assert(has_cell(blocks[0].table, 0, 0, L"A", MD_BOLD));
    assert(has_cell(blocks[0].table, 1, 1, L"2", MD_NORMAL));
  }

  // 4. Таблица между абзацами: окружающий текст не съедается, в RTF таблица
  //    лежит между «До:» и «После:».
  {
    auto blocks = render(L"До:\n| A | B |\n| --- | --- |\n| 1 | 2 |\nПосле:\n");
    dump("таблица между абзацами", blocks);
    assert(blocks.size() == 3);
    assert(blocks[0].kind == Block::kText && blocks[2].kind == Block::kText);
    assert(blocks[1].kind == Block::kTable);
    auto runs = text_runs(blocks);
    assert(has_run(runs, L"До:", MD_NORMAL));
    assert(has_run(runs, L"После:", MD_NORMAL));
    string rtf = to_rtf(blocks, 9000);
    assert(rtf.find("\\u1044?") != string::npos);  // «Д» — кириллица \u-эскейпами
    assert(rtf.find("\\u1055?") != string::npos);  // «П»
    assert(rtf.find("\\u1044?") < rtf.find("\\trowd"));
    assert(rtf.find("\\trowd") < rtf.find("\\u1055?"));
  }

  // 5. Строка с «|», но не таблица (не начинается с «|») — остаётся текстом.
  {
    auto blocks = render(L"формула a | b — дизъюнкция\n");
    assert(has_run(text_runs(blocks), L"формула a | b — дизъюнкция", MD_NORMAL));
  }

  // 6. Инлайн в ячейках: жирный и код работают и внутри таблицы.
  {
    auto blocks = render(L"| a **b** c | `x` |\n| --- | --- |\n| d | e |\n");
    const Table& t = blocks[0].table;
    assert(has_cell(t, 0, 0, L"b", MD_BOLD));
    assert(has_cell(t, 0, 0, L"a ", MD_BOLD));
    assert(has_cell(t, 0, 1, L"x", MD_CODE));
  }

  // 7. Инлайн: жирный, курсив, код. Фон кода RTF не несёт — \highlight
  //    MSFTEDIT не рисует, фон ставит chatui пост-обработкой.
  {
    auto blocks = render(L"это **жирный**, *курсив* и `код`.\n");
    auto runs = text_runs(blocks);
    assert(has_run(runs, L"жирный", MD_BOLD));
    assert(has_run(runs, L"курсив", MD_ITALIC));
    assert(has_run(runs, L"код", MD_CODE));
    assert(has_run(runs, L"это ", MD_NORMAL));
    assert(has_run(runs, L", ", MD_NORMAL));
    string rtf = to_rtf(blocks, 9000);
    assert(rtf.find("\\f1\\fs17") != string::npos);
    assert(rtf.find("highlight") == string::npos);
  }

  // 8. Заголовки и список.
  {
    auto blocks = render(L"# Title\n## Sub\n### Deep\n- пункт один\n- пункт два\n");
    auto runs = text_runs(blocks);
    assert(has_run(runs, L"Title", MD_BOLD | MD_H1));
    assert(has_run(runs, L"Sub", MD_BOLD | MD_H2));
    assert(has_run(runs, L"Deep", MD_BOLD | MD_H3));
    assert(has_run(runs, L"  • ", MD_NORMAL));
    assert(has_run(runs, L"пункт один", MD_NORMAL));
    // заголовки залиты абзацной \shading, списки — нет
    assert(blocks[0].shade && blocks[1].shade && blocks[2].shade);
    assert(!blocks[3].shade);
    string rtf = to_rtf(blocks, 9000);
    assert(rtf.find("\\fs24") != string::npos);    // H1 — 12pt
    assert(rtf.find("\\fs21") != string::npos);    // H2
    assert(rtf.find("\\fs20") != string::npos);    // H3
    assert(rtf.find("\\shading10000\\cbpat2") != string::npos);
  }

  // 9. Код-блок: моноширинный, каждая строка — свой залитый блок.
  {
    auto blocks = render(L"```\nint x = 1;\nint y = 2;\n```\n");
    auto runs = text_runs(blocks);
    assert(has_run(runs, L"int x = 1;", MD_CODE));
    assert(has_run(runs, L"int y = 2;", MD_CODE));
    assert(blocks[0].kind == Block::kText && blocks[0].shade);
    assert(to_rtf(blocks, 9000).find("\\shading10000\\cbpat2") != string::npos);
  }

  // 10. Разделитель.
  {
    auto blocks = render(L"до\n\n---\n\nпосле\n");
    int hr = 0;
    for (auto& r : text_runs(blocks))
      if (r.text.find(L"────") == 0) ++hr;
    assert(hr == 1);
  }

  // 11. Последняя строка без \n не теряется.
  {
    auto blocks = render(L"хвост без перевода");
    assert(has_run(text_runs(blocks), L"хвост без перевода", MD_NORMAL));
  }

  // 12. Экранирование RTF: слэш удваивается, скобки — \{ \}.
  {
    string rtf = to_rtf(render(L"путь C:\\dir и {скобки}\n"), 9000);
    assert(rtf.find("C:\\\\dir") != string::npos);
    assert(rtf.find("\\{") != string::npos && rtf.find("\\}") != string::npos);
  }

  // 13. Ссылки [текст](url): MD_LINK + адрес в run; жирный внутри подписи
  //     сохраняет оба стиля и получает адрес; кривые скобки — литеральный
  //     текст; RTF несёт \ul (вид), клик добирает chatui пост-обработкой.
  {
    auto blocks = render(L"бери [Скачать](https://example.com/a.zip) и [док](http://x.ru)\n");
    auto runs = text_runs(blocks);
    int links = 0;
    for (auto& r : runs)
      if (r.style & MD_LINK) { ++links; assert(!r.url.empty()); }
    assert(links == 2);
    assert(has_run(runs, L"Скачать", MD_LINK));
    assert(has_run(runs, L"бери ", MD_NORMAL));
    for (auto& r : runs)
      if (r.text == L"Скачать") assert(r.url == L"https://example.com/a.zip");

    auto runs2 = text_runs(render(L"[**жирная** ссылка](https://y.ru)\n"));
    assert(has_run(runs2, L"жирная", MD_BOLD | MD_LINK));
    assert(has_run(runs2, L" ссылка", MD_LINK));
    for (auto& r : runs2)
      if (r.text == L"жирная") assert(r.url == L"https://y.ru");

    // нет «](» или нет «)» — не ссылка, текст как есть
    assert(has_run(text_runs(render(L"раз [с скобкой] без url\n")),
                   L"раз [с скобкой] без url", MD_NORMAL));
    assert(has_run(text_runs(render(L"два [с скобкой](без конца\n")),
                   L"два [с скобкой](без конца", MD_NORMAL));

    string rtf = to_rtf(render(L"[x](https://z.ru)\n"), 9000);
    assert(rtf.find("\\ul ") != string::npos); // подчёркивание — вид ссылки
  }

  // 14. Потолок рекурсии inline_runs: маркеры глубже kMaxDepth остаются
  // literal-текстом (ответ модели произвольной вложенности не клал стек).
  {
    auto ok = inline_runs(L"**a**", 0, 31); // последний вложенный уровень
    assert(has_run(ok, L"a", MD_BOLD));
    auto cap = inline_runs(L"**a**", 0, 32); // на потолке — literal, как «кривые скобки»
    assert(has_run(cap, L"**a**", MD_NORMAL));
    // Длинный чередующийся маркерный мусор должен рендериться без падения.
    wstring deep;
    for (int i = 0; i < 5000; ++i) deep += L"**";
    auto blocks = render(deep + L"\n");
    string rtf = to_rtf(blocks, 9000);
    assert(rtf.find("{\\rtf1") == 0);
  }

  // 15. Control-байты в RTF: NUL обрывал поток MSFTEDIT, \x01 — мусор;
  // табуляция/переводы (как раньше) проходят, em-dash и non-BMP — \uNNNN.
  {
    wstring code = L"`a";
    code += wchar_t(0x01);
    code += wchar_t(0x02);
    code += L"b\u2014""c`";
    string rtf = to_rtf(render(code + L"\n"), 9000);
    assert(rtf.find('\x01') == string::npos);
    assert(rtf.find('\x02') == string::npos);
    assert(rtf.find("\\u8212?") != string::npos); // em-dash через \u
  }

  std::printf("mdrender_test: ok\n");
  return 0;
}
