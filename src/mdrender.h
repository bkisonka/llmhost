// Мини-рендер Markdown для чата: блоки → runs (текст) и RTF для RichEdit.
// Жирный/курсив, инлайн-код, код-блоки, заголовки, списки, разделители.
// Таблицы — настоящим RTF (\trowd/\cellx/\cell/\row): рамки, заливка шапки,
// выравнивание колонок из :---: / ---:. Шапка — строки до сепаратора |---|,
// без сепаратора — первая строка, модели его забывают.
// Чистые функции без зависимостей — детерминированный тест рядом.
#pragma once
#include <algorithm>
#include <string>
#include <vector>

namespace lh::md {

// Стили — биты (S_* заняты макросами Windows SDK, поэтому MD_*).
constexpr int MD_NORMAL = 0;
constexpr int MD_BOLD = 1;
constexpr int MD_ITALIC = 2;
constexpr int MD_CODE = 4;  // моноширинный + фон
constexpr int MD_H1 = 8;
constexpr int MD_H2 = 16;
constexpr int MD_H3 = 32;
constexpr int MD_LINK = 64; // кликабельная ссылка, адрес в Run::url

struct Run {
  std::wstring text;
  int style = 0;
  std::wstring url; // для MD_LINK: куда ведёт ссылка
};

// Таблица как структура: ячейки — инлайн-runs (жирный и `код` работают и там).
struct Table {
  std::vector<std::vector<std::vector<Run>>> rows; // [строка][колонка] → runs
  size_t header_rows = 0;                          // первые строки — шапка
  std::vector<int> align; // колонка: 0 влево, 1 центр (:---:), 2 право (---:)
};

// Блок вывода: строка текста или таблица. shade — заливка абзаца целиком
// (код-блоки, заголовки): RTF \shading10000\cbpat рисует полную полосу,
// в отличие от \highlight на тексте, который MSFTEDIT не рисует вовсе.
struct Block {
  enum Kind { kText, kTable };
  Kind kind = kText;
  bool shade = false;
  std::vector<Run> runs;
  Table table;
};

inline std::wstring trim(const std::wstring& s) {
  size_t a = s.find_first_not_of(L" \t\r");
  if (a == std::wstring::npos) return L"";
  size_t b = s.find_last_not_of(L" \t\r");
  return s.substr(a, b - a + 1);
}

// Инлайн-сканер: **жирный**, *курсив*, `код`, [текст](url) — ссылка.
// Код перекрывает прочее; кривые скобки остаются литеральным текстом.
// Глубина рекурсии ограничена: чередование **«…**…[*…*]](… даёт
// линейную вложенность от длины строки, а строка — ответ модели (недоверенный
// вход); без потолка длинный ответ клал стек хоста (chatui/TUI). На пределе
// глубины маркеры этого уровня остаются literal-текстом, как «кривые скобки».
inline std::vector<Run> inline_runs(const std::wstring& s, int base, int depth = 0) {
  constexpr int kMaxDepth = 32;
  const bool nested = depth < kMaxDepth;
  std::vector<Run> out;
  std::wstring cur;
  auto flush = [&](int style) {
    if (!cur.empty()) { out.push_back({cur, style}); cur.clear(); }
  };
  size_t i = 0;
  while (i < s.size()) {
    if (nested && s.compare(i, 2, L"**") == 0) {
      size_t e = s.find(L"**", i + 2);
      if (e != std::wstring::npos && e > i + 2) {
        flush(base);
        for (auto& r : inline_runs(s.substr(i + 2, e - i - 2), base | MD_BOLD, depth + 1)) out.push_back(r);
        i = e + 2;
        continue;
      }
    }
    if (s[i] == L'`') {
      size_t e = s.find(L'`', i + 1);
      if (e != std::wstring::npos && e > i + 1) {
        flush(base);
        out.push_back({s.substr(i + 1, e - i - 1), MD_CODE});
        i = e + 1;
        continue;
      }
    }
    if (nested && s[i] == L'*') {
      size_t e = s.find(L'*', i + 1);
      if (e != std::wstring::npos && e > i + 1) {
        flush(base);
        for (auto& r : inline_runs(s.substr(i + 1, e - i - 1), base | MD_ITALIC, depth + 1)) out.push_back(r);
        i = e + 1;
        continue;
      }
    }
    if (nested && s[i] == L'[') {
      size_t e = s.find(L"](", i + 1);
      if (e != std::wstring::npos && e > i + 1) {
        size_t u = s.find(L')', e + 2);
        if (u != std::wstring::npos && u > e + 2) {
          flush(base);
          std::wstring url = s.substr(e + 2, u - e - 2);
          for (auto& r : inline_runs(s.substr(i + 1, e - i - 1), base | MD_LINK, depth + 1)) {
            r.url = url;
            out.push_back(std::move(r));
          }
          i = u + 1;
          continue;
        }
      }
    }
    cur += s[i];
    ++i;
  }
  flush(base);
  return out;
}

inline std::vector<std::wstring> split_table_row(const std::wstring& line) {
  std::vector<std::wstring> cells;
  std::wstring cur;
  for (size_t i = 1; i < line.size(); ++i) { // пропускаем первую «|»
    if (line[i] == L'|') { cells.push_back(trim(cur)); cur.clear(); continue; }
    cur += line[i];
  }
  cells.push_back(trim(cur));
  if (!cells.empty() && cells.back().empty()) cells.pop_back();
  return cells;
}

// Сепаратор |---|---|: true, если строка им является; align — выравнивание
// колонок (по двоеточиям), заполняется только при true.
inline bool table_separator(const std::wstring& line, std::vector<int>& align) {
  std::wstring t = trim(line);
  if (t.size() < 5 || t.front() != L'|' || t.back() != L'|') return false;
  for (wchar_t c : t)
    if (c != L'|' && c != L'-' && c != L':' && c != L' ') return false;
  align.clear();
  for (auto& cell : split_table_row(t)) {
    bool l = !cell.empty() && cell.front() == L':';
    bool r = !cell.empty() && cell.back() == L':';
    align.push_back(l && r ? 1 : (r ? 2 : 0));
  }
  return true;
}

// Сырые строки → Table: все строки добиваются до общего числа колонок,
// ячейки прогоняются через инлайн-рендер, шапка — жирным.
inline Table make_table(const std::vector<std::vector<std::wstring>>& raw, size_t header_rows,
                        const std::vector<int>& align) {
  Table t;
  size_t cols = 0;
  for (auto& r : raw) cols = (std::max)(cols, r.size());
  t.header_rows = (std::min)(header_rows, raw.size());
  t.align = align;
  t.align.resize(cols, 0);
  for (size_t i = 0; i < raw.size(); ++i) {
    std::vector<std::vector<Run>> cells;
    for (size_t c = 0; c < cols; ++c)
      cells.push_back(inline_runs(c < raw[i].size() ? raw[i][c] : L"",
                                  i < t.header_rows ? MD_BOLD : MD_NORMAL));
    t.rows.push_back(std::move(cells));
  }
  return t;
}

inline std::vector<Block> render(const std::wstring& src) {
  std::vector<Block> out;
  auto text_block = [&](std::vector<Run> runs, bool shade = false) {
    Block b;
    b.kind = Block::kText;
    b.shade = shade;
    b.runs = std::move(runs);
    out.push_back(std::move(b));
  };
  bool fence = false;
  size_t i = 0;
  while (i < src.size()) {
    size_t eol = src.find(L'\n', i);
    std::wstring line = src.substr(i, eol == std::wstring::npos ? src.size() - i : eol - i);
    i = eol == std::wstring::npos ? src.size() : eol + 1;

    std::wstring t = trim(line);
    if (t.rfind(L"```", 0) == 0) { fence = !fence; continue; }
    if (fence) { text_block({{line, MD_CODE}}, true); continue; }

    // Таблица: блок строк с «|». Сканируем С ТЕКУЩЕЙ строки — i уже на
    // следующей, без сдвига первая строка таблицы терялась.
    if (t.size() > 2 && t.front() == L'|') {
      std::vector<std::vector<std::wstring>> rows;
      std::vector<int> align;
      bool sep = false;
      size_t header_rows = 0;
      size_t k = i - line.size() - (eol == std::wstring::npos ? 0 : 1);
      while (k <= src.size()) {
        size_t e2 = src.find(L'\n', k);
        std::wstring l2 = src.substr(k, e2 == std::wstring::npos ? src.size() - k : e2 - k);
        std::wstring t2 = trim(l2);
        if (t2.empty() || t2.front() != L'|') break;
        std::vector<int> a;
        if (table_separator(t2, a)) {
          sep = true;
          align = a;
          header_rows = rows.size(); // строки до сепаратора — шапка
        } else {
          rows.push_back(split_table_row(t2));
        }
        if (e2 == std::wstring::npos) { k = src.size() + 1; break; }
        k = e2 + 1;
      }
      bool consistent = rows.size() >= 2;
      if (consistent)
        for (auto& r : rows)
          if (r.size() != rows[0].size()) { consistent = false; break; }
      if (rows.size() >= 2 && (sep || consistent)) {
        // Без сепаратора шапки нет — но первая строка жирная: у модели всегда
        // заголовок в первой строке, даже если сепаратор забыли.
        size_t head = sep ? (std::max<size_t>)(header_rows, 1) : 1;
        Block b;
        b.kind = Block::kTable;
        b.table = make_table(rows, head, sep ? align : std::vector<int>{});
        out.push_back(std::move(b));
        i = k;
        continue;
      }
    }

    if (t.rfind(L"### ", 0) == 0) {
      text_block(inline_runs(t.substr(4), MD_BOLD | MD_H3), true);
      continue;
    }
    if (t.rfind(L"## ", 0) == 0) {
      text_block(inline_runs(t.substr(3), MD_BOLD | MD_H2), true);
      continue;
    }
    if (t.rfind(L"# ", 0) == 0) {
      text_block(inline_runs(t.substr(2), MD_BOLD | MD_H1), true);
      continue;
    }
    if (t == L"---" || t == L"***" || t == L"___") {
      text_block({{L"────────────────────", MD_NORMAL}});
      continue;
    }
    if ((t.rfind(L"- ", 0) == 0 || t.rfind(L"* ", 0) == 0) && t.size() > 2) {
      std::vector<Run> runs{{L"  • ", MD_NORMAL}};
      for (auto& r : inline_runs(t.substr(2), MD_NORMAL)) runs.push_back(r);
      text_block(std::move(runs));
      continue;
    }
    if (t.empty()) { text_block({}); continue; }
    text_block(inline_runs(line, MD_NORMAL));
  }
  return out;
}

// ---- RTF -----------------------------------------------------------------------

// Текст RTF: {\} экранируются, всё вне ASCII — \uNNNN? (RTF — 7-битный).
inline void rtf_text(std::string& out, const std::wstring& s) {
  for (wchar_t c : s) {
    switch (c) {
      case L'\\': out += "\\\\"; break;
      case L'{': out += "\\{"; break;
      case L'}': out += "\\}"; break;
      default:
        // Control-байты в RTF-поток писать нельзя: NUL обрывает поток
        // MSFTEDIT, DEL и прочие <0x20 — мусор в разметке. Табуляцию и
        // переводы строки оставляем (как и раньше: \t\n\r проходили как
        // есть).
        if ((c < 32 && c != 0x09 && c != 0x0A && c != 0x0D) || c == 127) break;
        if (c < 128) { out += (char)c; break; }
        // \u в RTF — единицы UTF-16. (int)(short) ломал non-BMP на машинах
        // с 32-битным wchar_t: эмодзи превращались в отрицательное число.
        if (c > 0xFFFF) {
          unsigned long v = (unsigned long)c - 0x10000ul;
          out += "\\u" + std::to_string((int)(short)(0xD800u + (v >> 10))) + "?";
          out += "\\u" + std::to_string((int)(short)(0xDC00u + (v & 0x3FFu))) + "?";
        } else {
          out += "\\u" + std::to_string((int)(short)c) + "?";
        }
    }
  }
}

// Один run: те же шрифты/размеры/цвета, что носил EM_SETCHARFORMAT, —
// Segoe UI 9pt (тёмно-серый), код и заголовки — Consolas мельче. Фон текста
// (\highlight/\cb) MSFTEDIT не рисует — заливку даёт абзацная \shading
// у блока (Block::shade) или EM_SETCHARFORMAT после стрима (инлайн-код).
inline void rtf_run(std::string& out, const Run& r) {
  out += "{\\f0\\fs18\\cf1";
  if (r.style & MD_BOLD) out += "\\b";
  if (r.style & MD_ITALIC) out += "\\i";
  if (r.style & MD_LINK) out += "\\ul";
  if (r.style & (MD_CODE | MD_H1 | MD_H2 | MD_H3)) {
    out += "\\f1\\fs17";
    if (r.style & MD_H1) out += "\\fs24";
    else if (r.style & MD_H2) out += "\\fs21";
    else if (r.style & MD_H3) out += "\\fs20";
  }
  out += " ";
  rtf_text(out, r.text);
  out += "}";
}

// Ширины колонок (twips): пропорционально самому широкому содержимому,
// не уже 500; остаток от округления — последней колонке, правый край ровно
// на width. Слишком широких колонок не даём (потолок 60% таблицы).
inline std::vector<int> column_widths(const Table& t, int width) {
  size_t cols = t.rows.empty() ? 0 : t.rows[0].size();
  if (cols == 0) return {};
  std::vector<long long> content(cols, 3);
  for (auto& row : t.rows)
    for (size_t c = 0; c < row.size() && c < cols; ++c) {
      long long len = 0;
      for (auto& run : row[c]) len += (long long)run.text.size();
      content[c] = (std::max)(content[c], len);
    }
  long long sum = 0;
  for (long long v : content) sum += v;
  std::vector<int> cw(cols, 0);
  int cap = width * 3 / 5;
  int used = 0;
  for (size_t c = 0; c + 1 < cols; ++c) {
    int v = (int)(width * content[c] / sum);
    cw[c] = (std::min)((std::max)(v, 500), cap);
    used += cw[c];
  }
  cw[cols - 1] = (std::max)(width - used, 500);
  return cw;
}

// Таблица строками RTF: каждая строка — \trowd с описаниями ячеек (\cellx),
// затем содержимое с \cell и \row. Шапка — жирная (стили в runs) + заливка
// \clcbpat2. ВАЖНО: без \clshdng — проверено живым контролом, \clshdng10000
// у MSFTEDIT заливку ячейки убивает, а без него — рисует.
inline void rtf_table(std::string& out, const Table& t, int width) {
  std::vector<int> cw = column_widths(t, width);
  if (cw.empty()) return;
  static const char* const ALIGN[] = {"\\ql", "\\qc", "\\qr"};
  for (size_t i = 0; i < t.rows.size(); ++i) {
    bool head = i < t.header_rows;
    out += "\\trowd\\trgaph60\\trleft0";
    int x = 0;
    for (size_t c = 0; c < t.rows[i].size() && c < cw.size(); ++c) {
      x += cw[c];
      out += "\\clbrdrt\\brdrs\\brdrw4\\brdrcf3\\clbrdrl\\brdrs\\brdrw4\\brdrcf3"
             "\\clbrdrb\\brdrs\\brdrw4\\brdrcf3\\clbrdrr\\brdrs\\brdrw4\\brdrcf3";
      if (head) out += "\\clcbpat2";
      out += "\\cellx" + std::to_string(x);
    }
    for (size_t c = 0; c < t.rows[i].size(); ++c) {
      int a = c < t.align.size() ? t.align[c] : 0;
      if (a < 0 || a > 2) a = 0;
      out += "\\pard\\intbl";
      out += ALIGN[a];
      for (auto& run : t.rows[i][c]) rtf_run(out, run);
      out += "\\cell";
    }
    out += "\\row";
  }
  // После таблицы — вернуть абзацные умолчания, иначе \intbl липнет к хвосту.
  out += "\\pard\\f0\\fs18\\cf1 ";
}

// Блоки → RTF-документ целиком: chatui вливает его в RichEdit одним
// EM_STREAMIN (SF_RTF|SFF_SELECTION). table_width — ширина ленты в twips.
inline std::string to_rtf(const std::vector<Block>& blocks, int table_width) {
  // Цвета: 1 — текст #1a1a1a, 2 — фон кода/шапки #f2f2f2, 3 — рамки #aaaaaa.
  // \colortbl обязан быть группой в скобках — без них MSFTEDIT не узнаёт
  // таблицу цветов и выводит её «;» текстом (проверено живым контролом).
  std::string out =
      "{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0\\fswiss Segoe UI;}{\\f1\\fmodern Consolas;}}"
      "{\\colortbl;\\red26\\green26\\blue26;\\red242\\green242\\blue242;\\red170\\green170\\blue170;}"
      "\\pard\\f0\\fs18\\cf1 ";
  for (auto& b : blocks) {
    if (b.kind == Block::kTable) {
      rtf_table(out, b.table, table_width);
      continue;
    }
    // \pard на каждом абзаце сбрасывает заливку/отступ предыдущего;
    // залитые строки (код-блоки, заголовки) — с небольшими отступами.
    out += b.shade ? "\\pard\\shading10000\\cbpat2\\li60\\ri60" : "\\pard";
    for (auto& r : b.runs) rtf_run(out, r);
    out += "\\par ";
  }
  return out + "}";
}

} // namespace lh::md
