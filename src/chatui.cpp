#include "chatui.h"

#include "config.h"
#include "mdrender.h"
#include "proc.h"
#include "supervisor.h"
#include "util.h"
#include "winstyle.h"

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <richedit.h>
#include <commdlg.h>
#include <commctrl.h>
#include <shellapi.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace lh::chatui {

namespace {

using nlohmann::json;

constexpr wchar_t CHAT_CLASS[] = L"llmhost_chat";

constexpr int IDC_LOG = 201;    // RichEdit: лента диалога
constexpr int IDC_INPUT = 202;  // ввод сообщения (внутри компоуз-карточки)
constexpr int IDC_SEND = 203;   // отправить / стоп (иконка)
constexpr int IDC_ATTACH = 205; // скрепка: приложить документ/изображение
constexpr int IDC_SAVE = 206;   // «Сохранить ответ» (тихая кнопка)
constexpr int IDC_CLEAR = 207;  // «Очистить» — стереть переписку и контекст
constexpr int IDC_MODEL = 208;  // выбор модели (режим роутера)
constexpr int IDC_THINK = 209;  // галочка «Мысли» (enable_thinking)

constexpr UINT WM_CHAT_CHUNK = WM_APP + 21; // LPARAM → new std::string (фрагмент ответа)
constexpr UINT WM_CHAT_DONE = WM_APP + 22;  // WPARAM 1 — успех; LPARAM → new std::string
constexpr UINT WM_CHAT_PHASE = WM_APP + 23; // WPARAM 0 — размышления, 1 — начался ответ

constexpr size_t MAX_HISTORY = 16;      // сообщений в контексте (без системного)
constexpr size_t MAX_ATTACHMENTS = 4;   // документов и картинок за раз
constexpr size_t MAX_FILE_CHARS = 384 * 1024; // ~ предел контекста на текстовый файл
constexpr size_t MAX_IMAGE_BYTES = 8 << 20;   // картинка — не больше 8 МБ

// Палитра компоуз-бара: белый фон, карточка с тонкой рамкой, отправка —
// тёмно-серая «пилюля». Всё строго в оттенках серого. Кнопки рисуем сами —
// классические «квадратно-гнездовые» сюда не вписываются.
constexpr COLORREF CLR_BG = RGB(0xff, 0xff, 0xff);
constexpr COLORREF CLR_CARD_BORDER = RGB(0xd8, 0xd8, 0xd8);
constexpr COLORREF CLR_CARD_FOCUS = RGB(0x7a, 0x7a, 0x7a);
constexpr COLORREF CLR_SEND = RGB(0x2e, 0x2e, 0x2e);
constexpr COLORREF CLR_SEND_HOVER = RGB(0x41, 0x41, 0x41);
constexpr COLORREF CLR_SEND_DOWN = RGB(0x1f, 0x1f, 0x1f);
constexpr COLORREF CLR_STOP = RGB(0x4a, 0x4a, 0x4a);
constexpr COLORREF CLR_STOP_HOVER = RGB(0x5a, 0x5a, 0x5a);
constexpr COLORREF CLR_HOVER_BG = RGB(0xed, 0xed, 0xed);
constexpr COLORREF CLR_ICON = RGB(0x4a, 0x4a, 0x4a);
constexpr COLORREF CLR_ICON_OFF = RGB(0xb4, 0xb4, 0xb4);
constexpr COLORREF CLR_TEXT = RGB(0x1a, 0x1a, 0x1a);
constexpr COLORREF CLR_TEXT_OFF = RGB(0x9a, 0x9a, 0x9a);

// Геометрия: лента сверху (растягивается), снизу компоуз-карточка
// (скрепка | ввод | отправка) и строка статуса. Всё пересчитывается в
// relayout() при изменении размера окна — границы у окна перетаскиваемые.
constexpr int MARGIN = 12;      // внешний отступ
constexpr int CARD_H = 78;      // высота компоуз-карточки
constexpr int CARD_RAD = 10;    // радиус скругления карточки
constexpr int ATTACH_S = 34;    // скрепка
constexpr int INPUT_H = 58;     // поле ввода
constexpr int SEND_S = 38;      // отправка/стоп
constexpr int SAVE_W = 100, SAVE_H = 28; // «Сохранить»
constexpr int MODEL_W = 190;    // выбор модели
constexpr int THINK_W = 78;     // галочка «Мысли» рядом с ним
constexpr int MIN_W = 520, MIN_H = 400;  // ниже — ряд контролов не помещается

HWND g_hwnd = nullptr, g_log = nullptr, g_input = nullptr;
std::atomic<bool> g_streaming{false}, g_cancel{false};
bool g_input_focus = false;

// История: content — строка ИЛИ массив (текст + картинки для vision-моделей).
std::vector<std::pair<std::string, json>> g_history;
std::string g_pending;      // ответ, набираемый стримом
json g_pending_user;        // вопрос этого обмена: в историю коммитится парой с ответом
int g_answer_start = 0;     // позиция в ленте, где начинается ответ модели
int g_status_cp = -1;       // статусная строка В ЛЕНТЕ: где начинается, -1 — нет строки

// Живой статус генерации («не висит ли?»): фазы ждём → размышляю → отвечаю.
// Фазы приходят из pump_sse сообщением WM_CHAT_PHASE, раз в секунду их
// перерисовывает WM_TIMER(2); в фазе ответа считаем ток/с по пришедшим
// фрагментам. g_phase_t0 — момент начала текущей фазы (совпадает с тем,
// как потом считается финальное «размышлял N с · ответ X ток/с»).
int g_live_phase = -1; // -1 ждём первый токен, 0 размышления, 1 ответ
int g_live_pieces = 0;
std::chrono::steady_clock::time_point g_live_start{}, g_phase_t0{};

// Ссылки последнего(их) ответа(ов): диапазон в ленте → адрес. Диапазоны до
// g_answer_start не трогаются при вставке нового ответа, поэтому копятся.
struct LinkRange { int from, to; std::wstring url; };
std::vector<LinkRange> g_links;

HWND g_tip = nullptr;      // tracking-тултип с адресом ссылки
bool g_tip_shown = false;
bool g_log_hover = false;  // TrackMouseEvent запрошен

// Ссылка под символом cp (или nullptr).
const wchar_t* link_at(int cp) {
  for (auto& L : g_links)
    if (cp >= L.from && cp < L.to) return L.url.c_str();
  return nullptr;
}

void tip_show(HWND h, const wchar_t* url, const POINT& client_pt) {
  TOOLINFOW ti{};
  ti.cbSize = sizeof(ti);
  ti.hwnd = h;
  ti.uId = 1;
  ti.lpszText = (LPWSTR)url;
  SendMessageW(g_tip, TTM_UPDATETIPTEXTW, 0, (LPARAM)&ti);
  POINT sp = client_pt;
  ClientToScreen(h, &sp);
  SendMessageW(g_tip, TTM_TRACKPOSITION, 0, MAKELPARAM(sp.x + 14, sp.y + 22));
  SendMessageW(g_tip, TTM_TRACKACTIVATE, TRUE, (LPARAM)&ti);
  g_tip_shown = true;
}

void tip_hide() {
  if (!g_tip_shown) return;
  TOOLINFOW ti{};
  ti.cbSize = sizeof(ti);
  ti.hwnd = g_log;
  ti.uId = 1;
  SendMessageW(g_tip, TTM_TRACKACTIVATE, FALSE, (LPARAM)&ti);
  g_tip_shown = false;
}

void open_link(const std::wstring& url) {
  // Только http/https: file://, command: и прочие схемы наружу не пускаем.
  if (url.rfind(L"http://", 0) != 0 && url.rfind(L"https://", 0) != 0) return;
  HINSTANCE h = ShellExecuteW(g_hwnd, L"open", url.c_str(), nullptr, nullptr,
                              SW_SHOWNORMAL);
  logf("chat: клик по ссылке %s → %ld", wide_to_utf8(url).c_str(), (long)(INT_PTR)h);
}

// Клик и ховер обрабатывает сабкласс ленты: MSFTEDIT капризничает с EN_LINK
// (у пользователя клики не доходили), а EM_CHARFROMPOS + карта g_links
// детерминированы. EN_LINK оставлен только ради курсора-руки и на лог.
// Контекстные меню ввода и ленты: свои, русские, с честной доступностью
// пунктов (у msftedit штатное меню английское — в log_subclass мы его гасим).
enum { IDM_C_UNDO = 3101, IDM_C_CUT, IDM_C_COPY, IDM_C_PASTE, IDM_C_DEL, IDM_C_ALL };

void show_edit_menu(HWND parent, HWND target, int x, int y) {
  bool is_log = target == g_log;
  DWORD a = 0, b = 0;
  SendMessageW(target, EM_GETSEL, (WPARAM)&a, (LPARAM)&b);
  bool has_sel = b > a;
  bool can_undo = SendMessageW(target, EM_CANUNDO, 0, 0) != 0;
  bool can_paste = IsClipboardFormatAvailable(CF_UNICODETEXT) != 0;

  HMENU m = CreatePopupMenu();
  if (is_log) {
    // Лента read-only: только копирование и выделение.
    AppendMenuW(m, MF_STRING | (has_sel ? 0 : MF_GRAYED), IDM_C_COPY, L"Копировать");
    AppendMenuW(m, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(m, MF_STRING, IDM_C_ALL, L"Выделить всё");
  } else {
    AppendMenuW(m, MF_STRING | (can_undo ? 0 : MF_GRAYED), IDM_C_UNDO, L"Отменить");
    AppendMenuW(m, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(m, MF_STRING | (has_sel ? 0 : MF_GRAYED), IDM_C_CUT, L"Вырезать");
    AppendMenuW(m, MF_STRING | (has_sel ? 0 : MF_GRAYED), IDM_C_COPY, L"Копировать");
    AppendMenuW(m, MF_STRING | (can_paste ? 0 : MF_GRAYED), IDM_C_PASTE, L"Вставить");
    AppendMenuW(m, MF_STRING | (has_sel ? 0 : MF_GRAYED), IDM_C_DEL, L"Удалить");
    AppendMenuW(m, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(m, MF_STRING, IDM_C_ALL, L"Выделить всё");
  }
  SetForegroundWindow(parent); // иначе меню не закрывается по клику мимо
  int id = TrackPopupMenu(m, TPM_RIGHTBUTTON | TPM_RETURNCMD | TPM_NONOTIFY,
                          x, y, 0, parent, nullptr);
  DestroyMenu(m);
  switch (id) {
    case IDM_C_UNDO: SendMessageW(target, EM_UNDO, 0, 0); break;
    case IDM_C_CUT: SendMessageW(target, WM_CUT, 0, 0); break;
    case IDM_C_COPY: SendMessageW(target, WM_COPY, 0, 0); break;
    case IDM_C_PASTE: SendMessageW(target, WM_PASTE, 0, 0); break;
    case IDM_C_DEL: SendMessageW(target, WM_CLEAR, 0, 0); break;
    case IDM_C_ALL:
      SendMessageW(target, EM_SETSEL, 0, -1);
      SendMessageW(target, EM_SCROLLCARET, 0, 0);
      break;
  }
}

// Координаты WM_CONTEXTMENU: экран при мыши, курсор при Shift+F10/Apps.
POINT ctx_menu_point(LPARAM lp) {
  POINT pt{(short)LOWORD(lp), (short)HIWORD(lp)};
  if (lp == -1) GetCursorPos(&pt);
  return pt;
}

LRESULT CALLBACK log_subclass(HWND h, UINT msg, WPARAM wp, LPARAM lp,
                              UINT_PTR, DWORD_PTR) {
  switch (msg) {
    case WM_CONTEXTMENU: {
      // Гасим английское меню msftedit, показываем своё.
      POINT pt = ctx_menu_point(lp);
      show_edit_menu(g_hwnd, h, pt.x, pt.y);
      return 0;
    }
    case WM_MOUSEMOVE: {
      if (!g_log_hover) {
        g_log_hover = true;
        TRACKMOUSEEVENT tme{sizeof(tme), TME_LEAVE, h, 0};
        TrackMouseEvent(&tme);
      }
      POINT pt{(short)LOWORD(lp), (short)HIWORD(lp)};
      LRESULT hit = SendMessageW(h, EM_CHARFROMPOS, 0, (LPARAM)&pt);
      const wchar_t* url = link_at((int)(short)LOWORD(hit));
      if (url) tip_show(h, url, pt);
      else tip_hide();
      break;
    }
    case WM_MOUSELEAVE:
      g_log_hover = false;
      tip_hide();
      break;
    case WM_LBUTTONUP: {
      POINT pt{(short)LOWORD(lp), (short)HIWORD(lp)};
      LRESULT hit = SendMessageW(h, EM_CHARFROMPOS, 0, (LPARAM)&pt);
      int cp = (int)(short)LOWORD(hit);
      for (auto& L : g_links) {
        if (cp >= L.from && cp < L.to) {
          open_link(L.url);
          break;
        }
      }
      break;
    }
  }
  return DefSubclassProc(h, msg, wp, lp);
}

struct Attachment {
  std::string name;
  std::string content; // текст как есть; картинка — сырые байты
  bool image = false;
  std::string mime;    // для картинок: image/png и т.п.
};
std::vector<Attachment> g_attachments; // до следующей отправки

struct Geom {
  RECT log{}, card{}, attach{}, input{}, send{}, save{}, clear{}, model{}, think{};
  int bottom_y = 0; // верх карточки
};
Geom g_geo;

// Пересборка лейаута под текущий размер: лента забирает всё свободное
// место, карточка прижата к низу, скрепка слева и отправка справа в ней.
// Выбор модели (роутер): выпадающий список + снимок id'ов в порядке комбобокса.
HWND g_model_combo = nullptr;
std::vector<std::string> g_model_ids;
std::string g_chat_pick;      // явный выбор пользователя; пусто — решит прокси
std::string g_models_sig;     // сигнатура набора: не пересобирать список зря

// Список моделей: только разговаривающие (эмбеддеры чату ни к чему). Пересобираем
// когда меняется набор или их состояния; выбор пользователя сохраняем, если жив.
// Скрытое окно не опрашиваем: таймер тикает всегда, а здесь сетевой запрос.
void refresh_models(bool force) {
  if (!g_model_combo || !g_hwnd || !IsWindowVisible(g_hwnd)) return;
  auto ms = sup::chat_models();
  // Эмбеддеры чату не собеседники; наверху — то, что уже в памяти.
  std::vector<sup::ChatModel> rows;
  for (auto& m : ms)
    if (!m.embedding) rows.push_back(m);
  std::stable_sort(rows.begin(), rows.end(),
                   [](const sup::ChatModel& a, const sup::ChatModel& b) {
                     auto rk = [](const std::string& s) {
                       return s == "loaded" ? 0 : s == "loading" ? 1 : 2;
                     };
                     if (rk(a.state) != rk(b.state)) return rk(a.state) < rk(b.state);
                     return a.id < b.id;
                   });
  std::string sig;
  for (auto& m : rows) sig += m.id + ":" + m.state + ";";
  if (!force && sig == g_models_sig) return;
  g_models_sig = sig;
  SendMessageW(g_model_combo, CB_RESETCONTENT, 0, 0);
  g_model_ids.clear();
  int sel = -1, first_loaded = -1;
  for (auto& m : rows) {
    std::wstring label = utf8_to_wide(m.id);
    if (m.state == "loaded") {
      label += L"  (в памяти)";
      if (first_loaded < 0) first_loaded = (int)g_model_ids.size();
    } else if (m.state == "loading") {
      label += L"  (грузится…)";
    }
    int idx = (int)SendMessageW(g_model_combo, CB_ADDSTRING, 0, (LPARAM)label.c_str());
    g_model_ids.push_back(m.id);
    if (m.id == g_chat_pick) sel = idx;
  }
  if (sel < 0 && !g_model_ids.empty()) {
    // Умолчание супервизора: последняя загруженная (или единственная).
    std::string def = sup::chat_model();
    for (size_t k = 0; k < g_model_ids.size() && sel < 0; ++k)
      if (g_model_ids[k] == def) sel = (int)k;
    // Ни primary, ни совпадений — первая загруженная; совсем ничего
    // загруженного — первая строка становится ЯВНОЙ целью: прокси без
    // загруженных подставить нечего (пустой «model» = 400 у роутера).
    if (sel < 0 && first_loaded >= 0) sel = first_loaded;
    if (sel < 0) {
      sel = 0;
      g_chat_pick = g_model_ids[0];
    } else {
      g_chat_pick.clear(); // умолчание супервизора отвечает прокси
    }
  }
  if (sel >= 0) SendMessageW(g_model_combo, CB_SETCURSEL, sel, 0);
}

void relayout(HWND hwnd) {
  RECT rc{};
  GetClientRect(hwnd, &rc);
  int W = rc.right, H = rc.bottom;
  if (W < 200 || H < 200) return;

  Geom& g = g_geo;
  g.bottom_y = H - MARGIN - CARD_H - (SAVE_H + 12);
  g.log = {MARGIN, MARGIN, W - MARGIN, g.bottom_y - 8};
  g.card = {MARGIN, g.bottom_y, W - MARGIN, g.bottom_y + CARD_H};
  g.attach = {MARGIN + 14, g.bottom_y + (CARD_H - ATTACH_S) / 2,
              MARGIN + 14 + ATTACH_S, g.bottom_y + (CARD_H + ATTACH_S) / 2};
  int send_x = W - MARGIN - 14 - SEND_S;
  g.send = {send_x, g.bottom_y + (CARD_H - SEND_S) / 2,
            send_x + SEND_S, g.bottom_y + (CARD_H + SEND_S) / 2};
  g.input = {g.attach.right + 10, g.bottom_y + (CARD_H - INPUT_H) / 2,
             send_x - 10, g.bottom_y + (CARD_H + INPUT_H) / 2};
  g.save = {W - MARGIN - SAVE_W, g.card.bottom + 4, W - MARGIN,
            g.card.bottom + 4 + SAVE_H};
  // «Очистить» — слева от «Сохранить», одинакового с ним размера.
  g.clear = {W - MARGIN - SAVE_W * 2 - 8, g.card.bottom + 4,
             W - MARGIN - SAVE_W - 8, g.card.bottom + 4 + SAVE_H};
  // Выбор модели — в нижнем ряду слева, за ним «Мысли»; статусная строка
  // переехала в ленту (RichEdit), ряд стал просторным.
  g.model = {MARGIN, g.card.bottom + 4, MARGIN + MODEL_W, g.card.bottom + 4 + SAVE_H};
  g.think = {MARGIN + MODEL_W + 8, g.card.bottom + 4 + (SAVE_H - 20) / 2,
             MARGIN + MODEL_W + 8 + THINK_W, g.card.bottom + 4 + (SAVE_H + 20) / 2};

  auto place = [](HWND h, const RECT& r) {
    if (h) MoveWindow(h, r.left, r.top, r.right - r.left, r.bottom - r.top, TRUE);
  };
  place(g_log, g.log);
  place(g_input, g.input);
  place(GetDlgItem(hwnd, IDC_ATTACH), g.attach);
  place(GetDlgItem(hwnd, IDC_SEND), g.send);
  place(GetDlgItem(hwnd, IDC_MODEL), g.model);
  place(GetDlgItem(hwnd, IDC_THINK), g.think);
  place(GetDlgItem(hwnd, IDC_SAVE), g.save);
  place(GetDlgItem(hwnd, IDC_CLEAR), g.clear);
  InvalidateRect(hwnd, nullptr, TRUE);
}

std::wstring utf8_display(const std::string& s) { return utf8_to_wide(s); }

// Системный промпт: ориентация модели во времени — без неё «сегодня» и
// «который час» ей не понять. Идея подсмотрена в claw2, код не копировался.
std::string system_prompt() {
  static const wchar_t* const days[] = {L"воскресенье", L"понедельник", L"вторник",
                                        L"среда",       L"четверг",     L"пятница",
                                        L"суббота"};
  static const wchar_t* const months[] = {L"января",   L"февраля", L"марта",  L"апреля",
                                          L"мая",      L"июня",    L"июля",   L"августа",
                                          L"сентября", L"октября", L"ноября", L"декабря"};
  SYSTEMTIME st{};
  GetLocalTime(&st);
  wchar_t buf[128];
  swprintf(buf, 128, L"%s, %d %s %d г., %02d:%02d (местное время)", days[st.wDayOfWeek],
           st.wDay, months[st.wMonth - 1], st.wYear, st.wHour, st.wMinute);
  return "Ты — локальный ассистент llmhost, работаешь на компьютере пользователя. "
         "Отвечай на языке пользователя, кратко и по делу. Сегодня " +
         wide_to_utf8(buf) + ".";
}

// ---- рисовка -------------------------------------------------------------------

HFONT icon_font(int px) {
  static HFONT f16 = CreateFontW(-16, 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET,
                                 0, 0, CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE,
                                 L"Segoe MDL2 Assets");
  static HFONT f14 = CreateFontW(-14, 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET,
                                 0, 0, CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE,
                                 L"Segoe MDL2 Assets");
  return px >= 16 ? f16 : f14;
}

void round_rect(HDC dc, const RECT& r, int rad, COLORREF border, int border_w, COLORREF fill) {
  HBRUSH br = CreateSolidBrush(fill);
  HPEN pen = CreatePen(PS_SOLID, border_w, border);
  HGDIOBJ old_br = SelectObject(dc, br), old_pen = SelectObject(dc, pen);
  RoundRect(dc, r.left, r.top, r.right, r.bottom, rad * 2, rad * 2);
  SelectObject(dc, old_br);
  SelectObject(dc, old_pen);
  DeleteObject(br);
  DeleteObject(pen);
}

void draw_glyph(HDC dc, const RECT& r, const wchar_t* glyph, HFONT font, COLORREF color) {
  // DrawText с DT_VCENTER: у TextOut вертикального центрирования нет
  // (VTA_CENTER на него не действует — глиф висел на базовой линии).
  HGDIOBJ old_font = SelectObject(dc, font);
  int old_bk = SetBkMode(dc, TRANSPARENT);
  COLORREF old_col = SetTextColor(dc, color);
  RECT rc = r;
  DrawTextW(dc, glyph, -1, &rc, DT_CENTER | DT_VCENTER | DT_SINGLELINE | DT_NOPREFIX);
  SetTextColor(dc, old_col);
  SetBkMode(dc, old_bk);
  SelectObject(dc, old_font);
}

// Иконки Segoe MDL2 Assets (в Win10 есть всегда): E723 скрепка, E724 отправка,
// E71A стоп, E74E сохранение.
constexpr wchar_t GLYPH_ATTACH[] = L"\uE723";
constexpr wchar_t GLYPH_SEND[] = L"\uE724";
constexpr wchar_t GLYPH_STOP[] = L"\uE71A";
constexpr wchar_t GLYPH_SAVE[] = L"\uE74E";
constexpr wchar_t GLYPH_CLEAR[] = L"\uE74D"; // корзина

// Отправка/стоп — единственная «спец»-кнопка (тёмная пилюля): свой ховер
// и нажатие через сабкласс. Скрепка и «Сохранить» — общие ws::button.
bool g_send_hover = false, g_send_press = false;

LRESULT CALLBACK send_subclass(HWND h, UINT msg, WPARAM wp, LPARAM lp, UINT_PTR, DWORD_PTR) {
  switch (msg) {
    case WM_MOUSEMOVE:
      if (!g_send_hover) {
        g_send_hover = true;
        TRACKMOUSEEVENT tme{sizeof(tme), TME_LEAVE, h, 0};
        TrackMouseEvent(&tme);
        InvalidateRect(h, nullptr, TRUE);
      }
      break;
    case WM_MOUSELEAVE:
      g_send_hover = false;
      g_send_press = false;
      InvalidateRect(h, nullptr, TRUE);
      break;
    case WM_LBUTTONDOWN:
      g_send_press = true;
      InvalidateRect(h, nullptr, TRUE);
      break;
    case WM_LBUTTONUP:
      g_send_press = false;
      InvalidateRect(h, nullptr, TRUE);
      break;
  }
  return DefSubclassProc(h, msg, wp, lp);
}

HWND add_send_button(int x, int y, int w, int h) {
  HWND b = CreateWindowExW(0, L"BUTTON", nullptr,
                           WS_CHILD | WS_VISIBLE | WS_TABSTOP | BS_OWNERDRAW |
                               BS_DEFPUSHBUTTON,
                           x, y, w, h, g_hwnd, (HMENU)(INT_PTR)IDC_SEND, nullptr, nullptr);
  SetWindowSubclass(b, send_subclass, 1, 0);
  return b;
}

void draw_send(const DRAWITEMSTRUCT* dis) {
  HDC dc = dis->hDC;
  RECT r = dis->rcItem;
  // Тёмно-серая «пилюля»; во время генерации — стоп (тот же серый,
  // но иконка-квадрат — перепутать нельзя).
  bool stop = g_streaming;
  COLORREF fill = stop ? CLR_STOP : CLR_SEND;
  if (g_send_press) fill = CLR_SEND_DOWN;
  else if (g_send_hover) fill = stop ? CLR_STOP_HOVER : CLR_SEND_HOVER;
  round_rect(dc, r, 9, fill, 1, fill);
  draw_glyph(dc, r, stop ? GLYPH_STOP : GLYPH_SEND, icon_font(16), RGB(255, 255, 255));
}

int log_end(); // честный конец текста ленты (реализация ниже, у RichEdit-хелперов)

// Автопрокрутка ленты: следует за новым текстом, только если читатель стоит
// внизу; прокрутил вверх почитать — не дёргаем до конца ответа.
bool feed_at_bottom() {
  if (!g_log) return true;
  SCROLLINFO si{};
  si.cbSize = sizeof(si);
  si.fMask = SIF_ALL;
  if (!GetScrollInfo(g_log, SB_VERT, &si)) return true;
  if (si.nMax <= 0 || si.nPage <= 0) return true; // прокрутки нет — всегда «внизу»
  return si.nPos + (int)si.nPage + 2 >= si.nMax;  // нижняя строка на экране
}

void feed_scroll_bottom() {
  if (g_log) SendMessageW(g_log, EM_SCROLL, SB_BOTTOM, 0);
}

// Статус генерации живёт В ЛЕНТЕ — серым курсивом в самом конце
// («· размышляю… 12 с»), как договорились: места в нижнем ряду больше нет.
// g_status_cp — где строка начинается, -1 — строки ещё нет; пустой текст её убирает.
void set_status(const std::wstring& text) {
  if (!g_hwnd || !g_log) return;
  bool follow = feed_at_bottom(); // решаем ДО перерисовки: стоял внизу — едем вниз
  if (g_status_cp < 0) {
    if (text.empty()) return;
    g_status_cp = log_end();
  }
  SendMessageW(g_log, EM_SETREADONLY, FALSE, 0);
  CHARRANGE cr{g_status_cp, log_end()};
  SendMessageW(g_log, EM_EXSETSEL, 0, (LPARAM)&cr);
  if (text.empty()) {
    SendMessageW(g_log, EM_REPLACESEL, FALSE, (LPARAM)L"");
    g_status_cp = -1;
  } else {
    std::wstring line = L"· " + text;
    SendMessageW(g_log, EM_REPLACESEL, FALSE, (LPARAM)line.c_str());
    CHARRANGE rng{g_status_cp, log_end()};
    SendMessageW(g_log, EM_EXSETSEL, 0, (LPARAM)&rng);
    CHARFORMAT2W cf{};
    cf.cbSize = sizeof(cf);
    cf.dwMask = CFM_COLOR | CFM_ITALIC | CFM_BOLD;
    cf.dwEffects = CFE_ITALIC;
    cf.crTextColor = CLR_TEXT_OFF;
    SendMessageW(g_log, EM_SETCHARFORMAT, SCF_SELECTION, (LPARAM)&cf);
    CHARRANGE end{log_end(), log_end()};
    SendMessageW(g_log, EM_EXSETSEL, 0, (LPARAM)&end);
    if (follow) feed_scroll_bottom();
  }
  SendMessageW(g_log, EM_SETREADONLY, TRUE, 0);
}

// Раз в секунду во время генерации (и сразу на смене фазы): секунды ждём/
// размышляю тикают, скорость ответа — по фактически пришедшим фрагментам.
void update_stream_status() {
  if (!g_hwnd || !g_streaming) return;
  auto now = std::chrono::steady_clock::now();
  wchar_t buf[96];
  if (g_live_phase == 0) {
    double s = std::chrono::duration<double>(now - g_phase_t0).count();
    if (s < 1.0) swprintf(buf, 96, L"размышляю… <1 с");
    else swprintf(buf, 96, L"размышляю… %.0f с", s);
  } else if (g_live_phase == 1) {
    double s = std::chrono::duration<double>(now - g_phase_t0).count();
    swprintf(buf, 96, L"отвечаю… %.1f ток/с", s > 0.2 ? g_live_pieces / s : 0.0);
  } else {
    swprintf(buf, 96, L"ждём ответа… %.0f с",
             std::chrono::duration<double>(now - g_live_start).count());
  }
  set_status(buf);
}

// Кнопки перерисовываются по смене режима (отправка ↔ стоп, доступность):
// скрепку и «Сохранить» на время генерации гасим честным EnableWindow —
// общий ws::draw_button разукрасит сам.
void refresh_buttons() {
  if (!g_hwnd) return;
  EnableWindow(GetDlgItem(g_hwnd, IDC_ATTACH), !g_streaming);
  EnableWindow(GetDlgItem(g_hwnd, IDC_SAVE), !g_streaming);
  EnableWindow(GetDlgItem(g_hwnd, IDC_CLEAR), !g_streaming);
  EnableWindow(GetDlgItem(g_hwnd, IDC_MODEL), !g_streaming); // модель не меняем посреди ответа
  EnableWindow(GetDlgItem(g_hwnd, IDC_THINK), !g_streaming); // и размышления тоже
  InvalidateRect(GetDlgItem(g_hwnd, IDC_SEND), nullptr, TRUE);
}

void refresh_attachment_status() {
  if (g_attachments.empty()) { set_status(L""); return; }
  std::wstring names;
  for (auto& a : g_attachments) {
    if (!names.empty()) names += L", ";
    names += utf8_display(a.name);
  }
  set_status(L"Прикреплено: " + names);
}

// Конец ленты в CP-координатах RichEdit. GetWindowTextLength не годится:
// он считает CRLF по два символа, а позиции CHARRANGE — по одному (абзац = 1),
// и с каждой строкой счётчик уползает вперёд — выделение начала резать
// середину ответа (задвоенные начала «ПожалПожалуйста…» — ровно это).
int log_end() {
  CHARRANGE cr{-1, -1}; // cpMax = -1 → в самый конец текста
  SendMessageW(g_log, EM_EXSETSEL, 0, (LPARAM)&cr);
  SendMessageW(g_log, EM_EXGETSEL, 0, (LPARAM)&cr);
  return (int)cr.cpMax;
}

void append_text(const std::wstring& text, bool bold) {
  // Решаем ДО вставки, как во всех путях ленты: стоял внизу — едем вниз.
  // EM_SCROLLCARET тут не годится: у RichEdit он молчит без фокуса, а фокус
  // при отправке живёт во вводе — «Вы:» оставалась за нижней кромкой, и
  // feed_at_bottom() дальше всю генерацию отвечал false.
  bool follow = feed_at_bottom();
  // ES_READONLY блокирует и EM_REPLACESEL — на время вставки снимаем.
  SendMessageW(g_log, EM_SETREADONLY, FALSE, 0);
  CHARRANGE cr{log_end(), log_end()};
  SendMessageW(g_log, EM_EXSETSEL, 0, (LPARAM)&cr);
  // Формат задаём ПОЛНОСТЬЮ, а не только жирность: RichEdit наследует формат
  // в точке вставки — после серо-курсивной статусной строки «Вы:» уезжал бы
  // серым курсивом.
  CHARFORMAT2W cf{};
  cf.cbSize = sizeof(cf);
  cf.dwMask = CFM_BOLD | CFM_ITALIC | CFM_COLOR;
  cf.dwEffects = bold ? CFE_BOLD : 0;
  cf.crTextColor = CLR_TEXT;
  SendMessageW(g_log, EM_SETCHARFORMAT, SCF_SELECTION, (LPARAM)&cf);
  SendMessageW(g_log, EM_REPLACESEL, FALSE, (LPARAM)text.c_str());
  if (follow) feed_scroll_bottom();
  SendMessageW(g_log, EM_SETREADONLY, TRUE, 0);
}

// Подсказка сверху ленты — она же возвращается после «Очистить».
void append_welcome() {
  append_text(L"Модель — выбранная в трее; сон её не помешает: запрос разбудит.\n"
              L"Вопрос — и Enter. Скрепка приложит документ или изображение "
              L"(для vision-моделей), «Сохранить» запишет последний ответ, "
              L"«Очистить» сотрёт переписку. Галочка «Мысли» включает и выключает "
              L"размышления рассуждающих моделей.\n\n",
              false);
}

// ---- вложения ----------------------------------------------------------------

bool image_mime_of(const std::wstring& ext, std::string& mime) {
  if (ext == L"png") { mime = "image/png"; return true; }
  if (ext == L"jpg" || ext == L"jpeg") { mime = "image/jpeg"; return true; }
  if (ext == L"webp") { mime = "image/webp"; return true; }
  if (ext == L"gif") { mime = "image/gif"; return true; }
  if (ext == L"bmp") { mime = "image/bmp"; return true; }
  return false;
}

void attach_file() {
  wchar_t buf[1024] = {};
  OPENFILENAMEW of{};
  of.lStructSize = sizeof(of);
  of.hwndOwner = g_hwnd;
  of.lpstrFilter = L"Изображения и тексты\0*.png;*.jpg;*.jpeg;*.webp;*.gif;*.bmp;"
                   L"*.txt;*.md;*.csv;*.json;*.log;*.xml;*.yaml;*.yml;*.ini;*.cfg;"
                   L"*.py;*.js;*.ts;*.c;*.cpp;*.h;*.java;*.sh;*.bat;*.html;*.css\0"
                   L"Изображения (для vision-моделей)\0*.png;*.jpg;*.jpeg;*.webp;*.gif;*.bmp\0"
                   L"Все файлы\0*.*\0";
  of.lpstrFile = buf;
  of.nMaxFile = 1024;
  of.Flags = OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;
  of.lpstrTitle = L"Приложить документ или изображение";
  if (!GetOpenFileNameW(&of)) return;

  std::wstring path(buf);
  std::wstring ext;
  size_t dot = path.find_last_of(L'.');
  if (dot != std::wstring::npos) ext = path.substr(dot + 1);
  for (auto& c : ext) c = (wchar_t)towlower(c);
  Attachment att;
  att.image = image_mime_of(ext, att.mime);

  if (lh::file_size(path) > MAX_IMAGE_BYTES) {
    set_status(L"файл больше 8 МБ — слишком велик для контекста");
    return;
  }
  // Читаем сами: текст проверяем на бинарность по первым 64 КБ, картинку
  // берём как есть (байты уйдут в base64 data-URI).
  FILE* f = _wfopen(path.c_str(), L"rb");
  if (!f) { set_status(L"файл не открывается"); return; }
  char head[65536];
  size_t got = fread(head, 1, sizeof(head), f);
  if (!att.image && memchr(head, 0, got)) {
    fclose(f);
    set_status(L"похоже на бинарный файл — анализировать можно только текст");
    return;
  }
  att.content.assign(head, got);
  char chunk[65536];
  while (att.content.size() < MAX_FILE_CHARS && (got = fread(chunk, 1, sizeof(chunk), f)) > 0)
    att.content.append(chunk, got);
  fclose(f);
  if (att.content.empty()) { set_status(L"файл пуст или не читается"); return; }
  if (!att.image && att.content.size() >= MAX_FILE_CHARS) {
    att.content.resize(MAX_FILE_CHARS);
    att.content += "\n… [файл обрезан до предела контекста]";
  }
  std::wstring file = path;
  size_t slash = file.find_last_of(L"\\/");
  if (slash != std::wstring::npos) file = file.substr(slash + 1);
  att.name = wide_to_utf8(file);
  if (g_attachments.size() >= MAX_ATTACHMENTS) {
    set_status(L"максимум " + std::to_wstring(MAX_ATTACHMENTS) + L" файлов за раз");
    return;
  }
  g_attachments.push_back(std::move(att));
  refresh_attachment_status();
}

// ---- рендер Markdown ---------------------------------------------------------

using namespace lh::md;

// Ширина ленты в twips: таблицам растянуться на окно с учётом DPI.
int log_width_twips() {
  RECT rc{};
  GetClientRect(g_log, &rc);
  HDC dc = GetDC(g_log);
  int dpi = dc ? GetDeviceCaps(dc, LOGPIXELSX) : 96;
  if (dc) ReleaseDC(g_log, dc);
  if (dpi <= 0) dpi = 96;
  int twips = (int)((long long)rc.right * 1440 / dpi);
  return (std::max)(twips - 200, 1440); // маленький отступ от краёв
}

DWORD CALLBACK rtf_in_cb(DWORD_PTR cookie, LPBYTE buf, LONG cb, LONG* moved) {
  auto* s = (std::string*)cookie;
  // (std::min) — скобки против min-макроса из windows.h (смоук без NOMINMAX).
  LONG n = cb > 0 ? (LONG)(std::min<size_t>)((size_t)cb, s->size()) : 0;
  if (n > 0) {
    memcpy(buf, s->data(), (size_t)n);
    s->erase(0, (size_t)n);
  }
  *moved = n;
  return 0;
}

// Фон инлайн-кода: RTF-ридер MSFTEDIT не рисует \highlight/\cb (проверено
// пиксельной пробой живого контрола), поэтому кодовые runs после стрима
// красим сами — EM_SETCHARFORMAT фон рисует всегда. Ищем фрагменты по
// порядку следования в документе, код-блоки уже закрыты абзацной заливкой.
void paint_code_bg(const std::vector<Block>& blocks) {
  int end = log_end();
  int from = g_answer_start;
  int painted = 0;
  for (auto& b : blocks) {
    if (b.kind != Block::kText || b.shade) continue;
    for (auto& r : b.runs) {
      if (!(r.style & MD_CODE) || r.text.empty()) continue;
      FINDTEXTW ft{};
      ft.chrg.cpMin = from;
      ft.chrg.cpMax = end;
      ft.lpstrText = (LPWSTR)r.text.c_str();
      LRESULT pos = SendMessageW(g_log, EM_FINDTEXT, FR_DOWN | FR_MATCHCASE, (LPARAM)&ft);
      if (pos < 0) return; // разошлись с текстом — дальше бессмысленно
      CHARRANGE cr{(LONG)pos, (LONG)(pos + (LONG)r.text.size())};
      SendMessageW(g_log, EM_EXSETSEL, 0, (LPARAM)&cr);
      CHARFORMAT2W cf{};
      cf.cbSize = sizeof(cf);
      cf.dwMask = CFM_BACKCOLOR;
      cf.crBackColor = RGB(0xf2, 0xf2, 0xf2);
      SendMessageW(g_log, EM_SETCHARFORMAT, SCF_SELECTION, (LPARAM)&cf);
      from = cr.cpMax;
      ++painted;
    }
  }
  if (painted) logf("chat: фон инлайн-кода — %d фрагментов", painted);
}

// Ссылки: RTF несёт только \ul (вид), кликабельность CFE_LINK и карта
// «диапазон → адрес» ставятся после стрима — EN_LINK отдаёт лишь диапазон,
// без своей карты адрес не достать. Ищем подписи по порядку следования.
void paint_links(const std::vector<Block>& blocks) {
  int end = log_end();
  int from = g_answer_start;
  int painted = 0;
  auto handle = [&](const Run& r) {
    if (!(r.style & MD_LINK) || r.url.empty()) return;
    FINDTEXTW ft{};
    ft.chrg.cpMin = from;
    ft.chrg.cpMax = end;
    ft.lpstrText = (LPWSTR)r.text.c_str();
    LRESULT pos = SendMessageW(g_log, EM_FINDTEXT, FR_DOWN | FR_MATCHCASE, (LPARAM)&ft);
    if (pos < 0) {
      // Разошлись с текстом — ссылка не кликабельна; узнаем почему из лога.
      logf("chat: ссылка НЕ найдена в ленте: '%s' → %s",
           wide_to_utf8(r.text).c_str(), wide_to_utf8(r.url).c_str());
      return;
    }
    CHARRANGE cr{(LONG)pos, (LONG)(pos + (LONG)r.text.size())};
    CHARFORMAT2W cf{};
    cf.cbSize = sizeof(cf);
    cf.dwMask = CFM_LINK | CFM_COLOR | CFM_UNDERLINE;
    cf.dwEffects = CFE_LINK | CFE_UNDERLINE;
    cf.crTextColor = RGB(0x4a, 0x4a, 0x4a); // серый — синий в тему запрещён
    SendMessageW(g_log, EM_SETCHARFORMAT, SCF_SELECTION, (LPARAM)&cf);
    g_links.push_back({(int)pos, (int)cr.cpMax, r.url});
    from = (int)cr.cpMax;
    ++painted;
  };
  for (auto& b : blocks) {
    if (b.kind == Block::kText) {
      for (auto& r : b.runs) handle(r);
    } else {
      for (auto& row : b.table.rows)
        for (auto& cell : row)
          for (auto& r : cell) handle(r);
    }
  }
  if (painted) logf("chat: ссылок размечено — %d", painted);
}

// Готовый ответ: Markdown → RTF одним куском, STREAMIN вливает его вместо
// выделенного сырого текста. Так таблицы попадают в RichEdit настоящими
// строками RTF — с рамками, заливкой шапки и выравниванием колонок.
void insert_markdown(const std::string& md_utf8) {
  std::vector<Block> blocks = md::render(utf8_to_wide(md_utf8));
  std::string rtf = md::to_rtf(blocks, log_width_twips());
  logf("chat: ответ — %zu блоков → rtf %zu байт", blocks.size(), rtf.size());
  bool follow = feed_at_bottom(); // итоговый рендер не уводит читающего вверх

  SendMessageW(g_log, EM_SETREADONLY, FALSE, 0);
  CHARRANGE cr{g_answer_start, log_end()};
  SendMessageW(g_log, EM_EXSETSEL, 0, (LPARAM)&cr);
  EDITSTREAM es{};
  es.dwCookie = (DWORD_PTR)&rtf;
  es.pfnCallback = rtf_in_cb;
  // Возвращаемое значение EM_STREAMIN у MSFTEDIT — не число байт (проверено
  // живым контролом: полный текст при «49 из 1089»), поэтому верим только dwError.
  SendMessageW(g_log, EM_STREAMIN, SF_RTF | SFF_SELECTION, (LPARAM)&es);
  if (es.dwError != 0) {
    // Провал стрима MSFTEDIT компенсирует обрезкой до начала выделения —
    // вернём на место хотя бы сырой ответ, чтобы он не пропал.
    logf("chat: STREAMIN не прошёл (err=%lu), возвращаю сырой текст",
         (unsigned long)es.dwError);
    CHARRANGE tail{log_end(), log_end()};
    SendMessageW(g_log, EM_EXSETSEL, 0, (LPARAM)&tail);
    std::wstring raw = utf8_display(md_utf8);
    SendMessageW(g_log, EM_REPLACESEL, FALSE, (LPARAM)raw.c_str());
  } else {
    paint_code_bg(blocks);
    paint_links(blocks);
  }
  int fin = log_end();
  CHARRANGE end{fin, fin};
  SendMessageW(g_log, EM_EXSETSEL, 0, (LPARAM)&end);
  if (follow) feed_scroll_bottom();
  SendMessageW(g_log, EM_SETREADONLY, TRUE, 0);
}

// ---- рабочий поток: HTTP-стрим к собственному прокси -------------------------

// Фазы потока: размышления (reasoning_content — b11223 выносит их из content,
// в ленту они не идут) и ответ считаем раздельно, иначе у reasoning-моделей
// один общий «ток/с» в разы ниже реальной скорости генерации.
struct StreamStats {
  int pieces = 0;        // фрагменты ответа
  int reason_pieces = 0; // фрагменты размышлений
  int phase_sent = -1;   // фаза, о которой UI уже извещён (WM_CHAT_PHASE)
  std::string finish;    // finish_reason финального чанка (stop/length/…)
  std::chrono::steady_clock::time_point first_reason{}; // первый фрагмент размышлений
  std::chrono::steady_clock::time_point first_answer{}; // первый фрагмент ответа
  std::chrono::steady_clock::time_point last_chunk{};   // последний фрагмент любого типа
};

// Разбор SSE-потока llama-server: строки «data: {…}», финал «data: [DONE]».
// [DONE] не обрывает приём — иначе httplib сочтёт ответ отменённым; false —
// только явный «Стоп».
bool pump_sse(const std::string& data, std::string& pending, StreamStats& st) {
  pending += data;
  size_t pos;
  while ((pos = pending.find('\n')) != std::string::npos) {
    std::string line = pending.substr(0, pos);
    pending.erase(0, pos + 1);
    if (line.rfind("data: ", 0) != 0) continue;
    std::string payload = line.substr(6);
    if (payload == "[DONE]") { pending.clear(); return true; }
    try {
      auto j = json::parse(payload);
      auto& choices = j.at("choices");
      if (choices.is_array() && !choices.empty()) {
        auto& ch = choices[0];
        if (ch.contains("finish_reason") && !ch["finish_reason"].is_null())
          st.finish = ch["finish_reason"].get<std::string>();
        if (!ch.contains("delta")) continue;
        auto& delta = ch["delta"];
        auto now = std::chrono::steady_clock::now();
        if (delta.contains("reasoning_content")) {
          std::string piece = delta.value("reasoning_content", "");
          if (!piece.empty()) {
            if (!st.reason_pieces) {
              st.first_reason = now;
              if (st.phase_sent < 0) {
                st.phase_sent = 0;
                PostMessageW(g_hwnd, WM_CHAT_PHASE, 0, 0);
              }
            }
            ++st.reason_pieces;
            st.last_chunk = now;
          }
        }
        if (delta.contains("content")) {
          std::string piece = delta.value("content", "");
          if (!piece.empty()) {
            if (!st.pieces) {
              st.first_answer = now;
              st.phase_sent = 1;
              PostMessageW(g_hwnd, WM_CHAT_PHASE, 1, 0);
            }
            ++st.pieces;
            st.last_chunk = now;
            PostMessageW(g_hwnd, WM_CHAT_CHUNK, 0, (LPARAM)new std::string(std::move(piece)));
          }
        }
      }
    } catch (...) { /* неполный/странный фрагмент — пропустить */ }
  }
  return true;
}

// Сообщение пользователя: текст + вложения одним user-сообщением. С картинками
// content становится массивом (OpenAI-формат, понимают vision-сборки
// llama-server): текст и image_url с data-URI.
json compose_user(const std::string& text, const std::vector<Attachment>& atts) {
  std::string out = text;
  for (auto& a : atts)
    if (!a.image)
      out += "\n\n--- Файл: " + a.name + " ---\n" + a.content + "\n--- конец файла ---";
  if (out.empty()) out = "Проанализируй приложенные файлы."; // вложения без текста
  bool any_image = false;
  for (auto& a : atts) any_image = any_image || a.image;
  if (!any_image) return json(out);
  json arr = json::array();
  arr.push_back({{"type", "text"}, {"text", out}});
  for (auto& a : atts)
    if (a.image)
      arr.push_back({{"type", "image_url"},
                     {"image_url", {{"url", "data:" + a.mime + ";base64," +
                                               lh::base64_encode(a.content)}}}});
  return arr;
}

void start_stream(const std::string& user_text, const std::vector<Attachment>& attachments) {
  // Вопрос уйдёт в историю только парой с ответом (WM_CHAT_DONE): шаблоны
  // новых моделей требуют строгого чередования user/assistant, а промах —
  // голый 500 без текста. До успеха g_history не трогаем.
  g_pending_user = compose_user(user_text, attachments);
  auto history = g_history;
  history.emplace_back("user", g_pending_user);

  config::Config cfg = config::current();
  json messages = json::array();
  messages.push_back({{"role", "system"}, {"content", system_prompt()}});
  for (auto& [role, content] : history)
    messages.push_back({{"role", role}, {"content", content}});
  json req = {{"model", g_chat_pick.empty() ? sup::chat_model() : g_chat_pick},
              {"stream", true},
              {"max_tokens", 4096},
              {"temperature", 0.7},
              {"messages", messages}};
  // Выключенные размышления — штатным ключом шаблона: модели без него
  // (и серверные сборки старее) просто игнорируют незнакомую переменную.
  if (!cfg.chat_think) req["chat_template_kwargs"] = json{{"enable_thinking", false}};
  std::string body = req.dump();

  g_streaming = true;
  g_cancel = false;
  g_live_phase = -1;
  g_live_pieces = 0;
  g_live_start = std::chrono::steady_clock::now();
  SetTimer(g_hwnd, 2, 1000, nullptr); // тикающий статус генерации
  update_stream_status();
  std::thread([body, port = cfg.openai_port, auth = cfg.openai_auth] {
    httplib::Client cli("127.0.0.1", port);
    cli.set_connection_timeout(std::chrono::seconds(5));
    cli.set_read_timeout(std::chrono::seconds(300));

    auto t0 = std::chrono::steady_clock::now();
    std::string pending, err;
    StreamStats st;
    bool ok = false;
    // В этой версии httplib у Client::Post нет приёмника ответа — собираем
    // запрос руками: content_receiver получает тело по мере прихода (SSE).
    httplib::Request req;
    req.method = "POST";
    req.path = "/v1/chat/completions";
    req.headers = {{"Content-Type", "application/json"}};
    if (auth) req.headers.emplace("Authorization", "Bearer " + config::client_token());
    req.body = body;
    req.content_receiver = [&](const char* data, size_t len, uint64_t, uint64_t) {
      if (g_cancel) return false; // «Стоп» — httplib оборвёт запрос
      return pump_sse(std::string(data, len), pending, st);
    };
    auto res = cli.send(req);
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (g_cancel) err = "остановлено";
    else if (!res) {
      err = "прокси не отвечает";
      logf("chat: прокси недоступен: %s", httplib::to_string(res.error()).c_str());
    } else if (res->status != 200) {
      err = "код " + std::to_string(res->status);
      try {
        err += ": " + json::parse(res->body).at("error").at("message").get<std::string>();
      } catch (...) {}
      // Тело ошибки может быть пустым (голый 500 httplib — исключение в
      // обработчике llama-server): в дампе хвоста его вывода обычно причина.
      std::string tail = sup::server_log_tail();
      std::string line = lh::proc::last_meaningful_line(tail);
      logf("chat: прокси вернул %d, тело: %.*s%s%s", res->status,
           (int)(std::min<size_t>)(res->body.size(), 512), res->body.c_str(),
           line.empty() ? "" : " | server: ", line.c_str());
    } else ok = true;

    wchar_t buf[192];
    auto plural = [](int n) -> const wchar_t* {
      if (n % 10 == 1 && n % 100 != 11) return L"токен";
      if (n % 10 >= 2 && n % 10 <= 4 && (n % 100 < 10 || n % 100 >= 20)) return L"токена";
      return L"токенов";
    };
    // Размышления: от первого reasoning-фрагмента до первого фрагмента ответа;
    // скорость — только по ответной части (до последнего фрагмента потока).
    if (ok && st.reason_pieces > 0 && st.pieces > 0 && st.first_reason < st.first_answer) {
      double reason = std::chrono::duration<double>(st.first_answer - st.first_reason).count();
      double answer = std::chrono::duration<double>(st.last_chunk - st.first_answer).count();
      const wchar_t* rs = reason < 1.0 ? L"<1 с" : nullptr;
      if (answer > 0.3) {
        if (rs) swprintf(buf, 192, L"готово · размышлял %s · ответ %.1f ток/с", rs, st.pieces / answer);
        else swprintf(buf, 192, L"готово · размышлял %.0f с · ответ %.1f ток/с", reason, st.pieces / answer);
      } else if (rs) swprintf(buf, 192, L"готово · размышлял %s", rs);
      else swprintf(buf, 192, L"готово · размышлял %.0f с", reason);
    } else if (ok && st.reason_pieces > 0 && st.pieces == 0) {
      // Мысль была, ответа не случилось: чаще всего лимит токенов (2048)
      // кончился на размышлениях (finish_reason=length) — сказать честно.
      double reason = std::chrono::duration<double>(st.last_chunk - st.first_reason).count();
      if (st.finish == "length")
        swprintf(buf, 192, L"готово · размышлял %.0f с · лимит токенов кончился на размышлениях — ответа не было",
                 reason);
      else if (reason >= 1.0)
        swprintf(buf, 192, L"готово · размышлял %.0f с · ответа не было", reason);
      else
        swprintf(buf, 192, L"готово · ответа не было");
    } else if (ok && st.pieces > 0 && secs > 0.3)
      swprintf(buf, 192, L"готово · %d %s · %.1f ток/с", st.pieces, plural(st.pieces), st.pieces / secs);
    else if (ok) swprintf(buf, 192, L"готово");
    else swprintf(buf, 192, L"%s", utf8_to_wide(err).c_str());
    PostMessageW(g_hwnd, WM_CHAT_DONE, ok ? 1 : 0, (LPARAM)new std::string(wide_to_utf8(buf)));
  }).detach();
}

void send_from_input() {
  if (g_streaming) return; // во время генерации «Отправить» — это «Стоп»
  // Длина — по факту содержимого: фиксированный буфер резал длинные вопросы.
  int len = GetWindowTextLengthW(g_input);
  std::wstring wbuf(len + 1, L'\0');
  GetWindowTextW(g_input, wbuf.data(), len + 1);
  wbuf.resize(len);
  std::string text = wide_to_utf8(wbuf);
  bool only_space = true;
  for (char c : text)
    if (c != ' ' && c != '\t' && c != '\r' && c != '\n') { only_space = false; break; }
  if (only_space && g_attachments.empty()) return;

  // Финальная строка статуса прошлого обмена («· готово · …») остаётся в
  // ленте отдельным абзацем: закрываем её переводом строки, иначе «Вы:»
  // прилипает к её хвосту. Якорь сбрасываем — строка стала обычным текстом.
  if (g_status_cp >= 0) {
    append_text(L"\n", false);
    g_status_cp = -1;
  }
  append_text(L"Вы: ", true);
  append_text(utf8_display(text) + L"\n", false);
  for (auto& a : g_attachments) {
    append_text(a.image ? L"  📷 " : L"  📎 ", false);
    append_text(utf8_display(a.name) + L" (" +
                    std::to_wstring(a.content.size() / 1024) + L" КБ)\n",
                false);
  }
  SetWindowTextW(g_input, L"");
  append_text(L"llmhost: ", true);
  g_answer_start = log_end();
  g_status_cp = -1; // статусная строка этого обмена ещё не создана
  g_pending.clear();

  // Отправка — воля читателя: якорим ленту к низу, даже если он уехал вверх
  // почитать историю. Без этого весь обмен стримит за нижней кромкой — follow
  // в чанках и статусе false, пока он сам не доскроллит.
  feed_scroll_bottom();

  // Вложения уходят в контекст и сбрасываются — каждая отправка самодостаточна.
  std::vector<Attachment> sent = std::move(g_attachments);
  g_attachments.clear();
  refresh_attachment_status();

  g_streaming = true; // перерисует отправку в «стоп»
  refresh_buttons();
  set_status(L"будим модель / генерируем…");
  start_stream(text, sent);
}

// Готовый документ по итогам работы: последний ответ модели → файл.
void save_reply() {
  const json* last = nullptr;
  for (auto it = g_history.rbegin(); it != g_history.rend(); ++it)
    if (it->first == "assistant" && it->second.is_string()) { last = &it->second; break; }
  if (!last) { set_status(L"пока нечего сохранять — сначала получите ответ"); return; }

  wchar_t buf[MAX_PATH] = L"llmhost-ответ.md";
  OPENFILENAMEW of{};
  of.lStructSize = sizeof(of);
  of.hwndOwner = g_hwnd;
  of.lpstrFilter = L"Markdown\0*.md\0Текст\0*.txt\0Все файлы\0*.*\0";
  of.lpstrFile = buf;
  of.nMaxFile = MAX_PATH;
  of.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST;
  of.lpstrDefExt = L"md";
  of.lpstrTitle = L"Сохранить ответ как документ";
  if (!GetSaveFileNameW(&of)) return;

  std::wstring path(buf);
  std::string md = last->get<std::string>();
  // BOM, чтобы Markdown-файлы корректно открывались в Windows-редакторах.
  if (!write_small_file(path, "\xEF\xBB\xBF" + md))
    set_status(L"не удалось записать файл");
    else
      set_status(L"сохранено: " + path);
}

// Стереть всё: ленту, контекст модели, вложения. История в g_history тоже
// уходит — следующий вопрос модель получит с чистого листа.
void clear_chat() {
  SendMessageW(g_log, EM_SETREADONLY, FALSE, 0);
  SendMessageW(g_log, WM_SETTEXT, 0, (LPARAM)L"");
  SendMessageW(g_log, EM_SETREADONLY, TRUE, 0);
  g_history.clear();
  g_pending.clear();
  g_attachments.clear();
  g_links.clear(); // диапазоны умерли вместе с текстом ленты
  g_answer_start = 0;
  g_status_cp = -1; // лента пуста — статусной строки нет
  refresh_attachment_status();
  append_welcome();
  set_status(L"история очищена");
}

LRESULT CALLBACK wndproc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
  g_hwnd = hwnd;
  switch (msg) {
    case WM_CREATE: {
      // Без WS_CHILD окно стало бы скрытым top-level: текст в него писался бы
      // (SendMessage работает), но в окне чата — пустота.
      g_log = CreateWindowExW(0, MSFTEDIT_CLASS, nullptr,
                              WS_CHILD | WS_VISIBLE | ES_READONLY | ES_MULTILINE |
                                  WS_VSCROLL | ES_AUTOVSCROLL,
                              0, 0, 0, 0, g_hwnd, (HMENU)(INT_PTR)IDC_LOG, nullptr, nullptr);
      SendMessageW(g_log, WM_SETFONT, (WPARAM)lh::ui_font(), TRUE);
      // ENM_LINK — курсор-рука над CFE_LINK; клик обрабатывает log_subclass.
      SendMessageW(g_log, EM_SETEVENTMASK, 0, ENM_LINK);
      // Tracking-тултип: адрес ссылки при наведении (сам адрес в ленте скрыт).
      g_tip = CreateWindowExW(WS_EX_TOPMOST, TOOLTIPS_CLASS, nullptr,
                              WS_POPUP | TTS_NOPREFIX | TTS_ALWAYSTIP,
                              0, 0, 0, 0, g_hwnd, nullptr, GetModuleHandleW(nullptr),
                              nullptr);
      if (g_tip) {
        TOOLINFOW ti{};
        ti.cbSize = sizeof(ti);
        ti.hwnd = g_log;
        ti.uId = 1;
        ti.uFlags = TTF_TRACK | TTF_ABSOLUTE;
        SendMessageW(g_tip, TTM_ADDTOOLW, 0, (LPARAM)&ti);
        SendMessageW(g_tip, TTM_SETMAXTIPWIDTH, 0, 420);
      }
      SetWindowSubclass(g_log, log_subclass, 3, 0);
      // Перенос по словам включён (EM_SETTARGETDEVICE не отключаем): таблицы
      // теперь настоящие RTF, длинный текст ячейки заворачивается сам.

      g_input = CreateWindowExW(0, L"EDIT", nullptr,
                                WS_CHILD | WS_VISIBLE | WS_TABSTOP | ES_MULTILINE |
                                    ES_AUTOVSCROLL,
                                0, 0, 0, 0, g_hwnd, (HMENU)(INT_PTR)IDC_INPUT, nullptr,
                                nullptr);
      SendMessageW(g_input, WM_SETFONT, (WPARAM)lh::ui_font(), TRUE);

      add_send_button(0, 0, SEND_S, SEND_S);
      ws::button(g_hwnd, IDC_ATTACH, 0, 0, ATTACH_S, ATTACH_S, nullptr, GLYPH_ATTACH);
      ws::button(g_hwnd, IDC_SAVE, 0, 0, SAVE_W, SAVE_H, L"Сохранить", GLYPH_SAVE);
      ws::button(g_hwnd, IDC_CLEAR, 0, 0, SAVE_W, SAVE_H, L"Очистить", GLYPH_CLEAR);

      // Выбор модели (роутер): высота окна комбобокса = закрытая часть + список.
      g_model_combo = CreateWindowExW(0, L"COMBOBOX", nullptr,
                                      WS_CHILD | WS_VISIBLE | WS_TABSTOP | CBS_DROPDOWNLIST |
                                          WS_VSCROLL,
                                      0, 0, 0, 0, g_hwnd, (HMENU)(INT_PTR)IDC_MODEL,
                                      nullptr, nullptr);
      SendMessageW(g_model_combo, WM_SETFONT, (WPARAM)lh::ui_font(), TRUE);

      // Размышления модели: выключено — в запрос уходит enable_thinking=false.
      HWND think = CreateWindowExW(0, L"BUTTON", L"Мысли",
                                   WS_CHILD | WS_VISIBLE | WS_TABSTOP | BS_AUTOCHECKBOX,
                                   0, 0, 0, 0, g_hwnd, (HMENU)(INT_PTR)IDC_THINK,
                                   nullptr, nullptr);
      SendMessageW(think, WM_SETFONT, (WPARAM)lh::ui_font(), TRUE);
      SendMessageW(think, BM_SETCHECK,
                   config::current().chat_think ? BST_CHECKED : BST_UNCHECKED, 0);

      SendMessageW(hwnd, DM_SETDEFID, IDC_SEND, 0);
      relayout(hwnd); // WM_SIZE уже прошёл до создания детей — раскладываем сами
      SetTimer(hwnd, 1, 2000, nullptr); // список моделей живёт своей жизнью
      append_welcome();
      return 0;
    }
    case WM_TIMER:
      if (wp == 1) refresh_models(false);
      else if (wp == 2) update_stream_status();
      return 0;
    case WM_SIZE:
      relayout(hwnd);
      ws::round_window(hwnd); // регион (Win10) должен жить по размеру окна
      return 0;
    case WM_GETMINMAXINFO:
      ws::min_size(lp, MIN_W, MIN_H);
      return 0;
    case WM_PAINT: {
      PAINTSTRUCT ps;
      HDC dc = BeginPaint(hwnd, &ps);
      // Компоуз-карточка: белая, скруглённая, с тонкой рамкой; когда ввод в
      // фокусе — рамка темнее и толще (2px).
      RECT all{};
      GetClientRect(hwnd, &all);
      FillRect(dc, &all, (HBRUSH)GetStockObject(WHITE_BRUSH));
      RECT card = g_geo.card;
      if (g_input_focus)
        round_rect(dc, card, CARD_RAD, CLR_CARD_FOCUS, 2, CLR_BG);
      else
        round_rect(dc, card, CARD_RAD, CLR_CARD_BORDER, 1, CLR_BG);
      EndPaint(hwnd, &ps);
      return 0;
    }
    case WM_CTLCOLOREDIT:
      // Ввод внутри карточки: белый фон без рамок — рамку рисует карточка.
      SetBkColor((HDC)wp, CLR_BG);
      SetTextColor((HDC)wp, CLR_TEXT);
      return (LRESULT)GetStockObject(WHITE_BRUSH);
    case WM_DRAWITEM: {
      auto* dis = (const DRAWITEMSTRUCT*)lp;
      if (dis->CtlID == IDC_SEND) draw_send(dis);
      else ws::draw_button(dis); // скрепка, «Сохранить» — общий плоский стиль
      return TRUE;
    }
    case WM_COMMAND: {
      int id = LOWORD(wp);
      int code = HIWORD(wp);
      if (id == IDC_INPUT && (code == EN_SETFOCUS || code == EN_KILLFOCUS)) {
        g_input_focus = code == EN_SETFOCUS;
        InvalidateRect(hwnd, nullptr, FALSE); // рамка карточки: серая ↔ акцент
        return 0;
      }
      if (id == IDC_SEND) {
        if (g_streaming) g_cancel = true; // «Стоп»
        else send_from_input();
      } else if (id == IDC_ATTACH && !g_streaming) {
        attach_file();
      } else if (id == IDC_MODEL && code == CBN_SELCHANGE) {
        int idx = (int)SendMessageW(g_model_combo, CB_GETCURSEL, 0, 0);
        if (idx >= 0 && idx < (int)g_model_ids.size()) {
          g_chat_pick = g_model_ids[idx];
          logf("chat: цель чата — %s", g_chat_pick.c_str());
        }
      } else if (id == IDC_THINK && code == BN_CLICKED) {
        // Настройка живёт в config.json — переживает рестарт приложения.
        bool on = SendMessageW(GetDlgItem(hwnd, IDC_THINK), BM_GETCHECK, 0, 0) == BST_CHECKED;
        config::Config c = config::current();
        if (c.chat_think != on) {
          c.chat_think = on;
          config::save(c);
          config::set_current(c);
          logf("chat: размышления %s", on ? "включены" : "выключены");
        }
      } else if (id == IDC_SAVE && !g_streaming) {
        save_reply();
      } else if (id == IDC_CLEAR && !g_streaming) {
        // Кнопки «Очистить» и «Сохранить» соседи — случайный клик не должен
        // сжигать переписку, поэтому короткое подтверждение.
        if (MessageBoxW(hwnd, L"Очистить всю переписку? Контекст модели тоже сбросится.",
                        L"llmhost — чат", MB_OKCANCEL | MB_ICONQUESTION) == IDOK)
          clear_chat();
      } else if (id == IDCANCEL && g_streaming) {
        g_cancel = true; // Esc во время генерации = стоп
      }
      return 0;
    }
    case WM_CHAT_PHASE: {
      // Смена фазы генерации: пересчитать t0, чтобы тикающий статус и финальная
      // строка мерили одинаково. Сообщения приходят строго до WM_CHAT_DONE
      // (та же очередь рабочего потока), но гард от поздних не помешает.
      if (!g_streaming) return 0;
      g_live_phase = wp ? 1 : 0;
      g_phase_t0 = std::chrono::steady_clock::now();
      if (!wp) g_live_pieces = 0;
      update_stream_status();
      return 0;
    }
    case WM_CHAT_CHUNK: {
      std::string* s = (std::string*)lp;
      g_pending += *s;
      ++g_live_pieces;
      // Автопрокрутка решается ДО вставки: стоял внизу — едем вниз, прокрутил
      // вверх почитать — не трогаем.
      bool follow = feed_at_bottom();
      // Фрагмент ответа — ПЕРЕД статусной строкой (она всегда последняя).
      // Формат задаём явно: слева может быть и жирное «llmhost:», и серый
      // курсив статуса — вставка не должна наследовать ни тот, ни другой.
      std::wstring w = utf8_display(*s);
      SendMessageW(g_log, EM_SETREADONLY, FALSE, 0);
      int at = g_status_cp >= 0 ? g_status_cp : log_end();
      int before = log_end();
      CHARRANGE ins{at, at};
      SendMessageW(g_log, EM_EXSETSEL, 0, (LPARAM)&ins);
      CHARFORMAT2W cf{};
      cf.cbSize = sizeof(cf);
      cf.dwMask = CFM_COLOR | CFM_ITALIC | CFM_BOLD;
      cf.crTextColor = CLR_TEXT;
      SendMessageW(g_log, EM_SETCHARFORMAT, SCF_SELECTION, (LPARAM)&cf);
      SendMessageW(g_log, EM_REPLACESEL, FALSE, (LPARAM)w.c_str());
      if (g_status_cp >= 0) g_status_cp += log_end() - before;
      CHARRANGE end{log_end(), log_end()};
      SendMessageW(g_log, EM_EXSETSEL, 0, (LPARAM)&end);
      if (follow) feed_scroll_bottom();
      SendMessageW(g_log, EM_SETREADONLY, TRUE, 0);
      delete s;
      return 0;
    }
    case WM_CHAT_DONE: {
      std::string* s = (std::string*)lp;
      if (wp) {
        // Успех: сырой Markdown стрима заменяем стилизованным RTF-рендером —
        // выделяем ответ и вливаем вместо него RTF одним STREAMIN.
        insert_markdown(g_pending);
        g_status_cp = -1; // статусная строка стёрлась вместе с сырым ответом
        logf("chat: ответ отрендерен, лента %d симв.", log_end());

        // Самодиагностика: фактический текст и фактический шрифт начала
        // ответа — читать изнутри процесса (снаружи RichEdit не отдаёт).
        {
          SendMessageW(g_log, EM_SETREADONLY, FALSE, 0);
          TEXTRANGEW tr{};
          wchar_t buf[64] = {};
          tr.chrg.cpMin = g_answer_start;
          tr.chrg.cpMax = g_answer_start + 60;
          tr.lpstrText = buf;
          SendMessageW(g_log, EM_GETTEXTRANGE, 0, (LPARAM)&tr);
          CHARRANGE probe{g_answer_start, g_answer_start + 1};
          SendMessageW(g_log, EM_EXSETSEL, 0, (LPARAM)&probe);
          CHARFORMAT2W cf{};
          cf.cbSize = sizeof(cf);
          SendMessageW(g_log, EM_GETCHARFORMAT, SCF_SELECTION, (LPARAM)&cf);
          logf("chat: факт: шрифт='%S' h=%d bold=%d текст='%S'",
               cf.szFaceName, (int)cf.yHeight,
               (int)((cf.dwEffects & CFE_BOLD) != 0), buf);
          int fin = log_end();
          CHARRANGE end{(LONG)fin, (LONG)fin};
          SendMessageW(g_log, EM_EXSETSEL, 0, (LPARAM)&end);
          SendMessageW(g_log, EM_SETREADONLY, TRUE, 0);
        }
        g_history.emplace_back("user", g_pending_user);
        g_history.emplace_back("assistant", json(g_pending));
        if (g_history.size() > MAX_HISTORY) {
          // Режем парами с начала: разрыв пары снова ломает чередование ролей.
          size_t excess = g_history.size() - MAX_HISTORY;
          g_history.erase(g_history.begin(),
                          g_history.begin() + (excess % 2 ? excess + 1 : excess));
        }
        append_text(L"\n", false);
      }
      g_pending.clear();
      KillTimer(hwnd, 2); // тикающий статус больше не нужен
      g_live_phase = -1;
      set_status(utf8_display(*s));
      g_streaming = false;
      refresh_buttons();
      delete s;
      return 0;
    }
    case WM_CONTEXTMENU: {
      // Лента приходит своим subclass-ом; сюда долетает правый клик по вводу.
      if ((HWND)wp == g_input) {
        POINT pt = ctx_menu_point(lp);
        show_edit_menu(hwnd, g_input, pt.x, pt.y);
        return 0;
      }
      return DefWindowProcW(hwnd, msg, wp, lp);
    }
    case WM_NOTIFY: {
      // EN_LINK здесь только протоколируем: открытие делает сабкласс ленты
      // (log_subclass), иначе один клик давал бы два окна браузера.
      auto* nm = (NMHDR*)lp;
      if (nm->idFrom == IDC_LOG && nm->code == EN_LINK) {
        auto* el = (ENLINK*)lp;
        logf("chat: EN_LINK msg=%u cp=[%ld,%ld)", el->msg, (long)el->chrg.cpMin,
             (long)el->chrg.cpMax);
        return 0;
      }
      return DefWindowProcW(hwnd, msg, wp, lp);
    }
    case WM_CLOSE:
      ShowWindow(hwnd, SW_HIDE);
      return 0;
    default:
      return DefWindowProcW(hwnd, msg, wp, lp);
  }
}

} // namespace

void open() {
  static const bool richedit = [] {
    // Только системный msftedit.dll: копия рядом с exe (флэшка!) исполнилась
    // бы внутри llmhost — классическая DLL-подмена.
    LoadLibraryExW(L"msftedit.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    return true;
  }();
  (void)richedit;

  if (g_hwnd && IsWindow(g_hwnd)) {
    ShowWindow(g_hwnd, SW_SHOWNORMAL);
    SetForegroundWindow(g_hwnd);
    refresh_models(true); // в скрытом окне список не обновлялся — наверстать
    feed_scroll_bottom(); // открываем на последней реплике, а не где-то в истории
    return;
  }

  WNDCLASSEXW wc{};
  wc.cbSize = sizeof(wc);
  wc.lpfnWndProc = wndproc;
  wc.hInstance = GetModuleHandleW(nullptr);
  wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
  wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
  wc.hIcon = wc.hIconSm = LoadIconW(GetModuleHandleW(nullptr), MAKEINTRESOURCEW(1));
  wc.lpszClassName = CHAT_CLASS;
  RegisterClassExW(&wc);

  DWORD style = WS_OVERLAPPEDWINDOW; // перетаскиваемые границы + максимизация
  RECT r{0, 0, 660, 520};
  AdjustWindowRect(&r, style, FALSE);
  CreateWindowExW(0, CHAT_CLASS, L"llmhost — чат", style,
                  CW_USEDEFAULT, CW_USEDEFAULT, r.right - r.left, r.bottom - r.top,
                  nullptr, nullptr, wc.hInstance, nullptr);
  if (g_hwnd) {
    ws::round_window(g_hwnd);
    ShowWindow(g_hwnd, SW_SHOWNORMAL);
    SetForegroundWindow(g_hwnd);
    SetFocus(g_input);
    refresh_models(true); // окно видно — первый наполн списка моделей
    feed_scroll_bottom();
  }
}

bool filter(MSG& msg) {
  if (!g_hwnd || !IsWindow(g_hwnd)) return false;
  // Enter во вводе: без модификаторов — отправить (во время генерации
  // игнорируем: кнопка в этот момент «Стоп», случайный Enter её не жмёт);
  // Ctrl+Enter / Shift+Enter — перевод строки (редактор без ES_WANTRETURN
  // сам Enter не вставляет). Перехват до IsDialogMessageW: Enter через
  // дефолт-кнопку не работает — DM_SETDEFID из WM_CREATE недействителен.
  if (msg.message == WM_KEYDOWN && msg.wParam == VK_RETURN && msg.hwnd == g_input) {
    const bool ctrl = GetKeyState(VK_CONTROL) < 0;
    const bool shift = GetKeyState(VK_SHIFT) < 0;
    if (ctrl || shift) {
      SendMessageW(g_input, EM_REPLACESEL, TRUE, (LPARAM)L"\r\n"); // с undo
    } else if (!g_streaming) {
      SendMessageW(g_hwnd, WM_COMMAND, MAKEWPARAM(IDC_SEND, 0), 0);
    }
    return true; // клавишу съели — ни дефолт-кнопки, ни писка
  }
  return IsDialogMessageW(g_hwnd, &msg) != 0;
}

} // namespace lh::chatui
