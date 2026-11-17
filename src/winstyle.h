// Единый стиль окон llmhost: закруглённые углы (нативно DWM на Win11,
// регион на Win10), перетаскиваемые границы, плоские серые кнопки вместо
// классических. Header-only, используется чатом, настройками и каталогом.
#pragma once
#include "util.h"

#include <windows.h>
#include <dwmapi.h>
#include <commctrl.h>

#include <map>

namespace lh::ws {

// Закругление углов окна. На Win11+ — настройка DWM (учитывает максимизацию
// и Snap Layouts); на Win10 — скругляющий регион (углы честно круглые,
// но DWM-тень пропадает — приемлемая плата на старой системе).
inline void round_window(HWND hwnd) {
  int pref = 2; // DWMWCP_ROUND
  if (DwmSetWindowAttribute(hwnd, 33 /*DWMWA_WINDOW_CORNER_PREFERENCE*/, &pref,
                            sizeof(pref)) == S_OK)
    return;
  RECT wr{};
  GetWindowRect(hwnd, &wr);
  int w = wr.right - wr.left, h = wr.bottom - wr.top;
  HRGN rgn = IsZoomed(hwnd) ? CreateRectRgn(0, 0, w, h)
                            : CreateRoundRectRgn(0, 0, w + 1, h + 1, 28, 28);
  SetWindowRgn(hwnd, rgn, TRUE); // прежний регион освободит система
}

// Минимальный размер окна (WM_GETMINMAXINFO): меньше — контролы наезжают
// друг на друга. Размер — внешний, как в ptMinTrackSize.
inline void min_size(LPARAM lp, int min_w, int min_h) {
  auto* mmi = (MINMAXINFO*)lp;
  mmi->ptMinTrackSize = {min_w, min_h};
}

namespace detail {

inline std::map<HWND, bool>& hover() {
  static std::map<HWND, bool> m;
  return m;
}

inline LRESULT CALLBACK btn_sub(HWND h, UINT msg, WPARAM wp, LPARAM lp, UINT_PTR, DWORD_PTR) {
  switch (msg) {
    case WM_MOUSEMOVE:
      if (!hover()[h]) {
        hover()[h] = true;
        TRACKMOUSEEVENT tme{sizeof(tme), TME_LEAVE, h, 0};
        TrackMouseEvent(&tme);
        InvalidateRect(h, nullptr, TRUE);
      }
      break;
    case WM_MOUSELEAVE:
      hover()[h] = false;
      InvalidateRect(h, nullptr, TRUE);
      break;
    case WM_NCDESTROY:
      hover().erase(h);
      break;
  }
  return DefSubclassProc(h, msg, wp, lp);
}

inline void round_rect(HDC dc, const RECT& r, int rad, COLORREF border, int bw, COLORREF fill) {
  HBRUSH br = CreateSolidBrush(fill);
  HPEN pen = CreatePen(PS_SOLID, bw, border);
  HGDIOBJ ob = SelectObject(dc, br), op = SelectObject(dc, pen);
  RoundRect(dc, r.left, r.top, r.right, r.bottom, rad * 2, rad * 2);
  SelectObject(dc, ob);
  SelectObject(dc, op);
  DeleteObject(br);
  DeleteObject(pen);
}

} // namespace detail

// Плоская тихая кнопка: прозрачная в покое, при наведении — светло-серая
// скруглённая плашка, в нажатии — темнее. Глиф Segoe MDL2 Assets — опционально
// слева от текста; кнопка-по-умолчанию (Enter) получает тонкую серую рамку.
// Рисует ws::draw_button из WM_DRAWITEM родителя.
inline HWND button(HWND parent, int id, int x, int y, int w, int h,
                   const wchar_t* text, const wchar_t* glyph = nullptr, bool def = false) {
  HWND b = CreateWindowExW(0, L"BUTTON", text,
                           WS_CHILD | WS_VISIBLE | WS_TABSTOP | BS_OWNERDRAW |
                               (def ? BS_DEFPUSHBUTTON : BS_PUSHBUTTON),
                           x, y, w, h, parent, (HMENU)(INT_PTR)id, nullptr, nullptr);
  SendMessageW(b, WM_SETFONT, (WPARAM)lh::ui_font(), TRUE);
  SetPropW(b, L"lh_glyph", (HANDLE)glyph); // литерал — живёт всю сессию
  SetWindowSubclass(b, detail::btn_sub, 2, 0);
  return b;
}

inline void draw_button(const DRAWITEMSTRUCT* d) {
  HDC dc = d->hDC;
  RECT r = d->rcItem;
  bool disabled = (d->itemState & ODS_DISABLED) != 0;
  bool hover = detail::hover()[d->hwndItem] && !disabled;
  bool pressed = hover && (d->itemState & ODS_SELECTED) != 0;

  FillRect(dc, &r, (HBRUSH)GetStockObject(WHITE_BRUSH));
  if (pressed)
    detail::round_rect(dc, r, 5, RGB(0xe0, 0xe0, 0xe0), 1, RGB(0xe0, 0xe0, 0xe0));
  else if (hover)
    detail::round_rect(dc, r, 5, RGB(0xed, 0xed, 0xed), 1, RGB(0xed, 0xed, 0xed));
  if (d->itemState & ODS_DEFAULT)
    detail::round_rect(dc, r, 5, RGB(0xad, 0xad, 0xad), 1, RGB(255, 255, 255));

  const wchar_t* glyph = (const wchar_t*)GetPropW(d->hwndItem, L"lh_glyph");
  int x = r.left;
  if (glyph && *glyph) {
    HFONT f = CreateFontW(-14, 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET, 0, 0,
                          CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE,
                          L"Segoe MDL2 Assets");
    HGDIOBJ of = SelectObject(dc, f);
    int ob = SetBkMode(dc, TRANSPARENT);
    COLORREF oc = SetTextColor(dc, disabled ? RGB(0xb4, 0xb4, 0xb4) : RGB(0x4a, 0x4a, 0x4a));
    RECT gr = {r.left + 8, r.top, r.left + 24, r.bottom};
    DrawTextW(dc, glyph, -1, &gr, DT_VCENTER | DT_SINGLELINE | DT_CENTER | DT_NOPREFIX);
    SetTextColor(dc, oc);
    SetBkMode(dc, ob);
    SelectObject(dc, of);
    DeleteObject(f);
    x = gr.right + 4;
  }

  wchar_t text[128] = {};
  int tn = GetWindowTextW(d->hwndItem, text, 128);
  if (tn > 0) {
    int ob = SetBkMode(dc, TRANSPARENT);
    COLORREF oc = SetTextColor(dc, disabled ? RGB(0x9a, 0x9a, 0x9a) : RGB(0x1a, 0x1a, 0x1a));
    RECT tr = {x + 2, r.top, r.right - 4, r.bottom};
    UINT fmt = DT_VCENTER | DT_SINGLELINE | DT_NOPREFIX;
    fmt |= glyph && *glyph ? DT_LEFT : DT_CENTER;
    DrawTextW(dc, text, -1, &tr, fmt);
    SetTextColor(dc, oc);
    SetBkMode(dc, ob);
  }
}

} // namespace lh::ws
