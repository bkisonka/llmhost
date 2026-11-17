// Флаг остановки демона: сюда сходятся /v0/quit (control API) и сигналы
// SIGTERM/SIGINT. На Windows роль играл tray::quit() (WM_QUIT); на Linux
// никакого GUI нет — main просто ждёт сброса флага.
#pragma once
#include <atomic>

namespace lh::quit {

inline std::atomic<bool>& flag() {
  static std::atomic<bool> f{false};
  return f;
}
inline void request() { flag().store(true); }
inline bool requested() { return flag().load(); }

} // namespace lh::quit
