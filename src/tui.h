// tui.h — полноэкранный интерфейс управления llmhost (в духе mc).
#pragma once

#include <string>

namespace lh::tui {

// Запуск; блокируется до выхода (F10/q/Ctrl-C). Терминал переводится в
// полный экран (альтернативный буфер, cbreak) и восстанавливается при выходе.
// port — control API, token — Bearer-ключ (может быть пуст).
int run(int port, const std::string& token);

}  // namespace lh::tui
