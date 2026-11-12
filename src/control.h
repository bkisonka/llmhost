// Control API :control_port — управление демоном из скриптов и CLI.
// Все запросы требуют Bearer-токен (файл token.txt рядом с config.json).
#pragma once

namespace lh::control {

bool start(int port); // false — порт занят, стартовать демона нельзя
void stop();

} // namespace lh::control
