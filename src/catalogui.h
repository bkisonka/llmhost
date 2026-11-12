// Окно «Каталог моделей»: поиск по HuggingFace, светофор «влезет ли»,
// выбор кванта, скачивание и загрузка — без браузера, чистый WinAPI.
#pragma once
#include <windows.h>

namespace lh::catalogui {

void open();          // показать/поднять окно (создаётся один раз)
bool filter(MSG& msg); // маршрутизация Enter/Tab из цикла трея; true — съедено

} // namespace lh::catalogui
