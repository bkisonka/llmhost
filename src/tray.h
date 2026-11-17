// Трей: иконка, меню (модели, выгрузить, настройки), тултип со статусом.
#pragma once
#include <windows.h>
#include <string>

namespace lh::tray {

bool init(HINSTANCE hInst);
void run();            // цикл сообщений; выходит после quit()

// Из чужих потоков: можно звать в любой момент.
void notify_state();   // статус менялся — обновить тултип
void balloon(const std::wstring& text); // всплывающее уведомление у значка
void quit();           // завершить приложение (control /v0/quit)

// Удаление модели в корзину: выгрузит (роутер/классика), заберёт группу
// шардов целиком. Пустая строка — успех, иначе — причина отказа.
std::wstring delete_model_to_bin(const std::wstring& path);

} // namespace lh::tray
