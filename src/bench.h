// Замер скорости установленных сборок llama-server на одной модели:
// спавн, /health, генерация фиксированного промпта, timings из ответа сервера.
// Победитель запоминается в конфиг (engine_build), заодно проверяется -ub 256.
// Асинхронно, как pull/engine; слот модели должен быть свободен — замеру
// нужна вся видеопамять.
#pragma once
#include <string>

namespace lh::bench {

// model_path пуст — берём модель слота (и спящего). false — err с причиной.
bool start(const std::string& model_path, std::string& err);
std::string status_json();
bool busy();

} // namespace lh::bench
