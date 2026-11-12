// Сеть: скачаем системой. WinHTTP в части окружений не резолвит имена, а
// системный curl умеет всё сразу — TLS, докачку (-C -), повторы. На Windows
// это %SystemRoot%\System32\curl.exe (с 10 1809), на Linux — фиксированный
// /usr/bin/curl; в обоих случаях запускаем его как ребёнка (умрёт вместе с
// демоном). SHA256: на Linux — системный sha256sum (на Windows был BCrypt).
#pragma once
#include <functional>
#include <string>
#include <cstdint>

namespace lh::dl {

struct Progress {
  uint64_t done = 0, total = 0; // total 0 — размер неизвестен
  double speed = 0;             // байт/с, сглаженная
};

// false — отменить (скачанное останется в .part для продолжения).
using ProgressFn = std::function<bool(const Progress&)>;

// SHA256 файла (системный sha256sum, ребёнок). Пусто — не посчитали;
// вызывающий обязан трактовать как отказ (fail-closed), не как допуск.
std::string sha256_file(const std::string& path);

// GET в файл. Появляется только проверенный файл: до этого — dest.part.
bool download(const std::string& url, const std::string& dest, const std::string& sha256_hex,
              const ProgressFn& on_progress, std::string& err);

// Малые GET/POST для API (ответ до 16 МБ). Не 2xx — ошибка с кодом.
bool get_string(const std::string& url, std::string& out, std::string& err);
bool post_json(const std::string& url, const std::string& body, std::string& out, std::string& err);

// Первые max_bytes ответа (Range-запрос) — читать заголовок удалённого .gguf
// до скачивания. true, если байты получены. Если сервер Range проигнорировал,
// curl оборвёт передачу по --max-filesize — возьмём то, что успело прийти.
bool get_partial(const std::string& url, size_t max_bytes, std::string& out, std::string& err);

} // namespace lh::dl
