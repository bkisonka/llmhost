// Каталог HuggingFace: поиск GGUF-репозиториев и предиктивная оценка «влезет ли»
// до скачивания. Точную прикидку даёт настоящий заголовок .gguf, дочитанный
// из сети Range-запросом (первые 512 КБ); пока его нет — вилка по имени модели
// (catalog_est.h). Ответы — готовые JSON для control API и окна каталога.
#pragma once
#include <string>

namespace lh::catalog {

// Поиск: q — текст, limit ≤ 50. JSON {"results":[...]} либо err.
std::string search_json(const std::string& q, int limit, std::string& err);

// Один репозиторий: список квантов с размерами, светофор и рекомендация.
// JSON {"repo":..., "files":[...]} либо err.
std::string detail_json(const std::string& repo, std::string& err);

} // namespace lh::catalog
