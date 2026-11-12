// Скачивание моделей с HuggingFace: /v0/pull "org/repo" (+ фильтр файлов) —
// llama-модели в models_dir; kind=image — художники в image-каталог (встроенный
// каталог image_catalog или репо с конвенцией recipes/ + категории). Один
// single-flight стейт на оба вида: прогресс /v0/pull общий, поле kind различает.
// SHA256 берётся из LFS-метаданных (paths-info), докачка и проверка — в download.cpp.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace lh::pull {

// false — ошибка параметров (синхронно, без потока); true — скачивание запущено (202).
bool start(const std::string& repo_in, const std::string& file_filter, std::string& err);

// Image-модель («художник»): repo_or_alias — имя во встроенном каталоге
// (image_catalog) или «org/repo» с раскладкой-конвенцией (recipes/*.json +
// diffusion_models/ + text_encoders/ + vae/, по одному файлу на слот).
// По завершении — привязка в llmhost-image.json (+ рецепт), модель сразу
// видна в /v0/image и селекторе прокси. false/true — как в start.
bool start_image(const std::string& repo_or_alias, std::string& err);

// state: idle | resolving | downloading | done | error | stopped (+ kind:
// llm|image, done/total/speed/error; поле stopping — пока отмена не дошла
// до конца)
std::string status_json();
bool busy();

// Остановить идущее скачивание: curl глушится на ближайшем тике прогресса
// (≤250 мс), .part остаётся на диске — повторный pull докачивает. false —
// скачивание не идёт (err с причиной). Конечное состояние — stopped.
bool cancel(std::string& err);

// Недокачанная загрузка: <models_dir>\<org>\<repo>\<файл>.part. Появляется,
// когда скачивание оборвалось (обрыв связи, перезапуск llmhost); докачивается
// через start(repo, file) — curl продолжает с места останова.
struct PartDownload {
  std::string path;  // путь .part
  std::string file;  // целевое имя (без .part)
  std::string repo;  // "org/repo" для докачки; пусто — путь не разложить
  uint64_t size = 0;  // сколько уже скачано
};

// Сканировать папку моделей на .part-хвосты (для меню трея).
std::vector<PartDownload> scan_parts(const std::string& models_dir);

} // namespace lh::pull
