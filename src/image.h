// Каталог image-моделей (второй движок — sd-server из stable-diffusion.cpp).
// Раскладка каталога — категориями sd-server: diffusion_models/, text_encoders/,
// vae/ и рецепты recipes/*.json. Рецепт — источник правды о слотах и дефолтах
// генерации, но НЕ об именах файлов: их выбирает клиент при загрузке, а парсер
// рецептов отвергает неизвестные поля (check_keys в апстриме) — расширить его
// привязками нельзя. Поэтому привязка «модель → файлы слотов» живёт отдельно,
// в llmhost-image.json в корне каталога: единственный источник правды о
// привязках, ребёнку sd-server он не виден (не рецеп — сканером игнорируется).
// Рецепт для записи обязателен; битые записи пропускаются с diag в журнал.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "config.h"

namespace lh::image {

// Каталог image-моделей: image-models-dir, а пусто — общий models_dir.
// Единая точка: спавн ребёнка, pull image-моделей и удаление обязаны
// сходиться в один каталог, иначе привязки и файлы разъезжаются.
std::string models_dir_resolved(const config::Config& cfg);

// Корректный id модели/рецепта: [A-Za-z0-9._-], 1..128 символов. Правило
// одно для скана, писателя привязок и id из репо-конвенции.
bool id_ok(const std::string& s);

// Имя верхнеуровневого каталога-категории sd-server? Сравнение без регистра —
// сканер sd-server сравнивает имена каталогов так же. Потребители: сканер
// llama-реестра (modelreg) не должен индексировать эти каталоги — энкодеры
// легитимные GGUF (мусорные «чат-модели»), VAE — «не gguf»-карты.
bool is_catalog_dir_name(const std::string& name);

// Одна модель каталога: файлы-слоты резолвлены и существуют на диске.
struct Model {
  std::string id;        // уникальный id из llmhost-image.json
  std::string recipe;    // id рецепта (recipes/<recipe>.json)
  std::string name;      // человекочитаемое имя (из рецепта, иначе id)
  std::string diffusion; // diffusion_models/*.gguf — абсолютный путь
  std::string llm;       // text_encoders/* — текстовый энкодер (gguf)
  std::string vae;       // vae/* — декодер (safetensors; tae не используем)
  uint64_t bytes = 0;    // сумма весов — предполёт VRAM image-движка
  // Дефолты генерации из рецепта (ребёнку — CLI-дефолтами; -1/пусто — нет).
  int steps = -1;
  double cfg = -1.0;
  std::string sampler;
  std::string scheduler;
  // Фичи рецепта (массив «features» — тот же словарь, что у sd-server:
  // ref_images, img_cfg, flow_shift…). Прокси гейтит /v1/images/edits по
  // наличию «ref_images»: апстрим на моделях без фичи падает на GGML_ASSERT
  // (z_image.hpp:808, инцидент 09.11).
  std::vector<std::string> features;
};

// Скан каталога: llmhost-image.json + рецепты. Нет файла привязок — пустой
// каталог (не ошибка: движок просто не настроен). Пути слотов — относительно
// dir; абсолютные, «..» и control-символы — отказ записи.
std::vector<Model> scan(const std::string& dir);

// Есть ли файл привязок (настроен ли движок хоть на одну модель).
bool catalog_exists(const std::string& dir);

// Запись привязки «модель → файлы слотов» — единственный писатель
// llmhost-image.json (pull дописывает, удаление стирает). Поля — как в скане,
// пути относительно dir. Чужие записи и неизвестные ключи существующей записи
// сохраняются; запись атомарная (tmp+rename).
struct Binding {
  std::string id, recipe, diffusion_model, llm, vae;
};

// Дописать/заменить запись с этим id. false — err (плохой id/путь, битый
// существующий файл, отказ записи).
bool binding_upsert(const std::string& dir, const Binding& b, std::string& err);

// Убрать запись. Найденная запись возвращается в removed (может быть nullptr).
// false — записи нет или err.
bool binding_remove(const std::string& dir, const std::string& id, Binding* removed,
                    std::string& err);

} // namespace lh::image
