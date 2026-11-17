// Каталог image-моделей: привязки llmhost-image.json + рецепты sd-server.
#include "image.h"
#include "util.h"

#include <unistd.h>

#include <cassert>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

using namespace lh;
using namespace lh::image;

namespace {

void put(const fs::path& p, const std::string& body) {
  fs::create_directories(p.parent_path());
  std::ofstream f(p, std::ios::binary);
  f << body;
}

const char* kRecipe = R"({
  "id": "z-image-turbo",
  "name": "Z-Image Turbo",
  "slots": [{"id": "diffusion_model", "required": true},
            {"id": "llm", "required": true},
            {"id": "vae", "required": false}],
  "defaults": {"sampling_method": "euler", "cfg_scale": 1, "steps": 9},
  "features": ["flow_shift", "ref_images", 42]
})";

const char* kBinding = R"({
  "models": [
    {"id": "z-image-turbo", "recipe": "z-image-turbo",
     "diffusion_model": "diffusion_models/z_image_turbo-Q8_0.gguf",
     "llm": "text_encoders/qwen3-4b-Q4_K_M.gguf",
     "vae": "vae/ae.safetensors"}
  ]
})";

// Заготовка каталога: рецепты + файлы-слоты. Возвращает корень.
std::string make_catalog() {
  fs::path d = fs::temp_directory_path() /
               ("llmhost-image-test-" + std::to_string(::getpid()));
  fs::remove_all(d);
  put(d / "recipes" / "z-image-turbo.json", kRecipe);
  put(d / "diffusion_models" / "z_image_turbo-Q8_0.gguf", "DIFF");
  put(d / "text_encoders" / "qwen3-4b-Q4_K_M.gguf", "LLM");
  put(d / "vae" / "ae.safetensors", "VAE");
  put(d / "llmhost-image.json", kBinding);
  return d.string();
}

} // namespace

int main() {
  // is_catalog_dir_name: без регистра, только категории sd-server.
  assert(is_catalog_dir_name("diffusion_models"));
  assert(is_catalog_dir_name("Text_Encoders"));
  assert(is_catalog_dir_name("VAE"));
  assert(is_catalog_dir_name("recipes"));
  assert(!is_catalog_dir_name("unsloth"));
  assert(!is_catalog_dir_name("llama.cpp"));
  assert(!is_catalog_dir_name(""));

  // Нет файла привязок — пустой каталог (не ошибка).
  {
    std::string d = make_catalog();
    fs::remove(fs::path(d) / "llmhost-image.json");
    auto v = scan(d);
    assert(v.empty());
    assert(!catalog_exists(d));
    fs::remove_all(d);
  }

  // Полная запись: рецепт даёт имя и дефолты, пути резолвятся, размеры суммируются.
  {
    std::string d = make_catalog();
    assert(catalog_exists(d));
    auto v = scan(d);
    assert(v.size() == 1);
    const Model& m = v[0];
    assert(m.id == "z-image-turbo");
    assert(m.name == "Z-Image Turbo");
    assert(m.recipe == "z-image-turbo");
    assert(m.diffusion.find("diffusion_models/z_image_turbo-Q8_0.gguf") != std::string::npos);
    assert(m.llm.find("text_encoders/") != std::string::npos);
    assert(m.vae.find("vae/") != std::string::npos);
    assert(m.bytes == 4 + 3 + 3); // "DIFF" + "LLM" + "VAE"
    assert(m.steps == 9);
    assert(m.cfg == 1.0);
    assert(m.sampler == "euler");
    // Фичи рецепта: строки собираются, не-строки (42) пропускаются молча.
    assert(m.features.size() == 2);
    assert(m.features[0] == "flow_shift" && m.features[1] == "ref_images");
    fs::remove_all(d);
  }

  // Отсутствующий файл слота — запись пропущена.
  {
    std::string d = make_catalog();
    fs::remove(fs::path(d) / "vae" / "ae.safetensors");
    assert(scan(d).empty());
    fs::remove_all(d);
  }

  // «..» и абсолютный путь — отказ записи (не резолвятся за пределы каталога).
  {
    std::string d = make_catalog();
    put(fs::path(d) / "llmhost-image.json", R"({
      "models": [{"id": "escape", "recipe": "z-image-turbo",
                  "diffusion_model": "../secret.gguf",
                  "llm": "text_encoders/qwen3-4b-Q4_K_M.gguf",
                  "vae": "vae/ae.safetensors"}]
    })");
    assert(scan(d).empty());
    put(fs::path(d) / "llmhost-image.json", R"({
      "models": [{"id": "abs", "recipe": "z-image-turbo",
                  "diffusion_model": "/etc/passwd",
                  "llm": "text_encoders/qwen3-4b-Q4_K_M.gguf",
                  "vae": "vae/ae.safetensors"}]
    })");
    assert(scan(d).empty());
    fs::remove_all(d);
  }

  // Нет рецепта — запись пропущена (рецепт обязателен).
  {
    std::string d = make_catalog();
    fs::remove(fs::path(d) / "recipes" / "z-image-turbo.json");
    assert(scan(d).empty());
    fs::remove_all(d);
  }

  // Дубликат id — скрываются все записи с ним.
  {
    std::string d = make_catalog();
    put(fs::path(d) / "llmhost-image.json", R"({
      "models": [
        {"id": "dup", "recipe": "z-image-turbo",
         "diffusion_model": "diffusion_models/z_image_turbo-Q8_0.gguf",
         "llm": "text_encoders/qwen3-4b-Q4_K_M.gguf", "vae": "vae/ae.safetensors"},
        {"id": "dup", "recipe": "z-image-turbo",
         "diffusion_model": "diffusion_models/z_image_turbo-Q8_0.gguf",
         "llm": "text_encoders/qwen3-4b-Q4_K_M.gguf", "vae": "vae/ae.safetensors"}
      ]
    })");
    assert(scan(d).empty());
    fs::remove_all(d);
  }

  // Битый json привязок — пусто, не падение.
  {
    std::string d = make_catalog();
    put(fs::path(d) / "llmhost-image.json", "{ невалидный ");
    assert(scan(d).empty());
    fs::remove_all(d);
  }

  // Значение не того типа («id»: 123) — type_error не вылетает из скана,
  // запись пропускается с diag.
  {
    std::string d = make_catalog();
    put(fs::path(d) / "llmhost-image.json", R"({
      "models": [{"id": 123, "recipe": "z-image-turbo",
                  "diffusion_model": "diffusion_models/z_image_turbo-Q8_0.gguf",
                  "llm": "text_encoders/qwen3-4b-Q4_K_M.gguf",
                  "vae": "vae/ae.safetensors"}]
    })");
    assert(scan(d).empty());
    fs::remove_all(d);
  }

  // Симлинк-побег: rel внутри каталога, цель — снаружи. Отказ записи.
  {
    std::string d = make_catalog();
    fs::path outside = fs::temp_directory_path() /
                       ("llmhost-image-outside-" + std::to_string(::getpid()) + ".gguf");
    { std::ofstream f(outside, std::ios::binary); f << "SECRET"; }
    std::error_code ec;
    fs::create_symlink(outside, fs::path(d) / "diffusion_models" / "evil.gguf", ec);
    if (!ec) {
      put(fs::path(d) / "llmhost-image.json", R"({
        "models": [{"id": "evil", "recipe": "z-image-turbo",
                    "diffusion_model": "diffusion_models/evil.gguf",
                    "llm": "text_encoders/qwen3-4b-Q4_K_M.gguf",
                    "vae": "vae/ae.safetensors"}]
      })");
      assert(scan(d).empty());
    }
    fs::remove(outside);
    fs::remove_all(d);
  }

  // ---- Писатель привязок: единственный способ править llmhost-image.json ----
  // upsert в каталог без файла привязок — создаёт файл, скан видит модель.
  {
    std::string d = make_catalog();
    fs::remove(fs::path(d) / "llmhost-image.json");
    std::string err;
    Binding b{"new-artist", "z-image-turbo",
              "diffusion_models/z_image_turbo-Q8_0.gguf",
              "text_encoders/qwen3-4b-Q4_K_M.gguf", "vae/ae.safetensors"};
    assert(binding_upsert(d, b, err));
    auto v = scan(d);
    assert(v.size() == 1);
    assert(v[0].id == "new-artist");

    // Замена существующей записи: чужие поля (пометки руками) сохраняются.
    put(fs::path(d) / "llmhost-image.json", R"({
      "models": [{"id": "new-artist", "recipe": "z-image-turbo",
                  "diffusion_model": "diffusion_models/z_image_turbo-Q8_0.gguf",
                  "llm": "text_encoders/qwen3-4b-Q4_K_M.gguf",
                  "vae": "vae/ae.safetensors", "note": "ручная пометка"}]
    })");
    b.vae = "vae/ae.safetensors";
    assert(binding_upsert(d, b, err));
    std::string raw = read_small_file((fs::path(d) / "llmhost-image.json").string());
    assert(raw.find("note") != std::string::npos);

    // Вторая запись добавляется, remove стирает только свою, файл остаётся
    // валидным (атомарная запись — rename, а не обрезка поверх).
    Binding second{"second-artist", "z-image-turbo",
                   "diffusion_models/z_image_turbo-Q8_0.gguf",
                   "text_encoders/qwen3-4b-Q4_K_M.gguf", "vae/ae.safetensors"};
    assert(binding_upsert(d, second, err));
    assert(scan(d).size() == 2);
    Binding removed;
    assert(binding_remove(d, "new-artist", &removed, err));
    assert(removed.recipe == "z-image-turbo");
    assert(removed.diffusion_model == "diffusion_models/z_image_turbo-Q8_0.gguf");
    auto rest = scan(d);
    assert(rest.size() == 1 && rest[0].id == "second-artist");
    assert(!binding_remove(d, "new-artist", nullptr, err)); // записи больше нет
    Binding missed;
    assert(binding_remove(d, "new-artist", &missed, err) == false);

    // Отказы: плохой id, пустой recipe, небезопасный путь слота.
    Binding bad{"плохой id", "z-image-turbo", "diffusion_models/x.gguf",
                "text_encoders/y.gguf", "vae/z.safetensors"};
    assert(!binding_upsert(d, bad, err));
    bad.id = "ok-id";
    bad.recipe = "";
    assert(!binding_upsert(d, bad, err));
    bad.recipe = "z-image-turbo";
    bad.diffusion_model = "../escape.gguf";
    assert(!binding_upsert(d, bad, err));
    bad.diffusion_model = "/etc/passwd";
    assert(!binding_upsert(d, bad, err));
    // После отказов каталог не пострадал.
    assert(scan(d).size() == 1);
    fs::remove_all(d);
  }

  // Единая точка каталога: пустой image-models-dir — общий models-dir.
  {
    config::Config c;
    c.image_models_dir = "";
    c.models_dir = "/data/models";
    assert(image::models_dir_resolved(c) == "/data/models");
    c.image_models_dir = "/data/images";
    assert(image::models_dir_resolved(c) == "/data/images");
  }
}
