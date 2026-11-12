// Встроенный каталог художников: источники слотов для pull image-моделей.
// Реальная раскладка на HF — много-репо: диффузия и энкодеры живут у квантизаторов
// (unsloth), VAE — в comfy-репо оригинала; единого «репо модели» не существует,
// оригиналы BFL/Tongyi — gated-safetensors. Поэтому entry = рецепт (встроенный,
// дословно формат sd-server: recipes/<id>.json) + слоты с точными репо и файлами.
// Квант в записи фиксированный (Q8_0 — рабочая точка хоста); свои раскладки —
// репо с конвенцией (см. pull: recipes/ + категории) или правка привязок руками.
// dest — имя на диске (категория выводится из слота): VAE общие у нескольких
// моделей, повторный pull существующий файл не перекачивает.
#pragma once
#include <string>
#include <vector>

namespace lh::image_catalog {

struct SlotSource {
  const char* slot; // поле привязки: diffusion_model | llm | vae
  const char* repo; // HF-репозиторий файла ("org/name")
  const char* file; // точное имя файла в репо
  const char* dest; // имя файла на диске внутри категории
};

struct Entry {
  const char* id;     // id каталога и рецепта (совпадают — как в живых привязках)
  const char* note;   // человеческая подпись для списков
  const char* recipe; // рецепт sd-server целиком (кладётся в recipes/<id>.json)
  std::vector<SlotSource> slots;
};

inline const std::vector<Entry>& all() {
  static const std::vector<Entry> k = {
      {"z-image-turbo", "Z-Image Turbo (Tongyi), 9 шагов, турбо-дистилл",
       R"({
  "schema_version": 1,
  "id": "z-image-turbo",
  "name": "Z-Image Turbo",
  "model_url": "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo",
  "slots": [
    {
      "id": "diffusion_model",
      "required": true
    },
    {
      "id": "llm",
      "required": true,
      "label": "Qwen 3 Text Encoder"
    },
    {
      "id": "vae",
      "required": false
    },
    {
      "id": "tae",
      "required": false
    }
  ],
  "required_any": [
    [
      "vae",
      "tae"
    ]
  ],
  "defaults": {
    "sampling_method": "euler",
    "cfg_scale": 1,
    "steps": 9,
    "width": 1024,
    "height": 1024
  },
  "features": [
    "flow_shift"
  ]
}
)",
       {{"diffusion_model", "unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q8_0.gguf",
         "z-image-turbo-Q8_0.gguf"},
        {"llm", "unsloth/Qwen3-4B-Instruct-2507-GGUF",
         "Qwen3-4B-Instruct-2507-Q4_K_M.gguf", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"},
        {"vae", "Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors",
         "flux1-ae.safetensors"}}},
      {"flux2-klein-4b", "FLUX.2 Klein 4B (Black Forest Labs), 4 шага",
       R"({
  "schema_version": 1,
  "id": "flux2-klein-4b",
  "name": "FLUX.2 Klein 4B",
  "size_multiple": 16,
  "description": "Supports reference-image editing. The FLUX.2 small decoder is an alternative VAE.",
  "model_url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B",
  "slots": [
    {
      "id": "diffusion_model",
      "required": true
    },
    {
      "id": "llm",
      "required": true,
      "label": "Qwen 3 4B Text Encoder"
    },
    {
      "id": "vae",
      "required": false
    },
    {
      "id": "tae",
      "required": false
    }
  ],
  "required_any": [
    [
      "vae",
      "tae"
    ]
  ],
  "defaults": {
    "cfg_scale": 1,
    "steps": 4,
    "width": 1024,
    "height": 1024,
    "sampling_method": "euler"
  },
  "features": [
    "flow_shift",
    "img_cfg",
    "ref_images",
    "slg"
  ]
}
)",
       {{"diffusion_model", "unsloth/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q8_0.gguf",
         "flux-2-klein-4b-Q8_0.gguf"},
        {"llm", "unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf",
         "Qwen3-4B-Q4_K_M.gguf"},
        {"vae", "Comfy-Org/Flux-2-Klein-4B", "split_files/vae/flux2-vae.safetensors",
         "flux2-ae.safetensors"}}},
      {"flux3-klein", "FLUX.3 Image Klein (Black Forest Labs), 16 шагов, ref-images",
       R"({
  "schema_version": 1,
  "id": "flux3-klein",
  "name": "FLUX.3 Image Klein",
  "size_multiple": 16,
  "description": "Supports text-to-image and reference-image editing.",
  "model_url": "https://huggingface.co/black-forest-labs/flux-3-image-klein",
  "slots": [
    {
      "id": "diffusion_model",
      "required": true
    },
    {
      "id": "llm",
      "required": true,
      "label": "Qwen 3 VL 4B Text Encoder"
    },
    {
      "id": "vae",
      "required": false
    },
    {
      "id": "tae",
      "required": false
    }
  ],
  "required_any": [
    [
      "vae",
      "tae"
    ]
  ],
  "defaults": {
    "sampling_method": "euler",
    "cfg_scale": 1,
    "steps": 16,
    "width": 1024,
    "height": 1024
  },
  "features": [
    "flow_shift",
    "ref_images"
  ]
}
)",
       {{"diffusion_model", "unsloth/flux-3-image-klein-GGUF",
         "flux-3-image-klein-Q8_0.gguf", "flux-3-image-klein-Q8_0.gguf"},
        {"llm", "unsloth/Qwen3-VL-4B-Instruct-GGUF", "Qwen3-VL-4B-Instruct-Q4_K_M.gguf",
         "Qwen3-VL-4B-Instruct-Q4_K_M.gguf"},
        {"vae", "Comfy-Org/Flux-2-Klein-4B", "split_files/vae/flux2-vae.safetensors",
         "flux2-ae.safetensors"}}},
  };
  return k;
}

inline const Entry* find(const std::string& id) {
  for (const auto& e : all())
    if (id == e.id) return &e;
  return nullptr;
}

} // namespace lh::image_catalog
