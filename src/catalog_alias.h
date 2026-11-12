// Короткие имена для быстрой установки: «lhctl pull qwen3» (и то же в
// POST /v0/pull) без понимания репозиториев и квантов — квант под вашу
// карту выбирает каталог (recommended), vision-проекции (mmproj)
// прицепляются автоматически.
//
// Таблица курируемая: имена проверены живьём (2026-10-01, топ по
// скачиваниям HF + семейства под рукой). Новая строка = новое имя;
// регистр не важен. Дефолт семейства — разумная середина для карт
// 8–12 ГБ; размерные варианты — отдельными именами.
#pragma once
#include <cstddef>

namespace lh::alias {

struct Entry {
  const char* name; // что набирает человек
  const char* repo; // org/repo на HuggingFace
  const char* note; // одна строка для подсказки
};

inline constexpr Entry kAll[] = {
    {"qwen3",        "unsloth/Qwen3-8B-GGUF",                          "8B — универсальный дефолт"},
    {"qwen3-30b",    "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",       "MoE 30B (эксперты могут лечь в RAM)"},
    {"qwen3-4b",     "unsloth/Qwen3-4B-Instruct-2507-GGUF",            "лёгкая для слабых карт"},
    {"qwen3-0.6b",   "unsloth/Qwen3-0.6B-GGUF",                        "крошка: черновик для спекуляции"},
    {"qwen3-coder",  "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",      "кодинг, MoE 30B"},
    {"gemma-4",      "unsloth/gemma-4-12b-it-GGUF",                    "12B, vision (mmproj скачается сам)"},
    {"gemma-4-26b",  "unsloth/gemma-4-26B-A4B-it-GGUF",                "MoE 26B, vision"},
    {"llama",        "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",       "классика 8B"},
    {"llama-3b",     "unsloth/Llama-3.2-3B-Instruct-GGUF",              "лёгкая llama"},
    {"ministral",    "unsloth/Ministral-3-3B-Instruct-2512-GGUF",       "3B от Mistral, vision"},
    {"granite",      "ibm-granite/granite-4.2-8b-GGUF",                "8B от IBM"},
    {"granite-3b",   "ibm-granite/granite-4.2-3b-GGUF",                "лёгкая granite"},
    {"granite-30b",  "ibm-granite/granite-4.2-30b-GGUF",               "MoE 30B от IBM"},
    {"phi",          "unsloth/Phi-4-mini-instruct-GGUF",               "4-mini от Microsoft"},
    {"embed",        "nomic-ai/nomic-embed-text-v1.5-GGUF",            "эмбеддинги для RAG (/v1/embeddings)"},
};

inline constexpr size_t kCount = sizeof(kAll) / sizeof(kAll[0]);

// Точное имя (регистр не важен); nullptr — не алиас.
inline const Entry* find(const char* name) {
  if (!name || !*name) return nullptr;
  for (const Entry& e : kAll) {
    const char *a = e.name, *b = name;
    while (*a && *b) {
      char ca = *a >= 'A' && *a <= 'Z' ? *a + 32 : *a;
      char cb = *b >= 'A' && *b <= 'Z' ? *b + 32 : *b;
      if (ca != cb) break;
      ++a; ++b;
    }
    if (!*a && !*b) return &e;
  }
  return nullptr;
}

} // namespace lh::alias
