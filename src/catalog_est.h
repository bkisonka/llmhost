// Предиктивная оценка «влезет ли» для моделей, которых ещё нет на диске.
// Из репозитория на HuggingFace известны только имя, размеры файлов и — после
// Range-выборки — настоящий заголовок GGUF. Пока заголовка нет, размеры модели
// угадываются по имени («Qwen3-8B» → 8 млрд параметров), KV-кэш — по бакету
// размеров (типовые формы семейств llama/qwen/mistral). Это вилка для светофора
// в списке поиска; точная прикидка — fit::estimate по реальному заголовку.
// Чистые функции без зависимостей — отдельный тест.
#pragma once
#include "fit.h" // dense_min_layers — бар общий с fit::dense_split

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace lh::catalog::est {

// Параметры из имени: «Qwen3-8B» → 8e9, «SmolLM2-135M» → 1.35e8, «Llama-3.2-1B»
// → 1e9. 0 — в имени размера нет. Для MoE («235B-A22B») берём первый — общий.
inline double params_from_name(const std::string& s) {
  for (size_t i = 0; i < s.size(); ++i) {
    if (!std::isdigit((unsigned char)s[i])) continue;
    double val = 0;
    size_t j = i;
    for (; j < s.size() && std::isdigit((unsigned char)s[j]); ++j) val = val * 10 + (s[j] - '0');
    if (j < s.size() && s[j] == '.' && j + 1 < s.size() && std::isdigit((unsigned char)s[j + 1])) {
      ++j;
      double frac = 0.1;
      for (; j < s.size() && std::isdigit((unsigned char)s[j]); ++j, frac /= 10)
        val += (s[j] - '0') * frac;
    }
    if (j >= s.size()) break;
    char unit = (char)((s[j] >= 'a' && s[j] <= 'z') ? s[j] - 32 : s[j]);
    bool next_word = j + 1 < s.size() && std::isalnum((unsigned char)s[j + 1]);
    if (unit == 'B' && !next_word) return val * 1e9;
    if (unit == 'M' && !next_word) return val * 1e6;
    i = j; // не размер («3.2-1B», «v0.2») — ищем дальше
  }
  return 0;
}

// Квант из имени файла/репозитория: «...-Q4_K_M.gguf» → «Q4_K_M». Порядок — от
// длинных паттернов к коротким, чтобы «Q4_K_M» не съелся «Q4_K».
inline std::string quant_from_name(const std::string& name_in) {
  static const char* const quants[] = {
      "Q8_K_XL", "Q8_0", "Q6_K_XL", "Q6_K", "Q5_K_XL", "Q5_K_M", "Q5_K_S", "Q5_K", "Q5_0",
      "Q5_1", "Q4_K_XL", "Q4_K_M", "Q4_K_S", "Q4_K", "Q4_0", "Q4_1", "Q3_K_XL", "Q3_K_L",
      "Q3_K_M", "Q3_K_S", "Q2_K_XL", "Q2_K_S", "Q2_K",
      "IQ4_XS", "IQ4_NL", "IQ3_XS", "IQ3_XXS", "IQ3_S", "IQ3_M", "IQ2_S", "IQ2_M",
      "IQ2_XS", "IQ2_XXS", "IQ1_S", "IQ1_M", "MXFP4", "BF16", "F16", "F32", "TQ1_0", "TQ2_0",
  };
  std::string low = name_in;
  for (char& c : low) c = (char)std::tolower((unsigned char)c);
  // Q4_K_XL тоже бывает (в iq-квантах Q8_0 у эмбеддингов) — XL добавлен выше.
  for (const char* q : quants) {
    std::string qlow = q;
    for (char& c : qlow) c = (char)std::tolower((unsigned char)c);
    if (low.find(qlow) != std::string::npos) return q;
  }
  return {};
}

// Байтов на параметр по кванту (средние по замерам файлов llama.cpp).
inline double bytes_per_param(const std::string& quant) {
  if (quant == "F32") return 4.0;
  if (quant == "BF16" || quant == "F16") return 2.0;
  if (quant == "Q8_0") return 1.06;
  if (quant == "Q6_K") return 0.82;
  if (quant == "Q5_K_M" || quant == "Q5_K_S" || quant == "Q5_K" || quant == "Q5_0" ||
      quant == "Q5_1")
    return 0.69;
  if (quant == "Q4_K_M" || quant == "Q4_K" || quant == "Q4_0" || quant == "Q4_1") return 0.61;
  if (quant == "Q4_K_S" || quant == "IQ4_XS" || quant == "IQ4_NL") return 0.57;
  if (quant == "Q3_K_L" || quant == "Q3_K_XL") return 0.57;
  if (quant == "Q3_K_M" || quant == "IQ3_M") return 0.53;
  if (quant == "MXFP4") return 0.53;
  if (quant == "Q3_K_S" || quant == "IQ3_S") return 0.50;
  if (quant == "IQ3_XS") return 0.48;
  if (quant == "IQ3_XXS") return 0.44;
  if (quant == "Q2_K" || quant == "IQ2_M") return 0.41;
  if (quant == "Q2_K_S" || quant == "IQ2_S") return 0.39;
  if (quant == "IQ2_XS") return 0.37;
  if (quant == "IQ2_XXS") return 0.33;
  if (quant == "IQ1_M") return 0.30;
  if (quant == "IQ1_S") return 0.27;
  return 0.61; // в имени кванта нет — считаем как Q4_K_M
}

// KV-байт на токен по бакету параметров: 2 (K+V) * n_layers * n_embd_kv * 2 байта
// для типовых форм. Сознательно с запасом: светофор не должен обещать лишнего.
inline uint64_t kv_per_token(double params) {
  if (params <= 0) return 0;
  if (params <= 1e9) return 131072;
  if (params <= 4e9) return 131072;
  if (params <= 10e9) return 147456;
  if (params <= 20e9) return 180224;
  if (params <= 40e9) return 286720;
  if (params <= 80e9) return 393216;
  return 524288;
}

// Слои по параметрам для плотных семейств llama/qwen/mistral/gemma — заведомо
// НЕ БОЛЬШЕ реальных (у qwen3-8B их 36, у 27B — 62, у 70B — 80): чем меньше
// угаданное число слоёв, тем крупнее «слой» и консервативнее вердикт.
inline int layers_guess(double params) {
  if (params <= 1e9) return 24;
  if (params <= 8e9) return 32;
  if (params <= 14e9) return 40;
  if (params <= 32e9) return 56;
  if (params <= 80e9) return 72;
  return 88;
}

// Грубая проверка частичного сплита плотной модели (заголовка нет): максимум
// слоёв на GPU при контексте не меньше ctx_min, KV — по бакету параметров.
// Возвращает false, если даже бар fit::dense_min_layers не обеспечивает
// минимальный контекст. Поднимает: ngl_out/layers_out/ctx_out/gpu_need_out.
inline bool dense_partial_coarse(double params, uint64_t weights, bool cuda,
                                 uint64_t vram_free, bool kv_q8, uint32_t ctx_min,
                                 int& ngl_out, int& layers_out, uint32_t& ctx_out,
                                 uint64_t& gpu_need_out) {
  ngl_out = 0; layers_out = 0; ctx_out = 0; gpu_need_out = 0;
  if (!cuda || params <= 0 || !weights || !vram_free) return false;
  if (ctx_min < 512) ctx_min = 512;
  layers_out = layers_guess(params);
  const uint64_t kv_tok = kv_per_token(params) / (kv_q8 ? 2 : 1);
  const uint64_t ovh = 400ull << 20; // вызывается только при cuda
  for (int n = layers_out; n >= fit::dense_min_layers(layers_out); --n) {
    uint64_t w = (uint64_t)((double)weights * n / layers_out);
    for (uint32_t c = 8192u;; c >>= 1) { // заголовка нет — потолок как у fit_json
      uint64_t need = w + kv_tok * c + ovh + (uint64_t)c * 65536;
      if (need <= vram_free) {
        if (c >= ctx_min) {
          ngl_out = n;
          ctx_out = c;
          gpu_need_out = need;
          return true;
        }
        break;
      }
      if (c <= 512) break;
    }
  }
  return false;
}

struct Coarse {
  bool known = false;      // параметры угаданы по имени
  double params = 0;       // млрд — как вышло из имени
  uint64_t weights = 0, kv = 0, overhead = 0, total = 0;
  bool fits = false;       // целиком в свободную VRAM
  bool partial = false;    // веса влезают, целиком нет — попробуют меньший ctx
  bool dense_partial = false; // CUDA: часть слоёв на GPU, хвост в RAM — грузить можно
  int ngl = 0, layers = 0;    // для dense_partial: сколько из скольких слоёв на GPU
  uint32_t split_ctx = 0;     // контекст, который влезает при этом сплите
  uint64_t gpu_total = 0;     // VRAM под GPU-часть при dense_partial
};

// Оценка по имени репозитория. cuda — оверхед бэкенда выше (см. fit.cpp);
// kv_q8 — умная настройка включила квантованный KV-кэш (вдвое меньше).
inline Coarse coarse(const std::string& name, uint32_t ctx, bool cuda, uint64_t vram_free,
                     bool kv_q8 = false) {
  Coarse c;
  c.params = params_from_name(name);
  if (c.params <= 0) return c;
  c.known = true;
  c.weights = (uint64_t)(c.params * bytes_per_param(quant_from_name(name)));
  c.kv = kv_per_token(c.params) * ctx / (kv_q8 ? 2 : 1);
  c.overhead = ((cuda ? 400ull : 90ull) << 20) + (uint64_t)ctx * 65536;
  c.total = c.weights + c.kv + c.overhead;
  if (vram_free > 0) {
    c.fits = c.total <= vram_free;
    c.partial = !c.fits && c.weights + c.overhead <= vram_free;
    // Плотная крупнее VRAM: на CUDA честнее «частично, часть слоёв в RAM»,
    // чем «мимо» — llama.cpp режет слои по -ngl одинаково для любых моделей.
    int ngl, layers;
    uint32_t sctx;
    uint64_t need;
    if (!c.fits && !c.partial &&
        dense_partial_coarse(c.params, c.weights, cuda, vram_free, kv_q8,
                             ctx >= 512 ? ctx : 4096, ngl, layers, sctx, need)) {
      c.dense_partial = true;
      c.ngl = ngl;
      c.layers = layers;
      c.split_ctx = sctx;
      c.gpu_total = need;
    }
  }
  return c;
}


// Файл-спутник репозитория: MTP-голова или vision-проекция.
struct CompanionFile {
  std::string kind;    // "mtp" или "mmproj"
  std::string filter;  // что передать в pull
  std::string quant;   // quant_from_name пути
  uint64_t size = 0;
};

inline bool quant_is_float(const std::string& q) {
  return q == "BF16" || q == "F16" || q == "F32";
}

inline int quant_bits(const std::string& q) {
  for (char c : q)
    if (c >= '0' && c <= '9') return c - '0';
  return quant_is_float(q) ? 16 : 0;
}

// Q4_K_XL и Q4_K_M — одна семья Q4. IQ4_XS — IQ4. BF16 остаётся собой.
inline std::string quant_family(const std::string& q) {
  if (q.size() >= 3 && q[0] == 'I' && q[1] == 'Q' && q[2] >= '0' && q[2] <= '9')
    return q.substr(0, 3);
  if (quant_is_float(q)) return q;
  if (q.size() >= 2 && q[0] == 'Q' && q[1] >= '0' && q[1] <= '9') return q.substr(0, 2);
  return q;
}

inline int companion_rank(const std::string& model_q, const std::string& cand_q) {
  if (model_q.empty()) return quant_is_float(cand_q) ? 10 : 40;
  if (!cand_q.empty() && cand_q == model_q) return 100;
  if (!cand_q.empty() && quant_family(model_q) == quant_family(cand_q)) return 80;
  if (quant_is_float(cand_q) && !quant_is_float(model_q)) return 10;
  return 30;
}

// По одному файлу каждого вида. Тот же квант, затем та же семья (Q4 к Q4),
// иначе ближайшая разрядность. Полный BF16 не выигрывает у кванта модели.
inline std::vector<std::string> pick_companion_filters(const std::string& model_quant,
                                                      const std::vector<CompanionFile>& files) {
  std::vector<std::string> out;
  for (const char* kind : {"mtp", "mmproj"}) {
    const CompanionFile* best = nullptr;
    for (auto& f : files) {
      if (f.kind != kind || f.filter.empty()) continue;
      if (!best) {
        best = &f;
        continue;
      }
      int rb = companion_rank(model_quant, best->quant);
      int rf = companion_rank(model_quant, f.quant);
      if (rf != rb) {
        if (rf > rb) best = &f;
        continue;
      }
      if (!model_quant.empty()) {
        int db = std::abs(quant_bits(best->quant) - quant_bits(model_quant));
        int df = std::abs(quant_bits(f.quant) - quant_bits(model_quant));
        if (df != db) {
          if (df < db) best = &f;
          continue;
        }
        double bb = bytes_per_param(best->quant);
        double bf = bytes_per_param(f.quant);
        if (bf != bb) {
          if (bf > bb) best = &f;
          continue;
        }
      }
      if (f.size < best->size) best = &f;
    }
    if (best) out.push_back(best->filter);
  }
  return out;
}

} // namespace lh::catalog::est
