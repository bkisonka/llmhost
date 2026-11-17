// Эвристики каталога: параметры и квант из имени, KV-бакеты, грубая оценка.
#include "catalog_alias.h"
#include "catalog_est.h"

#include <cassert>
#include <string>
#include <vector>
#include <cmath>

using namespace lh::catalog::est;

int main() {
  // Параметры из имени.
  assert(params_from_name("Qwen3-8B-GGUF") == 8e9);
  assert(params_from_name("Llama-3.2-1B-Instruct-GGUF") == 1e9);
  assert(params_from_name("SmolLM2-135M-Instruct-GGUF") == 135e6);
  assert(params_from_name("Qwen_QwQ-32B-GGUF") == 32e9);
  assert(params_from_name("gpt-oss-20b-gguf") == 20e9);
  assert(params_from_name("Mistral-Nemo-12B") == 12e9);
  // MoE: первый (общий) размер.
  assert(params_from_name("Qwen3-235B-A22B-GGUF") == 235e9);
  // Не размеры — не ловим: версии и прочие числа.
  assert(params_from_name("Mistral-7B-Instruct-v0.2-GGUF") == 7e9);
  assert(params_from_name("no-size-here") == 0);
  assert(params_from_name("Gemma-2-2b-it") == 2e9);

  // Квант из имени.
  assert(quant_from_name("Qwen3-8B-Q4_K_M-00001-of-00003.gguf") == "Q4_K_M");
  assert(quant_from_name("model.Q8_0.gguf") == "Q8_0");
  assert(quant_from_name("iq4_xs-something") == "IQ4_XS");
  assert(quant_from_name("plain-name") == "");
  assert(quant_from_name("Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf") == "Q4_K_XL");

  // Спутники: к Q4_K_XL берём Q4-голову, не BF16. Проекция, даже если она F16,
  // едет тоже — другого файла у модели нет.
  {
    std::vector<CompanionFile> files = {
        {"mtp", "MTP/mtp-Q4_K_M.gguf", "Q4_K_M", 2600},
        {"mtp", "MTP/mtp-BF16.gguf", "BF16", 7200},
        {"mmproj", "mmproj-F16.gguf", "F16", 800},
    };
    auto got = pick_companion_filters("Q4_K_XL", files);
    assert(got.size() == 2);
    assert(got[0] == "MTP/mtp-Q4_K_M.gguf");
    assert(got[1] == "mmproj-F16.gguf");
  }
  // Одна голова, пусть и BF16, всё равно качается.
  {
    std::vector<CompanionFile> files = {{"mtp", "MTP/mtp-BF16.gguf", "BF16", 7200}};
    auto got = pick_companion_filters("Q4_K_M", files);
    assert(got.size() == 1);
    assert(got[0] == "MTP/mtp-BF16.gguf");
  }
  // Внутри Q4 точнее более тяжёлый квант: Q4_K_M, не Q4_K_S.
  {
    std::vector<CompanionFile> files = {
        {"mtp", "MTP/mtp-Q4_K_S.gguf", "Q4_K_S", 2000},
        {"mtp", "MTP/mtp-Q4_K_M.gguf", "Q4_K_M", 2400},
    };
    auto got = pick_companion_filters("Q4_K", files);
    assert(got.size() == 1);
    assert(got[0] == "MTP/mtp-Q4_K_M.gguf");
  }

  // Квант-фолбэк и таблица байтов: неизвестный квант считается как Q4_K_M.
  assert(bytes_per_param("") == bytes_per_param("Q4_K_M"));
  assert(bytes_per_param("F16") > bytes_per_param("Q4_K_M"));

  // KV-бакеты монотонны по размеру модели.
  assert(kv_per_token(1e9) > 0);
  assert(kv_per_token(1e9) <= kv_per_token(8e9));
  assert(kv_per_token(8e9) <= kv_per_token(70e9));

  // Грубая оценка: 8B при Q4_K_M — примерно 5 ГБ весов; со светофором согласен.
  Coarse c = coarse("Qwen3-8B-GGUF", 4096, /*cuda=*/true, /*vram_free=*/8ull << 30);
  assert(c.known);
  assert(c.params == 8e9);
  assert(c.weights > 4.2e9 && c.weights < 5.6e9);
  assert(c.kv > 0 && c.overhead > (400ull << 20));
  assert(c.total == c.weights + c.kv + c.overhead);
  // Влезает в 8 ГБ: при Q4_K_M ~4.9 + KV ~0.6 + оверхед ~0.66 ≈ 6.2 ГБ.
  assert(c.fits && !c.partial);
  // В 4 ГБ — частично: веса+оверхед ~5.6 > 4, но на CUDA llama.cpp режет слои
  // по -ngl одинаково для любых моделей — бар 32/4 слоёв набирается.
  Coarse tight = coarse("Qwen3-8B-GGUF", 4096, true, 4ull << 30);
  assert(!tight.fits);
  assert(tight.dense_partial && tight.ngl >= 8 && tight.ngl < 32 && tight.split_ctx >= 4096);
  // Маленькая модель влезает, средняя — частично.
  Coarse small = coarse("SmolLM2-135M-Instruct-GGUF", 4096, true, 8ull << 30);
  assert(small.fits);
  // 27B плотная (слои угаданы как 56, бар 14) на карте с 11,5 ГБ: «частично»,
  // а не «мимо»; VRAM под GPU-часть в пределах свободной.
  Coarse dense = coarse("Qwen3.8-27B-GGUF", 4096, true, 11500ull << 20);
  assert(dense.known && !dense.fits && !dense.partial);
  assert(dense.dense_partial);
  assert(dense.ngl >= 14 && dense.ngl < 56 && dense.split_ctx >= 4096);
  assert(dense.gpu_total <= 11500ull << 20);
  // Тот же случай без CUDA — поведение прежнее: «мимо».
  assert(!coarse("Qwen3.8-27B-GGUF", 4096, false, 11500ull << 20).dense_partial);
  // Неизвестное имя — не угадываем.
  assert(!coarse("some-random-repo", 4096, true, 8ull << 30).known);
  // Без GPU нет вердиктов.
  Coarse nogpu = coarse("Qwen3-8B-GGUF", 4096, false, 0);
  assert(nogpu.known && !nogpu.fits && !nogpu.partial);

  std::puts("catalog_test: ok");

  // Таблица коротких имён: каждое находится (в любом регистре), мусор — нет,
  // репозитории в формате org/repo и без повторов имён.
  {
    assert(lh::alias::find("qwen3") != nullptr);
    assert(lh::alias::find("QWEN3") != nullptr);
    assert(lh::alias::find("Gemma-4") != nullptr);
    assert(lh::alias::find("") == nullptr);
    assert(lh::alias::find("no-such-model") == nullptr);
    for (const lh::alias::Entry& e : lh::alias::kAll) {
      assert(lh::alias::find(e.name) == &e);            // сама себя находит
      assert(std::string(e.repo).find('/') != std::string::npos); // org/repo
      assert(std::string(e.name).find('/') == std::string::npos); // имя — одно слово
      assert(e.note && *e.note);
    }
  }
    return 0;
}
