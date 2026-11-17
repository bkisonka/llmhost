// Различение падений по stderr-кольцу: OOM ствола против падения по черновику.
// Кейс 07.10: чужая MTP-голова ловила cudaMalloc OOM («failed to create MTP
// context») — failover винил память ствола и закольцовывался (502-буря).
#include "proc.h"

#include <string>

using lh::proc::looks_oom;
using lh::proc::looks_draft_failure;

// Репродукция TURBO с чужой головой (07.10): OOM-строка соседствует с MTP.
static const char* kTurboTail =
    "llama_model_load_impl: compute buffer size 1.8 GiB\n"
    "CUDA error: out of memory\n"
    "  cudaMalloc failed: out of memory\n"
    "srv_log: failed to create MTP context\n";

// Честная несовместимость головы: GGML_ASSERT без OOM.
static const char* kWidthAssert =
    "GGML_ASSERT: ggml/src/ggml.c:123: MTP input row width must match the "
    "target h_nextn width\n"
    "could not attach MTP module to model\n";

// Чистый OOM ствола: веса не влезли, draft ни при чём.
static const char* kTrunkOom =
    "llama_model_load_impl: loading model took 1321 ms\n"
    "llama_context: unable to allocate CUDA0 buffer\n"
    "llama_context: failed to allocate compute buffers\n";

int main() {
  // looks_oom — как раньше, плюс OOM из MTP-эпизода.
  if (!looks_oom("CUDA error: out of memory")) return 1;
  if (!looks_oom(kTrunkOom)) return 2;
  if (!looks_oom(kTurboTail)) return 3;
  if (looks_oom("loading model took 1321 ms")) return 4;

  // Вина draft — все три формата из репродукции, включая регистр и соседний OOM.
  if (!looks_draft_failure(kTurboTail)) return 10;
  if (!looks_draft_failure(kWidthAssert)) return 11;
  if (!looks_draft_failure("srv_log: failed to load draft model 'mtp-x.gguf'")) return 12;
  if (!looks_draft_failure("draft model load failed")) return 13;

  // Чистый OOM ствола — НЕ вина draft: чернить черновик по нему нельзя.
  if (looks_draft_failure(kTrunkOom)) return 14;
  if (looks_draft_failure("loading model took 1321 ms")) return 15;
  // «mtp» в имени файла/модели — не маркер падения.
  if (looks_draft_failure("loading model mtp-Qwen3.8-27B-Q4_0.gguf")) return 16;
  return 0;
}
