// Чтение заголовка GGUF: только метаданные до тензоров.
// Массивы (например, токенизатор на сотни тысяч строк) пропускаются seek'ом —
// разбор даже большой модели занимает миллисекунды.
#pragma once
#include <string>
#include <cstdint>
#include <vector>

namespace lh::gguf {

struct Meta {
  std::string name;        // general.name, пусто — возьмём имя файла
  std::string arch;        // general.architecture (llama, qwen2, …)
  std::string quant;       // человекочитаемый квант из general.file_type
  uint32_t file_type = 0;
  uint64_t n_layers = 0;
  uint64_t n_ctx_train = 0;
  uint64_t n_embd = 0;
  uint64_t n_head = 0;
  uint64_t n_head_kv = 0;
  uint64_t n_experts = 0;       // MoE: всего экспертов (0 — плотная модель)
  uint64_t n_experts_used = 0;  // MoE: активных на токен
  // Суммарный размер тензоров-экспертов (имена с «exps» — тот же подстрочный
  // паттерн, по которому -ot exps=CPU увозит их в RAM). Читается из тензорного
  // списка; 0 — неизвестно (списка нет / не дочитали / модель не MoE).
  uint64_t expert_bytes = 0;
  // Те же байты по блокам — ровно те тензоры, что двигает --n-cpu-moe N
  // апстрима (blk.N.ffn_(up|down|gate|gate_up)_(ch|)exps). Индекс вектора —
  // номер блока; блоки без экспертных тензоров (плотные у first-k-dense
  // семей) имеют 0, поэтому сумма первых N — точный вес выноса. nextn-тензоры
  // (blk.N.nextn.ffn_*) сюда не попадают: их регэксп ncmoe не двигает
  // (полный вынос exps=CPU берёт, частичный — нет). Пусто — раскладка
  // неизвестна (обход сорвался / список не читался): частичный сплит не
  // планируем, чтобы не рисковать неточной математикой.
  std::vector<uint64_t> expert_bytes_by_block;
  uint64_t nextn_predict_layers = 0;  // встроенный MTP: <arch>.nextn_predict_layers > 0
  // SWA (sliding-window attention): окно токенов и число локальных (оконных)
  // слоёв. Окно — <arch>.attention.sliding_window; локальные — из
  // sliding_window_pattern (массив флагов по слою или скаляр-«каждый N-й»).
  // llama4 ключей не несёт вовсе — выводится при разборе (окно 8192, каждый
  // 4-й слой глобальный), как это делает сам движок. У SWA-моделей классы
  // слоёв различаются и размером головы, и числом KV-голов (gemma4:
  // локальные 8×256, глобальные 2×512). 0 — знаний нет (слепая
  // консервативная аналитика).
  uint64_t n_swa = 0;
  uint64_t n_swa_layers = 0;
  uint64_t n_embd_head_k = 0;      // <arch>.attention.key_length (глобальные слои)
  uint64_t n_embd_head_k_swa = 0;  // <arch>.attention.key_length_swa
  uint64_t n_head_kv_local = 0;    // KV-головы локальных слоёв (если однородны)
  uint64_t n_head_kv_global = 0;   // KV-головы глобальных слоёв (если однородны)
  uint64_t n_vocab = 0;            // <arch>.vocab_size — для оценки логит-буфера
                                   // (ubatch × vocab × 4 байта) в планах впритык
  // Гибридность из меты — для арок, которых таблица семейств в fit.h не знает
  // (qwen4exp и будущие): токенный KV на каждом N-м слое (полный attention),
  // остальные слои — рекуррентные, с состоянием постоянного размера.
  uint32_t full_attn_interval = 0;  // <arch>.full_attention_interval, 0 — нет ключа
  bool has_ssm = false;             // <arch>.ssm.* присутствуют (mamba/GDN-слои)
  // Тензоры вне GPU-бюджета: per_layer_token_embd (n-gram PLE-таблицы qwen4exp,
  // четверть файла) — движок на карту их не берёт, читается из тензорного
  // списка; вычитается из GPU-части весов, иначе план считает их картой.
  uint64_t cpu_extra_bytes = 0;
};

bool parse_header(const std::string& path, Meta& out, std::string& err);
// То же из буфера (первые N байт удалённого .gguf, скачанные Range-запросом).
// Толерантно к обрезке: если нужные поля уже собраны, а кончились данные
// посреди токенизатора — успех.
bool parse_header_mem(const char* data, size_t len, Meta& out, std::string& err);
const char* quant_name(uint32_t file_type);

} // namespace lh::gguf
