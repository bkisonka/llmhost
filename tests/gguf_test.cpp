// Тест парсера GGUF: файл собирается из байтов прямо в тесте —
// бинарных фикстур в репозитории нет.
#include "gguf.h"
#include "util.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <filesystem>

using namespace lh;

static void put(std::vector<uint8_t>& b, const void* p, size_t n) {
  const uint8_t* s = (const uint8_t*)p;
  b.insert(b.end(), s, s + n);
}
static void put_u32(std::vector<uint8_t>& b, uint32_t v) { put(b, &v, 4); }
static void put_u64(std::vector<uint8_t>& b, uint64_t v) { put(b, &v, 8); }
static void put_str(std::vector<uint8_t>& b, const std::string& s) {
  put_u64(b, s.size());
  put(b, s.data(), s.size());
}

static void put_kv_string(std::vector<uint8_t>& b, const std::string& key, const std::string& val) {
  put_str(b, key);
  put_u32(b, 8); // T_STRING
  put_str(b, val);
}
static void put_kv_u32(std::vector<uint8_t>& b, const std::string& key, uint32_t v) {
  put_str(b, key);
  put_u32(b, 4); // T_UINT32
  put_u32(b, v);
}
static void put_kv_u64(std::vector<uint8_t>& b, const std::string& key, uint64_t v) {
  put_str(b, key);
  put_u32(b, 10); // T_UINT64
  put_u64(b, v);
}
static void put_kv_str_array(std::vector<uint8_t>& b, const std::string& key,
                             const std::vector<std::string>& items) {
  put_str(b, key);
  put_u32(b, 9); // T_ARRAY
  put_u32(b, 8); // из строк
  put_u64(b, items.size());
  for (auto& s : items) put_str(b, s);
}
static void put_kv_f32_array(std::vector<uint8_t>& b, const std::string& key, size_t count) {
  put_str(b, key);
  put_u32(b, 9);   // T_ARRAY
  put_u32(b, 6);   // из f32
  put_u64(b, count);
  b.insert(b.end(), count * 4, 0xAB);
}
static void put_kv_bool_array(std::vector<uint8_t>& b, const std::string& key,
                              const std::vector<uint8_t>& items) {
  put_str(b, key);
  put_u32(b, 9); // T_ARRAY
  put_u32(b, 7); // из bool
  put_u64(b, items.size());
  for (uint8_t v : items) b.push_back(v);
}
static void put_kv_u32_array(std::vector<uint8_t>& b, const std::string& key,
                             const std::vector<uint32_t>& items) {
  put_str(b, key);
  put_u32(b, 9); // T_ARRAY
  put_u32(b, 4); // из u32
  put_u64(b, items.size());
  for (uint32_t v : items) put_u32(b, v);
}

static void put_tensor(std::vector<uint8_t>& b, const std::string& name, uint64_t offset) {
  put_str(b, name);
  put_u32(b, 2);            // n_dims
  put_u64(b, 256);          // dims[0]
  put_u64(b, 1);            // dims[1] — форма не участвует, размер берём из офсетов
  put_u32(b, 0);            // тип F32
  put_u64(b, offset);
}

static std::string write_gguf(const std::vector<uint8_t>& bytes) {
  std::string path =
      (std::filesystem::temp_directory_path() / "llmhost_gguf_test.gguf").string();
  FILE* f = fopen(path.c_str(), "wb");
  assert(f);
  fwrite(bytes.data(), 1, bytes.size(), f);
  fclose(f);
  return path;
}

int main() {
  // Проверка quant_name на устойчивость к произвольным значениям — вызов, который
  // не должен падать ни на каком file_type.
  (void)gguf::quant_name(999);

  std::vector<uint8_t> b;
  put(b, "GGUF", 4);
  put_u32(b, 3);        // версия
  put_u64(b, 0);        // тензоров нет — нас интересует только заголовок
  put_u64(b, 10);       // kv-пар
  put_kv_string(b, "general.architecture", "llama");
  put_kv_string(b, "general.name", "TestModel");
  put_kv_u32(b, "general.file_type", 15);          // Q4_K_M
  put_kv_u32(b, "llama.block_count", 32);
  put_kv_u32(b, "llama.context_length", 8192);
  put_kv_u32(b, "llama.embedding_length", 4096);
  put_kv_u32(b, "llama.attention.head_count", 32);
  put_kv_u32(b, "llama.attention.head_count_kv", 8);
  put_kv_u32(b, "llama.expert_count", 8);          // MoE-метаданные
  put_kv_u32(b, "llama.expert_used_count", 2);
  // Массивы — большой токенизатор и табличка оценок: парсер обязан их перепрыгнуть.
  put_kv_str_array(b, "tokenizer.ggml.tokens", {"a", "bb", "ccc"});
  put_kv_f32_array(b, "tokenizer.ggml.scores", 2);
  put_kv_f32_array(b, "tokenizer.ggml.token_type", 100000);

  std::string path = write_gguf(b);
  gguf::Meta m;
  std::string err;
  bool ok = gguf::parse_header(path, m, err);
  assert(ok && err.empty());
  assert(m.name == "TestModel");
  assert(m.arch == "llama");
  assert(m.n_layers == 32);
  assert(m.n_ctx_train == 8192);
  assert(m.n_embd == 4096);
  assert(m.n_head == 32);
  assert(m.n_head_kv == 8);
  assert(m.n_experts == 8);
  assert(m.n_experts_used == 2);
  assert(std::string(m.quant) == "Q4_K_M");
  assert(m.expert_bytes == 0); // тензорного списка нет — экспертные байты неизвестны
  std::filesystem::remove(path);

  // SWA-метаданные gemma4: окно скаляром, паттерн — булев массив по слою,
  // KV-головы — массивом по слою (первый элемент — fallback-скаляр),
  // размерности голов — скалярами. Корреляция: глобальные слои (False в
  // паттерне) — 2 KV-головы, локальные — 8. Числа — как у реальных файлов.
  {
    std::vector<uint8_t> g;
    put(g, "GGUF", 4);
    put_u32(g, 3);
    put_u64(g, 0);      // тензоров нет
    put_u64(g, 10);     // kv-пар
    put_kv_string(g, "general.architecture", "gemma4");
    put_kv_u32(g, "gemma4.block_count", 6);
    put_kv_u32(g, "gemma4.context_length", 262144);
    put_kv_u32(g, "gemma4.embedding_length", 2816);
    put_kv_u32(g, "gemma4.attention.head_count", 16);
    put_kv_u32(g, "gemma4.attention.key_length", 512);
    put_kv_u32(g, "gemma4.attention.key_length_swa", 256);
    put_kv_u32(g, "gemma4.attention.sliding_window", 1024);
    put_kv_u32_array(g, "gemma4.attention.head_count_kv", {8, 8, 8, 8, 8, 2});
    put_kv_bool_array(g, "gemma4.attention.sliding_window_pattern", {1, 1, 1, 1, 1, 0});
    std::string gp = write_gguf(g);
    gguf::Meta gm;
    ok = gguf::parse_header(gp, gm, err);
    assert(ok && err.empty());
    assert(gm.arch == "gemma4" && gm.n_layers == 6 && gm.n_ctx_train == 262144);
    assert(gm.n_swa == 1024 && gm.n_swa_layers == 5);
    assert(gm.n_embd_head_k == 512 && gm.n_embd_head_k_swa == 256);
    assert(gm.n_head_kv == 8);
    assert(gm.n_head_kv_local == 8 && gm.n_head_kv_global == 2);
    // Разбор из памяти (Range-выборка) — те же поля.
    gguf::Meta gmem;
    ok = gguf::parse_header_mem((const char*)g.data(), g.size(), gmem, err);
    assert(ok && gmem.n_swa_layers == 5 && gmem.n_head_kv_global == 2);
    std::filesystem::remove(gp);
  }

  // llama4: SWA-ключей в GGUF нет вовсе — движок выводит их для арки сам
  // (окно 8192, каждый 4-й слой глобальный, iswa-кэш), выводим и мы.
  // Числа — с реального unsloth Llama-4-Scout-17B: 48 слоёв, KV-головы 8,
  // голова 128.
  {
    std::vector<uint8_t> l;
    put(l, "GGUF", 4);
    put_u32(l, 3);
    put_u64(l, 0);    // тензоров нет
    put_u64(l, 6);    // kv-пар
    put_kv_string(l, "general.architecture", "llama4");
    put_kv_u32(l, "llama4.block_count", 48);
    put_kv_u32(l, "llama4.embedding_length", 5120);
    put_kv_u32(l, "llama4.attention.head_count", 40);
    put_kv_u32(l, "llama4.attention.head_count_kv", 8);
    put_kv_u32(l, "llama4.attention.key_length", 128);
    std::string lp = write_gguf(l);
    gguf::Meta lm;
    ok = gguf::parse_header(lp, lm, err);
    assert(ok && err.empty());
    assert(lm.arch == "llama4" && lm.n_layers == 48);
    assert(lm.n_swa == 8192 && lm.n_swa_layers == 36);
    assert(lm.n_head_kv == 8 && lm.n_embd_head_k == 128);
    // Разбор из памяти — тот же вывод.
    gguf::Meta lmem;
    ok = gguf::parse_header_mem((const char*)l.data(), l.size(), lmem, err);
    assert(ok && lmem.n_swa == 8192 && lmem.n_swa_layers == 36);
    std::filesystem::remove(lp);
  }

  // Не-GGUF файл.
  std::vector<uint8_t> junk = {'n', 'o', 'p', 'e', 1, 2, 3, 4};
  std::string junk_path = write_gguf(junk);
  ok = gguf::parse_header(junk_path, m, err);
  assert(!ok);
  assert(err.find("GGUF") != std::string::npos);
  std::filesystem::remove(junk_path);

  // Обрезанный файл: магия есть, заголовка нет.
  std::vector<uint8_t> cut = {'G', 'G', 'U', 'F', 3, 0, 0, 0};
  std::string cut_path = write_gguf(cut);
  ok = gguf::parse_header(cut_path, m, err);
  assert(!ok);
  std::filesystem::remove(cut_path);

  // Разбор из памяти: полный буфер — как файл.
  gguf::Meta mem;
  ok = gguf::parse_header_mem((const char*)b.data(), b.size(), mem, err);
  assert(ok && err.empty());
  assert(mem.name == "TestModel" && mem.n_layers == 32 && mem.n_head_kv == 8);

  // Урезанный буфер (как Range-выборка удалённого .gguf): главное собрано — успех,
  // даже если данные кончились посреди токенизатора. 300 байт — середина kv-поля
  // head_count_kv: слои и embd уже разобраны.
  std::vector<uint8_t> half(b.begin(), b.begin() + 300);
  ok = gguf::parse_header_mem((const char*)half.data(), half.size(), mem, err);
  assert(ok && mem.n_layers == 32 && mem.n_embd == 4096);

  // Совсем пустой хвост: главное не собрано — ошибка.
  std::vector<uint8_t> tail(b.begin(), b.begin() + 40);
  ok = gguf::parse_header_mem((const char*)tail.data(), tail.size(), mem, err);
  assert(!ok);

  // Тензорный список: размер тензора — разница офсетов соседей, экспертные —
  // имена с «exps» (тот же паттерн, по которому -ot exps=CPU увозит их в RAM).
  {
    auto header = [](uint64_t tensors) {
      std::vector<uint8_t> t;
      put(t, "GGUF", 4);
      put_u32(t, 3);
      put_u64(t, tensors);
      put_u64(t, 2);
      put_kv_u32(t, "llama.block_count", 4);
      put_kv_string(t, "general.architecture", "llama");
      return t;
    };
    // Обычный MoE: expert-тензор внутри блока, файл кончается output.weight.
    std::vector<uint8_t> t = header(4);
    put_tensor(t, "token_embd.weight", 0);
    put_tensor(t, "blk.0.attn_q.weight", 1024);
    put_tensor(t, "blk.0.ffn_down_exps.weight", 2048);
    put_tensor(t, "output.weight", 4096);
    std::string tpath = write_gguf(t);
    gguf::Meta tm;
    ok = gguf::parse_header(tpath, tm, err);
    assert(ok && err.empty());
    assert(tm.expert_bytes == 4096 - 2048); // один экспертный тензор: спан 2048 байт
    assert(tm.expert_bytes_by_block.size() == 1 && tm.expert_bytes_by_block[0] == 2048);
    std::filesystem::remove(tpath);

    // Офсеты в списке не по порядку — сортировка восстанавливает спаны.
    t = header(3);
    put_tensor(t, "blk.0.attn_q.weight", 2048); // высший офсет — не экспертный
    put_tensor(t, "token_embd.weight", 0);
    put_tensor(t, "blk.0.ffn_up_exps.weight", 1024);
    tpath = write_gguf(t);
    ok = gguf::parse_header(tpath, tm, err);
    assert(ok && tm.expert_bytes == 2048 - 1024);
    std::filesystem::remove(tpath);

    // Шаред-эксперты («shexp») не содержат «exps» — остаются на GPU, в сумму
    // не попадают; два экспертных тензора суммируются.
    t = header(5);
    put_tensor(t, "token_embd.weight", 0);
    put_tensor(t, "blk.0.ffn_gate_shexp.weight", 1024);
    put_tensor(t, "blk.0.ffn_up_exps.weight", 2048);
    put_tensor(t, "blk.0.ffn_down_exps.weight", 4096);
    put_tensor(t, "output.weight", 6144);
    tpath = write_gguf(t);
    ok = gguf::parse_header(tpath, tm, err);
    assert(ok && tm.expert_bytes == 6144 - 2048);
    assert(tm.expert_bytes_by_block.size() == 1 &&
           tm.expert_bytes_by_block[0] == 6144 - 2048);
    std::filesystem::remove(tpath);

    // Поблочная раскладка (--n-cpu-moe): плотный первый блок (нечего выносить),
    // сплит-регэксп различает варианты имён; nextn-тензоры (blk.N.nextn.ffn_*)
    // в поблочный вес не попадают — их ncmoe не двигает, только полный вынос.
    t = header(8);
    put_tensor(t, "token_embd.weight", 0);
    put_tensor(t, "blk.0.attn_q.weight", 1024);        // плотный блок: attn без exps
    put_tensor(t, "blk.0.ffn_down.weight", 2048);      // …и плотный ffn (без exps)
    put_tensor(t, "blk.1.ffn_gate_up_exps.weight", 4096); // сплётенный gate_up
    put_tensor(t, "blk.2.ffn_gate_exps.weight", 8192);
    put_tensor(t, "blk.2.ffn_down_chexps.weight", 16384); // вариант «ch»
    put_tensor(t, "blk.3.nextn.ffn_gate_exps.weight", 32768); // nextn — не для ncmoe
    put_tensor(t, "output.weight", 65536);
    tpath = write_gguf(t);
    ok = gguf::parse_header(tpath, tm, err);
    assert(ok);
    // Полный вынос (exps=CPU) берёт и nextn: четыре спана подряд 4096…65536.
    assert(tm.expert_bytes == 65536 - 4096);
    assert(tm.expert_bytes_by_block.size() == 3);
    assert(tm.expert_bytes_by_block[0] == 0);            // плотный блок пуст
    assert(tm.expert_bytes_by_block[1] == 8192 - 4096);  // gate_up_exps
    assert(tm.expert_bytes_by_block[2] == 32768 - 8192); // gate_exps + down_chexps
    std::filesystem::remove(tpath);

    // Хвост файла — экспертный тензор: размерится концом файла минус офсет
    // от начала секции данных (нужны и размер файла, и general.alignment).
    t = header(2);
    put_tensor(t, "token_embd.weight", 0);
    put_tensor(t, "blk.0.ffn_up_exps.weight", 1024);
    {
      const uint64_t list_end = t.size();
      const uint64_t data_start = (list_end + 31) / 32 * 32; // выравнивание 32
      t.insert(t.end(), (size_t)(data_start - list_end) + 2048 + 512, 0xAA); // «данные»
      tpath = write_gguf(t);
      ok = gguf::parse_header(tpath, tm, err);
      assert(ok && tm.expert_bytes == t.size() - data_start - 1024);
      std::filesystem::remove(tpath);
    }
    // ...а в выборке из памяти офсет конца неизвестен — замер сдаётся (0).
    t = header(2);
    put_tensor(t, "token_embd.weight", 0);
    put_tensor(t, "blk.0.ffn_up_exps.weight", 1024);
    tpath = write_gguf(t);
    ok = gguf::parse_header(tpath, tm, err);
    assert(ok && tm.expert_bytes == 0);
    std::filesystem::remove(tpath);

    // Плотная модель: экспертных имён нет — ноль (не MoE).
    t = header(2);
    put_tensor(t, "token_embd.weight", 0);
    put_tensor(t, "output.weight", 1024);
    tpath = write_gguf(t);
    ok = gguf::parse_header(tpath, tm, err);
    assert(ok && tm.expert_bytes == 0);
    std::filesystem::remove(tpath);

    // Урезанная выборка памяти, обрывающаяся посреди списка тензоров: заголовок
    // собран — успех, экспертные байты не дочитаны (0).
    t = header(4);
    put_tensor(t, "token_embd.weight", 0);
    put_tensor(t, "blk.0.attn_q.weight", 1024);
    put_tensor(t, "blk.0.ffn_down_exps.weight", 2048);
    put_tensor(t, "output.weight", 4096);
    std::vector<uint8_t> cut2(t.begin(), t.begin() + (t.size() - 20));
    ok = gguf::parse_header_mem((const char*)cut2.data(), cut2.size(), tm, err);
    assert(ok && tm.n_layers == 4 && tm.expert_bytes == 0);
  }

  // Гибридность из меты: интервал токенного KV и ssm-ключ.
  {
    std::vector<uint8_t> h;
    put(h, "GGUF", 4);
    put_u32(h, 3);
    put_u64(h, 0);
    put_u64(h, 4);
    put_kv_string(h, "general.architecture", "qwen4exp");
    put_kv_u32(h, "qwen4exp.block_count", 48);
    put_kv_u32(h, "qwen4exp.full_attention_interval", 4);
    put_kv_u32(h, "qwen4exp.ssm.state_size", 128);
    std::string hp = write_gguf(h);
    gguf::Meta hm;
    ok = gguf::parse_header(hp, hm, err);
    assert(ok && hm.arch == "qwen4exp" && hm.n_layers == 48);
    assert(hm.full_attn_interval == 4 && hm.has_ssm);
    std::filesystem::remove(hp);
  }

  // CPU-тензоры (per_layer_token_embd): сумма спанов в cpu_extra_bytes,
  // экспертная сумма не задета; хвост-файл — CPU-тензор размерится концом.
  {
    auto hdr = [](uint64_t tensors) {
      std::vector<uint8_t> t;
      put(t, "GGUF", 4);
      put_u32(t, 3);
      put_u64(t, tensors);
      put_u64(t, 1);
      put_kv_string(t, "general.architecture", "qwen4exp");
      return t;
    };
    // per_layer_token_embd в середине: спан 2048, эксперты не тронуты.
    std::vector<uint8_t> t = hdr(4);
    put_tensor(t, "token_embd.weight", 0);
    put_tensor(t, "per_layer_token_embd.weight", 1024);
    put_tensor(t, "blk.0.ffn_up_exps.weight", 3072);
    put_tensor(t, "output.weight", 6144);
    std::string tp = write_gguf(t);
    gguf::Meta tm2;
    ok = gguf::parse_header(tp, tm2, err);
    assert(ok && tm2.cpu_extra_bytes == 2048);
    assert(tm2.expert_bytes == 6144 - 3072);
    std::filesystem::remove(tp);
    // Хвост — CPU-тензор: размерится концом файла.
    t = hdr(2);
    put_tensor(t, "token_embd.weight", 0);
    put_tensor(t, "per_layer_token_embd.weight", 1024);
    {
      const uint64_t list_end = t.size();
      const uint64_t data_start = (list_end + 31) / 32 * 32;
      t.insert(t.end(), (size_t)(data_start - list_end) + 1024 + 512, 0xAA);
      tp = write_gguf(t);
      ok = gguf::parse_header(tp, tm2, err);
      assert(ok && tm2.cpu_extra_bytes == t.size() - data_start - 1024);
      std::filesystem::remove(tp);
    }
  }

  // Враждебные скаляры: assign() обязан обрезать их до потолка планировщика —
  // без потолка block_count=1e9 вешает циклы раскладки, а произведения скаляров
  // замыкаются по модулю 2^64 в ложное «всезет» (см. fit_test).
  {
    std::vector<uint8_t> v;
    put(v, "GGUF", 4);
    put_u32(v, 3);
    put_u64(v, 0); // тензоров нет
    put_u64(v, 8); // kv-пар
    put_kv_string(v, "general.architecture", "llama");
    put_kv_u64(v, "llama.block_count", 1000000000ull);
    put_kv_u64(v, "llama.embedding_length", ~(uint64_t)0);
    put_kv_u64(v, "llama.attention.head_count", 1ull << 40);
    put_kv_u64(v, "llama.attention.head_count_kv", 1ull << 40);
    put_kv_u64(v, "llama.vocab_size", ~(uint64_t)0);
    put_kv_u64(v, "llama.context_length", 1ull << 48);
    put_kv_u64(v, "llama.expert_count", 1ull << 48);
    std::string vp = write_gguf(v);
    gguf::Meta vm;
    bool ook = gguf::parse_header(vp, vm, err);
    assert(ook && err.empty());
    assert(vm.n_layers == 65535);
    assert(vm.n_embd == (1ull << 20));
    assert(vm.n_head == (1ull << 20));
    assert(vm.n_head_kv == (1ull << 20));
    assert(vm.n_vocab == (1ull << 24));
    assert(vm.n_ctx_train == (1ull << 26));
    assert(vm.n_experts == (1ull << 16));
    std::filesystem::remove(vp);

    // Мусорный офсет в тензорном списке: хвост-тензор не может влезть в файл —
    // обход сдаётся, expert_bytes остаётся 0 (вызывающие возьмут консервативную
    // оценку), разбор заголовка при этом успешен.
    std::vector<uint8_t> t;
    put(t, "GGUF", 4);
    put_u32(t, 3);
    put_u64(t, 1); // один тензор
    put_u64(t, 1); // одна kv-пара
    put_kv_string(t, "general.architecture", "llama");
    put_tensor(t, "blk.0.ffn_up_exps.weight", ~(uint64_t)0);
    std::string tp2 = write_gguf(t);
    gguf::Meta tm3;
    ook = gguf::parse_header(tp2, tm3, err);
    assert(ook && tm3.expert_bytes == 0 && tm3.cpu_extra_bytes == 0);
    std::filesystem::remove(tp2);
  }

  std::puts("gguf_test: ok");
  return 0;
}
