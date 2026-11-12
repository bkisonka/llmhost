#include "gguf.h"
#include "util.h"

#include <cstdio>
#include <cstring>
#include <functional>
#include <algorithm>
#include <vector>

namespace lh::gguf {

namespace {

// gguf_type из llama.cpp.
enum Type : uint32_t {
  T_UINT8 = 0, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32,
  T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64,
};

uint64_t fixed_size(uint32_t t) {
  switch (t) {
    case T_UINT8: case T_INT8: case T_BOOL: return 1;
    case T_UINT16: case T_INT16: return 2;
    case T_UINT32: case T_INT32: case T_FLOAT32: return 4;
    case T_UINT64: case T_INT64: case T_FLOAT64: return 8;
    default: return 0; // string и array — не фиксированные
  }
}

class Reader {
public:
  explicit Reader(const std::string& path) : f_(fopen(path.c_str(), "rb")) {}
  Reader(const char* buf, size_t len) : buf_(buf), len_(len) {}
  ~Reader() { if (f_) fclose(f_); }
  Reader(const Reader&) = delete;
  Reader& operator=(const Reader&) = delete;
  bool ok() const { return f_ != nullptr || buf_ != nullptr; }

  bool bytes(void* dst, size_t n) {
    if (buf_) {
      if (len_ - pos_ < n) { pos_ = len_; return false; }
      memcpy(dst, buf_ + pos_, n);
      pos_ += n;
      return true;
    }
    return fread(dst, 1, n, f_) == n;
  }
  bool u32(uint32_t& v) { return bytes(&v, 4); }
  bool u64(uint64_t& v) { return bytes(&v, 8); }

  bool skip(uint64_t n) {
    if (buf_) {
      if (len_ - pos_ < n) { pos_ = len_; return false; }
      pos_ += (size_t)n;
      return true;
    }
    // fseeko: off_t на x86-64 Linux 64-битный — GGUF крупнее 4 ГиБ проходит.
    return fseeko(f_, (off_t)n, SEEK_CUR) == 0;
  }

  // Позиция чтения: конец тензорного списка — это начало секции данных
  // (с точностью до выравнивания), от неё считаются офсеты тензоров.
  uint64_t pos() const {
    if (buf_) return pos_;
    return f_ ? (uint64_t)ftello(f_) : 0;
  }

  bool str(std::string& s) {
    uint64_t len = 0;
    if (!u64(len) || len > (1u << 24)) return false; // длиннее 16 МБ — файл битый
    s.resize(len);
    return len == 0 || bytes(s.data(), (size_t)len);
  }

  bool skip_str() {
    uint64_t len = 0;
    return u64(len) && len <= (1u << 24) && skip(len);
  }

  // Значение любого типа. Для целых фиксированных зовёт on_scalar(тип, значение),
  // остальные (строки, массивы, float) просто пропускает.
  bool value(uint32_t type, const std::function<void(uint32_t, uint64_t)>& on_scalar) {
    if (uint64_t sz = fixed_size(type)) {
      uint64_t v = 0;
      if (!bytes(&v, (size_t)sz)) return false;
      if (on_scalar) on_scalar(type, v);
      return true;
    }
    if (type == T_STRING) {
      std::string s;
      return str(s);
    }
    if (type == T_ARRAY) {
      uint32_t elem = 0;
      uint64_t count = 0;
      if (!u32(elem) || !u64(count)) return false;
      if (uint64_t esz = fixed_size(elem)) {
        // count*esz мог переполниться на битом файле — парсер рассинхронизировался бы
        if (esz && count > ~(uint64_t)0 / esz) return false;
        return skip(count * esz);
      }
      if (elem == T_STRING) {
        for (uint64_t i = 0; i < count; ++i)
          if (!skip_str()) return false;
        return true;
      }
      return false; // вложенные массивы форматом не предусмотрены
    }
    return false;
  }

private:
  FILE* f_ = nullptr;
  const char* buf_ = nullptr; // режим памяти: Range-выборка удалённого файла
  size_t len_ = 0, pos_ = 0;
};

bool is_int(uint32_t t) {
  return t == T_UINT32 || t == T_INT32 || t == T_UINT64 || t == T_INT64;
}

// Потолок скаляров заголовка. Meta читается из недоверенного файла и кормит
// планировщика (fit, sup_plan, sup_harness, catalog_est): без потолка
// block_count = 1e9 в килобайтном файле вешал циклы раскладки на минуты, а
// произведения скаляров замыкались по модулю 2^64 в «всё влезает» (план
// впритык → OOM ребёнка). Реальные модели далеко ниже этих границ: слои
// единицы-сотни, embd/головы десятки тысяч, vocab ≤ 2^18, ctx_train ≤ 2^21.
constexpr uint64_t kMaxLayers = 65535;      // запас под полные циклы ngl/N
constexpr uint64_t kMaxDim = 1u << 20;      // embd, головы, key_length
constexpr uint64_t kMaxCtx = 1ull << 26;    // ctx_train и sliding_window (токены)
constexpr uint64_t kMaxVocab = 1ull << 24;  // vocab_size
constexpr uint64_t kMaxExperts = 1u << 16;  // expert_count / used
constexpr uint64_t kMaxInterval = 1u << 16; // full_attention_interval

uint64_t capv(uint64_t v, uint64_t max) { return v > max ? max : v; }

void assign(const std::string& key, uint32_t type, uint64_t v, Meta& m) {
  if (!is_int(type)) return;
  auto ends = [&](const char* suffix) {
    size_t n = strlen(suffix);
    return key.size() >= n && key.compare(key.size() - n, n, suffix) == 0;
  };
  if (ends(".block_count")) m.n_layers = capv(v, kMaxLayers);
  else if (ends(".context_length")) m.n_ctx_train = capv(v, kMaxCtx);
  else if (ends(".embedding_length")) m.n_embd = capv(v, kMaxDim);
  else if (ends(".attention.head_count")) m.n_head = capv(v, kMaxDim);
  else if (ends(".attention.head_count_kv")) m.n_head_kv = capv(v, kMaxDim);
  else if (ends(".attention.key_length")) m.n_embd_head_k = capv(v, kMaxDim);
  else if (ends(".attention.key_length_swa")) m.n_embd_head_k_swa = capv(v, kMaxDim);
  else if (ends(".attention.sliding_window")) m.n_swa = capv(v, kMaxCtx);
  else if (ends(".expert_count")) m.n_experts = capv(v, kMaxExperts);
  else if (ends(".expert_used_count")) m.n_experts_used = capv(v, kMaxExperts);
  else if (ends(".nextn_predict_layers")) m.nextn_predict_layers = capv(v, kMaxLayers);
  else if (ends(".vocab_size")) m.n_vocab = capv(v, kMaxVocab);
  else if (ends(".full_attention_interval")) m.full_attn_interval = (uint32_t)capv(v, kMaxInterval);
  else if (ends(".ssm.state_size")) m.has_ssm = true;
}

// Строка из недоверенного файла в сообщение об ошибке (оно уходит в журнал):
// control-символы — в точку, длина — до 64: иначе «\n…» из ключа метаданных
// печатался бы в log.txt как есть.
std::string printable_key(const std::string& s) {
  std::string out;
  out.reserve(std::min<size_t>(s.size(), 64));
  for (size_t i = 0; i < s.size() && i < 64; ++i) {
    unsigned char c = (unsigned char)s[i];
    out += (c < 0x20 || c == 0x7F) ? '.' : (char)c;
  }
  return out;
}

bool ends_with(const std::string& s, const char* suffix) {
  size_t n = strlen(suffix);
  return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

// Блок экспертных тензоров, которые двигает --n-cpu-moe N апстрима, или -1.
// Регэксп апстрима (common/common.h: LLM_FFN_EXPS_REGEX + llm_ffn_exps_block_regex):
// blk\.<N>\.ffn_(up|down|gate|gate_up)_(ch|)exps — проверяем его точным
// перечислением хвостов. nextn-тензоры (blk.N.nextn.ffn_*) и прочие имена
// с «exps» дают -1: полный вынос (-ot exps=CPU) их берёт, частичный — нет,
// и побайтовый учёт частичного обязан это отражать.
int ncmoe_block_of(const std::string& name) {
  if (name.compare(0, 4, "blk.") != 0) return -1;
  size_t i = 4;
  long long blk = 0;
  while (i < name.size() && isdigit((unsigned char)name[i])) {
    blk = blk * 10 + (name[i] - '0');
    if (blk > 100000) return -1; // не номер блока — мусор имени
    ++i;
  }
  if (i == 4 || i >= name.size() || name[i] != '.') return -1;
  static const char* const kTails[] = {
      "ffn_up_exps",   "ffn_down_exps",   "ffn_gate_exps",   "ffn_gate_up_exps",
      "ffn_up_chexps", "ffn_down_chexps", "ffn_gate_chexps", "ffn_gate_up_chexps"};
  const std::string tail = name.substr(i + 1);
  // Регэксп апстрима ищется в имени (не якорен в конец): за «exps» обычно
  // стоит «.weight» (иные суффиксы писателей тоже допустимы).
  for (const char* t : kTails)
    if (tail.compare(0, strlen(t), t) == 0) return (int)blk;
  return -1;
}

} // namespace

bool do_parse(Reader& r, Meta& out, std::string& err, bool tolerant, uint64_t file_size = 0) {
  err.clear(); // успех должен оставлять err пустым, а не наследовать чужой
  char magic[4] = {};
  if (!r.bytes(magic, 4) || memcmp(magic, "GGUF", 4) != 0) { err = "это не GGUF"; return false; }
  uint32_t version = 0;
  if (!r.u32(version) || (version != 2 && version != 3)) { err = "версия GGUF не поддерживается"; return false; }
  uint64_t tensor_count = 0, kv_count = 0;
  if (!r.u64(tensor_count) || !r.u64(kv_count)) { err = "заголовок обрезан"; return false; }

  Meta m;
  uint32_t alignment = 32; // general.alignment, спецификация — дефолт 32
  uint64_t swa_stride = 0; // скалярный паттерн SWA («каждый N-й слой глобальный»)
  std::vector<uint64_t> swa_pattern; // флаг «слой локальный» по слою
  std::vector<uint64_t> head_kv_arr; // KV-головы по слою
  // Толерантный режим (урезанная Range-выборка): главное уже собрано, а данные
  // кончились посреди токенизаторных массивов — не ошибка.
  auto bail = [&](const std::string& msg) {
    if (tolerant && m.n_layers && m.n_embd) { out = m; return true; }
    err = msg;
    return false;
  };
  for (uint64_t i = 0; i < kv_count; ++i) {
    std::string key;
    uint32_t type = 0;
    if (!r.str(key) || !r.u32(type)) return bail("метаданные обрезаны");

    // Строки: интересны два ключа, остальные пропускаем не читая.
    if (type == T_STRING) {
      if (key == "general.name" || key == "general.architecture") {
        std::string s;
        if (!r.str(s)) return bail("метаданные обрезаны");
        if (key == "general.name") m.name = s; else m.arch = s;
      } else if (!r.skip_str()) {
        return bail("метаданные обрезаны");
      }
      continue;
    }
    if (type == T_UINT32 && key == "general.file_type") {
      if (!r.u32(m.file_type)) return bail("метаданные обрезаны");
      continue;
    }
    // Малые массивы «по слою»: SWA-паттерн (флаг локального слоя) и
    // head_count_kv (KV-головы; у gemma4 это массив, скаляра нет — берём
    // первый элемент, слои считаем однородными). Остальные массивы
    // пропускаются seek'ом, как и прежде.
    if (type == T_ARRAY &&
        (ends_with(key, ".attention.sliding_window_pattern") ||
         ends_with(key, ".attention.head_count_kv"))) {
      uint32_t elem = 0;
      uint64_t count = 0;
      if (!r.u32(elem) || !r.u64(count)) return bail("метаданные обрезаны");
      const uint64_t esz = fixed_size(elem);
      const bool pattern = ends_with(key, ".attention.sliding_window_pattern");
      // Лимит и защита от переполнения — как в generic-пропуске; больше
      // слоёв, чем 4096, метаданные на слой не носят — это не наш формат.
      if (esz > 0 && count > 0 && count <= 4096 && count <= ~(uint64_t)0 / esz) {
        if (pattern) {
          uint64_t nonzero = 0;
          swa_pattern.resize((size_t)count);
          for (uint64_t j = 0; j < count; ++j) {
            uint64_t v = 0;
            if (!r.bytes(&v, (size_t)esz)) return bail("метаданные обрезаны");
            swa_pattern[(size_t)j] = (v != 0);
            nonzero += (v != 0);
          }
          m.n_swa_layers = nonzero;
        } else {
          head_kv_arr.resize((size_t)count);
          for (uint64_t j = 0; j < count; ++j) {
            uint64_t v = 0;
            if (!r.bytes(&v, (size_t)esz)) return bail("метаданные обрезаны");
            head_kv_arr[(size_t)j] = v;
          }
          if (is_int(elem) && head_kv_arr[0] > 0) m.n_head_kv = head_kv_arr[0];
        }
        continue;
      }
      if (!r.value(type, {})) return bail("значение «" + printable_key(key) + "» не разбирается");
      continue;
    }
    if (!r.value(type, [&](uint32_t t, uint64_t v) {
          if (key == "general.alignment" && t == T_UINT32 && v > 0) alignment = (uint32_t)v;
          if (ends_with(key, ".attention.sliding_window_pattern") && is_int(t))
            swa_stride = v; // n_layers может идти позже — разрешаем после цикла
          assign(key, t, v, m);
        }))
      return bail("значение «" + printable_key(key) + "» не разбирается");
  }
  // Скалярный SWA-паттерн: каждый N-й слой глобальный, остальные локальные.
  if (swa_stride > 1 && m.n_layers > 1 && m.n_swa_layers == 0)
    m.n_swa_layers = m.n_layers - (m.n_layers + swa_stride - 1) / swa_stride;
  // llama4: писатель GGUF не несёт SWA-ключей (ни окна, ни паттерна) — движок
  // выводит их для арки сам: окно 8192, каждый 4-й слой глобальный (iswa-кэш:
  // глобальные — полный контекст, локальные — окно). Выводим так же, иначе
  // аналитика считает все слои полным контекстом (Scout, KV 131k: 13 ГиБ
  // вместо ~4) и планировщик зря держит карту занятой, срезая блоки экспертов.
  if (m.arch == "llama4" && m.n_swa == 0 && m.n_layers > 1) {
    m.n_swa = 8192;
    if (m.n_swa_layers == 0) m.n_swa_layers = m.n_layers - (m.n_layers + 3) / 4;
  }
  // Корреляция KV-голов с классами слоёв: в SWA-моделях глобальные и
  // локальные слои держат разное число KV-голов (gemma4: глобальные 2,
  // локальные 8). Оба класса однородны — фиксируем; иначе оставляем 0,
  // оценка возьмёт общий fallback (консервативно).
  if (!swa_pattern.empty() && swa_pattern.size() == head_kv_arr.size() &&
      swa_pattern.size() == m.n_layers) {
    uint64_t l0 = 0, g0 = 0;
    bool l_set = false, g_set = false, l_uniform = true, g_uniform = true;
    for (size_t i = 0; i < head_kv_arr.size(); ++i) {
      if (swa_pattern[i]) {
        if (!l_set) { l0 = head_kv_arr[i]; l_set = true; }
        else if (head_kv_arr[i] != l0) l_uniform = false;
      } else {
        if (!g_set) { g0 = head_kv_arr[i]; g_set = true; }
        else if (head_kv_arr[i] != g0) g_uniform = false;
      }
    }
    if (l_set && l_uniform) m.n_head_kv_local = l0;
    if (g_set && g_uniform) m.n_head_kv_global = g0;
  }

  // Тензорный список: имя, форма, тип, офсет. Размеры тензоров по типам квантов
  // не считаем (таблица размеров блока — лишний дрейф за спецификацией): размер
  // тензора — разница офсетов соседей (писатель кладёт тензоры в файл подряд,
  // выравнивание ≤31 байт на стыке — шум). Экспертные — имена с «exps»: сумма
  // ровно то, что увезёт -ot exps=CPU в RAM. Писатели бывают разные (unsloth
  // кладёт последним именно экспертный тензор), поэтому последний (старший
  // офсет) размерится тоже — концом файла минус офсет от начала секции данных;
  // без файла (выборка из памяти, офсет конца неизвестен) экспертный хвост
  // сдаёт замер (0 → консервативная оценка). Сбой обхода (обрезанный буфер,
  // битый список) не валит разбор заголовка — expert_bytes остаётся нулём,
  // вызывающие откатятся к оценке. Попутно раскладываем ncmoe-тензоры (те, что
  // двигает --n-cpu-moe N) по блокам — точный вес частичного выноса.
  if (tensor_count > 0 && tensor_count < (1ull << 22)) {
    struct Ten {
      uint64_t off;
      bool exps;
      bool cpu_only; // per_layer_token_embd: движок на карту не берёт
      int blk; // блок ncmoe-тензора, -1 — не экспертный/не ncmoe
    };
    std::vector<Ten> tens;
    // reserve по min(заявлено, реальночитаемо): заголовок с tensor_count≈2^22
    // в килобайтном файле иначе вызывает ~64 МиБ transient-аллокацию на
    // каждый разбор (обход оборвётся на первом же мусоре — резерв уже не вернуть).
    tens.reserve((size_t)std::min<uint64_t>(tensor_count, 65536));
    bool walk_ok = true;
    for (uint64_t i = 0; i < tensor_count && walk_ok; ++i) {
      std::string name;
      uint32_t n_dims = 0, ttype = 0;
      uint64_t toff = 0;
      if (!r.str(name) || !r.u32(n_dims) || n_dims == 0 || n_dims > 8 ||
          !r.skip((uint64_t)n_dims * 8) || !r.u32(ttype) || !r.u64(toff))
        walk_ok = false;
      else
        tens.push_back({toff, name.find("exps") != std::string::npos,
                        name.find("per_layer_token_embd") != std::string::npos,
                        ncmoe_block_of(name)});
    }
    std::sort(tens.begin(), tens.end(),
              [](const Ten& a, const Ten& b) { return a.off < b.off; });
    // Последний тензор — концом файла: секция данных заканчивается файлом,
    // начало её — конец тензорного списка, выровненный на general.alignment.
    // Размерится хвост-тензор любого из учитываемых классов (писатели бывают
    // разные: unsloth кладёт последним именно экспертный).
    uint64_t last_span = 0;
    if (walk_ok && (tens.back().exps || tens.back().cpu_only)) {
      if (file_size > 0 && alignment > 0) {
        const uint64_t data_start = ((r.pos() + alignment - 1) / alignment) * alignment;
        // Без сложения data_start + off: враждебный офсет замкнул бы сумму
        // по модулю 2^64 и «прошёл» проверку.
        if (data_start <= file_size && file_size - data_start > tens.back().off)
          last_span = file_size - data_start - tens.back().off;
      }
      walk_ok = last_span > 0; // размерить не смогли — замер сдаётся
    }
    uint64_t expert_sum = 0, cpu_sum = 0, total = 0;
    int max_blk = -1;
    for (const Ten& t : tens)
      if (t.blk > max_blk) max_blk = t.blk;
    std::vector<uint64_t> by_block(max_blk >= 0 ? (size_t)max_blk + 1 : 0, 0);
    for (size_t i = 0; walk_ok && i + 1 < tens.size(); ++i) {
      const uint64_t span = tens[i + 1].off - tens[i].off;
      total += span;
      if (tens[i].exps) expert_sum += span;
      if (tens[i].cpu_only) cpu_sum += span;
      if (tens[i].blk >= 0) by_block[(size_t)tens[i].blk] += span;
    }
    if (walk_ok && last_span > 0) {
      total += last_span;
      if (tens.back().exps) expert_sum += last_span;
      if (tens.back().cpu_only) cpu_sum += last_span;
      if (tens.back().blk >= 0) by_block[(size_t)tens.back().blk] += last_span;
    }
    if (walk_ok && expert_sum > 0 && expert_sum < total) {
      m.expert_bytes = expert_sum;
      if (!by_block.empty()) m.expert_bytes_by_block = std::move(by_block);
    }
    if (walk_ok && cpu_sum > 0 && cpu_sum < total) m.cpu_extra_bytes = cpu_sum;
  }

  m.quant = quant_name(m.file_type);
  out = m;
  return true;
}

bool parse_header(const std::string& path, Meta& out, std::string& err) {
  // Размер файла нужен для размерения последнего тензора (см. обход ниже).
  uint64_t fsize = lh::file_size(path);
  Reader r(path);
  if (!r.ok()) { err = "файл не открывается"; return false; }
  return do_parse(r, out, err, /*tolerant=*/false, fsize);
}

bool parse_header_mem(const char* data, size_t len, Meta& out, std::string& err) {
  Reader r(data, len);
  if (!r.ok()) { err = "пустая выборка"; return false; }
  return do_parse(r, out, err, /*tolerant=*/true);
}

const char* quant_name(uint32_t ft) {
  switch (ft) {
    case 0: return "F32";
    case 1: return "F16";
    case 2: return "Q4_0";
    case 3: return "Q4_1";
    case 7: return "Q8_0";
    case 8: return "Q5_0";
    case 9: return "Q5_1";
    case 10: return "Q2_K";
    case 11: return "Q3_K_S";
    case 12: return "Q3_K_M";
    case 13: return "Q3_K_L";
    case 14: return "Q4_K_S";
    case 15: return "Q4_K_M";
    case 16: return "Q5_K_S";
    case 17: return "Q5_K_M";
    case 18: return "Q6_K";
    case 19: return "IQ2_XXS";
    case 20: return "IQ2_XS";
    case 21: return "Q2_K_S";
    case 22: return "IQ3_XS";
    case 23: return "IQ3_XXS";
    case 24: return "IQ1_S";
    case 25: return "IQ4_NL";
    case 26: return "IQ3_S";
    case 27: return "IQ3_M";
    case 28: return "IQ2_S";
    case 29: return "IQ2_M";
    case 30: return "IQ4_XS";
    case 31: return "IQ1_M";
    case 32: return "BF16";
    default: return "?";
  }
}

} // namespace lh::gguf
