// Реестр: шард-парсер и спутники удаления (companions_of).
#include "image.h"
#include "modelreg.h"
#include "util.h"

#include <unistd.h>

#include <cassert>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace lh;
using namespace lh::modelreg;

namespace fs = std::filesystem;

namespace {

Model m(const std::string& path, uint64_t size = 1000) {
  Model r;
  r.path = path;
  r.size = size;
  return r;
}

bool has(const std::vector<Companion>& cs, const std::string& p, const char* kind) {
  for (auto& c : cs)
    if (c.path == p && c.kind == kind) return true;
  return false;
}

} // namespace

int main() {
  // shard_info: канонические имена llama.cpp.
  {
    int i, c;
    shard_info("Qwen3-30B-A3B-Q4_K_M-00001-of-00003.gguf", i, c);
    assert(i == 1 && c == 3);
    shard_info("base-00002-of-00007.gguf", i, c);
    assert(i == 2 && c == 7);
    shard_info("m-12345-of-00099.gguf", i, c);
    assert(i == 12345 && c == 99);
    shard_info("plain.gguf", i, c);
    assert(i == 0 && c == 1);
    shard_info("x-of-.gguf", i, c); // мусор
    assert(i == 0 && c == 1);
    shard_info("some-Q4_K_M-of-00003.gguf", i, c); // перед «-of-» нет номера
    assert(i == 0 && c == 1);
    shard_info("big-00001-of-0003a.gguf", i, c); // число с мусором
    assert(i == 0 && c == 1);
  }
  // shard_base: группа — общее имя без номера.
  {
    assert(shard_base("base-00001-of-00003.gguf") == "base-");
    assert(shard_base("base-00003-of-00003.gguf") == "base-");
    assert(shard_base("Qwen3-30B-A3B-Q4_K_M-00002-of-00003.gguf") ==
           "Qwen3-30B-A3B-Q4_K_M-");
    assert(shard_base("plain.gguf") == "plain");
    // Фильтр pull — база и «*». База уже с тире, «-*» даёт «--*» и не
    // накрывает шарды Hugging Face.
    const char* shard = "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf";
    std::string filter = shard_base(shard) + "*";
    assert(filter == "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-*");
    assert(lh::wildcard_match(filter.c_str(), shard));
    assert(!lh::wildcard_match((shard_base(shard) + "-*").c_str(), shard));
    assert(registry_stem(shard) == "Qwen3.8-Flash-Next-UD-Q4_K_XL");
    assert(!is_shard_continuation(shard));
    assert(is_shard_continuation(
        "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf"));
    assert(registry_stem(
               "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf") ==
           "Qwen3.8-Flash-Next-UD-Q4_K_XL");
    assert(!is_shard_continuation("plain.gguf"));
    assert(registry_stem("/d/repo/plain.gguf") == "plain");
  }
  // На диске группа — одна модель: сумма частей, хвосты не самостоятельны.
  {
    namespace fs = std::filesystem;
    fs::path dir = fs::temp_directory_path() / "llmhost-shard-group";
    fs::remove_all(dir);
    fs::create_directories(dir);
    auto touch = [&](const char* name, int n) {
      std::ofstream o(dir / name, std::ios::binary);
      o << std::string(n, 'x');
    };
    touch("M-00001-of-00003.gguf", 10);
    touch("M-00002-of-00003.gguf", 20);
    touch("M-00003-of-00003.gguf", 30);
    touch("other.gguf", 7);
    std::string first = (dir / "M-00001-of-00003.gguf").string();
    assert(shard_group_bytes(first) == 60);
    assert(shard_group_bytes((dir / "M-00002-of-00003.gguf").string()) == 60);
    assert(shard_group_bytes((dir / "other.gguf").string()) == 7);
    assert(!is_shard_continuation(first));
    assert(is_shard_continuation((dir / "M-00002-of-00003.gguf").string()));
    assert(registry_stem(first) == "M");
    fs::remove_all(dir);
  }
  // Одиночная модель: mmproj и MTP-модуль — её спутники.
  {
    std::vector<Model> list = {
        m("/d/org/repo/Qwen3-8B-Q4_K_M.gguf", 5000),
        m("/d/org/repo/mmproj-F16.gguf", 300),
        m("/d/org/repo/MTP/mtp-module.gguf", 200),
    };
    auto cs = companions_of("/d/org/repo/Qwen3-8B-Q4_K_M.gguf", list);
    assert(cs.size() == 2);
    assert(has(cs, "/d/org/repo/mmproj-F16.gguf", "mmproj"));
    assert(has(cs, "/d/org/repo/MTP/mtp-module.gguf", "mtp"));
  }
  // mtp-* без подпапки — тоже модуль.
  {
    std::vector<Model> list = {
        m("/d/repo/model.gguf"),
        m("/d/repo/mtp-8b.gguf", 200),
    };
    auto cs = companions_of("/d/repo/model.gguf", list);
    assert(cs.size() == 1);
    assert(has(cs, "/d/repo/mtp-8b.gguf", "mtp"));
  }
  // В папке осталась другая основная модель — общие модули не спутники.
  {
    std::vector<Model> list = {
        m("/d/org/repo/Q4_K_M.gguf", 5000),
        m("/d/org/repo/Q8_0.gguf", 9000),
        m("/d/org/repo/mmproj-F16.gguf", 300),
        m("/d/org/repo/MTP/mtp.gguf", 200),
    };
    assert(companions_of("/d/org/repo/Q4_K_M.gguf", list).empty());
    assert(companions_of("/d/org/repo/Q8_0.gguf", list).empty());
  }
  // Шард-группа: остальные части уходят вместе с первой.
  {
    std::vector<Model> list = {
        m("/d/repo/big-00001-of-00003.gguf", 5000),
        m("/d/repo/big-00002-of-00003.gguf", 5000),
        m("/d/repo/big-00003-of-00003.gguf", 4000),
        m("/d/repo/mmproj-F16.gguf", 300),
    };
    auto cs = companions_of("/d/repo/big-00001-of-00003.gguf", list);
    assert(cs.size() == 3);
    assert(has(cs, "/d/repo/big-00002-of-00003.gguf", "shard"));
    assert(has(cs, "/d/repo/big-00003-of-00003.gguf", "shard"));
    assert(has(cs, "/d/repo/mmproj-F16.gguf", "mmproj"));
  }
  // Две шард-группы в одной папке: у второй своя mmproj остаётся нужной.
  {
    std::vector<Model> list = {
        m("/d/repo/a-00001-of-00002.gguf"),
        m("/d/repo/a-00002-of-00002.gguf"),
        m("/d/repo/b-00001-of-00002.gguf"),
        m("/d/repo/b-00002-of-00002.gguf"),
        m("/d/repo/mmproj-F16.gguf", 300),
    };
    auto cs = companions_of("/d/repo/a-00001-of-00002.gguf", list);
    assert(cs.size() == 1);
    assert(has(cs, "/d/repo/a-00002-of-00002.gguf", "shard"));
  }
  // Квант во вложенной папке, голова в MTP/ репозитория — как после
  // скачивания. mmproj сверху по-прежнему не её: проекция ищется только
  // в папке самого файла.
  {
    std::vector<Model> list = {
        m("/d/repo/sub/model.gguf"),
        m("/d/repo/mmproj-F16.gguf", 300),
        m("/d/repo/MTP/mtp.gguf", 200),
    };
    auto cs = companions_of("/d/repo/sub/model.gguf", list);
    assert(cs.size() == 1);
    assert(has(cs, "/d/repo/MTP/mtp.gguf", "mtp"));
    assert(mtp_place("/d/repo/sub/model.gguf", "/d/repo/MTP/mtp.gguf") == 2);
    assert(mtp_place("/d/repo/UD/model.gguf", "/d/repo/UD/MTP/head.gguf") == 1);
    assert(mtp_place("/d/repo/model.gguf", "/d/repo/MTP/head.gguf") == 1);
    assert(mtp_place("/d/repo/sub/model.gguf", "/d/repo/mtp-other.gguf") == 0);
    assert(mtp_place("/d/repo/sub/model.gguf", "/d/other/MTP/head.gguf") == 0);
  }
  // Два кванта одного репозитория делят голову в MTP/. Удаление одного
  // её не забирает.
  {
    std::vector<Model> list = {
        m("/d/repo/UD-Q4_K_XL/a-00001-of-00004.gguf"),
        m("/d/repo/UD-Q3_K_XL/b.gguf"),
        m("/d/repo/MTP/mtp.gguf", 200),
    };
    assert(companions_of("/d/repo/UD-Q4_K_XL/a-00001-of-00004.gguf", list).empty());
    assert(companions_of("/d/repo/UD-Q3_K_XL/b.gguf", list).empty());
  }
  // Голова остаётся спутником, когда других моделей репозитория уже нет.
  {
    std::vector<Model> list = {
        m("/d/repo/UD-Q4_K_XL/a-00001-of-00004.gguf"),
        m("/d/repo/MTP/mtp.gguf", 200),
    };
    auto cs = companions_of("/d/repo/UD-Q4_K_XL/a-00001-of-00004.gguf", list);
    assert(cs.size() == 1);
    assert(has(cs, "/d/repo/MTP/mtp.gguf", "mtp"));
  }
  // Удаляемой модели может не быть в скане (лимит, гонка) — группа по имени.
  {
    std::vector<Model> list = {m("/d/repo/a-00002-of-00002.gguf")};
    auto cs = companions_of("/d/repo/a-00001-of-00002.gguf", list);
    assert(cs.size() == 1);
    assert(has(cs, "/d/repo/a-00002-of-00002.gguf", "shard"));
  }

  // Мета шард-группы: заголовочная часть несёт метаданные (tensor_count = 0 —
  // раскладка unsloth), тензорные списки живут в частях 2..N — экспертные
  // байты, CPU-тензоры и поблочная раскладка суммируются по всем частям.
  {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "llmhost_modelreg_group";
    fs::remove_all(dir);
    fs::create_directories(dir);
    auto wr = [&](const std::string& name, const std::vector<uint8_t>& b) {
      std::ofstream f(dir / name, std::ios::binary);
      f.write((const char*)b.data(), (std::streamsize)b.size());
    };
    auto u32 = [&](std::vector<uint8_t>& b, uint32_t v) {
      for (int i = 0; i < 4; ++i) b.push_back((v >> (8 * i)) & 0xFF);
    };
    auto u64 = [&](std::vector<uint8_t>& b, uint64_t v) {
      for (int i = 0; i < 8; ++i) b.push_back((v >> (8 * i)) & 0xFF);
    };
    auto str = [&](std::vector<uint8_t>& b, const std::string& s) {
      u64(b, s.size());
      b.insert(b.end(), s.begin(), s.end());
    };
    auto kv_str = [&](std::vector<uint8_t>& b, const std::string& k,
                      const std::string& v) {
      str(b, k);
      u32(b, 8);
      str(b, v);
    };
    auto kv_u32 = [&](std::vector<uint8_t>& b, const std::string& k, uint32_t v) {
      str(b, k);
      u32(b, 4);
      u32(b, v);
    };
    auto tensor = [&](std::vector<uint8_t>& b, const std::string& name,
                      uint64_t off) {
      str(b, name);
      u32(b, 2);
      u64(b, 256);
      u64(b, 1);
      u32(b, 0);
      u64(b, off);
    };
    auto gguf_head = [&](std::vector<uint8_t>& b, uint64_t tensors, uint64_t kvs) {
      b.insert(b.end(), {'G', 'G', 'U', 'F'});
      u32(b, 3);
      u64(b, tensors);
      u64(b, kvs);
    };
    // Часть 1: только метаданные.
    {
      std::vector<uint8_t> b;
      gguf_head(b, 0, 3);
      kv_str(b, "general.architecture", "qwen4exp");
      kv_u32(b, "qwen4exp.block_count", 4);
      kv_u32(b, "qwen4exp.full_attention_interval", 4);
      wr("grp-00001-of-00002.gguf", b);
    }
    // Часть 2: тензорный список; файл кончается per_layer_token_embd —
    // CPU-тензор размерится концом файла.
    {
      std::vector<uint8_t> b;
      gguf_head(b, 4, 1);
      kv_str(b, "general.architecture", "qwen4exp");
      tensor(b, "token_embd.weight", 0);
      tensor(b, "blk.0.ffn_up_exps.weight", 1024);
      tensor(b, "blk.1.ffn_up_exps.weight", 3072);
      tensor(b, "per_layer_token_embd.weight", 6144);
      const uint64_t list_end = b.size();
      const uint64_t data_start = (list_end + 31) / 32 * 32;
      b.insert(b.end(), (size_t)(data_start - list_end) + 6144 + 256, 0xAA);
      wr("grp-00002-of-00002.gguf", b);
    }
    gguf::Meta mm;
    assert(model_meta((dir / "grp-00001-of-00002.gguf").string(), mm));
    assert(mm.arch == "qwen4exp" && mm.n_layers == 4);
    assert(mm.full_attn_interval == 4);
    assert(mm.expert_bytes == 2048 + 3072); // два блока, спаны одной части
    assert(mm.expert_bytes_by_block.size() == 2);
    assert(mm.expert_bytes_by_block[0] == 2048);
    assert(mm.expert_bytes_by_block[1] == 3072);
    assert(mm.cpu_extra_bytes > 0);
    // Часть 2 сама по себе — та же группа: частей после неё нет, мета своя.
    gguf::Meta m2;
    assert(model_meta((dir / "grp-00002-of-00002.gguf").string(), m2));
    assert(m2.expert_bytes == mm.expert_bytes && m2.cpu_extra_bytes == mm.cpu_extra_bytes);
    // Обычный файл вне группы — чистый заголовок.
    gguf::Meta solo;
    assert(model_meta((dir / "..").string() + "/nope.gguf", solo) == false);
    fs::remove_all(dir);
  }

  // ---- mtp_matches_trunk: модуль вяжется к своему стволу ------------------
  // Пары реестра из рабочего инстанса: квантовые хвосты разной длины отсекаются,
  // шард-группа сравнивается по базовому имени, регистр не важен.
  assert(mtp_matches_trunk("/m/Qwen3.8-27B-UD-Q4_K_XL.gguf",
                           "/m/mtp-Qwen3.8-27B-Q4_0.gguf"));
  assert(mtp_matches_trunk("/m/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf",
                           "/m/mtp-Qwen3-Next-80B-A3B-Instruct-Q8_0.gguf"));
  assert(mtp_matches_trunk("/m/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
                           "/m/repo/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf"));
  assert(mtp_matches_trunk("/m/gemma-4-12b-it-UD-Q5_K_XL.gguf",
                           "/m/mtp-gemma-4-12B-it-Q8_0.gguf")); // 12B против 12b
  assert(mtp_matches_trunk("/m/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
                           "/m/repo/MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf"));
  assert(mtp_matches_trunk("/m/X-UD-Q4_K_XL.gguf", "/m/mtp-X-Q4_K_M.gguf"));
  // Чужие головы: словарь общий и мета-фильтры их пропускали, привязка — нет.
  assert(!mtp_matches_trunk("/m/Qwen3.6-35B-A3B-NVFP4-MTP-TURBO.gguf",
                            "/m/mtp-Qwen3.8-27B-Q4_0.gguf"));
  assert(!mtp_matches_trunk("/m/qwen3.5-4b-nvfp4.gguf",
                            "/m/mtp-Qwen3.8-27B-Q4_0.gguf"));
  assert(!mtp_matches_trunk("/m/qwen3.5-4b-nvfp4.gguf",
                            "/m/mtp-gemma-4-12B-it-Q8_0.gguf"));
  // «shared» — не токен имени ствола: модуль мимо.
  assert(!mtp_matches_trunk("/m/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
                            "/m/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf"));
  // Частичное совпадение токена — не совпадение.
  assert(!mtp_matches_trunk("/m/Qwen3.5-4B-NVFP4.gguf", "/m/mtp-Qwen3-Q8_0.gguf"));
  // Пользовательский суффикс (frspec-owngen…) — база шире ствола, мимо.
  assert(!mtp_matches_trunk(
      "/m/Qwen3.6-35B-A3B-NVFP4-MTP-TURBO.gguf",
      "/m/mtp-Ornith-1.5-35B-A3B-NVFP4-frspec-owngen32768.gguf"));

  // ---- mmproj: сигнатура и родство проекции со стволом --------------------
  // Реальные имена машины разработки (29.10): рукименные «ствол-mmproj-формат»
  // и апстримные «mmproj-*» pull-раскладки.
  assert(mmproj_core_of("/m/qwen36-mmproj-BF16.gguf") == "qwen36");
  assert(mmproj_core_of("/m/mmproj-gemma4-BF16.gguf") == "gemma4");
  assert(mmproj_core_of("/m/gemma12-mmproj-BF16.gguf") == "gemma12");
  // Форматные/квантовые хвосты и стоп-слово pull в ядро не входят.
  assert(mmproj_core_of("/m/mmproj-F16.gguf").empty());
  assert(mmproj_core_of("/m/mmproj-model-f16.gguf").empty());
  assert(mmproj_core_of("/m/mmproj-Muse-Glimmer-30B-Q8_0.gguf") ==
         "museglimmer30b");
  // Свои проекции: сокращение семьи — подпоследовательность имени ствола.
  assert(mmproj_matches_trunk("/m/Qwen3.6-35B-A3B-NVFP4-MTP-TURBO.gguf",
                              "/m/qwen36-mmproj-BF16.gguf"));
  assert(mmproj_matches_trunk("/m/Qwen3.8-27B-UD-Q4_K_XL.gguf",
                              "/m/qwen38-mmproj-BF16.gguf"));
  assert(mmproj_matches_trunk("/m/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf",
                              "/m/mmproj-gemma4-BF16.gguf"));
  assert(mmproj_matches_trunk("/m/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
                              "/m/gemma12-mmproj-BF16.gguf"));
  assert(mmproj_matches_trunk("/m/Muse-Glimmer-30B-UD-Q4_K_XL.gguf",
                              "/m/mmproj-Muse-Glimmer-30B-Q8_0.gguf"));
  // Чужие: урок 29.10 — «первая попавшаяся» прицепляла gemma-проекцию к Qwen.
  assert(!mmproj_matches_trunk("/m/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf",
                               "/m/qwen36-mmproj-BF16.gguf"));
  assert(!mmproj_matches_trunk("/m/Qwen3.6-35B-A3B-NVFP4-MTP-TURBO.gguf",
                               "/m/mmproj-gemma4-BF16.gguf"));
  assert(!mmproj_matches_trunk("/m/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf",
                               "/m/qwen38-mmproj-BF16.gguf"));
  // «gemma12» различает 12B и 26B одного семейства — иначе длин общих ядер
  // недостаточно: «gemma4» вяжется к обоим, ядро длиннее выигрывает.
  assert(mmproj_matches_trunk("/m/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
                              "/m/mmproj-gemma4-BF16.gguf"));
  // Легаси-имя без сигнатуры родства не доказывает; короткое ядро — тоже.
  assert(!mmproj_matches_trunk("/m/Qwen3.8-27B-UD-Q4_K_XL.gguf",
                               "/m/mmproj-F16.gguf"));
  assert(!mmproj_matches_trunk("/m/qwen3.5-4b-nvfp4.gguf",
                               "/m/qwen36-mmproj-BF16.gguf"));

  // ---- is_mmproj_module: «mmproj» в любом месте имени ----------------------
  assert(is_mmproj_module("/m/mmproj-F16.gguf"));
  assert(is_mmproj_module("/m/qwen36-mmproj-BF16.gguf"));
  assert(is_mmproj_module("/m/MMProj-Muse-Glimmer-30B-Q8_0.gguf"));
  assert(!is_mmproj_module("/m/Qwen3.6-35B-A3B-NVFP4-MTP-TURBO.gguf"));

  // ---- скан: категории sd-server в корне моделей не индексируются ----------
  // Энкодеры — легитимные GGUF (мусорные «чат-модели»), VAE — «не gguf».
  {
    fs::path d = fs::temp_directory_path() /
                 ("llmhost-modelreg-test-" + std::to_string(::getpid()));
    fs::remove_all(d);
    auto put = [](const fs::path& p) {
      fs::create_directories(p.parent_path());
      std::ofstream f(p, std::ios::binary);
      f << "x";
    };
    put(d / "Qwen3-8B-Q4_K_M.gguf");
    put(d / "text_encoders" / "qwen3-4b-Q4_K_M.gguf"); // верхний уровень — мимо
    put(d / "vae" / "ae.safetensors");                 // и не gguf вовсе
    put(d / "diffusion_models" / "z.gguf");
    put(d / "unsloth" / "Nested-Q4_K_M.gguf");         // не категория — сканируется
    put(d / "unsloth" / "vae" / "deep.gguf");          // глубже первого уровня — тоже
    auto v = scan(d.string());
    size_t top = 0, nested = 0, catalog = 0;
    for (auto& m : v) {
      std::string rel = fs::relative(m.path, d).string();
      std::string first = rel.substr(0, rel.find('/'));
      if (lh::image::is_catalog_dir_name(first))
        ++catalog; // верхнеуровневая категория — из реестра исключена
      else if (rel == "Qwen3-8B-Q4_K_M.gguf")
        ++top;
      else
        ++nested;
    }
    assert(top == 1);
    assert(nested == 2); // unsloth/Nested + unsloth/vae/deep: глубже корня — не исключение
    assert(catalog == 0);
    fs::remove_all(d);
  }
  return 0;
}
