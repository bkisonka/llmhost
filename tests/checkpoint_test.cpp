// Каталог именованных чекпоинтов и канонизатор prefix_hash: писатель
// с сохранением чужих полей, sanitize,
// sha256-векторы, «смысл тот же — форма другая» → один хэш.
#include "checkpoint.h"

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
using lh::checkpoint::Entry;
using lh::checkpoint::find;
using lh::checkpoint::list;
using lh::checkpoint::prefix_hash;
using lh::checkpoint::remove;
using lh::checkpoint::remove_model;
using lh::checkpoint::sanitize_name;
using lh::checkpoint::sha256_hex;
using lh::checkpoint::upsert;

static const char* MODELS = "/tmp/lh-checkpoint-test-models";

#define REQUIRE(cond, err)                                                   \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::printf("FAIL: %s — %s\n", #cond, std::string(err).c_str());       \
      assert(false);                                                         \
    }                                                                        \
  } while (0)

static void wipe() {
  std::error_code ec;
  fs::remove_all(MODELS, ec);
  fs::create_directories(MODELS, ec);
}

int main() {
  // sanitize_name: пользовательский ввод в имя файла напрямую не едет.
  {
    assert(sanitize_name("step 7!") == "step_7_");
    assert(sanitize_name("../../../etc") == ".._.._.._etc");
    assert(sanitize_name(std::string(100, 'a')).size() == 64);
    assert(sanitize_name("") == "checkpoint");
    assert(sanitize_name("ok-1._x") == "ok-1._x");
  }

  // sha256: известные векторы (FIPS 180-4).
  {
    assert(sha256_hex("") ==
           "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    assert(sha256_hex("abc") ==
           "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    assert(sha256_hex("pfp1|") ==
           sha256_hex("pfp1|")); // детерминизм
  }

  // Канонизатор: одинаковые по смыслу, разные по форме → один хэш; разные по
  // смыслу → разный; мультимодальный content → хэш недоступен (пусто).
  {
    const char* a = R"([{"role":"user","content":"привет"},{"role":"assistant","content":"ответ"}])";
    // Другой порядок ключей, пробелы вне строк, лишние поля.
    const char* b =
        R"([ { "content" : "привет" , "role":"user" , "temperature":0.7 }, {"role":"assistant","content":"ответ","tool_calls":null} ])";
    const char* c = R"([{"role":"user","content":"привет"},{"role":"assistant","content":"другой ответ"}])";
    nlohmann::json ja = nlohmann::json::parse(a);
    nlohmann::json jb = nlohmann::json::parse(b);
    nlohmann::json jc = nlohmann::json::parse(c);
    std::string ha = prefix_hash(ja), hb = prefix_hash(jb), hc = prefix_hash(jc);
    assert(!ha.empty());
    assert(ha == hb); // смысл тот же, форма другая
    assert(ha != hc); // смысл другой
    // Мультимодальный (массивный) content — честно недоступен.
    assert(prefix_hash(nlohmann::json::parse(
               R"([{"role":"user","content":[{"type":"text","text":"x"}]}])")).empty());
    // Без role / не массив — недоступен.
    assert(prefix_hash(nlohmann::json::parse(R"([{"content":"x"}])")).empty());
    assert(prefix_hash(nlohmann::json(42)).empty());
    // Отсутствующий content (assistant с tool_calls) — пустая строка.
    auto d = nlohmann::json::parse(R"([{"role":"assistant","tool_calls":[]}])");
    auto e = nlohmann::json::parse(R"([{"role":"assistant","content":null}])");
    assert(prefix_hash(d) == prefix_hash(e) && !prefix_hash(d).empty());
  }

  // Каталог: upsert/список/поиск, чужие поля переживают, remove, remove_model.
  {
    wipe();
    std::string err;
    Entry e;
    e.model = "/data/models/x.gguf";
    e.name = sanitize_name("step7");
    e.files.push_back({1, "x.ckpt.step7.s1.bin", 123456, 4321});
    e.files.push_back({2, "x.ckpt.step7.s2.bin", 654, 0});
    e.prefix_hash = "abc123";
    e.created = 1730000000.0;
    REQUIRE(upsert(MODELS, e, err), err);

    // Чужой ключ верхнего уровня и чужие поля записи — переживают upsert:
    // каталог правим руками, минуя писателя модуля, затем едет upsert.
    {
      nlohmann::json j;
      j["written_by"] = "посторонняя утилита";
      j["checkpoints"] = nlohmann::json::array({
          nlohmann::json{{"model", e.model},
                         {"name", e.name},
                         {"files", nlohmann::json::array()},
                         {"prefix_hash", "abc123"},
                         {"created", 1730000000.0},
                         {"foreign_field", "не трогать"}}});
      std::ofstream f(std::string(MODELS) + "/llmhost-checkpoints.json");
      f << j.dump();
    }
    Entry e2 = e;
    e2.name = "step8";
    REQUIRE(upsert(MODELS, e2, err), err);
    auto all = list(MODELS);
    assert(all.size() == 2);
    {
      std::ifstream f(std::string(MODELS) + "/llmhost-checkpoints.json");
      nlohmann::json j = nlohmann::json::parse(f);
      assert(j.value("written_by", "") == "посторонняя утилита");
      bool foreign_kept = false;
      for (const auto& c : j["checkpoints"])
        if (c.value("name", "") == e.name)
          foreign_kept = c.value("foreign_field", "") == "не трогать";
      assert(foreign_kept);
    }
    // Обновление той же записи: поля заменяются, чужие не вычищаются.
    e.files[0].tokens = 4322;
    REQUIRE(upsert(MODELS, e, err), err);
    {
      std::ifstream f(std::string(MODELS) + "/llmhost-checkpoints.json");
      nlohmann::json j = nlohmann::json::parse(f);
      bool foreign_kept = false;
      for (const auto& c : j["checkpoints"])
        if (c.value("name", "") == e.name)
          foreign_kept = c.value("foreign_field", "") == "не трогать";
      assert(foreign_kept);
    }
    auto again = find(MODELS, e.model, "step7");
    assert(again && again->files.size() == 2 && again->files[0].tokens == 4322 &&
           again->files[0].file == "x.ckpt.step7.s1.bin");

    // Чужая модель — чужой каталог.
    Entry other;
    other.model = "/data/models/y.gguf";
    other.name = "step7";
    other.files.push_back({1, "y.ckpt.step7.s1.bin", 10, 1});
    REQUIRE(upsert(MODELS, other, err), err);
    assert(list(MODELS, e.model).size() == 2);
    assert(list(MODELS, other.model).size() == 1);

    // Удаление по имени: чужое имя не задевается; вернувшаяся запись — та,
    // что лежала в каталоге (токены последнего обновления).
    Entry removed;
    REQUIRE(remove(MODELS, e.model, "step7", &removed, err), err);
    assert(removed.files.size() == 2 && removed.files[0].tokens == 4322);
    assert(!remove(MODELS, e.model, "step7", nullptr, err)); // второй раз — отказ
    assert(list(MODELS, e.model).size() == 1);
    assert(list(MODELS, other.model).size() == 1);

    // Удаление модели целиком.
    REQUIRE(remove_model(MODELS, e.model, err), err);
    assert(list(MODELS, e.model).empty());
    assert(list(MODELS, other.model).size() == 1);

    // Битый каталог — fail-closed.
    {
      std::ofstream f(std::string(MODELS) + "/llmhost-checkpoints.json");
      f << "{\"checkpoints\": [ сломано";
    }
    assert(list(MODELS).empty());
    assert(!upsert(MODELS, e, err));
  }

  std::puts("checkpoint_test: ok");
  return 0;
}
