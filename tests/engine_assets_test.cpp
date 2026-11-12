// Достройка сборок манифеста из ассетов релиза: паттерны, скоуп платформы,
// версии, companion, idempotency по known-id.
#include "engine_assets.h"

#include <cassert>
#include <cstdio>
#include <string>

using namespace lh::engine;

static AssetIndex b12000_assets() {
  return {
      {"llama-b12000-bin-ubuntu-vulkan-x64.tar.gz", {35471500, "aaa"}},
      {"llama-b12000-bin-ubuntu-cuda-12.8-x64.tar.gz", {182526908, "bbb"}},
      {"cudart-llama-b12000-bin-ubuntu-cuda-12.8-x64.tar.gz", {594371567, "ccc"}},
      {"llama-b12000-bin-ubuntu-cuda-13.4-x64.tar.gz", {190000000, "ddd"}},
      {"cudart-llama-b12000-bin-ubuntu-cuda-13.4-x64.tar.gz", {595000000, "eee"}},
      // чужие паттерны — не подмешиваются никогда
      {"llama-b12000-bin-ubuntu-sycl-fp16-x64.tar.gz", {300000000, "fff"}},
      {"llama-b12000-bin-ubuntu-rocm-10.0-x64.tar.gz", {400000000, "ggg"}},
      {"llama-b12000-bin-ubuntu-arm64.tar.gz", {100000000, "hhh"}},
  };
}

static nlohmann::json ubuntu_build(const std::string& id, const std::string& tag,
                                   const std::string& name) {
  return {
      {"id", id},
      {"files", nlohmann::json::array({
                    {{"url", "https://github.com/ggml-org/llama.cpp/releases/"
                             "download/" + tag + "/" + name},
                     {"sha256", "old-" + id},
                     {"size", 1}},
                })},
  };
}

static nlohmann::json seed() { // наш живой сид: vulkan + cuda12
  return nlohmann::json::array({
      ubuntu_build("vulkan", "b11963", "llama-b11963-bin-ubuntu-vulkan-x64.tar.gz"),
      ubuntu_build("cuda12", "b11963", "llama-b11963-bin-ubuntu-cuda-12.8-x64.tar.gz"),
  });
}

int main() {
  // 1) cuda13 достраивается; известные id не трогаются; чужие паттерны мимо
  {
    AddedBuilds r = builds_from_assets(b12000_assets(), seed(), "b12000");
    assert(r.added == "cuda13");
    assert(r.builds.size() == 3);
    const auto& cb = r.builds[2];
    assert(cb["id"] == "cuda13");
    assert(cb["files"].size() == 2);
    assert(cb["files"][0]["url"] ==
           "https://github.com/ggml-org/llama.cpp/releases/download/b12000/"
           "llama-b12000-bin-ubuntu-cuda-13.4-x64.tar.gz");
    assert(cb["files"][0]["sha256"] == "ddd");
    assert(cb["files"][0]["size"] == 190000000u);
    assert(cb["files"][1]["url"].get<std::string>().find(
               "cudart-llama-b12000-bin-ubuntu-cuda-13.4-x64.tar.gz") !=
           std::string::npos);
    assert(cb["files"][1]["sha256"] == "eee");
  }
  // 2) companion-тарболл отсутствует — сборка целиком не добавляется
  {
    AssetIndex a = b12000_assets();
    a.erase("cudart-llama-b12000-bin-ubuntu-cuda-13.4-x64.tar.gz");
    AddedBuilds r = builds_from_assets(a, seed(), "b12000");
    assert(r.added.empty());
    assert(r.builds.size() == 2);
  }
  // 3) сид без cuda — обе ветки, по возрастанию (авто-подбор берёт старшую
  //    совместимую с драйвером первой)
  {
    nlohmann::json s = seed();
    s.erase(1);
    AddedBuilds r = builds_from_assets(b12000_assets(), s, "b12000");
    assert(r.added == "cuda12, cuda13");
    assert(r.builds.size() == 3);
    assert(r.builds[1]["id"] == "cuda12");
    assert(r.builds[2]["id"] == "cuda13");
  }
  // 4) vulkan в сиде нет — достраивается он, cuda не дублируется
  {
    nlohmann::json s = nlohmann::json::array({seed()[1]});
    AssetIndex a;
    a["llama-b12000-bin-ubuntu-vulkan-x64.tar.gz"] = {35471500, "aaa"};
    a["llama-b12000-bin-ubuntu-cuda-12.8-x64.tar.gz"] = {182526908, "bbb"};
    a["cudart-llama-b12000-bin-ubuntu-cuda-12.8-x64.tar.gz"] = {594371567, "ccc"};
    AddedBuilds r = builds_from_assets(a, s, "b12000");
    assert(r.added == "vulkan");
    assert(r.builds.size() == 2);
    assert(r.builds[0]["id"] == "cuda12");
    assert(r.builds[1]["id"] == "vulkan");
  }
  // 5) win-сид — скоуп вин-репо, нам мимо
  {
    nlohmann::json s = nlohmann::json::array({ubuntu_build(
        "vulkan", "b11963", "llama-b11963-bin-win-vulkan-x64.zip")});
    AddedBuilds r = builds_from_assets(b12000_assets(), s, "b12000");
    assert(r.added.empty());
    assert(r.builds.size() == 1);
  }
  // 6) платформа не определилась (нет url) — не трогаем
  {
    nlohmann::json s = nlohmann::json::array(
        {{{"id", "x"}, {"files", nlohmann::json::array()}}});
    AddedBuilds r = builds_from_assets(b12000_assets(), s, "b12000");
    assert(r.added.empty());
  }
  // 7) cuda13 уже в сиде — дубля нет
  {
    nlohmann::json s = seed();
    s.push_back(ubuntu_build("cuda13", "b11963",
                             "llama-b11963-bin-ubuntu-cuda-13.4-x64.tar.gz"));
    AddedBuilds r = builds_from_assets(b12000_assets(), s, "b12000");
    assert(r.added.empty());
    assert(r.builds.size() == 3);
  }
  // 8) кривые формы версии пропускаются молча
  {
    AssetIndex a;
    a["llama-b12000-bin-ubuntu-cuda-13-x64.tar.gz"] = {1, "x"};     // без минора
    a["llama-b12000-bin-ubuntu-cuda-13.4.1-x64.tar.gz"] = {1, "x"}; // три части
    a["llama-b12000-bin-ubuntu-cuda-v13.4-x64.tar.gz"] = {1, "x"};  // не цифры
    AddedBuilds r = builds_from_assets(a, seed(), "b12000");
    assert(r.added.empty());
  }
  // 9) ассет чужого тега не матчится
  {
    AssetIndex a;
    a["llama-b11999-bin-ubuntu-cuda-13.4-x64.tar.gz"] = {1, "x"};
    a["cudart-llama-b11999-bin-ubuntu-cuda-13.4-x64.tar.gz"] = {1, "x"};
    AddedBuilds r = builds_from_assets(a, seed(), "b12000");
    assert(r.added.empty());
  }
  // 10) builds не массив — вход без изменений
  {
    AddedBuilds r =
        builds_from_assets(b12000_assets(), nlohmann::json::object(), "b12000");
    assert(r.added.empty());
    assert(r.builds.is_object());
  }
  std::printf("engine_assets: ok\n");
  return 0;
}
