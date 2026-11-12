// Достройка сборок манифеста из ассетов релиза (чистая функция, без сети).
// Подстановка тегов в pin() обновляет только то, что в манифесте уже есть;
// новая вариата апстрима (вторая cuda-ветка — 12.8 и 13.4 живут в релизе
// одновременно) без достройки вечно оставалась бы сборкой-сиротой: «есть в
// релизе, в манифест не попадёт». Скоуп — linux-паттерны имён ubuntu-x64:
// win-сид достраивает вин-репо своим кодом (его паттерны другие), сид без
// ubuntu-url — платформа неизвестна — не трогаем вовсе. sycl/rocm/openvino/
// arm64 мы сознательно не раздаём — паттернов для них нет.
#pragma once
#include <nlohmann/json.hpp>

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace lh::engine {

// Имя ассета релиза → {размер, sha256-hex}. Заполняет pin() из digest'ов
// GitHub API — в индекс попадают только ассеты с sha256-digest'ом.
using AssetIndex = std::map<std::string, std::pair<uint64_t, std::string>>;

struct AddedBuilds {
  nlohmann::json builds; // массив builds «как станет» (вход, если added пуст)
  std::string added;     // достроенные id, «, »-разделённые
};

inline AddedBuilds builds_from_assets(const AssetIndex& assets,
                                      const nlohmann::json& builds,
                                      const std::string& new_tag) {
  AddedBuilds out;
  out.builds = builds;
  if (!builds.is_array()) return out;

  // Скоуп по платформе текущего манифеста.
  bool win = false, ubuntu = false;
  for (const auto& b : builds) {
    if (!b.is_object()) continue;
    for (const auto& f : b.value("files", nlohmann::json::array())) {
      std::string url = f.value("url", std::string());
      if (url.find("-win-") != std::string::npos) win = true;
      if (url.find("-ubuntu-") != std::string::npos) ubuntu = true;
    }
  }
  if (win || !ubuntu) return out;

  auto known = [&](const std::string& id) {
    for (const auto& b : out.builds)
      if (b.is_object() && b.value("id", std::string()) == id) return true;
    return false;
  };
  const std::string base =
      "https://github.com/ggml-org/llama.cpp/releases/download/" + new_tag + "/";
  // files собираются локально: ассет не нашёлся — сборка целиком не
  // добавляется (без companion cublas/cudart ggml-cuda не грузится).
  auto add_build = [&](const std::string& id,
                       const std::vector<std::string>& names) {
    nlohmann::json files = nlohmann::json::array();
    for (const auto& name : names) {
      auto it = assets.find(name);
      if (it == assets.end() || it->second.second.empty()) return false;
      files.push_back({{"url", base + name},
                       {"sha256", it->second.second},
                       {"size", it->second.first}});
    }
    out.builds.push_back({{"id", id}, {"files", files}});
    if (!out.added.empty()) out.added += ", ";
    out.added += id;
    return true;
  };

  // vulkan: llama-<tag>-bin-ubuntu-vulkan-x64.tar.gz
  if (!known("vulkan"))
    add_build("vulkan", {"llama-" + new_tag + "-bin-ubuntu-vulkan-x64.tar.gz"});

  // cuda-<V>-x64: id = cuda<мажор>, по возрастанию версии — авто-подбор
  // (первая пиннутая cuda) берёт самую старую: новые CUDA требуют более
  // новых драйверов. Версии строго <мажор>.<минор>: не знали форму — пропуск.
  std::map<std::pair<int, int>, std::string> vers;
  const std::string marker = "-bin-ubuntu-cuda-";
  const std::string suffix = "-x64.tar.gz";
  for (const auto& [name, szdig] : assets) {
    if (name.rfind("llama-" + new_tag + "-", 0) != 0) continue;
    size_t m = name.find(marker);
    if (m == std::string::npos) continue;
    std::string tail = name.substr(m + marker.size());
    if (tail.size() <= suffix.size() ||
        tail.compare(tail.size() - suffix.size(), suffix.size(), suffix) != 0)
      continue;
    std::string v = tail.substr(0, tail.size() - suffix.size());
    size_t dot = v.find('.');
    if (dot == std::string::npos || v.find('.', dot + 1) != std::string::npos)
      continue;
    std::string maj_s = v.substr(0, dot), min_s = v.substr(dot + 1);
    if (maj_s.empty() || min_s.empty()) continue;
    bool digits = true;
    for (char c : maj_s) digits &= isdigit((unsigned char)c) != 0;
    for (char c : min_s) digits &= isdigit((unsigned char)c) != 0;
    if (!digits) continue;
    int maj = std::atoi(maj_s.c_str());
    if (maj <= 0) continue;
    vers[{maj, std::atoi(min_s.c_str())}] = v;
  }
  for (const auto& [vmajmin, v] : vers) {
    std::string id = "cuda" + std::to_string(vmajmin.first);
    if (known(id)) continue;
    add_build(id, {"llama-" + new_tag + "-bin-ubuntu-cuda-" + v + "-x64.tar.gz",
                   "cudart-llama-" + new_tag + "-bin-ubuntu-cuda-" + v +
                       "-x64.tar.gz"});
  }
  return out;
}

} // namespace lh::engine
