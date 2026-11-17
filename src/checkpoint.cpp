#include "checkpoint.h"

#include "util.h"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <filesystem>

namespace lh::checkpoint {

namespace {

constexpr const char* kCatalogFile = "llmhost-checkpoints.json";
constexpr const char* kArrayKey = "checkpoints";

// Тот же каркас чтения/записи, что у привязок image (binding_upsert): файл
// может содержать чужие ключи верхнего уровня — писатель трогает только свой
// массив. Битый файл — fail-closed (правка по битому каталогу молча теряла
// бы записи).
bool read_catalog(const std::string& models_dir, nlohmann::json& j, std::string& err) {
  std::string raw = lh::read_small_file(catalog_path(models_dir));
  if (raw.empty()) {
    j = nlohmann::json{{kArrayKey, nlohmann::json::array()}};
    return true;
  }
  try {
    j = nlohmann::json::parse(raw);
  } catch (const nlohmann::json::exception& e) {
    err = std::string("битый ") + kCatalogFile + ": " + e.what();
    return false;
  }
  if (!j.is_object() || !j.contains(kArrayKey) || !j[kArrayKey].is_array()) {
    err = std::string(kCatalogFile) + ": ожидается {\"" + kArrayKey + "\": [...]}";
    return false;
  }
  return true;
}

bool save_catalog(const std::string& models_dir, const nlohmann::json& j, std::string& err) {
  const std::string path = catalog_path(models_dir);
  const std::string tmp = path + ".tmp." + lh::random_hex(6);
  std::error_code ec;
  bool ok = lh::write_small_file(tmp, j.dump(2) + "\n") &&
            (std::filesystem::rename(tmp, path, ec), !ec);
  if (!ok) {
    err = "запись " + path + " не удалась" + (ec ? ": " + ec.message() : "");
    std::error_code ec2;
    std::filesystem::remove(tmp, ec2);
    return false;
  }
  return true;
}

bool entry_eq(const nlohmann::json& e, const std::string& model, const std::string& name) {
  return e.is_object() && e.value("model", "") == model && e.value("name", "") == name;
}

nlohmann::json fields_of(const Entry& e) {
  nlohmann::json files = nlohmann::json::array();
  for (auto& f : e.files)
    files.push_back({{"slot", f.slot},
                     {"file", f.file},
                     {"bytes", f.bytes},
                     {"tokens", f.tokens}});
  return {{"model", e.model},
          {"name", e.name},
          {"files", std::move(files)},
          {"prefix_hash", e.prefix_hash},
          {"created", e.created}};
}

void entry_of(const nlohmann::json& e, Entry& out) {
  out.model = e.value("model", "");
  out.name = e.value("name", "");
  out.prefix_hash = e.value("prefix_hash", std::string());
  out.created = e.value("created", 0.0);
  if (e.contains("files") && e["files"].is_array())
    for (const auto& f : e["files"]) {
      if (!f.is_object()) continue;
      SlotFile sf;
      sf.slot = f.value("slot", 0);
      sf.file = f.value("file", "");
      sf.bytes = f.value("bytes", (uint64_t)0);
      sf.tokens = f.value("tokens", 0LL);
      out.files.push_back(std::move(sf));
    }
}

} // namespace

std::string sanitize_name(const std::string& raw) {
  // Незнакомый символ — в «_», не выбрасывать: выбрасывание схлопывало разные
  // имена в одно («step 7» и «step7» → «step7») и второй save молча
  // перезаписывал первый снимок. Грамматика — как sanitize_id в overrides_kv.h.
  std::string out;
  out.reserve(raw.size());
  for (unsigned char c : raw)
    out += (std::isalnum(c) || c == '.' || c == '-' || c == '_') ? (char)c : '_';
  if (out.size() > 64) out.resize(64); // потолок имени; файл — стем + «.ckpt.» + имя
  return out.empty() ? "checkpoint" : out;
}

std::string catalog_path(const std::string& models_dir) {
  return models_dir + "/" + kCatalogFile;
}

std::vector<Entry> list(const std::string& models_dir, const std::string& model) {
  std::vector<Entry> out;
  nlohmann::json j;
  std::string err;
  if (!read_catalog(models_dir, j, err)) return out;
  for (const auto& e : j[kArrayKey])
    if (e.is_object() && (model.empty() || e.value("model", "") == model)) {
      Entry en;
      entry_of(e, en);
      if (!en.name.empty()) out.push_back(std::move(en));
    }
  return out;
}

std::optional<Entry> find(const std::string& models_dir, const std::string& model,
                          const std::string& name) {
  for (auto& e : list(models_dir, model))
    if (e.name == name) return e;
  return std::nullopt;
}

bool upsert(const std::string& models_dir, const Entry& e, std::string& err) {
  if (e.model.empty() || e.name.empty() || e.files.empty()) {
    err = "чекпоинт: пустые model/name/files — запись отклонена";
    return false;
  }
  nlohmann::json j;
  if (!read_catalog(models_dir, j, err)) return false;
  nlohmann::json fields = fields_of(e);
  nlohmann::json* target = nullptr;
  for (auto& x : j[kArrayKey])
    if (entry_eq(x, e.model, e.name)) target = &x;
  if (target)
    target->update(fields); // чужие поля записи сохраняются (прецедент binding_upsert)
  else
    j[kArrayKey].push_back(std::move(fields));
  return save_catalog(models_dir, j, err);
}

bool remove(const std::string& models_dir, const std::string& model,
            const std::string& name, Entry* removed, std::string& err) {
  nlohmann::json j;
  if (!read_catalog(models_dir, j, err)) return false;
  nlohmann::json found;
  auto& arr = j[kArrayKey];
  for (size_t i = 0; i < arr.size(); ++i)
    if (entry_eq(arr[i], model, name)) {
      found = std::move(arr[i]);
      arr.erase(i);
      break;
    }
  if (found.is_null()) {
    err = "чекпоинт «" + name + "» не найден";
    return false;
  }
  if (removed) entry_of(found, *removed);
  return save_catalog(models_dir, j, err);
}

bool remove_model(const std::string& models_dir, const std::string& model, std::string& err) {
  nlohmann::json j;
  if (!read_catalog(models_dir, j, err)) return false;
  auto& arr = j[kArrayKey];
  for (size_t i = arr.size(); i > 0; --i)
    if (arr[i - 1].is_object() && arr[i - 1].value("model", "") == model) arr.erase(i - 1);
  return save_catalog(models_dir, j, err);
}

// ---- SHA256 (FIPS 180-4, компактная реализация без зависимостей) ------------

namespace {

struct Sha256 {
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  uint8_t buf[64] = {};
  uint64_t len = 0;
  size_t fill = 0;

  static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

  void block(const uint8_t* p) {
    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = (uint32_t)p[i * 4] << 24 | (uint32_t)p[i * 4 + 1] << 16 |
             (uint32_t)p[i * 4 + 2] << 8 | (uint32_t)p[i * 4 + 3];
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t t1 = hh + S1 + ch + K[i] + w[i];
      const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = S0 + maj;
      hh = g; g = f; f = e; e = d + t1;
      d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }

  void update(const std::string& data) {
    for (unsigned char ch : data) {
      buf[fill++] = ch;
      if (fill == 64) {
        block(buf);
        fill = 0;
        len += 64;
      }
    }
  }

  std::string finish() {
    const uint64_t bits = (len + fill) * 8;
    buf[fill++] = 0x80;
    if (fill > 56) {
      while (fill < 64) buf[fill++] = 0;
      block(buf);
      fill = 0;
    }
    while (fill < 56) buf[fill++] = 0;
    for (int i = 7; i >= 0; --i) buf[fill++] = (uint8_t)(bits >> (i * 8));
    block(buf);
    char out[65];
    for (int i = 0; i < 8; ++i) std::snprintf(out + i * 8, 9, "%08x", h[i]);
    return std::string(out, 64);
  }
};

} // namespace

std::string sha256_hex(const std::string& data) {
  Sha256 s;
  s.update(data);
  return s.finish();
}

std::string prefix_hash(const nlohmann::json& messages) {
  if (!messages.is_array()) return {};
  // Длина-префиксное кадрирование: границы роль/content не перепутать ни при
  // каком содержимом строк. «pfp1|» — версия канонизатора внутри хэша.
  std::string canon = "pfp1|";
  for (const auto& m : messages) {
    if (!m.is_object()) return {};
    const auto role = m.find("role");
    if (role == m.end() || !role->is_string()) return {};
    std::string text;
    const auto content = m.find("content");
    if (content != m.end() && !content->is_null()) {
      if (!content->is_string()) return {}; // мультимодальный — хэш недоступен
      text = content->get<std::string>();
    }
    const std::string& r = role->get<std::string>();
    canon += std::to_string(r.size()) + ":" + r + std::to_string(text.size()) + ":" + text;
  }
  return sha256_hex(canon);
}

} // namespace lh::checkpoint
