// Единый разбор SSE-стрима /v1/chat/completions для клиентов демона.
// Раньше протокол «data: {json} → [DONE] → delta.reasoning_content|content»
// был отрукописан в каждом клиенте (lhctl, tui; chatui — Windows-only,
// заведёт его при следующей правке Windows-репо) и расползался.
//
// Использование:
//   lh::sse::Chat sse;
//   sse.on_reasoning = [&](const std::string& r) { ... };  // «мысли»
//   sse.on_content   = [&](const std::string& c) { ... };  // ответ
//   // в content_receiver:
//   //   if (!sse.feed(d, n)) /* [DONE] */;
//   //   return true;  // разрывать соединение не нужно — сервер закроет сам
//   // при не-200: sse.raw_tail() — хвост сырого потока для error.message.
#pragma once
#include <functional>
#include <nlohmann/json.hpp>
#include <string>

namespace lh::sse {

struct Chat {
  std::function<void(const std::string&)> on_reasoning;
  std::function<void(const std::string&)> on_content;

  // Последние 64 КиБ сырого потока: ошибки приходят обычным JSON — хвост
  // уходит в разбор «error.message» при не-200.
  const std::string& raw_tail() const { return raw; }

  // Прокормить кусок сети. true — стрим жив; false — получен [DONE].
  bool feed(const char* d, size_t n) {
    raw.append(d, n);
    if (raw.size() > (64u << 10)) raw.erase(0, raw.size() - (64u << 10));
    sse.append(d, n);
    size_t nl;
    while ((nl = sse.find('\n')) != std::string::npos) {
      std::string line = sse.substr(0, nl);
      sse.erase(0, nl + 1);
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (line.rfind("data: ", 0) != 0) continue;
      std::string payload = line.substr(6);
      if (payload == "[DONE]") return false;
      try {
        // Чанк живёт в локали: auto& от json::parse(...).at(...) —
        // ссылка на временку, UB на первом же обращении.
        auto chunk = nlohmann::json::parse(payload);
        auto& delta = chunk.at("choices").at(0).at("delta");
        std::string r2 = delta.value("reasoning_content", std::string());
        if (!r2.empty() && on_reasoning) on_reasoning(r2);
        std::string c = delta.value("content", std::string());
        if (!c.empty() && on_content) on_content(c);
      } catch (...) {} // битый кусок стрима не должен ронять диалог
    }
    return true;
  }

 private:
  std::string raw, sse;
};

} // namespace lh::sse
