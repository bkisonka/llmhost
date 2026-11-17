// Парковка ждущих запросов (очередь пробуждения harness-режима). Запрос к
// нерезидентной модели, которой мешают занятые резиденты, занимает место на
// «парковке» с ключом-моделью и спит до notify / TTL. Глубина ограничена
// НА КЛЮЧ: толпа ждущих одну модель не должна есть лимит целиком (чужие
// модели в очереди не виноваты) — сверх лимита немедленный отказ, а не
// бесконечная память под ждунами. Потокобезопасно, без зависимостей —
// тестируется в отрыве от супервизора.
//
// Ожидание — по поколениям: notify_all увеличивает счётчик, wait будится,
// когда счётчик ушёл вперёд снимка вызывающего, и обновляет снимок. Так нет
// ни потерянного будильника (notify между «попытался и не вышло» и «ляг
// спать» не пропадает — поколение уже ушло), ни тугой петли (услышанный
// notify не заставляет wait вернуться мгновенно второй раз).
#pragma once
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <list>
#include <map>
#include <mutex>
#include <string>

namespace lh::park {

class Lot {
public:
  explicit Lot(size_t depth_max) : depth_max_(depth_max) {}

  // Итог ожидания: проснулся по notify (можно пробовать снова) или по TTL.
  enum class Wake { Notified, Expired };

  // Занять место. Адрес тикета — паспорт места: его ждут в wait() и им же
  // отпускают в leave(). nullptr — места для ключа кончилось (глубина).
  const void* enter(const std::string& key) {
    std::lock_guard l(m_);
    if (per_key_[key] >= depth_max_) return nullptr;
    ++per_key_[key];
    waiters_.push_back(Ticket{key});
    return &waiters_.back();
  }

  // Отпустить место — в любом исходе ожидания (RAII-обёртка на стороне
  // вызывающего). nullptr безопасен (место не занималось).
  void leave(const void* ticket) {
    if (!ticket) return;
    std::lock_guard l(m_);
    for (auto it = waiters_.begin(); it != waiters_.end(); ++it) {
      if (&*it != ticket) continue;
      --per_key_[it->key];
      if (per_key_[it->key] == 0) per_key_.erase(it->key);
      waiters_.erase(it);
      return;
    }
  }

  // Спать, пока счётчик поколений не уйдёт вперёд seen_gen, или до deadline.
  // seen_gen — снимок вызывающего: метод его обновляет (паттерн «передай и
  // получи обратно»), цикл повторных попыток просто передаёт тот же uint64_t.
  // Expired — TTL истёк.
  Wake wait(const void* ticket, std::chrono::steady_clock::time_point deadline,
            uint64_t& seen_gen) {
    std::unique_lock l(m_);
    auto mine = std::find_if(waiters_.begin(), waiters_.end(),
                             [&](const Ticket& t) { return &t == ticket; });
    if (mine == waiters_.end()) return Wake::Expired; // leave() уже был
    while (gen_ == seen_gen) {
      if (cv_.wait_until(l, deadline) == std::cv_status::timeout)
        return Wake::Expired;
    }
    seen_gen = gen_;
    return Wake::Notified;
  }

  // Разбудить всех: ждущие пересчитают своё условие и либо уйдут делать
  // дело, либо лягут снова. Экономии на «разбудить одного» нет сознательно:
  // будим всех при каждом изменении занятости — пробуждение дешёвое, а
  // ошибок с потерянным персональным будильником не бывает.
  void notify_all() {
    {
      std::lock_guard l(m_);
      ++gen_;
    }
    cv_.notify_all();
  }

  // Снимок поколения для первого wait() (дальше wait сам обновляет снимок).
  uint64_t generation_snapshot() const {
    std::lock_guard l(m_);
    return gen_;
  }

  size_t depth(const std::string& key) const {
    std::lock_guard l(m_);
    auto it = per_key_.find(key);
    return it == per_key_.end() ? 0 : it->second;
  }

  // Общая глубина (все ключи) — для метрик.
  size_t depth_total() const {
    std::lock_guard l(m_);
    size_t n = 0;
    for (auto& [k, v] : per_key_) n += v;
    return n;
  }

private:
  struct Ticket {
    std::string key;
  };
  mutable std::mutex m_;
  std::condition_variable cv_;
  std::list<Ticket> waiters_;            // адреса стабильны: erase только в leave
  std::map<std::string, size_t> per_key_;
  size_t depth_max_;
  uint64_t gen_ = 0;
};

} // namespace lh::park
