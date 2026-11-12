// Юнит-тесты паркинга (lh::park::Lot): глубина на ключ, отпускание места,
// TTL, пробуждение по notify, отсутствие тугой петли после notify.
#include "park.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

using lh::park::Lot;
using Wake = Lot::Wake;

static int g_failed = 0;
#define CHECK(cond)                                                          \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);            \
      ++g_failed;                                                            \
    }                                                                        \
  } while (0)

struct SlotGuard {
  Lot& lot;
  const void* t;
  explicit SlotGuard(Lot& l, const void* ticket) : lot(l), t(ticket) {}
  ~SlotGuard() { lot.leave(t); }
};

static void test_depth_per_key() {
  Lot lot(2);
  SlotGuard a1(lot, lot.enter("A"));
  CHECK(a1.t);
  SlotGuard a2(lot, lot.enter("A"));
  CHECK(a2.t);
  CHECK(lot.depth("A") == 2);
  CHECK(lot.enter("A") == nullptr); // третий на том же ключе — отказ
  SlotGuard b1(lot, lot.enter("B"));
  CHECK(b1.t);                      // чужой ключ не конкурирует
  CHECK(lot.depth_total() == 3);
}

static void test_leave_releases() {
  Lot lot(1);
  {
    SlotGuard g(lot, lot.enter("A"));
    CHECK(g.t);
    CHECK(lot.enter("A") == nullptr);
  } // guard отпустил
  CHECK(lot.enter("A") != nullptr); // место снова свободно
  CHECK(lot.depth("A") == 1);
  lot.leave(nullptr); // не падает
}

static void test_ttl_expires() {
  Lot lot(4);
  const void* t = lot.enter("A");
  CHECK(t);
  uint64_t gen = lot.generation_snapshot();
  auto t0 = std::chrono::steady_clock::now();
  Wake w = lot.wait(t, std::chrono::steady_clock::now() + std::chrono::milliseconds(120),
                    gen);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0)
                .count();
  CHECK(w == Wake::Expired);
  CHECK(ms >= 100 && ms < 2000);
  lot.leave(t);
}

static void test_notify_wakes() {
  Lot lot(4);
  const void* t = lot.enter("A");
  CHECK(t);
  std::atomic<bool> woke{false};
  std::thread th([&] {
    uint64_t gen = lot.generation_snapshot();
    Wake w = lot.wait(t, std::chrono::steady_clock::now() + std::chrono::seconds(5), gen);
    woke = w == Wake::Notified;
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  lot.notify_all();
  th.join();
  CHECK(woke.load());
  lot.leave(t);
}

static void test_no_busy_loop_after_notify() {
  // Прослушанный notify не должен ронять следующий wait мгновенно:
  // после notify второе ожидание с тем же снимком спит до TTL.
  Lot lot(4);
  const void* t = lot.enter("A");
  CHECK(t);
  uint64_t gen = lot.generation_snapshot();
  lot.notify_all();
  Wake w1 = lot.wait(t, std::chrono::steady_clock::now() + std::chrono::seconds(5), gen);
  CHECK(w1 == Wake::Notified); // поколение уже ушло — вернулись сразу
  CHECK(gen == 1);
  auto t0 = std::chrono::steady_clock::now();
  Wake w2 = lot.wait(t, std::chrono::steady_clock::now() + std::chrono::milliseconds(120),
                     gen); // тот же снимок: спим честно
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0)
                .count();
  CHECK(w2 == Wake::Expired);
  CHECK(ms >= 100 && ms < 2000);
  lot.leave(t);
}

static void test_leave_wakes_nothing_but_safe() {
  // leave несуществующего тикета — no-op; двойной leave — no-op.
  Lot lot(1);
  lot.leave(reinterpret_cast<const void*>(1));
  const void* t = lot.enter("A");
  lot.leave(t);
  lot.leave(t);
  CHECK(lot.depth_total() == 0);
  CHECK(lot.enter("A") != nullptr);
}

int main() {
  test_depth_per_key();
  test_leave_releases();
  test_ttl_expires();
  test_notify_wakes();
  test_no_busy_loop_after_notify();
  test_leave_wakes_nothing_but_safe();
  std::printf(g_failed ? "park_test: FAILED (%d)\n" : "park_test: ok\n", g_failed);
  return g_failed ? 1 : 0;
}
