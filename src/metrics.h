// Агрегатор наблюдаемости прокси + рендер Prometheus-текста для /v0/metrics.
// Прокси видит каждый запрос — здесь только счётчики; телеметрия карт и
// состояния моделей приходят в render() уже собранными (control).
#pragma once
#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>

namespace lh::metrics {

// usage из ответа движка; seen — встретился ли хоть один из ключей
// (OpenAI-имена или сырые tokens_evaluated/tokens_predicted).
struct Usage {
  uint64_t prompt = 0, completion = 0;
  bool seen = false;
};

// Копит хвост потока (последние КиБ): llama.cpp кладёт usage в финальном
// SSE-событии и в конце обычного JSON — хвоста хватает даже когда событие
// разрезано между чанками. Потокобезопасно: feed зовёт читатель ответа,
// finish — поток завершения.
class UsageSniffer {
public:
  void feed(const char* data, size_t n);
  Usage finish();

private:
  std::string tail_; // последние kTail байт потока
};

// Итог одного запроса прокси.
struct Outcome {
  const char* endpoint = "other"; // chat | completion | embeddings | other
  const char* kind = "ok";        // ok | upstream_error | unreachable |
                                  // client_gone | auth | bad_request | unavailable
  int status = 200;
  double seconds = 0;
  uint64_t bytes = 0;
  Usage usage;
};

void observe(const std::string& model, const Outcome& o);

// Очередь пробуждения (harness-режим). depth зовёт супервизор при входе/выходе
// в парковку (текущее число ждущих на модель), wait — по завершении ожидания
// (сколько просидел), timeout — при честном отказе по TTL.
void queue_depth(const std::string& model, int waiting);
void queue_wait(const std::string& model, uint64_t ms);
void queue_timeout(const std::string& model);

// Имя семейства по пути запроса (кардинальность меток — константа).
const char* endpoint_of(const std::string& target);

// Разбор prometheus-текста ребёнка (llamacpp:*): ключ без меток → значение.
// Метки отбрасываются — у ребёнка они константы его собственной телеметрии.
nlohmann::json parse_engine(const std::string& text);

// Порог выборки sustained-скорости: тот же, что у замеров в профилях форм
// (gen_n ≥ 128) — короткий сэмпл в пару не годится.
constexpr int kSustainMinTokens = 128;

// Дельта-сэмплер sustained-скоростей: копит
// предыдущее значение пожизненных счётчиков ребёнка и выдаёт скорость окна,
// когда сессия декодировала хотя бы kSustainMinTokens токенов. Скорость —
// Δтокенов / Δсекунд декодирования движка: паузы между запросами в число не
// попадают (сэмпл сравним с warmup-замером чистого декодирования). Смена
// pid или отскок счётчиков назад — состояние ребёнка свежее, накопление с
// нуля. Окно без прироста — молчание, база НЕ сдвигается (медленная сессия
// копит несколько тиков).
class SustainDelta {
public:
  struct Sample {
    double tps = 0;
    int n = 0; // токенов окна (0 — сэмпла нет)
  };
  Sample feed(long long pid, double tokens_now, double dec_secs_now);
  void reset();

private:
  long long pid_ = -1;
  double tokens_ = 0;
  double secs_ = 0;
  bool has_ = false;
};

// Prometheus-текст. status — sup::status_json() (карты, version),
// models_state — sup::models_state() (живые состояния по id),
// engine — телеметрия движков от control (sup::engine_metrics(): массив
// {model, series} с /metrics живых детей; пусто — серии не пишутся).
std::string render(const nlohmann::json& status, const nlohmann::json& models_state,
                   const nlohmann::json& engine = nlohmann::json());

} // namespace lh::metrics
