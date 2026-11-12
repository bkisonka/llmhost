// Железо: видеокарты через NVML (без SDK — dlopen "libnvidia-ml.so.1");
// не-NVIDIA и системы без NVML — без телеметрии (vram = 0, планировка вслепую,
// но честно). Vulkan в системе. Выбор сборки движка: CUDA по умолчанию,
// Vulkan — запасной путь.
#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace lh::hw {

enum class Build { Vulkan, Cuda12, Cuda13, None };

struct Gpu {
  int index = 0;                      // индекс карты; для CUDA совпадает с ним
  std::string name;                   // только при CUDA_DEVICE_ORDER=PCI_BUS_ID
  std::string uuid;                   // "GPU-..." (NVML); пусто — не NVIDIA
  int cc_major = -1, cc_minor = -1;   // compute capability
  uint64_t vram_total = 0, vram_free = 0; // байты; 0 — телеметрии нет
  bool nvidia = false;
  bool vulkan = false;                // libvulkan.so.1 загружается
  Build build = Build::None;          // сборка движка для этого ПК
  uint64_t ram_total = 0, ram_avail = 0;  // системная память; 0 — не узнали
};

// Карта 0 (совместимость: существующие вызовы смотрят только на неё).
Gpu query();
// Все карты; без NVML (AMD/Intel на Linux) — одна запись с тем, что удалось
// узнать (vram = 0): планировщик живёт вслепую, честнее выдумывать цифры.
std::vector<Gpu> query_all();
// Телеметрия кэшируется на секунду (каждый /v0/status её дёргает). Точечные
// замеры «прямо сейчас» (след ребёнка в профиле) обязаны начаться с чистого
// запроса — иначе снимок наследует чужую секунду.
void flush_telemetry_cache();

const char* build_name(Build b);

} // namespace lh::hw
