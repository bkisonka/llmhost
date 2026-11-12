#include "hw.h"

#include "util.h"

#ifdef _WIN32
#include <windows.h>
#endif

#ifndef _WIN32
#include <dlfcn.h>
#endif

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <chrono>
#include <mutex>
#include <utility>

namespace lh::hw {

namespace {

typedef int nvmlReturn_t;
typedef void* nvmlDevice_t;
struct nvmlMemory_st { unsigned long long total, free, used; };

typedef nvmlReturn_t (*FnInit)();
typedef nvmlReturn_t (*FnShutdown)();
typedef nvmlReturn_t (*FnCount)(unsigned*);
typedef nvmlReturn_t (*FnHandle)(unsigned, nvmlDevice_t*);
typedef nvmlReturn_t (*FnName)(nvmlDevice_t, char*, unsigned);
typedef nvmlReturn_t (*FnCc)(nvmlDevice_t, int*, int*);
typedef nvmlReturn_t (*FnMem)(nvmlDevice_t, nvmlMemory_st*);
typedef nvmlReturn_t (*FnUuid)(nvmlDevice_t, char*, unsigned);

struct Nvml {
#ifdef _WIN32
  HMODULE dll = nullptr;
#else
  void* dll = nullptr;
#endif
  FnInit init = nullptr;
  FnShutdown shutdown = nullptr;
  FnCount count = nullptr;
  FnHandle handle = nullptr;
  FnName name = nullptr;
  FnCc cc = nullptr;
  FnMem mem = nullptr;
  FnUuid uuid = nullptr;
  bool ok = false;
};

Nvml load_nvml() {
  Nvml n;
#ifdef _WIN32
  // Только системные каталоги: nvml.dll рядом с exe (флэшка!) исполнился бы
  // внутри llmhost — классическая DLL-подмена.
  n.dll = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (!n.dll) {
    // Старые драйверы держали nvml.dll не в System32 (абсолютный путь).
    wchar_t buf[MAX_PATH + 1] = {};
    UINT k = GetEnvironmentVariableW(L"ProgramFiles", buf, MAX_PATH);
    if (k && k <= MAX_PATH) {
      std::wstring pf =
          std::wstring(buf) + L"\\NVIDIA Corporation\\NVSMI\\nvml.dll";
      n.dll = LoadLibraryExW(pf.c_str(), nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    }
  }
  if (!n.dll) return n;
  auto req = [&](const char* a, const char* b) -> void* {
    FARPROC p = GetProcAddress(n.dll, a);
    return (void*)(p ? p : GetProcAddress(n.dll, b));
  };
#else
  // Драйвер кладёт библиотеку в системные каталоги ldconfig'а; она же
  // тянется по кэшу линкера — подмена чужим файлом отсечена rpath/путями.
  n.dll = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!n.dll) n.dll = dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_LOCAL); // dev- symlink
  if (!n.dll) return n;
  auto req = [&](const char* a, const char* b) -> void* {
    void* p = dlsym(n.dll, a);
    return p ? p : dlsym(n.dll, b);
  };
#endif
  n.init = (FnInit)req("nvmlInit_v2", "nvmlInit");
  n.shutdown = (FnShutdown)req("nvmlShutdown", "nvmlShutdown");
  n.count = (FnCount)req("nvmlDeviceGetCount_v2", "nvmlDeviceGetCount");
  n.handle = (FnHandle)req("nvmlDeviceGetHandleByIndex_v2", "nvmlDeviceGetHandleByIndex");
  n.name = (FnName)req("nvmlDeviceGetName", "nvmlDeviceGetName");
  n.cc = (FnCc)req("nvmlDeviceGetCudaComputeCapability", "nvmlDeviceGetCudaComputeCapability");
  n.mem = (FnMem)req("nvmlDeviceGetMemoryInfo", "nvmlDeviceGetMemoryInfo");
  n.uuid = (FnUuid)req("nvmlDeviceGetUUID", "nvmlDeviceGetUUID");
  n.ok = n.init && n.shutdown && n.count && n.handle && n.name && n.cc && n.mem && n.uuid;
  return n;
}

bool cc_at_least(const Gpu& g, int major, int minor) {
  return g.cc_major > major || (g.cc_major == major && g.cc_minor >= minor);
}

// Системная память: MemTotal/MemAvailable из /proc/meminfo, значения в кБ.
bool read_meminfo(unsigned long long& total, unsigned long long& avail) {
  std::ifstream f("/proc/meminfo");
  if (!f) return false;
  bool got_t = false, got_a = false;
  std::string key;
  unsigned long long val = 0;
  std::string unit;
  while (f >> key >> val >> unit) {
    if (key == "MemTotal:") { total = val * 1024ULL; got_t = true; }
    else if (key == "MemAvailable:") { avail = val * 1024ULL; got_a = true; }
    if (got_t && got_a) return true;
  }
  return got_t; // MemAvailable нет на древних ядрах — тогда avail остаётся 0
}

} // namespace

const char* build_name(Build b) {
  switch (b) {
    case Build::Vulkan: return "vulkan";
    case Build::Cuda12: return "cuda12";
    case Build::Cuda13: return "cuda13";
    case Build::None: return "none";
  }
  return "none";
}

// Кэш на секунду: query зовёт каждый /v0/status и супервизор (vram_settle),
// а NVML init/shutdown на каждый чих — всё под общим мьютексом.
namespace {
struct HwCache {
  bool has = false;
  std::chrono::steady_clock::time_point at;
  Gpu card0;
  std::vector<Gpu> all;
};
HwCache g_cache;
std::mutex g_cache_mtx;

void refresh_cache() {
  // LH_NO_NVML=1 — притвориться машиной без NVML: проверка фолбэк-пути
  // телеметрии на NVIDIA-машине (на AMD/Intel он и так основной).
  const bool no_nvml = ::getenv("LH_NO_NVML") != nullptr;
  static Nvml real_nvml = load_nvml();
  static Nvml fake_nvml; // ok=false — все ветки NVML мимо
  Nvml& n = no_nvml ? fake_nvml : real_nvml;
  g_cache.all.clear();

#ifdef _WIN32
  // vulkan-1.dll — загрузчик Vulkan из System32; грузим только для проверки.
  HMODULE vulkan = LoadLibraryExW(L"vulkan-1.dll", nullptr,
                                  LOAD_LIBRARY_SEARCH_SYSTEM32);
  const bool vk = vulkan != nullptr;
  if (vulkan) FreeLibrary(vulkan);

  unsigned long long ram_total = 0, ram_avail = 0;
  MEMORYSTATUSEX ms{};
  ms.dwLength = sizeof(ms);
  if (GlobalMemoryStatusEx(&ms)) {
    ram_total = ms.ullTotalPhys;
    ram_avail = ms.ullAvailPhys;
  }
#else
  void* vulkan = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
  const bool vk = vulkan != nullptr;
  if (vulkan) dlclose(vulkan);

  unsigned long long ram_total = 0, ram_avail = 0;
  read_meminfo(ram_total, ram_avail);
#endif

  // NVML не любит параллельные init/shutdown из разных потоков — мы под
  // мьютексом кэша, так что серии вызовов сериализованы.
  if (n.ok && n.init() == 0) {
    unsigned cnt = 0;
    if (n.count(&cnt) == 0) {
      for (unsigned i = 0; i < cnt; ++i) {
        nvmlDevice_t d = nullptr;
        if (n.handle(i, &d) != 0) continue;
        Gpu g;
        g.index = (int)i;
        g.nvidia = true;
        char nm[96] = {};
        if (n.name(d, nm, sizeof(nm)) == 0) g.name = nm;
        char uu[128] = {};
        if (n.uuid(d, uu, sizeof(uu)) == 0) g.uuid = uu;
        n.cc(d, &g.cc_major, &g.cc_minor);
        nvmlMemory_st mem{};
        if (n.mem(d, &mem) == 0) { g.vram_total = mem.total; g.vram_free = mem.free; }
        g.vulkan = vk;
        g.ram_total = ram_total;
        g.ram_avail = ram_avail;
        // CUDA 13 не запускается на картах старше Turing (CC < 7.5) — им CUDA 12.
        g.build = vk ? Build::Vulkan
                     : (g.nvidia ? (cc_at_least(g, 7, 5) ? Build::Cuda13 : Build::Cuda12)
                                 : Build::None);
        g_cache.all.push_back(g);
      }
    }
    n.shutdown();
  }
  if (g_cache.all.empty()) {
    // Без NVML (AMD/Intel, либо NVIDIA с испорченным NVML) телеметрии нет:
    // одна запись с тем, что знаем, vram = 0 — вслепую честнее выдуманных
    // цифр. Планировщик на такой карте считает по файлу модели и RAM-гейтам.
    Gpu g;
    g.vulkan = vk;
    g.ram_total = ram_total;
    g.ram_avail = ram_avail;
    g.build = vk ? Build::Vulkan : Build::None;
    g_cache.all.push_back(g);
  }
  // Отладочный крюк: проверить раскладку мульти-GPU без второго железа —
  // клонируем реальную карту. Только для ручных тестов (LH_FORCE_GPU_COUNT=2..8).
  {
    const char* buf = ::getenv("LH_FORCE_GPU_COUNT");
    if (buf && g_cache.all.size() == 1) {
      const int want = atoi(buf);
      if (want > 1 && want <= 8) {
        const Gpu real = g_cache.all[0];
        for (int i = 1; i < want; ++i) {
          Gpu clone = real;
          clone.index = i;
          clone.uuid += "-clone";
          g_cache.all.push_back(clone);
        }
      }
    }
  }
  g_cache.card0 = g_cache.all[0];
  g_cache.has = true;
  g_cache.at = std::chrono::steady_clock::now();
}

HwCache snapshot() {
  std::lock_guard l(g_cache_mtx);
  auto now = std::chrono::steady_clock::now();
  if (!g_cache.has || now - g_cache.at >= std::chrono::seconds(1)) refresh_cache();
  return g_cache;
}
} // namespace

Gpu query() {
  return snapshot().card0;
}

std::vector<Gpu> query_all() {
  return snapshot().all;
}

void flush_telemetry_cache() {
  std::lock_guard l(g_cache_mtx);
  g_cache.has = false;
}

} // namespace lh::hw
