cmake_minimum_required(VERSION 3.20)
project(llmhost C CXX)

# Заморозка beta 12: без SO_REUSEPORT два демона на одном порту — честный EADDRINUSE.
add_compile_definitions(LH_NO_REUSEPORT)
# Заморозка beta 12: format(printf) на logf + ошибки формата —
# std::string в varargs и счёт аргументов ловятся компилятором.
add_compile_options(-Werror=format)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

if (WIN32)
  add_compile_definitions(WIN32_LEAN_AND_MEAN NOMINMAX UNICODE _UNICODE _CRT_SECURE_NO_WARNINGS)
  # Статический рантайм (/MT): публичный zip должен запускаться на чистой
  # Windows без Visual C++ Redistributable (проверено: /MD импортировал
  # VCRUNTIME140.dll). Рост exe ~0,3 МБ — несущественно.
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# Статическая линковка релизных бинарей (Linux): аналогично /MT на Windows
# выше — тарболл запускается на любом дистрибутиве без системных зависимостей.
# В процессе весь HTTP ходит на числовые адреса, DNS/TLS делает внешний curl,
# dlopen(NVML/Vulkan) из статического glibc работает (проверено смоком).
option(LLMHOST_STATIC "Статически линковать llmhostd и lhctl (Linux-релиз)" OFF)

# Ядро без UI: демон llmhostd и (на Windows) GUI-приложение собирают одни и те
# же файлы. На Linux UI-файлов нет вовсе: tray/chatui/catalogui/wizard живут
# только в Windows-репо.
set(CORE_SRC
  src/util.cpp
  src/config.cpp
  src/gguf.cpp
  src/modelreg.cpp
  src/image.cpp
  src/mtp.cpp
  src/hw.cpp
  src/fit.cpp
  src/catalog.cpp
  src/proc.cpp
  src/download.cpp
  src/engine.cpp
  src/bench.cpp
  src/pull.cpp
  src/sup_plan.cpp
  src/sup_overrides.cpp
  src/sup_spawn.cpp
  src/sup_draft.cpp
  src/sup_status.cpp
  src/sup_state.cpp
  src/sup_harness.cpp
  src/sup_watch.cpp
  src/sup_image.cpp
  src/supervisor.cpp
  src/profiles.cpp
  src/proxy.cpp
  src/control.cpp
  src/metrics.cpp
)

# Headless-демон: держит llama-server и даёт системе OpenAI-совместимый
# endpoint (:openai_port) и control API (:control_port).
add_executable(llmhostd
  src/daemond.cpp
  src/tokens.cpp
  ${CORE_SRC}
)
target_include_directories(llmhostd PRIVATE vendor src)
if (WIN32)
  target_link_libraries(llmhostd PRIVATE ws2_32 winhttp bcrypt advapi32 shell32 user32 gdi32 psapi)
  target_sources(llmhostd PRIVATE vendor/miniz.c)
  if (MSVC)
    set_source_files_properties(vendor/miniz.c PROPERTIES COMPILE_OPTIONS "/w")
  else()
    # mingw: рантайм-DLL (libstdc++-6/libgcc_s_seh) не тащим за собой —
    # вшиваем; на чистой Windows машине их нет.
    target_link_options(llmhostd PRIVATE -static-libgcc -static-libstdc++)
  endif()
else()
  find_package(Threads REQUIRED)
  target_link_libraries(llmhostd PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
  if (LLMHOST_STATIC)
    target_link_options(llmhostd PRIVATE -static)
  endif()
endif()

# manifest.json нужен демону рядом с бинарником (пины сборок движка).
# Только затравка: в живом билд-каталоге манифест принадлежит каналу
# «lhctl engine update» — пересборка его не трогает (cmake/manifest_seed.cmake).
add_custom_command(TARGET llmhostd POST_BUILD
  COMMAND ${CMAKE_COMMAND}
          -DSRC=${CMAKE_SOURCE_DIR}/manifest.json
          -DDST=$<TARGET_FILE_DIR:llmhostd>
          -P ${CMAKE_SOURCE_DIR}/cmake/manifest_seed.cmake)

if (WIN32)
  # GUI-приложение Windows (трей + чат): то же ядро + UI-слой. На Linux не собирается.
  add_executable(llmhost WIN32
    src/main.cpp
    ${CORE_SRC}
    src/catalogui.cpp
    src/chatui.cpp
    src/tray.cpp
    src/wizard.cpp
    src/app.rc
  )
  target_include_directories(llmhost PRIVATE vendor src)
  target_link_libraries(llmhost PRIVATE ws2_32 winhttp bcrypt advapi32 shell32 user32 gdi32 comctl32 comdlg32 dwmapi pdh)
  if (MSVC)
    target_compile_options(llmhost PRIVATE /W4 /permissive- /utf-8)
    target_link_options(llmhost PRIVATE "/MANIFEST:NO")
  endif()
endif()

# Консольный клиент control API: status / models / load / unload / quit.
# termui — панели/таблицы для интерактивного вывода (пайпы не задеты).
add_executable(lhctl src/lhctl.cpp src/util.cpp src/termui.cpp src/tui.cpp)
target_include_directories(lhctl PRIVATE vendor src)
if (WIN32)
  target_link_libraries(lhctl PRIVATE ws2_32 winhttp bcrypt advapi32 shell32 user32 gdi32)
  if (MSVC)
    target_compile_options(lhctl PRIVATE /W4 /permissive- /utf-8)
  endif()
else()
  find_package(Threads REQUIRED)
  target_link_libraries(lhctl PRIVATE Threads::Threads)
  if (LLMHOST_STATIC)
    target_link_options(lhctl PRIVATE -static)
  endif()
endif()

enable_testing()

# Тест парсера GGUF: файл создаётся сам из байтов — бинарных фикстур в репозитории нет.
add_executable(gguf_test tests/gguf_test.cpp src/gguf.cpp src/util.cpp)
target_include_directories(gguf_test PRIVATE vendor src)
if (WIN32)
  target_link_libraries(gguf_test PRIVATE bcrypt ws2_32 advapi32)
endif()
if (MSVC)
  target_compile_options(gguf_test PRIVATE /W4 /permissive- /utf-8)
endif()
add_test(NAME gguf COMMAND gguf_test)

add_executable(fit_test tests/fit_test.cpp src/fit.cpp)
target_include_directories(fit_test PRIVATE vendor src)
if (MSVC)
  target_compile_options(fit_test PRIVATE /W4 /permissive- /utf-8)
endif()
add_test(NAME fit COMMAND fit_test)

add_executable(config_test tests/config_test.cpp src/config.cpp src/util.cpp)
target_include_directories(config_test PRIVATE vendor src)
if (WIN32)
  target_link_libraries(config_test PRIVATE bcrypt ws2_32 advapi32)
endif()
if (MSVC)
  target_compile_options(config_test PRIVATE /W4 /permissive- /utf-8)
endif()
add_test(NAME config COMMAND config_test)

add_executable(util_test tests/util_test.cpp src/util.cpp)
target_include_directories(util_test PRIVATE vendor src)
if (WIN32)
  target_link_libraries(util_test PRIVATE bcrypt ws2_32 advapi32)
endif()
if (MSVC)
  target_compile_options(util_test PRIVATE /W4 /permissive- /utf-8)
endif()
add_test(NAME util COMMAND util_test)

# Эвристики каталога — header-only, сетевое не тестируем: только чистые функции.
add_executable(catalog_test tests/catalog_test.cpp)
target_include_directories(catalog_test PRIVATE src)
if (MSVC)
  target_compile_options(catalog_test PRIVATE /W4 /permissive- /utf-8)
endif()
add_test(NAME catalog COMMAND catalog_test)

# Выбор языка интерфейса: приоритет LH_LANG → хинт конфига → локаль; фолбэк tr().
add_executable(i18n_test tests/i18n_test.cpp)
target_include_directories(i18n_test PRIVATE src)
if (MSVC)
  target_compile_options(i18n_test PRIVATE /W4 /permissive- /utf-8)
endif()
add_test(NAME i18n COMMAND i18n_test)

# Лестница попыток загрузки — тоже header-only.
add_executable(plan_test tests/plan_test.cpp)
target_include_directories(plan_test PRIVATE src)
if (MSVC)
  target_compile_options(plan_test PRIVATE /W4 /permissive- /utf-8)
endif()
add_test(NAME plan COMMAND plan_test)

# Рендер Markdown для чата — тоже header-only.
add_executable(mdrender_test tests/mdrender_test.cpp)
target_include_directories(mdrender_test PRIVATE src)
if (MSVC)
  target_compile_options(mdrender_test PRIVATE /W4 /permissive- /utf-8)
endif()
add_test(NAME mdrender COMMAND mdrender_test)

# Агрегатор метрик: счётчики, снифер usage, рендер Prometheus-текста.
add_executable(metrics_test tests/metrics_test.cpp src/metrics.cpp)
target_include_directories(metrics_test PRIVATE vendor src)
if (MSVC)
  target_compile_options(metrics_test PRIVATE /W4 /permissive- /utf-8)
endif()
add_test(NAME metrics COMMAND metrics_test)

# Паркинг очереди пробуждения: глубина на ключ, TTL, поколения notify.
add_executable(park_test tests/park_test.cpp)
target_include_directories(park_test PRIVATE src)
if (MSVC)
  target_compile_options(park_test PRIVATE /W4 /permissive- /utf-8)
endif()
find_package(Threads REQUIRED)
target_link_libraries(park_test PRIVATE Threads::Threads)
add_test(NAME park COMMAND park_test)

# Достройка сборок манифеста из ассетов релиза: паттерны, скоуп, версии.
add_executable(engine_assets_test tests/engine_assets_test.cpp)
target_include_directories(engine_assets_test PRIVATE vendor src)
if (MSVC)
  target_compile_options(engine_assets_test PRIVATE /W4 /permissive- /utf-8)
endif()
add_test(NAME engine_assets COMMAND engine_assets_test)

# Хранилище профилей: ключ «id+форма», миграция, флаг попытки, штамп релиза.
add_executable(profiles_test tests/profiles_test.cpp src/profiles.cpp src/util.cpp)
target_include_directories(profiles_test PRIVATE vendor src)
if (MSVC)
  target_compile_options(profiles_test PRIVATE /W4)
endif()
add_test(NAME profiles COMMAND profiles_test)

# Рендер терминала: ширина UTF-8/ANSI, обрезка, сжатие таблиц, рамки.
add_executable(termui_test tests/termui_test.cpp src/termui.cpp)
target_include_directories(termui_test PRIVATE vendor src)
if (WIN32)
  target_compile_options(termui_test PRIVATE /W4)
endif()
add_test(NAME termui COMMAND termui_test)

add_executable(image_test tests/image_test.cpp src/image.cpp src/config.cpp src/util.cpp)
target_include_directories(image_test PRIVATE vendor src)
if (WIN32)
  target_compile_options(image_test PRIVATE /W4 /permissive- /utf-8)
  target_link_libraries(image_test PRIVATE bcrypt ws2_32 advapi32)
endif()
add_test(NAME image COMMAND image_test)

add_executable(modelreg_test tests/modelreg_test.cpp src/modelreg.cpp src/image.cpp src/config.cpp src/gguf.cpp src/util.cpp)
target_include_directories(modelreg_test PRIVATE vendor src)
if (WIN32)
  target_compile_options(modelreg_test PRIVATE /W4 /permissive- /utf-8)
  target_link_libraries(modelreg_test PRIVATE bcrypt ws2_32 advapi32)
endif()
add_test(NAME modelreg COMMAND modelreg_test)

# Эвристики stderr-кольца: OOM ствола против падения по черновику (кейс 07.10).
add_executable(proc_test tests/proc_test.cpp src/proc.cpp src/util.cpp)
target_include_directories(proc_test PRIVATE vendor src)
if (WIN32)
  target_link_libraries(proc_test PRIVATE bcrypt ws2_32 advapi32)
else()
  find_package(Threads REQUIRED)
  target_link_libraries(proc_test PRIVATE Threads::Threads)
endif()
if (MSVC)
  target_compile_options(proc_test PRIVATE /W4 /permissive- /utf-8)
endif()
add_test(NAME proc COMMAND proc_test)

# Тесты честные в любой конфигурации: Release несёт NDEBUG, assert() вырезается
# и прогоны становятся пустыми. Снимаем NDEBUG у всех тестовых таргетов.
foreach(t gguf_test fit_test config_test util_test catalog_test i18n_test plan_test mdrender_test metrics_test park_test profiles_test termui_test modelreg_test image_test)
  if (MSVC)
    target_compile_options(${t} PRIVATE /UNDEBUG)
  else()
    target_compile_options(${t} PRIVATE -UNDEBUG)
  endif()
  # Тестовые бинарники не лежат рядом с llmhostd: в релизной папке им не место.
  set_target_properties(${t} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/tests)
endforeach()
