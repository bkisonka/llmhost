// Реестр моделей: рекурсивный скан папки, у каждого *.gguf — заголовок.
// Разбор дешёвый (только заголовок), кэш по (размер, mtime).
#pragma once
#include <string>
#include <vector>
#include "gguf.h"

namespace lh::modelreg {

struct Model {
  std::string path;
  std::string name;      // general.name или имя файла
  std::string arch;
  std::string quant;
  uint64_t size = 0;
  uint64_t mtime = 0;    // время изменения: замена файла тем же размером
  gguf::Meta meta;
};

std::vector<Model> scan(const std::string& dir);

// Файл — MTP-модуль (родной ускоритель спекулятивного декодирования, кладётся
// в репозитории модели: подпапка MTP или имя mtp-*). Как основную модель его
// не грузят — он прицепляется черновиком, поэтому секцию в пресете ему не заводят.
bool is_mtp_module(const std::string& path);

// Где MTP-модуль лежит относительно модели.
// 1 — в поддереве папки файла (в том числе в её MTP/).
// 2 — в папке MTP/ на уровень выше: квант скачан в подпапку репозитория,
//     голова — в MTP/ рядом с этой подпапкой.
// 0 — планировщик этот модуль к модели не цепляет.
int mtp_place(const std::string& model_path, const std::string& mtp_path);

// Модуль принадлежит этому стволу: имя после «mtp-» без квантового хвоста
// («Q4_K_M», «Q8_0») — префикс имени модели по токенам. Близость папки —
// не родство: в плоской папке чужие головы с общим словарём проходили
// мета-фильтры, роняли ребёнка и съедали лестницу загрузки (27.10: три
// чужих draft на qwen3.5-4b за три загрузки). Сравнение без учёта
// регистра — регистр имён в папках гуляет (12B/12b).
bool mtp_matches_trunk(const std::string& model_path, const std::string& mtp_path);

// Сигнатурное ядро имени проекции: стем без токена «mmproj» и без квантово-
// форматных хвостов (f16/bf16/q4_k_xl…), строчными, слитно.
// «qwen36-mmproj-BF16.gguf» → «qwen36»; «mmproj-F16.gguf» → «» — легаси-имя
// pull-раскладки, сигнатуры в нём нет (родство даёт каталог репозитория).
std::string mmproj_core_of(const std::string& mmproj_path);

// Родство проекции и ствола по имени (аналог mtp_matches_trunk): непустое
// ядро проекции — подпоследовательность нормализованного имени ствола.
// «gemma12» вяжется к «gemma-4-12B-…», но не к «gemma-4-26B-…»; «gemma4» —
// к обоим, их разбирает find_mmproj по длине ядра. Ядро короче 4 символов
// родства не доказывает (ловит случайные подпоследовательности).
bool mmproj_matches_trunk(const std::string& trunk_path,
                          const std::string& mmproj_path);

// Файл — vision-проекция: «mmproj» в любом месте имени (регистр не важен) —
// и апстримное «mmproj-F16.gguf», и ручное «qwen36-mmproj-BF16.gguf».
// Самостоятельной моделью не является — прикладывается к основной через
// --mmproj, чтобы она видела картинки. Из списков моделей и пресета
// исключается.
bool is_mmproj_module(const std::string& path);

// Спутник модели — файл из репозитория модели, который имеет смысл только при
// ней. kind: «shard» — часть шард-группы той же модели, «mmproj» —
// vision-проекция, «mtp» — родной MTP-модуль.
struct Companion {
  std::string path;
  std::string kind;
  uint64_t size = 0;
};

// Разбор шардинга llama.cpp: «base-00001-of-00003.gguf» → (1, 3); обычный
// файл → (0, 1).
void shard_info(const std::string& path, int& index, int& count);

// Базовое имя шард-группы (в согласии с shard_info): «base-00001-of-00003» →
// «base-»; у файла вне группы — стем без «.gguf».
std::string shard_base(const std::string& path);

// Часть 2..N шард-группы. Самостоятельной моделью не является: движок
// открывает первую часть и находит остальные по имени.
bool is_shard_continuation(const std::string& path);

// Имя для реестра. У шард-группы — база без «-00001-of-00004» (одна и та же
// у всех частей). У обычного файла — стем без «.gguf».
std::string registry_stem(const std::string& path);

// Байты весов. У шард-группы — сумма частей в той же папке (сколько уже
// лежит на диске). У обычного файла — его размер.
uint64_t shard_group_bytes(const std::string& path);

// Мета модели с учётом шард-группы. Заголовок первой части несёт метаданные
// модели, но тензорный список у писателей вроде unsloth живёт в частях 2..N
// (в первой tensor_count = 0), поэтому байты экспертов и CPU-тензоры — это
// сумма списков всех частей. Для файла вне группы — обычный parse_header.
bool model_meta(const std::string& path, gguf::Meta& out);

// Что уйдёт с диска вместе с моделью, если удалить её сейчас: части её
// шард-группы и модули (mmproj, MTP), лежащие в её репозитории. Модуль,
// нужный ещё живой основной модели, не спутник: mmproj ищется рядом с
// моделью (как find_mmproj), MTP — там, где его берёт find_draft: поддерево папки файла или MTP/ на уровень выше.
// list — полный скан реестра, scan().
std::vector<Companion> companions_of(const std::string& model_path,
                                     const std::vector<Model>& list);

} // namespace lh::modelreg
