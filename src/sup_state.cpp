#include "sup_internal.h"

namespace lh::sup::detail {

std::mutex g_mtx;

// Карта реестра harness: id → путь модели. Пишут harness_registry_maps
// (init и вотчер реестра), читают статус, чат-подсказка и будилка.
std::map<std::string, std::string> g_id_map;
std::set<std::string> g_emb;              // эмбеддерные модели: чат их не трогает
std::map<std::string, std::string> g_sys; // id → system-промпт (инъекция на прокси)
// Применение реестра идёт по очереди (вотчер, enable_mtp, ручное применение):
// параллельная перестройка карты посреди чужой загрузки убивает её ребёнка.
std::mutex g_apply_mtx;
// Отпечаток «уже применённого» реестра: apply_registry_change зовут не только
// вотчер (enable_mtp применяет сразу) — без общего гроссбука вотчер повторил
// бы перестройку через 2 с после ручного применения.
std::mutex g_fp_mtx;
std::string g_applied_fp;

std::mutex g_busy_mtx;
std::set<std::string> g_busy_set; // модели, где идёт генерация

} // namespace lh::sup::detail
