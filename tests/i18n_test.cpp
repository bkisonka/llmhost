// Выбор языка интерфейса: приоритет LH_LANG → хинт конфига → локаль машины;
// tr() при промахе таблицы возвращает ключ (русский) — дыра покрытия видна,
// а не молча ломает вывод. Среда процесса — глобальная: каждый сценарий
// сначала чистит LH_LANG/LANG/LC_*.
#include "i18n.h"

#include <cassert>
#include <cstdlib>
#include <string>

using namespace lh::i18n;

static void clean_env() {
  ::unsetenv("LH_LANG");
  ::unsetenv("LANG");
  ::unsetenv("LC_ALL");
  ::unsetenv("LC_MESSAGES");
}

int main() {
  // Чистая среда без локали — английский (правило: не-ру машина).
  clean_env();
  set_lang_hint(nullptr);
  assert(lang() == kEn);

  // Русская локаль (кодировка отрезается) — русский.
  ::setenv("LANG", "ru_RU.UTF-8", 1);
  set_lang_hint(nullptr);  // сброс решённого — перерешать
  assert(lang() == kRu);

  // Приоритет LC_ALL над LANG; не-ru префикс — английский.
  ::setenv("LC_ALL", "en_US.UTF-8", 1);
  set_lang_hint(nullptr);
  assert(lang() == kEn);
  ::setenv("LC_ALL", "ru_RU", 1);
  set_lang_hint(nullptr);
  assert(lang() == kRu);
  ::unsetenv("LC_ALL");

  // LH_LANG форсится против локали; регистр и суффикс не важны.
  ::setenv("LH_LANG", "EN", 1);
  assert(lang() == kEn);
  ::setenv("LH_LANG", "ru_RU", 1);
  assert(lang() == kRu);
  ::setenv("LH_LANG", "фигня", 1);
  assert(lang() == kRu);  // мусор игнорируется → локаль ru
  ::unsetenv("LANG");
  set_lang_hint(nullptr);
  assert(lang() == kEn);  // локали нет, мусор в LH_LANG — английский
  ::unsetenv("LH_LANG");

  // Хинт конфига бьёт локаль, но не LH_LANG.
  ::setenv("LANG", "ru_RU.UTF-8", 1);
  set_lang_hint("en");
  assert(lang() == kEn);
  ::setenv("LH_LANG", "ru", 1);
  assert(lang() == kRu);
  ::unsetenv("LH_LANG");
  set_lang_hint("auto");  // auto → снова детект локали
  assert(lang() == kRu);

  // tr(): промах по пустой таблице — ключ как есть, в обоих языках.
  set_lang_hint("en");
  assert(std::string(tr("нет такого ключа")) == "нет такого ключа");
  set_lang_hint("ru");
  assert(std::string(tr("нет такого ключа")) == "нет такого ключа");

  return 0;
}
