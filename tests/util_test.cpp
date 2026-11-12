// Тесты мелкой обвязки: wildcards, нормализация имён HF-репозиториев,
// квотинг аргументов, санитизация путей из сети, URL-кодирование,
// маленькие файлы (потолок чтения, приватная запись).
#include "util.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#ifndef _WIN32
#include <sys/stat.h>
#endif

using lh::normalize_hf_repo;
using lh::wildcard_match;

static int failed = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    fprintf(stderr, "FAIL: %s\n", what);
    ++failed;
  }
}

int main() {
  // ---- wildcards ---------------------------------------------------------
  check(wildcard_match("*", ""), "MATCH '*' ''");
  check(wildcard_match("*", "anything"), "MATCH '*' 'anything'");
  check(wildcard_match("*.gguf", "model.gguf"), "MATCH '*.gguf' 'model.gguf'");
  check(!wildcard_match("*.gguf", "model.gguf.txt"), "NO MATCH '*.gguf' 'model.gguf.txt'");
  check(wildcard_match("*Q4_K_M*", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
        "MATCH '*Q4_K_M*' 'Llama-...-Q4_K_M.gguf'");
  check(wildcard_match("*q4_k_m*", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
        "MATCH без регистра");
  check(wildcard_match("*-00001-of-00002.gguf", "model-00001-of-00002.gguf"),
        "MATCH шард №1");
  check(!wildcard_match("*-00001-of-00002.gguf", "model-00002-of-00002.gguf"),
        "NO MATCH шард №2");
  check(wildcard_match("a?c", "abc"), "MATCH 'a?c' 'abc'");
  check(!wildcard_match("a?c", "abbc"), "NO MATCH 'a?c' 'abbc'");
  check(wildcard_match("", ""), "MATCH '' ''");
  check(!wildcard_match("", "x"), "NO MATCH '' 'x'");

  // ---- normalize_hf_repo -------------------------------------------------
  const char* good[] = {
      "org/repo",
      "huggingface.co/org/repo",
      "https://huggingface.co/org/repo/resolve/main/model.gguf",
      "org/repo/",
      "/org/repo",
  };
  for (const char* s : good) {
    std::string r = s, err;
    check(normalize_hf_repo(r, err) && r == "org/repo", s);
  }
  const char* bad[] = {"", "justname", "a/b/c", "https://huggingface.co/org/repo/tree/main",
                       "../evil", "org/re po", "org/../repo", "org/.."};
  for (const char* s : bad) {
    std::string r = s, err;
    check(!normalize_hf_repo(r, err) && !err.empty(), s);
  }

  // ---- safe_rel_path (пути из HF API → папка моделей) ---------------------
  check(lh::safe_rel_path("model.gguf"), "SAFE простое имя");
  check(lh::safe_rel_path("sub/dir/model-8B.gguf"), "SAFE подпапка");
  check(lh::safe_rel_path("Qwen3 8B.gguf"), "SAFE пробел в имени");
  check(lh::safe_rel_path("\xD0\xA4.gguf"), "SAFE не-ASCII имя");
  check(!lh::safe_rel_path("../x.gguf"), "UNSAFE ..");
  check(!lh::safe_rel_path("a/../b.gguf"), "UNSAFE .. в середине");
  check(!lh::safe_rel_path("C:/x.gguf"), "UNSAFE диск");
  check(!lh::safe_rel_path("a/b:c.gguf"), "UNSAFE двоеточие");
  check(!lh::safe_rel_path("a\".gguf"), "UNSAFE кавычка");
  check(!lh::safe_rel_path("/abs.gguf"), "UNSAFE абсолютный");
  check(!lh::safe_rel_path("a//b.gguf"), "UNSAFE пустой компонент");
  check(!lh::safe_rel_path("a/"), "UNSAFE хвостовой слэш");

  // ---- safe_local_filename (имена с диска → пресет/журналы) ----------------
  check(lh::safe_local_filename("/models/Qwen3 8B.gguf"), "SAFEFILE обычный путь");
  check(lh::safe_local_filename("/models/Ф-model.gguf"), "SAFEFILE кириллица");
  check(lh::safe_local_filename("/x/a[b](c);d=.gguf"), "SAFEFILE спецсимволы печати");
  check(!lh::safe_local_filename("/x\n[evil]\ny.gguf"), "UNSAFEFILE перевод строки");
  check(!lh::safe_local_filename("/x\ry.gguf"), "UNSAFEFILE CR");
  check(!lh::safe_local_filename(std::string("/x\x01y.gguf")), "UNSAFEFILE control-байт");
  check(!lh::safe_local_filename(""), "UNSAFEFILE пустой путь");

  // ---- percent_encode_path ------------------------------------------------
  check(lh::percent_encode_path("model.gguf") == "model.gguf", "ENC простой");
  check(lh::percent_encode_path("a b/c d.gguf") == "a%20b/c%20d.gguf", "ENC пробелы");
  check(lh::percent_encode_path("\xD0\xA4.gguf") == "%D0%A4.gguf", "ENC не-ASCII");

  // ---- expand_env ---------------------------------------------------------
  check(lh::expand_env("$LLMHOST_NO_SUCH_VAR" "x") == "$LLMHOST_NO_SUCH_VAR" "x",
        "ENV несуществующая остаётся как есть");
  {
    // Существующая переменная разворачивается ($VAR и ${VAR}).
    putenv((char*)"LLMHOST_TEST_VAR=/tmp/zz");
    check(lh::expand_env("$LLMHOST_TEST_VAR/m.gguf") == "/tmp/zz/m.gguf", "ENV $VAR");
    check(lh::expand_env("${LLMHOST_TEST_VAR}/m.gguf") == "/tmp/zz/m.gguf", "ENV ${VAR}");
  }

  // ---- base64 (картинки в data-URI чата) -----------------------------------
  check(lh::base64_encode("") == "", "B64 пустое");
  check(lh::base64_encode("hello") == "aGVsbG8=", "B64 hello");
  check(lh::base64_encode("abc") == "YWJj", "B64 без паддинга");
  check(lh::base64_encode("he") == "aGU=", "B64 один паддинг");
  check(lh::base64_encode(std::string("\x00\x01\xff", 3)) == "AAH/", "B64 бинарные");

  // ---- replace_all (пин релиза движка: тег в имени ассета) -----------------
  check(lh::replace_all("llama-b11344-bin-ubuntu-cuda-12.8-x64.tar.gz", "b11344",
                        "b11379") == "llama-b11379-bin-ubuntu-cuda-12.8-x64.tar.gz",
        "REPLACE тег в имени ассета");
  check(lh::replace_all("b11344-b11344", "b11344", "b11379") == "b11379-b11379",
        "REPLACE все вхождения");
  check(lh::replace_all("abc", "x", "y") == "abc", "REPLACE нет вхождений");
  check(lh::replace_all("bbb", "b", "") == "", "REPLACE удаление всех");
  check(lh::replace_all("abc", "", "X") == "abc", "REPLACE пустая подстрока — без изменений");
  check(lh::replace_all("", "a", "b") == "", "REPLACE пустая строка");
  check(lh::replace_all("aaa", "a", "aa") == "aaaaaa", "REPLACE рост без зацикливания");

  // ---- release_from_version_output (факт релиза из --version llama-server) --
  check(lh::release_from_version_output(
            "version: 0.6.0-dev (build 11382, commit 11fe02151)\n"
            "built with GNU 13.3.0 for Linux x86_64") == "b11382",
        "RELVER стандартный вывод --version");
  check(lh::release_from_version_output("ggml: build 123 extra") == "b123",
        "RELVER без скобок");
  check(lh::release_from_version_output("build abc") == "", "RELVER не число");
  check(lh::release_from_version_output("build ") == "", "RELVER пустое число");
  check(lh::release_from_version_output("ничего похожего") == "", "RELVER нет build");
  check(lh::release_from_version_output("") == "", "RELVER пустая строка");

  // ---- parse_draft_acceptance (сторож спекуляции; формат плавает по пинам) --
  {
    double r;
    long long a, g;
    // b11429: слова accepted/generated в скобках
    check(lh::parse_draft_acceptance(
              "slot print_timing: id  3 | task 243 | draft acceptance = 0.66051 "
              "(  286 accepted /   433 generated), mean len =  2.97",
              r, a, g) &&
              r > 0.66 && r < 0.67 && a == 286 && g == 433,
          "ACCEPT вариант b11429 (слова в скобках)");
    // числовой вариант без слов
    check(lh::parse_draft_acceptance(
              "slot print_timing: id 0 | task 9 | draft acceptance = 0.979 ( 1234/ 1261)",
              r, a, g) &&
              r > 0.978 && a == 1234 && g == 1261,
          "ACCEPT числовой вариант");
    // «accept rate»
    check(lh::parse_draft_acceptance(
              "slot print_timing: draft accept rate = 0.512 ( 100 accepted / 200 generated)",
              r, a, g) &&
              r > 0.511 && a == 100 && g == 200,
          "ACCEPT accept rate");
    // гипотетический будущий формат
    check(lh::parse_draft_acceptance(
              "spec_draft: accept=0.979 n=1234/1261", r, a, g) &&
              r > 0.978 && a == 1234 && g == 1261,
          "ACCEPT spec_draft");
    // только rate, без счётчиков — распознать, счётчики нулевые
    check(lh::parse_draft_acceptance(
              "draft acceptance = 0.750", r, a, g) &&
              r > 0.749 && a == 0 && g == 0,
          "ACCEPT только rate");
    // не наша строка
    check(!lh::parse_draft_acceptance(
              "slot print_timing: id 1 | task 2 | n_tokens = 256", r, a, g),
          "ACCEPT чужая строка отвергнута");
    check(!lh::parse_draft_acceptance("", r, a, g), "ACCEPT пустая строка");
  }

  // ---- маленькие файлы: потолок размера и приватная запись --------------------
  {
    namespace fs = std::filesystem;
    std::error_code ec;
    std::string big = (fs::temp_directory_path() / "lh_util_big.bin").string();
    {
      std::ofstream f(big, std::ios::binary);
      const std::string chunk(65536, 'x');
      for (int i = 0; i < 130; ++i) f << chunk; // 8.3 МиБ — за потолком 8 МиБ
    }
    check(lh::read_small_file(big).empty(), "read_small_file: >8 МиБ — пусто, без OOM");
    fs::remove(big, ec);

    std::string small = (fs::temp_directory_path() / "lh_util_priv.txt").string();
    fs::remove(small, ec);
    check(lh::write_private_file(small, "secret"), "write_private_file: записал");
    check(lh::read_small_file(small) == "secret", "read/write_private roundtrip");
#ifndef _WIN32
    struct stat st;
    // umask может лишь сузить права (0600 → 0400) — проверяем, что ни одной
    // групповой/сторонней биты и исполняемого нет.
    check(::stat(small.c_str(), &st) == 0 && (st.st_mode & 0777u) <= 0600 &&
              (st.st_mode & 0077u) == 0,
          "write_private_file: не уже 0600 сразу при создании");
#endif
    fs::remove(small, ec);
  }

  if (failed) {
    fprintf(stderr, "%d проверок не прошло\n", failed);
    return 1;
  }
  printf("ok\n");
  return 0;
}
