#!/usr/bin/env bash
# Сборка Linux-релиза: статический билд + тесты -> dist/llmhost-<ver>-linux-x64.tar.gz
# (+ .sha256); ассеты для GitHub Releases копируются в dist/gh-release-<ver>/
# (аналог scripts/package_release.bat, который пакует Windows-zip).
#
#   scripts/package_release.sh [ВЕРСИЯ]        # по умолчанию — текущая (LH_VERSION)
#
# На GitHub Releases загрузить из dist/gh-release-<ver>/ четыре ассета:
# install.sh, llmhost-<ver>-linux-x64.tar.gz, llmhost-<ver>-linux-x64.tar.gz.sha256
# и манифест; заметки — в тело релиза (полный текст в RELEASE_NOTES.txt
# внутри тарболла).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VER="${1:-0.6.0-beta14}"
NAME="llmhost-$VER-linux-x64"
DIST="$ROOT/dist"
BUILD="$ROOT/build-linux-pkg"

# 1. Статический билд в отдельном каталоге (рабочий build-linux не трогаем).
echo "== Сборка (Release, LLMHOST_STATIC=ON): $BUILD"
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DLLMHOST_STATIC=ON
cmake --build "$BUILD" -j"$(nproc)"

# 2. Юнит-тесты (тестовые таргеты не линкуются статически — им статика не нужна).
echo "== Тесты"
ctest --test-dir "$BUILD" --output-on-failure

# 3. Стейджинг: только нужное пользователю (как в Windows-zip).
#    Манифест — с публичной note (scripts/showcase_texts.py, общий с
#    экспортёром витрины источник переписываний): репозиторная note несёт
#    историю пинов и нарратив рабочего окружения — в релиз она не едет.
STAGE="$DIST/$NAME"
rm -rf "$STAGE"
mkdir -p "$STAGE"
PUBTMP="$(mktemp -d /tmp/showcase-texts.XXXXXX)"
trap 'rm -rf "$PUBTMP"' EXIT
python3 "$ROOT/scripts/showcase_texts.py" "$PUBTMP"
install -m755 "$BUILD/llmhostd" "$STAGE/llmhostd"
install -m755 "$BUILD/lhctl" "$STAGE/lhctl"
install -m644 "$PUBTMP/manifest.json" "$STAGE/manifest.json"
install -m644 "$ROOT/README.md" "$STAGE/README.md"
# Языковые варианты README: ссылки-переключатель в README.md работают
# и в тарболле (docs/ рядом с корнем архива).
mkdir -p "$STAGE/docs"
for f in README.en.md README.zh.md README.pt-br.md; do
  install -m644 "$ROOT/docs/$f" "$STAGE/docs/$f"
done
install -m644 "$ROOT/RELEASE_NOTES.txt" "$STAGE/RELEASE_NOTES.txt"
install -m644 "$ROOT/LICENSE" "$STAGE/LICENSE"
install -m644 "$ROOT/scripts/install.sh" "$STAGE/install.sh"

# Проверка, что бинари действительно статические (ldd на статику отвечает
# «not a dynamic executable» и ненулевой код).
for bin in llmhostd lhctl; do
  if ldd "$STAGE/$bin" >/dev/null 2>&1; then
    echo "ОШИБКА: $STAGE/$bin не статический" >&2
    exit 1
  fi
done

# 4. Тарболл + sha256 (внутри тарболла — папка с именем релиза).
mkdir -p "$DIST"
TARBALL="$DIST/$NAME.tar.gz"
tar -czf "$TARBALL" -C "$DIST" "$NAME"
(cd "$DIST" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256")

# 5. Ассеты для GitHub Releases (4: тарболл, .sha256, install.sh, манифест).
GH="$DIST/gh-release-$VER"
mkdir -p "$GH"
cp -f "$TARBALL" "$DIST/$NAME.tar.gz.sha256" "$ROOT/scripts/install.sh" "$GH/"
install -m644 "$PUBTMP/manifest.json" "$GH/manifest.json"

echo
echo "Готово: $TARBALL"
cat "$DIST/$NAME.tar.gz.sha256"
echo "Ассеты для релиза: $GH (install.sh, тарболл, sha256, манифест)"
