#!/bin/sh
# Установщик llmhost (Linux): скачивает релизный тарболл с GitHub Releases,
# проверяет SHA256, ставит llmhostd/lhctl/manifest.json в ~/.local/lib/llmhost,
# симлинк lhctl в ~/.local/bin с записью в PATH, systemd user unit и ставит
# движок llama.cpp. Каталог моделей спрашивает (умолчание показано) и пишет
# его в [general] llmhost.conf — единый конфиг приложения.
#
#   curl -fsSL https://github.com/bkisonka/llmhost/releases/latest/download/install.sh | sh
#
# Флаги:
#   --version ВЕРСИЯ     конкретный релиз (тег) вместо последнего
#   --engine auto|cuda12|vulkan|none   какой движок ставить (по умолчанию auto:
#                        cuda12 при живом nvidia-smi, иначе vulkan)
#   --no-engine          не ставить движок (синоним --engine none)
#   --no-start           не трогать systemd (запуск вручную: ~/.local/lib/llmhost/llmhostd)
#   --models-dir ПУТЬ    каталог моделей без вопроса (иначе спросит, если есть
#                        терминал; в безлюдной среде — умолчание)
#   --help               эта справка
#
# Обновление = повторный запуск той же команды: демон останавливается, бинари
# заменяются, служба поднимается. Данные и модели не трогаются — они живут в
# ~/.local/share/llmhost (или $LH_DATA_DIR).
#
# Переменные окружения (нестандартные случаи, зеркала, испытания):
#   LLMHOST_REPO=owner/name    репозиторий публикации (по умолчанию зашит ниже)
#   LLMHOST_BASE_URL=URL       базовый URL вместо GitHub Releases; тарболл ищется
#                              как <URL>/llmhost-<версия>-linux-x64.tar.gz
#   LLMHOST_DIR=ПУТЬ           куда ставить бинари (по умолчанию ~/.local/lib/llmhost)
#   LLMHOST_BIN_DIR=ПУТЬ       симлинк lhctl + запись в PATH (по умолчанию ~/.local/bin)
#   LLMHOST_MODELS_DIR=ПУТЬ    каталог моделей — то же, что --models-dir
#
# Скрипт POSIX sh. stdin НЕ читает (запускается через пайп): флаги и переменные
# окружения. Единственный вопрос — каталог моделей: при живом терминале читаем
# /dev/tty (в интерактивном запуске — stdin), иначе тихо берём умолчание.
set -eu

REPO="${LLMHOST_REPO:-bkisonka/llmhost}"
VER=""          # пусто = последний релиз (через GitHub API)
ENGINE="auto"   # auto | cuda12 | vulkan | none
NO_START=0
MODELS_DIR="${LLMHOST_MODELS_DIR:-}"

say() { printf '%s\n' "$*"; }
warn() { printf 'ПРЕДУПРЕЖДЕНИЕ: %s\n' "$*" >&2; }
die() { printf 'ОШИБКА: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
install.sh — установщик llmhost (Linux)

  curl -fsSL https://github.com/bkisonka/llmhost/releases/latest/download/install.sh | sh

Флаги:
  --version ВЕРСИЯ     конкретный релиз (тег) вместо последнего
  --engine auto|cuda12|vulkan|none   движок (по умолчанию auto: cuda12 при
                       живом nvidia-smi, иначе vulkan)
  --no-engine          не ставить движок
  --no-start           не трогать systemd (ручной запуск: ~/.local/bin/llmhostd)
  --models-dir ПУТЬ    каталог моделей без вопроса (умолчание:
                       ${LH_DATA_DIR:-~/.local/share/llmhost}/models)

Обновление = повторный запуск той же команды. Данные и модели не трогаются.

Переменные: LLMHOST_REPO=owner/name, LLMHOST_BASE_URL=URL (зеркало),
LLMHOST_DIR=ПУТЬ (по умолчанию ~/.local/lib/llmhost),
LLMHOST_BIN_DIR=ПУТЬ (симлинк lhctl и запись в PATH, по умолчанию ~/.local/bin),
LLMHOST_MODELS_DIR=ПУТЬ (каталог моделей; путь пишется в [general]
llmhost.conf и меняется правкой этого файла).
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --version) [ $# -ge 2 ] || die "--version: укажите версию"; VER="$2"; shift 2 ;;
    --version=*) VER="${1#*=}"; shift ;;
    --engine) [ $# -ge 2 ] || die "--engine: укажите auto|cuda12|vulkan|none"; ENGINE="$2"; shift 2 ;;
    --engine=*) ENGINE="${1#*=}"; shift ;;
    --no-engine) ENGINE="none"; shift ;;
    --no-start) NO_START=1; shift ;;
    --models-dir) [ $# -ge 2 ] || die "--models-dir: укажите путь"; MODELS_DIR="$2"; shift 2 ;;
    --models-dir=*) MODELS_DIR="${1#*=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "неизвестный аргумент: $1 (справка: install.sh --help)" ;;
  esac
done

case "$ENGINE" in auto|cuda12|vulkan|none) ;; *) die "--engine=$ENGINE: ожидалось auto|cuda12|vulkan|none" ;; esac

command -v curl >/dev/null 2>&1 || die "нужен curl (установите пакет curl)"
command -v sha256sum >/dev/null 2>&1 || die "нужен sha256sum (coreutils)"

ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64) ;;
  aarch64|arm64) die "сборки для arm64 пока нет — пребилды llama.cpp тоже только x64" ;;
  *) die "неизвестная архитектура: $ARCH" ;;
esac

if [ -z "${LLMHOST_BASE_URL:-}" ]; then
  case "$REPO" in */*) ;; *) die "LLMHOST_REPO=$REPO: ожидалось owner/name" ;; esac
fi

# --- Каталог моделей (вопрос + запись в llmhost.conf) -----------------------
# Тот же порядок выбора каталога данных, что у демона (util.cpp app_dir):
# $LH_DATA_DIR → $XDG_DATA_HOME/llmhost → ~/.local/share/llmhost.
if [ -n "${LH_DATA_DIR:-}" ]; then
  DATA_DIR="$LH_DATA_DIR"
elif [ -n "${XDG_DATA_HOME:-}" ] && [ "${XDG_DATA_HOME#/}" != "$XDG_DATA_HOME" ]; then
  DATA_DIR="$XDG_DATA_HOME/llmhost"
else
  DATA_DIR="$HOME/.local/share/llmhost"
fi
CONF="$DATA_DIR/llmhost.conf"
DEFAULT_MODELS_DIR="$DATA_DIR/models"

if [ -z "$MODELS_DIR" ]; then
  printf 'Каталог для скачанных с HuggingFace моделей [%s]: ' \
    "$DEFAULT_MODELS_DIR" >&2
  answer=""
  if [ -t 0 ]; then
    # Интерактивный запуск (sh install.sh): вопрос читается из stdin.
    IFS= read -r answer || answer=""
  elif [ -t 2 ]; then
    # Пайп curl|sh при живом терминале: stdin занят скриптом — читаем /dev/tty.
    { IFS= read -r answer < /dev/tty || answer=""; }
  fi
  MODELS_DIR="${answer:-$DEFAULT_MODELS_DIR}"
  if [ -t 0 ] || [ -t 2 ]; then
    say "Каталог моделей: $MODELS_DIR"
  fi
fi
# ~ и относительные пути — в абсолютные: конфиг парсит демон из любого cwd.
case "$MODELS_DIR" in
  "~"*) MODELS_DIR="$HOME${MODELS_DIR#\~}" ;;
  /*) ;;
  *) MODELS_DIR="$PWD/$MODELS_DIR" ;;
esac
# Управляющие символы в путях не пропускаем: значение едет в INI-конфиг
# (строка = значение) и в systemd-юнит.
case "$MODELS_DIR" in *[[:cntrl:]]*) die "путь моделей содержит управляющие символы" ;; esac
case "$DATA_DIR" in
  *[[:cntrl:]]*|*[[:space:]]*|*'"'*|*"'"*|*\\*)
    die "каталог данных ($DATA_DIR) не годится: управляющие символы, пробелы или кавычки недопустимы для строки Environment в systemd-юните" ;;
esac

# Вписать models-dir в [general] llmhost.conf: заменить существующую строку,
# дописать после заголовка секции или создать блок в начале файла. Существующие
# комментарии и per-model секции не трогаются; прочие ключи демон допишет сам.
# Значение через ENVIRON, не через -v: awk в -v обрабатывает бэкслеши,
# путь с «\» исказился бы (или стал бы инъекцией строки).
write_models_dir() { # $1 = путь conf, $2 = каталог
  if [ ! -f "$1" ] || [ ! -s "$1" ]; then
    {
      printf '# Настройки llmhost. Правится вручную или через lhctl set;\n'
      printf '# [general] применяется демоном на лету (порты — после рестарта).\n'
      printf '[general]\nmodels-dir = %s\n' "$2"
    } > "$1"
    return 0
  fi
  if grep -qE '^[[:space:]]*models-dir[[:space:]]*=' "$1"; then
    VAL="$2" awk '
      BEGIN { v = ENVIRON["VAL"] }
      !done && /^[[:space:]]*models-dir[[:space:]]*=/ { print "models-dir = " v; done = 1; next }
      { print }
    ' "$1" > "$1.tmp" && mv "$1.tmp" "$1"
  elif VAL="$2" awk '
      BEGIN { v = ENVIRON["VAL"] }
      !ins && /^[[:space:]]*\[general\][[:space:]]*$/ { print; print "models-dir = " v; ins = 1; next }
      { print }
      END { exit ins ? 0 : 42 }
    ' "$1" > "$1.tmp" && mv "$1.tmp" "$1"; then
    :
  else
    # Секции [general] нет — блок в начало (перед комментариями и секциями).
    { printf '[general]\nmodels-dir = %s\n\n' "$2"; cat "$1"; } > "$1.tmp" \
      && mv "$1.tmp" "$1"
  fi
}


TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
trap 'exit 1' INT TERM

# --- URL тарболла -----------------------------------------------------------
# Через GitHub API — чтобы не зависеть от конвенции имён тегов: берём
# browser_download_url ассета по суффиксу. Без API (BASE_URL) — имя из версии.
TARBALL=""
URL_TGZ=""
if [ -n "${LLMHOST_BASE_URL:-}" ]; then
  # Зеркало/испытания: тарболл с версией в имени; без --version — алиас без версии.
  if [ -n "$VER" ]; then
    TARBALL="llmhost-$VER-linux-x64.tar.gz"
  else
    TARBALL="llmhost-linux-x64.tar.gz"
  fi
  URL_TGZ="$LLMHOST_BASE_URL/$TARBALL"
else
  API="https://api.github.com/repos/$REPO/releases/latest"
  if [ -n "$VER" ]; then
    API="https://api.github.com/repos/$REPO/releases/tags/$VER"
  fi
  say "Ищу релиз: $API"
  URL_TGZ="$(curl -fsSL --connect-timeout 15 "$API" \
    | sed -n 's/.*"browser_download_url"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
    | grep '/[^/]*linux-x64\.tar\.gz$' | head -n 1)" || true
  if [ -z "$URL_TGZ" ]; then
    die "релиз не найден (нет ассета *linux-x64.tar.gz). Проверьте LLMHOST_REPO=$REPO${VER:+ и тег $VER}"
  fi
  TARBALL="$(basename "$URL_TGZ")"
fi

# --- Скачивание и проверка хэша (fail-closed, как при скачивании моделей) ---
say "Скачиваю $URL_TGZ"
curl -fsSL --retry 3 --retry-delay 2 --connect-timeout 15 -o "$TMP/$TARBALL" "$URL_TGZ"
say "Скачиваю контрольную сумму"
if curl -fsSL --retry 3 --retry-delay 2 --connect-timeout 15 -o "$TMP/$TARBALL.sha256" "$URL_TGZ.sha256"; then
  EXPECTED="$(cut -d' ' -f1 "$TMP/$TARBALL.sha256" | tr -d '\r')"
  ACTUAL="$(sha256sum "$TMP/$TARBALL" | cut -d' ' -f1)"
  if printf '%s' "$EXPECTED" | grep -qE '^[0-9a-fA-F]{64}$'; then
    if [ "$ACTUAL" != "$EXPECTED" ]; then
      die "SHA256 не сошёлся: ожидалось $EXPECTED, получено $ACTUAL — файл НЕ устанавливается"
    fi
    say "SHA256 ok: $ACTUAL"
  else
    warn "файл .sha256 не похож на хэш — продолжаю без проверки"
  fi
else
  warn "sha256-файл недоступен — продолжаю без проверки"
fi

# --- Распаковка -------------------------------------------------------------
mkdir -p "$TMP/unpack"
tar -xzf "$TMP/$TARBALL" -C "$TMP/unpack"
SRC="$TMP/unpack/${TARBALL%.tar.gz}"
[ -f "$SRC/llmhostd" ] || SRC="$TMP/unpack"
[ -f "$SRC/llmhostd" ] || die "в тарболле нет llmhostd"
[ -f "$SRC/lhctl" ] || die "в тарболле нет lhctl"
[ -f "$SRC/manifest.json" ] || die "в тарболле нет manifest.json"

# --- Остановка старой версии (если была) ------------------------------------
# cp поверх запущенного бинаря = Text file busy, поэтому сначала стоп.
LLMHOST_DIR="${LLMHOST_DIR:-$HOME/.local/lib/llmhost}"
LLMHOST_BIN_DIR="${LLMHOST_BIN_DIR:-$HOME/.local/bin}"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT="$UNIT_DIR/llmhostd.service"

if [ "$NO_START" = 0 ]; then
  if [ -f "$UNIT" ] && systemctl --user stop llmhostd.service >/dev/null 2>&1; then
    say "Остановлена служба llmhostd.service"
  elif [ -x "$LLMHOST_DIR/lhctl" ]; then
    "$LLMHOST_DIR/lhctl" quit >/dev/null 2>&1 || true
    say "Демон остановлен через lhctl quit (если был запущен)"
  fi
fi

# --- Установка файлов -------------------------------------------------------
mkdir -p "$LLMHOST_DIR" "$LLMHOST_BIN_DIR"
cp -f "$SRC/llmhostd" "$SRC/lhctl" "$SRC/manifest.json" "$LLMHOST_DIR/"
chmod 755 "$LLMHOST_DIR/llmhostd" "$LLMHOST_DIR/lhctl"
for f in README.md RELEASE_NOTES.txt LICENSE; do
  if [ -f "$SRC/$f" ]; then
    cp -f "$SRC/$f" "$LLMHOST_DIR/"
  fi
done
ln -sf "$LLMHOST_DIR/lhctl" "$LLMHOST_BIN_DIR/lhctl"
say "Установлено: $LLMHOST_DIR (lhctl: $LLMHOST_BIN_DIR/lhctl)"

# --- Конфиг: каталог моделей в llmhost.conf ---------------------------------
mkdir -p "$DATA_DIR"
write_models_dir "$CONF" "$MODELS_DIR"
say "Конфиг: $CONF (models-dir = $MODELS_DIR)"
say "Поменять потом можно правкой этого файла — ключ models-dir в [general]"

# --- PATH: чтобы lhctl работал из любой оболочки ----------------------------
# Симлинк — только lhctl: демон юнит запускает по абсолютному пути из юнита,
# а llmhostd в PATH провоцирует ручные запуски мимо systemd.
ensure_path() { # $1 = файл; идемпотентно по литералу пути
  if grep -qF -- "$LLMHOST_BIN_DIR" "$1" 2>/dev/null; then
    return 0
  fi
  printf '\n# llmhost (install.sh): CLI-клиент lhctl\nexport PATH="%s:$PATH"\n' \
    "$LLMHOST_BIN_DIR" >> "$1"
  say "PATH: $LLMHOST_BIN_DIR дописан в $1 (вступит в силу в новой сессии)"
}

case ":$PATH:" in
  *":$LLMHOST_BIN_DIR:"*)
    say "PATH уже содержит $LLMHOST_BIN_DIR"
    ;;
  *)
    export PATH="$LLMHOST_BIN_DIR:$PATH"
    ensure_path "$HOME/.profile"               # login-шеллы (SSH), создаст при отсутствии
    if [ -f "$HOME/.bash_profile" ]; then      # если есть, он затеняет .profile у bash
      ensure_path "$HOME/.bash_profile"
    fi
    case "${SHELL:-}" in
      *zsh) ensure_path "${ZDOTDIR:-$HOME}/.zprofile" ;;
    esac
    ;;
esac

# --- systemd user unit (та же схема, что scripts/install-systemd.sh) --------
if [ "$NO_START" = 0 ] && command -v systemctl >/dev/null 2>&1; then
  # Данные демона должны лечь туда же, куда установщик писал llmhost.conf:
  # user-менеджер systemd не наследует шелловый LH_DATA_DIR.
  LH_DATA_DIR_ENV=""
  if [ -n "${LH_DATA_DIR:-}" ]; then
    LH_DATA_DIR_ENV="Environment=LH_DATA_DIR=$LH_DATA_DIR"
  fi
  # daemon-reload работает только при живом user-менеджере — он же и проверка.
  if mkdir -p "$UNIT_DIR" && systemctl --user daemon-reload >/dev/null 2>&1; then
    cat > "$UNIT" <<EOF
# llmhostd — headless-демон llmhost (OpenAI + control API).
# Сгенерировано install.sh; правки сюда — и systemctl --user daemon-reload.
[Unit]
Description=llmhostd — локальный LLM-супервизор
After=network-online.target
Wants=network-online.target
# Защита от рестарт-лупа (например, чужой процесс держит pid-файл):
# 5 неудачных стартов за 30 с — юнит в failed, и это видно.
StartLimitIntervalSec=30
StartLimitBurst=5

[Service]
ExecStart=$LLMHOST_DIR/llmhostd
${LH_DATA_DIR_ENV:-}
# Лог и в файл (ротация 1 МиБ), и в stderr → journalctl --user -u llmhostd.
Environment=LH_LOG_STDERR=1
# ВСЕГДА поднимать: чистый выход (lhctl quit, SIGTERM) — тоже авария для
# серверного демона. Единственный штатный способ остановить — systemctl --user stop.
Restart=always
RestartSec=2
# Дети не переживают родителя (замена Windows Job Object): стоп/рестарт юнита
# убивает всё дерево — llama-server, роутер и его детей-моделей.
KillMode=control-group
TimeoutStopSec=30

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    systemctl --user enable --now llmhostd.service
    say "Служба установлена и запущена: systemctl --user status llmhostd"
    say "Чтобы служба поднималась без входа в сессию (headless): loginctl enable-linger"
  else
    warn "systemd user-сессия недоступна — запустите вручную: $LLMHOST_DIR/llmhostd"
  fi
fi

# --- Движок llama.cpp (демон сам качает его по manifest.json) ----------------
if [ "$NO_START" = 1 ]; then
  say "Движок не ставился (--no-start): после запуска демона выполните lhctl engine install cuda12"
elif [ "$ENGINE" != "none" ]; then
  if "$LLMHOST_DIR/lhctl" status >/dev/null 2>&1; then
    if [ "$ENGINE" = "auto" ]; then
      if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        ENGINE="cuda12"
      else
        ENGINE="vulkan"
      fi
    fi
    say "Устанавливаю движок ($ENGINE) — крупная загрузка, это займёт время"
    if "$LLMHOST_DIR/lhctl" engine install "$ENGINE"; then
      say "Движок $ENGINE установлен"
    else
      warn "движок не поставился — позже вручную: $LLMHOST_BIN_DIR/lhctl engine install $ENGINE"
    fi
  else
    warn "демон не отвечает — движок не ставился. Если демон на нестандартном порту: LHCTL_PORT=ПОРТ lhctl engine install cuda12"
  fi
fi

# --- Итог -------------------------------------------------------------------
say
say "Готово. Дальше:"
say "  $LLMHOST_BIN_DIR/lhctl status        — состояние демона"
say "  $LLMHOST_BIN_DIR/lhctl pull <модель> — скачать модель (в $MODELS_DIR)"
say "  $LLMHOST_BIN_DIR/lhctl chat <модель> — диалог в терминале"
say "Данные: $DATA_DIR (движок, модели, лог); конфиг — $CONF"
say "Настройки: lhctl set — таблица и запись; правка $CONF применяется на лету"
say "Справка: $LLMHOST_DIR/README.md (раздел Linux; LAN-доступ — «Внешняя служба»)"
