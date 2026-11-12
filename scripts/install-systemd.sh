#!/usr/bin/env bash
# Установка llmhostd как systemd user-службы: ~/.config/systemd/user/llmhostd.service.
#
#   scripts/install-systemd.sh [ПУТЬ_К_llmhostd]   # установить и запустить (enable --now)
#   scripts/install-systemd.sh --no-start [ПУТЬ]   # установить, не запускать
#
# Путь по умолчанию — build-linux/llmhostd рядом со скриптом. Данные демон берёт
# из $LH_DATA_DIR → $XDG_DATA_HOME/llmhost → ~/.local/share/llmhost (как всегда);
# заданный $LH_DATA_DIR прописывается в юнит строкой Environment=LH_DATA_DIR=…
# Конфиг приложения — data-dir/llmhost.conf (INI): [general] — настройки,
# остальные секции — per-model переопределения.
# Управление: systemctl --user start|stop|restart|status llmhostd,
# журнал: journalctl --user -u llmhostd -f.
set -euo pipefail

NO_START=0
[ "${1:-}" = "--no-start" ] && { NO_START=1; shift; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXE="${1:-$HERE/../build-linux/llmhostd}"
EXE="$(readlink -f "$EXE")"
[ -x "$EXE" ] || { echo "нет исполняемого $EXE" >&2; exit 1; }

UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT="$UNIT_DIR/llmhostd.service"
mkdir -p "$UNIT_DIR"
# Данные демона — туда же, куда смотрит шелл: user-менеджер systemd не
# наследует LH_DATA_DIR, нестандартный каталог фиксируем строкой Environment.
LH_DATA_DIR_ENV=""
if [ -n "${LH_DATA_DIR:-}" ]; then
  LH_DATA_DIR_ENV="Environment=LH_DATA_DIR=$LH_DATA_DIR"
fi
cat > "$UNIT" <<EOF
# llmhostd — headless-демон llmhost (OpenAI :7330 + control :7331).
# Сгенерировано scripts/install-systemd.sh; правки сюда — и daemon-reload.
[Unit]
Description=llmhostd — локальный LLM-супервизор
After=network-online.target
Wants=network-online.target
# Защита от рестарт-лупа (например, чужой процесс держит pid-файл):
# 5 неудачных стартов за 30 с — юнит в failed, и это видно.
StartLimitIntervalSec=30
StartLimitBurst=5

[Service]
ExecStart=$EXE
${LH_DATA_DIR_ENV:-}
# Лог и в файл (ротация 1 МиБ), и в stderr → journalctl --user -u llmhostd.
Environment=LH_LOG_STDERR=1
# ВСЕГДА поднимать: чистый выход (lhctl quit, SIGTERM) — тоже авария для
# серверного демона. Единственный штатный способ остановить — systemctl --user stop.
Restart=always
RestartSec=2
# Главный механизм «дети не переживают родителя» (замена Windows Job Object):
# стоп/рестарт юнита убивает всё дерево — llama-server, роутер и его детей-моделей.
KillMode=control-group
TimeoutStopSec=30

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
echo "юнит записан: $UNIT ($EXE)"
if [ "$NO_START" = 1 ]; then
  echo "запуск не выполнен (--no-start): systemctl --user enable --now llmhostd"
else
  systemctl --user enable --now llmhostd
  systemctl --user status llmhostd --no-pager -n 3 || true
fi
# Чтобы служба поднималась при старте машины без входа в сессию — lingering
# (может попросить пароль; без неё юнит стартует при первом входе пользователя):
#   loginctl enable-linger
