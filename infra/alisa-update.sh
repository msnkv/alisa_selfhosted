#!/usr/bin/env bash
# Автообновление сервера Алисы из git. Запускается таймером systemd раз в
# минуту (alisa-update.timer), как устроено и почему — MIGRATION.md.
#
# Один прогон:
#   1. git fetch; нового коммита нет — выходим.
#   2. Сервер занят (идёт ответ, был вопрос меньше QUIET_S назад или ждёт
#      «подтверждаю») — выходим, через минуту попробуем снова.
#   3. git pull, зависимости, если менялись, перезапуск трёх служб.
#   4. Все поднялись и не падают — готово. Нет — откат на прошлый коммит,
#      коммит помечается плохим и больше не ставится.
#
# Ставится КОПИЕЙ в /usr/local/bin, а не запускается из репозитория:
# bash читает скрипт по ходу выполнения, и git pull посреди прогона
# подменил бы его на лету. Плохой коммит не сломает и сам откат.

set -euo pipefail

CONF=${ALISA_UPDATE_CONF:-/etc/alisa-update.conf}
ROOT=/opt/alisa_selfhosted
BRANCH=master
QUIET_S=120          # столько секунд без вопросов, чтобы перезапускать
HEALTH_WAIT_S=300    # столько ждём, пока сервер прогреет модели
SETTLE_S=30          # столько бот и напоминания должны прожить без падений
SERVICES="alisa-reminders alisa-server alisa-telegram"
NOTIFY=1             # писать владельцу в Telegram об обновлении и откате
# shellcheck source=/dev/null
[ -f "$CONF" ] && . "$CONF"

OWNER=$(stat -c %U "$ROOT")
PY="$ROOT/.venv/bin/python"
STATE="$ROOT/state"
BAD="$STATE/update-bad-commit"

log() { echo "alisa-update: $*"; }
# git и pip — от владельца репозитория, с его HOME: там ключ для GitHub.
as_owner() { sudo -H -u "$OWNER" "$@"; }
git_() { as_owner git -C "$ROOT" "$@"; }

env_value() {
    # || true: без .env pipefail иначе обрывал бы весь прогон через set -e.
    sed -n "s/^$1=//p" "$ROOT/.env" 2>/dev/null | tail -n 1 | tr -d "\r\"'" || true
}

notify() {
    [ "$NOTIFY" = 1 ] || return 0
    local token chat
    token=$(env_value TELEGRAM_BOT_TOKEN)
    # Первый ID в списке — владелец, как и для напоминаний.
    chat=$(env_value TELEGRAM_ALLOWED_USER_ID | cut -d, -f1 | tr -d ' ')
    [ -n "$token" ] && [ -n "$chat" ] || return 0
    curl -s --max-time 10 -o /dev/null \
        --data-urlencode "chat_id=$chat" --data-urlencode "text=$1" \
        "https://api.telegram.org/bot$token/sendMessage" || true
}

# quiet | busy | down. Адрес и токен берём из настроек самого сервера.
server_state() {
    (cd "$ROOT" && as_owner "$PY" - "$QUIET_S" 2>/dev/null) <<'EOF' || echo unknown
import json, sys, urllib.error, urllib.request
from assistant.config import load_config

cfg = load_config()
quiet_s = float(sys.argv[1])
request = urllib.request.Request(
    f"http://127.0.0.1:{cfg.server.port}/status",
    headers={"X-Token": cfg.server.token},
)
try:
    with urllib.request.urlopen(request, timeout=5) as response:
        status = json.load(response)
except urllib.error.HTTPError:
    print("unknown")
except OSError:
    print("down")
else:
    idle = not status["working"] and not status["drafts"] and status["idle_s"] >= quiet_s
    print("quiet" if idle else "busy")
EOF
}

install_requirements() {
    local reqs
    reqs=$(mktemp)
    chmod 644 "$reqs"
    # pywebrtc-audio — только для голосового «стоп» спутника, колёса под Windows.
    grep -v pywebrtc "$ROOT/requirements.txt" > "$reqs"
    (cd "$ROOT" && as_owner "$PY" -m pip install -q -r "$reqs" -r requirements-telegram.txt)
    local code=$?
    rm -f "$reqs"
    return $code
}

restart_all() {
    # shellcheck disable=SC2086
    systemctl restart $SERVICES
}

healthy() {
    local port waited=0
    port=$(cd "$ROOT" && as_owner "$PY" -c \
        "from assistant.config import load_config; print(load_config().server.port)") || return 1
    # /health отвечает только после прогрева моделей: раньше сервер не слушает.
    until curl -sf --max-time 5 "http://127.0.0.1:$port/health" >/dev/null; do
        waited=$((waited + 5))
        [ "$waited" -lt "$HEALTH_WAIT_S" ] || { log "сервер не ответил за $HEALTH_WAIT_S с"; return 1; }
        sleep 5
    done
    sleep "$SETTLE_S"
    local unit
    for unit in $SERVICES; do
        # Restart=always поднимает упавшую службу сама, поэтому «active»
        # мало: смотрим ещё, не перезапускалась ли она после нашего рестарта.
        if [ "$(systemctl show -p ActiveState --value "$unit")" != active ] ||
           [ "$(systemctl show -p NRestarts --value "$unit")" != 0 ]; then
            log "$unit не держится"
            return 1
        fi
    done
}

# --- прогон ---------------------------------------------------------------

cd "$ROOT"
mkdir -p "$STATE"
chown "$OWNER": "$STATE"

git_ fetch --quiet origin "$BRANCH"
old=$(git_ rev-parse HEAD)
new=$(git_ rev-parse "origin/$BRANCH")
[ "$old" != "$new" ] || exit 0

if [ "$(cat "$BAD" 2>/dev/null)" = "$new" ]; then
    exit 0  # этот коммит уже откатывали, ждём следующего
fi
if ! git_ merge-base --is-ancestor HEAD "origin/$BRANCH"; then
    log "на сервере свои коммиты, обновление вручную"
    exit 0
fi
if [ -n "$(git_ status --porcelain --untracked-files=no)" ]; then
    log "на сервере правлены файлы из git, обновление вручную: git status"
    exit 0
fi

state=$(server_state)
case "$state" in
    quiet|down) ;;  # упавшему серверу мешать нечем, а новый коммит его, может, и чинит
    busy) exit 0 ;;
    *) log "не понял, занят ли сервер: /status не ответил"; exit 0 ;;
esac

subject=$(git_ log -1 --format=%s "$new")
log "обновляюсь ${old:0:7} -> ${new:0:7}: $subject"
changed=$(git_ diff --name-only "$old" "$new")
git_ merge --quiet --ff-only "origin/$BRANCH"

ok=1
if echo "$changed" | grep -q '^requirements'; then
    log "изменились зависимости, ставлю"
    install_requirements || ok=0
fi
if [ "$ok" = 1 ]; then
    restart_all
    healthy || ok=0
fi

if [ "$ok" = 1 ]; then
    rm -f "$BAD"
    log "готово: ${new:0:7}"
    notify "Алиса обновилась: ${new:0:7} $subject"
    exit 0
fi

log "откатываюсь на ${old:0:7}"
echo "$new" > "$BAD"
git_ reset --quiet --hard "$old"
if echo "$changed" | grep -q '^requirements'; then
    install_requirements || log "зависимости после отката не встали"
fi
restart_all
if healthy; then
    notify "Обновление ${new:0:7} не взлетело и откачено на ${old:0:7}. Смотри: journalctl -u alisa-server"
else
    log "после отката сервер тоже не поднялся"
    notify "Обновление ${new:0:7} откачено, но сервер не поднялся и после отката. Нужны руки."
fi
exit 1
