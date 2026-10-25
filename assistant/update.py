"""Обновление спутника с сервера: код больше не носим на флешке.

Как устроено. Сервер при запуске запоминает файлы, нужные спутнику, —
ровно те, что лежат на диске в этот момент, то есть тот код, которым
он сам работает. Спутник перед запуском (`client.bat`) сверяет свои
файлы по контрольным суммам и скачивает изменившиеся. Версия — сумма
по всем файлам разом; сервер пишет её в каждый ответ, и спутник,
увидев новую, после ответа выходит, а `client.bat` запускает его
заново уже с обновлением. То есть «перезапустил сервер — ноутбук
подтянулся сам».

Что отдаётся — строго список ниже. `data/`, `.env`,
`config.local.yaml` и модели не отдаются никогда: у спутника своё,
а личного ему не положено. `.bat` тоже не отдаём: cmd читает
выполняемый файл кусками по ходу, и подмена посреди работы ломает его.

Запуск на спутнике: `python -m assistant.update [АДРЕС]`. Без адреса —
`server.url` из конфига. Сервер не отвечает — молча работаем со
старым кодом, это не повод не запускаться.

Тут только стандартная библиотека и pyyaml: обновлялка должна
работать, даже когда остальное сломано.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Что нужно спутнику. Пакет целиком — потому что проще, чем следить,
# кто кого импортирует, а лишний файл весит килобайты.
PATTERNS = ("assistant/*.py", "config.yaml", "requirements-client.txt")
REQUIREMENTS = "requirements-client.txt"
BACKUP = ".update_backup"

# Заголовок с версией в каждом ответе сервера.
VERSION_HEADER = "X-Client-Version"
# Код выхода спутника «сервер обновился, запусти меня заново».
RESTART_CODE = 3


def collect(root: Path = ROOT) -> dict[str, bytes]:
    """Файлы для спутника: путь через «/» -> содержимое."""
    files = {}
    for pattern in PATTERNS:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                files[path.relative_to(root).as_posix()] = path.read_bytes()
    return files


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest(files: dict[str, bytes]) -> dict[str, str]:
    return {name: digest(data) for name, data in files.items()}


def version(sums: dict[str, str]) -> str:
    """Одна короткая строка на весь набор: совпала — обновлять нечего."""
    joined = "\n".join(f"{name} {value}" for name, value in sorted(sums.items()))
    return digest(joined.encode("utf-8"))[:12]


def local_version(root: Path = ROOT) -> str:
    return version(manifest(collect(root)))


# --- сторона спутника -----------------------------------------------------


def _get(url: str, token: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"X-Token": token})
    with urllib.request.urlopen(request, timeout=timeout) as answer:
        return answer.read()


def update(url: str, token: str = "", root: Path = ROOT, *, timeout: float = 10.0) -> bool:
    """Подтягивает изменившиеся файлы. True — что-то обновилось."""
    if "://" not in url:
        url = "http://" + url
    url = url.rstrip("/")

    try:
        remote = json.loads(_get(url + "/client/manifest", token, timeout))
    except Exception as exc:  # noqa: BLE001
        print(f"Обновление пропущено: сервер {url} не отдал список файлов "
              f"({type(exc).__name__}: {exc}). Работаю со старым кодом.")
        return False

    files: dict[str, str] = remote["files"]
    mine = manifest(collect(root))
    changed = [name for name, value in files.items() if mine.get(name) != value]
    # Устаревшие модули удаляем только внутри пакета: так спутник
    # не тащит за собой файл, которого на сервере уже нет.
    stale = [name for name in mine if name not in files and name.startswith("assistant/")]
    if not changed and not stale:
        print(f"Код спутника свежий (версия {remote['version']}).")
        return False

    # Сперва всё скачиваем и проверяем во временную папку. Оборвалась
    # сеть на середине — на месте не тронуто ничего.
    with tempfile.TemporaryDirectory(prefix="alisa-update-") as temp:
        staged = Path(temp)
        for name in changed:
            try:
                data = _get(url + "/client/file?path=" + urllib.parse.quote(name), token, timeout)
            except Exception as exc:  # noqa: BLE001
                print(f"Обновление отменено: {name} не скачался ({type(exc).__name__}: {exc}).")
                return False
            if digest(data) != files[name]:
                print(f"Обновление отменено: {name} пришёл повреждённым.")
                return False
            target = staged / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

        # Прежние версии — в .update_backup: если новый код не
        # заведётся, можно вернуть руками.
        backup = root / BACKUP
        shutil.rmtree(backup, ignore_errors=True)
        for name in changed + stale:
            current = root / name
            if current.exists():
                (backup / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(current, backup / name)

        for name in changed:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged / name, target)
        for name in stale:
            (root / name).unlink(missing_ok=True)

    print(f"Спутник обновлён до версии {remote['version']}:")
    for name in changed:
        print(f"  {name}")
    for name in stale:
        print(f"  {name} (удалён)")

    if REQUIREMENTS in changed:
        print("Поменялись зависимости, ставлю...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(root / REQUIREMENTS)],
            check=False,
        )
        if result.returncode != 0:
            print("pip не справился. Повтори руками: "
                  "py -3 -m pip install -r requirements-client.txt")
    return True


def main(argv: list[str]) -> int:
    # Аргументы те же, что у client.bat: адрес, если есть, — первый
    # без чёрточек, остальное (--voice и т.п.) нам не нужно.
    address = next((arg for arg in argv if not arg.startswith("-")), "")

    import yaml

    def read(name: str) -> dict:
        path = ROOT / name
        if not path.exists():
            return {}
        return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("server") or {}

    # Только ветка server из двух файлов: полный load_config тянет
    # .env и прочее, а обновлялке это не нужно и может быть сломано.
    server = {**read("config.yaml"), **read("config.local.yaml")}
    address = address or server.get("url") or ""
    if not address:
        print("Обновление пропущено: адрес сервера не задан (server.url).")
        return 0
    update(address, str(server.get("token") or ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
