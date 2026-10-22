"""Журнал диалогов: что спрашивали, что отвечали, в каком режиме.

Зачем отдельно от обычных логов. В `logs/assistant.log` пишется всё
подряд, и человеку это читать можно, а машине — нет. Здесь другое: по
строке на обмен, в JSON, с устойчивым набором полей. Такой файл можно
потом разобрать скриптом: посмотреть, как владелец спрашивает, что
переспрашивает, где поправляет — и при желании собрать из этого набор
для обучения своей модели.

Поэтому главное требование к этому файлу — **стабильная схема**. Поля
можно добавлять, но не переименовывать и не выкидывать: иначе старые
записи перестанут складываться с новыми.

Где лежит: `logs/dialogue.jsonl`, папка в `.gitignore`. Это личные
разговоры, наружу они не уезжают.

Про группировку: одна «беседа» — это всё до сброса памяти. `forget()`
начинает новую. Без этого уточнения («а сколько соли?») нельзя будет
отличить от самостоятельных вопросов, а ровно они и интересны: из них
видно, чего в первом ответе не хватило.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

# Признаки недовольства ответом. Нужны не для поведения ассистента, а
# чтобы потом эти места было легко найти в журнале: именно они говорят,
# как надо было ответить. Список намеренно грубый - лучше пометить
# лишнее, чем потерять нужное.
_COMPLAINT = re.compile(
    r"\bне\s+то\b|не\s+об\s+этом|не\s+про\s+это|я\s+не\s+(?:про|об)\s+это"
    r"|неправильно|ты\s+не\s+понял|не\s+понял[ао]?\s+вопрос"
    r"|\bкороче\b|\bпроще\b|\bне\s+так\b|я\s+спрашивал"
    r"|при\s+чём\s+(?:тут|здесь)|это\s+не\s+ответ",
    re.IGNORECASE,
)


def looks_like_complaint(text: str) -> bool:
    """Похоже, что предыдущий ответ не устроил."""
    return bool(_COMPLAINT.search(text or ""))


class DialogueLog:
    """Пишет по строке JSON на каждый обмен."""

    def __init__(self, path: Path, *, enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled
        self._lock = threading.Lock()
        self.conversation = uuid.uuid4().hex[:12]
        self.turn = 0

    def new_conversation(self) -> None:
        """Память сбросили — дальше это уже другой разговор."""
        self.conversation = uuid.uuid4().hex[:12]
        self.turn = 0

    def append(self, record: dict) -> None:
        if not self.enabled:
            return
        self.turn += 1
        line = {
            "at": datetime.now().isoformat(timespec="seconds"),
            "conversation": self.conversation,
            "turn": self.turn,
            **record,
        }
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(line, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001
            # Журнал не должен ронять ответ: если не записалось - ну и ладно,
            # но в обычный лог об этом скажем.
            log.warning("не записал обмен в журнал диалогов: %s", exc)


def read_all(path: Path) -> list[dict]:
    """Читает журнал, пропуская битые строки.

    Битые строки возможны: файл дописывается на ходу, и если процесс
    убили посреди записи, последняя строка может быть неполной.
    """
    if not path.is_file():
        return []
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("строка %d журнала диалогов битая, пропускаю", number)
    return records


def describe(path: Path, last: int = 5) -> str:
    """Короткая сводка по журналу — для команды --history."""
    records = read_all(path)
    if not records:
        return f"Журнал диалогов пуст или ещё не создан: {path}"

    conversations = {r.get("conversation") for r in records}
    complaints = [r for r in records if r.get("complaint")]
    with_tools = [r for r in records if r.get("tools")]
    follow_ups = [r for r in records if (r.get("turn") or 1) > 1]
    modes: dict[str, int] = {}
    for r in records:
        for key in ("detail", "long_form"):
            if r.get(key):
                modes[key] = modes.get(key, 0) + 1
        if r.get("mode"):
            modes[r["mode"]] = modes.get(r["mode"], 0) + 1

    lines = [
        f"Журнал диалогов: {path}",
        f"  обменов:        {len(records)}",
        f"  бесед:          {len(conversations)}",
        f"  уточнений:      {len(follow_ups)} (вопрос не первый в беседе)",
        f"  с поиском:      {len(with_tools)}",
        f"  недовольства:   {len(complaints)}",
    ]
    if modes:
        lines.append(
            "  режимы:         "
            + ", ".join(f"{name} {count}" for name, count in sorted(modes.items()))
        )

    lines.append(f"\nПоследние {min(last, len(records))}:")
    for r in records[-last:]:
        mark = "!" if r.get("complaint") else " "
        lines.append(f"  {mark} [{r.get('at', '')}] {r.get('question', '')}")
        lines.append(f"      → {r.get('answer', '')}")
    return "\n".join(lines)
