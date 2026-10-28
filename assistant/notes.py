"""Голосовые заметки: «запиши в заметки, что…» дописывает строку в data/notes.md.

Это первая запись на диск по голосовой команде, а whisper иногда
ошибается, поэтому сперва предпросмотр и только после «подтверждаю» —
запись. Модель здесь не участвует: текст заметки вырезается из фразы
регулярным выражением. Так быстрее, бесплатно и ничего не перевирается.

Читать заметки отдельно не нужно: notes.md лежит в data/ и уезжает в
промпт вместе с остальными файлами, перечитываясь при изменении.
"""

from __future__ import annotations

import re
import threading
import time
from datetime import datetime
from pathlib import Path

from .reminder_dialogue import INTENT as REMINDER_INTENT, NO, YES
from .streaming import check_cancelled

FILE_NAME = "notes.md"
CONFIRM_S = 600

# «запиши в заметки, что…», «запомни в заметках…», «добавь заметку: …»,
# «заметка: …», «новая заметка …». Хвост после триггера — это и есть
# заметка; пустой хвост — значит спросим, что записать.
TRIGGER = re.compile(
    r"^\s*(?:"
    r"(?:запиши|запомни|сохрани|добавь|сделай)\s+(?:в\s+)?(?:мои\s+)?"
    r"(?:заметк[иуа]|заметках|заметку)"
    r"|(?:новая\s+)?заметка"
    r")\b[\s,:—-]*(?:что\s+)?(?P<text>.*)$",
    re.I | re.S,
)
DELETE = re.compile(
    r"^\s*(?:удали|сотри|отмени|убери)\s+последнюю\s+заметку\b", re.I
)
# Только строки, которые пишет сам ассистент: руками дописанное в файл
# «удали последнюю заметку» не тронет.
LINE = re.compile(r"^- \d{4}-\d{2}-\d{2} \S+ \d{2}:\d{2} — ")
WEEKDAYS = ("пн", "вт", "ср", "чт", "пт", "сб", "вс")
HEADER = (
    "# Заметки\n\n"
    "Надиктовано голосом, по строке на заметку, новые внизу. "
    "Строку можно поправить или удалить руками.\n\n"
)

# Сессий много (спутник, Telegram, браузер), файл один.
_FILE_LOCK = threading.Lock()


def note_request(question: str) -> str | None:
    """Текст заметки из фразы; "" — просили записать, но не сказали что."""
    if REMINDER_INTENT.search(question):
        return None
    match = TRIGGER.match(question)
    if not match:
        return None
    return match["text"].strip().rstrip(".").strip()


def note_line(text: str, now: datetime) -> str:
    return f"- {now:%Y-%m-%d} {WEEKDAYS[now.weekday()]} {now:%H:%M} — {text}\n"


class NoteBook:
    """Файл notes.md: дописать строку и снять последнюю."""

    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / FILE_NAME

    def append(self, text: str, now: datetime | None = None) -> None:
        line = note_line(text, now or datetime.now())
        with _FILE_LOCK:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fresh = not self.path.exists() or not self.path.read_text(encoding="utf-8").strip()
            with self.path.open("a", encoding="utf-8", newline="\n") as file:
                if fresh:
                    file.write(HEADER)
                file.write(line)

    def remove_last(self) -> str | None:
        """Снять последнюю заметку; вернуть её текст или None, если снимать нечего."""
        with _FILE_LOCK:
            if not self.path.exists():
                return None
            lines = self.path.read_text(encoding="utf-8").splitlines(keepends=True)
            for index in range(len(lines) - 1, -1, -1):
                if LINE.match(lines[index]):
                    removed = lines.pop(index)
                    self.path.write_text("".join(lines), encoding="utf-8", newline="\n")
                    return LINE.sub("", removed).strip()
        return None


class NoteDialogue:
    """Предпросмотр и подтверждение заметки в пределах одной сессии."""

    def __init__(self, data_dir: Path) -> None:
        self.book = NoteBook(data_dir)
        self.pending: str | None = None
        self.awaiting_text = False
        self.expires = 0.0

    @property
    def active(self) -> bool:
        return self.pending is not None or self.awaiting_text

    def reset(self) -> None:
        self.pending = None
        self.awaiting_text = False
        self.expires = 0.0

    def handle(self, question: str, cancelled=None) -> str | None:
        """Ответ про заметку или None, если реплика не про заметки."""
        normalized = re.sub(r"[^\w\s]", "", question.lower()).strip()
        if self.expires and time.monotonic() >= self.expires:
            was_pending = self.pending is not None
            self.reset()
            if was_pending and normalized in YES:
                return "Подтверждение устарело. Повтори заметку."

        if DELETE.match(question):
            self.reset()
            check_cancelled(cancelled)
            removed = self.book.remove_last()
            if removed is None:
                return "Удалять нечего: надиктованных заметок нет."
            return f"Удалила заметку: {removed}."

        if self.active and normalized in NO:
            self.reset()
            return "Заметка не записана."
        if self.pending is not None and normalized in YES | {"запиши", "да запиши"}:
            check_cancelled(cancelled)
            self.book.append(self.pending)
            self.reset()
            return "Записала."

        text = note_request(question)
        if text is None and self.awaiting_text:
            # Ответ на «что записать?» — это и есть заметка.
            text = question.strip().rstrip(".").strip()
        if text is None:
            # Несвязанная реплика не оставляет скрытого подтверждения на потом.
            self.reset()
            return None
        if not text:
            self.reset()
            self.awaiting_text = True
            self.expires = time.monotonic() + CONFIRM_S
            return "Что записать?"
        self.reset()
        self.pending = text[:500]
        self.expires = time.monotonic() + CONFIRM_S
        return f"Записать: {self.pending}? Скажи «подтверждаю» или «отмена»."
