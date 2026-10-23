"""Разбор просьбы моделью, точное подтверждение кодом и клиент службы.

Модель может предложить дату, но не может сохранить напоминание.
Сохранение доступно только после отдельной явной реплики пользователя.
"""

from __future__ import annotations

import json
import http.client
import re
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from .reminders import exact_time, parse_local_time
from .streaming import check_cancelled

INTENT = re.compile(r"\b(?:напомни|напомнить|напоминание|напоминания)\b", re.I)
YES = {"да", "подтверждаю", "да подтверждаю", "сохрани", "сохрани напоминание",
       "да сохрани", "да сохрани напоминание", "подтверждаю напоминание"}
NO = {"нет", "отмена", "отмени", "не надо", "не сохраняй"}
SCHEMA = {
    "type": "function", "function": {
        "name": "propose_reminder",
        "description": "Предложить напоминание или уточнить недостающую дату/время. Ничего не сохраняет.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "Что напомнить, до 500 символов."},
            "local_time": {"type": "string", "description": "YYYY-MM-DDTHH:MM:SS в указанном поясе, либо пустая строка."},
            "question": {"type": "string", "description": "Уточнение, если время неоднозначно/не указано; иначе пустая строка."},
        }, "required": ["text", "local_time", "question"], "additionalProperties": False},
    },
}


class ReminderError(ValueError):
    pass


class ReminderClient:
    def __init__(self, settings):
        self.url = settings.url.rstrip("/")
        if self.url and "://" not in self.url:
            self.url = "http://" + self.url
        self.token = settings.token

    def request(self, path, body=None):
        request = urllib.request.Request(
            self.url + path, data=None if body is None else json.dumps(body).encode(),
            headers={"X-Reminder-Token": self.token, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 400:
                try:
                    message = json.load(exc)["error"]
                except (ValueError, KeyError):
                    message = "Служба отклонила параметры напоминания."
                raise ReminderError(message) from None
            raise ReminderError("Нет доступа к службе напоминаний или она недоступна.") from None
        except (OSError, ValueError, http.client.HTTPException):
            raise ReminderError("Не удалось проверить результат в службе напоминаний. "
                                "При сохранении повтори «подтверждаю»: дубль не создастся.") from None


class ReminderDialogue:
    def __init__(self, settings):
        self.client = ReminderClient(settings)
        self.pending = None
        self.context = []
        self.expires = 0

    def reset(self):
        self.pending = None
        self.context.clear()
        self.expires = 0

    def handle(self, question, session_id, extract, cancelled=None):
        normalized = re.sub(r"[^\w\s]", "", question.lower()).strip()
        if self.expires and time.monotonic() >= self.expires:
            was_pending = self.pending is not None
            self.reset()
            if was_pending and normalized in YES:
                return "Подтверждение устарело. Повтори просьбу с датой и временем."
        if normalized in NO and (self.pending or self.context):
            self.reset()
            return "Напоминание не сохранено."
        if self.pending and normalized in YES:
            check_cancelled(cancelled)
            try:
                result = self.client.request("/reminders", self.pending)
            except ReminderError as exc:
                return str(exc)
            self.reset()
            if result["status"] == "cancelled":
                return "Это напоминание уже отменено."
            return (f"Напоминание сохранено: {exact_time(result['due'], result['zone'])}, "
                    f"{result['text']}. Доставка в Telegram.")

        if normalized == "мои напоминания":
            self.reset()
            if not self.client.url:
                return "Служба напоминаний не настроена: укажи reminders.url и REMINDERS_TOKEN."
            try:
                rows = self.client.request("/reminders")["reminders"]
            except ReminderError as exc:
                return str(exc)
            if not rows:
                return "Ожидающих напоминаний нет."
            return "\n".join(
                f"{row['id']}: {exact_time(row['due'], row['zone'])} — {row['text']} "
                f"[{row['status']}]" for row in rows
            )
        cancellation = re.fullmatch(r"отмени напоминание ([a-f0-9]{32})", normalized)
        if cancellation:
            self.reset()
            if not self.client.url:
                return "Служба напоминаний не настроена."
            check_cancelled(cancelled)
            try:
                self.client.request("/cancel", {"id": cancellation[1]})
            except ReminderError as exc:
                return str(exc)
            return "Напоминание отменено."

        intent = bool(INTENT.search(question))
        if not intent and not self.context:
            # Несвязанная реплика не оставляет скрытого подтверждения на потом.
            self.reset()
            return None
        if not self.client.url:
            return "Напоминания пока не подключены. Укажи reminders.url и REMINDERS_TOKEN, " \
                   "запусти службу на постоянно включённой машине. Ничего не сохранено."
        self.pending = None  # старое «да» больше не может сохранить старую дату
        try:
            health = self.client.request("/health")
            zone, service_now = health["zone"], health["now"]
            now = datetime.fromtimestamp(service_now, ZoneInfo(zone))
            messages = [{"role": "system", "content": (
                "Разбери просьбу о разовом напоминании через propose_reminder. "
                f"Сейчас {now.isoformat(timespec='seconds')}, часовой пояс {zone}. "
                "Даты считай относительно этого времени, а не времени компьютера. "
                "Если дата или время неоднозначны, local_time оставь пустым и задай question. "
                "В частности 'в семь' без утра/вечера означает необходимость уточнения. "
                "Не выбирай время по своему усмотрению. Повторяющиеся напоминания не поддерживаются. "
                "Не обещай сохранения или отправки. Всегда вызови инструмент."
            )}, *self.context, {"role": "user", "content": question}]
            check_cancelled(cancelled)
            data = extract(messages, SCHEMA)
            check_cancelled(cancelled)
            if not isinstance(data, dict):
                self.reset()
                return "Не удалось разобрать напоминание. Укажи текст, дату и точное время."
            clarification = data.get("question")
            if clarification:
                self.context = [*messages[1:], {"role": "assistant", "content": str(clarification)}][-8:]
                self.expires = time.monotonic() + 600
                return "Ничего не сохранено. " + str(clarification)[:500]
            text, local_time = data.get("text"), data.get("local_time")
            if not isinstance(text, str) or not 1 <= len(text.strip()) <= 500 or not isinstance(local_time, str):
                raise ReminderError("Укажи текст, дату и точное время напоминания.")
            due = parse_local_time(local_time, zone, service_now)
            self.pending = {"id": uuid.uuid4().hex, "session": session_id,
                            "text": text.strip(), "local_time": local_time, "zone": zone}
            self.context.clear()
            self.expires = time.monotonic() + 600
            return (f"Напомнить {exact_time(due, zone)}: {text.strip()}. "
                    "Сохранить? Скажи «подтверждаю» или «отмена». Пока не сохранено.")
        except (ReminderError, ValueError, KeyError, OverflowError) as exc:
            self.reset()
            return f"Напоминание не подготовлено: {exc}"
