"""Постоянная служба напоминаний: SQLite, HTTP и доставка в Telegram.

Запуск на постоянно включённой машине: python -m assistant.reminders.
Модели, микрофон, DeepSeek и polling Telegram здесь не нужны.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)
LEASE_SECONDS = 600  # больше всех таймаутов и повторов Telegram.call


def parse_local_time(value: str, zone: str, now: float) -> int:
    """Никаких догадок при пропущенном/двойном часе перехода на летнее время."""
    if "T" not in value:
        raise ValueError("Нужны и дата, и время в формате YYYY-MM-DDTHH:MM:SS.")
    local = datetime.fromisoformat(value)
    if local.tzinfo is not None:
        raise ValueError("Укажи местное время без смещения, часовой пояс задан службой.")
    tz = ZoneInfo(zone)
    candidates = set()
    for fold in (0, 1):
        aware = local.replace(tzinfo=tz, fold=fold)
        stamp = int(aware.timestamp())
        if datetime.fromtimestamp(stamp, tz).replace(tzinfo=None) == local:
            candidates.add(stamp)
    if len(candidates) != 1:
        raise ValueError("Это время пропущено или повторяется при переводе часов. Выбери другое.")
    due = candidates.pop()
    if due <= now:
        raise ValueError("Это время уже прошло. Уточни будущую дату и время.")
    return due


def exact_time(stamp: float, zone: str) -> str:
    local = datetime.fromtimestamp(stamp, ZoneInfo(zone))
    offset = local.strftime("%z")
    return f"{local:%d.%m.%Y в %H:%M:%S} ({zone}, UTC{offset[:3]}:{offset[3:]})"


class ReminderStore:
    def __init__(self, path: Path, zone: str, owner: int, clock=time.time):
        ZoneInfo(zone)  # ошибка настройки до запуска сервера
        if owner <= 0:
            raise ValueError("Нужен положительный TELEGRAM_ALLOWED_USER_ID.")
        self.path, self.zone, self.owner, self.clock = Path(path), zone, owner, clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS reminders (
                    id TEXT PRIMARY KEY, session TEXT NOT NULL, text TEXT NOT NULL,
                    due INTEGER NOT NULL, zone TEXT NOT NULL, chat_id INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt REAL NOT NULL DEFAULT 0, lease_until REAL NOT NULL DEFAULT 0,
                    lease TEXT, sent_at REAL, error TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS reminder_due ON reminders(status, due, next_attempt);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def create(self, data):
        key, text, session = data.get("id"), data.get("text"), data.get("session")
        if not isinstance(key, str) or len(key) != 32:
            raise ValueError("Некорректный ID напоминания.")
        uuid.UUID(hex=key)
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 500:
            raise ValueError("Текст напоминания должен содержать от 1 до 500 символов.")
        if not isinstance(session, str) or not 1 <= len(session) <= 128:
            raise ValueError("Некорректная сессия.")
        if data.get("zone") != self.zone:
            raise ValueError("Часовой пояс службы изменился. Запроси новое подтверждение.")
        local_time = data.get("local_time")
        if not isinstance(local_time, str):
            raise ValueError("Не указаны дата и время.")
        # Повтор сохранения после потери ответа разрешён даже после due.
        due = parse_local_time(local_time, self.zone, float('-inf'))
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT * FROM reminders WHERE id=?", (key,)).fetchone()
            if existing:
                if (existing["text"], existing["due"], existing["session"], existing["chat_id"]) != (
                    text.strip(), due, session, self.owner,
                ):
                    raise ValueError("Этот ID уже принадлежит другому напоминанию.")
                return dict(existing)
            if due <= self.clock():
                raise ValueError("Это время уже прошло. Запроси новую дату и подтверждение.")
            db.execute(
                "INSERT INTO reminders(id,session,text,due,zone,chat_id) VALUES(?,?,?,?,?,?)",
                (key, session, text.strip(), due, self.zone, self.owner),
            )
            return dict(db.execute("SELECT * FROM reminders WHERE id=?", (key,)).fetchone())

    def active(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute(
                "SELECT * FROM reminders WHERE status IN ('pending','sending','failed') "
                "ORDER BY due LIMIT 50"
            )]

    def cancel(self, key):
        if not isinstance(key, str) or len(key) != 32:
            raise ValueError("Укажи полный ID напоминания из списка.")
        with self.connect() as db:
            row = db.execute("SELECT * FROM reminders WHERE id=?", (key,)).fetchone()
            if not row or row["status"] not in ("pending", "failed"):
                raise ValueError("Можно отменить только ожидающее напоминание; отправка могла уже начаться.")
            changed = db.execute(
                "UPDATE reminders SET status='cancelled' WHERE id=? AND status IN ('pending','failed')", (key,)
            ).rowcount
            if not changed:
                raise ValueError("Отправка уже началась; отменить её нельзя.")

    def claim(self):
        now, lease = self.clock(), uuid.uuid4().hex
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM reminders WHERE chat_id=? AND due<=? AND "
                "((status IN ('pending','failed') AND next_attempt<=?) OR (status='sending' AND lease_until<=?)) "
                "ORDER BY due LIMIT 1", (self.owner, now, now, now),
            ).fetchone()
            if row is None:
                return None
            db.execute("UPDATE reminders SET status='sending', lease=?, lease_until=?, "
                       "attempts=attempts+1 WHERE id=?", (lease, now + LEASE_SECONDS, row["id"]))
            return dict(db.execute("SELECT * FROM reminders WHERE id=?", (row["id"],)).fetchone())

    def finish(self, row, error=None):
        now = self.clock()
        # 400/401/403/404 требуют исправления настроек или разблокировки бота.
        permanent = error is not None and error.code in (400, 401, 403, 404)
        status = 'sent' if error is None else ('failed' if permanent else 'pending')
        retry = max(min(3600, 30 * 2 ** min(row["attempts"], 7)), getattr(error, "retry_after", 0))
        if permanent:
            retry = max(3600, retry)  # после исправления токена/блокировки снова доставим
        with self.connect() as db:
            db.execute(
                "UPDATE reminders SET status=?, sent_at=?, next_attempt=?, error=?, "
                "lease=NULL, lease_until=0 WHERE id=? AND lease=? AND status='sending'",
                (status, now if error is None else None, now + retry,
                 str(error) if error else '', row["id"], row["lease"]),
            )


def deliver_one(store, telegram):
    from .telegram_bot import TelegramError

    row = store.claim()
    if row is None:
        return False
    try:
        telegram.text(row["chat_id"], f"Напоминание: {row['text']}\n"
                      f"На {exact_time(row['due'], row['zone'])}.")
    except TelegramError as exc:
        store.finish(row, exc)
        log.warning("напоминание %s: ошибка доставки Telegram %s", row["id"], exc.code)
    else:
        store.finish(row)
    return True


def make_handler(store, token):
    if not token:
        raise ValueError("Нужен REMINDERS_TOKEN.")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def reply(self, status, body):
            blob = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def authorized(self):
            if not hmac.compare_digest(self.headers.get("X-Reminder-Token", "").encode(), token.encode()):
                self.reply(403, {"error": "Нет доступа к службе напоминаний."})
                return False
            return True

        def do_GET(self):
            if not self.authorized():
                return
            if self.path == "/health":
                self.reply(200, {"zone": store.zone, "now": store.clock()})
            elif self.path == "/reminders":
                try:
                    self.reply(200, {"reminders": store.active()})
                except sqlite3.Error:
                    self.reply(503, {"error": "Хранилище напоминаний недоступно."})
            else:
                self.reply(404, {"error": "Неизвестный адрес."})

        def do_POST(self):
            if not self.authorized():
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16384:
                    raise ValueError("Некорректный размер запроса.")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError("Нужен JSON-объект.")
                if self.path == "/reminders":
                    result = store.create(data)
                elif self.path == "/cancel":
                    store.cancel(data.get("id"))
                    result = {"cancelled": True}
                else:
                    self.reply(404, {"error": "Неизвестный адрес."})
                    return
                self.reply(200, result)
            except (ValueError, TypeError, OverflowError) as exc:
                self.reply(400, {"error": str(exc)})
            except sqlite3.Error:
                self.reply(503, {"error": "Хранилище напоминаний недоступно."})

    return Handler


def main():
    from .config import load_config
    from .telegram_bot import Telegram

    logging.basicConfig(level=logging.INFO)
    cfg = load_config().reminders
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    try:
        if not bot_token or not cfg.token:
            raise ValueError("Укажи TELEGRAM_BOT_TOKEN и REMINDERS_TOKEN в .env.")
        store = ReminderStore(cfg.database, cfg.timezone, int(os.getenv("TELEGRAM_ALLOWED_USER_ID", "0")))
        server = ThreadingHTTPServer((cfg.host, cfg.port), make_handler(store, cfg.token))
    except (ValueError, KeyError, OSError, sqlite3.Error) as exc:
        print(f"Служба напоминаний не запущена: {exc}")
        return 1
    server.daemon_threads = True
    stopped = threading.Event()
    telegram = Telegram(bot_token)

    def schedule():
        while not stopped.is_set():
            try:
                if deliver_one(store, telegram):
                    continue
            except Exception:
                log.exception("ошибка планировщика; запись остаётся в SQLite")
            stopped.wait(2)

    worker = threading.Thread(target=schedule, name="reminders", daemon=True)
    worker.start()
    print(f"Напоминания: {cfg.host}:{cfg.port}, {cfg.timezone}, БД {cfg.database}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stopped.set()
        server.server_close()
        worker.join(timeout=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
