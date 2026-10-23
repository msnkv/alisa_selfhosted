"""Постоянство, подтверждение и доставка без настоящего Telegram/DeepSeek."""

import json
import subprocess
import sys
import tempfile
import threading
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from assistant.config import RemindersConfig, load_config
from assistant.llm import Assistant
from assistant.reminder_dialogue import ReminderClient, ReminderDialogue, ReminderError
from assistant.reminders import LEASE_SECONDS, ReminderStore, deliver_one, make_handler, parse_local_time
from assistant.telegram_bot import TelegramError
from assistant.streaming import AnswerCancelled

NOW = datetime(2030, 1, 1, 12, tzinfo=timezone.utc).timestamp()
LOCAL_TIME = "2030-01-02T07:00:00"


class StoreFixture:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.clock = Mock(return_value=NOW)
        self.path = Path(self.tmp.name) / "state" / "reminders.sqlite3"
        self.store = ReminderStore(self.path, "Europe/Moscow", 123, self.clock)
        self.payload = {"id": uuid.uuid4().hex, "session": "satellite:test",
                        "text": "поставить тесто", "local_time": LOCAL_TIME, "zone": "Europe/Moscow"}

    def create(self):
        return self.store.create(self.payload)

    def status(self, key=None):
        with self.store.connect() as db:
            return dict(db.execute("SELECT * FROM reminders WHERE id=?", (key or self.payload["id"],)).fetchone())

    def due(self):
        row = self.create()
        self.clock.return_value = row["due"] + 1
        return row


class StoreTests(StoreFixture, unittest.TestCase):
    def test_restart_delivers_overdue_without_alisa(self):
        self.due()
        restarted = ReminderStore(self.path, "Europe/Moscow", 123, self.clock)
        telegram = Mock()
        self.assertTrue(deliver_one(restarted, telegram))
        telegram.text.assert_called_once()
        chat, text = telegram.text.call_args.args
        self.assertEqual(chat, 123)
        self.assertIn("поставить тесто", text)
        self.assertIn("02.01.2030 в 07:00:00", text)
        self.assertEqual(self.status()["status"], "sent")
        self.assertFalse(deliver_one(restarted, telegram))

    def test_idempotent_create_even_after_due_time(self):
        original = self.due()
        self.assertEqual(self.create()["id"], original["id"])
        self.assertEqual(len(self.store.active()), 1)
        with self.assertRaises(ValueError):
            self.store.create({**self.payload, "text": "другая задача"})

    def test_timezone_conversion_and_ambiguous_clock_changes(self):
        self.assertEqual(self.create()["due"], datetime(2030, 1, 2, 4, tzinfo=timezone.utc).timestamp())
        for local in ("2026-03-29T02:30:00", "2026-10-25T02:30:00"):
            with self.subTest(local=local), self.assertRaises(ValueError):
                parse_local_time(local, "Europe/Berlin", float('-inf'))

    def test_invalid_or_past_input_is_not_saved(self):
        for change in ({"local_time": "2020-01-01T07:00:00"}, {"local_time": "2030-01-02"},
                       {"local_time": "2030-01-02T07:00:00+03:00"}, {"text": " "},
                       {"text": "x" * 501}, {"zone": "UTC"}, {"session": ""}, {"id": "x"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.store.create({**self.payload, **change})
        self.assertEqual(self.store.active(), [])

    def test_concurrent_workers_claim_only_once(self):
        self.due()
        with ThreadPoolExecutor(8) as pool:
            claims = list(pool.map(lambda _: self.store.claim(), range(8)))
        self.assertEqual(sum(row is not None for row in claims), 1)

    def test_crashed_worker_lease_recovers_and_stale_result_is_ignored(self):
        self.due()
        first = self.store.claim()
        self.assertIsNone(self.store.claim())
        self.clock.return_value += LEASE_SECONDS + 1
        second = ReminderStore(self.path, "Europe/Moscow", 123, self.clock).claim()
        self.assertNotEqual(first["lease"], second["lease"])
        self.store.finish(first)
        self.assertEqual(self.status()["status"], "sending")
        self.store.finish(second)
        self.assertEqual(self.status()["status"], "sent")

    def test_network_retry_survives_restart_and_respects_backoff(self):
        self.due()
        telegram = Mock()
        telegram.text.side_effect = TelegramError(0, "нет связи")
        with self.assertLogs("assistant.reminders", level="WARNING"):
            deliver_one(self.store, telegram)
        state = self.status()
        self.assertEqual(state["status"], "pending")
        self.assertGreater(state["next_attempt"], self.clock())
        restarted = ReminderStore(self.path, "Europe/Moscow", 123, self.clock)
        self.assertFalse(deliver_one(restarted, telegram))
        self.clock.return_value = state["next_attempt"] + 1
        telegram.text.side_effect = None
        self.assertTrue(deliver_one(restarted, telegram))
        self.assertEqual(self.status()["status"], "sent")

    def test_permanent_telegram_failure_stays_visible(self):
        self.due()
        telegram = Mock()
        telegram.text.side_effect = TelegramError(403, "бот заблокирован")
        with self.assertLogs("assistant.reminders", level="WARNING"):
            deliver_one(self.store, telegram)
        self.assertEqual(self.store.active()[0]["status"], "failed")
        self.assertIsNone(self.store.claim())
        self.clock.return_value += 3601
        self.assertIsNotNone(self.store.claim())

    def test_cancel_prevents_delivery_but_cannot_revoke_inflight_send(self):
        self.due()
        self.store.cancel(self.payload["id"])
        self.assertIsNone(self.store.claim())
        self.assertEqual(self.status()["status"], "cancelled")
        with self.assertRaises(ValueError):
            self.store.cancel(self.payload["id"])
        self.payload["id"] = uuid.uuid4().hex
        self.clock.return_value = NOW
        self.due()
        self.store.claim()
        with self.assertRaises(ValueError):
            self.store.cancel(self.payload["id"])

    def test_changing_owner_does_not_redirect_old_reminders(self):
        self.due()
        other = ReminderStore(self.path, "Europe/Moscow", 999, self.clock)
        self.assertIsNone(other.claim())


class HTTPAndDialogueTests(StoreFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.store, "secret"))
        self.server.daemon_threads = True
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        def close():
            self.server.shutdown()
            self.server.server_close()
            thread.join()
        self.addCleanup(close)
        self.settings = RemindersConfig(url=f"http://127.0.0.1:{self.server.server_port}", token="secret")
        self.flow = ReminderDialogue(self.settings)
        self.extract = Mock(return_value={"text": "поставить тесто", "local_time": LOCAL_TIME, "question": ""})

    def handle(self, question, flow=None):
        with patch("assistant.reminder_dialogue.time.time", self.clock):
            return (flow or self.flow).handle(question, "satellite:test", self.extract)

    def test_http_auth_applies_to_reads_and_writes(self):
        client = ReminderClient(replace(self.settings, token="wrong"))
        for path, body in (("/health", None), ("/reminders", None), ("/reminders", self.payload), ("/cancel", {"id": self.payload["id"]})):
            with self.subTest(path=path), self.assertRaises(ReminderError):
                client.request(path, body)
        self.assertEqual(self.store.active(), [])

    def test_preview_does_not_save_confirm_saves_and_no_model_can_bypass_it(self):
        preview = self.handle("Напомни завтра в семь утра поставить тесто")
        self.assertIn("02.01.2030 в 07:00:00", preview)
        self.assertIn("Europe/Moscow, UTC+03:00", preview)
        self.assertIn("Пока не сохранено", preview)
        self.assertEqual(self.store.active(), [])
        result = self.handle("Да.")
        self.assertIn("сохранено", result)
        self.assertEqual(len(self.store.active()), 1)
        self.extract.assert_called_once()
        self.assertIsNone(self.handle("да"))

    def test_ambiguous_hour_clarifies_then_previews(self):
        self.extract.return_value = {"text": "поставить тесто", "local_time": "", "question": "В семь утра или вечера?"}
        self.assertIn("утра или вечера", self.handle("Напомни завтра в семь поставить тесто"))
        self.assertEqual(self.store.active(), [])
        self.extract.return_value = {"text": "поставить тесто", "local_time": LOCAL_TIME, "question": ""}
        self.assertIn("07:00:00", self.handle("Утра"))
        messages = self.extract.call_args.args[0]
        self.assertIn("Europe/Moscow", messages[0]["content"])
        self.assertIn("2030-01-01", messages[0]["content"])
        self.assertIn("Напомни завтра", messages[1]["content"])
        self.assertEqual(self.store.active(), [])

    def test_cancel_and_unrelated_reply_invalidate_preview(self):
        for reply in ("отмена", "Какая погода?"):
            self.handle("Напомни завтра утром поставить тесто")
            self.handle(reply)
            self.assertIsNone(self.handle("да"))
            self.assertEqual(self.store.active(), [])

    def test_confirm_cannot_cross_sessions_or_restart(self):
        self.handle("Напомни завтра утром поставить тесто")
        self.assertIsNone(self.handle("да", ReminderDialogue(self.settings)))
        self.assertEqual(self.store.active(), [])

    def test_expired_preview_cannot_be_confirmed(self):
        self.handle("Напомни завтра утром поставить тесто")
        self.flow.expires = 1
        self.assertIn("устарело", self.handle("подтверждаю"))
        self.assertEqual(self.store.active(), [])

    def test_lost_create_response_can_be_retried_without_duplicate(self):
        self.handle("Напомни завтра утром поставить тесто")
        request = self.flow.client.request
        def lose_response(path, body=None):
            request(path, body)
            raise ReminderError("Не удалось проверить сохранение. Повтори подтверждение.")
        with patch.object(self.flow.client, "request", side_effect=lose_response):
            self.assertIn("проверить", self.handle("подтверждаю"))
        self.assertEqual(len(self.store.active()), 1)
        self.assertIn("сохранено", self.handle("подтверждаю"))
        self.assertEqual(len(self.store.active()), 1)

    def test_past_date_rejected_without_saving(self):
        self.extract.return_value["local_time"] = "2020-01-01T07:00:00"
        self.assertIn("прошло", self.handle("Напомни поставить тесто"))
        self.assertIsNone(self.handle("да"))
        self.assertEqual(self.store.active(), [])

    def test_list_and_explicit_cancel_use_stored_id(self):
        self.handle("Напомни завтра утром поставить тесто")
        self.handle("подтверждаю")
        row = self.store.active()[0]
        self.assertIn(row["id"], self.handle("Мои напоминания"))
        self.assertIn("отменено", self.handle(f"Отмени напоминание {row['id']}"))
        self.assertEqual(self.store.active(), [])

    def test_missing_service_never_claims_save(self):
        flow = ReminderDialogue(RemindersConfig())
        self.assertIn("Ничего не сохранено", self.handle("Напомни завтра", flow))
        self.extract.assert_not_called()

    def test_missing_structured_call_cannot_hallucinate_saved(self):
        self.extract.return_value = None
        self.assertIn("Не удалось разобрать", self.handle("Напомни завтра"))
        self.assertEqual(self.store.active(), [])

    def test_http_ignores_arbitrary_recipient(self):
        result = ReminderClient(self.settings).request("/reminders", {**self.payload, "chat_id": 999})
        self.assertEqual(result["chat_id"], 123)

    def test_http_rejects_oversized_or_non_object_request(self):
        client = ReminderClient(self.settings)
        for data in ([], {**self.payload, "text": "x" * 20000}):
            with self.subTest(data_type=type(data)), self.assertRaises(ReminderError):
                client.request("/reminders", data)
        self.assertEqual(self.store.active(), [])

    def test_cancelled_preview_cannot_leave_a_hidden_confirmation(self):
        cfg = load_config()
        cfg = replace(cfg, reminders=self.settings, log_dir=Path(self.tmp.name), dialogue_log=False,
                      search=replace(cfg.search, enabled=False))
        assistant = Assistant(cfg)
        call = NS(function=NS(name="propose_reminder", arguments=json.dumps(self.extract.return_value)))
        cancelled = threading.Event()
        with patch.object(assistant, "_step", return_value=("", [call], None)):
            with self.assertRaises(AnswerCancelled):
                assistant.ask("Напомни завтра утром поставить тесто", cancel_event=cancelled,
                              on_sentence=lambda text: cancelled.set())
        self.assertIsNone(assistant.reminders.pending)
        self.assertEqual(self.store.active(), [])

    def test_assistant_streams_preview_and_confirm_and_forget_clears_draft(self):
        cfg = load_config()
        cfg = replace(cfg, reminders=self.settings, log_dir=Path(self.tmp.name), dialogue_log=False,
                      search=replace(cfg.search, enabled=False))
        assistant = Assistant(cfg).new_session("satellite:test")
        assistant.long_form = True
        call = NS(function=NS(name="propose_reminder", arguments=json.dumps(self.extract.return_value)))
        sentences, steps = [], []
        with patch.object(assistant, "_step", return_value=("Сохранено!", [call], None)) as api:
            with patch("assistant.reminder_dialogue.time.time", self.clock):
                answer = assistant.ask("Напомни завтра в семь утра поставить тесто",
                                       on_sentence=sentences.append, on_step=steps.append)
                self.assertEqual(sentences, [answer.text])
                self.assertEqual(steps, [False])
                self.assertFalse(assistant.will_be_long_form("да"))
                self.assertEqual(self.store.active(), [])
                other = assistant.new_session("browser:test")
                self.assertIsNone(other.reminders.pending)
                assistant.forget()
                self.assertIsNone(assistant.reminders.pending)
                assistant.ask("Напомни завтра в семь утра поставить тесто")
                saved = assistant.ask("подтверждаю")
                self.assertIn("сохранено", saved.text)
                self.assertEqual(len(self.store.active()), 1)
                self.assertEqual(api.call_count, 2)


class LightweightTests(unittest.TestCase):
    def test_service_imports_without_model_audio_and_opus_packages(self):
        script = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'numpy','openai','torch','av','sounddevice','vosk','faster_whisper'}:
            raise ImportError(fullname)
sys.meta_path.insert(0, Block())
import assistant.reminders, assistant.telegram_bot
'''
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
