"""Изоляция разговоров через настоящий сервер, без моделей и API-ключей.

python -m unittest discover -s tests -p 'test_*.py' -v
"""

import base64
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from assistant.client import Satellite
from assistant.config import load_config
from assistant.history import read_all
from assistant.llm import Assistant
from assistant.server import Brain, LEGACY_SESSION, make_handler


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        data = root / "data"
        data.mkdir()
        (data / "garage.md").write_text("Тестовая полка.", encoding="utf-8")
        cfg = load_config()
        self.cfg = replace(
            cfg, search=replace(cfg.search, enabled=False), data_dir=data,
            log_dir=root, dialogue_log=True,
        )
        self.assistant = Assistant(self.cfg)
        # Не обращаемся к сети. Проверяем реальные ask, историю, режимы и
        # журнал, подменяя только ответ API, а не логику работы с памятью.
        self.prompts = []

        def step(assistant, messages, tools, on_sentence, max_tokens):
            self.prompts.append((assistant.session_id, messages, tools))
            return "Ответ принят.", [], None

        self.patch_step = patch.object(Assistant, "_step", autospec=True, side_effect=step)
        self.patch_step.start()
        self.addCleanup(self.patch_step.stop)
        self.stt = SimpleNamespace(transcribe=Mock(return_value=SimpleNamespace(text="Вопрос голосом")))
        self.tts = SimpleNamespace(sample_rate=48000, synthesize=Mock(return_value=np.zeros(100, dtype=np.float32)))
        self.brain = Brain(self.cfg, self.assistant, self.stt, self.tts)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.brain, "test-token"))
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join()
        self.tmp.cleanup()

    def post(self, session=None, *, text="Вопрос", path="/ask", audio=None, token="test-token"):
        headers = {"X-Token": token, "Content-Type": "audio/ogg" if audio is not None else "application/json"}
        if session is not None:
            headers["X-Session-ID"] = session
        request = urllib.request.Request(
            self.url + path, headers=headers,
            data=audio if audio is not None else json.dumps({"text": text}).encode(),
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            return exc.code, json.load(exc)

    def session(self, key):
        return self.brain._sessions[key][0]

    def questions(self, key):
        return [m["content"] for m in self.session(key).history if m["role"] == "user"]

    def test_interleaved_channels_keep_their_own_context(self):
        for key in ("telegram:123", "browser:abc", "satellite:kitchen"):
            self.assertEqual(self.post(key, text=f"Начало {key}")[0], 200)
        for key in ("browser:abc", "satellite:kitchen", "telegram:123"):
            self.post(key, text=f"Уточнение {key}")
        for key, messages, _ in self.prompts:
            questions = [m["content"] for m in messages if m["role"] == "user"]
            self.assertTrue(all(key in question for question in questions))
        self.assertEqual(self.assistant.history, [])

    def test_story_mode_and_forget_do_not_affect_other_channels(self):
        self.post("telegram:123", text="Расскажи сказку")
        self.post("browser:abc", text="Где инструмент?")
        self.assertTrue(self.session("telegram:123").long_form)
        self.assertFalse(self.session("browser:abc").long_form)
        self.post("browser:abc", text="Забудь")
        self.assertTrue(self.session("telegram:123").long_form)
        self.assertEqual(self.questions("telegram:123"), ["Расскажи сказку"])
        self.post("telegram:123", text="Забудь")
        self.assertFalse(self.session("telegram:123").long_form)
        self.assertEqual(self.questions("telegram:123"), [])

    def test_explicit_forget_endpoint_only_clears_selected_session(self):
        self.post("telegram:123", text="Телеграм")
        self.post("satellite:kitchen", text="Кухня")
        before = self.session("telegram:123").dialogue.conversation
        status, result = self.post("satellite:kitchen", path="/forget")
        self.assertEqual((status, result), (200, {"ok": True}))
        self.assertEqual(self.questions("satellite:kitchen"), [])
        self.assertEqual(self.questions("telegram:123"), ["Телеграм"])
        self.assertEqual(self.session("telegram:123").dialogue.conversation, before)

    def test_story_in_telegram_does_not_disable_browser_tools(self):
        schemas = [{"type": "function", "function": {"name": "web_search"}}]
        self.assistant.tools = SimpleNamespace(SCHEMAS=schemas)
        self.post("telegram:123", text="Расскажи сказку")
        self.post("browser:abc", text="Какая погода?")
        self.assertIsNone(self.prompts[0][2])
        self.assertEqual(self.prompts[1][2], schemas)

    def test_voice_text_and_voice_forget_use_same_session(self):
        key = "telegram:123"
        self.post(key, text="Текстом")
        status, result = self.post(key, audio=b"test ogg")
        self.assertEqual(status, 200)
        self.assertEqual(self.questions(key), ["Текстом", "Вопрос голосом"])
        self.assertTrue(base64.b64decode(result["audio"]).startswith(b"RIFF"))
        self.post("browser:abc", text="Браузер")
        self.stt.transcribe.return_value.text = "Забудь"
        status, result = self.post(key, audio=b"test ogg")
        self.assertEqual(status, 200)
        self.assertTrue(base64.b64decode(result["audio"]).startswith(b"RIFF"))
        self.assertEqual(self.questions(key), [])
        self.assertEqual(self.questions("browser:abc"), ["Браузер"])

    def test_old_client_has_separate_legacy_session(self):
        self.post(text="Старый клиент")
        self.post("browser:abc", text="Новый клиент")
        self.post(text="Уточнение старого")
        self.assertEqual(self.questions(LEGACY_SESSION), ["Старый клиент", "Уточнение старого"])
        self.post(path="/forget")
        self.assertEqual(self.questions(LEGACY_SESSION), [])
        self.assertEqual(self.questions("browser:abc"), ["Новый клиент"])

    def test_session_id_does_not_replace_auth_and_is_validated(self):
        self.assertEqual(self.post("telegram:123", token="wrong")[0], 403)
        for key in ("", "bad/session", "x" * 129):
            self.assertEqual(self.post(key)[0], 400)
        self.assertEqual(len(self.brain._sessions), 0)
        self.assertEqual(self.post("x" * 128)[0], 200)

    def test_journal_turns_and_conversations_are_separate(self):
        self.post("telegram:123", text="Первый")
        self.post("browser:abc", text="Первый")
        self.post("telegram:123", text="Второй")
        self.post("telegram:123", text="Забудь")
        self.post("telegram:123", text="Новый")
        self.post("browser:abc", text="Второй")
        rows = read_all(self.cfg.log_dir / self.cfg.dialogue_file)
        telegram = [row for row in rows if row["session_id"] == "telegram:123"]
        browser = [row for row in rows if row["session_id"] == "browser:abc"]
        self.assertEqual([row["turn"] for row in telegram], [1, 2, 1])
        self.assertEqual([row["turn"] for row in browser], [1, 2])
        self.assertEqual(telegram[0]["conversation"], telegram[1]["conversation"])
        self.assertNotEqual(telegram[0]["conversation"], telegram[2]["conversation"])
        self.assertEqual(browser[0]["conversation"], browser[1]["conversation"])
        self.assertTrue(set(r["conversation"] for r in telegram).isdisjoint(r["conversation"] for r in browser))

    def test_session_does_not_inherit_local_dialogue_or_story_mode(self):
        self.assistant.ask("Расскажи сказку")
        self.post("telegram:123", text="Новый вопрос")
        session = self.session("telegram:123")
        self.assertEqual(self.questions("telegram:123"), ["Новый вопрос"])
        self.assertFalse(session.long_form)
        self.assertTrue(self.assistant.long_form)
        self.assertIs(session.knowledge, self.assistant.knowledge)
        self.assertIs(session.tools, self.assistant.tools)
        self.assistant.forget()
        self.assertEqual(self.assistant.history, [])
        self.assertEqual(self.questions("telegram:123"), ["Новый вопрос"])

    def test_memory_limit_is_applied_per_session(self):
        limit = self.cfg.agent.memory_turns
        for i in range(limit + 2):
            self.post("telegram:123", text=f"Реплика {i}")
        self.post("browser:abc", text="Единственная реплика")
        self.assertEqual(len(self.questions("telegram:123")), limit)
        self.assertEqual(self.questions("browser:abc"), ["Единственная реплика"])

    def test_idle_and_capacity_limits_do_not_mix_context(self):
        with patch("assistant.server.MAX_SESSIONS", 2), patch("assistant.server.SESSION_IDLE_S", 10):
            with patch("assistant.server.time.monotonic", return_value=100):
                self.brain.answer_text("Один", session_id="one")
                self.brain.answer_text("Два", session_id="two")
            with patch("assistant.server.time.monotonic", return_value=105):
                self.brain.answer_text("Уточнение", session_id="one")
                self.brain.answer_text("Три", session_id="three")
            self.assertEqual(list(self.brain._sessions), ["one", "three"])
            with patch("assistant.server.time.monotonic", return_value=116):
                self.brain.answer_text("Заново", session_id="one")
            self.assertEqual(list(self.brain._sessions), ["one"])
            self.assertEqual(self.questions("one"), ["Заново"])

    def test_concurrent_clients_do_not_mix_histories(self):
        def talk(key):
            for i in range(2):
                self.assertEqual(self.post(key, text=f"{key} вопрос {i}")[0], 200)

        keys = ["telegram:123", "browser:abc", "satellite:kitchen"]
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(talk, keys))
        for key in keys:
            self.assertEqual(self.questions(key), [f"{key} вопрос 0", f"{key} вопрос 1"])

    def test_satellite_sends_same_id_for_question_and_reset(self):
        cfg = replace(self.cfg, server=replace(
            self.cfg.server, url=self.url, token="test-token", session_id="satellite:kitchen",
        ))
        satellite = Satellite(cfg, self.url, microphone=None)
        satellite._post("/ask", b'{"text":"Kitchen"}', "application/json")
        self.post("browser:abc", text="Browser")
        self.assertEqual(self.questions("satellite:kitchen"), ["Kitchen"])
        satellite.forget()
        self.assertEqual(self.questions("satellite:kitchen"), [])
        self.assertEqual(self.questions("browser:abc"), ["Browser"])
        self.assertEqual(Satellite(cfg, self.url, microphone=None).session_id, satellite.session_id)

    def test_satellites_get_different_ids_by_default(self):
        cfg = replace(self.cfg, server=replace(self.cfg.server, session_id=""))
        first = Satellite(cfg, self.url, microphone=None)
        second = Satellite(cfg, self.url, microphone=None)
        self.assertTrue(first.session_id.startswith("satellite:"))
        self.assertNotEqual(first.session_id, second.session_id)


if __name__ == "__main__":
    unittest.main()
