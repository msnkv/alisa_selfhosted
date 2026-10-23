"""Потоковый HTTP и остановка ответа без настоящих моделей и динамика."""

import json
import threading
import urllib.error
import urllib.request
import numpy as np
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import unittest

import test_sessions
from assistant.client import Satellite
from assistant.llm import Assistant
from assistant.streaming import AnswerCancelled, Requests


def chunk(text="", calls=None):
    return NS(choices=[NS(delta=NS(content=text, tool_calls=calls or []))], usage=None)


class StreamingTests(unittest.TestCase):
    def setUp(self):
        test_sessions.SessionTests.setUp(self)
        self.patch_step.stop()
        self.block = False
        self.release = threading.Event()
        self.first_generated = threading.Event()
        self.closed = threading.Event()
        self.responses = []

        def api(assistant, messages, tools, *, stream, max_tokens):
            self.assertTrue(stream)
            def generate():
                try:
                    yield chunk("Первая фраза. ")
                    self.first_generated.set()
                    while self.block and not self.release.wait(0.02):
                        yield chunk()
                    yield chunk("Вторая фраза.")
                finally:
                    self.closed.set()
            return generate()

        self.api = patch.object(Assistant, "_call_api", autospec=True, side_effect=api)
        self.api.start()
        self.addCleanup(self.api.stop)

    def tearDown(self):
        self.release.set()
        for response in self.responses:
            response.close()
        test_sessions.SessionTests.tearDown(self)

    def open(self, text="Расскажи сказку", session="satellite:test", request="test-request", audio=None):
        response = urllib.request.urlopen(urllib.request.Request(
            self.url + "/ask-stream",
            data=json.dumps({"text": text}).encode() if audio is None else audio,
            headers={"Content-Type": "application/json" if audio is None else "audio/wav",
                     "X-Token": "test-token", "X-Session-ID": session, "X-Request-ID": request},
        ), timeout=3)
        self.responses.append(response)
        return response

    def cancel(self, session="satellite:test", request="test-request", token="test-token"):
        with urllib.request.urlopen(urllib.request.Request(
            self.url + "/cancel", data=b"{}",
            headers={"X-Token": token, "X-Session-ID": session, "X-Request-ID": request},
        ), timeout=2) as response:
            return response.status

    def until_sentence(self, response):
        while True:
            raw = response.readline()
            self.assertTrue(raw, "поток завершился без звука")
            event = json.loads(raw)
            if event["type"] == "sentence":
                return event

    def rest(self, response):
        return [json.loads(raw) for raw in response]

    def test_first_audio_arrives_before_generation_finishes(self):
        self.block = True
        response = self.open()
        first = self.until_sentence(response)
        self.assertEqual(first["text"], "Первая фраза.")
        self.assertTrue(first["audio"])
        self.assertFalse(self.closed.is_set())
        self.release.set()
        rest = self.rest(response)
        self.assertEqual([e["type"] for e in rest], ["sentence", "done"])
        self.assertEqual(rest[-1]["text"], "Первая фраза. Вторая фраза.")

    def test_ordinary_answer_waits_for_final_step(self):
        self.block = True
        response = self.open(text="Где инструмент?")
        self.assertTrue(self.first_generated.wait(1))
        self.assertFalse(self.tts.synthesize.called)
        self.release.set()
        events = self.rest(response)
        self.assertEqual(len([e for e in events if e["type"] == "sentence"]), 2)
        self.assertEqual(events[-1]["type"], "done")

    def test_tool_step_text_is_discarded(self):
        tools = NS(SCHEMAS=[{"type": "function", "function": {"name": "web_search"}}], call=Mock(return_value="найдено"))
        self.assistant.tools = tools
        calls = [NS(index=0, id="call-1", function=NS(name="web_search", arguments="{}"))]
        sequence = [iter([chunk("Не озвучивать. ", calls)]), iter([chunk("Финальный ответ.")])]
        with patch.object(Assistant, "_call_api", side_effect=sequence):
            events = self.rest(self.open(text="Какая погода?"))
        self.assertEqual([e["text"] for e in events if e["type"] == "sentence"], ["Финальный ответ."])
        tools.call.assert_called_once()

    def test_cancel_closes_model_stream_and_does_not_save_partial_answer(self):
        self.block = True
        response = self.open()
        self.until_sentence(response)
        self.assertEqual(self.cancel(), 200)
        self.assertEqual(self.rest(response), [])
        self.assertTrue(self.closed.wait(1))
        assistant = self.brain._sessions["satellite:test"][0]
        self.assertEqual(assistant.history, [])
        self.assertFalse(assistant.long_form)

    def test_other_session_cannot_cancel_request(self):
        self.block = True
        response = self.open()
        self.until_sentence(response)
        self.cancel(session="browser:other")
        event = self.brain.requests.active[("satellite:test", "test-request")]
        self.assertFalse(event.is_set())
        self.release.set()
        self.assertEqual(self.rest(response)[-1]["type"], "done")

    def test_cancel_releases_model_when_client_stops_reading_audio(self):
        synthesized = threading.Event()
        samples = np.zeros(48000 * 20, dtype=np.float32)
        self.tts.synthesize.side_effect = lambda text: (synthesized.set(), samples)[1]

        def generate():
            try:
                for _ in range(100):
                    yield chunk("Длинная фраза. ")
            finally:
                self.closed.set()

        with patch.object(Assistant, "_call_api", return_value=generate()):
            # Читаем только HTTP-заголовки. Буфер TCP заполнится аудио.
            response = self.open()
            self.assertTrue(synthesized.wait(1))
            self.cancel()
            self.assertTrue(self.closed.wait(1), "отмена застряла в записи в сокет")
            response.close()
        self.assertTrue(self.brain._lock.acquire(timeout=1))
        self.brain._lock.release()

    def test_cancel_while_waiting_for_models_and_before_request_arrives(self):
        self.brain._lock.acquire()
        try:
            response = self.open(audio=b"voice")
            self.assertEqual(json.loads(response.readline())["type"], "start")
            self.cancel()
            self.assertEqual(self.rest(response), [])
            self.stt.transcribe.assert_not_called()
        finally:
            self.brain._lock.release()
        self.cancel(request="early")
        self.assertEqual(self.rest(self.open(request="early")), [])

    def test_cancel_requires_authentication(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.cancel(token="wrong")
        self.assertEqual(caught.exception.code, 403)

    def test_duplicate_request_id_is_rejected(self):
        self.block = True
        response = self.open()
        self.until_sentence(response)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.open()
        self.assertEqual(caught.exception.code, 400)
        self.cancel()
        self.rest(response)

    def test_voice_uses_same_session_and_forget_still_speaks(self):
        self.stt.transcribe.return_value.text = "Расскажи сказку"
        events = self.rest(self.open(audio=b"voice"))
        self.assertEqual(events[1], {"type": "heard", "text": "Расскажи сказку"})
        self.assertEqual(self.brain._sessions["satellite:test"][0].history[0]["content"], "Расскажи сказку")
        self.stt.transcribe.return_value.text = "Забудь"
        events = self.rest(self.open(request="forget", audio=b"voice"))
        self.assertTrue(next(e for e in events if e["type"] == "sentence")["audio"])
        self.assertEqual(self.brain._sessions["satellite:test"][0].history, [])

    def test_streaming_tts_error_is_reported_and_unlocks_models(self):
        self.tts.synthesize.side_effect = RuntimeError("model failed")
        with self.assertLogs("assistant.server", level="ERROR"):
            events = self.rest(self.open())
        self.assertEqual(events[-1]["type"], "error")
        self.tts.synthesize.side_effect = None
        events = self.rest(self.open(request="retry"))
        self.assertEqual(events[-1]["type"], "done")

    def test_empty_model_response_speaks_fallback(self):
        with patch.object(Assistant, "_call_api", return_value=iter([])):
            events = self.rest(self.open())
        sentences = [e for e in events if e["type"] == "sentence"]
        self.assertEqual(len(sentences), 1)
        self.assertTrue(sentences[0]["audio"])
        self.assertEqual(sentences[0]["text"], events[-1]["text"])

    def test_satellite_plays_first_chunk_before_model_finishes(self):
        self.block = True
        self.stt.transcribe.return_value.text = "Расскажи сказку"
        satellite = Satellite(self.cfg, self.url, microphone=None)
        satellite.cfg = NS(name=self.cfg.name, server=NS(token="test-token", timeout_s=3))
        def play(samples, rate):
            if satellite.play.call_count == 1:
                self.assertFalse(self.closed.is_set())
            self.release.set()
        satellite.play = Mock(side_effect=play)
        satellite.stop_playback = Mock()
        result = satellite._stream_with_filler(b"voice")
        self.assertEqual(result["type"], "done")
        self.assertEqual(satellite.play.call_count, 2)

    def test_satellite_ctrl_c_stops_response_but_not_client(self):
        self.block = True
        self.stt.transcribe.return_value.text = "Расскажи сказку"
        satellite = Satellite(self.cfg, self.url, microphone=None)
        satellite.cfg = NS(name=self.cfg.name, server=NS(token="test-token", timeout_s=3))
        satellite.play = Mock(side_effect=KeyboardInterrupt)
        satellite.stop_playback = Mock()
        self.assertEqual(satellite._stream_with_filler(b"voice"), {"cancelled": True})
        self.assertTrue(self.closed.wait(1))
        satellite.stop_playback.assert_called_once()

    def test_satellite_button_event_stops_current_playback(self):
        self.block = True
        self.stt.transcribe.return_value.text = "Расскажи сказку"
        satellite = Satellite(self.cfg, self.url, microphone=None)
        satellite.cfg = NS(name=self.cfg.name, server=NS(token="test-token", timeout_s=3))
        stopped = threading.Event()
        playing = threading.Event()
        playback_ended = threading.Event()
        satellite.play = Mock(side_effect=lambda *args: (playing.set(), playback_ended.wait(2)))
        satellite.stop_playback = Mock(side_effect=playback_ended.set)
        result = []
        worker = threading.Thread(target=lambda: result.append(satellite._stream_with_filler(b"voice", stopped)))
        worker.start()
        self.assertTrue(playing.wait(2))
        stopped.set()
        worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result, [{"cancelled": True}])
        satellite.stop_playback.assert_called_once()

    def test_early_cancel_bookkeeping_is_bounded(self):
        requests = Requests()
        for i in range(200):
            requests.cancel("s", str(i))
        self.assertEqual(len(requests.early), 128)
        self.assertTrue(requests.start("s", "199").is_set())
        self.assertFalse(requests.start("other", "199").is_set())
