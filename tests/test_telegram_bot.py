"""Проверка клиента через настоящий HTTP, без Telegram-токена и моделей.

python -m unittest discover -s tests -p test_telegram_bot.py -v
"""

import base64
import io
import json
import math
import struct
import threading
import unittest
import wave
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch

import av

from assistant.config import load_config
from assistant.telegram_bot import (
    MAX_INPUT_BYTES, AliceServer, Telegram, TelegramError, handle_message, wav_to_voice,
)


def make_wav(rate=48000, channels=1):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"".join(
            struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate))) * channels
            for i in range(rate // 5)
        ))
    return buffer.getvalue()


class BotTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.questions = []
        self.wav = make_wav()
        self.ogg = wav_to_voice(self.wav)
        self.result = {"text": "Ответ Алисы.", "audio": base64.b64encode(self.wav).decode()}
        self.server_status = 200
        self.downloads = 0
        test = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, code, body, kind="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                test.downloads += 1
                self.reply(200, test.ogg, "audio/ogg")

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                kind = self.headers["Content-Type"]
                if self.path == "/ask":
                    test.questions.append((
                        body, kind, self.headers.get("X-Token"), self.headers.get("X-Session-ID"),
                    ))
                    self.reply(test.server_status, json.dumps(test.result).encode())
                    return
                method = self.path.rsplit("/", 1)[-1]
                if kind.startswith("multipart/"):
                    mail = BytesParser(policy=default).parsebytes(
                        f"Content-Type: {kind}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body
                    )
                    params = {
                        part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
                        for part in mail.iter_parts()
                    }
                else:
                    params = json.loads(body)
                test.calls.append((method, params))
                if method == "getFile":
                    result = {"file_path": "voice/input.oga"}
                elif method == "badToken":
                    self.reply(401, b'{"ok":false,"error_code":401,"description":"secret-token"}')
                    return
                else:
                    result = True
                self.reply(200, json.dumps({"ok": True, "result": result}).encode())

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        host = f"127.0.0.1:{self.httpd.server_port}"
        self.telegram = Telegram("secret-token")
        self.telegram.api_url = f"http://{host}/bot/"
        self.telegram.file_url = f"http://{host}/file/"
        self.alice = AliceServer(SimpleNamespace(url=host, token="alice-secret", timeout_s=2))

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join()

    def message(self, **extra):
        return {"chat": {"id": 123, "type": "private"}, "from": {"id": 123}, **extra}

    def handle(self, message):
        handle_message(message, self.telegram, self.alice, {123})

    def assert_voice(self):
        method, params = self.calls[-1]
        self.assertEqual(method, "sendVoice")
        self.assertEqual(params["chat_id"], b"123")
        self.assertTrue(params["voice"].startswith(b"OggS"))
        with av.open(io.BytesIO(params["voice"])) as audio:
            self.assertEqual(audio.streams.audio[0].codec_context.name, "opus")
            self.assertGreater(sum(frame.samples for frame in audio.decode(audio=0)), 8000)

    def test_text_roundtrip_sends_json_text_and_playable_voice(self):
        self.handle(self.message(text="Где болгарка?"))
        body, kind, token, session_id = self.questions[0]
        self.assertEqual(json.loads(body), {"text": "Где болгарка?"})
        self.assertEqual((kind, token), ("application/json", "alice-secret"))
        self.assertEqual(session_id, "telegram:123")
        self.assertEqual(self.calls[0], ("sendMessage", {"chat_id": 123, "text": "Ответ Алисы."}))
        self.assert_voice()

    def test_voice_roundtrip_forwards_original_ogg(self):
        self.handle(self.message(voice={"file_id": "incoming-voice", "file_size": len(self.ogg)}))
        self.assertEqual(self.questions[0], (self.ogg, "audio/ogg", "alice-secret", "telegram:123"))
        self.assertEqual(self.calls[0], ("getFile", {"file_id": "incoming-voice"}))
        self.assert_voice()

    def test_strangers_and_groups_cannot_reach_server(self):
        self.handle({**self.message(text="Секреты?"), "from": {"id": 456}})
        self.handle({**self.message(text="Секреты?"), "chat": {"id": -1, "type": "group"}})
        self.assertEqual(self.questions, [])
        self.assertEqual(self.calls, [])

    def test_start_without_access_only_returns_id(self):
        handle_message(self.message(text="/start"), self.telegram, self.alice, set())
        self.assertIn("123", self.calls[0][1]["text"])
        self.assertEqual(self.questions, [])

    def test_server_error_has_explanation_without_fake_answer(self):
        self.server_status = 503
        self.handle(self.message(text="Привет"))
        self.assertIn("недоступен", self.calls[0][1]["text"])
        self.assertEqual(len(self.calls), 1)

    def test_text_survives_missing_and_corrupt_audio(self):
        for audio in (None, "broken-base64"):
            with self.subTest(audio=audio):
                self.calls.clear()
                self.result = {"text": "Ответ", "audio": audio}
                self.handle(self.message(text="Вопрос"))
                self.assertEqual(self.calls[0][1]["text"], "Ответ")
                self.assertEqual([name for name, _ in self.calls], ["sendMessage", "sendMessage"])

    def test_oversized_voice_is_rejected_before_download(self):
        self.handle(self.message(voice={"file_id": "large", "file_size": MAX_INPUT_BYTES + 1}))
        self.assertEqual(self.downloads, 0)
        self.assertEqual(self.questions, [])
        self.assertIn("покороче", self.calls[0][1]["text"])

    def test_long_answer_is_split_without_losing_text(self):
        text = "Длинная сказка. " * 500
        self.result["text"] = text
        self.handle(self.message(text="Расскажи сказку"))
        parts = [params["text"] for method, params in self.calls if method == "sendMessage"]
        self.assertEqual("".join(parts), text)
        self.assertTrue(all(len(part) <= 2000 for part in parts))
        self.assert_voice()

    def test_telegram_http_error_does_not_expose_token(self):
        with self.assertRaises(TelegramError) as caught:
            self.telegram.call("badToken", {})
        self.assertEqual(caught.exception.code, 401)
        self.assertNotIn("secret-token", str(caught.exception))

    def test_rate_limit_retries_rejected_send_only(self):
        with patch.object(self.telegram, "_call", side_effect=[
            TelegramError(429, "Too many requests", 2), True,
        ]) as call, patch("assistant.telegram_bot.time.sleep") as sleep:
            self.assertTrue(self.telegram.call("sendMessage", {"text": "Ответ"}))
            self.assertEqual(call.call_count, 2)
            sleep.assert_called_once_with(2)

    def test_polling_advances_offset_and_uses_only_bot_dependencies(self):
        from assistant.telegram_bot import main

        responses = [
            {"username": "alice_test"}, {},
            [{"update_id": 42, "message": self.message(text="Вопрос")}],
            KeyboardInterrupt,
        ]
        with patch("assistant.telegram_bot.Telegram", return_value=self.telegram), patch.object(
            self.telegram, "call", side_effect=responses
        ) as call, patch("assistant.telegram_bot.handle_message") as handle, patch.dict(
            "os.environ", {"TELEGRAM_BOT_TOKEN": "secret-token", "TELEGRAM_ALLOWED_USER_ID": "123"}
        ):
            self.assertEqual(main(), 0)
            handle.assert_called_once()
            self.assertEqual(call.call_args_list[-1].args[1]["offset"], 43)

    def test_resampling_stereo_22050_to_mono_opus(self):
        blob = wav_to_voice(make_wav(rate=22050, channels=2))
        with av.open(io.BytesIO(blob)) as audio:
            self.assertEqual(audio.streams.audio[0].codec_context.name, "opus")
            self.assertEqual(audio.streams.audio[0].codec_context.sample_rate, 48000)
            self.assertEqual(audio.streams.audio[0].codec_context.channels, 1)
            self.assertGreater(sum(frame.samples for frame in audio.decode(audio=0)), 8000)

    def test_local_config_server_url_is_used(self):
        import tempfile
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp, patch("assistant.config.load_dotenv"):
            path = Path(tmp) / "config.yaml"
            path.write_text((root / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
            path.with_name("config.local.yaml").write_text(
                'server:\n  url: "192.168.31.35:8765"\n  token: "local-token"\n', encoding="utf-8"
            )
            server = AliceServer(load_config(path).server)
        self.assertEqual(server.url, "http://192.168.31.35:8765")
        self.assertEqual(server.token, "local-token")


if __name__ == "__main__":
    unittest.main()
