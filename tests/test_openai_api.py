"""OpenAI-совместимый вход для Home Assistant: /v1/models и /v1/chat/completions.

Проверяем пакетом openai, тем же, каким ходит большинство клиентов:
если он понимает наш ответ, поймёт и HA. Модели и DeepSeek не нужны —
Brain подменён.
"""

import threading
import unittest
from http.server import ThreadingHTTPServer
from types import SimpleNamespace as NS

from openai import OpenAI

from assistant.server import make_handler, openai_question, openai_session


class FakeBrain:
    def __init__(self):
        self.cfg = NS(name="тест")
        self.phrases = {}
        self.calls = []

    def answer_text(self, question, *, session_id, speak=True, heard=""):
        self.calls.append((question, session_id, speak))
        return {"heard": question, "text": f"ответ на «{question}»", "seconds": 0, "tools": []}


class OpenAiApiTests(unittest.TestCase):
    def serve(self, token=""):
        brain = FakeBrain()
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(brain, token))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        return brain, f"http://127.0.0.1:{httpd.server_port}/v1"

    def test_chat_plain_and_stream(self):
        brain, url = self.serve(token="secret")
        client = OpenAI(base_url=url, api_key="secret", max_retries=0)

        self.assertEqual([m.id for m in client.models.list()], ["alisa"])

        messages = [
            {"role": "system", "content": "ты умный дом"},
            {"role": "user", "content": "привет"},
            {"role": "assistant", "content": "здравствуй"},
            {"role": "user", "content": [{"type": "text", "text": "где болгарка?"}]},
        ]
        reply = client.chat.completions.create(model="alisa", messages=messages, user="kitchen")
        self.assertEqual(reply.choices[0].message.content, "ответ на «где болгарка?»")
        # Только последняя реплика, сессия из user, без синтеза.
        self.assertEqual(brain.calls[-1], ("где болгарка?", "kitchen", False))

        stream = client.chat.completions.create(model="alisa", messages=messages, stream=True)
        text = "".join(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices)
        self.assertEqual(text, "ответ на «где болгарка?»")
        self.assertEqual(brain.calls[-1][1], "ha")

    def test_wrong_key(self):
        _, url = self.serve(token="right")
        client = OpenAI(base_url=url, api_key="wrong", max_retries=0)
        with self.assertRaises(Exception) as caught:
            client.chat.completions.create(model="alisa", messages=[{"role": "user", "content": "x"}])
        self.assertEqual(getattr(caught.exception, "status_code", None), 401)

    def test_helpers(self):
        self.assertEqual(openai_question({"messages": [{"role": "system", "content": "s"}]}), "")
        self.assertEqual(openai_question({}), "")
        self.assertEqual(openai_session({"user": "плохой id!"}, None), "ha")
        self.assertEqual(openai_session({"user": "a"}, "b"), "b")


if __name__ == "__main__":
    unittest.main()
