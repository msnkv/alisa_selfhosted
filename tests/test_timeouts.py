"""Бюджет времени агента: медленная модель не должна держать ответ минутами."""

import threading
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import httpx

import test_sessions
from assistant.llm import NO_MODEL, TOOL_SKIPPED, Assistant, StepTimeout


def chunk(text="", calls=None):
    return NS(choices=[NS(delta=NS(content=text, tool_calls=calls or []))], usage=None)


def search_call(index=0):
    return NS(index=index, id=f"call-{index}", function=NS(name="web_search", arguments="{}"))


class HangingStream:
    """Поток, который молчит, пока его не закроют, — как DeepSeek под нагрузкой."""

    def __init__(self):
        self.closed = threading.Event()

    def __iter__(self):
        yield chunk("Обрывок. ")
        self.closed.wait(5)
        raise httpx.ReadError("stream closed")

    def close(self):
        self.closed.set()


class TimeoutTests(unittest.TestCase):
    def setUp(self):
        test_sessions.SessionTests.setUp(self)
        self.patch_step.stop()
        self.tools = NS(SCHEMAS=[{"type": "function", "function": {"name": "web_search"}}],
                        call=Mock(return_value="найдено"))
        self.assistant.tools = self.tools

    def test_tool_step_timeout_goes_to_final_answer(self):
        sequence = [httpx.ReadTimeout("slow"), iter([chunk("Ответ по найденному.")])]
        steps = []
        with patch.object(Assistant, "_call_api", side_effect=sequence) as api:
            answer = self.assistant.ask("Какая погода?", on_sentence=lambda s: None, on_step=steps.append)
        self.assertEqual(answer.text, "Ответ по найденному.")
        self.assertIsNone(api.call_args_list[1].args[1], "на последнем шаге инструментов нет")
        self.assertEqual(steps, [True, False])

    def test_final_step_timeout_says_so(self):
        self.assistant.tools = None
        with patch.object(Assistant, "_call_api", side_effect=httpx.ReadTimeout("slow")):
            answer = self.assistant.ask("Привет", on_sentence=lambda s: None, on_step=lambda used: None)
        self.assertEqual(answer.text, NO_MODEL)
        self.assertEqual(self.assistant.history, [])

    def test_call_timeout_never_exceeds_budget(self):
        self.assistant.cfg = replace(self.cfg, agent=replace(self.cfg.agent, total_timeout_s=5))
        with patch.object(Assistant, "_call_api", return_value=iter([chunk("Да.")])) as api:
            self.assistant.ask("Привет", on_sentence=lambda s: None, on_step=lambda used: None)
        self.assertLessEqual(api.call_args.kwargs["timeout"], 5)

    def test_hanging_stream_is_closed_by_deadline(self):
        stream = HangingStream()
        started = time.perf_counter()
        with self.assertRaises(StepTimeout):
            Assistant._read_stream(stream, lambda s: None, timeout=0.2)
        self.assertLess(time.perf_counter() - started, 2)

    def test_tools_are_skipped_after_deadline(self):
        self.assistant.cfg = replace(self.cfg, agent=replace(self.cfg.agent, total_timeout_s=0.2))
        self.tools.call.side_effect = lambda *args: (time.sleep(0.3), "найдено")[1]
        sequence = [iter([chunk("", [search_call(0), search_call(1)])]), iter([chunk("Ответ.")])]
        with patch.object(Assistant, "_call_api", side_effect=sequence) as api:
            answer = self.assistant.ask("Какая погода?", on_sentence=lambda s: None, on_step=lambda used: None)
        self.assertEqual(answer.text, "Ответ.")
        self.tools.call.assert_called_once()
        messages = api.call_args_list[1].args[0]
        self.assertIn(TOOL_SKIPPED, [m.get("content") for m in messages])


if __name__ == "__main__":
    unittest.main()
