"""Голосовые заметки: разбор фразы, подтверждение и файл — на временной папке."""

import tempfile
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from assistant.config import load_config
from assistant.llm import Assistant
from assistant.notes import NoteBook, NoteDialogue, note_request
from assistant.prompt import Knowledge


class ParseTests(unittest.TestCase):
    def test_trigger_phrases(self):
        cases = {
            "Запиши в заметки, что ключи от гаража у соседа.": "ключи от гаража у соседа",
            "запиши заметку: купить фильтр для пылесоса": "купить фильтр для пылесоса",
            "Заметка: шланг порвался": "шланг порвался",
            "запомни в заметках что дрель отдал Пете": "дрель отдал Пете",
            "добавь в заметки — краска закончилась": "краска закончилась",
            "запиши заметку": "",
        }
        for phrase, text in cases.items():
            with self.subTest(phrase=phrase):
                self.assertEqual(note_request(phrase), text)

    def test_not_a_note(self):
        for phrase in ("что у меня в заметках?", "где лежит болгарка",
                       "напомни записать в заметки про шланг", "заметками я не пользуюсь"):
            with self.subTest(phrase=phrase):
                self.assertIsNone(note_request(phrase))


class DialogueTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.path = self.dir / "notes.md"
        self.dialogue = NoteDialogue(self.dir)

    def test_preview_does_not_write_confirm_writes(self):
        reply = self.dialogue.handle("Запиши в заметки, что ключи у соседа")
        self.assertIn("ключи у соседа", reply)
        self.assertFalse(self.path.exists())
        self.assertEqual(self.dialogue.handle("подтверждаю"), "Записала.")
        text = self.path.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("# Заметки"))
        self.assertRegex(text, r"\n- \d{4}-\d{2}-\d{2} \S\S \d{2}:\d{2} — ключи у соседа\n$")
        # Повторное «да» второй строки не добавит.
        self.assertIsNone(self.dialogue.handle("да"))
        self.assertEqual(text, self.path.read_text(encoding="utf-8"))

    def test_cancel_and_unrelated_reply_drop_preview(self):
        self.dialogue.handle("заметка: шланг порвался")
        self.assertEqual(self.dialogue.handle("отмена"), "Заметка не записана.")
        self.dialogue.handle("заметка: шланг порвался")
        self.assertIsNone(self.dialogue.handle("где болгарка"))
        self.assertIsNone(self.dialogue.handle("подтверждаю"))
        self.assertFalse(self.path.exists())

    def test_expired_preview_cannot_be_confirmed(self):
        with patch("assistant.notes.time.monotonic", return_value=0):
            self.dialogue.handle("заметка: шланг порвался")
        with patch("assistant.notes.time.monotonic", return_value=601):
            self.assertIn("устарело", self.dialogue.handle("подтверждаю"))
        self.assertFalse(self.path.exists())

    def test_asks_what_to_write(self):
        self.assertEqual(self.dialogue.handle("запиши заметку"), "Что записать?")
        self.assertIn("Краска кончилась", self.dialogue.handle("Краска кончилась."))
        self.dialogue.handle("подтверждаю")
        self.assertIn("— Краска кончилась\n", self.path.read_text(encoding="utf-8"))

    def test_remove_last_touches_only_dictated_lines(self):
        self.assertIn("нечего", self.dialogue.handle("удали последнюю заметку"))
        book = NoteBook(self.dir)
        book.append("первая", datetime(2026, 10, 28, 9, 5))
        book.append("вторая", datetime(2026, 10, 29, 9, 5))
        with self.path.open("a", encoding="utf-8") as file:
            file.write("руками дописано\n")
        self.assertEqual(self.dialogue.handle("Удали последнюю заметку."), "Удалила заметку: вторая.")
        text = self.path.read_text(encoding="utf-8")
        self.assertIn("- 2026-10-28 ср 09:05 — первая\n", text)
        self.assertIn("руками дописано", text)
        self.assertNotIn("вторая", text)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        cfg = load_config()
        self.cfg = replace(cfg, data_dir=self.dir, log_dir=self.dir, dialogue_log=False,
                           search=replace(cfg.search, enabled=False))

    def test_assistant_handles_note_without_model(self):
        assistant = Assistant(self.cfg).new_session("satellite:test")
        sentences, steps = [], []
        with patch.object(assistant, "_step", side_effect=AssertionError("модель не нужна")):
            preview = assistant.ask("Запиши в заметки, что дрель у Пети",
                                    on_sentence=sentences.append, on_step=steps.append)
            self.assertEqual(sentences, [preview.text])
            self.assertEqual(steps, [False])
            self.assertEqual(preview.tools_used, ["note"])
            self.assertFalse(assistant.will_be_long_form("подтверждаю"))
            other = assistant.new_session("browser:test")
            self.assertIsNone(other.notes.pending)
            assistant.forget()
            self.assertIsNone(assistant.notes.pending)
            assistant.ask("Запиши в заметки, что дрель у Пети")
            self.assertEqual(assistant.ask("подтверждаю").text, "Записала.")
        self.assertIn("дрель у Пети", (self.dir / "notes.md").read_text(encoding="utf-8"))

    def test_notes_go_last_in_prompt(self):
        for name in ("about.md", "notes.md", "recipes.md", "README.md"):
            (self.dir / name).write_text(name, encoding="utf-8")
        names = [path.name for path in Knowledge(self.dir).files()]
        self.assertEqual(names, ["about.md", "recipes.md", "notes.md"])


if __name__ == "__main__":
    unittest.main()
