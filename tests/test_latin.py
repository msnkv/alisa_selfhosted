"""Латиница перед синтезом: Silero её глотает, поэтому переводим в кириллицу."""

import unittest

from assistant.tts import latin_to_cyrillic


class LatinTests(unittest.TestCase):
    def test_live_case(self):
        # Живой ответ 29.10.2026, прозвучавший как «с и».
        self.assertEqual(
            latin_to_cyrillic("живёт на мини-пк с Linux и VPN, это сугои!"),
            "живёт на мини-пк с линукс и вэпээн, это сугои!",
        )

    def test_dictionary_and_hyphen(self):
        self.assertEqual(latin_to_cyrillic("Wi-Fi и Home Assistant"), "вай-фай и хоум ассистант")

    def test_unknown_acronym_spelled(self):
        self.assertEqual(latin_to_cyrillic("порт RTX"), "порт ар-ти-икс")

    def test_unknown_word_transliterated(self):
        self.assertEqual(latin_to_cyrillic("Makita"), "макита")
        self.assertEqual(latin_to_cyrillic("Bosch"), "бош")
        self.assertEqual(latin_to_cyrillic("systemd"), "системд")

    def test_cyrillic_untouched(self):
        text = "В твоих записях: на два килограмма капусты."
        self.assertEqual(latin_to_cyrillic(text), text)

    def test_no_latin_left(self):
        out = latin_to_cyrillic("Samsung Galaxy, iPhone, SSD и HDMI-кабель")
        self.assertNotRegex(out, "[A-Za-z]")


if __name__ == "__main__":
    unittest.main()
