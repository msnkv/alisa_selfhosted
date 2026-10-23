"""Необязательный прогон настоящих AEC/Vosk на локальных WAV, без устройств.

Файлы в models/voice-stop-check: stop.wav (отдельное «стоп»), alisa.wav
(речь, включая слово «стоп»), negative.wav (похожие слова без команды).
Если модели или записей нет, тесты пропускаются; ничего не скачивается.

Локальный набор: официальная vosk-model-small-ru-0.22 в models/.
WAV сгенерированы на macOS командой /usr/bin/say -v Milena -o <путь>
--data-format=LEI16@<частота> '<текст>' (только в файл, без динамика):
  stop.wav, 16000: Стоп.
  alisa.wav, 48000: Сейчас я расскажу тебе историю про кота, который жил
    в маленьком доме у леса. Однажды он услышал громкий голос. Стоп!
    Подожди меня, сказала девочка. Они вместе пошли дальше по тропинке.
  negative.wav, 16000: Где лежит мой телефон? Сколько сейчас времени?
    Пожалуйста, продолжай рассказ. Стол. Сто. Суп. Потоп. Кто там?
    Что случилось? Не останавливайся.
Все переносы в тексте заменяются пробелами. Другой синтез/диктор может
изменить распознавание: это конкретный регрессионный набор, не универсальный
замер качества. Модель и WAV остаются в игнорируемой папке models/.
"""

import json
from pathlib import Path
import unittest
import wave

import numpy as np

from assistant.voice_stop import FRAME, RATE, StopDetector, make_verifier, resample


class RecordedSpeechTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1] / "models"
        model = root / "vosk-model-small-ru-0.22"
        audio = root / "voice-stop-check"
        if not model.is_dir() or not all((audio / f"{name}.wav").is_file() for name in ("stop", "alisa", "negative")):
            raise unittest.SkipTest("нет локальной модели Vosk и WAV для речевой проверки")
        try:
            from vosk import Model, SetLogLevel
            from pywebrtc_audio import AudioProcessor
        except ImportError:
            raise unittest.SkipTest("нужны vosk и pywebrtc-audio")
        SetLogLevel(-1)
        cls.model = Model(str(model))
        cls.processor = AudioProcessor
        cls.recordings = {}
        for name in ("stop", "alisa", "negative"):
            with wave.open(str(audio / f"{name}.wav"), "rb") as handle:
                assert handle.getnchannels() == 1 and handle.getsampwidth() == 2
                samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32768
                if len(samples) < 1600:
                    raise ValueError(f"пустая или слишком короткая запись {name}")
                cls.recordings[name] = resample(samples, handle.getframerate(), RATE)

    def run_audio(self, near, far):
        from vosk import KaldiRecognizer

        rec = KaldiRecognizer(self.model, 16000, json.dumps(["стоп", "[unk]"], ensure_ascii=False))
        rec.SetWords(True)
        detector = StopDetector(
            self.processor(sample_rate=RATE, echo_cancellation=True, noise_suppression=True),
            rec, make_verifier(self.model, 0.9),
        )
        return [i / RATE for i in range(0, len(near), FRAME)
                if detector.process(near[i:i+FRAME], far[i:i+FRAME], playing=bool(np.any(far[i:i+FRAME])))]

    def reference(self, name):
        samples = np.pad(self.recordings[name], (0, RATE * 2))
        return np.pad(samples, (0, (-len(samples)) % FRAME))

    def echo(self, far):
        near = np.zeros_like(far)
        near[3360:] += far[:-3360] * 0.5  # 70 мс задержки
        near[4320:] += far[:-4320] * 0.12  # ещё одно отражение через 20 мс
        return near

    def test_actual_stop_in_silence(self):
        reference = self.reference("alisa")
        near = np.zeros_like(reference)
        voice = self.recordings["stop"]
        near[RATE:RATE+len(voice)] = voice
        self.assertEqual(len(self.run_audio(near, np.zeros_like(near))), 1)

    def test_actual_stop_over_alisa_speech(self):
        far = self.reference("alisa")
        voice = self.recordings["stop"]
        recognised = 0
        for second in (4, 6, 8):
            near = self.echo(far)
            at = RATE * second
            near[at:at+len(voice)] += voice
            found = self.run_audio(near, far)
            self.assertLessEqual(len(found), 1, f"команда на {second} секунде: {found}")
            for moment in found:
                self.assertTrue(second < moment < second + 2)
            recognised += bool(found)
        # Консервативное подтверждение допускает пропуски. На этой записи
        # (тот же синтетический голос для обеих сторон) распознаны 2 из 3.
        # Это минимальный порог регрессии, НЕ обещание 100% распознавания.
        self.assertGreaterEqual(recognised, 2)

    def test_alisa_echo_alone_does_not_stop(self):
        far = self.reference("alisa")
        # Тут намеренно нет дополнительной текстовой блокировки:
        # проверяем собственно связку AEC/ASR на речи с «стоп».
        self.assertEqual(self.run_audio(self.echo(far), far), [])

    def test_other_words_do_not_stop(self):
        near = self.reference("negative")
        self.assertEqual(self.run_audio(near, np.zeros_like(near)), [])
