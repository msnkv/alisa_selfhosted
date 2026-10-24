"""«Стоп» отдельной фразой во время записи вопроса.

Первая часть — record_phrase с подменённым микрофоном: исключение
StopWord и сбой ловца. Вторая — настоящий Vosk на записях из
models/voice-stop-check (делает tests/make_voice_stop_wavs.py); без
модели и WAV пропускается.
"""

from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import wave

import numpy as np

from assistant.mic import FRAME, SAMPLE_RATE, Microphone, StopWord
from assistant.voice_stop import PhraseStop


class LoudVad:
    """Всё громче тишины — речь."""

    name = "test"

    def reset(self):
        pass

    def probability(self, frame):
        return 1.0 if np.abs(frame).max() > 0.01 else 0.0


def fake_sounddevice(audio):
    """sounddevice, чей поток сразу отдаёт всю запись кадрами по FRAME."""

    class Stream:
        def __init__(self, *, callback, **_kwargs):
            self.callback = callback

        def start(self):
            padded = np.pad(audio, (0, (-len(audio)) % FRAME))
            for i in range(0, len(padded), FRAME):
                self.callback(padded[i:i + FRAME].reshape(-1, 1), FRAME, None, None)

        def close(self):
            pass

    return types.SimpleNamespace(InputStream=Stream)


class FakeStop:
    def __init__(self, at_frame=None, finish=False, broken=False):
        self.at_frame, self.on_finish, self.broken = at_frame, finish, broken
        self.frames = 0

    def reset(self):
        self.frames = 0

    def feed(self, frame):
        if self.broken:
            raise RuntimeError("vosk упал")
        self.frames += 1
        return self.frames == self.at_frame

    def finish(self):
        return self.on_finish


class RecordPhraseTests(unittest.TestCase):
    def record(self, stop):
        mic = Microphone(silence_s=0.3, min_speech_s=0.1, wait_speech_s=2)
        mic._vad = LoudVad()
        mic.stop_word = stop
        speech = np.concatenate([
            np.zeros(SAMPLE_RATE // 2), np.full(SAMPLE_RATE // 2, 0.1), np.zeros(SAMPLE_RATE),
        ]).astype(np.float32)
        with patch.dict(sys.modules, {"sounddevice": fake_sounddevice(speech)}):
            return mic, mic.record_phrase()

    def test_without_stop_records_as_before(self):
        _, audio = self.record(None)
        self.assertGreater(len(audio), 0)

    def test_stop_mid_recording_raises(self):
        with self.assertRaises(StopWord):
            self.record(FakeStop(at_frame=20))

    def test_stop_in_tail_raises(self):
        with self.assertRaises(StopWord):
            self.record(FakeStop(finish=True))

    def test_broken_detector_does_not_break_recording(self):
        with self.assertLogs("assistant.mic", level="WARNING"):
            mic, audio = self.record(FakeStop(broken=True))
        self.assertGreater(len(audio), 0)
        self.assertIsNone(mic.stop_word)


class RecordedSpeechTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1] / "models"
        model = root / "vosk-model-small-ru-0.22"
        cls.audio = root / "voice-stop-check"
        if not model.is_dir() or not (cls.audio / "stop.wav").is_file():
            raise unittest.SkipTest("нет локальной модели Vosk и WAV для речевой проверки")
        try:
            from vosk import Model, SetLogLevel
        except ImportError:
            raise unittest.SkipTest("нужен vosk")
        SetLogLevel(-1)
        cls.model = Model(str(model))

    def load(self, name):
        path = self.audio / f"{name}.wav"
        if not path.is_file():
            self.skipTest(f"нет {name}.wav, перегенерируй make_voice_stop_wavs.py")
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
        samples = samples.astype(np.float32) / 32768
        step = rate // SAMPLE_RATE
        samples = samples[: len(samples) // step * step].reshape(-1, step).mean(axis=1)
        # Как в жизни: тишина до и после, кадры по 512 отсчётов.
        return np.pad(samples, (SAMPLE_RATE // 2, SAMPLE_RATE))

    def stopped(self, samples):
        detector = PhraseStop(self.model, 0.9)
        detector.reset()
        padded = np.pad(samples, (0, (-len(samples)) % FRAME))
        for i in range(0, len(padded), FRAME):
            if detector.feed(padded[i:i + FRAME]):
                return True
        return detector.finish()

    def test_single_stop(self):
        self.assertTrue(self.stopped(self.load("stop")))

    def test_stop_twice(self):
        self.assertTrue(self.stopped(self.load("stop_twice")))

    def test_similar_words_do_not_stop(self):
        # «Стол. Сто. Штопор. Стопка. Стой. Постой» и обычные вопросы.
        self.assertFalse(self.stopped(self.load("negative")))

    def test_story_with_stop_inside_does_not_stop(self):
        self.assertFalse(self.stopped(self.load("alisa")))


if __name__ == "__main__":
    unittest.main()
