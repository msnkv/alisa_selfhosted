"""Wyoming-сервисы для Home Assistant: протокол, без настоящих моделей.

Живой круг на whisper и Silero (синтез -> распознавание) проверен
руками 25.10.2026, см. JOURNAL.md. Здесь — что HA получает правильные
события: описание, текст по звуку, звук по тексту.
"""

import asyncio
import socket
import time
import unittest
from types import SimpleNamespace as NS

import numpy as np
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info
from wyoming.tts import Synthesize, SynthesizeVoice

from assistant import wyoming_server


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FakeBrain:
    def __init__(self):
        self.stt = object()
        self.tts = NS(sample_rate=48000)
        self.heard = None
        self.spoken = []

    def transcribe_samples(self, samples):
        self.heard = samples
        return "где болгарка"

    def synthesize(self, text, speaker=None):
        self.spoken.append((text, speaker))
        return np.full(4800, 0.5, dtype=np.float32)  # десятая секунда


class WyomingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.brain = FakeBrain()
        cls.cfg = NS(
            name="тест",
            stt=NS(model="large-v3-turbo", language="ru"),
            tts=NS(model="v5_6_ru"),
            wyoming=NS(host="127.0.0.1", stt_port=free_port(), tts_port=free_port()),
        )
        wyoming_server.start(cls.brain, cls.cfg)
        for port in (cls.cfg.wyoming.stt_port, cls.cfg.wyoming.tts_port):
            for _ in range(50):
                try:
                    socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
                    break
                except OSError:
                    time.sleep(0.05)

    def test_transcribe(self):
        async def run():
            async with AsyncTcpClient("127.0.0.1", self.cfg.wyoming.stt_port) as client:
                await client.write_event(Describe().event())
                info = Info.from_event(await client.read_event())
                self.assertEqual(info.asr[0].models[0].languages, ["ru"])
                await client.write_event(Transcribe(language="ru").event())
                # Секунда звука в 48 кГц стерео — должна дойти как 16 кГц моно.
                await client.write_event(AudioStart(rate=48000, width=2, channels=2).event())
                chunk = np.zeros(48000 * 2, dtype="<i2").tobytes()
                await client.write_event(AudioChunk(rate=48000, width=2, channels=2, audio=chunk).event())
                await client.write_event(AudioStop().event())
                event = await client.read_event()
                return Transcript.from_event(event).text

        self.assertEqual(asyncio.run(run()), "где болгарка")
        self.assertAlmostEqual(len(self.brain.heard), 16000, delta=50)
        self.assertEqual(self.brain.heard.dtype, np.float32)

    def test_synthesize(self):
        async def run():
            async with AsyncTcpClient("127.0.0.1", self.cfg.wyoming.tts_port) as client:
                await client.write_event(Describe().event())
                info = Info.from_event(await client.read_event())
                self.assertIn("baya", [s.name for s in info.tts[0].voices[0].speakers])
                await client.write_event(
                    # Голос по протоколу передаётся только вместе с
                    # названием модели — так его шлёт и HA.
                    Synthesize(text="Привет.", voice=SynthesizeVoice(name="v5_6_ru", speaker="baya")).event()
                )
                start = AudioStart.from_event(await client.read_event())
                audio = b""
                while True:
                    event = await client.read_event()
                    if AudioStop.is_type(event.type):
                        break
                    audio += AudioChunk.from_event(event).audio
                return start, audio

        start, audio = asyncio.run(run())
        self.assertEqual((start.rate, start.width, start.channels), (48000, 2, 1))
        self.assertEqual(len(audio), 4800 * 2)
        self.assertEqual(self.brain.spoken[-1], ("Привет.", "baya"))


if __name__ == "__main__":
    unittest.main()
