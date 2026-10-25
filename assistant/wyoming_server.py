"""Распознавание и синтез как Wyoming-сервисы для Home Assistant.

Этап 3: HA работает передатчиком, а слух и голос остаются на ПК, где
видеокарта. HA говорит с такими сервисами по протоколу Wyoming —
строки JSON и звук поверх обычного TCP. Подключаются они в HA
интеграцией Wyoming по адресу ПК и порту.

Почему внутри того же `--serve`, а не отдельной программой. Отдельная
программа загрузила бы вторую копию whisper и Silero: лишние
гигабайты видеопамяти и секунды прогрева. Здесь сервисы зовут те же
модели через `Brain` под тем же замком, так что спутник, страница и HA
просто встают в очередь друг за другом.

Сервис распознавания: HA шлёт звук кусками, мы копим, переводим в
16 кГц моно и по концу звука отдаём текст. Сервис синтеза: HA шлёт
текст, мы отдаём звук. Длинный текст режется на куски, и каждый уходит
сразу после синтеза — HA начинает играть, не дожидаясь всего ответа.

Потоковый протокол синтеза (текст по кусочкам, пока модель пишет) не
заявляем: наш агент отдаёт HA ответ целиком, см. server.py.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from functools import partial

import numpy as np
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import (
    AsrModel, AsrProgram, Attribution, Describe, Info, TtsProgram, TtsVoice, TtsVoiceSpeaker,
)
from wyoming.server import AsyncEventHandler, AsyncTcpServer
from wyoming.tts import Synthesize

from .tts import SPEAKERS, split_for_synthesis

log = logging.getLogger(__name__)

# whisper ест именно такое: 16 кГц, моно, см. mic.py.
STT_RATE = 16000
# Сколько отсчётов звука в одном сообщении синтеза. Около сорока
# миллисекунд при 48 кГц: мелко, чтобы HA начинал играть сразу.
TTS_CHUNK = 2048

WHISPER = Attribution(name="faster-whisper", url="https://github.com/SYSTRAN/faster-whisper")
SILERO = Attribution(name="Silero", url="https://github.com/snakers4/silero-models")


def stt_info(cfg) -> Info:
    model = AsrModel(
        name=cfg.stt.model, attribution=WHISPER, installed=True,
        description=f"whisper {cfg.stt.model}", version=None, languages=[cfg.stt.language],
    )
    return Info(asr=[AsrProgram(
        name="alisa-whisper", attribution=WHISPER, installed=True,
        description=f"{cfg.name}: распознавание на ПК", version=None, models=[model],
    )])


def tts_info(cfg) -> Info:
    voice = TtsVoice(
        name=cfg.tts.model, attribution=SILERO, installed=True,
        description=f"Silero {cfg.tts.model}", version=None, languages=["ru"],
        speakers=[TtsVoiceSpeaker(name=name) for name in SPEAKERS],
    )
    return Info(tts=[TtsProgram(
        name="alisa-silero", attribution=SILERO, installed=True,
        description=f"{cfg.name}: синтез на ПК", version=None, voices=[voice],
    )])


class SttHandler(AsyncEventHandler):
    def __init__(self, reader, writer, *, brain, info: Info) -> None:
        super().__init__(reader, writer)
        self.brain = brain
        self.info = info
        self.converter = AudioChunkConverter(rate=STT_RATE, width=2, channels=1)
        self.audio = bytearray()

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.info.event())
            return True
        if Transcribe.is_type(event.type):
            # Язык и модель у нас одни, просьбу HA о другом не слушаем.
            return True
        if AudioStart.is_type(event.type):
            self.audio.clear()
            # Новый преобразователь на каждую фразу: в нём хранится
            # состояние пересчёта частоты от прошлой.
            self.converter = AudioChunkConverter(rate=STT_RATE, width=2, channels=1)
            return True
        if AudioChunk.is_type(event.type):
            self.audio += self.converter.convert(AudioChunk.from_event(event)).audio
            return True
        if AudioStop.is_type(event.type):
            samples = np.frombuffer(bytes(self.audio), dtype="<i2").astype(np.float32) / 32768.0
            self.audio.clear()
            started = time.perf_counter()
            loop = asyncio.get_running_loop()
            try:
                text = await loop.run_in_executor(None, self.brain.transcribe_samples, samples)
            except Exception:  # noqa: BLE001
                log.exception("Wyoming: распознавание упало")
                text = ""
            log.info(
                "Wyoming: %.1f с звука распознано за %.2f с: %r",
                len(samples) / STT_RATE, time.perf_counter() - started, text,
            )
            await self.write_event(Transcript(text=text, language="ru").event())
            # Как у эталонного wyoming-faster-whisper: одна фраза на
            # соединение, HA на следующую откроет новое.
            return False
        return True


class TtsHandler(AsyncEventHandler):
    def __init__(self, reader, writer, *, brain, info: Info) -> None:
        super().__init__(reader, writer)
        self.brain = brain
        self.info = info

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.info.event())
            return True
        if not Synthesize.is_type(event.type):
            return True

        request = Synthesize.from_event(event)
        speaker = request.voice.speaker if request.voice else None
        if speaker not in SPEAKERS:
            speaker = None
        rate = self.brain.tts.sample_rate
        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        seconds = 0.0
        await self.write_event(AudioStart(rate=rate, width=2, channels=1).event())
        for part in split_for_synthesis(request.text):
            try:
                samples = await loop.run_in_executor(None, self.brain.synthesize, part, speaker)
            except Exception:  # noqa: BLE001
                log.exception("Wyoming: синтез упал на %r", part)
                break
            pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
            seconds += len(pcm) / rate
            for at in range(0, len(pcm), TTS_CHUNK):
                await self.write_event(AudioChunk(
                    rate=rate, width=2, channels=1, audio=pcm[at:at + TTS_CHUNK].tobytes(),
                ).event())
        await self.write_event(AudioStop().event())
        log.info(
            "Wyoming: %.1f с речи синтезировано за %.2f с: %r",
            seconds, time.perf_counter() - started, request.text,
        )
        return True


async def _serve(brain, cfg) -> None:
    servers = []
    if brain.stt is not None:
        handler = partial(SttHandler, brain=brain, info=stt_info(cfg))
        servers.append(AsyncTcpServer(cfg.wyoming.host, cfg.wyoming.stt_port).run(handler))
    if brain.tts is not None:
        handler = partial(TtsHandler, brain=brain, info=tts_info(cfg))
        servers.append(AsyncTcpServer(cfg.wyoming.host, cfg.wyoming.tts_port).run(handler))
    await asyncio.gather(*servers)


def start(brain, cfg) -> threading.Thread:
    """Поднимает сервисы в своём потоке со своим циклом asyncio.

    HTTP-сервер у нас из стандартной библиотеки и живёт в потоках, а
    Wyoming — на asyncio. Отдельный поток с отдельным циклом мирит их
    без переделки того, что уже работает.
    """
    def run() -> None:
        try:
            asyncio.run(_serve(brain, cfg))
        except Exception:  # noqa: BLE001
            log.exception("Wyoming-сервисы упали")
            print("  Wyoming-сервисы упали, подробности в журнале.")

    thread = threading.Thread(target=run, name="wyoming", daemon=True)
    thread.start()
    return thread
