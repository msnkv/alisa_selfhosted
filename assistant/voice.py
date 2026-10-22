"""Голосовой цикл: нажал клавишу, спросил голосом, услышал ответ.

Клавиша здесь временная — на шаге 6 её заменит wake word. Но порядок
именно такой намеренно: с клавишей видно, что всё остальное работает, и
капризную настройку wake word можно делать на уже говорящем ассистенте.

Остановку записи клавишей не делаем: в комнате руки заняты, поэтому запись
обрывается сама, когда человек замолчал (см. mic.py).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from .config import Config
from .llm import Assistant

log = logging.getLogger(__name__)

EXIT_WORDS = {"выход", "exit", "quit", "q"}
FORGET_WORDS = {"забудь", "сброс", "новая тема"}

# Пока ассистент ищет в интернете, проходит от четырёх до двенадцати секунд.
# Тишина такой длины из колонки выглядит как «сломалось», поэтому сразу
# говорим заглушку. Она синтезируется заранее, при прогреве.
FILLER_PHRASE = "Секунду, смотрю."


@dataclass
class Timings:
    """Из чего сложилась задержка между «договорил» и «услышал ответ»."""

    record: float = 0.0
    stt: float = 0.0
    llm: float = 0.0
    synth: float = 0.0

    @property
    def to_first_sound(self) -> float:
        """Сколько человек ждёт в тишине после того, как договорил."""
        return self.stt + self.llm + self.synth

    def describe(self) -> str:
        return (
            f"запись {self.record:.1f} с | до первого звука "
            f"{self.to_first_sound:.1f} с = распознавание {self.stt:.1f} "
            f"+ модель {self.llm:.1f} + синтез {self.synth:.1f}"
        )


class VoiceLoop:
    def __init__(
        self,
        cfg: Config,
        assistant: Assistant,
        stt,
        tts,
        microphone,
    ) -> None:
        self.cfg = cfg
        self.assistant = assistant
        self.stt = stt
        self.tts = tts
        self.microphone = microphone
        self._filler_audio = None

    # --- подготовка -------------------------------------------------------

    def warm_up(self) -> None:
        """Грузит все модели заранее.

        Без этого первая фраза ждёт загрузки распознавания и синтеза —
        это секунды, и выглядит как будто ассистент сломался.
        """
        print("Готовлю модели...")

        started = time.perf_counter()
        self.microphone.warm_up()
        print(f"  определение конца фразы: {time.perf_counter() - started:.1f} с")

        started = time.perf_counter()
        self.stt.warm_up()
        print(
            f"  распознавание: {time.perf_counter() - started:.1f} с"
            f" ({self.stt.actual_device}/{self.stt.actual_compute_type})"
        )

        started = time.perf_counter()
        self.tts.warm_up()
        # Заглушку синтезируем один раз здесь, чтобы в бою она звучала
        # мгновенно, а не через полсекунды после решения искать.
        self._filler_audio = self.tts.synthesize(FILLER_PHRASE)
        print(f"  синтез речи: {time.perf_counter() - started:.1f} с")

    # --- один круг --------------------------------------------------------

    def _say_filler(self, name: str, arguments: dict) -> None:
        detail = arguments.get("query") or arguments.get("url") or ""
        print(f"  ... {name}: {detail}")
        if self._filler_audio is None:
            return
        try:
            self.tts.play_async(self._filler_audio)
        except Exception as exc:  # noqa: BLE001
            log.warning("заглушка не прозвучала: %s", exc)

    def listen_and_answer(self) -> Timings | None:
        """Один круг: послушать, распознать, ответить голосом."""
        timings = Timings()

        print("  слушаю...")
        started = time.perf_counter()
        audio = self.microphone.record_phrase(
            on_speech_start=lambda: print("  говоришь...")
        )
        timings.record = time.perf_counter() - started

        if len(audio) == 0:
            print("  речи не услышал")
            return None

        started = time.perf_counter()
        transcript = self.stt.transcribe(audio)
        timings.stt = time.perf_counter() - started

        if not transcript.text:
            print("  не разобрал, повтори")
            self.tts.say("Не разобрала, повтори пожалуйста.")
            return None

        print(f"\n  ты: {transcript.text}")

        started = time.perf_counter()
        answer = self.assistant.ask(transcript.text, on_tool_call=self._say_filler)
        timings.llm = time.perf_counter() - started

        print(f"  {self.cfg.name}: {answer.text}")

        started = time.perf_counter()
        samples = self.tts.synthesize(answer.text)
        timings.synth = time.perf_counter() - started

        # Заглушка могла ещё звучать - обрываем, ответ важнее.
        self.tts.stop()
        self.tts.play(samples)

        log.info("голосовой круг: %s", timings.describe())
        print(f"  [{timings.describe()}]")
        return timings

    # --- цикл -------------------------------------------------------------

    def run(self) -> int:
        self.warm_up()
        print(
            f"\n{self.cfg.name} слушает. Нажми Enter и говори."
            "\nЗапись остановится сама, когда замолчишь. «выход» — закончить.\n"
        )

        while True:
            try:
                command = input("[Enter] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0

            if command in EXIT_WORDS:
                return 0

            if command in FORGET_WORDS:
                self.assistant.forget()
                print("  разговор забыт")
                print()
                continue

            try:
                self.listen_and_answer()
            except Exception as exc:  # noqa: BLE001
                print(f"  ошибка: {type(exc).__name__}: {exc}")
                log.exception("круг голосового цикла упал")
            print()
