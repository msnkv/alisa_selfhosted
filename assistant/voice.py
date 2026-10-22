"""Голосовой цикл. Два режима: по клавише и по кодовой фразе.

Режим с клавишей сделан первым намеренно: он позволил убедиться, что
распознавание, модель и синтез работают, до того как добавилась капризная
настройка wake word. Он остаётся в проекте как отладочный — когда wake
word шалит, удобно проверить всё остальное без него.

Остановку записи клавишей не делаем ни в одном режиме: в комнате руки
заняты, поэтому запись обрывается сама, когда человек замолчал (mic.py).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from .config import Config
from .llm import Assistant
from .prompt import FORGOTTEN_REPLY, is_forget
from .tts import SpeechQueue

log = logging.getLogger(__name__)

EXIT_WORDS = {"выход", "exit", "quit", "q"}

# Пока ассистент ищет в интернете, проходит от четырёх до шестнадцати секунд.
# Тишина такой длины из колонки выглядит как «сломалось», поэтому сразу
# говорим заглушку. Она синтезируется заранее, при прогреве.
FILLER_PHRASE = "Секунду, смотрю."

# Отзыв на кодовую фразу: без него непонятно, услышал ассистент или нет.
ACK_PHRASE = "Да?"

# Как часто можно повторять заглушку. Вопрос с несколькими поисками
# занимает до тридцати секунд, и одной фразы в начале на это не хватает:
# дальше идёт полминуты тишины, которая выглядит как поломка. Но и
# подряд её говорить нельзя - именно это звучало как каша. Поэтому
# повторяем, только когда канал звука свободен и прошло хотя бы столько.
FILLER_GAP_S = 8.0


@dataclass
class Timings:
    """Из чего сложилась задержка между «договорил» и «услышал ответ»."""

    record: float = 0.0
    stt: float = 0.0
    llm: float = 0.0
    synth: float = 0.0
    # Замеренное время до первого звука. В потоковом режиме складывать его
    # из частей нельзя: синтез идёт внахлёст с генерацией, поэтому просто
    # отмечаем момент, когда динамик заговорил.
    first_sound: float = 0.0

    @property
    def to_first_sound(self) -> float:
        """Сколько человек ждёт в тишине после того, как договорил."""
        if self.first_sound > 0:
            return self.first_sound
        return self.stt + self.llm + self.synth

    def describe(self) -> str:
        if self.first_sound > 0:
            return (
                f"запись {self.record:.1f} с | до первого звука "
                f"{self.first_sound:.1f} с (поток) | ответ целиком "
                f"{self.stt + self.llm:.1f} с"
            )
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
        wake=None,
    ) -> None:
        self.cfg = cfg
        self.assistant = assistant
        self.stt = stt
        self.tts = tts
        self.microphone = microphone
        self.wake = wake
        self._filler_audio = None
        self._ack_audio = None
        # Очередь звука текущего круга. Нужна заглушке: она обязана идти
        # тем же каналом, что и ответ, иначе два потока дерутся за вывод.
        self._speech = None
        # Когда заглушка звучала последний раз - чтобы не частить.
        self._filler_at = 0.0

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
        # Заглушку и отзыв синтезируем один раз здесь, чтобы в бою они
        # звучали мгновенно, а не через полсекунды после события.
        self._filler_audio = self.tts.synthesize(FILLER_PHRASE)
        self._ack_audio = self.tts.synthesize(ACK_PHRASE)
        print(f"  синтез речи: {time.perf_counter() - started:.1f} с")

        if self.wake is not None:
            started = time.perf_counter()
            self.wake.warm_up()
            print(f"  кодовая фраза: {time.perf_counter() - started:.1f} с")

    # --- один круг --------------------------------------------------------

    def _say_filler(self, name: str, arguments: dict) -> None:
        """Сказать «секунду, смотрю», если тишину больше нечем закрыть."""
        detail = arguments.get("query") or arguments.get("url") or ""
        print(f"  ... {name}: {detail}")
        if self._filler_audio is None:
            return

        speech = self._speech
        if speech is not None and speech.committed:
            # Ответ уже пошёл в динамик - заглушка теперь только помешает.
            return

        # Не чаще, чем раз в FILLER_GAP_S. Инструмент вызывается по
        # несколько раз на вопрос, и без этого на четырёх вызовах звучало
        # четыре «секунду, смотрю» подряд.
        now = time.perf_counter()
        if now - self._filler_at < FILLER_GAP_S:
            return
        self._filler_at = now
        try:
            # play_async идёт через тот же замок на вывод, что и ответ,
            # поэтому наложиться на него не может: кто успел, того и
            # динамик, а второй ждёт.
            self.tts.play_async(self._filler_audio)
        except Exception as exc:  # noqa: BLE001
            log.warning("заглушка не прозвучала: %s", exc)

    def listen_and_answer(
        self, *, wait_speech_s: float | None = None
    ) -> Timings | None:
        """Один круг: послушать, распознать, ответить голосом.

        wait_speech_s короче обычного на уточнении без кодовой фразы:
        там микрофон открыт на всякий случай, и висеть долго незачем.
        """
        timings = Timings()

        print("  слушаю...")
        started = time.perf_counter()
        audio = self.microphone.record_phrase(
            on_speech_start=lambda: print("  говоришь..."),
            wait_speech_s=wait_speech_s,
        )
        timings.record = time.perf_counter() - started
        # Момент «человек договорил»: от него и считается ожидание в тишине.
        after_record = time.perf_counter()

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

        # «Забудь» голосом: сменить тему, не трогая клавиатуру.
        if is_forget(transcript.text):
            self.assistant.forget()
            print("  разговор забыт")
            self.tts.say(FORGOTTEN_REPLY)
            return None

        # В потоковом режиме озвучка идёт параллельно генерации: очередь
        # получает предложения по одному и начинает говорить с первого,
        # не дожидаясь конца ответа.
        speech = SpeechQueue(self.tts) if self.cfg.tts.stream else None
        self._speech = speech
        self._filler_at = 0.0
        if speech is not None:
            speech.start()

        # Модель пишет текст на любом шаге, в том числе на том, где
        # вызывает инструмент. Синтез начинаем сразу, а говорить даём
        # только когда шаг закончился и оказался текстовым: иначе
        # присказка вроде «сейчас уточню цены» звучит, а потом тот же
        # ответ приходит ещё раз - и получается ответ, сказанный дважды.
        unspoken: list[str] = []
        step_text: list[str] = []

        def on_sentence(sentence: str) -> None:
            step_text.append(sentence)
            speech.add(sentence)

        def on_step(used_tools: bool) -> None:
            if used_tools:
                unspoken.extend(step_text)
                speech.discard()
            else:
                speech.commit()
            step_text.clear()

        # В разговоре инструментов не будет, значит шаг заведомо
        # текстовый - разрешаем говорить сразу, не дожидаясь конца
        # генерации. Иначе сказку пришлось бы слушать после минуты тишины.
        if speech is not None and self.assistant.will_be_long_form(transcript.text):
            speech.commit()

        started = time.perf_counter()
        try:
            answer = self.assistant.ask(
                transcript.text,
                on_tool_call=self._say_filler,
                on_sentence=on_sentence if speech is not None else None,
                on_step=on_step if speech is not None else None,
            )
        except Exception:
            if speech is not None:
                speech.stop()
            raise
        finally:
            self._speech = None
        timings.llm = time.perf_counter() - started

        # Что модель сказала по ходу, но вслух не пошло. Печатаем, чтобы
        # было видно: в логах иначе непонятно, куда делись её слова.
        if unspoken:
            print(f"  {self.cfg.name} (не вслух): {' '.join(unspoken)}")
        print(f"  {self.cfg.name}: {answer.text}")

        if speech is not None:
            if not speech.committed:
                # Досюда доходим, когда ответа как такового не было:
                # цикл сдался и подставил свою фразу. Её и говорим.
                speech.add(answer.text)
                speech.commit()
            # Модель договорила, но динамик ещё догоняет остаток.
            speech.close()
            speech.wait()
            if speech.first_sound_at is not None:
                timings.first_sound = speech.first_sound_at - after_record
        else:
            started = time.perf_counter()
            samples = self.tts.synthesize(answer.text)
            timings.synth = time.perf_counter() - started

            # Заглушка могла ещё звучать - обрываем, ответ важнее.
            self.tts.stop()
            self.tts.play(samples)

        log.info("голосовой круг: %s", timings.describe())
        print(f"  [{timings.describe()}]")
        return timings

    # --- режим с клавишей -------------------------------------------------

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

            if is_forget(command):
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

    # --- режим с кодовой фразой -------------------------------------------

    def run_wake(self) -> int:
        if self.wake is None:
            print("Кодовая фраза выключена в config.yaml")
            return 1

        self.warm_up()
        spoken = self.cfg.wake.model.replace("_", " ")
        print(
            f"\n{self.cfg.name} ждёт кодовую фразу: «{spoken}»."
            "\nСкажи её, потом задай вопрос. Ctrl+C — закончить.\n"
        )

        while True:
            try:
                heard = self.wake.wait()
            except KeyboardInterrupt:
                print()
                return 0

            if not heard:
                continue

            print("  да?")
            if self._ack_audio is not None:
                try:
                    self.tts.play_async(self._ack_audio)
                except Exception as exc:  # noqa: BLE001
                    log.warning("отзыв не прозвучал: %s", exc)

            # Первый круг ждёт речь как обычно, уточнения - меньше.
            # Требовать кодовую фразу перед «да, расскажи подробнее»
            # бессмысленно: ассистент сам это предложил.
            wait = None
            while True:
                try:
                    timings = self.listen_and_answer(wait_speech_s=wait)
                except KeyboardInterrupt:
                    print()
                    return 0
                except Exception as exc:  # noqa: BLE001
                    print(f"  ошибка: {type(exc).__name__}: {exc}")
                    log.exception("круг голосового цикла упал")
                    break

                if timings is None or self.cfg.wake.follow_up_s <= 0:
                    break
                wait = self.cfg.wake.follow_up_s
                print(
                    f"  (уточняй сразу, без кодовой фразы — слушаю"
                    f" {wait:.0f} с)"
                )

            # Пауза, чтобы ассистент не поймал кодовую фразу в собственном
            # ответе из колонок. С наушниками не нужна, с динамиками важна.
            if self.cfg.wake.cooldown_s > 0:
                time.sleep(self.cfg.wake.cooldown_s)
            print()
