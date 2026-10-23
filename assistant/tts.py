"""Озвучка ответов: Silero на русском.

Почему Silero, а не Piper: модели v5 сами расставляют ударения и разруливают
омографы. Без этого синтезатор говорит «зАмок» там, где нужен «замОк», а
размечать ударения руками в каждом ответе модели невозможно. Плюс у v5 есть
вопросительная интонация, а ассистент часто заканчивает вопросом
(«рассказать по шагам?»). Лицензия моделей CC BY-NC — для дома подходит.

Почему на процессоре, а не на видеокарте: распознавание речи на шаге 5
работает через CTranslate2, а не через torch, поэтому видеопамять торчу
вообще не нужна — пусть вся достаётся распознаванию. А Silero на
шестнадцати потоках успевает быстрее, чем это слышно.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SILERO_REPO = "snakers4/silero-models"

# Весь вывод звука в процессе проходит через этот замок. Причина дорогая:
# sounddevice держит один поток вывода в своей глобальной переменной, и
# каждый sd.play останавливает предыдущий и записывает туда себя. Два
# одновременных вызова из разных потоков эту переменную портят -
# живыми остаются два потока вывода, а библиотека знает про один.
# Слышно это было как два голоса сразу, а sd.wait() при этом возвращался
# раньше, чем звук кончился. Поэтому: один играет - остальные ждут.
_SOUND_LOCK = threading.Lock()

# Сколько символов Silero берёт за один вызов. Замерено на одной и той
# же фразе, повторённой нужное число раз: 620 символов проходят, 868 уже
# падают с «Model couldn't generate your text, probably it's too long».
# Сама библиотека предупреждает про тысячу, но это неправда - предел
# внутри на числе токенов, а не символов. Шестьсот с запасом.
#
# Поймано сказкой: модель написала четыре тысячи символов, сервер отдал
# их в синтез одним куском и ответил пятисоткой. На ПК этого не видно,
# там потоковая озвучка режет ответ на предложения сама.
MAX_SYNTH_CHARS = 600

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")

# Голоса русской модели v5. Послушать все: .\alisa.ps1 --voices "фраза"
SPEAKERS = ("aidar", "baya", "kseniya", "xenia", "eugene")

# Фраза для образцов голосов: тут и числа словами, и вопрос в конце,
# и шипящие — слышно, как голос справляется с тем, что ему реально дадут.

def split_for_synthesis(text: str, limit: int = MAX_SYNTH_CHARS) -> list[str]:
    """Режет текст на куски, которые Silero осилит за один вызов.

    Режем по границам предложений и набиваем кусок до предела: чем
    меньше кусков, тем меньше швов в речи. Шов слышен как лишняя пауза,
    потому что Silero дописывает тишину в конец каждого куска.

    Если одно предложение само длиннее предела (бывает: модель умеет
    выдать period на двести слов), оно режется по запятым, а в крайнем
    случае по словам. Пустых кусков не возвращаем.
    """
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []

    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for sentence in _SENTENCE_SPLIT.split(text):
        if not sentence.strip():
            continue
        # Предложение не влезает даже само по себе - дробим его.
        if len(sentence) > limit:
            flush()
            chunks.extend(_split_hard(sentence, limit))
            continue
        if len(current) + len(sentence) + 1 > limit:
            flush()
        current = f"{current} {sentence}".strip()
    flush()
    return chunks


def _split_hard(sentence: str, limit: int) -> list[str]:
    """Длинное предложение: по запятым, а что не влезло - по словам."""
    parts: list[str] = []
    current = ""
    for piece in re.split(r"(?<=[,;:])\s+", sentence):
        if len(piece) > limit:
            if current.strip():
                parts.append(current.strip())
                current = ""
            words = piece.split()
            for word in words:
                if len(current) + len(word) + 1 > limit:
                    parts.append(current.strip())
                    current = ""
                current = f"{current} {word}".strip()
            continue
        if len(current) + len(piece) + 1 > limit:
            parts.append(current.strip())
            current = ""
        current = f"{current} {piece}".strip()
    if current.strip():
        parts.append(current.strip())
    return parts


SAMPLE_TEXT = (
    "Болгарки лежат на левом стеллаже, третий уровень. "
    "Курс доллара сейчас восемьдесят один рубль двенадцать копеек. "
    "Рассказать про остальные полки?"
)


class SileroTTS:
    name = "silero"

    def __init__(
        self,
        *,
        model_id: str = "v5_6_ru",
        speaker: str = "xenia",
        sample_rate: int = 48000,
        device: str = "cpu",
        models_dir: Path | None = None,
        output_device: int | str | None = None,
    ) -> None:
        self.model_id = model_id
        self.speaker = speaker
        self.sample_rate = sample_rate
        self.device = device
        self.models_dir = models_dir
        self.output_device = output_device
        self._model = None
        # Какие именованные аргументы принимает apply_tts — выясняем на
        # первом вызове: у разных версий моделей они немного разные.
        self._accent_supported = True

    # --- загрузка ---------------------------------------------------------

    @property
    def model(self):
        """Модель грузится при первом обращении.

        Импорт torch и чтение весов занимают секунды, а селфтест и список
        устройств должны работать быстро и без скачивания моделей.
        """
        if self._model is None:
            import torch

            if self.models_dir is not None:
                self.models_dir.mkdir(parents=True, exist_ok=True)
                torch.hub.set_dir(str(self.models_dir))

            started = time.perf_counter()
            # trust_repo обязателен: иначе torch.hub ждёт подтверждения
            # с клавиатуры и в голосовом режиме это повиснет намертво.
            model, _ = torch.hub.load(
                repo_or_dir=SILERO_REPO,
                model="silero_tts",
                language="ru",
                speaker=self.model_id,
                trust_repo=True,
            )
            model.to(torch.device(self.device))
            self._model = model
            log.info(
                "Silero %s загружена на %s за %.2f с",
                self.model_id, self.device, time.perf_counter() - started,
            )
        return self._model

    def warm_up(self) -> float:
        """Прогрев: грузит модель и синтезирует короткую фразу.

        На шагах 5 и 6 это надо делать заранее, при старте, иначе первый же
        ответ будет ждать загрузки модели несколько секунд.
        """
        started = time.perf_counter()
        self.synthesize("Готова.")
        return time.perf_counter() - started

    # --- синтез -----------------------------------------------------------

    def synthesize(self, text: str, speaker: str | None = None) -> np.ndarray:
        """Озвучивает текст любой длины.

        Длинный режется на куски и склеивается — иначе Silero падает,
        см. `split_for_synthesis`. Снаружи это незаметно, и так и
        задумано: предел принадлежит Silero, значит и обходить его
        должна обёртка Silero, а не каждый вызывающий по отдельности.
        """
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)

        parts = split_for_synthesis(text)
        if len(parts) > 1:
            log.info("текст длинный (%d символов), режу на %d куска", len(text), len(parts))
            pieces = [self._synthesize_one(part, speaker) for part in parts]
            pieces = [piece for piece in pieces if len(piece)]
            if not pieces:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(pieces)
        return self._synthesize_one(text, speaker)

    def _synthesize_one(self, text: str, speaker: str | None = None) -> np.ndarray:
        """Один вызов Silero. Текст обязан быть в пределах его лимита."""
        started = time.perf_counter()
        kwargs = {
            "text": text,
            "speaker": speaker or self.speaker,
            "sample_rate": self.sample_rate,
        }
        if self._accent_supported:
            kwargs["put_accent"] = True
            kwargs["put_yo"] = True

        try:
            audio = self.model.apply_tts(**kwargs)
        except TypeError as exc:
            if not self._accent_supported:
                raise
            log.warning("модель не приняла put_accent/put_yo (%s), пробую без них", exc)
            self._accent_supported = False
            kwargs.pop("put_accent", None)
            kwargs.pop("put_yo", None)
            audio = self.model.apply_tts(**kwargs)

        samples = audio.detach().cpu().numpy().astype(np.float32)
        seconds = len(samples) / self.sample_rate
        elapsed = time.perf_counter() - started
        log.info(
            "синтез %s: %.2f с речи за %.2f с (быстрее реального времени в %.1f раза)",
            kwargs["speaker"], seconds, elapsed,
            seconds / elapsed if elapsed > 0 else 0,
        )
        return samples

    def say_short(self, text: str) -> np.ndarray:
        """Синтез служебной фразы: то же самое, но без тишины в конце.

        Silero дописывает в конец каждой фразы около четверти секунды
        тишины. В ответе это незаметно и даже к месту, а в служебной
        фразе — прямая задержка: «Ммм, сейчас» звучит перед ответом и
        всё это время держит замок на выводе звука. Замерено: хвост
        ровно 0,23–0,27 с у всех фраз, полезный звук кончается раньше.
        Восемь сотых оставляем, чтобы не звучало обрубленным.
        """
        samples = self.synthesize(text)
        loud = np.flatnonzero(np.abs(samples) > 0.01)
        if not len(loud):
            return samples
        end = int(loud[-1]) + int(0.08 * self.sample_rate)
        return samples[: min(end, len(samples))]

    # --- вывод ------------------------------------------------------------

    def say(self, text: str, speaker: str | None = None) -> None:
        """Синтезирует и проигрывает, дожидаясь конца."""
        samples = self.synthesize(text, speaker)
        if len(samples) == 0:
            return
        self.play(samples)

    def play(self, samples: np.ndarray) -> None:
        """Проигрывает и ждёт конца. Под замком — см. `_SOUND_LOCK`."""
        import sounddevice as sd

        with _SOUND_LOCK:
            sd.play(samples, samplerate=self.sample_rate, device=self.output_device)
            sd.wait()

    def play_async(self, samples: np.ndarray) -> None:
        """Запускает воспроизведение и сразу возвращает управление.

        Нужно для фразы-заглушки: пока она звучит, ассистент уже должен
        искать. Если ждать её конца, заглушка сама добавит полторы секунды
        к задержке, от которой мы ею же и спасаемся.

        Внутри — обычный `play` в отдельном потоке, чтобы и эта фраза
        прошла через замок на вывод, а не в обход него.
        """
        threading.Thread(
            target=self.play, args=(samples,), name="tts-async", daemon=True
        ).start()

    def stop(self) -> None:
        """Обрывает то, что играет сейчас.

        Готовый ответ важнее недоговорённой заглушки.
        """
        import sounddevice as sd

        sd.stop()

    def save(self, text: str, path: Path, speaker: str | None = None) -> Path:
        import soundfile as sf

        samples = self.synthesize(text, speaker)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), samples, self.sample_rate)
        return path


class SpeechQueue:
    """Озвучка ответа по предложениям, с синтезом заранее.

    Как это работает и зачем так сложно.

    Модель отвечает в несколько шагов: сходить в поиск, прочитать
    страницу, и только потом ответить. Текст она при этом может писать на
    любом шаге — в том числе на том, где вызывает инструмент. Озвучивать
    такой текст нельзя: он не ответ, а присказка вроде «сейчас уточню
    цены», и дальше тот же ответ придёт ещё раз. Вживую это звучало как
    ответ, сказанный дважды.

    Но и ждать конца всей генерации, чтобы начать синтез, расточительно.
    Поэтому разведены два момента: **синтез начинается сразу**, как
    только предложение готово, а **воспроизведение — только после
    `commit`**, то есть когда шаг закончился и оказался текстовым. Если
    шаг оказался с вызовом инструмента, зовётся `discard`, и
    насинтезированное просто выбрасывается.

    Выброс сделан через номер поколения, а не через чистку очередей: поток
    синтеза может быть в середине работы, и отнимать у него очередь —
    гонка. Вместо этого у каждой фразы есть номер, `discard` его
    увеличивает, и динамик молча пропускает всё с устаревшим номером.

    Потоков два: пока динамик играет текущее предложение, синтезатор
    готовит следующее. Синтез быстрее реального времени примерно в
    пятнадцать раз, поэтому речь идёт без разрывов.
    """

    def __init__(self, tts: SileroTTS) -> None:
        self.tts = tts
        self._texts: queue.Queue = queue.Queue()
        # Ограничение не случайное: синтез быстрее реального времени
        # примерно в пятнадцать раз, и на длинной сказке он успел бы
        # насинтезировать двадцать минут звука в память, пока играет
        # первая минута. Это сотни мегабайт. С пределом синтезатор
        # просто ждёт динамик.
        self._audio: queue.Queue = queue.Queue(maxsize=4)
        self._synth: threading.Thread | None = None
        self._player: threading.Thread | None = None
        self._stopped = threading.Event()
        # Разрешение говорить. Снято, пока не ясно, текстовый ли это шаг.
        self._allowed = threading.Event()
        self._generation = 0
        # Момент, когда из динамика пошёл первый звук ответа.
        self.first_sound_at: float | None = None
        self.committed = False
        self.spoken: list[str] = []

    def start(self) -> None:
        self._synth = threading.Thread(
            target=self._synth_loop, name="tts-synth", daemon=True
        )
        self._player = threading.Thread(
            target=self._play_loop, name="tts-play", daemon=True
        )
        self._synth.start()
        self._player.start()

    # --- вход -------------------------------------------------------------

    def add(self, sentence: str) -> None:
        """Отдать предложение в синтез. Говорить его пока нельзя."""
        if self._stopped.is_set():
            return
        sentence = sentence.strip()
        if sentence:
            self._texts.put((sentence, self._generation))

    def commit(self) -> None:
        """Шаг оказался текстовым: это ответ, можно говорить."""
        self.committed = True
        self._allowed.set()

    def discard(self) -> None:
        """Шаг оказался вызовом инструмента: сказанное там не ответ."""
        self._generation += 1
        self._allowed.clear()

    def close(self) -> None:
        """Предложений больше не будет; дать потокам договорить."""
        self._texts.put(None)

    def wait(self, timeout: float | None = None) -> None:
        for thread in (self._synth, self._player):
            if thread is not None:
                thread.join(timeout)

    def stop(self) -> None:
        """Оборвать всё — например, если вопрос упал с ошибкой."""
        import sounddevice as sd

        self._stopped.set()
        # Разрешение ставим тоже, иначе динамик навсегда повиснет на нём.
        self._allowed.set()
        self._texts.put(None)
        # Без ожидания: очередь звука ограничена, и обычный put мог бы
        # повиснуть именно тогда, когда мы пытаемся всё оборвать.
        try:
            self._audio.put_nowait(None)
        except queue.Full:
            pass
        sd.stop()

    # --- потоки -----------------------------------------------------------

    def _synth_loop(self) -> None:
        while True:
            item = self._texts.get()
            if item is None or self._stopped.is_set():
                # None дальше по цепочке: он скажет динамику, что всё.
                self._audio.put(None)
                return
            sentence, generation = item
            if generation != self._generation:
                continue
            try:
                samples = self.tts.synthesize(sentence)
            except Exception as exc:  # noqa: BLE001
                # Одно неудачное предложение не должно рушить весь ответ.
                log.warning("не синтезировала «%s»: %s", sentence, exc)
                continue
            if len(samples):
                self._audio.put((sentence, samples, generation))

    def _play_loop(self) -> None:
        while True:
            item = self._audio.get()
            if item is None or self._stopped.is_set():
                return
            sentence, samples, generation = item
            if generation != self._generation:
                # Сказанное на шаге с инструментом - не ответ.
                log.info("не озвучиваю «%s»: шаг оказался не ответом", sentence)
                continue

            self._allowed.wait()
            if self._stopped.is_set() or generation != self._generation:
                continue

            if self.first_sound_at is None:
                self.first_sound_at = time.perf_counter()
            try:
                self.tts.play(samples)
            except Exception as exc:  # noqa: BLE001
                log.warning("не проиграла «%s»: %s", sentence, exc)
            self.spoken.append(sentence)


ENGINES = {"silero": SileroTTS}


def make_tts(cfg) -> SileroTTS:
    """Создаёт озвучку по настройкам из config.yaml."""
    try:
        factory = ENGINES[cfg.tts.engine]
    except KeyError:
        raise ValueError(
            f"неизвестный движок озвучки {cfg.tts.engine!r}, "
            f"доступны: {', '.join(ENGINES)}"
        ) from None

    return factory(
        model_id=cfg.tts.model,
        speaker=cfg.tts.speaker,
        sample_rate=cfg.tts.sample_rate,
        device=cfg.tts.device,
        models_dir=cfg.models_dir,
        output_device=cfg.tts.output_device,
    )


def describe_devices() -> str:
    """Список аудиоустройств — нужен, чтобы выбрать USB-гарнитуру."""
    import sounddevice as sd

    lines = []
    try:
        default_in, default_out = sd.default.device
    except Exception:  # noqa: BLE001
        default_in, default_out = -1, -1

    for number, device in enumerate(sd.query_devices()):
        marks = []
        if device["max_input_channels"] > 0:
            marks.append("вход")
        if device["max_output_channels"] > 0:
            marks.append("выход")
        if number == default_in:
            marks.append("вход по умолчанию")
        if number == default_out:
            marks.append("выход по умолчанию")
        lines.append(f"  {number:3} {device['name']}  [{', '.join(marks)}]")
    return "\n".join(lines)
