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
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SILERO_REPO = "snakers4/silero-models"

# Голоса русской модели v5. Послушать все: .\alisa.ps1 --voices "фраза"
SPEAKERS = ("aidar", "baya", "kseniya", "xenia", "eugene")

# Фраза для образцов голосов: тут и числа словами, и вопрос в конце,
# и шипящие — слышно, как голос справляется с тем, что ему реально дадут.
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
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)

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

    # --- вывод ------------------------------------------------------------

    def say(self, text: str, speaker: str | None = None) -> None:
        """Синтезирует и проигрывает, дожидаясь конца."""
        samples = self.synthesize(text, speaker)
        if len(samples) == 0:
            return
        self.play(samples)

    def play(self, samples: np.ndarray) -> None:
        import sounddevice as sd

        sd.play(samples, samplerate=self.sample_rate, device=self.output_device)
        sd.wait()

    def play_async(self, samples: np.ndarray) -> None:
        """Запускает воспроизведение и сразу возвращает управление.

        Нужно для фразы-заглушки: пока она звучит, ассистент уже должен
        искать. Если ждать её конца, заглушка сама добавит полторы секунды
        к задержке, от которой мы ею же и спасаемся.
        """
        import sounddevice as sd

        sd.play(samples, samplerate=self.sample_rate, device=self.output_device)

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
