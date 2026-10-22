"""Wake word: ассистент просыпается на кодовую фразу вместо клавиши.

Готовые модели openWakeWord все англоязычные, поэтому по умолчанию стоит
`hey_jarvis`. Своё русское слово — это обучение модели, оно в планах
этапа 4 и здесь не делается.

Почему это оставлено на последний шаг: тут придётся подбирать порог,
ловить ложные срабатывания от телевизора и разговоров, и заниматься этим
гораздо приятнее, когда всё остальное уже работает.
"""

from __future__ import annotations

import logging
import queue
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# openWakeWord ждёт кадры по 1280 отсчётов - это 80 мс на 16 кГц.
CHUNK = 1280

# Что доступно без обучения своей модели.
BUILTIN_MODELS = ("hey_jarvis", "alexa", "hey_mycroft", "hey_rhasspy")


class WakeWord:
    name = "openwakeword"

    def __init__(
        self,
        *,
        model: str = "hey_jarvis",
        threshold: float = 0.5,
        vad_threshold: float = 0.5,
        device: int | str | None = None,
        cooldown_s: float = 1.0,
    ) -> None:
        self.model_name = model
        self.threshold = threshold
        self.vad_threshold = vad_threshold
        self.device = device
        self.cooldown_s = cooldown_s
        self._model = None

    # --- загрузка ---------------------------------------------------------

    @staticmethod
    def _ensure_models(model_name: str) -> None:
        """Докачивает модель и общие слои, если их ещё нет.

        Кладутся они внутрь venv, в resources/models самого openwakeword.
        Это чужая папка, зато не нужно таскать пути: модель потом
        запрашивается по имени. Вес около пяти мегабайт, так что потерять
        их при пересборке venv не страшно.
        """
        import openwakeword
        import openwakeword.utils as utils

        resources = Path(openwakeword.__file__).parent / "resources" / "models"
        needed = [
            resources / "melspectrogram.onnx",
            resources / "embedding_model.onnx",
        ]
        have_wakeword = any(resources.glob(f"{model_name}*.onnx"))

        if all(path.exists() for path in needed) and have_wakeword:
            return

        log.info("докачиваю модели wake word (%s)", model_name)
        utils.download_models([model_name])

    @property
    def model(self):
        if self._model is None:
            from openwakeword.model import Model

            self._ensure_models(self.model_name)
            started = time.perf_counter()
            self._model = Model(
                wakeword_models=[self.model_name],
                # На Windows tflite не поддерживается, только onnx.
                # По умолчанию библиотека просит tflite и падает.
                inference_framework="onnx",
                # Встроенный VAD отсекает срабатывания на музыке и шуме:
                # без него телевизор будит ассистента заметно чаще.
                vad_threshold=self.vad_threshold,
            )
            log.info(
                "wake word %s загружен за %.2f с (порог %.2f, vad %.2f)",
                self.model_name, time.perf_counter() - started,
                self.threshold, self.vad_threshold,
            )
        return self._model

    def warm_up(self) -> float:
        started = time.perf_counter()
        self.model.predict(np.zeros(CHUNK, dtype=np.int16))
        self.model.reset()
        return time.perf_counter() - started

    # --- ожидание ---------------------------------------------------------

    def _stream(self):
        import sounddevice as sd

        chunks: queue.Queue[np.ndarray] = queue.Queue()

        def callback(indata, _frames, _time_info, status):
            if status:
                log.debug("микрофон wake word: %s", status)
            chunks.put(indata[:, 0].copy())

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            # openWakeWord работает с целочисленным звуком.
            dtype="int16",
            blocksize=CHUNK,
            device=self.device,
            callback=callback,
        )
        return stream, chunks

    def wait(self, *, timeout_s: float | None = None) -> bool:
        """Ждёт кодовую фразу. True — услышал, False — вышло время.

        Поток открывается заново на каждое ожидание: так заодно
        выбрасывается звук, накопившийся пока ассистент говорил, и он
        не будит сам себя через колонки.
        """
        self.model.reset()
        stream, chunks = self._stream()
        started = time.perf_counter()

        with stream:
            while True:
                if timeout_s is not None and time.perf_counter() - started > timeout_s:
                    return False
                try:
                    frame = chunks.get(timeout=1.0)
                except queue.Empty:
                    log.warning("микрофон молчит - данные не идут")
                    continue

                scores = self.model.predict(frame)
                best = max(scores.values()) if scores else 0.0
                if best >= self.threshold:
                    log.info("wake word услышан, уверенность %.2f", best)
                    self.model.reset()
                    return True

    def monitor(self, seconds: float = 30.0, report_from: float = 0.05) -> None:
        """Печатает уверенность модели - для подбора порога.

        Полезно, чтобы посмотреть, что даёт твой голос, а что телевизор
        и обычный разговор в комнате.
        """
        stream, chunks = self._stream()
        started = time.perf_counter()
        peak = 0.0
        triggers = 0

        print(
            f"Слушаю {seconds:.0f} с. Скажи «{self.model_name.replace('_', ' ')}»"
            f" несколько раз. Порог {self.threshold}."
        )
        with stream:
            while time.perf_counter() - started < seconds:
                try:
                    frame = chunks.get(timeout=1.0)
                except queue.Empty:
                    continue
                scores = self.model.predict(frame)
                best = max(scores.values()) if scores else 0.0
                peak = max(peak, best)
                if best >= self.threshold:
                    triggers += 1
                    print(f"  {time.perf_counter()-started:5.1f} с  СРАБОТАЛО  {best:.2f}")
                    self.model.reset()
                elif best >= report_from:
                    print(f"  {time.perf_counter()-started:5.1f} с  близко     {best:.2f}")

        print(f"\nМаксимум {peak:.2f}, срабатываний {triggers}.")
        if triggers == 0 and peak > 0.2:
            print("Близко, но порога не хватило — попробуй опустить wake.threshold.")
        elif triggers == 0:
            print("Не услышал. Проверь микрофон (--mic) и произношение.")


def make_wake_word(cfg) -> WakeWord:
    return WakeWord(
        model=cfg.wake.model,
        threshold=cfg.wake.threshold,
        vad_threshold=cfg.wake.vad_threshold,
        device=cfg.wake.input_device,
        cooldown_s=cfg.wake.cooldown_s,
    )
