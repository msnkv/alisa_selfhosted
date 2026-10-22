"""Микрофон: запись фразы с определением, что человек договорил.

Ждать нажатия второй клавиши, чтобы остановить запись, неудобно — в
комнате руки заняты. Поэтому запись останавливается сама, когда человек
замолчал. Решает это silero-vad: крошечная модель, которая по окну из
пятисот двенадцати отсчётов говорит вероятность речи. Она же понадобится
на шаге 6, когда клавиша исчезнет совсем.
"""

from __future__ import annotations

import logging
import queue
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Whisper и silero-vad оба работают на шестнадцати килогерцах.
SAMPLE_RATE = 16000
# silero-vad требует ровно столько отсчётов в окне.
FRAME = 512
FRAME_SECONDS = FRAME / SAMPLE_RATE

SILERO_VAD_REPO = "snakers4/silero-vad"


class SileroVad:
    """Вероятность речи в окне. Модель весит около двух мегабайт."""

    name = "silero-vad"

    def __init__(self, models_dir: Path | None = None) -> None:
        import torch

        if models_dir is not None:
            models_dir.mkdir(parents=True, exist_ok=True)
            torch.hub.set_dir(str(models_dir))

        self._torch = torch
        self._model, _ = torch.hub.load(
            repo_or_dir=SILERO_VAD_REPO,
            model="silero_vad",
            trust_repo=True,
        )
        self._model.eval()

    def reset(self) -> None:
        # У модели есть внутреннее состояние: без сброса начало новой
        # фразы оценивается с оглядкой на предыдущую.
        self._model.reset_states()

    def probability(self, frame: np.ndarray) -> float:
        with self._torch.no_grad():
            tensor = self._torch.from_numpy(frame)
            return float(self._model(tensor, SAMPLE_RATE).item())


class EnergyVad:
    """Запасной вариант: громкость вместо модели.

    Нужен, если silero-vad не скачался (нет сети, заблокирован GitHub).
    В тихой комнате работает приемлемо, в шумной хуже: отличить речь от
    гудящего холодильника по одной громкости невозможно.
    """

    name = "громкость"

    def __init__(self, threshold: float = 0.01) -> None:
        self.threshold = threshold

    def reset(self) -> None:
        pass

    def probability(self, frame: np.ndarray) -> float:
        level = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        return 1.0 if level > self.threshold else 0.0


def make_vad(models_dir: Path | None = None):
    try:
        return SileroVad(models_dir)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "silero-vad не загрузился (%s: %s), определяю конец фразы по громкости",
            type(exc).__name__, exc,
        )
        return EnergyVad()


class Microphone:
    def __init__(
        self,
        *,
        device: int | str | None = None,
        speech_threshold: float = 0.5,
        silence_s: float = 0.8,
        min_speech_s: float = 0.3,
        max_phrase_s: float = 15.0,
        wait_speech_s: float = 6.0,
        models_dir: Path | None = None,
    ) -> None:
        self.device = device
        self.speech_threshold = speech_threshold
        self.silence_s = silence_s
        self.min_speech_s = min_speech_s
        self.max_phrase_s = max_phrase_s
        self.wait_speech_s = wait_speech_s
        self.models_dir = models_dir
        self._vad = None

    @property
    def vad(self):
        if self._vad is None:
            self._vad = make_vad(self.models_dir)
            log.info("определение конца фразы: %s", self._vad.name)
        return self._vad

    def warm_up(self) -> float:
        started = time.perf_counter()
        self.vad.probability(np.zeros(FRAME, dtype=np.float32))
        return time.perf_counter() - started

    def record_phrase(self, *, on_speech_start=None) -> np.ndarray:
        """Пишет с микрофона, пока человек не договорит.

        Возвращает запись целиком, включая тишину в начале: обрезать её
        незачем, у whisper для этого свой фильтр.
        Пустой массив означает, что речи так и не услышали.
        """
        import sounddevice as sd

        self.vad.reset()
        chunks: queue.Queue[np.ndarray] = queue.Queue()

        def callback(indata, _frames, _time_info, status):
            if status:
                log.debug("микрофон: %s", status)
            chunks.put(indata[:, 0].copy())

        frames: list[np.ndarray] = []
        speech_seconds = 0.0
        silence_seconds = 0.0
        heard_speech = False
        started = time.perf_counter()

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=FRAME,
            device=self.device,
            callback=callback,
        ):
            while True:
                try:
                    frame = chunks.get(timeout=1.0)
                except queue.Empty:
                    log.warning("микрофон молчит - данные не идут")
                    break

                frames.append(frame)
                is_speech = self.vad.probability(frame) >= self.speech_threshold

                if is_speech:
                    speech_seconds += FRAME_SECONDS
                    silence_seconds = 0.0
                    if not heard_speech and speech_seconds >= self.min_speech_s:
                        heard_speech = True
                        if on_speech_start is not None:
                            on_speech_start()
                elif heard_speech:
                    silence_seconds += FRAME_SECONDS
                    if silence_seconds >= self.silence_s:
                        break

                elapsed = time.perf_counter() - started
                if heard_speech and elapsed >= self.max_phrase_s:
                    log.info("фраза оборвана по лимиту %.0f с", self.max_phrase_s)
                    break
                if not heard_speech and elapsed >= self.wait_speech_s:
                    log.info("речи не услышал за %.0f с", self.wait_speech_s)
                    return np.zeros(0, dtype=np.float32)

        audio = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
        log.info(
            "записано %.2f с, из них речи %.2f с",
            len(audio) / SAMPLE_RATE, speech_seconds,
        )
        return audio

    def measure_level(self, seconds: float = 3.0) -> tuple[float, float]:
        """Средняя и максимальная громкость за несколько секунд.

        Нужно, чтобы проверить микрофон, не трогая распознавание: если тут
        нули, дело в устройстве или в системных настройках, а не в коде.
        """
        import sounddevice as sd

        recording = sd.rec(
            int(seconds * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=self.device,
        )
        sd.wait()
        samples = recording[:, 0]
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        return rms, float(np.max(np.abs(samples)))


def make_microphone(cfg) -> Microphone:
    return Microphone(
        device=cfg.mic.input_device,
        speech_threshold=cfg.mic.speech_threshold,
        silence_s=cfg.mic.silence_s,
        min_speech_s=cfg.mic.min_speech_s,
        max_phrase_s=cfg.mic.max_phrase_s,
        wait_speech_s=cfg.mic.wait_speech_s,
        models_dir=cfg.models_dir,
    )
