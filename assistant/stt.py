"""Распознавание речи: faster-whisper на видеокарте.

faster-whisper работает не через torch, а через CTranslate2 — поэтому вся
видеопамять достаётся ему, а синтез речи спокойно живёт на процессоре.

Про Blackwell (RTX 50-й серии, sm_120): на старых сборках CTranslate2 режим
int8 падал с CUBLAS_STATUS_NOT_SUPPORTED, и проявлялось это не при загрузке
модели, а при первом же вычислении. Поэтому загрузка идёт по цепочке с
пробным распознаванием на каждом шаге: int8 на карте, потом float16 на
карте, потом процессор. Что реально заработало — видно в логе.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Whisper везде работает на шестнадцати килогерцах.
SAMPLE_RATE = 16000


@dataclass
class Transcript:
    text: str
    audio_seconds: float
    seconds: float
    language: str = "ru"
    language_probability: float = 0.0

    @property
    def speed(self) -> float:
        """Во сколько раз быстрее реального времени."""
        return self.audio_seconds / self.seconds if self.seconds > 0 else 0.0


def _register_cuda_dlls() -> None:
    """Показывает Windows, где лежат cublas и cudnn из пакетов pip.

    CUDA Toolkit мы не ставили, нужные библиотеки пришли пакетами
    nvidia-cublas-cu12 и nvidia-cudnn-cu12. Их dll лежат внутри
    site-packages, а Windows там не ищет.

    Тонкость, на которой я потерял время: одного os.add_dll_directory
    недостаточно. Эти каталоги просматривает только LoadLibraryEx с
    особым флагом, а CTranslate2 грузит cublas обычным LoadLibrary -
    и тот видит лишь системные каталоги и PATH. Симптом: модель на
    cuda загружается, а падает уже на первом вычислении с
    «Library cublas64_12.dll is not found or cannot be loaded».
    Поэтому каталоги дописываются и в PATH процесса.
    """
    if sys.platform != "win32":
        return

    base = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not base.is_dir():
        log.warning("пакеты nvidia-* не найдены в %s, видеокарта может не заработать", base)
        return

    directories = [str(path) for path in sorted(base.glob("*/bin"))]
    if not directories:
        log.warning("в %s нет каталогов bin с dll", base)
        return

    for directory in directories:
        try:
            os.add_dll_directory(directory)
        except OSError as exc:
            log.warning("не удалось добавить каталог dll %s: %s", directory, exc)

    current = os.environ.get("PATH", "")
    missing = [d for d in directories if d not in current]
    if missing:
        os.environ["PATH"] = os.pathsep.join(missing) + os.pathsep + current
    log.info("каталоги библиотек CUDA подключены: %s", ", ".join(
        Path(d).parent.name for d in directories
    ))


class WhisperSTT:
    name = "faster-whisper"

    def __init__(
        self,
        *,
        model: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "int8_float16",
        language: str = "ru",
        beam_size: int = 1,
        vad_filter: bool = True,
        models_dir: Path | None = None,
    ) -> None:
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.models_dir = models_dir
        self._model = None
        # Что в итоге заработало — пригодится в селфтесте и логах.
        self.actual_device = device
        self.actual_compute_type = compute_type

    # --- загрузка ---------------------------------------------------------

    def _attempts(self) -> list[tuple[str, str]]:
        attempts: list[tuple[str, str]] = []
        if self.device == "cuda":
            attempts.append(("cuda", self.compute_type))
            if self.compute_type != "float16":
                attempts.append(("cuda", "float16"))
        else:
            attempts.append((self.device, self.compute_type))
        # Процессор как последняя надежда: медленно, но работает всегда.
        attempts.append(("cpu", "int8"))
        return attempts

    @property
    def model(self):
        if self._model is None:
            _register_cuda_dlls()
            from faster_whisper import WhisperModel

            download_root = str(self.models_dir / "whisper") if self.models_dir else None
            silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
            errors = []

            for device, compute_type in self._attempts():
                started = time.perf_counter()
                try:
                    candidate = WhisperModel(
                        self.model_name,
                        device=device,
                        compute_type=compute_type,
                        download_root=download_root,
                    )
                    # Пробное распознавание обязательно: ошибки CUDA
                    # всплывают при вычислении, а не при загрузке модели.
                    segments, _ = candidate.transcribe(silence, language=self.language)
                    list(segments)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{device}/{compute_type}: {type(exc).__name__}: {exc}")
                    log.warning(
                        "%s на %s/%s не заработала: %s",
                        self.model_name, device, compute_type, exc,
                    )
                    continue

                self._model = candidate
                self.actual_device = device
                self.actual_compute_type = compute_type
                log.info(
                    "%s загружена на %s/%s за %.1f с",
                    self.model_name, device, compute_type,
                    time.perf_counter() - started,
                )
                break

            if self._model is None:
                raise RuntimeError(
                    "не удалось запустить распознавание речи:\n  "
                    + "\n  ".join(errors)
                )
        return self._model

    def warm_up(self) -> float:
        """Грузит модель и прогоняет секунду тишины.

        Без прогрева первая же фраза ждёт загрузки модели - это секунды.
        """
        started = time.perf_counter()
        _ = self.model
        return time.perf_counter() - started

    # --- распознавание ----------------------------------------------------

    def transcribe(self, audio: np.ndarray | str | Path) -> Transcript:
        if isinstance(audio, (str, Path)):
            audio_input: np.ndarray | str = str(audio)
            audio_seconds = 0.0
        else:
            audio_input = audio.astype(np.float32)
            audio_seconds = len(audio) / SAMPLE_RATE

        started = time.perf_counter()
        segments, info = self.model.transcribe(
            audio_input,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            # Без этого Whisper на тишине и шуме любит выдумывать фразы
            # вроде «Субтитры сделал DimaTorzok».
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        elapsed = time.perf_counter() - started

        transcript = Transcript(
            text=text,
            audio_seconds=audio_seconds or getattr(info, "duration", 0.0),
            seconds=elapsed,
            language=getattr(info, "language", self.language),
            language_probability=getattr(info, "language_probability", 0.0),
        )
        log.info(
            "распознано за %.2f с (%.1fx реального времени): %r",
            transcript.seconds, transcript.speed, transcript.text,
        )
        return transcript


ENGINES = {"faster-whisper": WhisperSTT}


def make_stt(cfg) -> WhisperSTT:
    try:
        factory = ENGINES[cfg.stt.engine]
    except KeyError:
        raise ValueError(
            f"неизвестный движок распознавания {cfg.stt.engine!r}, "
            f"доступны: {', '.join(ENGINES)}"
        ) from None

    return factory(
        model=cfg.stt.model,
        device=cfg.stt.device,
        compute_type=cfg.stt.compute_type,
        language=cfg.stt.language,
        beam_size=cfg.stt.beam_size,
        vad_filter=cfg.stt.vad_filter,
        models_dir=cfg.models_dir,
    )
