"""Голосовое «стоп»: duplex-аудио -> WebRTC AEC -> два прохода Vosk.

В ASR не попадает сырой микрофон при отказе AEC. Речь Алисы со словом
«стоп» дополнительно закрывает распознавание до затухания её эха.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
from collections import deque

import numpy as np

log = logging.getLogger(__name__)
RATE = 48000
FRAME = 480  # 10 мс, один кадр WebRTC
# Запас очереди кадров: три секунды. Полсекунды на старом ноуте не
# хватило: средний кадр успевает с огромным запасом, но единичная
# заминка потока (проверка слова, приём следующей фразы) длиннее.
QUEUE_FRAMES = 300
ECHO_TAIL_S = 0.5
WARMUP_S = 0.8
MIN_LEVEL = 0.003
OWN_STOP = re.compile(r"\bстоп\b", re.IGNORECASE)


def is_stop(result, min_confidence):
    words = result.get("result") or []
    return (
        result.get("text", "").strip().lower() == "стоп"
        and len(words) == 1 and words[0].get("word") == "стоп"
        and words[0].get("conf", 0) >= min_confidence
        and 0.12 <= words[0].get("end", 0) - words[0].get("start", 0) <= 1.5
    )


def make_verifier(model, min_confidence):
    from vosk import KaldiRecognizer

    recognizer = KaldiRecognizer(model, 16000)
    recognizer.SetWords(True)

    def verify(pcm):
        recognizer.Reset()
        recognizer.AcceptWaveform(pcm + bytes(16000))  # полсекунды тишины для конца фразы
        return is_stop(json.loads(recognizer.FinalResult()), min_confidence)

    return verify


def resample(samples, source_rate, target_rate):
    """Восходящая передискретизация; 48 кГц Silero остаются без изменений."""
    if source_rate == target_rate:
        return np.asarray(samples, dtype=np.float32)
    if source_rate > target_rate:
        raise ValueError("для голосового стоп частота ответа должна быть не выше 48 кГц")
    positions = np.arange(round(len(samples) * target_rate / source_rate)) * source_rate / target_rate
    return np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)


class StopDetector:
    """Один поток владеет и AEC, и распознавателем. Вне аудиоколбэка."""

    def __init__(self, processor, recognizer, verify, min_confidence=0.9):
        if not 0.8 <= min_confidence <= 1.0:
            raise ValueError("voice_stop.min_confidence должен быть от 0.8 до 1.0")
        self.processor = processor
        self.recognizer = recognizer
        self.min_confidence = min_confidence
        self.verify = verify
        self.rendered = 0
        self.tail = 0
        self.muted = False
        self.peak = 0.0
        # FIR перед 48 -> 16 кГц, чтобы не сложить высокие частоты в речь.
        axis = np.arange(63) - 31
        kernel = np.sinc(axis / 3) * np.hamming(63)
        self.kernel = (kernel / kernel.sum()).astype(np.float32)
        self.previous = np.zeros(62, dtype=np.float32)
        self.audio = deque(maxlen=300)  # очищенные последние три секунды, в памяти
        self.samples_seen = 0
        self.verifies = 0
        self.verify_worst = 0.0

    def reset(self):
        self.recognizer.Reset()
        self.audio.clear()
        # Vosk.Reset сбрасывает текст, но НЕ временную шкалу слов.
        # Считаем все переданные в ASR отсчёты с начала его жизни.
        self.peak = 0.0

    def process(self, near, far, *, blocked=False, playing=False):
        clean = np.asarray(self.processor.process(near, far), dtype=np.float32)
        if clean.shape != (FRAME,) or not np.isfinite(clean).all():
            raise ValueError("AEC вернул некорректный кадр")
        if float(np.mean(far ** 2)) > 1e-8:
            self.rendered += FRAME
            self.tail = round(ECHO_TAIL_S * RATE / FRAME)
        elif self.tail:
            self.tail -= 1
        filtered = np.convolve(np.concatenate((self.previous, clean)), self.kernel, mode="valid")[::3]
        self.previous = clean[-62:].copy()
        warming = self.rendered < WARMUP_S * RATE and (playing or self.tail > 0)
        if blocked or warming:
            if not self.muted:
                self.reset()
            self.muted = True
            self.peak = 0.0
            return False
        if self.muted:
            self.reset()
            self.muted = False
        self.audio.append(filtered)
        self.samples_seen += len(filtered)
        self.peak = max(self.peak, float(np.sqrt(np.mean(filtered ** 2))))
        pcm = (np.clip(filtered, -1, 1) * 32767).astype("<i2").tobytes()
        if not self.recognizer.AcceptWaveform(pcm):
            return False
        result = json.loads(self.recognizer.Result())
        accepted = False
        if is_stop(result, self.min_confidence) and self.peak >= MIN_LEVEL:
            word = result["result"][0]
            recorded = np.concatenate(self.audio)
            offset = self.samples_seen - len(recorded)
            start = round(word["start"] * 16000) - 2400
            end = round(word["end"] * 16000) + 2400
            if start >= offset - 2400 and end <= self.samples_seen + 2400:
                snippet = recorded[max(0, start - offset):max(0, end - offset)]
                if len(snippet) >= 1600:
                    pcm = (np.clip(snippet, -1, 1) * 32767).astype("<i2").tobytes()
                    began = time.perf_counter()
                    accepted = self.verify(pcm)
                    self.verifies += 1
                    self.verify_worst = max(self.verify_worst, time.perf_counter() - began)
        self.reset()
        return accepted


class VoiceStop:
    """Один duplex-поток даёт AEC точный эталон ВСЕГО звука спутника."""

    def __init__(self, model, settings, input_device=None, output_device=None):
        self.model = model
        self.settings = settings
        self.devices = (input_device, output_device)
        self.stream = None
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.shutdown = threading.Event()
        self.failed = threading.Event()
        self.frames = queue.Queue(maxsize=QUEUE_FRAMES)
        # Замеры за ответ: по ним видно, хватает ли запаса очереди.
        self.processed = 0
        self.busy_s = 0.0
        self.worst_s = 0.0
        self.deepest = 0
        self.detector = None
        self.samples = np.empty(0, dtype=np.float32)
        self.cursor = 0
        self.own_stop = False
        self.block_until = 0.0
        self.dac_end = 0.0
        self.error = ""

    def start(self, cancelled):
        import sounddevice as sd
        from pywebrtc_audio import AudioProcessor
        from vosk import KaldiRecognizer

        self.cancelled = cancelled
        processor = AudioProcessor(
            sample_rate=RATE, echo_cancellation=True, noise_suppression=True,
            auto_gain_control=False, stream_delay_ms=self.settings.echo_delay_ms,
        )
        # Грамматика только предлагает кандидата на очищенном сигнале.
        # Обычный ASR повторно проверяет короткий очищенный фрагмент:
        # ограниченная грамматика сама по себе даёт ложные команды.
        grammar = json.dumps(["стоп", "[unk]"], ensure_ascii=False)
        recognizer = KaldiRecognizer(self.model, 16000, grammar)
        recognizer.SetWords(True)
        detector = StopDetector(
            processor, recognizer, make_verifier(self.model, self.settings.min_confidence),
            self.settings.min_confidence,
        )
        self.detector = detector

        def listen():
            while not self.shutdown.is_set() and not cancelled.is_set():
                try:
                    near, far, blocked, playing = self.frames.get(timeout=0.1)
                except queue.Empty:
                    continue
                if self.failed.is_set():
                    continue
                self.deepest = max(self.deepest, self.frames.qsize() + 1)
                began = time.perf_counter()
                try:
                    detected = detector.process(near, far, blocked=blocked, playing=playing)
                    spent = time.perf_counter() - began
                    self.processed += 1
                    self.busy_s += spent
                    self.worst_s = max(self.worst_s, spent)
                    if detected and not self.failed.is_set() and not self.shutdown.is_set():
                        log.info("голосовой стоп: независимая команда после AEC")
                        cancelled.set()
                except Exception as exc:
                    self.fail(f"ошибка AEC/ASR: {type(exc).__name__}")

        self.worker = threading.Thread(target=listen, name="voice-stop", daemon=True)
        self.stream = sd.Stream(
            samplerate=RATE, blocksize=FRAME, channels=1, dtype="float32",
            device=self.devices, latency="high", callback=self.callback,
        )
        try:
            self.stream.start()
            self.worker.start()
        except BaseException:
            self.stream.close()
            self.stream = None
            raise

    def fail(self, reason):
        if not self.failed.is_set():
            self.error = reason
            self.failed.set()
            log.warning("голосовой стоп отключён для текущего ответа: %s", reason)

    def callback(self, indata, outdata, frames, timing, status):
        outdata.fill(0)
        if self.cancelled.is_set() or self.shutdown.is_set():
            self.done.set()
            return
        with self.lock:
            count = min(frames, len(self.samples) - self.cursor)
            playing = count > 0
            if count:
                outdata[:count, 0] = self.samples[self.cursor:self.cursor + count]
                self.cursor += count
                if self.own_stop:
                    self.block_until = time.monotonic() + max(0, timing.outputBufferDacTime - timing.currentTime) + frames / RATE + ECHO_TAIL_S
                if self.cursor == len(self.samples):
                    self.dac_end = timing.outputBufferDacTime + count / RATE
                    self.done.set()
            blocked = time.monotonic() < self.block_until
        if status or frames != FRAME:
            self.fail("сбой аудиобуфера; эталон эха ненадёжен")
            return
        if self.failed.is_set():
            return
        try:
            self.frames.put_nowait((indata[:, 0].copy(), outdata[:, 0].copy(), blocked, playing))
        except queue.Full:
            self.fail("обработка микрофона не успевает за звуком")

    def play(self, samples, rate, text=""):
        samples = resample(samples, rate, RATE)
        if not len(samples):
            return
        with self.lock:
            self.samples = samples
            self.cursor = 0
            self.own_stop = bool(OWN_STOP.search(text))
            self.done.clear()
        while not self.cancelled.is_set():
            if not self.stream.active:
                raise RuntimeError("duplex-аудиопоток остановился")
            if self.done.wait(0.01):
                remaining = self.dac_end - self.stream.time
                if remaining <= 0:
                    return
                self.cancelled.wait(min(0.01, remaining))

    def stop_audio(self):
        if self.stream is not None:
            self.stream.abort()
        self.done.set()

    def close(self):
        self.shutdown.set()
        if self.stream is not None:
            self.stream.abort()
            self.stream.close()
            self.stream = None
        if hasattr(self, "worker") and self.worker.is_alive():
            self.worker.join(timeout=1)
        if self.processed:
            log.log(logging.WARNING if self.failed.is_set() else logging.INFO, "%s", self.summary())

    def summary(self):
        """Одна строка для журнала: где тратилось время обработки."""
        line = (
            f"голосовой стоп: кадров {self.processed}, в среднем "
            f"{self.busy_s / self.processed * 1000:.2f} мс, худший {self.worst_s * 1000:.0f} мс "
            f"при бюджете {FRAME / RATE * 1000:.0f}, очередь до {self.deepest} из {QUEUE_FRAMES}"
        )
        if self.detector is not None and self.detector.verifies:
            line += (f", проверок слова {self.detector.verifies}, худшая "
                     f"{self.detector.verify_worst * 1000:.0f} мс")
        return line


def make_voice_stop(cfg, wake=None, model=None):
    # Проверяем AEC до загрузки модели; без него сырой микрофон не слушаем.
    from pywebrtc_audio import AudioProcessor  # noqa: F401
    from .wake import VoskWakeWord

    if model is None:
        source = wake if isinstance(wake, VoskWakeWord) else VoskWakeWord(
            model_dir=cfg.wake.model_dir, models_dir=cfg.models_dir, language=cfg.wake.language,
        )
        model = source.model
    return VoiceStop(model, cfg.voice_stop, cfg.mic.input_device, cfg.tts.output_device)
