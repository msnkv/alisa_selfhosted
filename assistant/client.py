"""Спутник: микрофон и динамик здесь, мозги на сервере.

Это вторая половина разделения. Клиент умеет ровно три вещи: услышать
кодовую фразу, записать фразу до конца, проиграть то, что прислал
сервер. Полное распознавание вопроса, модель ответов и синтез — на ПК.
Локально Vosk слушает кодовую фразу и команду «стоп» с эхоподавлением.

Почему кодовая фраза считается на клиенте, а не на сервере: иначе
спутник круглосуточно льёт звук из комнаты по сети. Это и расточительно,
и неприятно. Так же устроены и готовые решения.

Сейчас клиент запускается на том же ПК (`--client`) — так проверяется
сам обмен. Смысл он обретёт на отдельной железке: одноплатник с
USB-микрофоном и колонкой, где запускается этот же файл.

Что для этого ещё надо знать: определение конца фразы в `mic.py` тянет
torch, потому что использует silero через него. На одноплатнике это
лишнее — там надо будет взять onnx-копию silero, которая и так лежит
внутри openwakeword, либо обойтись `EnergyVad`. Распознавание и синтез
на спутник тащить не нужно вовсе.
"""

from __future__ import annotations

import io
import json
import logging
import threading
import time
import urllib.error
import urllib.request
import wave
import uuid

import numpy as np

from .phrases import FILLER_AFTER_S, FILLER_GAP_S, FILLER_LIMIT, NOD_AFTER_S, Pool
from .prompt import is_forget
from .server import encode_wav

log = logging.getLogger(__name__)

EXIT_WORDS = {"выход", "exit", "quit", "q"}

# Запись с микрофона у нас всегда в этой частоте, см. mic.py.
RECORD_RATE = 16000

# Сколько ждать первого звука, прежде чем сказать «не получилось». Сервер
# в худшем случае отвечает примерно за минуту: бюджет агента 35 с,
# запас последнего шага 15 с, плюс распознавание и синтез. Без этого
# предела спутник ждал бы ответа сколько угодно.
ANSWER_WAIT_S = 90.0


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Разбирает WAV от сервера. Своими руками, чтобы не тянуть soundfile."""
    with wave.open(io.BytesIO(data), "rb") as handle:
        rate = handle.getframerate()
        width = handle.getsampwidth()
        channels = handle.getnchannels()
        frames = handle.readframes(handle.getnframes())

    if width != 2:
        raise ValueError(f"ожидал WAV из целых по два байта, пришёл по {width}")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


class Satellite:
    def __init__(self, cfg, url: str, *, microphone, wake=None) -> None:
        self.cfg = cfg
        # Без схемы urllib не работает, а писать её руками каждый раз лень.
        if "://" not in url:
            url = "http://" + url
        self.url = url.rstrip("/")
        self.session_id = cfg.server.session_id or "satellite:" + uuid.uuid4().hex
        self._sound_lock = threading.Lock()
        self._response_cancel = None
        self._voice_stop = None
        self._stop_model = None
        self.microphone = microphone
        self.wake = wake
        # Служебные фразы в голосе ассистента: отзыв на кодовую фразу и
        # заглушки. Синтеза здесь нет, поэтому их готовит сервер, а
        # спутник забирает один раз при запуске и держит в памяти —
        # играть их надо мгновенно. Наборы и правила выбора — phrases.py.
        self.phrases: dict[str, Pool] = {}

    # --- обмен с сервером -------------------------------------------------

    def _post(self, path: str, body: bytes, kind: str) -> dict:
        request = urllib.request.Request(
            self.url + path,
            data=body,
            headers={
                "Content-Type": kind, "X-Token": self.cfg.server.token,
                "X-Session-ID": self.session_id,
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.cfg.server.timeout_s) as answer:
            return json.loads(answer.read().decode("utf-8"))

    def check(self) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=5) as answer:
                answer.read()
        except Exception as exc:  # noqa: BLE001
            print(f"Сервер {self.url} не отвечает: {type(exc).__name__}: {exc}")
            # Подсказываем оба варианта: из cmd файл .ps1 не
            # запускается, а открывается в блокноте.
            print("Запусти его на ПК: alisa.bat --serve (или .\\alisa.ps1 --serve)")
            return False
        print(f"Сервер {self.url} на связи.")
        self.load_phrases()
        return True

    def load_phrases(self) -> None:
        """Забирает у сервера служебные фразы: отзыв и заглушки."""
        import base64

        try:
            with urllib.request.urlopen(self.url + "/phrases", timeout=10) as answer:
                data = json.loads(answer.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.warning("служебные фразы забрать не удалось: %s", exc)
            print("  (служебных фраз не будет: сервер их не отдал)")
            return

        total = 0
        for name, items in data.items():
            ready = []
            for item in items:
                try:
                    samples, rate = decode_wav(base64.b64decode(item["wav"]))
                except Exception as exc:  # noqa: BLE001
                    log.warning("фраза «%s» не разобралась: %s", item.get("text"), exc)
                    continue
                ready.append((item.get("text", ""), samples, rate))
            self.phrases[name] = Pool(ready)
            total += len(ready)
        log.info("служебных фраз получено: %d", total)

    def say_phrase(self, name: str) -> bool:
        """Говорит фразу из набора. False — набора нет, сказать нечем."""
        pool = self.phrases.get(name)
        choice = pool.pick() if pool is not None else None
        if choice is None:
            return False
        text, samples, rate = choice
        print(f"  ({text})")
        try:
            self.play(samples, rate, text=text)
        except Exception as exc:  # noqa: BLE001
            log.warning("фраза не прозвучала: %s", exc)
        return True

    # --- звук -------------------------------------------------------------

    def play(self, samples: np.ndarray, rate: int, *, text: str = "") -> None:
        if self._voice_stop is not None:
            self._voice_stop.play(samples, rate, text)
            return
        import sounddevice as sd

        with self._sound_lock:
            if self._response_cancel is not None and self._response_cancel.is_set():
                return
            sd.play(samples, samplerate=rate, device=self.cfg.tts.output_device)
        sd.wait()

    def stop_playback(self) -> None:
        if self._voice_stop is not None:
            self._voice_stop.stop_audio()
            return
        import sounddevice as sd

        with self._sound_lock:
            sd.stop()

    def _start_voice_stop(self, stopped):
        if self.microphone is None or not self.cfg.voice_stop.enabled:
            return
        controller = None
        try:
            from .voice_stop import make_voice_stop

            controller = make_voice_stop(self.cfg, self.wake, self._stop_model)
            self._stop_model = controller.model
            controller.start(stopped)
            self._voice_stop = controller
            print("  Скажи «стоп», чтобы прервать ответ (эхоподавление включено).")
        except Exception as exc:
            if controller is not None:
                controller.close()
            log.warning("голосовой стоп недоступен: %s: %s", type(exc).__name__, exc)
            print(f"  Голосовой «стоп» недоступен ({type(exc).__name__}: {exc}). "
                  "Обнови requirements-client.txt, проверь модель Vosk и "
                  "аудиоустройства. Остановка: Ctrl+C.")

    def _stream_with_filler(self, body: bytes, cancel_event=None) -> dict:
        """Два WAV в очереди: принимаем следующую фразу, пока звучит текущая.

        Ctrl+C во время ожидания или воспроизведения отменяет только ответ.
        cancel_event позволяет сделать ту же кнопку в другом интерфейсе.
        """
        import base64
        import queue

        stopped = cancel_event if cancel_event is not None else threading.Event()
        self._response_cancel = stopped
        self._start_voice_stop(stopped)
        events = queue.Queue(maxsize=2)
        request_id = uuid.uuid4().hex

        def put(item):
            while not stopped.is_set():
                try:
                    events.put(item, timeout=0.1)
                    return
                except queue.Full:
                    pass

        def worker():
            request = urllib.request.Request(
                self.url + "/ask-stream", data=body,
                headers={"Content-Type": "audio/wav", "X-Token": self.cfg.server.token,
                         "X-Session-ID": self.session_id, "X-Request-ID": request_id},
            )
            try:
                with urllib.request.urlopen(request, timeout=self.cfg.server.timeout_s) as response:
                    while not stopped.is_set():
                        raw = response.readline(4 * 1024 * 1024 + 1)
                        if not raw:
                            break
                        if len(raw) > 4 * 1024 * 1024:
                            raise ValueError("слишком большой фрагмент звука")
                        put(json.loads(raw))
            except Exception as exc:
                put({"exception": exc})
            finally:
                put({"type": "eof"})

        thread = threading.Thread(target=worker, name="stream", daemon=True)
        thread.start()

        def stop_when_requested():
            stopped.wait()
            self.stop_playback()

        stopper = threading.Thread(target=stop_when_requested, name="stop-audio", daemon=True)
        stopper.start()
        started = time.perf_counter()
        completed = False
        sounded = False
        first_sound_s = None
        search = False
        at = NOD_AFTER_S
        nod = False
        fillers = 0
        print("  Ctrl+C — остановить текущий ответ.")
        try:
            while not stopped.is_set():
                try:
                    event = events.get(timeout=0.1)
                except queue.Empty:
                    waited = time.perf_counter() - started
                    if not sounded and waited >= ANSWER_WAIT_S:
                        # Сервер сам укладывается в минуту с небольшим,
                        # так что это уже не «долго думает», а «завис».
                        raise TimeoutError(f"ответа нет {ANSWER_WAIT_S:.0f} с")
                    if not sounded and waited >= at and fillers < FILLER_LIMIT:
                        self.say_phrase("nod" if not nod else "search" if search else "thinking")
                        fillers += nod
                        at = FILLER_AFTER_S if not nod else waited + FILLER_GAP_S
                        nod = True
                    continue
                if "exception" in event:
                    raise event["exception"]
                if stopped.is_set():
                    break
                kind = event.get("type")
                if kind == "search":
                    search = True
                elif kind == "heard":
                    print(f"  ты: {event['text']}")
                elif kind == "sentence":
                    print(f"  {self.cfg.name}: {event['text']}")
                    if event.get("audio"):
                        samples, rate = decode_wav(base64.b64decode(event["audio"]))
                        if not sounded:
                            first_sound_s = time.perf_counter() - started
                            print(f"  [первый звук через {first_sound_s:.2f} с]")
                        sounded = True
                        self.play(samples, rate, text=event["text"])
                elif kind == "done":
                    completed = True
                    event["first_sound_s"] = first_sound_s
                    return event
                elif kind == "error":
                    raise RuntimeError(event["text"])
                elif kind == "eof":
                    raise ConnectionError("сервер оборвал ответ до завершения")
        except KeyboardInterrupt:
            stopped.set()
        finally:
            # Даже при сетевой ошибке останавливаем серверную генерацию.
            stopped.set()
            stopper.join()
            if self._voice_stop is not None:
                self._voice_stop.close()
                if self._voice_stop.failed.is_set():
                    print("  Голосовой «стоп» был отключён: " + self._voice_stop.error)
                    if self._voice_stop.processed:
                        print("  " + self._voice_stop.summary())
                elif getattr(getattr(self._voice_stop, "detector", None), "candidates", 0):
                    # Слышал «стоп», но не засчитал — видно сразу, а не
                    # только в журнале. Подробности по каждому — там.
                    print("  [" + self._voice_stop.detector.tally() + "]")
                self._voice_stop = None
            self._response_cancel = None
            try:
                if not completed:
                    request = urllib.request.Request(
                        self.url + "/cancel", data=b"{}",
                        headers={"X-Token": self.cfg.server.token, "X-Session-ID": self.session_id,
                                 "X-Request-ID": request_id, "Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(request, timeout=3) as response:
                        response.read()
            except Exception as exc:
                log.warning("не удалось подтвердить остановку ответа: %s", type(exc).__name__)
            thread.join(timeout=0.2)
        print("  Ответ остановлен.")
        return {"cancelled": True}

    # --- один круг --------------------------------------------------------

    def ask_once(self, *, wait_speech_s: float | None = None) -> bool:
        """Записать вопрос, отправить, проиграть ответ.

        False означает «речи не услышал» — внешний цикл по этому решает,
        возвращаться ли к ожиданию кодовой фразы.
        """
        print("  слушаю...")
        started = time.perf_counter()
        audio = self.microphone.record_phrase(
            on_speech_start=lambda: print("  говоришь..."),
            wait_speech_s=wait_speech_s,
        )
        record_s = time.perf_counter() - started
        if len(audio) == 0:
            print("  речи не услышал")
            return False

        body = encode_wav(audio, RECORD_RATE)
        started = time.perf_counter()
        try:
            result = self._stream_with_filler(body)
        except urllib.error.HTTPError as exc:
            print(f"  сервер ответил {exc.code}: {exc.read().decode('utf-8', 'replace')}")
            # Вслух, а не только в консоль: человек в другой комнате
            # видит не экран, а тишину, и ждёт ответа зря.
            self.say_phrase("error")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"  не дошло до сервера: {type(exc).__name__}: {exc}")
            self.say_phrase("error")
            return True
        round_trip = time.perf_counter() - started

        if result.get("error"):
            print(f"  ошибка на сервере: {result['error']}")
            self.say_phrase("error")
            return True

        if result.get("cancelled"):
            return False
        # Ответ играется по ходу приёма, поэтому круг включает и речь:
        # задержку честно показывает только первый звук.
        first_sound = result.get("first_sound_s")
        line = (
            f"запись {record_s:.1f} с"
            + (f" | первый звук {first_sound:.1f} с" if first_sound is not None else "")
            + f" | весь ответ {round_trip:.1f} с (модель {result.get('seconds', 0):.1f} с)"
        )
        log.info("круг спутника: %s", line)
        print(f"  [{line}]")
        return True

    def forget(self) -> None:
        try:
            self._post("/forget", b"{}", "application/json")
            print("  разговор забыт")
        except Exception as exc:  # noqa: BLE001
            print(f"  не смог сбросить разговор: {type(exc).__name__}: {exc}")

    # --- режимы -----------------------------------------------------------

    def run(self) -> int:
        """По клавише. Отладочный режим, как и в локальном цикле."""
        if not self.check():
            return 1
        self.microphone.warm_up()
        print("\nНажми Enter и говори. «выход» — закончить.\n")

        while True:
            try:
                command = input("[Enter] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if command in EXIT_WORDS:
                return 0
            if is_forget(command):
                self.forget()
                print()
                continue
            self.ask_once()
            print()

    def run_wake(self) -> int:
        if self.wake is None:
            print("Кодовая фраза выключена в config.yaml")
            return 1
        if not self.check():
            return 1

        self.microphone.warm_up()
        started = time.perf_counter()
        self.wake.warm_up()
        print(f"Кодовая фраза готова за {time.perf_counter() - started:.1f} с")

        spoken = self.wake.spoken
        print(f"\nСпутник ждёт «{spoken}». Ctrl+C — закончить.\n")

        while True:
            try:
                heard = self.wake.wait()
            except KeyboardInterrupt:
                print()
                return 0
            if not heard:
                continue
            # Отзыв вслух, а не только в консоль. Без него из другого
            # конца комнаты непонятно, услышал он кодовую фразу или нет,
            # и человек повторяет её впустую.
            if not self.say_phrase("ack"):
                print("  да?")

            wait = None
            while True:
                try:
                    spoke = self.ask_once(wait_speech_s=wait)
                except KeyboardInterrupt:
                    print()
                    return 0
                except Exception as exc:  # noqa: BLE001
                    print(f"  ошибка: {type(exc).__name__}: {exc}")
                    log.exception("круг спутника упал")
                    break
                if not spoke or self.cfg.wake.follow_up_s <= 0:
                    break
                wait = self.cfg.wake.follow_up_s
                print(f"  (уточняй сразу, без кодовой фразы — слушаю {wait:.0f} с)")

            if self.cfg.wake.cooldown_s > 0:
                time.sleep(self.cfg.wake.cooldown_s)
            print()
