"""Спутник: микрофон и динамик здесь, мозги на сервере.

Это вторая половина разделения. Клиент умеет ровно три вещи: услышать
кодовую фразу, записать фразу до конца, проиграть то, что прислал
сервер. Распознавания, модели и синтеза тут нет вообще — они на ПК.

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

import numpy as np

from .prompt import is_forget
from .server import encode_wav

log = logging.getLogger(__name__)

EXIT_WORDS = {"выход", "exit", "quit", "q"}

# Запись с микрофона у нас всегда в этой частоте, см. mic.py.
RECORD_RATE = 16000

# Столько тишины человек терпит спокойно; дальше надо подать голос.
FILLER_AFTER_S = 1.5
# И повторять, пока идёт поиск: вопрос с несколькими попытками занимает
# до полуминуты, и молчание всё это время выглядит как поломка.
FILLER_GAP_S = 8.0


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
        self.microphone = microphone
        self.wake = wake
        # Фраза-заглушка в голосе ассистента. Синтеза здесь нет, поэтому
        # её готовит сервер, а спутник забирает один раз при запуске.
        self.filler = None
        self.filler_rate = 0

    # --- обмен с сервером -------------------------------------------------

    def _post(self, path: str, body: bytes, kind: str) -> dict:
        request = urllib.request.Request(
            self.url + path,
            data=body,
            headers={"Content-Type": kind, "X-Token": self.cfg.server.token},
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
            print("Запусти его на ПК командой .\\alisa.ps1 --serve")
            return False
        print(f"Сервер {self.url} на связи.")
        self.load_filler()
        return True

    def load_filler(self) -> None:
        """Забирает у сервера фразу «секунду, смотрю»."""
        try:
            with urllib.request.urlopen(self.url + "/filler.wav", timeout=10) as answer:
                self.filler, self.filler_rate = decode_wav(answer.read())
        except Exception as exc:  # noqa: BLE001
            log.warning("заглушку забрать не удалось: %s", exc)
            print("  (заглушки не будет: сервер её не отдал)")

    # --- звук -------------------------------------------------------------

    def play(self, samples: np.ndarray, rate: int) -> None:
        import sounddevice as sd

        sd.play(samples, samplerate=rate, device=self.cfg.tts.output_device)
        sd.wait()

    def _post_with_filler(self, body: bytes) -> dict:
        """Отправляет вопрос и подаёт голос, пока сервер думает.

        Без этого после «договорил» идёт полная тишина — на вопросе с
        поиском это пятнадцать секунд, и человек не понимает, живо ли
        оно вообще. Запрос уходит в отдельный поток, а главный ждёт его
        и в паузах играет заглушку.
        """
        outcome: dict = {}

        def worker() -> None:
            try:
                outcome["data"] = self._post("/ask", body, "audio/wav")
            except Exception as exc:  # noqa: BLE001
                outcome["error"] = exc

        thread = threading.Thread(target=worker, name="ask", daemon=True)
        thread.start()

        waited = FILLER_AFTER_S
        while True:
            thread.join(waited)
            if not thread.is_alive():
                break
            if self.filler is None:
                # Заглушки нет - просто ждём дальше, но молча.
                waited = FILLER_GAP_S
                continue
            self.play(self.filler, self.filler_rate)
            waited = FILLER_GAP_S

        if "error" in outcome:
            raise outcome["error"]
        return outcome["data"]

    def _say_server_answer(self, result: dict) -> None:
        if result.get("heard"):
            print(f"  ты: {result['heard']}")
        print(f"  {self.cfg.name}: {result.get('text', '')}")

        encoded = result.get("audio")
        if not encoded:
            print("  (сервер прислал только текст, синтез у него выключен)")
            return
        import base64

        samples, rate = decode_wav(base64.b64decode(encoded))
        self.play(samples, rate)

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
            result = self._post_with_filler(body)
        except urllib.error.HTTPError as exc:
            print(f"  сервер ответил {exc.code}: {exc.read().decode('utf-8', 'replace')}")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"  не дошло до сервера: {type(exc).__name__}: {exc}")
            return True
        round_trip = time.perf_counter() - started

        if result.get("error"):
            print(f"  ошибка на сервере: {result['error']}")
            return True

        self._say_server_answer(result)
        line = (
            f"запись {record_s:.1f} с | сервер {round_trip:.1f} с"
            f" (из них модель {result.get('seconds', 0):.1f} с)"
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

        spoken = self.cfg.wake.model.replace("_", " ")
        print(f"\nСпутник ждёт «{spoken}». Ctrl+C — закончить.\n")

        while True:
            try:
                heard = self.wake.wait()
            except KeyboardInterrupt:
                print()
                return 0
            if not heard:
                continue
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
