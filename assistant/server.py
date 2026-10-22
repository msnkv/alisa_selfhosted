"""Сервер: ПК отвечает на вопросы, пришедшие по домашней сети.

Зачем это нужно. Цель проекта — колонка в другой комнате, а значит
вопрос приходит откуда-то извне, а распознавание, модель и синтез
остаются здесь, где видеокарта. Это и есть разделение на сервер и
клиента: тяжёлое железо в одном месте, микрофон и динамик в другом.

Важно про сеть: клиент говорит с сервером по **локальной** сети, через
домашний роутер. Провайдер для этого не нужен, и когда интернет
пропадает, спутник по-прежнему достаёт ПК — отваливаются только
DeepSeek и веб-поиск. Подробнее про это в JOURNAL.md, раздел про
офлайн.

Почему обычный HTTP из стандартной библиотеки, а не WebSocket:
один вопрос — это один запрос и один ответ, то есть ровно то, что
обычный запрос и делает. Ни одной новой зависимости, отлаживается
через curl, и браузер умеет это из коробки. Потоковую передачу звука
по сети добавим, если задержка станет мешать: сейчас время уходит в
поиск, а не в передачу.

Запуск: `.\\alisa.ps1 --serve`
"""

from __future__ import annotations

import base64
import io
import json
import logging
import tempfile
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

log = logging.getLogger(__name__)

# Больше этого от клиента не принимаем: запись фразы это десятки
# килобайт, а не мегабайты. Заодно защита от случайного мусора.
MAX_BODY_BYTES = 8 * 1024 * 1024


def encode_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """Переводит float32 от Silero в WAV, который умеет играть всё.

    Пишем целыми числами: так файл вдвое меньше, а на слух разницы нет.
    """
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767).astype("<i2")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


class Brain:
    """Распознавание, модель и синтез вместе, под одним замком.

    Замок нужен, потому что модели не рассчитаны на одновременную
    работу из разных потоков: одна копия whisper, одна копия Silero.
    Для дома это и не нужно — вопросы задают по одному. Зато сервер
    может быть многопоточным и не зависать, пока обрабатывает запрос.
    """

    def __init__(self, cfg, assistant, stt, tts) -> None:
        self.cfg = cfg
        self.assistant = assistant
        self.stt = stt
        self.tts = tts
        self._lock = threading.Lock()

    def warm_up(self) -> None:
        print("Готовлю модели...")
        if self.stt is not None:
            started = time.perf_counter()
            self.stt.warm_up()
            print(f"  распознавание: {time.perf_counter() - started:.1f} с")
        if self.tts is not None:
            started = time.perf_counter()
            self.tts.warm_up()
            print(f"  синтез речи: {time.perf_counter() - started:.1f} с")

    def answer_audio(self, body: bytes, suffix: str) -> dict:
        """Звук с вопросом на входе, ответ текстом и звуком на выходе."""
        if self.stt is None:
            raise ValueError("распознавание выключено в config.yaml")

        # Пишем во временный файл и отдаём путь: faster-whisper сам
        # расшифрует формат через av, поэтому с телефона можно присылать
        # что угодно, а не только WAV.
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(body)
            path = Path(handle.name)
        try:
            with self._lock:
                transcript = self.stt.transcribe(path)
        finally:
            path.unlink(missing_ok=True)

        if not transcript.text:
            return {"heard": "", "text": "Не разобрала, повтори пожалуйста."}
        return self.answer_text(transcript.text, heard=transcript.text)

    def answer_text(self, question: str, *, heard: str = "") -> dict:
        with self._lock:
            answer = self.assistant.ask(question)
            audio = None
            if self.tts is not None:
                samples = self.tts.synthesize(answer.text)
                if len(samples):
                    audio = encode_wav(samples, self.tts.sample_rate)

        result = {
            "heard": heard or question,
            "text": answer.text,
            "seconds": round(answer.seconds, 2),
            "tools": answer.tools_used,
        }
        if audio is not None:
            result["sample_rate"] = self.tts.sample_rate
            result["audio"] = base64.b64encode(audio).decode("ascii")
        return result

    def forget(self) -> None:
        with self._lock:
            self.assistant.forget()


PAGE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__NAME__</title>
<style>
  :root { color-scheme: light dark; }
  body {
    margin: 0; padding: 24px 16px; font: 17px/1.5 system-ui, sans-serif;
    max-width: 34rem; margin-inline: auto;
  }
  h1 { font-size: 1.3rem; margin: 0 0 1rem; }
  button, input {
    font: inherit; width: 100%; box-sizing: border-box;
    padding: 14px; border-radius: 12px; border: 1px solid #8884;
  }
  button { margin-top: 10px; cursor: pointer; }
  /* touch-action: none - иначе палец на кнопке прокручивает страницу,
     и запись обрывается на середине фразы. */
  #talk { padding: 28px; font-size: 1.1rem; touch-action: none; }
  #talk.recording { background: #c0392b; color: white; }
  .note { color: #8889; font-size: 0.85rem; margin-top: 1.5rem; }
  .heard { color: #8889; font-size: 0.9rem; }
  #out { margin-top: 1.2rem; white-space: pre-wrap; }
</style>
</head>
<body>
<h1>__NAME__</h1>

<form id="ask">
  <input id="q" placeholder="Спроси текстом" autocomplete="off">
  <button type="submit">Спросить</button>
</form>

<button id="talk">Нажать и говорить</button>

<div id="out"></div>
<audio id="player" hidden></audio>

<p class="note">
Голос работает только на защищённом соединении: браузеры не дают доступ
к микрофону по обычному http, кроме localhost. С телефона по домашней
сети кнопка скажет об этом прямо. Текстом работает всегда.
</p>

<script>
const out = document.getElementById("out");
const player = document.getElementById("player");

function line(text, cls) {
  const node = document.createElement(cls === "p" ? "p" : "div");
  if (cls !== "p") node.className = cls;
  // textContent, а не innerHTML: это текст от модели, и вставлять его
  // как разметку незачем.
  node.textContent = text;
  return node;
}

function show(data) {
  out.textContent = "";
  if (data.error) { out.textContent = "Ошибка: " + data.error; return; }
  if (data.heard) out.append(line("услышала: " + data.heard, "heard"));
  out.append(line(data.text, "p"));
  if (data.seconds) {
    out.append(line(
      data.seconds + " с" +
        (data.tools && data.tools.length ? ", искала в интернете" : ""),
      "heard"
    ));
  }
  if (data.audio) {
    player.src = "data:audio/wav;base64," + data.audio;
    player.play().catch(() => {});
  }
}

async function send(options) {
  out.textContent = "Думаю...";
  try {
    const response = await fetch("/ask" + location.search, options);
    show(await response.json());
  } catch (error) {
    out.textContent = "Не достучался до сервера: " + error;
  }
}

document.getElementById("ask").onsubmit = (event) => {
  event.preventDefault();
  const field = document.getElementById("q");
  const text = field.value.trim();
  if (!text) return;
  field.value = "";
  send({
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
};

const talk = document.getElementById("talk");
let recorder = null;

async function startTalking() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    out.textContent =
      "Браузер не даёт микрофон по обычному http. Спрашивай текстом," +
      " либо открой страницу с самого ПК по адресу localhost.";
    return;
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (error) {
    out.textContent = "Микрофон недоступен: " + error.name;
    return;
  }
  const parts = [];
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = (event) => parts.push(event.data);
  recorder.onstop = () => {
    stream.getTracks().forEach((track) => track.stop());
    const blob = new Blob(parts, { type: recorder.mimeType });
    send({
      method: "POST",
      headers: { "Content-Type": blob.type || "audio/webm" },
      body: blob,
    });
  };
  recorder.start();
  talk.classList.add("recording");
  talk.textContent = "Говори, отпусти когда закончишь";
}

function stopTalking() {
  talk.classList.remove("recording");
  talk.textContent = "Нажать и говорить";
  if (recorder && recorder.state === "recording") recorder.stop();
  recorder = null;
}

talk.addEventListener("pointerdown", startTalking);
talk.addEventListener("pointerup", stopTalking);
// pointerleave тут намеренно нет: на телефоне палец легко съезжает с
// кнопки, и запись обрывалась бы на середине фразы.
talk.addEventListener("pointercancel", stopTalking);

// Экран не гаснет, пока страница открыта. Нужно, если телефон живёт
// в розетке и работает колонкой. Работает только на защищённом
// соединении, поэтому молча пропускаем, если нельзя.
async function keepAwake() {
  try {
    await navigator.wakeLock.request("screen");
  } catch (error) {
    /* не вышло - значит экран гасится как обычно */
  }
}
keepAwake();
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") keepAwake();
});
</script>
</body>
</html>
"""


def make_handler(brain: Brain, token: str):
    page = PAGE.replace("__NAME__", brain.cfg.name)

    class Handler(BaseHTTPRequestHandler):
        server_version = "alisa"
        # Иначе каждый запрос печатает строку в stderr поверх всего.
        def log_message(self, fmt, *args):  # noqa: D401
            log.info("%s - %s", self.address_string(), fmt % args)

        # --- мелочи -------------------------------------------------------

        def _send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, code: int, data: dict) -> None:
            self._send(
                code,
                json.dumps(data, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def _allowed(self, query: dict) -> bool:
            if not token:
                return True
            given = self.headers.get("X-Token") or (query.get("token") or [""])[0]
            return given == token

        # --- маршруты -----------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802
            parts = urlsplit(self.path)
            query = parse_qs(parts.query)

            if parts.path == "/health":
                self._send(200, b"ok\n", "text/plain; charset=utf-8")
                return
            if parts.path in ("/", "/index.html"):
                if not self._allowed(query):
                    self._send(403, b"token?\n", "text/plain; charset=utf-8")
                    return
                self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
                return
            self._send(404, b"nope\n", "text/plain; charset=utf-8")

        def do_POST(self) -> None:  # noqa: N802
            parts = urlsplit(self.path)
            query = parse_qs(parts.query)

            if parts.path not in ("/ask", "/forget"):
                self._send(404, b"nope\n", "text/plain; charset=utf-8")
                return
            if not self._allowed(query):
                self._send_json(403, {"error": "нужен токен"})
                return

            if parts.path == "/forget":
                brain.forget()
                self._send_json(200, {"ok": True})
                return

            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                self._send_json(400, {"error": "пустой запрос"})
                return
            if length > MAX_BODY_BYTES:
                self._send_json(413, {"error": "слишком большой запрос"})
                return

            body = self.rfile.read(length)
            kind = (self.headers.get("Content-Type") or "").split(";")[0].strip()

            try:
                if kind == "application/json":
                    question = (json.loads(body).get("text") or "").strip()
                    if not question:
                        self._send_json(400, {"error": "нет текста вопроса"})
                        return
                    result = brain.answer_text(question)
                else:
                    # Расширение нужно av, чтобы понять формат. Браузер
                    # присылает webm, наш спутник - wav.
                    suffix = ".wav" if "wav" in kind else ".webm"
                    result = brain.answer_audio(body, suffix)
            except Exception as exc:  # noqa: BLE001
                log.exception("запрос упал")
                self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
                return

            self._send_json(200, result)

    return Handler


def serve(cfg, assistant, stt, tts) -> int:
    brain = Brain(cfg, assistant, stt, tts)
    brain.warm_up()

    handler = make_handler(brain, cfg.server.token)
    httpd = ThreadingHTTPServer((cfg.server.host, cfg.server.port), handler)

    print(f"\n{cfg.name} слушает сеть на порту {cfg.server.port}.")
    print(f"  с этого ПК:     http://localhost:{cfg.server.port}/")
    print(f"  с телефона:     http://<адрес этого ПК>:{cfg.server.port}/")
    print("  адрес ПК покажет команда ipconfig")
    if not cfg.server.token:
        # Говорим прямо: в data/ лежат гараж, рецепты и бюджет, и любой
        # в домашней сети может про них спросить. Это осознанный выбор,
        # но он должен быть осознанным.
        print(
            "\n  ВНИМАНИЕ: токен не задан, спросить может любой в домашней"
            "\n  сети, в том числе про твои личные файлы. Включается так:"
            "\n  server.token в config.yaml, потом ?token=... в адресе."
        )
    print("\nCtrl+C — закончить.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        httpd.server_close()
    return 0
