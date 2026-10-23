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
import re
import tempfile
import threading
import time
import wave
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlsplit

import numpy as np

from .prompt import FORGOTTEN_REPLY, is_forget

if TYPE_CHECKING:
    from .llm import Assistant

log = logging.getLogger(__name__)

# Больше этого от клиента не принимаем: запись фразы это десятки
# килобайт, а не мегабайты. Заодно защита от случайного мусора.
MAX_BODY_BYTES = 8 * 1024 * 1024

# Сессия — контекст разговора, а не авторизация: X-Token проверяется отдельно.
LEGACY_SESSION = "legacy"
MAX_SESSIONS = 128
SESSION_IDLE_S = 24 * 60 * 60
SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


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
        self._sessions: OrderedDict[str, tuple[Assistant, float]] = OrderedDict()
        # Короткие служебные фразы готовятся здесь, а играет их спутник.
        # Синтеза у него нет и быть не должно, но молчать он тоже не
        # может: без отзыва непонятно, услышал ли он кодовую фразу, а
        # без заглушки пятнадцать секунд поиска выглядят как поломка.
        # Наборы те же, что у голосового цикла на ПК, см. phrases.py.
        self.phrases: dict[str, list[tuple[str, bytes]]] = {}

    def warm_up(self) -> None:
        print("Готовлю модели...")
        if self.stt is not None:
            started = time.perf_counter()
            self.stt.warm_up()
            print(f"  распознавание: {time.perf_counter() - started:.1f} с")
        if self.tts is not None:
            from .phrases import POOLS

            started = time.perf_counter()
            self.tts.warm_up()
            for name, texts in POOLS.items():
                ready = []
                for text in texts:
                    samples = self.tts.synthesize(text)
                    if len(samples):
                        ready.append((text, encode_wav(samples, self.tts.sample_rate)))
                self.phrases[name] = ready
            print(f"  синтез речи: {time.perf_counter() - started:.1f} с")

    def phrases_json(self) -> dict:
        """Все служебные фразы для спутника: текст и звук в base64.

        Одним запросом, а не по файлу на фразу: их полтора десятка, и
        спутник забирает их один раз при запуске.
        """
        return {
            name: [
                {"text": text, "wav": base64.b64encode(blob).decode("ascii")}
                for text, blob in ready
            ]
            for name, ready in self.phrases.items()
        }

    def _session(self, session_id: str) -> Assistant:
        """Вызывать под _lock. Ограничиваем число и срок хранения бесед."""
        now = time.monotonic()
        while self._sessions:
            key, (_, used_at) = next(iter(self._sessions.items()))
            if now - used_at < SESSION_IDLE_S:
                break
            self._sessions.pop(key)
        entry = self._sessions.pop(session_id, None)
        if entry is None:
            if len(self._sessions) >= MAX_SESSIONS:
                self._sessions.popitem(last=False)
            assistant = self.assistant.new_session(session_id)
        else:
            assistant = entry[0]
        self._sessions[session_id] = (assistant, now)
        return assistant

    def answer_audio(self, body: bytes, suffix: str, *, session_id: str = LEGACY_SESSION) -> dict:
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
        return self.answer_text(transcript.text, heard=transcript.text, session_id=session_id)

    def answer_text(
        self, question: str, *, heard: str = "", session_id: str = LEGACY_SESSION,
    ) -> dict:
        # «Забудь» обрабатываем здесь, а не в модели. Спутник текста не
        # видит - он отправляет звук, - поэтому проверить команду может
        # только сервер. Без этого модель отвечала «забыла», ничего при
        # этом не забывая.
        with self._lock:
            assistant = self._session(session_id)
            if is_forget(question):
                assistant.forget()
                return self._spoken(FORGOTTEN_REPLY, heard or question)
            answer = assistant.ask(question)
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

    def _spoken(self, text: str, heard: str) -> dict:
        """Готовый ответ без модели. Вызывается под _lock."""
        result = {"heard": heard, "text": text, "seconds": 0.0, "tools": []}
        if self.tts is not None:
            samples = self.tts.synthesize(text)
            if len(samples):
                result["sample_rate"] = self.tts.sample_rate
                result["audio"] = base64.b64encode(
                    encode_wav(samples, self.tts.sample_rate)
                ).decode("ascii")
        return result

    def forget(self, session_id: str = LEGACY_SESSION) -> None:
        with self._lock:
            self._session(session_id).forget()


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

<button id="talk">Говорить</button>
<div id="diag" class="note"></div>

<div id="out"></div>
<audio id="player" hidden></audio>

<p class="note">
Нажми «Говорить», скажи вопрос, нажми ещё раз. Голос работает только на
защищённом соединении: браузеры не дают микрофон по обычному http, кроме
localhost. Обойти это на своём телефоне можно так: в Chrome открой
chrome://flags, найди «Insecure origins treated as secure», впиши туда
адрес этой страницы вместе с портом и перезапусти браузер. Текстом
работает всегда.
</p>

<script>
const out = document.getElementById("out");
const player = document.getElementById("player");

// Сохраняем ID на этом браузере; getRandomValues работает и по HTTP в LAN.
// Если хранилище закрыто, разговор сохраняется до перезагрузки страницы.
const sessionId = (() => {
  const key = "alisa.session";
  try {
    const saved = localStorage.getItem(key);
    if (/^browser:[a-f0-9]{32}$/.test(saved || "")) return saved;
  } catch (_) {}
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  const id = "browser:" + Array.from(bytes, b => b.toString(16).padStart(2, "0")).join("");
  try { localStorage.setItem(key, id); } catch (_) {}
  return id;
})();

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
    options.headers = { ...options.headers, "X-Session-ID": sessionId };
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
const diag = document.getElementById("diag");
let recorder = null;
let stream = null;
let starting = false;

// Сразу видно, считает ли браузер страницу защищённой. Если флаг в
// chrome://flags не подхватился, это единственный способ понять это,
// не гадая.
diag.textContent = "микрофон: " + (
  !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia
    ? "недоступен в этом браузере"
    : window.isSecureContext
      ? "доступен"
      : "браузер считает страницу незащищённой, разрешения не даст"
);

function idle() {
  recorder = null;
  talk.classList.remove("recording");
  talk.textContent = "Говорить";
}

async function startTalking() {
  if (starting || recorder) return;
  starting = true;
  talk.textContent = "Включаю микрофон...";
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("браузер не даёт микрофон по обычному http");
    }
    if (typeof MediaRecorder === "undefined") {
      throw new Error("в этом браузере нет MediaRecorder");
    }
    // Поток держим открытым между вопросами: разрешение спрашивается
    // один раз, и следующая запись начинается сразу.
    if (!stream) {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    }
    const parts = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (event) => parts.push(event.data);
    recorder.onstop = () => {
      const type = recorder.mimeType || "audio/webm";
      idle();
      send({
        method: "POST",
        headers: { "Content-Type": type },
        body: new Blob(parts, { type }),
      });
    };
    recorder.start();
    talk.classList.add("recording");
    talk.textContent = "Говори... нажми, когда закончишь";
  } catch (error) {
    idle();
    out.textContent =
      "Микрофон не включился: " + (error.message || error.name) +
      ". Спрашивай текстом, либо впиши адрес этой страницы с портом в" +
      " chrome://flags, пункт «Insecure origins treated as secure», и" +
      " перезапусти браузер.";
  } finally {
    starting = false;
  }
}

// Два нажатия, а не удержание. При удержании первый раз поверх кнопки
// встаёт запрос разрешения: он съедает отпускание пальца, и запись
// либо не начиналась, либо не останавливалась — со стороны выглядело
// как «кнопка ничего не делает».
talk.addEventListener("click", () => {
  if (recorder && recorder.state === "recording") recorder.stop();
  else startTalking();
});

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
            if parts.path == "/phrases":
                # Спутник забирает их один раз при запуске и держит в
                # памяти: играть надо мгновенно, а не качать каждый раз.
                self._send_json(200, brain.phrases_json())
                return
            if parts.path in ("/filler.wav", "/ack.wav"):
                # Остались для отладки через браузер и curl: по одной
                # фразе из набора. Спутник ходит в /phrases.
                name = "ack" if parts.path == "/ack.wav" else "search"
                ready = brain.phrases.get(name) or []
                if not ready:
                    self._send(404, b"no audio\n", "text/plain; charset=utf-8")
                    return
                self._send(200, ready[0][1], "audio/wav")
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

            given_session = self.headers.get("X-Session-ID")
            session_id = LEGACY_SESSION if given_session is None else given_session
            if not SESSION_ID.fullmatch(session_id):
                self._send_json(400, {"error": "неверный X-Session-ID: 1–128 символов A-Z, a-z, 0-9, _ . : -"})
                return

            if parts.path == "/forget":
                brain.forget(session_id)
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
                    result = brain.answer_text(question, session_id=session_id)
                else:
                    # Расширение нужно av, чтобы понять формат. Браузер
                    # присылает webm, наш спутник - wav.
                    suffix = (
                        ".wav" if "wav" in kind else
                        ".ogg" if "ogg" in kind or "opus" in kind else ".webm"
                    )
                    result = brain.answer_audio(body, suffix, session_id=session_id)
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
