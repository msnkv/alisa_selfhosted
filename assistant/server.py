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
обычный запрос и делает. /ask сохраняет цельный JSON для Telegram и
старых клиентов, /ask-stream передаёт фразы как NDJSON браузеру и
спутнику. /cancel останавливает конкретный запрос в выбранной сессии.

Запуск: `.\\alisa.ps1 --serve`
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import select
import tempfile
import threading
import time
import uuid
import wave
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlsplit

import numpy as np

from . import update
from .prompt import FORGOTTEN_REPLY, is_forget
from .streaming import AnswerCancelled, Requests, check_cancelled
from .tts import split_for_synthesis

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
        self.requests = Requests()
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

        transcript = self.transcribe(body, suffix)
        if not transcript:
            return {"heard": "", "text": "Не разобрала, повтори пожалуйста."}
        return self.answer_text(transcript, heard=transcript, session_id=session_id)

    def _acquire(self, cancelled=None):
        while not self._lock.acquire(timeout=0.1):
            check_cancelled(cancelled)

    def transcribe(self, body: bytes, suffix: str, *, cancelled=None) -> str:
        if self.stt is None:
            raise ValueError("распознавание выключено в config.yaml")
        # Пишем во временный файл и отдаём путь: faster-whisper сам
        # расшифрует формат через av, поэтому с телефона можно присылать
        # что угодно, а не только WAV.
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(body)
            path = Path(handle.name)
        try:
            self._acquire(cancelled)
            try:
                check_cancelled(cancelled)
                transcript = self.stt.transcribe(path)
            finally:
                self._lock.release()
        finally:
            path.unlink(missing_ok=True)

        return transcript.text

    def transcribe_samples(self, samples: np.ndarray) -> str:
        """Готовый звук 16 кГц в текст. Для Wyoming, см. wyoming_server.py."""
        if self.stt is None:
            raise ValueError("распознавание выключено в config.yaml")
        with self._lock:
            return self.stt.transcribe(samples).text

    def synthesize(self, text: str, speaker: str | None = None) -> np.ndarray:
        """Текст в звук под общим замком. Для Wyoming."""
        if self.tts is None:
            raise ValueError("синтез выключен в config.yaml")
        with self._lock:
            return self.tts.synthesize(text, speaker)

    def stream_answer(self, question, *, body, suffix, session_id, cancelled, emit):
        """NDJSON: отдельный WAV на фразу, без сборки всего звука в памяти.

        Обычный ответ разрешаем после финального шага; у сказки инструментов
        нет, поэтому можно передавать фразы ещё во время генерации.
        """
        check_cancelled(cancelled)
        if question is None:
            question = self.transcribe(body, suffix, cancelled=cancelled)
        check_cancelled(cancelled)
        emit({"type": "heard", "text": question})
        # Ожидание занятой модели тоже можно отменить.
        self._acquire(cancelled)
        try:
            check_cancelled(cancelled)
            assistant = self._session(session_id)
            old_mode = assistant.long_form
            pending = []
            committed = False
            spoken_count = 0

            def say(text):
                nonlocal spoken_count
                for part in split_for_synthesis(text):
                    check_cancelled(cancelled)
                    event = {"type": "sentence", "text": part}
                    if self.tts is not None:
                        samples = self.tts.synthesize(part)
                        if len(samples):
                            event["audio"] = base64.b64encode(encode_wav(samples, self.tts.sample_rate)).decode("ascii")
                    check_cancelled(cancelled)
                    emit(event)
                    spoken_count += 1

            def sentence(text):
                check_cancelled(cancelled)
                if committed:
                    say(text)
                else:
                    pending.append(text)

            def step(used_tools):
                nonlocal committed
                if used_tools:
                    pending.clear()
                else:
                    committed = True
                    for text in pending:
                        say(text)
                    pending.clear()

            if not question or is_forget(question):
                if question:
                    assistant.forget()
                text = FORGOTTEN_REPLY if question else "Не разобрала, повтори пожалуйста."
                say(text)
                emit({"type": "done", "text": text, "seconds": 0, "tools": []})
                return
            committed = assistant.will_be_long_form(question)
            try:
                answer = assistant.ask(
                    question, on_sentence=sentence, on_step=step, cancel_event=cancelled,
                    on_tool_call=lambda name, args: emit({"type": "search"}),
                )
                check_cancelled(cancelled)
                if not spoken_count:
                    say(answer.text)
                emit({"type": "done", "text": answer.text, "seconds": answer.seconds, "tools": answer.tools_used})
            except AnswerCancelled:
                assistant.long_form = old_mode
                raise
        finally:
            self._lock.release()

    def answer_text(
        self, question: str, *, heard: str = "", session_id: str = LEGACY_SESSION,
        speak: bool = True,
    ) -> dict:
        """speak=False — только текст: для Home Assistant, он озвучивает сам."""
        # «Забудь» обрабатываем здесь, а не в модели. Спутник текста не
        # видит - он отправляет звук, - поэтому проверить команду может
        # только сервер. Без этого модель отвечала «забыла», ничего при
        # этом не забывая.
        with self._lock:
            assistant = self._session(session_id)
            if is_forget(question):
                assistant.forget()
                return self._spoken(FORGOTTEN_REPLY, heard or question, speak=speak)
            answer = assistant.ask(question)
            audio = None
            if self.tts is not None and speak:
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

    def _spoken(self, text: str, heard: str, *, speak: bool = True) -> dict:
        """Готовый ответ без модели. Вызывается под _lock."""
        result = {"heard": heard, "text": text, "seconds": 0.0, "tools": []}
        if self.tts is not None and speak:
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
<button id="stop" disabled>Стоп</button>
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

let active = null;
const stopButton = document.getElementById("stop");

function busy(value) {
  stopButton.disabled = !value;
  document.getElementById("q").disabled = value;
  document.getElementById("talk").disabled = value;
}

function playPart(run, audio) {
  return new Promise((resolve, reject) => {
    run.release = resolve;
    player.onended = () => { run.release = null; resolve(); };
    player.onerror = () => { run.release = null; reject(new Error("не удалось проиграть звук")); };
    player.src = "data:audio/wav;base64," + audio;
    player.play().catch(reject);
  });
}

async function cancelRun(run) {
  if (run.cancelled) return;
  run.cancelled = true;
  if (active === run) {
    active = null;
    busy(false);
    player.pause();
    player.removeAttribute("src");
    player.onended = null;
    player.onerror = null;
  }
  run.controller.abort();
  if (run.release) { run.release(); run.release = null; }
  try {
    await fetch("/cancel" + location.search, {
      method: "POST", headers: {"X-Session-ID": sessionId, "X-Request-ID": run.id},
    });
  } catch (_) {}
}

stopButton.onclick = () => {
  if (!active) return;
  const run = active;
  out.append(line("Остановлено.", "heard"));
  cancelRun(run);
};

async function send(options) {
  if (active) return;
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  const run = {id: Array.from(bytes, b => b.toString(16).padStart(2, "0")).join(""),
               controller: new AbortController(), release: null};
  active = run;
  busy(true);
  out.textContent = "Думаю...";
  let done = false;
  try {
    options.headers = {...options.headers, "X-Session-ID": sessionId, "X-Request-ID": run.id};
    options.signal = run.controller.signal;
    const response = await fetch("/ask-stream" + location.search, options);
    if (!response.ok) throw new Error("сервер ответил " + response.status);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    out.textContent = "";
    while (active === run && !done) {
      const chunk = await reader.read();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, {stream: true});
      let end;
      while ((end = buffer.indexOf("\\n")) >= 0 && active === run && !done) {
        const event = JSON.parse(buffer.slice(0, end));
        buffer = buffer.slice(end + 1);
        if (event.type === "heard") out.append(line("услышала: " + event.text, "heard"));
        if (event.type === "sentence") {
          out.append(line(event.text, "p"));
          if (event.audio) await playPart(run, event.audio);
        }
        if (event.type === "error") throw new Error(event.text);
        if (event.type === "done") done = true;
      }
    }
    if (!done && active === run) throw new Error("ответ оборвался до завершения");
  } catch (error) {
    if (active === run) out.append(line("Ошибка: " + error.message, "heard"));
  } finally {
    if (!done) await cancelRun(run);
    if (active === run) { active = null; busy(false); }
  }
}

document.getElementById("ask").onsubmit = (event) => {
  event.preventDefault();
  const field = document.getElementById("q");
  const text = field.value.trim();
  if (active || !text) return;
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


# --- OpenAI-совместимый вход для Home Assistant ----------------------------
#
# HA (этап 3) отдаёт текст вопроса «агенту разговора», а тот умеет
# ходить на любой сервер с OpenAI-совместимым API. Так наш агент
# подключается к HA без своей интеграции на Python: снаружи он
# выглядит как модель «alisa», внутри — тот же Assistant.ask со всеми
# правилами, data/ и поиском.
#
# Что от протокола берём, а что нет. HA присылает свою историю и свой
# системный промпт (и, может быть, инструменты для умного дома). Их не
# используем: правила, память диалога и инструменты у нас свои, а
# умный дом владельцу не нужен. Берём только последнюю реплику
# пользователя. Озвучивает ответ сам HA через Wyoming, поэтому здесь
# только текст.

OPENAI_MODEL = "alisa"
OPENAI_SESSION = "ha"


def openai_question(data: dict) -> str:
    """Текст последней реплики пользователя из запроса chat/completions."""
    for message in reversed(data.get("messages") or []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            # Новый формат: список частей, текст в частях типа text.
            content = " ".join(
                str(part.get("text") or "") for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return str(content or "").strip()
    return ""


def openai_session(data: dict, header: str | None) -> str:
    """Сессия HA: заголовок, поле user или общая «ha»."""
    for candidate in (header, data.get("user")):
        if candidate and SESSION_ID.fullmatch(str(candidate)):
            return str(candidate)
    return OPENAI_SESSION


def openai_reply(text: str, model: str) -> dict:
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        # Токены считает DeepSeek, а не мы; нули честнее выдумки.
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def openai_stream(text: str, model: str) -> list[bytes]:
    """Тот же ответ кусками SSE для клиентов, которые просят stream.

    Ответ к этому моменту уже готов целиком: озвучивать по фразам HA
    будет сам, а промежуточный текст шагов с поиском отдавать нельзя —
    его модель потом выбрасывает (см. «Озвучивается только финальный
    шаг» в CLAUDE.md).
    """
    base = {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
    }

    def chunk(delta: dict, finish: str | None) -> bytes:
        body = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        return b"data: " + json.dumps(body, ensure_ascii=False).encode("utf-8") + b"\n\n"

    return [
        chunk({"role": "assistant", "content": ""}, None),
        chunk({"content": text}, None),
        chunk({}, "stop"),
        b"data: [DONE]\n\n",
    ]


def make_handler(brain: Brain, token: str):
    page = PAGE.replace("__NAME__", brain.cfg.name)
    # Код для спутника снимаем один раз при запуске: отдаём ровно то,
    # чем работает сервер, а не то, что успели поправить на диске.
    client_files = update.collect()
    client_sums = update.manifest(client_files)
    client_version = update.version(client_sums)

    class Handler(BaseHTTPRequestHandler):
        server_version = "alisa"
        # Иначе каждый запрос печатает строку в stderr поверх всего.
        def log_message(self, fmt, *args):  # noqa: D401
            log.info("%s - %s", self.address_string(), fmt % args)

        def end_headers(self) -> None:
            # Версия кода спутника в каждом ответе: по ней спутник
            # узнаёт, что сервер перезапустили с новым кодом.
            self.send_header(update.VERSION_HEADER, client_version)
            super().end_headers()

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
            # Клиенты OpenAI шлют ключ так; в HA это поле «API key».
            bearer = self.headers.get("Authorization") or ""
            if not given and bearer.lower().startswith("bearer "):
                given = bearer[7:].strip()
            return given == token

        def _openai_chat(self, body: bytes) -> None:
            try:
                data = json.loads(body)
                if not isinstance(data, dict):
                    raise ValueError("ожидал JSON-объект")
            except ValueError as exc:
                self._send_json(400, {"error": {"message": f"плохой JSON: {exc}"}})
                return
            question = openai_question(data)
            if not question:
                self._send_json(400, {"error": {"message": "нет реплики пользователя"}})
                return
            session_id = openai_session(data, self.headers.get("X-Session-ID"))
            model = str(data.get("model") or OPENAI_MODEL)
            try:
                result = brain.answer_text(question, session_id=session_id, speak=False)
            except Exception as exc:  # noqa: BLE001
                log.exception("запрос из HA упал")
                self._send_json(500, {"error": {"message": f"{type(exc).__name__}: {exc}"}})
                return
            log.info("HA (%s): %r -> %r", session_id, question, result["text"])
            if not data.get("stream"):
                self._send_json(200, openai_reply(result["text"], model))
                return
            parts = openai_stream(result["text"], model)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(sum(len(part) for part in parts)))
            self.end_headers()
            for part in parts:
                self.wfile.write(part)

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
            if parts.path == "/v1/models":
                # HA при подключении проверяет адрес списком моделей.
                if not self._allowed(query):
                    self._send_json(401, {"error": {"message": "нужен токен"}})
                    return
                self._send_json(200, {"object": "list", "data": [{
                    "id": OPENAI_MODEL, "object": "model", "created": 0, "owned_by": "alisa",
                }]})
                return
            if parts.path in ("/client/manifest", "/client/file"):
                # Обновление спутника, см. update.py. Отдаётся только то,
                # что попало в снимок: путь не из списка — 404, так что
                # ни data/, ни .env отсюда не достать.
                if not self._allowed(query):
                    self._send(403, b"token?\n", "text/plain; charset=utf-8")
                    return
                if parts.path == "/client/manifest":
                    self._send_json(200, {"version": client_version, "files": client_sums})
                    return
                name = (query.get("path") or [""])[0]
                if name not in client_files:
                    self._send(404, b"nope\n", "text/plain; charset=utf-8")
                    return
                self._send(200, client_files[name], "application/octet-stream")
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

            if parts.path == "/v1/chat/completions":
                if not self._allowed(query):
                    self._send_json(401, {"error": {"message": "нужен токен"}})
                    return
                length = int(self.headers.get("Content-Length") or 0)
                if not 0 < length <= MAX_BODY_BYTES:
                    self._send_json(400, {"error": {"message": "пустой или слишком большой запрос"}})
                    return
                self._openai_chat(self.rfile.read(length))
                return
            if parts.path not in ("/ask", "/ask-stream", "/cancel", "/forget"):
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

            request_id = self.headers.get("X-Request-ID", "")
            if parts.path in ("/ask-stream", "/cancel") and not SESSION_ID.fullmatch(request_id):
                self._send_json(400, {"error": "нужен корректный X-Request-ID"})
                return
            if parts.path == "/cancel":
                brain.requests.cancel(session_id, request_id)
                self._send_json(200, {"ok": True})
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

            if parts.path == "/ask-stream":
                try:
                    question = None
                    if kind == "application/json":
                        question = str(json.loads(body).get("text") or "").strip()
                        if not question:
                            raise ValueError("нет текста вопроса")
                    cancelled = brain.requests.start(session_id, request_id)
                except (ValueError, AttributeError) as exc:
                    self._send_json(400, {"error": str(exc)})
                    return

                def emit(event):
                    check_cancelled(cancelled)
                    try:
                        payload = memoryview(json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n")
                        deadline = time.monotonic() + brain.cfg.server.timeout_s
                        while payload:
                            check_cancelled(cancelled)
                            # Медленный клиент создаёт обратное давление, но
                            # не блокирует проверку отмены в socket.sendall.
                            if not select.select([], [self.connection], [], 0.1)[1]:
                                if time.monotonic() >= deadline:
                                    raise TimeoutError("клиент перестал принимать ответ")
                                continue
                            try:
                                sent = self.connection.send(payload[:65536])
                            except BlockingIOError:
                                continue
                            if not sent:
                                raise ConnectionError("клиент отключился")
                            payload = payload[sent:]
                            deadline = time.monotonic() + brain.cfg.server.timeout_s
                    except OSError:
                        cancelled.set()
                        raise AnswerCancelled from None

                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.connection.setblocking(False)
                    self.close_connection = True
                    emit({"type": "start"})
                    suffix = ".wav" if "wav" in kind else ".ogg" if "ogg" in kind else ".webm"
                    brain.stream_answer(
                        question, body=body, suffix=suffix, session_id=session_id,
                        cancelled=cancelled, emit=emit,
                    )
                except (AnswerCancelled, OSError):
                    log.info("ответ %s в сессии %s остановлен", request_id, session_id)
                except Exception:
                    log.exception("потоковый запрос упал")
                    try:
                        emit({"type": "error", "text": "Не удалось подготовить ответ. Попробуй ещё раз."})
                    except AnswerCancelled:
                        pass
                finally:
                    brain.requests.finish(session_id, request_id)
                return

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
    print(f"  код для спутника: версия {update.local_version()}")
    if cfg.wyoming.enabled:
        # Импорт здесь: пакет wyoming нужен только когда это включено.
        from . import wyoming_server

        wyoming_server.start(brain, cfg)
        print(
            f"  для Home Assistant: распознавание — порт {cfg.wyoming.stt_port},"
            f" синтез — {cfg.wyoming.tts_port}, агент — http://<ПК>:{cfg.server.port}/v1"
        )
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
