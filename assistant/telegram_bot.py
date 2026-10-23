"""Telegram как текстовый и голосовой клиент сервера Алисы.

Запуск: python -m assistant.telegram_bot. Модели здесь не загружаются.
Адрес и токен сервера берутся из config.yaml + config.local.yaml,
TELEGRAM_BOT_TOKEN и TELEGRAM_ALLOWED_USER_ID — из .env.
"""

from __future__ import annotations

import base64
import http.client
import io
import json
import os
import time
import urllib.error
import urllib.request
import uuid

from .config import load_config

MAX_INPUT_BYTES = 8 * 1024 * 1024  # лимит /ask на сервере


def wav_to_voice(blob: bytes) -> bytes:
    """WAV от Silero -> OGG/Opus для sendVoice, без установки ffmpeg."""
    import av

    buffer = io.BytesIO()
    with av.open(io.BytesIO(blob)) as source, av.open(buffer, "w", format="ogg") as target:
        stream = target.add_stream("libopus", rate=48000)
        stream.layout = "mono"
        stream.bit_rate = 32000
        resampler = av.AudioResampler(format="flt", layout="mono", rate=48000)
        for frame in source.decode(audio=0):
            for ready in resampler.resample(frame):
                for packet in stream.encode(ready):
                    target.mux(packet)
        for ready in resampler.resample(None):
            for packet in stream.encode(ready):
                target.mux(packet)
        for packet in stream.encode(None):
            target.mux(packet)
    return buffer.getvalue()


class TelegramError(RuntimeError):
    def __init__(self, code: int, description: str, retry_after: int = 0):
        super().__init__(f"Telegram {code}: {description}")
        self.code = code
        self.retry_after = retry_after


class Telegram:
    def __init__(self, token: str):
        self.token = token
        self.api_url = f"https://api.telegram.org/bot{token}/"
        self.file_url = f"https://api.telegram.org/file/bot{token}/"

    def call(self, method: str, params: dict, *, voice: bytes | None = None):
        # 429 означает, что запрос отклонён, поэтому повтор безопасен.
        # После таймаута отправку не повторяем: сообщение могло дойти.
        for attempt in range(3):
            try:
                return self._call(method, params, voice=voice)
            except TelegramError as exc:
                if exc.code != 429 or attempt == 2 or exc.retry_after > 60:
                    raise
                time.sleep(max(1, exc.retry_after))

    def _call(self, method: str, params: dict, *, voice: bytes | None = None):
        if voice is None:
            body = json.dumps(params).encode("utf-8")
            kind = "application/json"
        else:
            boundary = uuid.uuid4().hex
            parts = []
            for key, value in params.items():
                parts.append(
                    f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"'
                    f"\r\n\r\n{value}\r\n".encode("utf-8")
                )
            parts.extend([
                f'--{boundary}\r\nContent-Disposition: form-data; name="voice"; '
                'filename="answer.ogg"\r\nContent-Type: audio/ogg\r\n\r\n'.encode(),
                voice,
                f"\r\n--{boundary}--\r\n".encode(),
            ])
            body = b"".join(parts)
            kind = f"multipart/form-data; boundary={boundary}"
        request = urllib.request.Request(
            self.api_url + method, data=body, headers={"Content-Type": kind}
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            # HTTPError содержит URL с секретным токеном. Его не печатаем.
            try:
                result = json.loads(exc.read())
            except (ValueError, UnicodeError):
                raise TelegramError(exc.code, "ошибка HTTP") from None
        except (OSError, http.client.HTTPException):
            raise TelegramError(0, "нет связи с Telegram") from None
        except (ValueError, UnicodeError):
            raise TelegramError(0, "Telegram вернул некорректный ответ") from None
        if not result.get("ok"):
            raise TelegramError(
                result.get("error_code", 0),
                str(result.get("description", "ошибка API")).replace(self.token, "<token>"),
                result.get("parameters", {}).get("retry_after", 0),
            )
        return result["result"]

    def download_voice(self, voice: dict) -> bytes:
        if voice.get("file_size", 0) > MAX_INPUT_BYTES:
            raise ValueError("Голосовое больше восьми мегабайт. Пришли запись покороче.")
        info = self.call("getFile", {"file_id": voice["file_id"]})
        try:
            with urllib.request.urlopen(self.file_url + info["file_path"], timeout=60) as response:
                blob = response.read(MAX_INPUT_BYTES + 1)
        except (OSError, http.client.HTTPException):
            raise TelegramError(0, "не удалось скачать голосовое") from None
        if len(blob) > MAX_INPUT_BYTES:
            raise ValueError("Голосовое больше восьми мегабайт. Пришли запись покороче.")
        return blob

    def text(self, chat_id: int, text: str) -> None:
        # Сказки могут быть длиннее лимита Telegram. 2000 символов вмещают
        # даже текст с эмодзи, которые занимают по две единицы UTF-16.
        for start in range(0, len(text), 2000):
            self.call("sendMessage", {"chat_id": chat_id, "text": text[start:start + 2000]})


class AliceServer:
    def __init__(self, cfg):
        self.url = cfg.url.rstrip("/")
        if "://" not in self.url:
            self.url = "http://" + self.url
        self.token = cfg.token
        self.timeout = cfg.timeout_s

    def ask(self, body: bytes, kind: str) -> dict:
        request = urllib.request.Request(
            self.url + "/ask", data=body,
            headers={"Content-Type": kind, "X-Token": self.token},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        if result.get("error"):
            raise ValueError("Сервер не смог обработать вопрос.")
        return result


def handle_message(message: dict, telegram: Telegram, alice: AliceServer, owner: int) -> None:
    if message.get("chat", {}).get("type") != "private":
        return
    chat_id = message["chat"]["id"]
    user_id = message.get("from", {}).get("id")
    text = (message.get("text") or "").strip()
    command = text.split(maxsplit=1)[0].split("@")[0] if text else ""
    if user_id != owner:
        if command == "/start":
            telegram.text(chat_id, f"Твой Telegram ID: {user_id}. Доступ к Алисе пока не настроен.")
        return
    if command in ("/start", "/help"):
        telegram.text(chat_id, "Пришли текст или голосовое. Верну ответ Алисы текстом и войсом. "
                      "Чтобы сбросить разговор, напиши «забудь».")
        return
    try:
        if message.get("voice"):
            body = telegram.download_voice(message["voice"])
            kind = "audio/ogg"
        elif text:
            body = json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")
            kind = "application/json"
        else:
            telegram.text(chat_id, "Пришли текст или голосовое сообщение.")
            return
    except ValueError as exc:
        telegram.text(chat_id, str(exc))
        return
    except TelegramError:
        telegram.text(chat_id, "Не удалось скачать голосовое из Telegram. Попробуй ещё раз.")
        return

    try:
        result = alice.ask(body, kind)
    except (OSError, ValueError, http.client.HTTPException):
        telegram.text(chat_id, "Сервер Алисы недоступен или не смог ответить. Попробуй позже.")
        return
    answer = result.get("text") or "Сервер прислал пустой ответ."
    telegram.text(chat_id, answer)
    if not result.get("audio"):
        telegram.text(chat_id, "Сервер прислал только текст. Проверь, что на нём включена озвучка.")
        return
    try:
        voice = wav_to_voice(base64.b64decode(result["audio"], validate=True))
    except Exception as exc:  # текст уже отправлен, о сбое звука сообщаем отдельно
        print(f"Не удалось преобразовать звук: {type(exc).__name__}")
        telegram.text(chat_id, "Текст готов, но не удалось подготовить войс.")
        return
    telegram.call("sendVoice", {"chat_id": chat_id}, voice=voice)


def main() -> int:
    cfg = load_config()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("Добавь TELEGRAM_BOT_TOKEN в .env (токен от @BotFather).")
        return 1
    try:
        owner = int(os.getenv("TELEGRAM_ALLOWED_USER_ID", "").strip() or "0")
        if owner < 0:
            raise ValueError
    except ValueError:
        print("TELEGRAM_ALLOWED_USER_ID должен быть положительным числом.")
        return 1
    telegram = Telegram(token)
    alice = AliceServer(cfg.server)
    try:
        me = telegram.call("getMe", {})
        if telegram.call("getWebhookInfo", {}).get("url"):
            print("У бота включён webhook. Используй отдельного бота или отключи webhook.")
            return 1
        print(f"Бот @{me['username']} запущен. Ctrl+C — остановить.")
        if not owner:
            print("Напиши боту /start, добавь показанный ID в TELEGRAM_ALLOWED_USER_ID в .env "
                  "и перезапусти. До этого вопросы к Алисе закрыты.")
        offset = 0
        while True:
            try:
                updates = telegram.call("getUpdates", {
                    "offset": offset, "timeout": 30, "allowed_updates": ["message"],
                })
                for update in updates:
                    try:
                        if "message" in update:
                            handle_message(update["message"], telegram, alice, owner)
                    except TelegramError as exc:
                        print(exc)
                    finally:
                        # Не повторяем запрос к модели при ошибке доставки в Telegram.
                        offset = update["update_id"] + 1
            except TelegramError as exc:
                if exc.code in (401, 404, 409):
                    print(f"{exc}. Проверь токен и что запущена только одна копия бота.")
                    return 1
                print(exc)
                time.sleep(min(30, max(3, exc.retry_after)))
    except TelegramError as exc:
        print(exc)
        return 1
    except KeyboardInterrupt:
        print("\nБот остановлен.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
