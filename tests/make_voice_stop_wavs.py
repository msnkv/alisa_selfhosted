"""Записи для test_voice_stop_speech.py нашим же Silero, без macOS.

Запуск из корня проекта:  .venv\\Scripts\\python.exe tests\\make_voice_stop_wavs.py

Кладёт в models/voice-stop-check (папка в .gitignore):
  alisa.wav     48 кГц, голос Алисы (tts.speaker), история со словом «Стоп!»;
  stop.wav      16 кГц, «Стоп.» другим голосом — это человек в комнате;
  stop_<голос>.wav  то же всеми остальными голосами, для замера;
  stop_twice.wav  «Стоп, стоп.» — так говорят после промаха;
  negative.wav  16 кГц, похожие слова без команды.
Голос человека намеренно не совпадает с голосом Алисы: иначе AEC и
распознавание проверяются в слишком удобных условиях.
"""

from pathlib import Path
import sys
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assistant.config import load_config  # noqa: E402
from assistant.tts import make_tts  # noqa: E402
from assistant.voice_stop import resample  # noqa: E402

ALISA = ("Сейчас я расскажу тебе историю про кота, который жил в маленьком доме у леса. "
         "Однажды он услышал громкий голос. Стоп! Подожди меня, сказала девочка. "
         "Они вместе пошли дальше по тропинке.")
NEGATIVE = ("Где лежит мой телефон? Сколько сейчас времени? Пожалуйста, продолжай рассказ. "
            "Стол. Сто. Суп. Потоп. Кто там? Что случилось? Не останавливайся. "
            "Штопор. Стопка. Сток. Стой. Постой.")
VOICES = ("aidar", "eugene", "kseniya", "xenia", "baya")


def save(path, samples, rate):
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def to16k(samples, rate):
    """48 -> 16 кГц прореживанием после усреднения: для речи достаточно."""
    if rate == 16000:
        return samples
    step = rate // 16000
    trimmed = samples[: len(samples) // step * step]
    return trimmed.reshape(-1, step).mean(axis=1)


def main():
    cfg = load_config()
    tts = make_tts(cfg)
    out = ROOT / "models" / "voice-stop-check"
    out.mkdir(parents=True, exist_ok=True)
    rate = tts.sample_rate
    alisa = cfg.tts.speaker
    save(out / "alisa.wav", resample(tts.synthesize(ALISA, alisa), rate, 48000), 48000)
    human = next(v for v in VOICES if v != alisa)
    for voice in VOICES:
        save(out / f"stop_{voice}.wav", to16k(tts.synthesize("Стоп.", voice), rate), 16000)
    save(out / "stop.wav", to16k(tts.synthesize("Стоп.", human), rate), 16000)
    save(out / "stop_twice.wav", to16k(tts.synthesize("Стоп, стоп.", human), rate), 16000)
    save(out / "negative.wav", to16k(tts.synthesize(NEGATIVE, human), rate), 16000)
    print(f"готово: {out}, Алиса — {alisa}, человек — {human}")


if __name__ == "__main__":
    main()
