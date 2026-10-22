"""Текстовый режим: вопрос с клавиатуры, ответ в консоль или голосом.

Нужен для отладки без микрофона. Запуск:
    .\\alisa.ps1
    .\\alisa.ps1 --selftest
    .\\alisa.ps1 --ask "где лежат болгарки?"
    .\\alisa.ps1 --ask "где лежат болгарки?" --speak
    .\\alisa.ps1 --voices
    .\\alisa.ps1 --devices
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace

from .config import ROOT, load_config, setup_logging
from .llm import Answer, Assistant
from .mic import make_microphone
from .prompt import build_system_prompt, human_now
from .stt import make_stt
from .tts import SAMPLE_TEXT, SPEAKERS, SpeechQueue, describe_devices, make_tts
from .voice import VoiceLoop
from .wake import make_wake_word

EXIT_WORDS = {"выход", "exit", "quit", "q"}
# Сменить тему, не перезапуская ассистента.
FORGET_WORDS = {"забудь", "сброс", "новая тема"}
SAMPLES_DIR = ROOT / "samples"


def _force_utf8() -> None:
    """Windows-консоль часто не в UTF-8, и кириллица падает с ошибкой."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _stats(answer: Answer) -> str:
    parts = [f"{answer.seconds:.2f} с"]
    if answer.tools_used:
        parts.append("инструменты: " + ", ".join(answer.tools_used))
    parts.append(f"вход {answer.tokens_in} (из кэша {answer.tokens_cached})")
    parts.append(f"выход {answer.tokens_out}")
    return "  [" + ", ".join(parts) + "]"


def _announce_tool(name: str, arguments: dict) -> None:
    """Печатает, что ассистент пошёл искать.

    В голосовом режиме (шаг 5) сюда встанет фраза-заглушка «секунду, смотрю».
    """
    detail = arguments.get("query") or arguments.get("url") or ""
    print(f"  ... {name}: {detail}")


def selftest(assistant: Assistant, tts) -> int:
    """Проверяет, что всё на месте, и делает живые запросы."""
    cfg = assistant.cfg
    ok = True

    print("Проверка окружения")
    print(f"  ассистент:   {cfg.name}, данные {cfg.owner}")
    print(f"  модель:      {cfg.llm.model} ({cfg.llm.base_url})")
    print(f"  размышления: {'включены' if cfg.llm.thinking else 'выключены'}")
    print(f"  ответ:       до {cfg.max_sentences} предложений, ~{cfg.max_words} слов")
    print(f"  сейчас:      {human_now()}")

    if assistant.tools is None:
        print("  интернет:    выключен в config.yaml")
    else:
        print(
            f"  интернет:    {cfg.search.backend}"
            f" (движок {cfg.search.engine}, регион {cfg.search.region}),"
            f" до {cfg.agent.max_steps} шагов, бюджет {cfg.agent.total_timeout_s:.0f} с"
        )

    if tts is None:
        print("  озвучка:     выключена в config.yaml")
    else:
        print(
            f"  озвучка:     {cfg.tts.engine} {cfg.tts.model},"
            f" голос {cfg.tts.speaker}, {cfg.tts.sample_rate} Гц, {cfg.tts.device}"
        )

    if cfg.wake.enabled:
        print(
            f"  кодовая фраза: {cfg.wake.model} (порог {cfg.wake.threshold},"
            f" фильтр речи {cfg.wake.vad_threshold})"
        )
    else:
        print("  кодовая фраза: выключена в config.yaml")

    if not cfg.stt.enabled:
        print("  слух:        выключен в config.yaml")
    else:
        print(
            f"  слух:        {cfg.stt.engine} {cfg.stt.model},"
            f" {cfg.stt.device}/{cfg.stt.compute_type}, язык {cfg.stt.language}"
        )

    key = cfg.llm.api_key
    if key:
        print(f"  ключ API:    найден, длина {len(key)} символов")
    else:
        print("  ключ API:    НЕ НАЙДЕН — вставь его в .env (DEEPSEEK_API_KEY=...)")
        ok = False

    files = assistant.knowledge.files()
    if files:
        print(f"  файлы в {cfg.data_dir.name}/:")
        for path in files:
            print(f"    {path.name}: {path.stat().st_size} байт")
    else:
        print(f"  файлы в {cfg.data_dir.name}/: НЕ НАЙДЕНЫ")
        ok = False

    prompt_len = len(
        build_system_prompt(
            name=cfg.name,
            owner=cfg.owner,
            max_sentences=cfg.max_sentences,
            max_words=cfg.max_words,
            knowledge=assistant.knowledge.text(),
            tools_available=assistant.tools is not None,
        )
    )
    print(f"  системный промпт: {prompt_len} символов")

    if not ok:
        print("\nЕсть проблемы, дальше не проверяю.")
        return 1

    print("\nПробую запрос к модели...")
    try:
        answer = assistant.ask("Ответь одним словом: готово")
    except Exception as exc:  # noqa: BLE001 — показываем пользователю как есть
        print(f"  ОШИБКА: {type(exc).__name__}: {exc}")
        return 1
    print(f"  ответ: {answer.text}")
    print(_stats(answer))

    if assistant.tools is not None:
        print("\nПробую поиск в интернете...")
        try:
            found = assistant.tools.web_search("проверка связи")
        except Exception as exc:  # noqa: BLE001
            print(f"  ОШИБКА: {type(exc).__name__}: {exc}")
            return 1
        print(f"  {found.splitlines()[0] if found else '(пусто)'}")
        if "ничего не вернул" in found:
            print("  поиск не отвечает — попробуй позже или смени engine в config.yaml")
            return 1

    if tts is not None:
        print("\nПробую озвучку (первый раз качает модель, это небыстро)...")
        try:
            elapsed = tts.warm_up()
        except Exception as exc:  # noqa: BLE001
            print(f"  ОШИБКА: {type(exc).__name__}: {exc}")
            return 1
        print(f"  модель загружена и синтез работает, {elapsed:.1f} с на прогрев")

    if cfg.stt.enabled:
        print("\nПробую распознавание (первый раз качает модель, это небыстро)...")
        stt = make_stt(cfg)
        try:
            elapsed = stt.warm_up()
        except Exception as exc:  # noqa: BLE001
            print(f"  ОШИБКА: {type(exc).__name__}: {exc}")
            return 1
        print(
            f"  работает на {stt.actual_device}/{stt.actual_compute_type},"
            f" {elapsed:.1f} с на прогрев"
        )
        if stt.actual_device != cfg.stt.device:
            print(
                f"  ВНИМАНИЕ: просили {cfg.stt.device}, получили"
                f" {stt.actual_device} — подробности в логе"
            )

    print("\nВсё работает.")
    return 0


def list_devices() -> int:
    print("Аудиоустройства:\n")
    print(describe_devices())
    print(
        "\nНомер нужного устройства впиши в config.yaml:"
        "\n  tts.output_device - куда играть"
    )
    return 0


def check_microphone(cfg) -> int:
    """Отдельная проверка микрофона, без распознавания.

    Если тут нули, дело в устройстве или в системных настройках, и лезть
    в распознавание бессмысленно.
    """
    microphone = make_microphone(cfg)
    print("Скажи что-нибудь, слушаю три секунды...")
    rms, peak = microphone.measure_level(3.0)
    print(f"  средняя громкость: {rms:.4f}")
    print(f"  максимум:          {peak:.4f}")

    if peak < 0.001:
        print("\nМикрофон молчит. Проверь, что он не выключен в системе")
        print("и что выбрано нужное устройство (--devices, потом mic.input_device).")
        return 1
    if peak < 0.02:
        print("\nСлышно очень тихо — подними громкость микрофона в системе.")
        return 1
    print("\nМикрофон слышит нормально.")
    return 0


def render_voices(tts, text: str) -> int:
    """Озвучивает одну фразу всеми голосами, сохраняет и проигрывает.

    Это замена абстрактному сравнению движков: слушаешь одну и ту же
    фразу пятью голосами на своих наушниках и выбираешь.
    """
    if tts is None:
        print("Озвучка выключена в config.yaml")
        return 1

    print(f"Фраза: {text}\n")
    for speaker in SPEAKERS:
        path = SAMPLES_DIR / f"{speaker}.wav"
        try:
            tts.save(text, path, speaker=speaker)
        except Exception as exc:  # noqa: BLE001
            print(f"  {speaker}: ОШИБКА {type(exc).__name__}: {exc}")
            continue
        print(f"  {speaker}: {path}")
        tts.say(text, speaker=speaker)

    print(f"\nФайлы лежат в {SAMPLES_DIR}")
    print("Понравившийся голос впиши в config.yaml, tts.speaker")
    return 0


def _respond(assistant: Assistant, tts, question: str) -> Answer | None:
    # С озвучкой говорим потоком, как в голосовом режиме: так тут слышно
    # ровно то, что услышишь из колонки, и видно замер до первого звука.
    speech = None
    if tts is not None and assistant.cfg.tts.stream:
        speech = SpeechQueue(tts)
        speech.start()

    started = time.perf_counter()
    try:
        answer = assistant.ask(
            question,
            on_tool_call=_announce_tool,
            on_sentence=speech.add if speech is not None else None,
        )
    except Exception as exc:  # noqa: BLE001
        if speech is not None:
            speech.stop()
        print(f"  ошибка: {type(exc).__name__}: {exc}")
        return None

    print(f"\n{answer.text}")
    print(_stats(answer))

    if speech is not None:
        if speech.first_sound_at is not None:
            print(f"  [до первого звука {speech.first_sound_at - started:.2f} с]")
        speech.close()
        speech.wait()
    elif tts is not None:
        try:
            tts.say(answer.text)
        except Exception as exc:  # noqa: BLE001
            print(f"  озвучка не удалась: {type(exc).__name__}: {exc}")
    return answer


def repl(assistant: Assistant, tts) -> int:
    print(f"{assistant.cfg.name}: текстовый режим. Пустая строка или «выход» — закончить.\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not question or question.lower() in EXIT_WORDS:
            return 0

        if question.lower() in FORGET_WORDS:
            assistant.forget()
            print("  разговор забыт")
            print()
            continue

        _respond(assistant, tts, question)
        print()


def main() -> int:
    _force_utf8()

    parser = argparse.ArgumentParser(description="Текстовый режим голосового помощника")
    parser.add_argument(
        "--selftest", action="store_true", help="проверить окружение, API, поиск и озвучку"
    )
    parser.add_argument("--ask", metavar="ВОПРОС", help="задать один вопрос и выйти")
    parser.add_argument(
        "--speak", action="store_true", help="озвучить ответ, а не только напечатать"
    )
    parser.add_argument(
        "--no-web", action="store_true", help="ответить без доступа в интернет"
    )
    parser.add_argument(
        "--voices", nargs="?", const=SAMPLE_TEXT, metavar="ФРАЗА",
        help="озвучить фразу всеми голосами и сохранить в samples/",
    )
    parser.add_argument(
        "--devices", action="store_true", help="показать аудиоустройства с номерами"
    )
    parser.add_argument(
        "--voice", action="store_true", help="голосовой режим: Enter, вопрос, ответ голосом"
    )
    parser.add_argument(
        "--mic", action="store_true", help="проверить микрофон: показать громкость"
    )
    parser.add_argument(
        "--wake", action="store_true",
        help="режим кодовой фразы: ассистент сам просыпается на слово",
    )
    parser.add_argument(
        "--wake-test", action="store_true",
        help="показать уверенность модели кодовой фразы, чтобы подобрать порог",
    )
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg)

    if args.devices:
        return list_devices()

    if args.no_web:
        cfg = replace(cfg, search=replace(cfg.search, enabled=False))

    # Озвучку создаём, но модель она подгрузит только когда понадобится.
    tts = make_tts(cfg) if cfg.tts.enabled else None

    if args.voices:
        return render_voices(tts, args.voices)

    if args.mic:
        return check_microphone(cfg)

    if args.wake_test:
        # Ни ключа, ни модели тут не нужно - только микрофон и wake word.
        make_wake_word(cfg).monitor()
        return 0

    assistant = Assistant(cfg)

    if args.voice or args.wake:
        if tts is None or not cfg.stt.enabled:
            print("Для голосового режима нужны включённые stt и tts в config.yaml")
            return 1
        wake = make_wake_word(cfg) if (args.wake and cfg.wake.enabled) else None
        if args.wake and wake is None:
            print("Кодовая фраза выключена: поставь wake.enabled в config.yaml")
            return 1
        loop = VoiceLoop(
            cfg, assistant, make_stt(cfg), tts, make_microphone(cfg), wake
        )
        return loop.run_wake() if args.wake else loop.run()

    if args.selftest:
        return selftest(assistant, tts)

    # В обычном режиме молчим, пока не попросили голос: отлаживать текстом
    # быстрее, чем слушать каждый ответ.
    speaker = tts if args.speak else None

    if args.ask:
        return 0 if _respond(assistant, speaker, args.ask) else 1
    return repl(assistant, speaker)


if __name__ == "__main__":
    sys.exit(main())
