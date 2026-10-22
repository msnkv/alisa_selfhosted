"""Текстовый режим: вопрос с клавиатуры, ответ в консоль.

Нужен для отладки без микрофона. Запуск:
    .\\alisa.ps1
    .\\alisa.ps1 --selftest
    .\\alisa.ps1 --ask "где лежат болгарки?"
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from .config import load_config, setup_logging
from .llm import Answer, Assistant
from .prompt import build_system_prompt, human_now

EXIT_WORDS = {"выход", "exit", "quit", "q"}


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
    """В текстовом режиме просто печатаем, что ассистент пошёл искать.

    В голосовом режиме (шаг 5) сюда встанет фраза-заглушка «секунду, смотрю».
    """
    detail = arguments.get("query") or arguments.get("url") or ""
    print(f"  ... {name}: {detail}")


def selftest(assistant: Assistant) -> int:
    """Проверяет, что всё на месте, и делает один живой запрос к API."""
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
        print("\nЕсть проблемы, запрос к модели не делаю.")
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
        first = found.splitlines()[0] if found else "(пусто)"
        print(f"  {first}")
        if "ничего не вернул" in found:
            print("  поиск не отвечает — попробуй позже или смени engine в config.yaml")
            return 1

    print("\nВсё работает.")
    return 0


def ask_once(assistant: Assistant, question: str) -> int:
    try:
        answer = assistant.ask(question, on_tool_call=_announce_tool)
    except Exception as exc:  # noqa: BLE001
        print(f"ошибка: {type(exc).__name__}: {exc}")
        return 1
    print(answer.text)
    print(_stats(answer))
    return 0


def repl(assistant: Assistant) -> int:
    print(f"{assistant.cfg.name}: текстовый режим. Пустая строка или «выход» — закончить.\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not question or question.lower() in EXIT_WORDS:
            return 0

        try:
            answer = assistant.ask(question, on_tool_call=_announce_tool)
        except Exception as exc:  # noqa: BLE001
            print(f"  ошибка: {type(exc).__name__}: {exc}\n")
            continue

        print(f"\n{answer.text}")
        print(_stats(answer) + "\n")


def main() -> int:
    _force_utf8()

    parser = argparse.ArgumentParser(description="Текстовый режим голосового помощника")
    parser.add_argument(
        "--selftest", action="store_true", help="проверить окружение, API и поиск"
    )
    parser.add_argument("--ask", metavar="ВОПРОС", help="задать один вопрос и выйти")
    parser.add_argument(
        "--no-web", action="store_true", help="ответить без доступа в интернет"
    )
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg)
    if args.no_web:
        cfg = replace(cfg, search=replace(cfg.search, enabled=False))
    assistant = Assistant(cfg)

    if args.selftest:
        return selftest(assistant)
    if args.ask:
        return ask_once(assistant, args.ask)
    return repl(assistant)


if __name__ == "__main__":
    sys.exit(main())
