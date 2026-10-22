"""Прогон набора вопросов из questions.yaml.

Смысл этой штуки: когда меняешь системный промпт, модель или бэкенд поиска,
руками перепроверять десяток вопросов невыносимо, а на глаз «кажется, стало
лучше» ничего не значит. Здесь прогон одной командой и таблица с цифрами.

Запуск:
    .\\test.ps1
    .\\test.ps1 --only fresh
    .\\test.ps1 --repeat 3
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

# Чтобы скрипт работал и как `python -m tests.run_questions`,
# и как `python tests/run_questions.py`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assistant.config import load_config, setup_logging  # noqa: E402
from assistant.llm import Answer, Assistant  # noqa: E402

QUESTIONS_FILE = Path(__file__).with_name("questions.yaml")

# Ответ длиннее бюджета - это не провал, а повод присмотреться:
# синтезатор будет бубнить долго, но ответ по сути верный.
LENGTH_TOLERANCE = 1.5


@dataclass
class Outcome:
    ask: str
    kind: str
    answer: Answer
    failures: list[str]
    warnings: list[str]

    @property
    def passed(self) -> bool:
        return not self.failures


def evaluate(case: dict, answer: Answer, max_words: int) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    text = answer.text
    lowered = text.lower()

    if not text.strip():
        failures.append("пустой ответ")
        return failures, warnings

    # Цифры читает синтезатор речи, и читает плохо.
    found_digits = re.findall(r"\d+", text)
    if found_digits:
        failures.append("цифры вместо слов: " + ", ".join(found_digits[:4]))

    expected_tools = str(case.get("tools", "any")).lower()
    used = bool(answer.tools_used)
    if expected_tools == "no" and used:
        failures.append("полез в интернет без нужды: " + ", ".join(answer.tools_used))
    elif expected_tools == "yes" and not used:
        failures.append("не полез в интернет, хотя нужны свежие данные")

    expect_any = [s for s in case.get("expect_any", []) if s]
    if expect_any and not any(s.lower() in lowered for s in expect_any):
        if answer.search_empty:
            # Ассистент отработал правильно: сходил в поиск, получил пустоту
            # и честно сказал об этом. Виноват поисковик, а не промпт.
            failures.append(
                "поиск вернулся пустым, ответа по сути нет "
                "(ассистент повёл себя верно, подвёл поисковик)"
            )
        else:
            failures.append(
                "нет ни одного из ожидаемых слов: " + ", ".join(expect_any[:4])
            )

    for forbidden in case.get("forbid", []):
        if forbidden.lower() in lowered:
            failures.append(f"встретилось запрещённое {forbidden!r}")

    words = len(text.split())
    if words > max_words * LENGTH_TOLERANCE:
        warnings.append(f"длинно: {words} слов при бюджете {max_words}")

    return failures, warnings


def run(cases: list[dict], assistant: Assistant, repeat: int) -> list[Outcome]:
    outcomes: list[Outcome] = []
    total = len(cases) * repeat
    done = 0

    for case in cases:
        for _ in range(repeat):
            done += 1
            print(f"[{done}/{total}] {case['ask']}", flush=True)
            try:
                answer = assistant.ask(case["ask"])
            except Exception as exc:  # noqa: BLE001
                answer = Answer(text="", seconds=0.0)
                outcomes.append(
                    Outcome(
                        ask=case["ask"],
                        kind=case.get("type", "?"),
                        answer=answer,
                        failures=[f"упало: {type(exc).__name__}: {exc}"],
                        warnings=[],
                    )
                )
                continue

            failures, warnings = evaluate(case, answer, assistant.cfg.max_words)
            outcomes.append(
                Outcome(
                    ask=case["ask"],
                    kind=case.get("type", "?"),
                    answer=answer,
                    failures=failures,
                    warnings=warnings,
                )
            )
    return outcomes


def report(outcomes: list[Outcome]) -> str:
    """Отчёт в markdown: его же пишем в файл, чтобы сравнивать прогоны."""
    passed = sum(1 for o in outcomes if o.passed)
    lines = [
        f"# Прогон {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"Пройдено {passed} из {len(outcomes)}.",
        "",
    ]

    by_kind: dict[str, list[Outcome]] = {}
    for outcome in outcomes:
        by_kind.setdefault(outcome.kind, []).append(outcome)

    for kind, group in by_kind.items():
        ok = sum(1 for o in group if o.passed)
        lines.append(f"## {kind}: {ok} из {len(group)}")
        lines.append("")
        for outcome in group:
            mark = "OK  " if outcome.passed else "ПЛОХО"
            tools = ", ".join(outcome.answer.tools_used) or "без инструментов"
            lines.append(f"- **{mark}** {outcome.ask}")
            lines.append(f"  - ответ: {outcome.answer.text or '(пусто)'}")
            lines.append(
                f"  - {outcome.answer.seconds:.2f} с, {tools}, "
                f"вход {outcome.answer.tokens_in} "
                f"(из кэша {outcome.answer.tokens_cached}), "
                f"выход {outcome.answer.tokens_out}"
            )
            for failure in outcome.failures:
                lines.append(f"  - ПРОБЛЕМА: {failure}")
            for warning in outcome.warnings:
                lines.append(f"  - заметка: {warning}")
        lines.append("")

    seconds = [o.answer.seconds for o in outcomes if o.answer.seconds > 0]
    if seconds:
        seconds.sort()
        lines.append(
            f"Задержка: медиана {seconds[len(seconds) // 2]:.2f} с, "
            f"минимум {seconds[0]:.2f} с, максимум {seconds[-1]:.2f} с."
        )
    return "\n".join(lines)


def print_table(outcomes: list[Outcome]) -> None:
    print()
    width = max(len(o.ask) for o in outcomes)
    for outcome in outcomes:
        mark = "OK   " if outcome.passed else "ПЛОХО"
        tools = "сеть" if outcome.answer.tools_used else "    "
        print(
            f"{mark} {outcome.kind:6} {tools} "
            f"{outcome.answer.seconds:5.2f}с  {outcome.ask:<{width}}"
        )
        for failure in outcome.failures:
            print(f"        └─ {failure}")
        for warning in outcome.warnings:
            print(f"        └─ (заметка) {warning}")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Прогон тестовых вопросов")
    parser.add_argument("--only", help="только вопросы этого типа: data, known, fresh")
    parser.add_argument("--repeat", type=int, default=1, help="сколько раз каждый вопрос")
    args = parser.parse_args()

    cases = yaml.safe_load(QUESTIONS_FILE.read_text(encoding="utf-8"))["questions"]
    if args.only:
        cases = [c for c in cases if c.get("type") == args.only]
        if not cases:
            print(f"нет вопросов типа {args.only!r}")
            return 1

    cfg = load_config()
    setup_logging(cfg)
    assistant = Assistant(cfg)

    outcomes = run(cases, assistant, max(1, args.repeat))
    print_table(outcomes)

    text = report(outcomes)
    out = cfg.log_dir / f"прогон-{datetime.now():%Y%m%d-%H%M}.md"
    out.write_text(text, encoding="utf-8")

    passed = sum(1 for o in outcomes if o.passed)
    print(f"\nПройдено {passed} из {len(outcomes)}. Отчёт: {out}")
    return 0 if passed == len(outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
