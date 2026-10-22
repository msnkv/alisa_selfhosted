"""Клиент DeepSeek и цикл вызова инструментов.

Модель может сходить в поиск, прочитать страницу, снова поискать — и только
потом ответить. Цикл ограничен и по числу шагов, и по общему времени: лучше
честное «не нашёл», чем ассистент, который молчит полминуты.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from openai import APIStatusError, OpenAI

from .config import Config
from .prompt import (
    Knowledge,
    build_system_prompt,
    clean_for_speech,
    looks_like_tool_markup,
)
from .search import EMPTY_RESULT_MARKER, SearchTools, make_backend

log = logging.getLogger(__name__)


class MissingApiKey(RuntimeError):
    pass


@dataclass
class Answer:
    """Ответ вместе с замерами — чтобы потом искать, где уходит время."""

    text: str
    seconds: float
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cached: int = 0
    steps: int = 1
    tools_used: list[str] = field(default_factory=list)
    # Поиск сходил, но вернулся пустым. Отличает «поисковик не ответил»
    # от «ассистент не справился» — важно при разборе прогона тестов.
    search_empty: bool = False


# Конец предложения: точка, восклицательный или вопросительный знак
# после как минимум двух букв, а за ним пробел. Двух букв — чтобы не
# разрывать сокращения вида «и т.д.»: там перед точкой одна буква.
# Пробел — чтобы не отдавать в синтез обрубок, пока модель ещё печатает.
_SENTENCE_END = re.compile(r"(?<=[^\W\d_]{2})[.!?…]+(?=\s)")


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Отрезает от буфера законченные предложения.

    Возвращает готовые предложения и остаток, который модель ещё дописывает.
    Резать надо по целому буферу, а не по обрезку до последнего знака:
    правило требует пробела после точки, и если отрезать буфер ровно по
    точке, последнее предложение этот пробел потеряет и не найдётся.
    """
    ready: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(buffer):
        sentence = clean_for_speech(buffer[start:match.end()])
        if sentence:
            ready.append(sentence)
        start = match.end()
    return ready, buffer[start:]


@dataclass
class _StreamedCall:
    """Вызов инструмента, собранный из кусочков потока.

    Поле function возвращает сам объект, чтобы остальной код не отличал
    этот вызов от того, что приходит обычным, непотоковым ответом:
    там обращаются к call.function.name и call.function.arguments.
    """

    id: str = ""
    name: str = ""
    arguments: str = ""

    @property
    def function(self) -> "_StreamedCall":
        return self


# На последнем шаге инструменты у модели забирают, но системный промпт
# по-прежнему требует «обязательно вызови поиск». Из этого противоречия
# она пытается вызвать инструмент текстом. Поэтому на последнем шаге
# говорим прямо: искать больше нечем, отвечай словами.
LAST_STEP_NUDGE = (
    "Инструментов больше нет, искать нечем. Ответь словами по тому, что "
    "уже нашла. Если надёжного ответа нет — честно скажи, что проверить "
    "не удалось. Никакой разметки и никаких вызовов инструментов в ответе."
)


# Что сказать, если инструменты отработали, а ответа всё равно нет.
GAVE_UP = (
    "Не смог найти надёжный ответ на этот вопрос, поэтому говорить ничего "
    "не буду, чтобы не ввести в заблуждение."
)


class Assistant:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.knowledge = Knowledge(cfg.data_dir)
        # Клиент создаём лениво: без ключа его конструктор падает, а селфтест
        # должен успеть сказать об этом по-человечески.
        self._client: OpenAI | None = None
        # Если API не примет отключение размышлений, выставим флаг и
        # перестанем его посылать, чтобы не ловить ошибку на каждом вопросе.
        self._thinking_supported = True

        # Последние реплики диалога. Нужны, чтобы работали уточнения:
        # ассистент сам предлагает «рассказать по шагам?», и без памяти
        # ответ «да, расскажи» уходит в пустоту - модель не знает, о чём
        # речь, и берёт первое похожее из data/.
        # Храним только текст, без вызовов инструментов: результаты поиска
        # в истории не нужны, а DeepSeek после вызова инструмента требует
        # возвращать ему и служебные поля - лишний повод для ошибок.
        self.history: list[dict] = []

        self.tools: SearchTools | None = None
        if cfg.search.enabled:
            backend = make_backend(
                cfg.search.backend,
                region=cfg.search.region,
                engine=cfg.search.engine,
                retries=cfg.search.retries,
                hedge_after_s=cfg.search.hedge_after_s,
                timeout_s=cfg.search.timeout_s,
            )
            self.tools = SearchTools(
                backend,
                max_results=cfg.search.max_results,
                page_chars=cfg.search.page_chars,
                fetch_timeout_s=cfg.search.fetch_timeout_s,
            )

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if not self.cfg.llm.api_key:
                raise MissingApiKey(
                    "не найден DEEPSEEK_API_KEY — вставь ключ в файл .env"
                )
            self._client = OpenAI(
                api_key=self.cfg.llm.api_key,
                base_url=self.cfg.llm.base_url,
                timeout=self.cfg.llm.timeout_s,
                max_retries=self.cfg.llm.max_retries,
            )
        return self._client

    def _extra_body(self) -> dict | None:
        if self.cfg.llm.thinking or not self._thinking_supported:
            return None
        # DeepSeek включает размышления по умолчанию; для голоса это лишние
        # секунды, поэтому гасим их явно.
        return {"thinking": {"type": "disabled"}}

    # --- основной вход ----------------------------------------------------

    def ask(
        self,
        question: str,
        *,
        on_tool_call: Callable[[str, dict], None] | None = None,
        on_sentence: Callable[[str], None] | None = None,
        on_step: Callable[[bool], None] | None = None,
    ) -> Answer:
        """Задаёт вопрос модели, при необходимости гоняя её по инструментам.

        on_tool_call вызывается перед каждым обращением к инструменту. На
        шаге 5 через него голосовой режим говорит «секунду, смотрю», чтобы
        в комнате не висела тишина.

        on_sentence включает потоковый режим: ответ читается по мере
        генерации, и каждое законченное предложение уходит в этот
        обработчик сразу. Голосовой цикл отдаёт его в очередь синтеза и
        начинает говорить, не дожидаясь конца ответа. Без обработчика
        всё работает как раньше, одним ответом — так ходят тесты и
        текстовый режим без озвучки.

        on_step зовётся после каждого шага и говорит, оказался шаг
        вызовом инструмента или текстом. Это нужно озвучке: текст с
        шага с инструментом — не ответ, а присказка, и говорить его
        нельзя, иначе ответ прозвучит дважды.
        """
        system = build_system_prompt(
            name=self.cfg.name,
            owner=self.cfg.owner,
            max_sentences=self.cfg.max_sentences,
            max_words=self.cfg.max_words,
            knowledge=self.knowledge.text(),
            tools_available=self.tools is not None,
        )
        # История идёт после системного промпта - значит большой неизменный
        # блок с данными из data/ по-прежнему кэшируется целиком.
        messages: list[dict] = [
            {"role": "system", "content": system},
            *self.history,
            {"role": "user", "content": question},
        ]

        started = time.perf_counter()
        deadline = started + self.cfg.agent.total_timeout_s
        totals = {"in": 0, "out": 0, "cached": 0}
        tools_used: list[str] = []
        search_empty = False
        nudged = False
        step = 0

        while True:
            step += 1
            out_of_time = time.perf_counter() >= deadline
            last_step = step > self.cfg.agent.max_steps or out_of_time
            # На последнем шаге инструменты не даём: модель обязана ответить
            # словами, иначе цикл не закончится никогда.
            tools = None if (last_step or self.tools is None) else self.tools.SCHEMAS
            if last_step and tools_used and not nudged:
                # Инструменты уже отработали, а теперь их отобрали -
                # надо сказать об этом прямо, иначе модель выдаст вызов
                # текстом. Сообщение идёт в конец, кэш промпта не рвётся.
                messages.append({"role": "user", "content": LAST_STEP_NUDGE})
                nudged = True

            content, calls, usage = self._step(messages, tools, on_sentence)
            self._add_usage(totals, usage)
            if on_step is not None:
                on_step(bool(calls))

            if not calls:
                answer = self._finish(
                    content, started, totals, step, tools_used, search_empty,
                )
                self._remember(question, answer.text)
                return answer

            if last_step:
                # Сюда попасть не должны (инструментов не давали), но если
                # модель всё равно просит — честно сдаёмся.
                break

            messages.append(self._assistant_turn(content, calls))
            for call in calls:
                name, arguments = self._parse_call(call)
                tools_used.append(name)
                if on_tool_call is not None:
                    on_tool_call(name, arguments)
                result = self.tools.call(name, arguments)
                if EMPTY_RESULT_MARKER in result:
                    search_empty = True
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    }
                )

            if time.perf_counter() >= deadline:
                log.warning(
                    "бюджет времени %.0f с исчерпан после %d шагов",
                    self.cfg.agent.total_timeout_s, step,
                )

        return self._finish(
            GAVE_UP, started, totals, step, tools_used, search_empty
        )

    # --- память диалога ---------------------------------------------------

    def _remember(self, question: str, answer: str) -> None:
        if self.cfg.agent.memory_turns <= 0 or not answer:
            return
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        # Держим только последние пары вопрос-ответ: ассистент отвечает
        # короткими фразами, дальше в прошлое помнить нечего, а каждая
        # лишняя реплика - это токены и шанс, что модель зацепится
        # за старую тему вместо новой.
        limit = self.cfg.agent.memory_turns * 2
        if len(self.history) > limit:
            del self.history[:-limit]

    def forget(self) -> None:
        """Забыть диалог. Нужно между независимыми вопросами в тестах."""
        self.history.clear()

    # --- внутренности -----------------------------------------------------

    def _create_kwargs(self, messages: list[dict], tools: list[dict] | None) -> dict:
        kwargs = {
            "model": self.cfg.llm.model,
            "messages": messages,
            "temperature": self.cfg.llm.temperature,
            "max_tokens": self.cfg.llm.max_tokens,
            "extra_body": self._extra_body(),
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return kwargs

    def _call_api(self, messages: list[dict], tools: list[dict] | None, *, stream: bool):
        kwargs = self._create_kwargs(messages, tools)
        if stream:
            kwargs["stream"] = True
            # Без этого в потоковом режиме расход токенов не приходит
            # вообще, и мы перестанем видеть, что кэш промпта работает.
            kwargs["stream_options"] = {"include_usage": True}
        try:
            return self.client.chat.completions.create(**kwargs)
        except APIStatusError as exc:
            # Отключение размышлений — единственный нестандартный параметр,
            # который мы посылаем. Если сервер его не понял, пробуем без него.
            if self._thinking_supported and kwargs.get("extra_body") is not None:
                log.warning(
                    "сервер не принял отключение размышлений (%s), повторяю без него",
                    exc.status_code,
                )
                self._thinking_supported = False
                kwargs["extra_body"] = None
                return self.client.chat.completions.create(**kwargs)
            raise

    def _step(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        on_sentence: Callable[[str], None] | None,
    ):
        """Один обмен с моделью. Возвращает текст, вызовы и расход токенов.

        Форма ответа одна и та же, потоковый режим или нет, поэтому цикл
        в ask про эту разницу ничего не знает.
        """
        if on_sentence is None:
            response = self._call_api(messages, tools, stream=False)
            message = response.choices[0].message
            return (
                message.content or "",
                list(getattr(message, "tool_calls", None) or []),
                response.usage,
            )
        return self._read_stream(
            self._call_api(messages, tools, stream=True), on_sentence
        )

    @staticmethod
    def _read_stream(stream, on_sentence: Callable[[str], None]):
        """Читает поток, отдавая законченные предложения по ходу дела.

        Тонкость: заранее неизвестно, чем окажется шаг — текстом или
        вызовом инструмента. Узнаём по тому, что приходит в кусочках, и
        поэтому озвучивать начинаем сразу, как только пошёл текст. На
        шаге с инструментами текста обычно нет вообще, так что лишнего
        ассистент не скажет.
        """
        pieces: list[str] = []
        buffer = ""
        calls: dict[int, _StreamedCall] = {}
        usage = None
        # Если в тексте пошла служебная разметка - это не ответ, а
        # попытка вызвать инструмент. Озвучивать такое нельзя.
        muted = False

        for chunk in stream:
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # Вызовы инструментов приходят по частям: имя в одном кусочке,
            # аргументы по буквам в следующих. Собираем их по номеру.
            for part in getattr(delta, "tool_calls", None) or []:
                call = calls.setdefault(part.index, _StreamedCall())
                if part.id:
                    call.id = part.id
                function = getattr(part, "function", None)
                if function is not None:
                    if function.name:
                        call.name = function.name
                    if function.arguments:
                        call.arguments += function.arguments

            text = getattr(delta, "content", None)
            if not text:
                continue
            pieces.append(text)
            buffer += text
            if not muted and looks_like_tool_markup(buffer):
                log.warning("модель выдаёт вызов инструмента текстом, молчу")
                muted = True
            if muted:
                continue
            ready, buffer = split_sentences(buffer)
            for sentence in ready:
                on_sentence(sentence)

        # Последнее предложение обычно без пробела на конце, поэтому
        # разбивка его не отдаёт - отдаём здесь.
        tail = "" if muted else clean_for_speech(buffer)
        if tail:
            on_sentence(tail)

        return "".join(pieces), [calls[index] for index in sorted(calls)], usage

    @staticmethod
    def _assistant_turn(content: str, calls) -> dict:
        """Ход модели с вызовами инструментов, пересобранный для истории."""
        return {
            "role": "assistant",
            "content": content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in calls
            ],
        }

    @staticmethod
    def _parse_call(call) -> tuple[str, dict]:
        name = call.function.name
        try:
            arguments = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            log.warning("не разобрал аргументы %s: %r", name, call.function.arguments)
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        return name, arguments

    @staticmethod
    def _add_usage(totals: dict, usage) -> None:
        if usage is None:
            return
        totals["in"] += getattr(usage, "prompt_tokens", 0) or 0
        totals["out"] += getattr(usage, "completion_tokens", 0) or 0
        totals["cached"] += getattr(usage, "prompt_cache_hit_tokens", 0) or 0

    def _finish(
        self,
        text: str,
        started: float,
        totals: dict,
        steps: int,
        tools_used: list[str],
        search_empty: bool = False,
    ) -> Answer:
        if looks_like_tool_markup(text):
            # Не ответ, а вызов инструмента текстом. Лучше честное
            # «не нашла», чем зачитанная вслух разметка.
            log.warning("вместо ответа пришла разметка вызова: %s", text[:200])
            text = ""
        answer = Answer(
            text=clean_for_speech(text) or GAVE_UP,
            seconds=time.perf_counter() - started,
            tokens_in=totals["in"],
            tokens_out=totals["out"],
            tokens_cached=totals["cached"],
            steps=steps,
            tools_used=tools_used,
            search_empty=search_empty,
        )
        log.info(
            "ответ за %.2f с, шагов %d, инструменты %s, вход %d (из кэша %d), выход %d",
            answer.seconds, answer.steps, tools_used or "не использованы",
            answer.tokens_in, answer.tokens_cached, answer.tokens_out,
        )
        log.info("ответ: %s", answer.text)
        # Цифры в ответе — это промах правила из промпта. Сами их не правим
        # (русские числительные с падежами надёжно не развернуть), но пишем
        # в лог: если промахов много, значит правило надо усиливать.
        if re.search(r"\d", answer.text):
            log.warning("в ответе остались цифры, синтезатор прочитает их криво: %s", answer.text)
        return answer
