"""Клиент DeepSeek и цикл вызова инструментов.

Модель может сходить в поиск, прочитать страницу, снова поискать — и только
потом ответить. Цикл ограничен и по числу шагов, и по общему времени: лучше
честное «не нашёл», чем ассистент, который молчит полминуты.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import time
import threading
from dataclasses import dataclass, field
from typing import Callable

# Не httpx, а httpx2: openai с третьей версии работает на нём, и его
# исключения от httpx не наследуют. С обычным httpx обрыв посреди потока
# пролетал мимо перехвата, а срок молчания не доходил до запроса.
import httpx2
from openai import APIConnectionError, APIStatusError, OpenAI

from .config import Config
from .history import DialogueLog, looks_like_complaint
from .prompt import (
    Knowledge,
    build_system_prompt,
    clean_for_speech,
    looks_like_tool_markup,
    KNOWLEDGE_WITHHELD,
    MODE_RECORDS,
    answer_mode,
    long_form_cancel,
    long_form_request,
    wants_detail,
)
from .search import EMPTY_RESULT_MARKER, SearchTools, make_backend
from .streaming import check_cancelled
from .reminder_dialogue import ReminderDialogue, INTENT as REMINDER_INTENT
from .notes import NoteDialogue, note_request

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

# Модель не ответила даже на последнем шаге. Это не «не нашла», а «сервис
# молчит», и говорить надо именно это: повторить вопрос имеет смысл.
NO_MODEL = "Не получилось дождаться ответа модели, попробуй спросить ещё раз чуть позже."

# Сколько времени гарантированно получает последний, текстовый шаг, даже
# если бюджет agent.total_timeout_s уже съеден поиском. Без запаса ответ,
# ради которого всё искали, обрывался бы сразу.
FINAL_STEP_S = 15.0

# Сколько молчания считать зависанием в потоковом ответе: ни байта до
# начала ответа или между кусками. Здоровый DeepSeek начинает за секунду
# даже с промптом на двадцать шесть тысяч токенов, а 24.10.2026 он
# полчаса то и дело молчал по 30 с подряд, и тогда же висел поиск.
# Только для потока: без потока ответ приходит целиком, и подробный
# ответ законно идёт дольше.
STALL_S = 8.0

# Меньше этого на повтор зависшего шага не тратимся — не успеет.
RETRY_MIN_S = 3.0

# Последний шаг после того, как шаг с инструментами не дождались: поиска
# не было вовсе, и LAST_STEP_NUDGE («ответь по тому, что нашла») не
# подходит. Без указания модель отвечает про погоду из головы.
NO_SEARCH_NUDGE = (
    "Поиск сейчас недоступен, проверить ничего не получилось. Если вопрос "
    "про свежие данные — погоду, курсы, цены, новости, расписания, — не "
    "называй никаких цифр и фактов, а честно скажи одной фразой, что "
    "сейчас проверить не получилось. На остальное отвечай как обычно. "
    "Никакой разметки и никаких вызовов инструментов в ответе."
)

# Что модель получает вместо результата, если до инструмента не дошли.
# Ответ на каждый вызов обязателен, иначе API отклонит следующий запрос.
TOOL_SKIPPED = "Время на поиск вышло, этот вызов не выполнялся. Отвечай по тому, что уже есть."


class StepTimeout(Exception):
    """Потоковый ответ модели не уложился в срок шага."""


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

        # Режим разговора. Липкий: включается фразой «давай поговорим» или
        # «расскажи сказку» и держится, пока не скажут «хватит» или
        # «забудь». Повторять фразу перед каждой репликой было бы странно.
        self.long_form = False
        self.session_id = "local"
        self.reminders = ReminderDialogue(cfg.reminders)
        self.notes = NoteDialogue(cfg.data_dir)

        # Журнал диалогов: по строке JSON на обмен, для разбора потом.
        # Пишется здесь, а не в голосовом цикле, намеренно: через ask
        # проходят все пути - текстовый режим, голос, сервер со
        # спутником и страница с телефона. Один вход, один журнал.
        self.dialogue = DialogueLog(
            cfg.log_dir / cfg.dialogue_file, enabled=cfg.dialogue_log
        )

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

    def new_session(self, session_id: str) -> Assistant:
        """Отдельный разговор с общими настройками, знаниями и инструментами.

        Копируем оболочку, а изменяемые данные разговора создаём заново.
        Whisper и Silero принадлежат серверу и здесь не загружаются.
        """
        session = copy.copy(self)
        session.history = []
        session.long_form = False
        session.session_id = session_id
        session.reminders = ReminderDialogue(self.cfg.reminders)
        session.notes = NoteDialogue(self.cfg.data_dir)
        session.dialogue = DialogueLog(self.dialogue.path, enabled=self.dialogue.enabled)
        return session

    def ask(
        self,
        question: str,
        *,
        on_tool_call: Callable[[str, dict], None] | None = None,
        on_sentence: Callable[[str], None] | None = None,
        on_step: Callable[[bool], None] | None = None,
        cancel_event: threading.Event | None = None,
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
        # Длинный ответ просят словами: «раскрой подробно», «детально»,
        # «говори долго». Тогда и предел в промпте другой, и места под
        # ответ надо больше - иначе он обрежется на середине фразы.
        # Разговор включается и выключается словами и держится между
        # репликами.
        check_cancelled(cancel_event)
        reminder_started = time.perf_counter()

        def extract_reminder(messages, schema):
            extra = {"cancel_event": cancel_event} if cancel_event is not None else {}
            _, calls, _ = self._step(messages, [schema], None, 600, **extra)
            if len(calls) != 1:
                return None
            name, data = self._parse_call(calls[0])
            return data if name == "propose_reminder" else None

        # Заметки раньше напоминаний: «подтверждаю» у них общее, а
        # реплика не про заметку сбрасывает черновик заметки и уходит дальше.
        local_text, local_kind = self.notes.handle(question, cancel_event), "note"
        if local_text is not None:
            # Иначе старое «подтверждаю» сохранило бы и брошенное напоминание.
            self.reminders.reset()
        else:
            local_text = self.reminders.handle(question, self.session_id, extract_reminder, cancel_event)
            local_kind = "reminder"
        if local_text is not None:
            answer = Answer(local_text, time.perf_counter() - reminder_started, tools_used=[local_kind])
            try:
                check_cancelled(cancel_event)
                if on_sentence is not None:
                    on_sentence(local_text)
                if on_step is not None:
                    on_step(False)
                check_cancelled(cancel_event)
            except BaseException:
                # Не оставляем подтверждение, если предпросмотр не дошёл.
                self.reminders.reset()
                self.notes.reset()
                raise
            self._remember(question, answer.text)
            self._log_turn(question, answer, None, False, False)
            return answer
        if long_form_cancel(question):
            self.long_form = False
        elif long_form_request(question):
            self.long_form = True
        long_form = self.long_form

        # Явная просьба про источник: «посмотри в моих записях» или
        # «поищи в интернете». Без неё модель решает сама, как и раньше.
        mode = answer_mode(question)
        # Вариант «без просьбы в записи не смотреть» - выключен по
        # умолчанию, см. data.only_on_request в config.yaml.
        knowledge = self.knowledge.text()
        if self.cfg.data_only_on_request and mode != MODE_RECORDS:
            knowledge = KNOWLEDGE_WITHHELD

        detail = wants_detail(question) and not long_form
        detail_words = self.cfg.detail_words if detail else 0
        story_words = self.cfg.story_words if long_form else 0
        if long_form:
            max_tokens = self.cfg.llm.story_max_tokens
        elif detail:
            max_tokens = self.cfg.llm.detail_max_tokens
        else:
            max_tokens = self.cfg.llm.max_tokens
        if detail:
            log.info("просят подробный ответ: бюджет %d слов", detail_words)
        if long_form:
            log.info("режим разговора: бюджет %d слов", story_words)
        if mode:
            log.info("просят источник: %s", mode)

        system = build_system_prompt(
            name=self.cfg.name,
            owner=self.cfg.owner,
            max_sentences=self.cfg.max_sentences,
            max_words=self.cfg.max_words,
            knowledge=knowledge,
            tools_available=self.tools is not None,
            city=self.cfg.city,
            persona=self.cfg.persona,
            detail_words=detail_words,
            mode=mode,
            story_words=story_words,
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
        timed_out = False
        step = 0

        while True:
            check_cancelled(cancel_event)
            step += 1
            remaining = deadline - time.perf_counter()
            last_step = step > self.cfg.agent.max_steps or remaining <= 0 or timed_out
            # На последнем шаге инструменты не даём: модель обязана ответить
            # словами, иначе цикл не закончится никогда.
            # В разговоре инструментов не даём совсем: лазить в интернет
            # посреди сказки незачем, а заодно это делает шаг заведомо
            # текстовым - и озвучка может начаться с первой фразы.
            tools = (
                None
                if (last_step or self.tools is None or long_form)
                else self.tools.SCHEMAS
            )
            if last_step and tools_used and not nudged:
                # Инструменты уже отработали, а теперь их отобрали -
                # надо сказать об этом прямо, иначе модель выдаст вызов
                # текстом. Сообщение идёт в конец, кэш промпта не рвётся.
                messages.append({"role": "user", "content": LAST_STEP_NUDGE})
                nudged = True
            elif timed_out and not nudged:
                # До поиска не дошли: модель не ответила на шаге с
                # инструментами. Без этого указания она на последнем шаге
                # выдумала погоду — «около семи градусов» (24.10.2026).
                messages.append({"role": "user", "content": NO_SEARCH_NUDGE})
                nudged = True

            # Бюджет проверяется между шагами, а один вызов модели сам по
            # себе может висеть llm.timeout_s, да ещё и с повтором. Поймано
            # 23.10.2026: два шага по 33 и 52 секунды, ответ через две
            # минуты при бюджете 35. Поэтому срок вызова режем по остатку,
            # а последнему шагу оставляем свой запас.
            # Повтор внутри клиента OpenAI выключен: он удваивал срок.
            # Повторяем сами, ниже, и только зависший шаг.
            limit = min(self.cfg.llm.timeout_s, max(remaining, FINAL_STEP_S) if last_step else remaining)
            step_deadline = time.perf_counter() + limit
            extra = {"cancel_event": cancel_event} if cancel_event is not None else {}
            # Зависший запрос повторяем один раз, по новому соединению, —
            # как попытки поиска внахлёст. Здоровый DeepSeek начинает
            # ответ за секунду, так что молчание дольше STALL_S — это
            # зависание, и второй запрос почти всегда проходит. Не
            # повторяем, если что-то уже прозвучало: начало сказки дважды.
            spoke = []
            speak = None if on_sentence is None else (lambda s: (spoke.append(s), on_sentence(s)))
            failure = None
            for attempt in (1, 2):
                try:
                    content, calls, usage = self._step(
                        messages, tools, speak, max_tokens,
                        timeout=step_deadline - time.perf_counter(), **extra,
                    )
                    failure = None
                    break
                except (APIConnectionError, httpx2.TransportError, StepTimeout) as exc:  # и таймауты
                    check_cancelled(cancel_event)
                    failure = exc
                    left = step_deadline - time.perf_counter()
                    if attempt == 2 or spoke or left < RETRY_MIN_S:
                        break
                    log.warning("модель молчит (%s), повторяю запрос, осталось %.0f с",
                                type(exc).__name__, left)
                    # Обрывок первой попытки озвучивать нельзя.
                    if on_step is not None:
                        on_step(True)
            if failure is not None:
                exc = failure
                # Недоговорённый обрывок озвучивать нельзя.
                if on_step is not None:
                    on_step(True)
                # Шаг без инструментов повторять незачем: следующий будет
                # таким же, а начало сказки могло уже прозвучать. Две
                # попытки подряд без ответа — сервис лежит, и последний шаг
                # тоже не ответит: честно говорим сразу, не ждём ещё.
                if tools is None or attempt == 2:
                    log.warning("модель не ответила за %.0f с: %s", limit, type(exc).__name__)
                    answer = self._finish(NO_MODEL, started, totals, step, tools_used, search_empty)
                    self._log_turn(question, answer, mode, detail, long_form)
                    return answer
                # Шаг с инструментами не дождались — отвечаем по тому, что
                # уже нашли, а не сдаёмся целиком.
                log.warning("модель не ответила за %.0f с (%s), перехожу к ответу",
                            limit, type(exc).__name__)
                timed_out = True
                continue
            check_cancelled(cancel_event)
            self._add_usage(totals, usage)
            if on_step is not None:
                on_step(bool(calls))

            if not calls:
                answer = self._finish(
                    content, started, totals, step, tools_used, search_empty,
                )
                self._remember(question, answer.text)
                self._log_turn(question, answer, mode, detail, long_form)
                return answer

            if last_step:
                # Сюда попасть не должны (инструментов не давали), но если
                # модель всё равно просит — честно сдаёмся.
                break

            messages.append(self._assistant_turn(content, calls))
            for call in calls:
                check_cancelled(cancel_event)
                name, arguments = self._parse_call(call)
                if time.perf_counter() >= deadline:
                    # Не качаем ещё одну страницу на восемь секунд, когда
                    # время уже вышло: следующий шаг всё равно последний.
                    log.info("бюджет исчерпан, %s не вызываю", name)
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": TOOL_SKIPPED})
                    continue
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

        answer = self._finish(
            GAVE_UP, started, totals, step, tools_used, search_empty
        )
        self._log_turn(question, answer, mode, detail, long_form)
        return answer

    # --- журнал диалогов --------------------------------------------------

    def _log_turn(
        self,
        question: str,
        answer: Answer,
        mode: str | None,
        detail: bool,
        long_form: bool,
    ) -> None:
        """Складывает обмен в журнал. Схему не ломать, см. history.py."""
        self.dialogue.append(
            {
                "question": question,
                "session_id": self.session_id,
                "answer": answer.text,
                # Пометка «предыдущий ответ не устроил». Нужна не
                # ассистенту, а тому, кто потом будет разбирать журнал:
                # именно эти места говорят, как надо было ответить.
                "complaint": looks_like_complaint(question),
                "seconds": round(answer.seconds, 2),
                "steps": answer.steps,
                "tools": answer.tools_used,
                "search_empty": answer.search_empty,
                "mode": mode,
                "detail": detail,
                "long_form": long_form,
                "tokens_in": answer.tokens_in,
                "tokens_cached": answer.tokens_cached,
                "tokens_out": answer.tokens_out,
                "model": self.cfg.llm.model,
                "data_files": [p.name for p in self.knowledge.files()],
            }
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
        # Разговор тоже заканчивается: «забудь» — это сброс всего.
        self.long_form = False
        self.reminders.reset()
        self.notes.reset()
        # И в журнале дальше идёт уже другая беседа.
        self.dialogue.new_conversation()

    def will_be_long_form(self, question: str) -> bool:
        """Будет ли ответ разговором — ещё до обращения к модели.

        Нужно голосовому циклу: в этом режиме инструментов нет, значит
        шаг гарантированно текстовый, и озвучку можно разрешить сразу,
        с первой фразы. Иначе сказку пришлось бы слушать после минуты
        тишины — сперва вся генерация, и только потом звук.
        """
        if REMINDER_INTENT.search(question) or self.reminders.pending or self.reminders.context:
            return False
        if note_request(question) is not None or self.notes.active:
            return False
        if long_form_cancel(question):
            return False
        return self.long_form or long_form_request(question)

    # --- внутренности -----------------------------------------------------

    def _create_kwargs(
        self, messages: list[dict], tools: list[dict] | None, max_tokens: int
    ) -> dict:
        kwargs = {
            "model": self.cfg.llm.model,
            "messages": messages,
            "temperature": self.cfg.llm.temperature,
            "max_tokens": max_tokens,
            "extra_body": self._extra_body(),
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return kwargs

    def _call_api(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        *,
        stream: bool,
        max_tokens: int,
        timeout: float | None = None,
    ):
        kwargs = self._create_kwargs(messages, tools, max_tokens)
        client = self.client
        if timeout is not None:
            if stream:
                # Таймаут httpx — это молчание между байтами, а не срок
                # всего ответа: весь ответ ограничивает таймер в _read_stream.
                stall = min(timeout, STALL_S)
                timeout = httpx2.Timeout(timeout, connect=stall, read=stall)
            client = client.with_options(timeout=timeout, max_retries=0)
        if stream:
            kwargs["stream"] = True
            # Без этого в потоковом режиме расход токенов не приходит
            # вообще, и мы перестанем видеть, что кэш промпта работает.
            kwargs["stream_options"] = {"include_usage": True}
        try:
            return client.chat.completions.create(**kwargs)
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
                return client.chat.completions.create(**kwargs)
            raise

    def _step(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        on_sentence: Callable[[str], None] | None,
        max_tokens: int,
        *, cancel_event: threading.Event | None = None,
        timeout: float | None = None,
    ):
        """Один обмен с моделью. Возвращает текст, вызовы и расход токенов.

        Форма ответа одна и та же, потоковый режим или нет, поэтому цикл
        в ask про эту разницу ничего не знает.
        """
        if on_sentence is None:
            response = self._call_api(
                messages, tools, stream=False, max_tokens=max_tokens, timeout=timeout,
            )
            message = response.choices[0].message
            return (
                message.content or "",
                list(getattr(message, "tool_calls", None) or []),
                response.usage,
            )
        started = time.perf_counter()
        stream = self._call_api(messages, tools, stream=True, max_tokens=max_tokens, timeout=timeout)
        return self._read_stream(
            stream,
            on_sentence,
            cancel_event=cancel_event,
            timeout=None if timeout is None else timeout - (time.perf_counter() - started),
        )

    @staticmethod
    def _read_stream(stream, on_sentence: Callable[[str], None], *, cancel_event=None, timeout=None):
        """Читает поток не дольше timeout секунд на весь ответ.

        Таймаут httpx считается между кусками, а не на весь ответ: поток,
        который капает по слову или шлёт пустые «я жив», он не оборвёт.
        Поэтому по сроку поток закрывает отдельный таймер.
        """
        close = getattr(stream, "close", None)
        expired = threading.Event()
        timer = None
        if timeout is not None and close is not None:
            def expire():
                expired.set()
                close()
            timer = threading.Timer(max(timeout, 0.0), expire)
            timer.daemon = True
            timer.start()
        try:
            return Assistant._read_stream_chunks(stream, on_sentence, cancel_event=cancel_event)
        except Exception:
            if expired.is_set():
                raise StepTimeout(f"поток не уложился в {timeout:.0f} с") from None
            raise
        finally:
            if timer is not None:
                timer.cancel()
            if close is not None:
                close()

    @staticmethod
    def _read_stream_chunks(stream, on_sentence: Callable[[str], None], *, cancel_event=None):
        """Читает поток, отдавая законченные предложения по ходу дела.

        Текст отдаём обработчику сразу, но разрешение воспроизведения
        обычного ответа даёт on_step только после финального шага.
        Генератор API закрывается и при завершении, и при отмене.
        """
        pieces: list[str] = []
        buffer = ""
        calls: dict[int, _StreamedCall] = {}
        usage = None
        # Если в тексте пошла служебная разметка - это не ответ, а
        # попытка вызвать инструмент. Озвучивать такое нельзя.
        muted = False

        for chunk in stream:
            check_cancelled(cancel_event)
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
        # То же с латиницей: синтез переведёт её в кириллицу сам, но коряво.
        if re.search(r"[A-Za-z]", answer.text):
            log.warning("в ответе латиница, синтезатор прочитает её криво: %s", answer.text)
        return answer
