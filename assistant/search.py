"""Два универсальных инструмента для модели: поиск и чтение страницы.

Отдельных инструментов под курс валют, погоду или маршруты намеренно нет.
Модель сама решает, что искать и надо ли открывать страницу. Специализи-
рованный инструмент добавим только если по логам станет видно, что поиск
стабильно не справляется с каким-то частым типом вопросов.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import trafilatura
from ddgs import DDGS
from trafilatura.settings import use_config

log = logging.getLogger(__name__)

# По этой подстроке и цикл, и прогон тестов отличают «поиск не ответил»
# от «ассистент сам не справился». Менять только вместе с llm.py.
EMPTY_RESULT_MARKER = "ничего не вернул"


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class DdgsBackend:
    """Поиск через ddgs — он опрашивает несколько поисковиков сразу.

    Ни ключей, ни аккаунта, ни тарифа: библиотека ходит на обычные
    поисковики так же, как браузер. Своего индекса у неё нет.
    """

    name = "ddgs"

    # Вопросительные слова в начале запроса поисковику не нужны, а на
    # повторной попытке короткий запрос срабатывает чаще.
    _FILLER = (
        "какая", "какой", "какое", "какие", "сколько", "что", "где",
        "когда", "почему", "скажи", "подскажи", "а",
    )

    def __init__(
        self,
        *,
        region: str = "ru-ru",
        engine: str = "auto",
        safesearch: str = "off",
        retries: int = 2,
        retry_pause_s: float = 0.7,
    ) -> None:
        self.region = region
        self.engine = engine
        self.safesearch = safesearch
        self.retries = retries
        self.retry_pause_s = retry_pause_s

    def _simplify(self, query: str) -> str:
        words = query.rstrip("?！!.").split()
        while words and words[0].lower().strip(",") in self._FILLER:
            words = words[1:]
        return " ".join(words) or query

    def _attempt(self, query: str, max_results: int) -> list[dict]:
        return DDGS().text(
            query,
            region=self.region,
            safesearch=self.safesearch,
            max_results=max_results,
            backend=self.engine,
        )

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        # Движки отвечают через раз: один и тот же запрос то приносит
        # результаты, то падает с «No results found», причём на всех движках
        # сразу. Проверено руками — это не про конкретные слова, а про
        # троттлинг на стороне поисковиков. Поэтому пробуем несколько раз,
        # а на последней попытке упрощаем запрос.
        attempts = [query] * max(1, self.retries)
        simplified = self._simplify(query)
        if simplified != query:
            attempts.append(simplified)

        for number, attempt_query in enumerate(attempts, 1):
            if number > 1:
                time.sleep(self.retry_pause_s)
            try:
                raw = self._attempt(attempt_query, max_results)
            except Exception as exc:
                log.warning(
                    "поиск %r попытка %d/%d не удалась (%s): %s",
                    attempt_query, number, len(attempts), type(exc).__name__, exc,
                )
                continue

            if raw:
                if number > 1:
                    log.info("поиск удался с попытки %d, запрос %r", number, attempt_query)
                return [
                    SearchResult(
                        title=(item.get("title") or "").strip(),
                        url=(item.get("href") or "").strip(),
                        snippet=(item.get("body") or "").strip(),
                    )
                    for item in raw
                ]

        # Для ассистента полная неудача — это просто пустой список:
        # он честно скажет, что проверить не удалось.
        log.warning("поиск %r не дал результатов за %d попыток", query, len(attempts))
        return []


BACKENDS = {
    "ddgs": DdgsBackend,
    # Сюда же встанет searxng, когда поднимем его на VDS (шаг 3).
}


def make_backend(name: str, **options):
    try:
        factory = BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"неизвестный бэкенд поиска {name!r}, доступны: {', '.join(BACKENDS)}"
        ) from None
    return factory(**options)


class SearchTools:
    """Набор инструментов, который видит модель."""

    def __init__(
        self,
        backend,
        *,
        max_results: int = 5,
        page_chars: int = 6000,
        fetch_timeout_s: int = 8,
    ) -> None:
        self.backend = backend
        self.max_results = max_results
        self.page_chars = page_chars

        # Страницу качаем с коротким таймаутом: ассистент отвечает голосом,
        # и ждать медленный сайт дольше нескольких секунд бессмысленно.
        self._fetch_config = use_config()
        self._fetch_config.set("DEFAULT", "DOWNLOAD_TIMEOUT", str(fetch_timeout_s))

    # --- сами инструменты -------------------------------------------------

    def web_search(self, query: str) -> str:
        started = time.perf_counter()
        results = self.backend.search(query, self.max_results)
        log.info(
            "web_search(%r) через %s: %d результатов за %.2f с",
            query, self.backend.name, len(results), time.perf_counter() - started,
        )

        if not results:
            return (
                f"Поиск по запросу «{query}» {EMPTY_RESULT_MARKER}. "
                "Попробуй переформулировать запрос или скажи, что проверить не удалось."
            )

        lines = [f"Результаты поиска по запросу «{query}»:", ""]
        for n, item in enumerate(results, 1):
            lines.append(f"{n}. {item.title}")
            lines.append(f"   {item.url}")
            if item.snippet:
                lines.append(f"   {item.snippet}")
            lines.append("")
        return "\n".join(lines).strip()

    def fetch_page(self, url: str) -> str:
        started = time.perf_counter()
        try:
            raw = trafilatura.fetch_url(url, config=self._fetch_config)
        except Exception as exc:
            log.warning("fetch_page(%r) упал: %s: %s", url, type(exc).__name__, exc)
            raw = None

        if raw is None:
            log.info(
                "fetch_page(%r): не скачалось за %.2f с",
                url, time.perf_counter() - started,
            )
            return (
                f"Страницу {url} открыть не удалось: она недоступна или "
                "закрыта от автоматического чтения. Попробуй другую ссылку."
            )

        # include_tables обязательно: на страницах с курсами, расписаниями и
        # ценами вся суть лежит именно в таблицах, без них остаётся обвязка.
        text = trafilatura.extract(
            raw, include_tables=True, include_comments=False, favor_recall=True
        ) or ""
        text = text.strip()
        log.info(
            "fetch_page(%r): %d символов за %.2f с",
            url, len(text), time.perf_counter() - started,
        )

        if not text:
            return (
                f"Со страницы {url} не удалось вытащить текст — похоже, "
                "содержимое рисуется скриптами. Попробуй другую ссылку."
            )

        if len(text) > self.page_chars:
            text = text[: self.page_chars] + "\n\n(текст обрезан)"
        return f"Текст страницы {url}:\n\n{text}"

    # --- то, что видит модель ---------------------------------------------

    SCHEMAS = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Поиск в интернете. Возвращает заголовки, ссылки и короткие "
                    "выдержки. Обязательно используй для всего, что меняется со "
                    "временем: курсы валют, погода, новости, цены, наличие товаров, "
                    "расписания транспорта, время в пути, события."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Поисковый запрос обычными словами, как написал бы человек."
                            ),
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_page",
                "description": (
                    "Скачивает страницу по ссылке и возвращает её текст. "
                    "Используй, когда в выдержках из поиска нет нужной точной "
                    "цифры или факта, но видно, что на странице он есть."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Ссылка из результатов поиска.",
                        }
                    },
                    "required": ["url"],
                },
            },
        },
    ]

    def call(self, name: str, arguments: dict) -> str:
        if name == "web_search":
            query = (arguments.get("query") or "").strip()
            if not query:
                return "Инструменту web_search нужен непустой запрос."
            return self.web_search(query)
        if name == "fetch_page":
            url = (arguments.get("url") or "").strip()
            if not url.startswith(("http://", "https://")):
                return (
                    "Инструменту fetch_page нужна ссылка, "
                    "начинающаяся с http или https."
                )
            return self.fetch_page(url)
        return f"Инструмента {name} нет."
