"""Два универсальных инструмента для модели: поиск и чтение страницы.

Отдельных инструментов под курс валют, погоду или маршруты намеренно нет.
Модель сама решает, что искать и надо ли открывать страницу. Специализи-
рованный инструмент добавим только если по логам станет видно, что поиск
стабильно не справляется с каким-то частым типом вопросов.
"""

from __future__ import annotations

import logging
import queue
import ssl
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import quote, urlsplit, urlunsplit

import trafilatura
from ddgs import DDGS

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
        hedge_after_s: float = 2.5,
        timeout_s: float = 18.0,
    ) -> None:
        self.region = region
        self.engine = engine
        self.safesearch = safesearch
        self.retries = retries
        self.hedge_after_s = hedge_after_s
        self.timeout_s = timeout_s

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

    @staticmethod
    def _convert(raw: list[dict]) -> list[SearchResult]:
        return [
            SearchResult(
                title=(item.get("title") or "").strip(),
                url=(item.get("href") or "").strip(),
                snippet=(item.get("body") or "").strip(),
            )
            for item in raw
        ]

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        """Ищет, пуская попытки внахлёст, и берёт первый непустой ответ.

        Движки отвечают через раз: один и тот же запрос то приносит
        результаты, то падает с «No results found», причём на всех движках
        сразу. Проверено руками — это не про конкретные слова, а про
        троттлинг на стороне поисковиков.

        Раньше попытки шли одна за другой, и неудача первой стоила всего
        её времени целиком. Замеры показали, почему это дорого: одна
        попытка занимает от двух до десяти секунд при медиане около пяти,
        а падает примерно каждая четвёртая. То есть дело не только в
        отказах, а в разбросе — и лечится он не ожиданием, а запуском
        второй попытки, не дожидаясь первой.

        Поэтому: пускаем первую, и если за `hedge_after_s` ответа нет,
        добавляем следующую, не отменяя предыдущую. Кто ответил первым —
        того и берём. Если попытка падает сразу, следующая стартует
        немедленно, без ожидания.
        """
        attempts = [query] * max(1, self.retries)
        # Последняя попытка идёт с упрощённым запросом: без «какая» и
        # «сколько» поисковики отвечают охотнее.
        simplified = self._simplify(query)
        if simplified != query:
            attempts.append(simplified)

        answers: queue.Queue = queue.Queue()
        started = time.perf_counter()
        # Общий срок на весь поиск. Без него висящая попытка тянет время
        # на себя: замеры дали двадцать секунд там, где ответа всё равно
        # не было. Голосу столько ждать нельзя.
        deadline = started + self.timeout_s

        def run(number: int, text: str) -> None:
            try:
                raw = self._attempt(text, max_results)
            except Exception as exc:  # noqa: BLE001
                answers.put((number, text, None, exc))
            else:
                answers.put((number, text, raw, None))

        def launch(number: int) -> None:
            # Потоки демонские: если ответ уже получен, оставшиеся попытки
            # никого не держат и не мешают процессу закончиться.
            threading.Thread(
                target=run,
                args=(number + 1, attempts[number]),
                name="search-%d" % (number + 1),
                daemon=True,
            ).start()

        launched = 0
        running = 0

        while launched < len(attempts) or running > 0:
            if running == 0:
                # Ждать нечего — либо это самая первая попытка, либо
                # предыдущая уже отвалилась. Пускаем следующую сразу.
                launch(launched)
                launched += 1
                running += 1

            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                log.warning(
                    "поиск %r: срок %.0f с вышел, попыток было %d",
                    query, self.timeout_s, launched,
                )
                break

            more = launched < len(attempts)
            try:
                number, text, raw, exc = answers.get(
                    timeout=min(self.hedge_after_s, remaining) if more else remaining
                )
            except queue.Empty:
                if more:
                    log.info(
                        "поиск %r: ответа нет за %.1f с, пускаю попытку %d внахлёст",
                        query, self.hedge_after_s, launched + 1,
                    )
                    launch(launched)
                    launched += 1
                    running += 1
                    continue
                log.warning(
                    "поиск %r: все попытки повисли, бросаю через %.0f с",
                    query, self.timeout_s,
                )
                break

            running -= 1
            if exc is not None:
                log.warning(
                    "поиск %r попытка %d/%d не удалась (%s): %s",
                    text, number, len(attempts), type(exc).__name__, exc,
                )
                continue
            if raw:
                log.info(
                    "поиск удался с попытки %d за %.2f с, запрос %r",
                    number, time.perf_counter() - started, text,
                )
                return self._convert(raw)
            log.info("поиск %r попытка %d вернулась пустой", text, number)

        # Для ассистента полная неудача — это просто пустой список:
        # он честно скажет, что проверить не удалось.
        log.warning(
            "поиск %r не дал результатов за %d попыток и %.2f с",
            query, launched, time.perf_counter() - started,
        )
        return []


BACKENDS = {
    "ddgs": DdgsBackend,
    # Сюда же встанет searxng, когда поднимем его на VDS (шаг 3).
}


def _make_ssl_context() -> ssl.SSLContext:
    """Контекст с набором корневых сертификатов из certifi.

    Системному хранилищу Windows не хватает промежуточных сертификатов для
    части сайтов - cbr.ru, например, отваливался с CERTIFICATE_VERIFY_FAILED.
    certifi уже стоит как зависимость, и с ним те же сайты открываются.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception as exc:  # noqa: BLE001
        log.warning("certifi недоступен (%s), беру системные сертификаты", exc)
        return ssl.create_default_context()


def encode_url(url: str) -> str:
    """Приводит ссылку к ascii: проценты в пути, punycode в домене.

    Поиск постоянно приносит ссылки с кириллицей - любая статья русской
    Википедии такая. urllib на них падает с UnicodeEncodeError, поэтому
    кодируем сами. Знак процента оставлен безопасным, чтобы уже
    закодированные ссылки не закодировались второй раз.
    """
    parts = urlsplit(url)
    host = parts.netloc
    try:
        host.encode("ascii")
    except UnicodeEncodeError:
        host = host.encode("idna").decode("ascii")

    return urlunsplit(
        (
            parts.scheme,
            host,
            quote(parts.path, safe="/%:@!$&'()*+,;="),
            quote(parts.query, safe="/%:@!$&'()*+,;=?"),
            quote(parts.fragment, safe="/%:@!$&'()*+,;=?"),
        )
    )


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

    # Сколько байт страницы читать максимум. Заодно ограничивает время,
    # которое можно провисеть на сайте, отдающем данные по капле.
    MAX_PAGE_BYTES = 2_000_000

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
        self.fetch_timeout_s = fetch_timeout_s
        self._ssl_context = _make_ssl_context()

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

    def _download(self, url: str) -> bytes | None:
        """Скачивает страницу с жёстким таймаутом и без повторов.

        Качаем сами, а не через trafilatura.fetch_url: та внутри повторяет
        запрос трижды, и на медленном сайте восьмисекундный таймаут
        превращается в двадцать четыре секунды. Проверено на kp.ru - один
        вопрос про погоду вылетел за весь бюджет цикла.
        """
        request = urllib.request.Request(
            encode_url(url),
            headers={
                # Без привычного User-Agent часть сайтов отдаёт заглушку.
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/141.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "ru,en;q=0.8",
            },
        )
        with urllib.request.urlopen(
            request, timeout=self.fetch_timeout_s, context=self._ssl_context
        ) as response:
            return response.read(self.MAX_PAGE_BYTES)

    def fetch_page(self, url: str) -> str:
        started = time.perf_counter()
        try:
            raw = self._download(url)
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
