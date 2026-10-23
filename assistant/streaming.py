"""Отмена сетевого ответа. Общий лёгкий модуль, без моделей и звука."""

import threading
from collections import OrderedDict


class AnswerCancelled(Exception):
    """Запрос остановлен пользователем; это не ошибка модели."""


def check_cancelled(event: threading.Event | None) -> None:
    if event is not None and event.is_set():
        raise AnswerCancelled


class Requests:
    """Отмена привязана к паре сессия/запрос и не ждёт замка моделей."""

    def __init__(self):
        self.lock = threading.Lock()
        self.active = {}
        self.early = OrderedDict()

    def start(self, session: str, request: str) -> threading.Event:
        key = (session, request)
        with self.lock:
            if key in self.active or len(self.active) >= 128:
                raise ValueError("запрос уже выполняется или очередь заполнена")
            event = threading.Event()
            if self.early.pop(key, False):
                event.set()
            self.active[key] = event
            return event

    def cancel(self, session: str, request: str) -> None:
        key = (session, request)
        with self.lock:
            event = self.active.get(key)
            if event is not None:
                event.set()
            else:
                # Стоп может прийти раньше самого запроса. Число таких
                # отметок ограничено; UUID запросов никогда не переиспользуем.
                self.early[key] = True
                if len(self.early) > 128:
                    self.early.popitem(last=False)

    def finish(self, session: str, request: str) -> None:
        with self.lock:
            self.active.pop((session, request), None)
