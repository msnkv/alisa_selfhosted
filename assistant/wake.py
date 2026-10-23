"""Wake word: ассистент просыпается на кодовую фразу вместо клавиши.

Готовые модели openWakeWord все англоязычные, поэтому по умолчанию стоит
`hey_jarvis`. Своё русское слово — это обучение модели, оно в планах
этапа 4 и здесь не делается.

Почему это оставлено на последний шаг: тут придётся подбирать порог,
ловить ложные срабатывания от телевизора и разговоров, и заниматься этим
гораздо приятнее, когда всё остальное уже работает.
"""

from __future__ import annotations

import logging
import queue
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# openWakeWord ждёт кадры по 1280 отсчётов - это 80 мс на 16 кГц.
CHUNK = 1280

# Что доступно без обучения своей модели.
BUILTIN_MODELS = ("hey_jarvis", "alexa", "hey_mycroft", "hey_rhasspy")


def open_stream(device, blocksize: int):
    """Поток с микрофона: целые числа, одна дорожка, 16 кГц.

    Общий для обоих бэкендов. Поток открывается заново на каждое
    ожидание — так выбрасывается звук, накопившийся пока ассистент
    говорил, и он не будит сам себя через колонки.
    """
    import sounddevice as sd

    chunks: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, _frames, _time_info, status):
        if status:
            log.debug("микрофон кодовой фразы: %s", status)
        chunks.put(indata[:, 0].copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=blocksize,
        device=device,
        callback=callback,
    )
    return stream, chunks


class WakeWord:
    name = "openwakeword"

    def __init__(
        self,
        *,
        model: str = "hey_jarvis",
        threshold: float = 0.5,
        vad_threshold: float = 0.5,
        device: int | str | None = None,
        cooldown_s: float = 1.0,
    ) -> None:
        self.model_name = model
        self.threshold = threshold
        self.vad_threshold = vad_threshold
        self.device = device
        self.cooldown_s = cooldown_s
        self._model = None

    @property
    def spoken(self) -> str:
        """Как позвать ассистента — для подсказки на экране."""
        return self.model_name.replace("_", " ")

    # --- загрузка ---------------------------------------------------------

    @staticmethod
    def _ensure_models(model_name: str) -> None:
        """Докачивает модель и общие слои, если их ещё нет.

        Кладутся они внутрь venv, в resources/models самого openwakeword.
        Это чужая папка, зато не нужно таскать пути: модель потом
        запрашивается по имени. Вес около пяти мегабайт, так что потерять
        их при пересборке venv не страшно.
        """
        import openwakeword
        import openwakeword.utils as utils

        resources = Path(openwakeword.__file__).parent / "resources" / "models"
        needed = [
            resources / "melspectrogram.onnx",
            resources / "embedding_model.onnx",
        ]
        have_wakeword = any(resources.glob(f"{model_name}*.onnx"))

        if all(path.exists() for path in needed) and have_wakeword:
            return

        log.info("докачиваю модели wake word (%s)", model_name)
        utils.download_models([model_name])

    @property
    def model(self):
        if self._model is None:
            from openwakeword.model import Model

            self._ensure_models(self.model_name)
            started = time.perf_counter()
            self._model = Model(
                wakeword_models=[self.model_name],
                # На Windows tflite не поддерживается, только onnx.
                # По умолчанию библиотека просит tflite и падает.
                inference_framework="onnx",
                # Встроенный VAD отсекает срабатывания на музыке и шуме:
                # без него телевизор будит ассистента заметно чаще.
                vad_threshold=self.vad_threshold,
            )
            log.info(
                "wake word %s загружен за %.2f с (порог %.2f, vad %.2f)",
                self.model_name, time.perf_counter() - started,
                self.threshold, self.vad_threshold,
            )
        return self._model

    def warm_up(self) -> float:
        started = time.perf_counter()
        self.model.predict(np.zeros(CHUNK, dtype=np.int16))
        self.model.reset()
        return time.perf_counter() - started

    # --- ожидание ---------------------------------------------------------

    def _stream(self):
        return open_stream(self.device, CHUNK)

    def wait(self, *, timeout_s: float | None = None) -> bool:
        """Ждёт кодовую фразу. True — услышал, False — вышло время.

        Поток открывается заново на каждое ожидание: так заодно
        выбрасывается звук, накопившийся пока ассистент говорил, и он
        не будит сам себя через колонки.
        """
        self.model.reset()
        stream, chunks = self._stream()
        started = time.perf_counter()

        with stream:
            while True:
                if timeout_s is not None and time.perf_counter() - started > timeout_s:
                    return False
                try:
                    frame = chunks.get(timeout=1.0)
                except queue.Empty:
                    log.warning("микрофон молчит - данные не идут")
                    continue

                scores = self.model.predict(frame)
                best = max(scores.values()) if scores else 0.0
                if best >= self.threshold:
                    log.info("wake word услышан, уверенность %.2f", best)
                    self.model.reset()
                    return True

    def monitor(self, seconds: float = 30.0, report_from: float = 0.05) -> None:
        """Печатает уверенность модели - для подбора порога.

        Полезно, чтобы посмотреть, что даёт твой голос, а что телевизор
        и обычный разговор в комнате.
        """
        stream, chunks = self._stream()
        started = time.perf_counter()
        peak = 0.0
        triggers = 0

        print(
            f"Слушаю {seconds:.0f} с. Скажи «{self.model_name.replace('_', ' ')}»"
            f" несколько раз. Порог {self.threshold}."
        )
        with stream:
            while time.perf_counter() - started < seconds:
                try:
                    frame = chunks.get(timeout=1.0)
                except queue.Empty:
                    continue
                scores = self.model.predict(frame)
                best = max(scores.values()) if scores else 0.0
                peak = max(peak, best)
                if best >= self.threshold:
                    triggers += 1
                    print(f"  {time.perf_counter()-started:5.1f} с  СРАБОТАЛО  {best:.2f}")
                    self.model.reset()
                elif best >= report_from:
                    print(f"  {time.perf_counter()-started:5.1f} с  близко     {best:.2f}")

        print(f"\nМаксимум {peak:.2f}, срабатываний {triggers}.")
        if triggers == 0 and peak > 0.2:
            print("Близко, но порога не хватило — попробуй опустить wake.threshold.")
        elif triggers == 0:
            print("Не услышал. Проверь микрофон (--mic) и произношение.")


class VoskWakeWord:
    """Кодовая фраза через распознавание с ограниченной грамматикой.

    Зачем, если есть openWakeWord: там все готовые модели английские, а
    своя русская — это обучение классификатора на синтетике, день работы
    с неясным исходом (см. CLAUDE.md). Здесь обучать нечего: берём
    маленькую русскую модель распознавания и говорим ей, что в эфире
    возможны только наши варианты фразы и «что-то другое». На такой
    грамматике декодер почти не ошибается и работает заметно быстрее,
    чем полноценное распознавание.

    Цена по сравнению с openWakeWord: это настоящее распознавание, а не
    крошечный классификатор, так что процессора оно ест больше. Зато
    фраза русская и меняется строкой в конфиге.
    """

    name = "vosk"

    # Восьмая секунды. На четверти отклик был заметно медленнее, а
    # нагрузка всё равно ничтожная — измерено, один процент ядра.
    CHUNK = 2000

    def __init__(
        self,
        *,
        phrase: str = "алиса тян",
        variants: tuple[str, ...] = (),
        model_dir: Path | None = None,
        language: str = "ru",
        min_confidence: float = 0.8,
        device: int | str | None = None,
        cooldown_s: float = 1.0,
    ) -> None:
        self.phrase = phrase.strip().lower()
        # Распознавание слышит фразу по-разному, поэтому принимаем
        # несколько написаний. Сама фраза всегда в списке.
        self.variants = tuple(
            dict.fromkeys([self.phrase, *(v.strip().lower() for v in variants if v)])
        )
        self.model_dir = model_dir
        self.language = language
        self.min_confidence = min_confidence
        self.device = device
        self.cooldown_s = cooldown_s
        self._model = None

    @property
    def spoken(self) -> str:
        return self.phrase

    # --- загрузка ---------------------------------------------------------

    @property
    def model(self):
        if self._model is None:
            from vosk import Model, SetLogLevel

            # Без этого kaldi сыплет в stderr десятки строк на загрузке.
            SetLogLevel(-1)
            started = time.perf_counter()
            if self.model_dir and self.model_dir.is_dir():
                self._model = Model(model_path=str(self.model_dir))
                where = str(self.model_dir)
            else:
                # Сама скачает маленькую модель языка в кэш пользователя.
                self._model = Model(lang=self.language)
                where = "кэш vosk"
            log.info(
                "модель vosk (%s) загружена за %.2f с из %s",
                self.language, time.perf_counter() - started, where,
            )
        return self._model

    def _recognizer(self):
        import json

        from vosk import KaldiRecognizer

        # Грамматика — главное здесь. Декодер выбирает только из наших
        # вариантов и «[unk]», то есть всё остальное. Без неё это было бы
        # обычное распознавание: медленнее и с кучей похожих слов.
        grammar = json.dumps([*self.variants, "[unk]"], ensure_ascii=False)
        recognizer = KaldiRecognizer(self.model, SAMPLE_RATE, grammar)
        # Без этого не будет уверенности по словам, а без неё всё
        # разваливается — см. heard_phrase.
        recognizer.SetWords(True)
        return recognizer

    def warm_up(self) -> float:
        started = time.perf_counter()
        recognizer = self._recognizer()
        recognizer.AcceptWaveform(bytes(self.CHUNK * 2))
        return time.perf_counter() - started

    # --- распознавание ----------------------------------------------------

    def check(self, result: dict) -> tuple[bool, str, float]:
        """Это наша фраза? Возвращает решение, что услышано и уверенность.

        Самое важное место во всём бэкенде. Одной грамматики
        НЕДОСТАТОЧНО, и это проверено замером: если разрешить декодеру
        только «алиса тян» и «что-то другое», он начинает укладывать в
        нашу фразу вообще всё подряд. На десяти посторонних фразах
        получилось шесть ложных срабатываний — «где лежат болгарки» и
        «маша, ты не видела мои ключи» обе слышались как «алиса тян».

        Спасает уверенность по словам. С порогом ноль восемь на тех же
        десяти фразах ложных не осталось ни одного, а все пять
        правильных по-прежнему срабатывают. Поэтому `SetWords(True)` и
        проверка `conf` — не украшение, их нельзя убирать.

        Плата: уверенность есть только в окончательном результате, не в
        промежуточном. То есть решение принимается, когда распознавание
        сочло фразу законченной. Замерено: для одиночной «Алиса-тян» это
        полсекунды после того, как человек договорил.
        """
        said = (result.get("text") or "").strip().lower()
        if not said:
            return False, "", 0.0

        # Слов нет — значит SetWords не сработал, уверенности взять
        # негде. Лучше не будить, чем будить вслепую.
        words = result.get("result") or []
        if not words:
            return False, said, 0.0

        spoken = [word.get("word", "") for word in words]
        best = None
        for variant in self.variants:
            needed = variant.split()
            for start in range(len(spoken) - len(needed) + 1):
                if spoken[start:start + len(needed)] != needed:
                    continue
                # Фраза должна стоять В НАЧАЛЕ реплики: перед ней может
                # быть только «что-то другое». Кодовую фразу говорят
                # первой, а вот посторонняя речь, куда декодер случайно
                # вписал наши слова, чаще всего имеет их в середине.
                # Замер: без этого правила «вчера ходил в магазин за
                # хлебом» слышалось как «тян [unk] алиса тян» и будило
                # ассистента.
                if any(word != "[unk]" for word in spoken[:start]):
                    continue
                window = words[start:start + len(needed)]
                value = min(word.get("conf", 0.0) for word in window)
                best = value if best is None else max(best, value)

        if best is None:
            return False, said, 0.0
        if best >= self.min_confidence:
            return True, said, best
        log.debug("похоже на фразу, но неуверенно: %r (%.2f)", said, best)
        return False, said, best

    def _read(self, recognizer, frame: np.ndarray) -> dict | None:
        """Окончательный результат кадра, если он готов."""
        import json

        if not recognizer.AcceptWaveform(frame.tobytes()):
            return None
        return json.loads(recognizer.Result())

    def wait(self, *, timeout_s: float | None = None) -> bool:
        recognizer = self._recognizer()
        stream, chunks = open_stream(self.device, self.CHUNK)
        started = time.perf_counter()

        with stream:
            while True:
                if timeout_s is not None and time.perf_counter() - started > timeout_s:
                    return False
                try:
                    frame = chunks.get(timeout=1.0)
                except queue.Empty:
                    log.warning("микрофон молчит - данные не идут")
                    continue
                result = self._read(recognizer, frame)
                if result is None:
                    continue
                heard, said, confidence = self.check(result)
                if heard:
                    log.info(
                        "кодовая фраза услышана: %r (уверенность %.2f)",
                        said, confidence,
                    )
                    return True

    def monitor(self, seconds: float = 30.0, report_from: float = 0.0) -> None:
        """Печатает, что слышит. Аналог подбора порога у openWakeWord.

        Порога тут нет, поэтому полезно другое: видеть, каким именно
        текстом распознаётся твоя фраза. Если она стабильно слышится
        иначе — это написание надо добавить в `wake.variants`.
        """
        recognizer = self._recognizer()
        stream, chunks = open_stream(self.device, self.CHUNK)
        started = time.perf_counter()
        triggers = 0
        seen: set[str] = set()
        # Для сводки в конце: уверенность попаданий, близких промахов
        # и сколько раз распозналось что-то постороннее.
        hits: list[float] = []
        misses: list[float] = []
        other: list[str] = []

        print(
            f"Слушаю {seconds:.0f} с. Скажи «{self.phrase}» несколько раз,"
            f" потом поговори о другом.\nПорог уверенности"
            f" {self.min_confidence}, варианты: {', '.join(self.variants)}"
        )
        with stream:
            while time.perf_counter() - started < seconds:
                try:
                    frame = chunks.get(timeout=1.0)
                except queue.Empty:
                    continue
                result = self._read(recognizer, frame)
                if result is None:
                    continue
                heard, said, confidence = self.check(result)
                if not said or said in seen:
                    continue
                seen.add(said)
                moment = time.perf_counter() - started
                if heard:
                    triggers += 1
                    hits.append(confidence)
                    print(f"  {moment:5.1f} с  СРАБОТАЛО  {said!r}  {confidence:.2f}")
                    recognizer = self._recognizer()
                    seen.clear()
                elif confidence:
                    misses.append(confidence)
                    print(f"  {moment:5.1f} с  близко     {said!r}  {confidence:.2f}")
                else:
                    other.append(said)
                    print(f"  {moment:5.1f} с  слышу      {said!r}")

        print()
        if hits:
            lowest, highest = min(hits), max(hits)
            print(
                f"Срабатываний {triggers}. Уверенность твоего голоса:"
                f" от {lowest:.2f} до {highest:.2f}."
            )
            # Советуем порог с запасом вниз от худшего попадания, но не
            # ниже 0.6: на замерах при 0.5 начинали лезть ложные.
            advice = max(0.6, round(lowest - 0.1, 2))
            if lowest < self.min_confidence:
                print(
                    f"Часть попаданий ниже порога {self.min_confidence:.2f}"
                    f" — поставь wake.min_confidence {advice:.2f}."
                )
            elif advice > self.min_confidence + 0.05:
                print(
                    f"Запас большой: порог можно поднять до {advice:.2f},"
                    " ложных станет ещё меньше."
                )
            else:
                print(f"Порог {self.min_confidence:.2f} подобран нормально.")
        else:
            print("Срабатываний нет.")

        if misses:
            best = max(misses)
            print(
                f"Близких промахов {len(misses)}, лучший {best:.2f}"
                f" при пороге {self.min_confidence:.2f}."
            )
            if best >= 0.65:
                print(f"Опусти wake.min_confidence до {best - 0.05:.2f}.")
            else:
                print(
                    "Это низко, и порог тут не поможет. Скажи фразу и чуть"
                    " передохни перед вопросом: слитная речь после неё"
                    " сбивает распознавание."
                )

        if not hits and not misses:
            print(
                "Фраза не распозналась ни разу. Посмотри строки «слышу»:"
                " если она стабильно звучит иначе — добавь это написание"
                " в wake.variants. Если строк нет совсем — проверь"
                " микрофон командой --mic."
            )
        if other:
            print(f"Посторонней речи распозналось: {len(other)} раз(а).")


def make_wake_word(cfg) -> WakeWord:
    if cfg.wake.engine == "vosk":
        return VoskWakeWord(
            phrase=cfg.wake.phrase,
            variants=tuple(cfg.wake.variants),
            model_dir=cfg.wake.model_dir,
            language=cfg.wake.language,
            min_confidence=cfg.wake.min_confidence,
            device=cfg.wake.input_device,
            cooldown_s=cfg.wake.cooldown_s,
        )
    if cfg.wake.engine != "openwakeword":
        raise ValueError(
            f"неизвестный движок кодовой фразы {cfg.wake.engine!r}, "
            f"доступны: openwakeword, vosk"
        )
    return WakeWord(
        model=cfg.wake.model,
        threshold=cfg.wake.threshold,
        vad_threshold=cfg.wake.vad_threshold,
        device=cfg.wake.input_device,
        cooldown_s=cfg.wake.cooldown_s,
    )
