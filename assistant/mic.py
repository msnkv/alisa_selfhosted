"""Микрофон: запись фразы с определением, что человек договорил.

Ждать нажатия второй клавиши, чтобы остановить запись, неудобно — в
комнате руки заняты. Поэтому запись останавливается сама, когда человек
замолчал. Решает это silero-vad: крошечная модель, которая по окну из
пятисот двенадцати отсчётов говорит вероятность речи. Она же понадобится
на шаге 6, когда клавиша исчезнет совсем.
"""

from __future__ import annotations

import logging
import queue
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Whisper и silero-vad оба работают на шестнадцати килогерцах.
SAMPLE_RATE = 16000
# silero-vad требует ровно столько отсчётов в окне.
FRAME = 512
FRAME_SECONDS = FRAME / SAMPLE_RATE

# Сколько ждать первый кадр после открытия микрофона. Поймано на
# ноутбуке-спутнике: сразу после того, как отзвучал «да?» или ответ,
# Windows иногда отдаёт первый кадр позже секунды, и запись сдавалась с
# «микрофон молчит», не дав сказать ни слова. Посреди записи по-прежнему
# хватает секунды тишины в данных, чтобы считать микрофон отвалившимся.
FIRST_FRAME_S = 3.0
FRAME_TIMEOUT_S = 1.0
# Не пришёл первый кадр — переоткрываем поток столько раз, потом сдаёмся.
REOPEN_ATTEMPTS = 1

# Фон для порога громкости меряем по медиане первых кадров, а не по
# одному первому: см. EnergyVad. Десять кадров — около трети секунды.
CALIBRATION_FRAMES = 10
# Тише этого — цифровой ноль, которым поток иногда начинается, а не
# тишина в комнате. Такие кадры в замер фона не берём.
SILENT_LEVEL = 1e-6

SILERO_VAD_REPO = "snakers4/silero-vad"


class StopWord(Exception):
    """Человек сказал «стоп» вместо вопроса: запись брошена, ждём кодовую фразу."""


def _db(level: float) -> str:
    """Уровень в децибелах: читать проще, чем 0.0043 против 0.0128."""
    if level <= 0:
        return "-inf"
    return f"{20 * np.log10(level):.0f}"


def _bar(level: float, width: int = 24) -> str:
    """Полоска громкости от -60 дБ до нуля."""
    if level <= 0:
        filled = 0
    else:
        filled = int(round(width * max(0.0, (20 * np.log10(level) + 60) / 60)))
    filled = min(width, filled)
    return "█" * filled + "·" * (width - filled)


class SileroVad:
    """Вероятность речи в окне. Модель весит около двух мегабайт."""

    name = "silero-vad"

    def __init__(self, models_dir: Path | None = None) -> None:
        import torch

        if models_dir is not None:
            models_dir.mkdir(parents=True, exist_ok=True)
            torch.hub.set_dir(str(models_dir))

        self._torch = torch
        self._model, _ = torch.hub.load(
            repo_or_dir=SILERO_VAD_REPO,
            model="silero_vad",
            trust_repo=True,
        )
        self._model.eval()

    def reset(self) -> None:
        # У модели есть внутреннее состояние: без сброса начало новой
        # фразы оценивается с оглядкой на предыдущую.
        self._model.reset_states()

    def probability(self, frame: np.ndarray) -> float:
        with self._torch.no_grad():
            tensor = self._torch.from_numpy(frame)
            return float(self._model(tensor, SAMPLE_RATE).item())


class EnergyVad:
    """Запасной вариант: громкость вместо модели.

    Нужен, если silero-vad не скачался (нет сети, заблокирован GitHub),
    и на спутнике, где torch ставить незачем.

    **Порог плавающий, и это не украшение, а исправление поломки.**
    Раньше тут стояло жёсткое значение 0,01, и оно наступило на живого
    человека: с пяти метров от ноутбука речь даёт уровень в разы ниже,
    чем та же речь в гарнитуру. Ассистент при этом не молчал — он писал
    звук, но не считал его речью, поэтому «говоришь...» не появлялось и
    фраза не отправлялась, пока человек не повышал голос. А в записи,
    которая в итоге уезжала на сервер, тихая часть была — её же потом
    распознавали целиком. Поломка выглядела как «слышит, но не
    реагирует».

    Фиксированный порог тут и не может работать: громкость зависит от
    расстояния, микрофона и комнаты, а не от того, речь это или нет.
    Поэтому меряется уровень тишины (фон), и речью считается то, что
    громче фона в `ratio` раз. Фон отслеживается постоянно: вниз
    быстро, вверх медленно — иначе собственная речь подняла бы его и
    сама себя заглушила.

    `floor_min` — нижняя граница порога. Без неё в полной тишине (фон
    около нуля) речью становится любой шорох.

    **Начальный фон — медиана первой трети секунды, а не первый кадр.**
    Поймано на чайнике 24.10.2026: после ответа ассистент принял шум
    кипящего чайника за речь и ушёл в цикл «не разобрала». Фон брался
    по одному кадру, а поток сразу после открытия бывает тише комнаты,
    вплоть до цифрового нуля. Фон выходил заниженным, а вверх он
    растёт медленно — секунд пятнадцать, — и ровный шум всё это время
    был «громче фона вдвое». Пока замер идёт, речью ничего не
    считается; звук при этом пишется, так что начало фразы не теряется.
    Если человек заговорил сразу и медиана вышла завышенной, она быстро
    опустится на первой же паузе между словами.
    """

    name = "громкость"

    def __init__(
        self,
        ratio: float = 2.0,
        floor_min: float = 0.0015,
        fixed: float | None = None,
    ) -> None:
        self.ratio = ratio
        self.floor_min = floor_min
        # Если задан - плавающий порог выключен. На случай, если в
        # какой-то комнате окажется, что так лучше.
        self.fixed = fixed
        self.floor = None
        self._first: list[float] = []
        self.level = 0.0
        self.threshold = fixed if fixed is not None else floor_min

    def reset(self) -> None:
        # Фон меряем заново на каждой фразе: между фразами в комнате
        # могло измениться что угодно.
        self.floor = None
        self._first = []

    def probability(self, frame: np.ndarray) -> float:
        level = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        self.level = level

        if self.floor is None:
            if level > SILENT_LEVEL:
                self._first.append(level)
            if len(self._first) >= CALIBRATION_FRAMES:
                self.floor = float(np.median(self._first))
        elif level < self.floor:
            self.floor += 0.2 * (level - self.floor)
        else:
            self.floor += 0.002 * (level - self.floor)

        if self.fixed is None:
            if self.floor is None:
                return 0.0
            self.threshold = max(self.floor_min, self.floor * self.ratio)
        return 1.0 if level > self.threshold else 0.0


def make_vad(
    models_dir: Path | None = None,
    engine: str = "auto",
    energy: dict | None = None,
):
    """Чем определять конец фразы.

    `silero` точнее, но тянет torch — полтора гигабайта ради одной
    маленькой модели. На спутнике это лишнее, поэтому там ставят
    `energy`: по громкости, без единой тяжёлой зависимости.

    `auto` пробует silero и молча откатывается, если torch нет.
    """
    settings = energy or {}
    if engine == "energy":
        return EnergyVad(**settings)
    try:
        return SileroVad(models_dir)
    except Exception as exc:  # noqa: BLE001
        if engine == "silero":
            raise
        log.warning(
            "silero-vad не загрузился (%s: %s), определяю конец фразы по громкости",
            type(exc).__name__, exc,
        )
        return EnergyVad(**settings)


class Microphone:
    def __init__(
        self,
        *,
        device: int | str | None = None,
        speech_threshold: float = 0.5,
        silence_s: float = 0.8,
        min_speech_s: float = 0.3,
        max_phrase_s: float = 15.0,
        wait_speech_s: float = 6.0,
        models_dir: Path | None = None,
        vad_engine: str = "auto",
        energy: dict | None = None,
    ) -> None:
        self.device = device
        self.speech_threshold = speech_threshold
        self.silence_s = silence_s
        self.min_speech_s = min_speech_s
        self.max_phrase_s = max_phrase_s
        self.wait_speech_s = wait_speech_s
        self.models_dir = models_dir
        self.vad_engine = vad_engine
        self.energy = energy
        self._vad = None
        # Ловец «стоп» отдельной фразой (voice_stop.PhraseStop). Ставит
        # цикл с Vosk-кодовой фразой: модель уже загружена ради неё.
        self.stop_word = None

    @property
    def vad(self):
        if self._vad is None:
            self._vad = make_vad(self.models_dir, self.vad_engine, self.energy)
            log.info("определение конца фразы: %s", self._vad.name)
        return self._vad

    def warm_up(self) -> float:
        started = time.perf_counter()
        self.vad.probability(np.zeros(FRAME, dtype=np.float32))
        return time.perf_counter() - started

    def record_phrase(
        self, *, on_speech_start=None, wait_speech_s: float | None = None
    ) -> np.ndarray:
        """Пишет с микрофона, пока человек не договорит.

        Возвращает запись целиком, включая тишину в начале: обрезать её
        незачем, у whisper для этого свой фильтр.
        Пустой массив означает, что речи так и не услышали. «Стоп»
        отдельной фразой бросает запись исключением StopWord.
        """
        import sounddevice as sd

        # Для уточнения без кодовой фразы ждать нужно меньше, чем на
        # обычном круге: если человек молчит, надо быстро вернуться к
        # ожиданию кодовой фразы, а не висеть с открытым микрофоном.
        wait_for_speech = (
            self.wait_speech_s if wait_speech_s is None else wait_speech_s
        )
        self.vad.reset()
        stop_word = self.stop_word
        if stop_word is not None:
            stop_word.reset()
        chunks: queue.Queue[np.ndarray] = queue.Queue()

        def heard_stop(check) -> bool:
            # Сбой распознавания не должен ронять запись вопроса:
            # отключаем «стоп» до перезапуска и пишем дальше.
            nonlocal stop_word
            if stop_word is None:
                return False
            try:
                return check(stop_word)
            except Exception as exc:  # noqa: BLE001
                log.warning("«стоп» при записи отключён: %s: %s", type(exc).__name__, exc)
                stop_word = self.stop_word = None
                return False

        def callback(indata, _frames, _time_info, status):
            if status:
                log.debug("микрофон: %s", status)
            chunks.put(indata[:, 0].copy())

        def open_stream():
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=FRAME,
                device=self.device,
                callback=callback,
            )
            stream.start()
            return stream

        # Первый кадр ждём отдельно и дольше: запуск микрофона сразу после
        # звука из динамика бывает медленным, а это не поломка.
        stream = None
        first_frame = None
        for attempt in range(1 + REOPEN_ATTEMPTS):
            opened = time.perf_counter()
            stream = open_stream()
            try:
                first_frame = chunks.get(timeout=FIRST_FRAME_S)
            except queue.Empty:
                stream.close()
                log.warning(
                    "микрофон молчит %.0f с после открытия, попытка %d из %d",
                    FIRST_FRAME_S, attempt + 1, 1 + REOPEN_ATTEMPTS,
                )
                continue
            log.info(
                "первый кадр с микрофона через %.2f с%s",
                time.perf_counter() - opened,
                " (после переоткрытия)" if attempt else "",
            )
            break
        if first_frame is None:
            log.warning("микрофон молчит - данные не идут")
            return np.zeros(0, dtype=np.float32)

        frames: list[np.ndarray] = []
        speech_seconds = 0.0
        silence_seconds = 0.0
        heard_speech = False
        # Время на «начни говорить» отсчитываем с момента, когда микрофон
        # заработал: ожидание его запуска не должно съедать это окно.
        started = time.perf_counter()

        # Не `with stream`: он запускает поток ещё раз, а этот уже запущен.
        try:
            while True:
                if first_frame is not None:
                    frame, first_frame = first_frame, None
                else:
                    try:
                        frame = chunks.get(timeout=FRAME_TIMEOUT_S)
                    except queue.Empty:
                        log.warning("микрофон молчит - данные не идут")
                        break

                frames.append(frame)
                if heard_stop(lambda s: s.feed(frame)):
                    raise StopWord
                is_speech = self.vad.probability(frame) >= self.speech_threshold

                if is_speech:
                    speech_seconds += FRAME_SECONDS
                    silence_seconds = 0.0
                    if not heard_speech and speech_seconds >= self.min_speech_s:
                        heard_speech = True
                        if on_speech_start is not None:
                            on_speech_start()
                elif heard_speech:
                    silence_seconds += FRAME_SECONDS
                    if silence_seconds >= self.silence_s:
                        break

                elapsed = time.perf_counter() - started
                if heard_speech and elapsed >= self.max_phrase_s:
                    log.info("фраза оборвана по лимиту %.0f с", self.max_phrase_s)
                    break
                if not heard_speech and elapsed >= wait_for_speech:
                    log.info("речи не услышал за %.0f с", wait_for_speech)
                    return np.zeros(0, dtype=np.float32)
        finally:
            stream.close()

        # Пауза конца фразы (1,3 с) обычно длиннее, чем нужно Vosk, но
        # если запись кончилась первой — «стоп» мог остаться в хвосте.
        if heard_stop(lambda s: s.finish()):
            raise StopWord

        audio =np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
        log.info(
            "записано %.2f с, из них речи %.2f с",
            len(audio) / SAMPLE_RATE, speech_seconds,
        )
        return audio

    def monitor(self, seconds: float | None = None) -> int:
        """Живые цифры с микрофона: что слышно и считается ли это речью.

        Сделано по той же причине, что и `--wake-test`: когда
        ассистент «слышит, но не отвечает», глазами в комнате не
        понять, чего не хватило — громкости, порога или времени.
        Здесь видно всё сразу: уровень, текущий порог и те самые
        моменты, когда появилось бы «говоришь...».

        Решения принимаются ровно тем же кодом, что в `record_phrase`:
        тот же кадр, тот же порог, те же `min_speech_s` и `silence_s`.
        Иначе проверять было бы нечего.
        """
        import sounddevice as sd

        # Печатаем не каждый кадр: тридцать строк в секунду не читаются.
        every = 8  # кадров, то есть примерно четверть секунды

        print(f"Слушаю микрофон ({self.vad.name}). Ctrl+C — закончить.")
        print("Говори с того места, где будешь стоять на самом деле.\n")
        print("  время   уровень                        дБ   порог   речь")

        chunks: queue.Queue[np.ndarray] = queue.Queue()

        def callback(indata, _frames, _time_info, status):
            if status:
                log.debug("микрофон: %s", status)
            chunks.put(indata[:, 0].copy())

        self.vad.reset()
        levels: list[float] = []
        speech_levels: list[float] = []
        phrases = 0
        speech_frames = 0
        total_frames = 0
        speech_seconds = 0.0
        silence_seconds = 0.0
        heard_speech = False
        phrase_started = 0.0
        batch: list[tuple[float, bool, float]] = []

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=FRAME,
                device=self.device,
                callback=callback,
            ):
                while True:
                    try:
                        frame = chunks.get(timeout=1.0)
                    except queue.Empty:
                        if total_frames:
                            break
                        print("  микрофон молчит — данные не идут вообще")
                        return 1

                    probability = self.vad.probability(frame)
                    is_speech = probability >= self.speech_threshold
                    level = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
                    threshold = getattr(self.vad, "threshold", None)

                    levels.append(level)
                    total_frames += 1
                    if is_speech:
                        speech_frames += 1
                        speech_levels.append(level)
                    batch.append((level, is_speech, threshold or 0.0))

                    # Время считаем по кадрам, а не по часам: так оно
                    # совпадает с длиной звука, и проверять монитор
                    # можно на заранее записанном файле.
                    now = total_frames * FRAME_SECONDS

                    # Те же правила, что в record_phrase.
                    if is_speech:
                        speech_seconds += FRAME_SECONDS
                        silence_seconds = 0.0
                        if not heard_speech and speech_seconds >= self.min_speech_s:
                            heard_speech = True
                            phrase_started = now
                            phrases += 1
                            print(f"        ► «говоришь...» ({_db(level)} дБ)")
                    elif heard_speech:
                        silence_seconds += FRAME_SECONDS
                        if silence_seconds >= self.silence_s:
                            print(
                                "        ◄ конец фразы, речи"
                                f" {speech_seconds:.1f} с,"
                                f" всего {now - phrase_started:.1f} с\n"
                            )
                            heard_speech = False
                            speech_seconds = 0.0
                            silence_seconds = 0.0
                            self.vad.reset()

                    if len(batch) >= every:
                        peak = max(item[0] for item in batch)
                        spoke = any(item[1] for item in batch)
                        bar_threshold = batch[-1][2]
                        print(
                            f"  {int(now) // 60}:{int(now) % 60:02d}"
                            f"  {_bar(peak)} {_db(peak):>5} дБ"
                            f"  {_db(bar_threshold):>5}"
                            f"  {'РЕЧЬ' if spoke else ''}"
                        )
                        batch.clear()

                    if seconds is not None and now >= seconds:
                        break
        except KeyboardInterrupt:
            print()

        return self._monitor_report(levels, speech_levels, phrases, speech_frames, total_frames)

    def _monitor_report(
        self,
        levels: list[float],
        speech_levels: list[float],
        phrases: int,
        speech_frames: int,
        total_frames: int,
    ) -> int:
        """Разбор того, что наслушали, с советом по настройке."""
        if not levels:
            print("Ничего не услышал.")
            return 1

        array = np.array(levels)
        quiet = float(np.percentile(array, 10))
        loud = float(np.percentile(array, 99))
        share = 100 * speech_frames / total_frames if total_frames else 0

        print("\nЧто получилось:")
        print(f"  тишина в комнате:  {_db(quiet):>5} дБ  ({quiet:.4f})")
        print(f"  самое громкое:     {_db(loud):>5} дБ  ({loud:.4f})")
        print(f"  речью посчитано:   {share:.0f}% времени, фраз {phrases}")

        vad = self.vad
        if not isinstance(vad, EnergyVad):
            print("\n  Порог тут задаёт silero (mic.speech_threshold),")
            print("  от громкости он не зависит — смотреть надо на долю речи.")
            return 0

        # Во сколько раз самое громкое превышает фон. Вблизи это
        # десятки, на пяти метрах бывает и втрое.
        margin = loud / quiet if quiet > 0 else float("inf")
        if vad.fixed is not None:
            print(f"  жёсткий порог:     {_db(vad.fixed):>5} дБ  ({vad.fixed:.4f})")
        else:
            print(
                f"  громче фона в:     {margin:.1f} раза"
                f"  (порог: фон × {vad.ratio:.1f})"
            )

        if phrases == 0 and vad.fixed is not None:
            print(
                "\n  Фраз не поймано, а порог тут жёсткий. Убери его:"
                "\n  mic.energy_threshold: null — тогда порог будет"
                " считаться от тишины."
            )
        elif phrases == 0 and margin <= vad.ratio:
            print(
                "\n  Речь слышно, но она не дотягивает до порога. Поставь в"
                f" config.yaml\n  mic.energy_ratio: {max(1.3, margin / 2):.1f}"
                f" (сейчас {vad.ratio:.1f})."
            )
        elif phrases == 0 and loud <= vad.floor_min:
            print(
                f"\n  Твой голос тише нижней границы. Поставь"
                f"\n  mic.energy_floor: {loud / 2:.4f} (сейчас {vad.floor_min:.4f})."
            )
        elif phrases == 0:
            print(
                f"\n  Порог речь проходит, но фраза не набрала"
                f" {self.min_speech_s:.1f} с подряд."
                "\n  Говори чуть длиннее или уменьши mic.min_speech_s."
            )
        elif share > 60:
            print(
                "\n  Речью считается больше половины времени — порог низкий."
                f"\n  Подними mic.energy_ratio, например до {vad.ratio * 1.5:.1f}."
            )
        else:
            print("\n  Настройка выглядит нормально.")
        return 0

    def measure_level(self, seconds: float = 3.0) -> tuple[float, float]:
        """Средняя и максимальная громкость за несколько секунд.

        Нужно, чтобы проверить микрофон, не трогая распознавание: если тут
        нули, дело в устройстве или в системных настройках, а не в коде.
        """
        import sounddevice as sd

        recording = sd.rec(
            int(seconds * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=self.device,
        )
        sd.wait()
        samples = recording[:, 0]
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        return rms, float(np.max(np.abs(samples)))


def make_microphone(cfg) -> Microphone:
    return Microphone(
        device=cfg.mic.input_device,
        speech_threshold=cfg.mic.speech_threshold,
        silence_s=cfg.mic.silence_s,
        min_speech_s=cfg.mic.min_speech_s,
        max_phrase_s=cfg.mic.max_phrase_s,
        wait_speech_s=cfg.mic.wait_speech_s,
        models_dir=cfg.models_dir,
        vad_engine=cfg.mic.vad,
        energy={
            "ratio": cfg.mic.energy_ratio,
            "floor_min": cfg.mic.energy_floor,
            "fixed": cfg.mic.energy_threshold,
        },
    )
