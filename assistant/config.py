"""Загрузка config.yaml и ключей из .env, настройка логов."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Корень проекта: на уровень выше папки assistant/
ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class LLMConfig:
    model: str
    base_url: str
    thinking: bool
    temperature: float
    max_tokens: int
    detail_max_tokens: int
    story_max_tokens: int
    timeout_s: float
    max_retries: int
    api_key: str


@dataclass(frozen=True)
class SearchConfig:
    enabled: bool
    backend: str
    engine: str
    region: str
    max_results: int
    retries: int
    hedge_after_s: float
    timeout_s: float
    page_chars: int
    fetch_timeout_s: int


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int
    total_timeout_s: float
    memory_turns: int


@dataclass(frozen=True)
class TTSConfig:
    enabled: bool
    engine: str
    model: str
    speaker: str
    sample_rate: int
    device: str
    output_device: int | str | None
    stream: bool


@dataclass(frozen=True)
class STTConfig:
    enabled: bool
    engine: str
    model: str
    device: str
    compute_type: str
    language: str
    beam_size: int
    vad_filter: bool


@dataclass(frozen=True)
class WakeConfig:
    enabled: bool
    engine: str
    phrase: str
    variants: tuple[str, ...]
    model_dir: Path | None
    language: str
    min_confidence: float
    model: str
    threshold: float
    vad_threshold: float
    input_device: int | str | None
    cooldown_s: float
    follow_up_s: float


@dataclass(frozen=True)
class MicConfig:
    input_device: int | str | None
    speech_threshold: float
    silence_s: float
    min_speech_s: float
    max_phrase_s: float
    wait_speech_s: float


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    token: str
    timeout_s: float
    url: str


@dataclass(frozen=True)
class Config:
    name: str
    owner: str
    city: str
    persona: str
    llm: LLMConfig
    search: SearchConfig
    agent: AgentConfig
    tts: TTSConfig
    stt: STTConfig
    mic: MicConfig
    wake: WakeConfig
    server: ServerConfig
    max_sentences: int
    max_words: int
    detail_words: int
    story_words: int
    data_dir: Path
    data_only_on_request: bool
    models_dir: Path
    log_dir: Path
    log_level: str
    dialogue_log: bool
    dialogue_file: str


def load_config(path: Path | None = None) -> Config:
    """Читает config.yaml и .env. Пути в конфиге считаются от корня проекта."""
    load_dotenv(ROOT / ".env")

    cfg_path = path or ROOT / "config.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    llm = raw["llm"]
    search = raw.get("search", {})
    agent = raw.get("agent", {})
    tts = raw.get("tts", {})
    stt = raw.get("stt", {})
    mic = raw.get("mic", {})
    wake = raw.get("wake", {})
    server = raw.get("server", {})

    return Config(
        name=raw["assistant"]["name"],
        owner=raw["assistant"]["owner"],
        city=str(raw["assistant"].get("city", "") or ""),
        persona=str(raw["assistant"].get("persona", "") or ""),
        llm=LLMConfig(
            model=llm["model"],
            base_url=llm["base_url"],
            thinking=bool(llm.get("thinking", False)),
            temperature=float(llm.get("temperature", 0.3)),
            max_tokens=int(llm.get("max_tokens", 400)),
            detail_max_tokens=int(llm.get("detail_max_tokens", 1400)),
            story_max_tokens=int(llm.get("story_max_tokens", 6000)),
            timeout_s=float(llm.get("timeout_s", 30)),
            max_retries=int(llm.get("max_retries", 1)),
            # Ключ только из окружения: в config.yaml, который лежит в git,
            # секретам не место.
            api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
        ),
        search=SearchConfig(
            enabled=bool(search.get("enabled", True)),
            backend=search.get("backend", "ddgs"),
            engine=search.get("engine", "auto"),
            region=search.get("region", "ru-ru"),
            max_results=int(search.get("max_results", 5)),
            retries=int(search.get("retries", 2)),
            hedge_after_s=float(search.get("hedge_after_s", 2.5)),
            timeout_s=float(search.get("timeout_s", 18)),
            page_chars=int(search.get("page_chars", 6000)),
            fetch_timeout_s=int(search.get("fetch_timeout_s", 8)),
        ),
        agent=AgentConfig(
            max_steps=int(agent.get("max_steps", 4)),
            total_timeout_s=float(agent.get("total_timeout_s", 35)),
            memory_turns=int(agent.get("memory_turns", 3)),
        ),
        tts=TTSConfig(
            enabled=bool(tts.get("enabled", True)),
            engine=tts.get("engine", "silero"),
            model=tts.get("model", "v5_6_ru"),
            speaker=tts.get("speaker", "xenia"),
            sample_rate=int(tts.get("sample_rate", 48000)),
            device=tts.get("device", "cpu"),
            output_device=tts.get("output_device"),
            stream=bool(tts.get("stream", True)),
        ),
        stt=STTConfig(
            enabled=bool(stt.get("enabled", True)),
            engine=stt.get("engine", "faster-whisper"),
            model=stt.get("model", "large-v3-turbo"),
            device=stt.get("device", "cuda"),
            compute_type=stt.get("compute_type", "int8_float16"),
            language=stt.get("language", "ru"),
            beam_size=int(stt.get("beam_size", 1)),
            vad_filter=bool(stt.get("vad_filter", True)),
        ),
        mic=MicConfig(
            input_device=mic.get("input_device"),
            speech_threshold=float(mic.get("speech_threshold", 0.5)),
            silence_s=float(mic.get("silence_s", 0.8)),
            min_speech_s=float(mic.get("min_speech_s", 0.3)),
            max_phrase_s=float(mic.get("max_phrase_s", 15)),
            wait_speech_s=float(mic.get("wait_speech_s", 6)),
        ),
        wake=WakeConfig(
            enabled=bool(wake.get("enabled", True)),
            engine=str(wake.get("engine", "openwakeword")),
            phrase=str(wake.get("phrase", "алиса тян")),
            variants=tuple(wake.get("variants", []) or ()),
            model_dir=(
                ROOT / wake["model_dir"] if wake.get("model_dir") else None
            ),
            language=str(wake.get("language", "ru")),
            min_confidence=float(wake.get("min_confidence", 0.8)),
            model=wake.get("model", "hey_jarvis"),
            threshold=float(wake.get("threshold", 0.5)),
            vad_threshold=float(wake.get("vad_threshold", 0.5)),
            input_device=wake.get("input_device"),
            cooldown_s=float(wake.get("cooldown_s", 1.0)),
            follow_up_s=float(wake.get("follow_up_s", 8.0)),
        ),
        server=ServerConfig(
            host=server.get("host", "0.0.0.0"),
            port=int(server.get("port", 8765)),
            # Пустой токен = спросить может любой в домашней сети.
            token=str(server.get("token", "") or ""),
            timeout_s=float(server.get("timeout_s", 90)),
            url=str(
                server.get("url")
                or "http://localhost:%d" % int(server.get("port", 8765))
            ),
        ),
        max_sentences=int(raw["answer"]["max_sentences"]),
        max_words=int(raw["answer"].get("max_words", 45)),
        detail_words=int(raw["answer"].get("detail_words", 300)),
        story_words=int(raw["answer"].get("story_words", 3000)),
        data_dir=ROOT / raw["data"]["dir"],
        data_only_on_request=bool(raw["data"].get("only_on_request", False)),
        models_dir=ROOT / raw.get("models", {}).get("dir", "models"),
        log_dir=ROOT / raw["logging"]["dir"],
        log_level=raw["logging"]["level"].upper(),
        dialogue_log=bool(raw["logging"].get("dialogue", True)),
        dialogue_file=str(raw["logging"].get("dialogue_file", "dialogue.jsonl")),
    )


def setup_logging(cfg: Config) -> None:
    """Логи в файл с подробностями и в консоль - только важное.

    В файл пишем всё: потом по нему смотрим, из чего сложилась задержка
    и какие инструменты дёргала модель.
    """
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(cfg.log_dir / "assistant.log", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )

    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    root = logging.getLogger()
    root.setLevel(getattr(logging, cfg.log_level, logging.INFO))
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console)

    # Клиент openai на каждом запросе пишет по строчке в INFO - не нужно.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    # ddgs и trafilatura тоже многословны. primp - это http-клиент внутри
    # ddgs, он пишет строчку на каждый опрошенный поисковик.
    logging.getLogger("ddgs").setLevel(logging.WARNING)
    logging.getLogger("primp").setLevel(logging.WARNING)
    logging.getLogger("trafilatura").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
