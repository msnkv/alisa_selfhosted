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
    page_chars: int
    fetch_timeout_s: int


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int
    total_timeout_s: float


@dataclass(frozen=True)
class TTSConfig:
    enabled: bool
    engine: str
    model: str
    speaker: str
    sample_rate: int
    device: str
    output_device: int | str | None


@dataclass(frozen=True)
class Config:
    name: str
    owner: str
    llm: LLMConfig
    search: SearchConfig
    agent: AgentConfig
    tts: TTSConfig
    max_sentences: int
    max_words: int
    data_dir: Path
    models_dir: Path
    log_dir: Path
    log_level: str


def load_config(path: Path | None = None) -> Config:
    """Читает config.yaml и .env. Пути в конфиге считаются от корня проекта."""
    load_dotenv(ROOT / ".env")

    cfg_path = path or ROOT / "config.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    llm = raw["llm"]
    search = raw.get("search", {})
    agent = raw.get("agent", {})
    tts = raw.get("tts", {})

    return Config(
        name=raw["assistant"]["name"],
        owner=raw["assistant"]["owner"],
        llm=LLMConfig(
            model=llm["model"],
            base_url=llm["base_url"],
            thinking=bool(llm.get("thinking", False)),
            temperature=float(llm.get("temperature", 0.3)),
            max_tokens=int(llm.get("max_tokens", 400)),
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
            page_chars=int(search.get("page_chars", 6000)),
            fetch_timeout_s=int(search.get("fetch_timeout_s", 8)),
        ),
        agent=AgentConfig(
            max_steps=int(agent.get("max_steps", 4)),
            total_timeout_s=float(agent.get("total_timeout_s", 35)),
        ),
        tts=TTSConfig(
            enabled=bool(tts.get("enabled", True)),
            engine=tts.get("engine", "silero"),
            model=tts.get("model", "v5_6_ru"),
            speaker=tts.get("speaker", "xenia"),
            sample_rate=int(tts.get("sample_rate", 48000)),
            device=tts.get("device", "cpu"),
            output_device=tts.get("output_device"),
        ),
        max_sentences=int(raw["answer"]["max_sentences"]),
        max_words=int(raw["answer"].get("max_words", 45)),
        data_dir=ROOT / raw["data"]["dir"],
        models_dir=ROOT / raw.get("models", {}).get("dir", "models"),
        log_dir=ROOT / raw["logging"]["dir"],
        log_level=raw["logging"]["level"].upper(),
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
