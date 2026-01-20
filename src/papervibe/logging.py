"""Logging setup for the PaperVibe CLI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

_CONSOLE = Console()

_LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "warn": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def get_console() -> Console:
    return _CONSOLE


def _parse_level(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized.isdigit():
            return int(normalized)
        if normalized in _LEVEL_MAP:
            return _LEVEL_MAP[normalized]
    return None


def _resolve_level(base_level: int, verbose: int, quiet: int) -> int:
    level = base_level - (10 * verbose) + (10 * quiet)
    return max(logging.DEBUG, min(level, logging.CRITICAL))


def setup_logging(
    verbose: int = 0,
    quiet: int = 0,
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    base_level = _parse_level(log_level) or logging.INFO
    console_level = _resolve_level(base_level, verbose, quiet)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG if log_file else console_level)

    console_handler = RichHandler(
        console=_CONSOLE,
        show_time=False,
        show_path=False,
        rich_tracebacks=True,
        markup=False,
    )
    console_handler.setLevel(console_level)
    root_logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        root_logger.addHandler(file_handler)

    logging.captureWarnings(True)
    return logging.getLogger("papervibe")
