"""Logging setup for the PaperVibe CLI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.traceback import Traceback

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


class ConsoleHandler(logging.Handler):
    def __init__(self, console: Console) -> None:
        super().__init__()
        self.console = console

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            style = _style_for_level(record.levelno)
            self.console.print(message, style=style, highlight=False)
            if record.exc_info and self.level <= logging.DEBUG:
                exc_type, exc_value, exc_traceback = record.exc_info
                if exc_type is not None:
                    traceback = Traceback.from_exception(
                        exc_type,
                        exc_value,
                        exc_traceback,
                        show_locals=False,
                    )
                    self.console.print(traceback)
        except Exception:
            self.handleError(record)


def _style_for_level(level: int) -> str:
    if level >= logging.CRITICAL:
        return "bold white on red"
    if level >= logging.ERROR:
        return "bold red"
    if level >= logging.WARNING:
        return "yellow"
    if level >= logging.INFO:
        return "white"
    return "dim"


def setup_logging(
    verbose: int = 0,
    quiet: int = 0,
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    base_level = _parse_level(log_level) or logging.INFO
    console_level = _resolve_level(base_level, verbose, quiet)

    app_logger = logging.getLogger("papervibe")
    app_logger.handlers.clear()
    app_logger.setLevel(logging.DEBUG if log_file else console_level)
    app_logger.propagate = False

    console_handler = ConsoleHandler(_CONSOLE)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    app_logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        app_logger.addHandler(file_handler)

    return app_logger
