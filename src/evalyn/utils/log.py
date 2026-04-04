"""Logging utilities for Evalyn.

Provides dual logging:
- Console: Rich-formatted output for human readability during the run.
- File: Append-only events.log in the run directory, structured as
  [ISO_TIMESTAMP] [LEVEL] [COMPONENT] message
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


# Module-level console instance for non-log output
console = Console()

# Track whether logging has been configured
_configured = False


class EventsFileHandler(logging.FileHandler):
    """Custom file handler that formats log records for the events.log file."""

    def __init__(self, filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(str(filepath), mode="a", encoding="utf-8")
        self.setFormatter(EventsFormatter())


class EventsFormatter(logging.Formatter):
    """Formats log records as: [ISO_TIMESTAMP] [LEVEL] [COMPONENT] message"""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        component = record.name.replace("evalyn.", "")
        return f"[{ts}] [{record.levelname}] [{component}] {record.getMessage()}"


def setup_logging(
    run_dir: Optional[Path] = None,
    *,
    verbose: bool = False,
) -> None:
    """Configure the Evalyn logging system.

    Call once at the start of a command. Sets up:
    - Rich console handler (always)
    - Events file handler (if run_dir is provided)

    Args:
        run_dir: Path to the current run directory. If provided, events.log
            is created there.
        verbose: If True, set log level to DEBUG. Otherwise INFO.
    """
    global _configured

    root = logging.getLogger("evalyn")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Clear existing handlers to avoid duplicates on re-call
    root.handlers.clear()

    # Console handler via Rich
    rich_handler = RichHandler(
        console=Console(stderr=True),
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        markup=True,
    )
    rich_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.addHandler(rich_handler)

    # File handler if run_dir is provided
    if run_dir:
        events_path = run_dir / "events.log"
        file_handler = EventsFileHandler(events_path)
        file_handler.setLevel(logging.DEBUG)
        root.addHandler(file_handler)

    _configured = True


def get_logger(component: str) -> logging.Logger:
    """Get a logger for a specific component.

    Args:
        component: Component name (e.g., "planner", "runner.fixer").
            Will be prefixed with "evalyn.".

    Returns:
        A configured Logger instance.
    """
    return logging.getLogger(f"evalyn.{component}")


def get_console_logger(component: str) -> logging.Logger:
    """Get a logger with console-only output.

    Useful for CLI commands that don't have a run directory yet.
    Ensures logging is set up even if setup_logging() hasn't been called.
    """
    global _configured
    if not _configured:
        setup_logging()
    return get_logger(component)
