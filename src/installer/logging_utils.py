"""Small, isolated file loggers for installer operations and child output."""

from contextlib import contextmanager
import logging
from pathlib import Path
from typing import Iterator


def log_path(path: Path) -> Path:
    """Use a .log suffix for both default and explicitly supplied destinations."""
    return Path(path).expanduser().with_suffix(".log")


class _FileHandler(logging.FileHandler):
    def handleError(self, record: logging.LogRecord) -> None:
        """A failed log write must fail the operation instead of being hidden."""
        raise


@contextmanager
def file_logger(path: Path, mode: str = "w") -> Iterator[logging.Logger]:
    """Open one .log file and release its handler on success or exception.

    A directly constructed logger has no process-global registration or parent
    handlers, so repeated installer runs cannot duplicate lines or retain open
    files. Each record is flushed by FileHandler as soon as it is emitted.
    """
    destination = log_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handler = _FileHandler(destination, mode=mode, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger = logging.Logger(f"installer.{destination.stem}", level=logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)
    try:
        yield logger
    finally:
        logger.removeHandler(handler)
        handler.close()
