"""Helpers for logging, timing, and OS operations."""

import logging
import sys
from collections.abc import Iterator
from pathlib import Path


def setup_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Create and configure a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        formatter = logging.Formatter(
            fmt="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger(__name__)


def read_file(path: Path) -> dict[str, str]:
    """Extract document content from a file.

    On failure, logs the error and returns an empty dict.
    """
    try:
        with Path.open(path, "r", encoding="utf-8") as f:
            content = f.read()
            return {"content": content, "source": str(path)}

    except (OSError, UnicodeDecodeError) as e:
        msg = f"Failed to read file {path}: {e}"
        logger.exception(msg)
        return {}


def search_files(dir_path: Path, extensions: list[str]) -> Iterator[Path]:
    """Search for files with given extensions in a directory."""
    for file_path in dir_path.rglob("*"):
        if file_path.suffix in extensions:
            yield file_path
