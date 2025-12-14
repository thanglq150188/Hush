"""File logging handlers."""

import logging
from pathlib import Path
from typing import Union, Literal
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

from ..config import HandlerConfig


class FileHandlerConfig(HandlerConfig):
    """File handler configuration (size-based rotation).

    Args:
        enabled: Enable this handler (default: True)
        level: Log level for this handler
        format_str: Custom format string
        filepath: Path to the log file
        max_bytes: Max file size before rotation (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)
        encoding: File encoding (default: utf-8)
    """

    type: Literal["file"] = "file"
    filepath: Union[str, Path]
    max_bytes: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5
    encoding: str = "utf-8"


class TimedFileHandlerConfig(HandlerConfig):
    """Time-based rotating file handler configuration.

    Args:
        enabled: Enable this handler (default: True)
        level: Log level for this handler
        format_str: Custom format string
        filepath: Path to the log file
        when: Rotation interval type ('S', 'M', 'H', 'D', 'midnight', 'W0'-'W6')
        interval: Rotation interval (default: 1)
        backup_count: Number of backup files to keep (default: 30)
        encoding: File encoding (default: utf-8)
    """

    type: Literal["timed_file"] = "timed_file"
    filepath: Union[str, Path]
    when: str = "midnight"
    interval: int = 1
    backup_count: int = 30
    encoding: str = "utf-8"


def create_file_handler(config: FileHandlerConfig) -> logging.Handler:
    """Create file handler with size-based rotation.

    Args:
        config: File handler configuration

    Returns:
        Rotating file handler
    """
    filepath = Path(config.filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    format_str = config.format_str or "[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s"

    handler = RotatingFileHandler(
        filepath,
        maxBytes=config.max_bytes,
        backupCount=config.backup_count,
        encoding=config.encoding,
    )
    handler.setLevel(getattr(logging, config.level.upper()))
    handler.setFormatter(logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S"))
    return handler


def create_timed_file_handler(config: TimedFileHandlerConfig) -> logging.Handler:
    """Create time-based rotating file handler.

    Args:
        config: Timed file handler configuration

    Returns:
        Timed rotating file handler
    """
    filepath = Path(config.filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    format_str = config.format_str or "[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s"

    handler = TimedRotatingFileHandler(
        filepath,
        when=config.when,
        interval=config.interval,
        backupCount=config.backup_count,
        encoding=config.encoding,
    )
    handler.setLevel(getattr(logging, config.level.upper()))
    handler.setFormatter(logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S"))
    return handler
