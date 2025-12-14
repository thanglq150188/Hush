"""Logging handlers package.

Each handler module defines:
- Config class (extends HandlerConfig)
- Factory function (takes config, returns logging.Handler)

Handlers are auto-registered when this package is imported.
"""

from .console import ConsoleHandlerConfig, NamedRichHandler, create_console_handler
from .file import (
    FileHandlerConfig,
    TimedFileHandlerConfig,
    create_file_handler,
    create_timed_file_handler,
)

__all__ = [
    # Console
    "ConsoleHandlerConfig",
    "NamedRichHandler",
    "create_console_handler",
    # File
    "FileHandlerConfig",
    "TimedFileHandlerConfig",
    "create_file_handler",
    "create_timed_file_handler",
]
