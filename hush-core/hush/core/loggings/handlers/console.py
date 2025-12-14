"""Console logging handlers."""

import sys
import copy
import codecs
import logging
from typing import Literal

from rich.logging import RichHandler
from rich.console import Console
from rich.highlighter import NullHighlighter

from ..config import HandlerConfig
from ..theme import LOGGING_THEME


class ConsoleHandlerConfig(HandlerConfig):
    """Console handler configuration.

    Args:
        enabled: Enable this handler (default: True)
        level: Log level for this handler
        format_str: Custom format string (ignored if use_rich=True)
        use_rich: Use Rich handler for beautiful colors (default: True)
    """

    type: Literal["console"] = "console"
    use_rich: bool = True


class NamedRichHandler(RichHandler):
    """RichHandler that includes logger name in output."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record with logger name and message colored by level."""
        # Map log levels to styles for both name and message
        # Use direct colors for WARNING/ERROR/CRITICAL to avoid conflict with custom tags
        level_styles = {
            "DEBUG": ("#8b949e", "#8b949e"),                       # Lighter gray (more readable)
            "INFO": ("white", "white"),                            # White
            "WARNING": ("#d29922", "#d29922"),                     # Yellow
            "ERROR": ("#f85149", "#f85149"),                       # Red
            "CRITICAL": ("bold reverse #b81c1c", "bold reverse #b81c1c"),  # White on dark red
        }
        name_style, msg_style = level_styles.get(record.levelname, ("muted", "log.message"))

        # Work on a copy to avoid affecting other handlers
        record = copy.copy(record)
        record.msg = f"[{name_style}]\\[{record.name}][/{name_style}] [{msg_style}]{record.msg}[/{msg_style}]"
        super().emit(record)


def create_console_handler(config: ConsoleHandlerConfig) -> logging.Handler:
    """Create console handler from config.

    Args:
        config: Console handler configuration

    Returns:
        Rich handler with colors or basic UTF-8 console handler
    """
    if config.use_rich:
        console = Console(
            theme=LOGGING_THEME,
            # Don't force terminal - let Rich auto-detect
            # This ensures plain text output when piped to files or in CI/CD
            highlight=False,
        )

        handler = NamedRichHandler(
            console=console,
            show_time=True,
            show_level=True,
            show_path=False,
            enable_link_path=False,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            tracebacks_extra_lines=1,
            tracebacks_theme="github-dark",
            omit_repeated_times=False,
            highlighter=NullHighlighter(),
        )
        handler.setLevel(getattr(logging, config.level.upper()))
        return handler
    else:
        format_str = config.format_str or '[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s'
        handler = logging.StreamHandler(
            stream=codecs.getwriter('utf-8')(sys.stdout.buffer)
        )
        handler.setLevel(getattr(logging, config.level.upper()))
        handler.setFormatter(logging.Formatter(format_str, datefmt='%H:%M:%S'))
        return handler
