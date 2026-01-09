"""Các logging handler cho console."""

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
    """Config cho console handler.

    Args:
        enabled: Bật handler này (mặc định: True)
        level: Log level cho handler này
        format_str: Format string tùy chỉnh (bị bỏ qua nếu use_rich=True)
        use_rich: Sử dụng Rich handler cho màu sắc đẹp (mặc định: True)
    """

    type: Literal["console"] = "console"
    use_rich: bool = True


class NamedRichHandler(RichHandler):
    """RichHandler bao gồm tên logger trong output."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit log record với tên logger và message được tô màu theo level."""
        # Map log level sang style cho cả name và message
        # Dùng màu trực tiếp cho WARNING/ERROR/CRITICAL để tránh xung đột với custom tag
        level_styles = {
            "DEBUG": ("#8b949e", "#8b949e"),                       # Xám nhạt (dễ đọc hơn)
            "INFO": ("white", "white"),                            # Trắng
            "WARNING": ("#d29922", "#d29922"),                     # Vàng
            "ERROR": ("#f85149", "#f85149"),                       # Đỏ
            "CRITICAL": ("bold reverse #b81c1c", "bold reverse #b81c1c"),  # Trắng trên nền đỏ đậm
        }
        name_style, msg_style = level_styles.get(record.levelname, ("muted", "log.message"))

        # Làm việc trên bản sao để không ảnh hưởng đến các handler khác
        record = copy.copy(record)
        record.msg = f"[{name_style}]\\[{record.name}][/{name_style}] [{msg_style}]{record.msg}[/{msg_style}]"
        super().emit(record)


def create_console_handler(config: ConsoleHandlerConfig) -> logging.Handler:
    """Tạo console handler từ config.

    Args:
        config: Config cho console handler

    Returns:
        Rich handler với màu sắc hoặc console handler UTF-8 cơ bản
    """
    if config.use_rich:
        console = Console(
            theme=LOGGING_THEME,
            # Không ép buộc terminal - để Rich tự phát hiện
            # Đảm bảo output plain text khi pipe sang file hoặc trong CI/CD
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
