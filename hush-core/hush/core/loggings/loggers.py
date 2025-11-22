"""Centralized logging setup for Hush Core.

This module provides a configured LOGGER instance that can be used throughout
the hush-core package. It uses Python's standard logging with UTF-8 console output.
"""

import sys
import codecs
import logging
from typing import Optional


def create_console_handler(level: str = "INFO", format_str: Optional[str] = None) -> logging.StreamHandler:
    """Create UTF-8 console handler.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_str: Custom format string for log messages

    Returns:
        Configured StreamHandler instance
    """
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    handler = logging.StreamHandler(
        stream=codecs.getwriter('utf-8')(sys.stdout.buffer)
    )
    handler.setLevel(getattr(logging, level.upper()))
    handler.setFormatter(logging.Formatter(format_str))
    return handler


def setup_logger(
    name: str = "hush.core",
    level: str = "INFO",
    console_output: bool = True,
    format_str: Optional[str] = None
) -> logging.Logger:
    """Setup logger with console handler.

    Args:
        name: Logger name (default: "hush.core")
        level: Logging level (default: "INFO")
        console_output: Enable console output (default: True)
        format_str: Custom format string for log messages

    Returns:
        Configured Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Add console handler
    if console_output:
        console_handler = create_console_handler(
            level=level,
            format_str=format_str
        )
        logger.addHandler(console_handler)

    return logger


# Default logger instance for hush.core
LOGGER = setup_logger()