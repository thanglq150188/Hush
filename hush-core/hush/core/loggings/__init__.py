"""Centralized logging setup for Hush Core.

This package provides a configured LOGGER instance and utilities for logging
throughout the hush-core package.

Modules:
    config: LogConfig and HandlerConfig base class
    theme: Logging theme and highlighters
    handlers: Console, file, and extensible handlers
    formatters: Log data formatting utilities

Example:
    from hush.core.loggings import LOGGER, LogConfig, setup_logger

    # Use default logger (console only)
    LOGGER.info("Hello world")

    # Create custom logger with multiple handlers
    config = LogConfig(
        name="my_app",
        level="DEBUG",
        handlers=[
            {"type": "console", "level": "INFO"},
            {"type": "file", "filepath": "logs/app.log", "level": "DEBUG"},
        ]
    )
    logger = setup_logger(config)

    # Extend with custom handlers (e.g., Kafka)
    # In external package: hush-loggings-kafka
    import hush.loggings.kafka  # Auto-registers on import

    config = LogConfig(
        handlers=[
            {"type": "console"},
            {"type": "kafka", "bootstrap_servers": "localhost:9092", "topic": "logs"},
        ]
    )
"""

import logging
from typing import Callable, Dict, Optional, Tuple, Type, Any, Union

from .config import LogConfig, HandlerConfig
from .handlers import (
    ConsoleHandlerConfig,
    FileHandlerConfig,
    TimedFileHandlerConfig,
    create_console_handler,
    create_file_handler,
    create_timed_file_handler,
)
from .formatters import format_log_data, log_break, LOG_INDENT
from .theme import LOGGING_THEME


# Handler registry: type -> (ConfigClass, FactoryFunction)
_HANDLER_REGISTRY: Dict[str, Tuple[Type[HandlerConfig], Callable[[HandlerConfig], logging.Handler]]] = {}


def register_handler(
    handler_type: str,
    config_class: Type[HandlerConfig],
    factory: Callable[[HandlerConfig], logging.Handler],
) -> None:
    """Register a handler type with its config class and factory.

    This allows extending the logging system with custom handlers.
    External packages call this to register their handlers.

    Args:
        handler_type: The handler type identifier (e.g., "kafka", "syslog")
        config_class: The config class for this handler (extends HandlerConfig)
        factory: Factory function that takes config and returns logging.Handler

    Example:
        # In external package (hush-loggings-kafka)
        from hush.core.loggings import HandlerConfig, register_handler

        class KafkaHandlerConfig(HandlerConfig):
            type: Literal["kafka"] = "kafka"
            bootstrap_servers: str
            topic: str

        def create_kafka_handler(config: KafkaHandlerConfig) -> logging.Handler:
            ...

        # Auto-register on import
        register_handler("kafka", KafkaHandlerConfig, create_kafka_handler)
    """
    _HANDLER_REGISTRY[handler_type] = (config_class, factory)


def _parse_handler_config(data: Union[HandlerConfig, Dict[str, Any]]) -> HandlerConfig:
    """Parse raw dict or HandlerConfig into the correct config class."""
    # Already a specific config class instance
    if isinstance(data, HandlerConfig) and type(data) is not HandlerConfig:
        return data

    # Convert HandlerConfig base to dict for re-parsing
    if isinstance(data, HandlerConfig):
        data = data.model_dump()

    handler_type = data.get("type")
    if handler_type and handler_type in _HANDLER_REGISTRY:
        config_class, _ = _HANDLER_REGISTRY[handler_type]
        return config_class(**data)

    # Fallback to base HandlerConfig (will fail later if used)
    return HandlerConfig(**data)


def _create_handler_from_config(config: HandlerConfig) -> Optional[logging.Handler]:
    """Create a handler from its configuration."""
    if not config.enabled:
        return None

    handler_type = config.type

    if handler_type in _HANDLER_REGISTRY:
        _, factory = _HANDLER_REGISTRY[handler_type]
        return factory(config)

    raise ValueError(f"Unknown handler type: {handler_type}. Did you forget to import the handler package?")


def add_handler(logger: logging.Logger, handler: logging.Handler) -> logging.Logger:
    """Add a handler to a logger.

    Args:
        logger: The logger to add handler to
        handler: The handler to add

    Returns:
        The logger with the handler added
    """
    logger.addHandler(handler)
    return logger


def remove_handlers(logger: logging.Logger) -> logging.Logger:
    """Remove all handlers from a logger.

    Args:
        logger: The logger to clear handlers from

    Returns:
        The logger with all handlers removed
    """
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    return logger


def setup_logger(config: Optional[LogConfig] = None) -> logging.Logger:
    """Setup logger from configuration.

    Args:
        config: Log configuration (default: LogConfig())

    Returns:
        Configured Logger instance
    """
    if config is None:
        config = LogConfig()

    logger = logging.getLogger(config.name)
    logger.setLevel(getattr(logging, config.level.upper()))
    logger.propagate = config.propagate

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Create and add handlers from config
    for handler_data in config.handlers:
        handler_config = _parse_handler_config(handler_data)
        handler = _create_handler_from_config(handler_config)
        if handler:
            add_handler(logger, handler)

    return logger


# Register built-in handlers
register_handler("console", ConsoleHandlerConfig, create_console_handler)
register_handler("file", FileHandlerConfig, create_file_handler)
register_handler("timed_file", TimedFileHandlerConfig, create_timed_file_handler)


# Default logger instance
LOGGER = setup_logger(LogConfig(name="hush.core"))


__all__ = [
    # Main exports
    "LOGGER",
    "LogConfig",
    "setup_logger",
    # Base config for extending
    "HandlerConfig",
    # Built-in handler configs
    "ConsoleHandlerConfig",
    "FileHandlerConfig",
    "TimedFileHandlerConfig",
    # Utilities
    "add_handler",
    "remove_handlers",
    "register_handler",
    "format_log_data",
    "log_break",
    "LOG_INDENT",
    # Theme
    "LOGGING_THEME",
]
