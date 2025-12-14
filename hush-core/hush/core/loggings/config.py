"""Logging configuration models."""

from typing import Optional, List, Union, Dict, Any

from pydantic import BaseModel, Field, ConfigDict


class HandlerConfig(BaseModel):
    """Base handler configuration.

    External packages extend this class to define their own handler configs.
    The 'type' field is used as a discriminator to look up the correct
    config class and factory from the registry.

    Args:
        type: Handler type identifier (e.g., "console", "file", "kafka")
        enabled: Enable this handler (default: True)
        level: Log level for this handler (default: DEBUG)
        format_str: Custom format string

    Example:
        # In external package (hush-loggings-kafka)
        class KafkaHandlerConfig(HandlerConfig):
            type: Literal["kafka"] = "kafka"
            bootstrap_servers: str
            topic: str
    """

    model_config = ConfigDict(extra="allow")

    type: str
    enabled: bool = True
    level: str = "DEBUG"
    format_str: Optional[str] = None


class LogConfig(BaseModel):
    """Logging configuration with flexible handler support.

    Handlers can be specified as:
    - Dict with 'type' key (will be parsed into correct config class)
    - HandlerConfig instance (or subclass)

    Args:
        name: Logger name (default: "hush")
        level: Root log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        handlers: List of handler configurations (dicts or HandlerConfig instances)
        propagate: Propagate to parent loggers (default: False)

    Example:
        # Basic usage with console only
        config = LogConfig()
        logger = setup_logger(config)

        # With multiple handlers (dict-based, works with YAML/JSON)
        config = LogConfig(
            name="my_app",
            level="DEBUG",
            handlers=[
                {"type": "console", "level": "INFO"},
                {"type": "file", "filepath": "logs/app.log", "level": "DEBUG"},
            ]
        )
        logger = setup_logger(config)

        # With external handler (after importing the package)
        import hush.loggings.kafka  # Auto-registers kafka handler

        config = LogConfig(
            handlers=[
                {"type": "console"},
                {"type": "kafka", "bootstrap_servers": "localhost:9092", "topic": "logs"},
            ]
        )
    """

    name: str = "hush"
    level: str = "INFO"
    handlers: List[Union[HandlerConfig, Dict[str, Any]]] = Field(
        default_factory=lambda: [{"type": "console"}]
    )
    propagate: bool = False
