#!/usr/bin/env python3
"""Demo script to showcase the logging system in Hush Core."""

from hush.core.loggings import LOGGER, LogConfig, setup_logger


def demo_console_only():
    """Demo with console handler only (default)."""
    config = LogConfig(
        name="demo.console",
        level="DEBUG",
        handlers=[
            {"type": "console", "level": "DEBUG"},
        ]
    )
    logger = setup_logger(config)

    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message", exc_info=True)


def demo_console_and_file():
    """Demo with both console and file handlers."""
    config = LogConfig(
        name="demo.file",
        level="DEBUG",
        handlers=[
            {"type": "console", "level": "INFO"},
            {"type": "file", "filepath": "logs/demo.log", "level": "DEBUG"},
        ]
    )
    logger = setup_logger(config)

    logger.debug("This goes to file only (console level is INFO)")
    logger.info("This goes to both console and file")
    logger.warning("Warning: something needs attention")
    logger.error("Error: something went wrong")
    logger.critical("Critical: immediate action required", exc_info=True)


def demo_plain_console():
    """Demo with plain console (no Rich formatting)."""
    config = LogConfig(
        name="demo.plain",
        level="DEBUG",
        handlers=[
            {"type": "console", "level": "DEBUG", "use_rich": False},
        ]
    )
    logger = setup_logger(config)

    logger.debug("Plain debug message")
    logger.info("Plain info message")
    logger.warning("Plain warning message")
    logger.error("Plain error message")


if __name__ == "__main__":
    print("\n--- Console Only (Rich) ---\n")
    demo_console_only()

    print("\n--- Console + File ---\n")
    demo_console_and_file()
    print("(Check logs/demo.log for file output)")

    print("\n--- Plain Console (no Rich) ---\n")
    demo_plain_console()

    print("\n--- Done ---\n")
