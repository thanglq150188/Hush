"""Log formatting utilities."""

from typing import Any

# Indent for multiline logs
# Rich auto-indents continuation lines to align with message start
LOG_INDENT = "  "  # Small additional indent for visual hierarchy


def log_break(text: str = "") -> str:
    """Add a line break with proper indentation for multiline logs.

    Use this when you want to break a long log message across lines.
    Rich automatically aligns continuation lines with the message start.

    Args:
        text: The text to put on the new line (optional)

    Returns:
        Text prefixed with newline and indent

    Example:
        LOGGER.info(f"Processing request{log_break('user_id: 123')}{log_break('action: update')}")
        # Output:
        # [12/12/25 00:59:32] INFO     [hush.core] Processing request
        #                                           user_id: 123
        #                                           action: update
    """
    return f"\n{LOG_INDENT}{text}"


def format_log_data(data: Any, max_length: int = 400, max_items: int = 3) -> str:
    """Smart formatting for log data that handles different types elegantly.

    Args:
        data: Data to format (dict, list, str, etc.)
        max_length: Maximum character length for the output
        max_items: Maximum number of items to show in collections

    Returns:
        Formatted string suitable for logging

    Example:
        >>> format_log_data({"key": "value", "list": [1, 2, 3]})
        "{key='value', list=<list len=3>}"
    """
    if data is None:
        return "None"

    if isinstance(data, dict):
        if not data:
            return "{}"

        items = []
        for i, (k, v) in enumerate(data.items()):
            if i >= max_items:
                items.append(f"... +{len(data) - max_items} more")
                break

            if isinstance(v, str):
                val_str = f"'{v[:50]}...'" if len(v) > 50 else f"'{v}'"
            elif isinstance(v, (list, dict)):
                val_str = f"<{type(v).__name__} len={len(v)}>"
            elif isinstance(v, bytes):
                val_str = f"<bytes len={len(v)}>"
            else:
                val_str = str(v)[:50]

            items.append(f"{k}={val_str}")

        result = "{" + ", ".join(items) + "}"

    elif isinstance(data, (list, tuple)):
        if not data:
            return "[]" if isinstance(data, list) else "()"

        items = []
        for i, item in enumerate(data):
            if i >= max_items:
                items.append(f"... +{len(data) - max_items} more")
                break
            items.append(str(item)[:50])

        bracket = ("[]" if isinstance(data, list) else "()")
        result = bracket[0] + ", ".join(items) + bracket[1]

    elif isinstance(data, str):
        if len(data) > max_length:
            result = f"'{data[:max_length]}...' (len={len(data)})"
        else:
            result = f"'{data}'"

    elif isinstance(data, bytes):
        result = f"<bytes len={len(data)}>"

    else:
        result = str(data)

    if len(result) > max_length:
        result = result[:max_length] + "..."

    return result
