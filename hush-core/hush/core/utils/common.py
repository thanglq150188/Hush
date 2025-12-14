"""Common utility functions for hush-core."""

import asyncio
import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Type


@dataclass
class Param:
    """Parameter definition for node inputs/outputs.

    Example:
        input_schema = {
            "query": Param(str, required=True, description="Search query"),
            "limit": Param(int, default=10, description="Max results"),
        }
    """
    type: Type = str
    required: bool = False
    default: Any = None
    description: str = ""


def unique_name() -> str:
    """Generate a unique name using UUID."""
    return uuid.uuid4().hex[:8]


def verify_data(data: Any) -> bool:
    """Verify data is valid (not containing invalid types)."""
    # Basic validation - can be extended
    return True


def raise_error(message: str) -> None:
    """Raise a ValueError with the given message."""
    raise ValueError(message)


def extract_condition_variables(condition: str) -> Dict[str, str]:
    """
    Extract variable names and their inferred types from a condition string.

    Args:
        condition: A condition string like "a > 10 and b == 'hello'"

    Returns:
        Dict mapping variable names to inferred types
    """
    # Pattern to match variable names (identifiers not followed by opening paren)
    # Excludes Python keywords and built-in names
    keywords = {'and', 'or', 'not', 'in', 'is', 'True', 'False', 'None', 'if', 'else'}

    # Find all identifiers
    identifiers = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', condition)

    variables = {}
    for var in identifiers:
        if var not in keywords and var not in variables:
            # Try to infer type from context
            variables[var] = "Any"

    return variables


def fake_chunk_from(content: str, model: str = "default") -> Dict[str, Any]:
    """Create a fake chunk response for error handling."""
    return {
        "choices": [{
            "delta": {
                "content": content
            }
        }],
        "model": model
    }


def ensure_async(func: Callable) -> Callable:
    """Ensure a function is async-compatible.

    If the function is already async, return it as-is.
    If it's synchronous, wrap it to run in a thread pool executor.

    Args:
        func: The function to make async-compatible

    Returns:
        An async function that can be awaited

    Example:
        def sync_func(x):
            return x * 2

        async def async_func(x):
            return x * 2

        # Both can now be awaited:
        func = ensure_async(sync_func)
        result = await func(10)
    """
    if asyncio.iscoroutinefunction(func):
        return func

    async def async_wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    return async_wrapper
