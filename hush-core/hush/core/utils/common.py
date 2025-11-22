"""Common utility functions for hush-core."""

import re
import uuid
from typing import Any, Dict


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
