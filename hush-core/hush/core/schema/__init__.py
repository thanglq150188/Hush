"""Schema definitions for hush-core.

This module provides a simplified ParamSet for schema definitions.
For full functionality, use hush-integrations which provides the complete
tools and schema system.
"""

from typing import Any, Dict, Optional, Iterator
from dataclasses import dataclass, field
import re


@dataclass
class Param:
    """Parameter definition with type and default value."""
    type: str = "Any"
    default: Any = None
    required: bool = False
    description: str = ""


class ParamSet:
    """
    Parameter set for defining input/output schemas.

    This is a simplified version for hush-core. For advanced functionality,
    use the full ParamSet from hush-integrations.
    """

    def __init__(self):
        self.params: Dict[str, Param] = {}

    @classmethod
    def new(cls) -> 'ParamSet':
        """Create a new ParamSet builder."""
        return cls()

    def var(self, definition: str, required: bool = False, description: str = "") -> 'ParamSet':
        """
        Add a variable to the parameter set.

        Args:
            definition: Variable definition like "name: type = default"
            required: Whether the parameter is required
            description: Parameter description

        Examples:
            .var("query: str")
            .var("count: int = 0")
            .var("items: List")
        """
        # Parse definition: "name: type = default" or "name: type"
        match = re.match(r'(\w+)\s*:\s*(\w+(?:\[[\w\s,\[\]]+\])?)\s*(?:=\s*(.+))?', definition.strip())
        if match:
            name = match.group(1)
            param_type = match.group(2)
            default_str = match.group(3)

            # Parse default value
            default = None
            if default_str is not None:
                default_str = default_str.strip()
                if default_str == 'None':
                    default = None
                elif default_str == 'True':
                    default = True
                elif default_str == 'False':
                    default = False
                elif default_str == '{}':
                    default = {}
                elif default_str == '[]':
                    default = []
                elif default_str.startswith('"') or default_str.startswith("'"):
                    default = default_str[1:-1]
                else:
                    try:
                        default = int(default_str)
                    except ValueError:
                        try:
                            default = float(default_str)
                        except ValueError:
                            default = default_str

            self.params[name] = Param(
                type=param_type,
                default=default,
                required=required,
                description=description
            )
        else:
            # Simple case: just a name
            self.params[definition.strip()] = Param(required=required, description=description)

        return self

    def build(self) -> 'ParamSet':
        """Build and return the ParamSet."""
        return self

    def __getitem__(self, key: str) -> Param:
        return self.params[key]

    def __setitem__(self, key: str, value: Param):
        self.params[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.params

    def __iter__(self) -> Iterator[str]:
        return iter(self.params)

    def __len__(self) -> int:
        return len(self.params)

    def keys(self):
        return self.params.keys()

    def values(self):
        return self.params.values()

    def items(self):
        return self.params.items()

    def get(self, key: str, default: Any = None) -> Optional[Param]:
        return self.params.get(key, default)


__all__ = [
    "Param",
    "ParamSet",
]
