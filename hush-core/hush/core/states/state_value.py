"""Variable cell storage for workflow state with context support."""

from typing import Any, Optional, Dict, List


DEFAULT = "DEFAULT"


class VariableCell:
    """Storage for a variable that can have multiple values in different contexts/loops."""

    def __init__(self, default_value: Any = None):
        self.contexts: Dict[str, Any] = {}
        self.versions: List[str] = []
        self.DEFAULT_VALUE = default_value
        self.DEFAULT_CONTEXT = "main"

    def __setitem__(self, context_id: Optional[str], value: Any) -> None:
        """Set value for a specific context."""
        if context_id is None:
            if not self.versions:
                context_id = self.DEFAULT_CONTEXT
            else:
                context_id = self.versions[-1]

        self.contexts[context_id] = value

        if context_id not in self.versions:
            self.versions.append(context_id)

    def __getitem__(self, context_id: Optional[str] = None) -> Any:
        """Get value from a specific context."""
        if context_id is None:
            if not self.versions:
                context_id = self.DEFAULT_CONTEXT
            else:
                context_id = self.versions[-1]

        return self.contexts.get(context_id, self.DEFAULT_VALUE)

    def get(self) -> Any:
        """Get value from the most recent context or default."""
        if self.versions:
            latest_context = self.versions[-1]
            return self.contexts[latest_context]
        return self.DEFAULT_VALUE
