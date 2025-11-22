"""State registry for managing workflow states."""

from typing import Dict, Any, Optional, ClassVar

from hush.core.states.workflow_state import WorkflowState
from hush.core.states.workflow_indexer import WorkflowIndexer


class StateRegistry:
    """
    Simple dict-based state manager for maximum performance.
    Designed for high concurrency with minimal overhead.
    """
    __slots__ = ('_states',)

    _self: ClassVar[Optional['StateRegistry']] = None

    @classmethod
    def self(cls) -> 'StateRegistry':
        """Get the singleton instance of the registry."""
        if cls._self is None:
            cls._self = cls()
        return cls._self

    def __init__(self):
        self._states: Dict[str, WorkflowState] = {}

    def create(
        self,
        indexer: WorkflowIndexer,
        inputs: Dict[str, Any] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> WorkflowState:
        """Create new state with given data."""
        if inputs is None:
            inputs = {}

        _state = WorkflowState(
            indexer=indexer,
            inputs=inputs,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id
        )

        self._states[_state.request_id] = _state
        return _state

    def __getitem__(self, request_id: str) -> Optional[WorkflowState]:
        """Access state by request id."""
        return self._states.get(request_id, None)

    def __iter__(self):
        """Enable iteration over state keys."""
        return iter(self._states.keys())
