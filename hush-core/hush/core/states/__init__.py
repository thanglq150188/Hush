"""Workflow state management v2 - simplified Cell-based design.

Design:
    StateSchema  - Defines structure with index-based O(1) resolution.
    MemoryState  - In-memory Cell-based storage.
    Ref          - Reference with operation chaining.
    Cell         - Multi-context value storage.

Example:
    from hush.core.states import StateSchema, MemoryState

    # From graph
    schema = StateSchema(graph)
    state = MemoryState(schema, inputs={"x": 5})

    # Access values
    state["node", "var", None] = "value"
    value = state["node", "var", None]

    # Debug
    schema.show()
    state.show()
"""

from hush.core.states.ref import Ref
from hush.core.states.schema import StateSchema
from hush.core.states.state import MemoryState
from hush.core.states.cell import Cell

__all__ = [
    "Ref",
    "StateSchema",
    "MemoryState",
    "Cell",
]
