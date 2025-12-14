"""Workflow state management.

Design:
    StateSchema  - Defines structure (links, defaults). Built once.
    BaseState    - Abstract interface for state backends.
    MemoryState  - In-memory dict storage (default).
    RedisState   - Redis storage for distributed apps.

Example:
    from hush.core.states import StateSchema, MemoryState, RedisState

    # BUILD (once)
    schema = StateSchema("my_workflow")
    schema.link("llm", "messages", "prompt", "output")
    schema.set("llm", "temperature", 0.7)

    # RUN with MemoryState (default)
    state = schema.create_state(inputs={"query": "hello"})

    # RUN with RedisState
    state = schema.create_state(
        inputs={"query": "hello"},
        state_class=RedisState,
        redis_client=redis_client
    )

    # Access values (with automatic redirect)
    state["prompt", "output", None] = "Hello"
    value = state["llm", "messages", None]  # Returns "Hello" via redirect

    # Debug
    schema.show()
    state.show()
"""

from hush.core.states.schema import StateSchema
from hush.core.states.base import BaseState
from hush.core.states.memory import MemoryState
from hush.core.states.redis import RedisState

__all__ = [
    "StateSchema",
    "BaseState",
    "MemoryState",
    "RedisState",
]
