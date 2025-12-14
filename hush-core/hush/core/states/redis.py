"""Redis-backed workflow state for distributed applications."""

import json
from typing import Any, Dict, Optional

from hush.core.states.schema import StateSchema
from hush.core.states.base import BaseState

__all__ = ["RedisState"]


class RedisState(BaseState):
    """Redis-backed workflow state for distributed applications.

    Suitable for multi-process/multi-server deployments.
    Values are JSON serialized for Redis storage.

    Example:
        import redis
        client = redis.Redis(host='localhost', port=6379)

        schema = StateSchema("my_workflow")
        schema.link("llm", "messages", "prompt", "output")

        state = RedisState(schema, redis_client=client, inputs={"query": "hello"})
        state["prompt", "output", None] = "Hello"
        value = state["llm", "messages", None]  # "Hello" via redirect
    """

    __slots__ = ("_redis", "_prefix", "_ttl")

    def __init__(
        self,
        schema: StateSchema,
        redis_client,
        inputs: Dict[str, Any] = None,
        prefix: str = "hush:state",
        ttl: int = 3600,
        user_id: str = None,
        session_id: str = None,
        request_id: str = None,
    ) -> None:
        """Initialize Redis state.

        Args:
            schema: StateSchema defining structure and redirects
            redis_client: Redis client instance
            inputs: Initial input values {var_name: value}
            prefix: Redis key prefix
            ttl: Time-to-live in seconds (default: 1 hour)
            user_id: User identifier (auto-generated if None)
            session_id: Session identifier (auto-generated if None)
            request_id: Request identifier (auto-generated if None)
        """
        self._redis = redis_client
        self._prefix = prefix
        self._ttl = ttl
        super().__init__(schema, inputs, user_id=user_id, session_id=session_id, request_id=request_id)

    def _make_key(self, node: str, var: str, ctx: Optional[str]) -> str:
        """Generate Redis key."""
        ctx_part = ctx or "main"
        return f"{self._prefix}:{self.request_id}:{node}:{var}:{ctx_part}"

    # =========================================================================
    # Abstract Method Implementations
    # =========================================================================

    def _get_value(self, node: str, var: str, ctx: Optional[str]) -> Any:
        key = self._make_key(node, var, ctx)
        value = self._redis.get(key)
        if value is None:
            return None
        return json.loads(value)

    def _set_value(self, node: str, var: str, ctx: Optional[str], value: Any) -> None:
        key = self._make_key(node, var, ctx)
        self._redis.setex(key, self._ttl, json.dumps(value))

    def _iter_keys(self):
        """Iterate over keys for this request."""
        pattern = f"{self._prefix}:{self.request_id}:*"
        for key in self._redis.scan_iter(match=pattern):
            # Parse key: prefix:request_id:node:var:ctx
            parts = key.decode().split(":")
            if len(parts) >= 5:
                node = parts[3]
                var = parts[4]
                ctx = parts[5] if len(parts) > 5 else None
                if ctx == "main":
                    ctx = None
                yield (node, var, ctx)

    # =========================================================================
    # Additional Methods
    # =========================================================================

    def __len__(self) -> int:
        """Count stored values for this request."""
        pattern = f"{self._prefix}:{self.request_id}:*"
        return sum(1 for _ in self._redis.scan_iter(match=pattern))

    def cleanup(self) -> None:
        """Delete all keys for this request."""
        pattern = f"{self._prefix}:{self.request_id}:*"
        keys = list(self._redis.scan_iter(match=pattern))
        if keys:
            self._redis.delete(*keys)
