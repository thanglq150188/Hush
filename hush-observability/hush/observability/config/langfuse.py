"""Langfuse configuration."""

from typing import Optional
from .base import TracerConfig


class LangfuseConfig(TracerConfig):
    """Configuration for Langfuse observability backend.

    Attributes:
        public_key: Public API key for Langfuse authentication
        secret_key: Secret API key for Langfuse authentication
        host: Langfuse server URL (default: cloud.langfuse.com)
        no_proxy: Proxy bypass setting for internal networks

    Example:
        ```python
        config = LangfuseConfig(
            public_key="pk-...",
            secret_key="sk-...",
            host="https://cloud.langfuse.com"
        )
        ```

        From YAML:
        ```yaml
        tracer:langfuse:
          _class: LangfuseConfig
          public_key: ${LANGFUSE_PUBLIC_KEY}
          secret_key: ${LANGFUSE_SECRET_KEY}
          host: https://cloud.langfuse.com
        ```
    """
    public_key: str
    secret_key: str
    host: str = "https://cloud.langfuse.com"
    no_proxy: Optional[str] = None
    enabled: bool = True
    sample_rate: float = 1.0

    @classmethod
    def default(cls) -> 'LangfuseConfig':
        """Create default config pointing to Langfuse cloud."""
        return cls(
            public_key="pk-...",  # Replace with actual key
            secret_key="sk-...",  # Replace with actual key
            host="https://cloud.langfuse.com"
        )
