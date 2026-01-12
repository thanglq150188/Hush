"""Langfuse configuration."""

from typing import Optional
from pydantic import BaseModel


class LangfuseConfig(BaseModel):
    """Configuration for Langfuse observability backend.

    Attributes:
        public_key: Public API key for Langfuse authentication
        secret_key: Secret API key for Langfuse authentication
        host: Langfuse server URL (default: cloud.langfuse.com)
        no_proxy: Proxy bypass setting for internal networks
        enabled: Whether tracing is enabled
        sample_rate: Sampling rate for traces (0.0 to 1.0)

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
        langfuse:
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
    def from_env(cls) -> 'LangfuseConfig':
        """Create config from environment variables.

        Environment variables:
            - LANGFUSE_PUBLIC_KEY
            - LANGFUSE_SECRET_KEY
            - LANGFUSE_HOST (optional)
        """
        import os
        return cls(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            no_proxy=os.environ.get("NO_PROXY"),
        )