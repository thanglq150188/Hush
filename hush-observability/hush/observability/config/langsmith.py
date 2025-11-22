"""LangSmith configuration - stub for future implementation."""

from .base import TracerConfig


class LangSmithConfig(TracerConfig):
    """Configuration for LangSmith observability backend.

    Note: This is a stub. Full implementation coming soon.

    Attributes:
        api_key: LangSmith API key
        project: Project name
        endpoint: LangSmith server endpoint
    """
    api_key: str
    project: str
    endpoint: str = "https://api.smith.langchain.com"
