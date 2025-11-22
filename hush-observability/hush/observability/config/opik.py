"""Opik configuration - stub for future implementation."""

from .base import TracerConfig


class OpikConfig(TracerConfig):
    """Configuration for Opik observability backend.

    Note: This is a stub. Full implementation coming soon.

    Attributes:
        api_key: Opik API key
        workspace: Workspace identifier
        endpoint: Opik server endpoint
    """
    api_key: str
    workspace: str
    endpoint: str = "https://www.comet.com/opik"
