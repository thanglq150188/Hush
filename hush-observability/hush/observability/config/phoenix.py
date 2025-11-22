"""Phoenix (Arize) configuration - stub for future implementation."""

from typing import Optional
from .base import TracerConfig


class PhoenixConfig(TracerConfig):
    """Configuration for Phoenix (Arize Phoenix) observability backend.

    Note: This is a stub. Full implementation coming soon.

    Attributes:
        endpoint: Phoenix server endpoint
        project_name: Optional project name for organizing traces
    """
    endpoint: str = "http://localhost:6006"
    project_name: Optional[str] = None
