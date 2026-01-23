"""Pytest configuration and shared fixtures for hush-providers tests."""

import os
import pytest
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from package root
load_dotenv(Path(__file__).parent.parent / ".env")

# Get config path from environment
CONFIGS_PATH = Path(os.environ.get("HUSH_CONFIG", ""))

from hush.core.registry import ResourceHub, set_global_hub


@pytest.fixture(scope="session", autouse=True)
def setup_resource_hub():
    """Setup ResourceHub with test configurations for the entire test session."""
    # Import plugins to auto-register config classes and factory handlers
    from hush.providers.registry import LLMPlugin, EmbeddingPlugin, RerankPlugin

    # Create hub from config file
    if not CONFIGS_PATH.exists():
        raise FileNotFoundError(
            f"Config file not found: {CONFIGS_PATH}. "
            "Set HUSH_CONFIG environment variable or create .env file."
        )
    hub = ResourceHub.from_yaml(CONFIGS_PATH)

    # Set as global and singleton
    set_global_hub(hub)
    ResourceHub.set_instance(hub)

    yield hub

    # Cleanup
    ResourceHub._instance = None


@pytest.fixture
def hub(setup_resource_hub):
    """Get the ResourceHub instance."""
    return setup_resource_hub
