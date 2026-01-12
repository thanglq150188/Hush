"""Pytest configuration and shared fixtures for hush-providers tests."""

import os
import pytest
from pathlib import Path

# Set config path before importing hush modules
CONFIGS_PATH = Path(__file__).parent.parent / "configs" / "resources.yaml"
os.environ["HUSH_CONFIG"] = str(CONFIGS_PATH)

from hush.core.registry import ResourceHub, set_global_hub


@pytest.fixture(scope="session", autouse=True)
def setup_resource_hub():
    """Setup ResourceHub with test configurations for the entire test session."""
    # Import plugins to auto-register config classes and factory handlers
    from hush.providers.registry import LLMPlugin, EmbeddingPlugin, RerankPlugin

    # Create hub from config file
    if CONFIGS_PATH.exists():
        hub = ResourceHub.from_yaml(CONFIGS_PATH)
    else:
        # Fallback to in-memory hub for CI/CD
        from hush.core.registry.storage import InMemoryConfigStorage
        hub = ResourceHub(InMemoryConfigStorage())

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
