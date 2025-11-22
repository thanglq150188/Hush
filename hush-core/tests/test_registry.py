"""Tests for the resource registry system."""

import pytest
from typing import Any, Type

from hush.core.registry import (
    ResourceHub,
    ResourcePlugin,
    ResourceConfig,
    FileConfigStorage,
)


# ============================================================================
# Test Fixtures: Custom Plugin
# ============================================================================

class MockServiceConfig(ResourceConfig):
    """Mock config for testing."""
    name: str = "default"
    host: str = "localhost"
    port: int = 8080


class MockService:
    """Mock service for testing."""
    def __init__(self, config: MockServiceConfig):
        self.config = config
        self.name = config.name
        self.host = config.host
        self.port = config.port

    def __repr__(self):
        return f"MockService({self.name}@{self.host}:{self.port})"


class MockServicePlugin(ResourcePlugin):
    """Mock plugin for testing."""

    @classmethod
    def config_class(cls) -> Type[ResourceConfig]:
        return MockServiceConfig

    @classmethod
    def create(cls, config: ResourceConfig) -> Any:
        if not isinstance(config, MockServiceConfig):
            raise ValueError(f"Expected MockServiceConfig, got {type(config)}")
        return MockService(config)

    @classmethod
    def resource_type(cls) -> str:
        return "mock"


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def hub():
    """Create an in-memory resource hub for testing."""
    hub = ResourceHub.from_memory()
    hub.register_plugin(MockServicePlugin)
    return hub


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    return MockServiceConfig(
        name="test-service",
        host="example.com",
        port=9000
    )


# ============================================================================
# Tests
# ============================================================================

def test_hub_creation():
    """Test creating different types of resource hubs."""
    # In-memory
    hub = ResourceHub.from_memory()
    assert hub is not None
    assert len(hub.keys()) == 0


def test_plugin_registration(hub):
    """Test registering plugins."""
    assert hub.has_plugin("mock")

    # Check plugin is working
    config = MockServiceConfig(name="test")
    key = hub.register(config)
    assert key == "mock:test"
    assert hub.has(key)


def test_resource_registration(hub, mock_config):
    """Test registering and retrieving resources."""
    # Register a resource
    key = hub.register(mock_config)
    assert key == "mock:test-service"

    # Retrieve it
    service = hub.get(key)
    assert isinstance(service, MockService)
    assert service.name == "test-service"
    assert service.host == "example.com"
    assert service.port == 9000


def test_resource_registration_custom_key(hub, mock_config):
    """Test registering with a custom key."""
    key = hub.register(mock_config, registry_key="custom:key")
    assert key == "custom:key"
    assert hub.has("custom:key")

    service = hub.get("custom:key")
    assert isinstance(service, MockService)


def test_resource_removal(hub, mock_config):
    """Test removing resources."""
    key = hub.register(mock_config)
    assert hub.has(key)

    # Remove it
    removed = hub.remove(key, persist=False)
    assert removed is True
    assert not hub.has(key)

    # Try to remove non-existent key
    removed = hub.remove("non-existent")
    assert removed is False


def test_clear_registry(hub, mock_config):
    """Test clearing all resources."""
    # Add multiple resources
    hub.register(MockServiceConfig(name="service1"))
    hub.register(MockServiceConfig(name="service2"))
    hub.register(MockServiceConfig(name="service3"))

    assert len(hub.keys()) == 3

    # Clear all
    hub.clear(persist=False)
    assert len(hub.keys()) == 0


def test_get_config(hub, mock_config):
    """Test retrieving config objects."""
    key = hub.register(mock_config)

    # Get the config
    config = hub.get_config(key)
    assert isinstance(config, MockServiceConfig)
    assert config.name == "test-service"
    assert config.host == "example.com"
    assert config.port == 9000


def test_key_existence(hub, mock_config):
    """Test checking key existence."""
    assert not hub.has("mock:test-service")

    key = hub.register(mock_config)
    assert hub.has(key)
    assert hub.has("mock:test-service")


def test_keys_listing(hub):
    """Test listing all keys."""
    assert hub.keys() == []

    hub.register(MockServiceConfig(name="service1"))
    hub.register(MockServiceConfig(name="service2"))

    keys = hub.keys()
    assert len(keys) == 2
    assert "mock:service1" in keys
    assert "mock:service2" in keys


def test_singleton_pattern():
    """Test singleton instance management."""
    # Create and set instance
    hub = ResourceHub.from_memory()
    ResourceHub.set_instance(hub)

    # Get singleton
    singleton = ResourceHub.instance()
    assert singleton is hub

    # Clean up for other tests
    ResourceHub._instance = None


def test_singleton_not_initialized():
    """Test singleton raises error if not initialized."""
    ResourceHub._instance = None

    with pytest.raises(RuntimeError, match="ResourceHub singleton not initialized"):
        ResourceHub.instance()


def test_multiple_plugins():
    """Test registering multiple plugins."""
    hub = ResourceHub.from_memory()

    # Create second plugin
    class AnotherServiceConfig(ResourceConfig):
        id: str = "default"

    class AnotherService:
        def __init__(self, config):
            self.id = config.id

    class AnotherServicePlugin(ResourcePlugin):
        @classmethod
        def config_class(cls):
            return AnotherServiceConfig

        @classmethod
        def create(cls, config):
            return AnotherService(config)

        @classmethod
        def resource_type(cls):
            return "another"

    # Register both plugins
    hub.register_plugins(MockServicePlugin, AnotherServicePlugin)

    assert hub.has_plugin("mock")
    assert hub.has_plugin("another")

    # Register resources from both
    hub.register(MockServiceConfig(name="service1"))
    hub.register(AnotherServiceConfig(id="service2"))

    assert hub.has("mock:service1")
    assert hub.has("another:service2")


def test_no_plugin_registered():
    """Test error when registering resource without plugin."""
    hub = ResourceHub.from_memory()

    config = MockServiceConfig(name="test")

    with pytest.raises(ValueError, match="No plugin registered for config type"):
        hub.register(config)


def test_get_nonexistent_key(hub):
    """Test error when getting non-existent key."""
    with pytest.raises(KeyError, match="Resource 'non-existent' not found"):
        hub.get("non-existent")


def test_get_config_nonexistent_key(hub):
    """Test error when getting config for non-existent key."""
    with pytest.raises(KeyError, match="Config 'non-existent' not found"):
        hub.get_config("non-existent")


# ============================================================================
# Test File Storage
# ============================================================================

def test_file_storage_yaml(tmp_path):
    """Test YAML file storage."""
    config_file = tmp_path / "resources.yaml"

    # Create hub with file storage
    hub = ResourceHub.from_yaml(config_file)
    hub.register_plugin(MockServicePlugin)

    # Register a resource (persist=True)
    config = MockServiceConfig(name="test", host="example.com", port=9000)
    key = hub.register(config, persist=True)

    # Close and reload
    hub.close()

    # Create new hub from same file
    hub2 = ResourceHub.from_yaml(config_file)
    hub2.register_plugin(MockServicePlugin)

    # Check resource is loaded
    assert hub2.has(key)
    service = hub2.get(key)
    assert service.name == "test"
    assert service.host == "example.com"
    assert service.port == 9000


def test_file_storage_json(tmp_path):
    """Test JSON file storage."""
    config_file = tmp_path / "resources.json"

    # Create hub with file storage
    hub = ResourceHub.from_json(config_file)
    hub.register_plugin(MockServicePlugin)

    # Register a resource
    config = MockServiceConfig(name="test", host="example.com", port=9000)
    key = hub.register(config, persist=True)

    # Close and reload
    hub.close()

    # Create new hub from same file
    hub2 = ResourceHub.from_json(config_file)
    hub2.register_plugin(MockServicePlugin)

    # Check resource is loaded
    assert hub2.has(key)
    service = hub2.get(key)
    assert service.name == "test"


def test_persistence_disabled(hub, mock_config):
    """Test registering without persistence."""
    key = hub.register(mock_config, persist=False)
    assert hub.has(key)

    # In-memory storage, so this is just a no-op test
    # In file storage, this would NOT save to disk


# ============================================================================
# Integration Tests
# ============================================================================

def test_workflow_integration():
    """Test using registry in a workflow context."""
    hub = ResourceHub.from_memory()
    hub.register_plugin(MockServicePlugin)

    # Register multiple services
    hub.register(MockServiceConfig(name="llm-service", port=8000))
    hub.register(MockServiceConfig(name="embedding-service", port=8001))
    hub.register(MockServiceConfig(name="cache-service", port=6379))

    # Simulate workflow using services
    llm = hub.get("mock:llm-service")
    embedding = hub.get("mock:embedding-service")
    cache = hub.get("mock:cache-service")

    assert llm.port == 8000
    assert embedding.port == 8001
    assert cache.port == 6379


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
