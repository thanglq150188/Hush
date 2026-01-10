"""Tests for the ResourceHub and resource registry system."""

import pytest
from typing import Any

from hush.core.utils.yaml_model import YamlModel
from hush.core.registry import (
    ResourceHub,
    ResourceFactory,
    register_config_class,
    register_factory_handler,
    get_config_class,
    CLASS_NAME_MAP,
    FACTORY_HANDLERS,
)


# ============================================================================
# Test Fixtures: Mock Config and Service
# ============================================================================

class MockServiceConfig(YamlModel):
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


def mock_service_factory(config: MockServiceConfig) -> MockService:
    """Factory function to create MockService from config."""
    return MockService(config)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_registries():
    """Clean up registries before and after each test."""
    # Save original state
    original_class_map = CLASS_NAME_MAP.copy()
    original_handlers = FACTORY_HANDLERS.copy()

    yield

    # Restore original state
    CLASS_NAME_MAP.clear()
    CLASS_NAME_MAP.update(original_class_map)
    FACTORY_HANDLERS.clear()
    FACTORY_HANDLERS.update(original_handlers)


@pytest.fixture
def hub(tmp_path):
    """Create a ResourceHub with YAML storage for testing."""
    config_file = tmp_path / "resources.yaml"
    hub = ResourceHub.from_yaml(config_file)

    # Register mock config and factory
    register_config_class(MockServiceConfig)
    register_factory_handler(MockServiceConfig, mock_service_factory)

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
# Tests: Config Class Registration
# ============================================================================

class TestConfigClassRegistration:
    """Test config class registration."""

    def test_register_single_class(self):
        """Test registering a single config class."""
        register_config_class(MockServiceConfig)
        assert "MockServiceConfig" in CLASS_NAME_MAP
        assert CLASS_NAME_MAP["MockServiceConfig"] == MockServiceConfig

    def test_get_config_class(self):
        """Test retrieving registered config class."""
        register_config_class(MockServiceConfig)
        result = get_config_class("MockServiceConfig")
        assert result == MockServiceConfig

    def test_get_nonexistent_class(self):
        """Test getting non-existent class returns None."""
        result = get_config_class("NonExistentConfig")
        assert result is None


# ============================================================================
# Tests: Factory Handler Registration
# ============================================================================

class TestFactoryHandlerRegistration:
    """Test factory handler registration."""

    def test_register_factory_handler(self):
        """Test registering a factory handler."""
        register_config_class(MockServiceConfig)
        register_factory_handler(MockServiceConfig, mock_service_factory)
        assert MockServiceConfig in FACTORY_HANDLERS

    def test_factory_creates_instance(self):
        """Test factory creates correct instance."""
        register_config_class(MockServiceConfig)
        register_factory_handler(MockServiceConfig, mock_service_factory)

        config = MockServiceConfig(name="test", host="localhost", port=8080)
        instance = ResourceFactory.create(config)

        assert isinstance(instance, MockService)
        assert instance.name == "test"
        assert instance.host == "localhost"
        assert instance.port == 8080

    def test_factory_no_handler_raises(self):
        """Test factory raises error when no handler registered."""
        class UnregisteredConfig(YamlModel):
            value: str = "test"

        config = UnregisteredConfig()
        with pytest.raises(ValueError, match="Không có factory handler"):
            ResourceFactory.create(config)


# ============================================================================
# Tests: ResourceHub Creation
# ============================================================================

class TestHubCreation:
    """Test creating ResourceHub instances."""

    def test_from_yaml(self, tmp_path):
        """Test creating hub from YAML file."""
        config_file = tmp_path / "test.yaml"
        hub = ResourceHub.from_yaml(config_file)
        assert hub is not None

    def test_from_json(self, tmp_path):
        """Test creating hub from JSON file."""
        config_file = tmp_path / "test.json"
        hub = ResourceHub.from_json(config_file)
        assert hub is not None


# ============================================================================
# Tests: Resource Registration
# ============================================================================

class TestResourceRegistration:
    """Test registering and retrieving resources."""

    def test_register_and_get(self, hub, mock_config):
        """Test registering and retrieving a resource."""
        key = hub.register(mock_config)

        # Verify key format
        assert "mockservice:" in key.lower() or "test-service" in key

        # Verify resource is retrievable
        assert hub.has(key)
        service = hub.get(key)
        assert isinstance(service, MockService)
        assert service.name == "test-service"
        assert service.host == "example.com"
        assert service.port == 9000

    def test_register_with_custom_key(self, hub, mock_config):
        """Test registering with a custom key."""
        key = hub.register(mock_config, registry_key="custom:my-service")
        assert key == "custom:my-service"
        assert hub.has("custom:my-service")

    def test_get_config(self, hub, mock_config):
        """Test retrieving config object."""
        key = hub.register(mock_config)
        config = hub.get_config(key)

        assert isinstance(config, MockServiceConfig)
        assert config.name == "test-service"


# ============================================================================
# Tests: Resource Removal
# ============================================================================

class TestResourceRemoval:
    """Test removing resources."""

    def test_remove_existing(self, hub, mock_config):
        """Test removing an existing resource."""
        key = hub.register(mock_config)
        assert hub.has(key)

        removed = hub.remove(key)
        assert removed is True
        assert not hub.has(key)

    def test_remove_nonexistent(self, hub):
        """Test removing non-existent key returns False."""
        removed = hub.remove("nonexistent:key")
        assert removed is False

    def test_clear_all(self, hub):
        """Test clearing all resources."""
        hub.register(MockServiceConfig(name="service1"))
        hub.register(MockServiceConfig(name="service2"))
        hub.register(MockServiceConfig(name="service3"))

        assert len(hub.keys()) == 3

        hub.clear()
        assert len(hub.keys()) == 0


# ============================================================================
# Tests: Key Operations
# ============================================================================

class TestKeyOperations:
    """Test key listing and checking operations."""

    def test_keys_empty(self, hub):
        """Test keys on empty hub."""
        assert hub.keys() == []

    def test_keys_with_resources(self, hub):
        """Test listing all keys."""
        hub.register(MockServiceConfig(name="service1"))
        hub.register(MockServiceConfig(name="service2"))

        keys = hub.keys()
        assert len(keys) == 2

    def test_has_existing(self, hub, mock_config):
        """Test has returns True for existing key."""
        key = hub.register(mock_config)
        assert hub.has(key) is True

    def test_has_nonexistent(self, hub):
        """Test has returns False for non-existent key."""
        assert hub.has("nonexistent:key") is False


# ============================================================================
# Tests: Error Handling
# ============================================================================

class TestErrorHandling:
    """Test error handling."""

    def test_get_nonexistent_raises(self, hub):
        """Test get raises KeyError for non-existent key."""
        with pytest.raises(KeyError, match="Không tìm thấy resource"):
            hub.get("nonexistent:key")

    def test_get_config_nonexistent_raises(self, hub):
        """Test get_config raises KeyError for non-existent key."""
        with pytest.raises(KeyError, match="Không tìm thấy config"):
            hub.get_config("nonexistent:key")


# ============================================================================
# Tests: Singleton Pattern
# ============================================================================

class TestSingletonPattern:
    """Test singleton instance management."""

    def test_set_and_get_instance(self, tmp_path):
        """Test setting and getting singleton instance."""
        # Clear any existing instance
        ResourceHub._instance = None

        config_file = tmp_path / "singleton.yaml"
        hub = ResourceHub.from_yaml(config_file)
        ResourceHub.set_instance(hub)

        singleton = ResourceHub.instance()
        assert singleton is hub

        # Clean up
        ResourceHub._instance = None

    def test_instance_not_initialized_raises(self):
        """Test instance raises error if not initialized."""
        ResourceHub._instance = None

        with pytest.raises(RuntimeError, match="chưa được khởi tạo"):
            ResourceHub.instance()


# ============================================================================
# Tests: File Persistence
# ============================================================================

class TestFilePersistence:
    """Test file-based persistence."""

    def test_yaml_persistence(self, tmp_path):
        """Test YAML file persistence."""
        config_file = tmp_path / "persist.yaml"

        # Register config class and handler
        register_config_class(MockServiceConfig)
        register_factory_handler(MockServiceConfig, mock_service_factory)

        # Create hub and register resource
        hub1 = ResourceHub.from_yaml(config_file)
        config = MockServiceConfig(name="persistent", host="example.com", port=9000)
        key = hub1.register(config)
        hub1.close()

        # Create new hub from same file
        hub2 = ResourceHub.from_yaml(config_file)

        # Verify resource is loaded
        assert hub2.has(key)
        service = hub2.get(key)
        assert service.name == "persistent"
        assert service.host == "example.com"
        assert service.port == 9000
        hub2.close()

    def test_json_persistence(self, tmp_path):
        """Test JSON file persistence."""
        config_file = tmp_path / "persist.json"

        # Register config class and handler
        register_config_class(MockServiceConfig)
        register_factory_handler(MockServiceConfig, mock_service_factory)

        # Create hub and register resource
        hub1 = ResourceHub.from_json(config_file)
        config = MockServiceConfig(name="json-service", host="json.example.com", port=8888)
        key = hub1.register(config)
        hub1.close()

        # Create new hub from same file
        hub2 = ResourceHub.from_json(config_file)

        # Verify resource is loaded
        assert hub2.has(key)
        service = hub2.get(key)
        assert service.name == "json-service"
        hub2.close()
