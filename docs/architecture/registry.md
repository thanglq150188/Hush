# Registry Architecture

Tài liệu này mô tả kiến trúc nội bộ của Registry system trong Hush.

## Overview

Registry system gồm 2 components chính:

```
┌─────────────────────────────────────────────────────────────────┐
│                        ConfigRegistry                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ _entries: Dict[category, Dict[type, ConfigEntry]]         │  │
│  │                                                           │  │
│  │  "llm"       → {"openai": ConfigEntry, "azure": ...}     │  │
│  │  "embedding" → {"embedding": ConfigEntry}                 │  │
│  │  "_class"    → {"OpenAIConfig": ConfigEntry, ...}        │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ REGISTRY.get_class() / REGISTRY.create()
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         ResourceHub                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ _cache: Dict[key, CacheEntry]                             │  │
│  │                                                           │  │
│  │  "llm:gpt-4"     → CacheEntry(config, instance)          │  │
│  │  "embedding:bge" → CacheEntry(config, instance)          │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ _storage: ConfigStorage (YAML/JSON)                       │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Structures

### ConfigEntry

```python
@dataclass
class ConfigEntry:
    config_class: Type[YamlModel]  # e.g., OpenAIConfig
    factory: Callable[[YamlModel], Any]  # e.g., LLMFactory.create
```

Chứa mapping từ config class đến factory function.

### CacheEntry

```python
@dataclass
class CacheEntry:
    config: YamlModel  # Parsed config object
    instance: Any = None  # Lazy-loaded instance
```

Chứa config và instance (lazy loaded).

## ConfigRegistry

Singleton registry quản lý type → config class mapping.

### Registration Flow

```
Plugin Import
     │
     ▼
REGISTRY.register(OpenAIConfig, LLMFactory.create)
     │
     ├──▶ _entries["llm"]["openai"] = ConfigEntry(...)
     │    (từ _category và _type của config class)
     │
     └──▶ _entries["_class"]["OpenAIConfig"] = ConfigEntry(...)
          (fallback lookup by class name)
```

### Lookup Flow

```
REGISTRY.get_class("openai", category="llm")
     │
     ├──▶ Try: _entries["llm"]["openai"]  ✓
     │
     └──▶ Fallback: _entries["_class"]["openai"]
```

### Key Methods

| Method | Mô tả |
|--------|-------|
| `register(config_class, factory)` | Register config class với factory |
| `get_class(type_name, category)` | Lookup config class by type |
| `get_entry(type_name, category)` | Lookup full ConfigEntry |
| `create(config)` | Create instance from config |

## ResourceHub

Quản lý config loading và instance lifecycle.

### Loading Flow

```
hub.get("llm:gpt-4")
     │
     ├──▶ Check _cache["llm:gpt-4"].instance
     │    (return if exists)
     │
     ├──▶ Load config from _storage
     │    config_data = _storage.load_one("llm:gpt-4")
     │
     ├──▶ Parse config
     │    type_name = config_data["type"]  # "openai"
     │    category = "llm"  # from key prefix
     │    config_class = REGISTRY.get_class(type_name, category)
     │    config = config_class.model_validate(config_data)
     │
     ├──▶ Create instance (lazy)
     │    instance = REGISTRY.create(config)
     │
     └──▶ Cache and return
          _cache["llm:gpt-4"] = CacheEntry(config, instance)
```

### Storage Backends

```python
class ConfigStorage(ABC):
    @abstractmethod
    def load_one(self, key: str) -> Optional[Dict]: ...

    @abstractmethod
    def load_all(self) -> Dict[str, Dict]: ...

    @abstractmethod
    def save(self, key: str, data: Dict): ...

    @abstractmethod
    def remove(self, key: str): ...
```

Built-in implementations:
- `YamlConfigStorage` - YAML file backend
- `JsonConfigStorage` - JSON file backend

## Config Class Requirements

Để register với REGISTRY, config class cần:

```python
from typing import ClassVar
from hush.core.utils.yaml_model import YamlModel

class MyConfig(YamlModel):
    _type: ClassVar[str] = "my-type"        # Type alias trong YAML
    _category: ClassVar[str] = "mycategory"  # Category namespace

    # Fields...
    api_key: str
    model: str
```

## Plugin Pattern

Plugins auto-register khi import:

```python
# hush/providers/registry/llm_plugin.py

class LLMPlugin:
    _registered = False

    @classmethod
    def register(cls):
        if cls._registered:
            return

        REGISTRY.register(LLMConfig, LLMFactory.create)
        REGISTRY.register(OpenAIConfig, LLMFactory.create)
        # ...

        cls._registered = True

# Auto-register on import
LLMPlugin.register()
```

## Folder Structure

```
hush-core/hush/core/registry/
├── __init__.py              # Public exports
├── config_registry.py       # ConfigRegistry, ConfigEntry, REGISTRY
├── resource_hub.py          # ResourceHub, CacheEntry
├── storage/                 # Storage backends
│   ├── __init__.py
│   ├── base.py              # ConfigStorage ABC
│   ├── yaml.py              # YamlConfigStorage
│   └── json.py              # JsonConfigStorage
└── shortcuts/               # Convenience utilities
    ├── __init__.py
    ├── global_hub.py        # get_hub(), set_global_hub()
    └── health.py            # HealthCheckResult
```

## Testing

Tests sử dụng `ConfigRegistry.reset()` để isolate:

```python
def test_something():
    ConfigRegistry.reset()  # Clear singleton

    # Register test configs
    REGISTRY.register(MockConfig, mock_factory)

    # Test...
```

## Design Decisions

1. **Singleton Pattern**: `ConfigRegistry` và global hub sử dụng singleton để đơn giản hóa access từ anywhere trong code.

2. **Lazy Loading**: Instances chỉ được tạo khi access lần đầu (`hub.get()`), giúp startup nhanh hơn.

3. **Category Namespacing**: Cho phép same type name trong different categories (e.g., `llm:openai` vs `storage:openai`).

4. **Fallback to Class Name**: Nếu type không tìm thấy trong category, fallback lookup by class name để backward compatible với `_class:` format cũ.
