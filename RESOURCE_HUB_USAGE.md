# Global RESOURCE_HUB and Auto-Registration Guide

## Overview

The Hush framework now features **automatic plugin registration** and a **global RESOURCE_HUB** for maximum convenience!

### Key Features

✅ **Auto-Registration**: Plugins automatically register themselves when imported
✅ **Global Hub**: `RESOURCE_HUB` is available globally - no setup needed
✅ **Zero Boilerplate**: No more manual `hub.register_plugin()` calls
✅ **Backwards Compatible**: Old patterns still work perfectly

---

## Quick Start (New Way)

```python
from hush.core import RESOURCE_HUB, WorkflowEngine, START, END, INPUT, OUTPUT
from hush.providers import LLMNode  # Plugin auto-registers!
from hush.providers.llms.config import OpenAIConfig

# Just register your config - plugins are already registered!
llm_config = OpenAIConfig(
    api_type="openai",
    api_key="your-key",
    base_url="https://api.openai.com/v1",
    model="gpt-4"
)

RESOURCE_HUB.register(llm_config, persist=False)

# Use nodes directly - they use the global hub
with WorkflowEngine(name="simple-workflow") as workflow:
    llm = LLMNode(
        name="chat",
        resource_key="openai:gpt-4",
        inputs={"messages": INPUT},
        outputs={"content": OUTPUT}
    )
    START >> llm >> END

workflow.compile()
```

**That's it!** No manual plugin registration needed! 🎉

---

## Configuration File Support

The global hub automatically looks for config files in this order:

1. **Environment variable**: `HUSH_CONFIG=/path/to/config.yaml`
2. **Current directory**: `./resources.yaml`
3. **User home**: `~/.hush/resources.yaml`
4. **Fallback**: In-memory storage

### Example: Using resources.yaml

Create `resources.yaml` in your project:

```yaml
llm:openai:gpt-4:
  _class: OpenAIConfig
  api_type: openai
  api_key: ${OPENAI_API_KEY}
  model: gpt-4
  base_url: https://api.openai.com/v1

embedding:bge-m3:
  _class: EmbeddingConfig
  api_type: hf
  model: BAAI/bge-m3
  dimensions: 1024
```

Then just import and use:

```python
from hush.core import RESOURCE_HUB
from hush.providers import LLMNode, EmbeddingNode

# Configs are automatically loaded from resources.yaml!
llm = RESOURCE_HUB.llm("openai:gpt-4")
embedding = RESOURCE_HUB.embedding("bge-m3")
```

---

## Advanced Usage

### Custom Global Hub

Override the default global hub:

```python
from hush.core import ResourceHub, set_global_hub

# Create custom hub
custom_hub = ResourceHub.from_yaml("my_config.yaml")

# Set as global
set_global_hub(custom_hub)

# Now all imports use this hub
from hush.providers import LLMNode  # Uses custom_hub
```

### Multiple Hubs (Advanced)

You can still create and use multiple hubs:

```python
from hush.core import ResourceHub

# Production hub
prod_hub = ResourceHub.from_yaml("prod_resources.yaml")

# Dev/test hub
dev_hub = ResourceHub.from_memory()

# Use specific hubs
prod_llm = prod_hub.llm("gpt-4")
dev_llm = dev_hub.llm("gpt-3.5-turbo")
```

### Creating Custom Plugins

Your custom plugins will auto-register too!

```python
from hush.core.registry import ResourcePlugin, ResourceConfig

class MyCustomPlugin(ResourcePlugin):
    @classmethod
    def config_class(cls):
        return MyConfig

    @classmethod
    def create(cls, config):
        return MyCustomResource(config)

    @classmethod
    def resource_type(cls):
        return "custom"

# That's it! Plugin auto-registers when this module is imported
```

---

## Migration Guide

### Before (Old Way)

```python
# ❌ Old: Lots of boilerplate
from hush.core.registry import ResourceHub
from hush.providers import LLMPlugin, EmbeddingPlugin, RerankPlugin

hub = ResourceHub.from_yaml("resources.yaml")

# Manual registration required
hub.register_plugins(LLMPlugin, EmbeddingPlugin, RerankPlugin)

# Set as singleton
ResourceHub.set_instance(hub)

# Finally use it
from hush.providers import LLMNode
node = LLMNode(...)
```

### After (New Way)

```python
# ✅ New: Just import and use!
from hush.core import RESOURCE_HUB
from hush.providers import LLMNode  # Auto-registers!

node = LLMNode(...)  # Just works!
```

---

## How It Works

### Auto-Registration Mechanism

When you import a plugin module, the `PluginMeta` metaclass automatically:

1. Detects the plugin class definition
2. Registers it with the global `RESOURCE_HUB`
3. Maps config classes (including subclasses) to the plugin

```python
# This import triggers auto-registration
from hush.providers import LLMPlugin  # Registers itself!

# Check it worked
from hush.core import RESOURCE_HUB
assert RESOURCE_HUB.has_plugin("llm")  # ✓ True
```

### Lazy Initialization

The global hub is created lazily on first access:

```python
from hush.core import RESOURCE_HUB  # Creates hub now

# Subsequent imports reuse the same instance
from hush.core.registry import get_hub
hub = get_hub()  # Same instance as RESOURCE_HUB
```

---

## Best Practices

### ✅ Do This

```python
# Import from top-level packages
from hush.core import RESOURCE_HUB
from hush.providers import LLMNode

# Use descriptive resource keys
RESOURCE_HUB.register(config, registry_key="llm:prod:gpt-4")

# Leverage auto-registration
# Just import and use - no setup needed!
```

### ❌ Avoid This

```python
# Don't manually register plugins unless you need a custom hub
hub = ResourceHub.from_memory()
hub.register_plugin(LLMPlugin)  # Not needed anymore!

# Don't create multiple global hubs
# Use set_global_hub() if you need to override
```

---

## Troubleshooting

### Plugins Not Auto-Registering?

Make sure you import the plugin module:

```python
# This doesn't import plugins
from hush.core import RESOURCE_HUB

# This does!
from hush.providers import LLMPlugin  # or LLMNode, which imports LLMPlugin
```

### Resource Not Found?

Check the registry keys:

```python
from hush.core import RESOURCE_HUB

# List all registered resources
print(RESOURCE_HUB.keys())

# Check if plugin is registered
print(RESOURCE_HUB._plugins.keys())
```

### Want to Reset Global Hub?

```python
from hush.core.registry import set_global_hub, ResourceHub

# Create fresh hub
fresh_hub = ResourceHub.from_memory()
set_global_hub(fresh_hub)
```

---

## Summary

The new auto-registration system makes Hush even more developer-friendly:

- **No more boilerplate**: Plugins register automatically
- **Global hub**: `RESOURCE_HUB` ready to use
- **Config file support**: Automatically loads from standard locations
- **Fully backwards compatible**: Old code still works

Just import and build! 🚀
