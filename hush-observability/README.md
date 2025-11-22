# Hush Observability

Flexible observability and tracing framework for Hush workflows with support for multiple backend providers.

## Features

✅ **Multiple Backend Support**
- Langfuse
- Azure Phoenix (Arize Phoenix)
- Opik
- LangSmith
- Easy to add more!

✅ **Unified Interface**
- Single API works with any backend
- Switch providers without code changes
- Async-first design with buffering

✅ **Resource Hub Integration**
- Auto-registers observability plugins
- Manage tracers like any other resource
- Configuration via YAML

✅ **Rich Tracing**
- Hierarchical traces with spans and generations
- Automatic parent-child relationships
- Token usage tracking
- Custom metadata and tags

## Quick Start

### Installation

```bash
# Install with Langfuse support
pip install hush-observability[langfuse]

# Install with all backends
pip install hush-observability[all]

# Install specific backends
pip install hush-observability[phoenix,opik]
```

### Basic Usage

```python
from hush.core import RESOURCE_HUB
from hush.observability import LangfuseConfig

# Register tracer config
config = LangfuseConfig(
    public_key="pk-...",
    secret_key="sk-...",
    host="https://cloud.langfuse.com"
)

RESOURCE_HUB.register(config, registry_key="tracer:langfuse", persist=False)

# Get tracer and use it
tracer = RESOURCE_HUB.tracer("langfuse")

# Add trace items
tracer.add_span(
    request_id="req-123",
    name="root",
    input={"query": "Hello"},
    user_id="user-123"
)

tracer.add_generation(
    request_id="req-123",
    name="llm-call",
    parent="root",
    model="gpt-4",
    input=[{"role": "user", "content": "Hello"}],
    output="Hi there!"
)

# Flush to backend
await tracer.flush("req-123")
```

### Using with Workflows

```python
from hush.core import WorkflowEngine, START, END
from hush.providers import LLMNode

# Tracer auto-integrates with workflow nodes
with WorkflowEngine(name="chat", tracer_key="langfuse") as workflow:
    llm = LLMNode(name="chat", resource_key="gpt-4", ...)
    START >> llm >> END

# Traces are automatically sent to Langfuse!
```

## Configuration

### YAML Configuration

```yaml
tracer:langfuse:
  _class: LangfuseConfig
  public_key: ${LANGFUSE_PUBLIC_KEY}
  secret_key: ${LANGFUSE_SECRET_KEY}
  host: https://cloud.langfuse.com
  no_proxy: "*.internal.com"

tracer:phoenix:
  _class: PhoenixConfig
  endpoint: http://localhost:6006

tracer:opik:
  _class: OpikConfig
  api_key: ${OPIK_API_KEY}
  workspace: my-workspace
```

### Python Configuration

```python
from hush.observability import LangfuseConfig, PhoenixConfig
from hush.core import RESOURCE_HUB

# Langfuse
langfuse_config = LangfuseConfig(
    public_key="pk-...",
    secret_key="sk-...",
    host="https://cloud.langfuse.com"
)
RESOURCE_HUB.register(langfuse_config, registry_key="tracer:langfuse")

# Phoenix
phoenix_config = PhoenixConfig(endpoint="http://localhost:6006")
RESOURCE_HUB.register(phoenix_config, registry_key="tracer:phoenix")
```

## Architecture

### Abstraction Layer

```
┌─────────────────────────────────────┐
│         Your Application            │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│      BaseTracer Interface           │
│  (add_span, add_generation, flush)  │
└──────────────┬──────────────────────┘
               │
       ┌───────┴────────┬─────────────┬──────────┐
       │                │             │          │
┌──────▼──────┐  ┌──────▼──────┐  ┌──▼───┐  ┌──▼──────┐
│  Langfuse   │  │   Phoenix   │  │ Opik │  │LangSmith│
│   Tracer    │  │   Tracer    │  │Tracer│  │ Tracer  │
└─────────────┘  └─────────────┘  └──────┘  └─────────┘
```

### Key Components

- **BaseTracer**: Abstract interface all tracers implement
- **AsyncTraceBuffer**: Hierarchical trace buffering with validation
- **TracerPlugin**: ResourceHub plugin for tracer management
- **Config Models**: Pydantic models for each backend

## Supported Backends

### Langfuse

```python
from hush.observability import LangfuseTracer, LangfuseConfig

config = LangfuseConfig(
    public_key="pk-...",
    secret_key="sk-...",
    host="https://cloud.langfuse.com"
)
tracer = LangfuseTracer(config)
```

### Azure Phoenix (Arize Phoenix)

```python
from hush.observability import PhoenixTracer, PhoenixConfig

config = PhoenixConfig(endpoint="http://localhost:6006")
tracer = PhoenixTracer(config)
```

### Opik

```python
from hush.observability import OpikTracer, OpikConfig

config = OpikConfig(
    api_key="...",
    workspace="my-workspace"
)
tracer = OpikTracer(config)
```

### LangSmith

```python
from hush.observability import LangSmithTracer, LangSmithConfig

config = LangSmithConfig(
    api_key="...",
    project="my-project"
)
tracer = LangSmithTracer(config)
```

## Advanced Features

### Hierarchical Tracing

```python
# Build parent-child relationships
tracer.add_span(request_id="req-1", name="root", parent=None)
tracer.add_generation(request_id="req-1", name="gen-1", parent="root")
tracer.add_span(request_id="req-1", name="child-span", parent="gen-1")

# Automatic validation and ordering
await tracer.flush("req-1")  # Flushes in correct hierarchy
```

### Updating Traces

```python
# Add initial trace
tracer.add_generation(
    request_id="req-1",
    name="llm",
    model="gpt-4",
    input=[{"role": "user", "content": "Hello"}]
)

# Update with output later
tracer.update_item(
    request_id="req-1",
    name="llm",
    output="Hi there!",
    usage_details={"prompt_tokens": 10, "completion_tokens": 5}
)
```

### Custom Metadata

```python
tracer.add_span(
    request_id="req-1",
    name="process",
    input={"data": "..."},
    # Trace-level metadata
    user_id="user-123",
    session_id="session-abc",
    tags=["production", "v2"],
    metadata={"environment": "prod", "region": "us-east-1"},
    # Custom data
    custom_field="value"
)
```

## Examples

See the `examples/` directory for complete examples:
- `examples/basic_tracing.py` - Basic tracing operations
- `examples/workflow_integration.py` - Integration with Hush workflows
- `examples/multi_backend.py` - Using multiple backends simultaneously
- `examples/advanced_hierarchies.py` - Complex trace hierarchies

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev,all]"

# Run tests
pytest tests/

# Run with coverage
pytest --cov=hush.observability tests/
```

### Adding a New Backend

1. Create config model in `hush/observability/config/`
2. Implement tracer in `hush/observability/tracers/`
3. Register in `__init__.py`
4. Add tests and examples

## License

MIT
