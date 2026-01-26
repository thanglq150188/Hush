# Kiến trúc Hush

Tài liệu này mô tả kiến trúc nội bộ của Hush framework dành cho contributors và plugin developers.

## Package Structure

```
hush/
├── hush-core/           # Core engine và abstractions
│   └── hush/core/
│       ├── engine/      # Hush execution engine
│       ├── nodes/       # Node types (Graph, Branch, Loop, Code, etc.)
│       ├── states/      # State management (MemoryState, RedisState)
│       ├── registry/    # Resource management (ConfigRegistry, ResourceHub)
│       ├── tracers/     # Tracing infrastructure
│       ├── streams/     # Streaming service
│       └── utils/       # Utilities (YamlModel, Param, etc.)
│
├── hush-providers/      # LLM/Embedding/Reranking implementations
│   └── hush/providers/
│       ├── llms/        # LLM providers (OpenAI, Azure, Gemini)
│       ├── embeddings/  # Embedding providers
│       ├── rerankers/   # Reranking providers
│       ├── nodes/       # Provider-specific nodes (LLMNode, EmbeddingNode)
│       └── registry/    # Plugin registration
│
└── hush-observability/  # External tracing adapters
    └── hush/observability/
        ├── backends/    # Langfuse, OpenTelemetry adapters
        └── plugin.py    # Auto-registration
```

## Key Components

### 1. Hush Engine

`Hush` là execution engine chính, chịu trách nhiệm:
- Parse graph topology từ `GraphNode`
- Resolve dependencies giữa các nodes
- Execute nodes theo thứ tự (parallel khi có thể)
- Manage state propagation

```python
engine = Hush(graph)
result = await engine.run(inputs={...})
```

### 2. Node System

Tất cả nodes kế thừa từ `BaseNode`:

| Node Type | Mô tả |
|-----------|-------|
| `GraphNode` | Container cho subgraph |
| `CodeNode` | Execute Python function |
| `BranchNode` | Conditional routing |
| `ForLoopNode` | Fixed iteration |
| `WhileLoopNode` | Conditional iteration |
| `MapNode` | Parallel iteration |
| `AsyncIterNode` | Async streaming iteration |

### 3. State Management

State system quản lý data flow giữa nodes:
- `StateSchema` - Define state structure
- `MemoryState` - In-memory state (default)
- `RedisState` - Distributed state
- `Ref` / `Cell` - Reactive references

Chi tiết: [State Architecture](state.md)

### 4. Registry System

Registry system quản lý resources và configs:
- `ConfigRegistry` - Type → Config class mapping
- `ResourceHub` - Config → Instance management
- Plugin pattern cho extensibility

Chi tiết: [Registry Architecture](registry.md)

### 5. Tracing

Tracing infrastructure cho observability:
- `BaseTracer` - Abstract interface
- `LocalTracer` - File-based tracing
- External adapters (Langfuse, OpenTelemetry)

## Execution Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Inputs    │───>│    Hush     │───>│   Outputs   │
└─────────────┘    │   Engine    │    └─────────────┘
                   └──────┬──────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   ┌─────────┐      ┌─────────┐      ┌─────────┐
   │  Node   │      │  Node   │      │  Node   │
   │    A    │─────>│    B    │─────>│    C    │
   └─────────┘      └─────────┘      └─────────┘
        │                 │                 │
        └─────────────────┴─────────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │    State    │
                   └─────────────┘
```

1. **Parse Phase**: Engine parse graph để xác định node dependencies
2. **Schedule Phase**: Xác định thứ tự execution (topological sort)
3. **Execute Phase**: Run nodes, propagate state
4. **Collect Phase**: Gather outputs từ terminal nodes

## Extension Points

### Adding New Node Types

1. Kế thừa từ `BaseNode`
2. Implement `_run()` method
3. Define inputs/outputs schema

### Adding New Providers

1. Create config class với `_type` và `_category`
2. Implement provider class
3. Create factory function
4. Register với `REGISTRY.register()`

### Adding New Storage Backends

1. Implement `ConfigStorage` interface
2. Provide `load_one()`, `load_all()`, `save()`, `remove()` methods

## Tiếp theo

- [Registry Architecture](registry.md) - Chi tiết về ResourceHub và ConfigRegistry
- [State Architecture](state.md) - Chi tiết về State system
