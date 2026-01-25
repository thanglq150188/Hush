# Hush AI

Async workflow orchestration engine for GenAI applications.

> **Note**: This is the meta-package. Install with `pip install hush-ai[standard]`

## Installation

```bash
# Core only - workflow engine with local tracing
pip install hush-ai[core]

# Standard - workflow engine + LLM providers (OpenAI)
pip install hush-ai[standard]

# Everything (lightweight)
pip install hush-ai[all]
```

## Installation Options

### Tiers

| Tier | Description |
|------|-------------|
| `hush-ai[core]` | Workflow engine + local SQLite tracing + web UI |
| `hush-ai[standard]` | Core + LLM/embedding/rerank providers (OpenAI) |
| `hush-ai[all]` | Standard + all providers + observability (light) |
| `hush-ai[full]` | All + heavy ML frameworks (torch, transformers) |

### LLM Providers

```bash
pip install hush-ai[openai]      # OpenAI
pip install hush-ai[azure]       # Azure OpenAI
pip install hush-ai[gemini]      # Google Gemini
pip install hush-ai[openai,gemini]  # Multiple providers
```

### Local Inference

```bash
pip install hush-ai[onnx]        # ONNX Runtime (lightweight)
pip install hush-ai[huggingface] # HuggingFace + PyTorch (heavy)
```

### Observability

```bash
pip install hush-ai[standard,langfuse]  # Add Langfuse tracing
pip install hush-ai[standard,otel]      # Add OpenTelemetry
```

## Quick Start

```python
from hush.core import Hush, Graph
from hush.core.nodes import FunctionNode
from hush.providers.llms import OpenAIChat

# Define nodes
llm = OpenAIChat(model="gpt-4o-mini")

@FunctionNode
async def process(state):
    response = await llm.chat([{"role": "user", "content": state["prompt"]}])
    return {"response": response.content}

# Build and run workflow
graph = Graph()
graph.add_node(process)

engine = Hush(graph)
result = await engine.run({"prompt": "Hello, world!"})
```

## Local Trace Viewer

Traces are automatically saved to `~/.hush/traces.db`. View them with:

```bash
python -m hush.core.ui.server
# Open http://localhost:8765
```

## Packages

- **hush-core**: Workflow engine, state management, local tracing
- **hush-providers**: LLM, embedding, reranking providers
- **hush-observability**: External tracing (Langfuse, OpenTelemetry)

## License

MIT
