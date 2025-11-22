# Hush Experiment

Experimental and testing project for the Hush ecosystem. This project is used to:

- Test the integration between `hush-core` and `hush-providers`
- Verify installation and dependency management
- Provide example implementations and usage patterns
- Demonstrate real-world workflows using Hush

## Project Structure

```
hush-experiment/
├── examples/           # Example scripts demonstrating Hush usage
│   ├── basic_embedding.py       # Simple embedding workflow
│   └── advanced_workflow.py     # Complex RAG pipeline with multiple providers
├── tests/              # Integration tests
│   └── integration/    # End-to-end integration tests
├── pyproject.toml      # Project configuration with local package dependencies
└── README.md           # This file
```

## Setup

### Prerequisites

- Python 3.10 or higher
- The parent `hush-core` and `hush-providers` packages (installed automatically from local paths)
- A valid `resources.yaml` configuration file in the parent directory

### Installation

From the `hush-experiment` directory:

```bash
# Install with lightweight providers (recommended for testing)
pip install -e ".[all]"

# Or install with full ML frameworks (if needed)
pip install -e ".[full]"

# Install dev dependencies for testing
pip install -e ".[dev]"
```

Using `uv` (faster, recommended):

```bash
# Install with lightweight providers
uv pip install -e ".[all]"

# With dev dependencies
uv pip install -e ".[dev]"
```

## Configuration

Before running examples, ensure you have a `resources.yaml` file in the parent directory (`../resources.yaml`). This file should define your provider configurations:

```yaml
# Example resources.yaml structure
embedding:
  bge-m3:
    backend: onnx
    model_name: BAAI/bge-m3
    # ... other config

llm:
  gpt-4:
    provider: openai
    model: gpt-4
    api_key: ${OPENAI_API_KEY}
    # ... other config

reranking:
  bge-m3:
    backend: onnx
    model_name: BAAI/bge-reranker-m3
    # ... other config
```

See the main `resources.yaml` in the root directory for a complete example.

## Running Examples

### Basic Embedding Example

This demonstrates a simple workflow that embeds text using the Hush ecosystem:

```bash
cd examples
python basic_embedding.py
```

**What it does:**
- Loads ResourceHub from YAML configuration
- Creates a simple embedding workflow
- Processes sample texts and generates embeddings

### Advanced Workflow Example

This demonstrates a complex RAG (Retrieval Augmented Generation) pipeline:

```bash
cd examples
python advanced_workflow.py
```

**What it does:**
- Embeds a user query
- Embeds document chunks
- Reranks documents based on relevance
- Generates an LLM response using top-ranked documents
- Demonstrates multi-node workflow orchestration

## Development

### Running Tests

```bash
# Run all integration tests
pytest tests/

# Run specific test file
pytest tests/integration/test_embedding.py

# Run with verbose output
pytest -v tests/
```

### Creating New Examples

1. Create a new file in `examples/`
2. Import required providers from `hush.providers`
3. Set up ResourceHub with necessary plugins
4. Build your workflow using WorkflowEngine
5. Add documentation explaining what the example demonstrates

Example template:

```python
"""Description of what this example demonstrates."""

import asyncio
from hush.core import WorkflowEngine, START, END, INPUT, OUTPUT
from hush.core.registry import ResourceHub
from hush.providers import YourNode, YourPlugin


async def main():
    # Setup
    hub = ResourceHub.from_yaml("../resources.yaml")
    hub.register_plugin(YourPlugin)
    ResourceHub.set_instance(hub)

    # Build workflow
    with WorkflowEngine(name="example") as workflow:
        # Add nodes and connections
        pass

    workflow.compile()
    result = await workflow.run(inputs={})
    return result


if __name__ == "__main__":
    asyncio.run(main())
```

## Testing Integration

This project is specifically designed to test the Hush ecosystem from a user's perspective. When developing new features in `hush-core` or `hush-providers`, test them here to ensure:

1. Installation works correctly
2. APIs are intuitive and ergonomic
3. Documentation is accurate
4. Dependencies are properly declared
5. Real-world usage patterns work as expected

## Testing Your Installation

Run the import test to verify everything is working:

```bash
uv run python test_import.py
```

This will verify that all key modules can be imported successfully.

## Common Issues

### ModuleNotFoundError: No module named 'hush.providers'

This usually means the namespace package setup isn't working. Make sure:

1. Both `hush-core/hush/__init__.py` and `hush-providers/hush/__init__.py` contain:
   ```python
   __path__ = __import__('pkgutil').extend_path(__path__, __name__)
   ```

2. Run `uv sync` to reinstall packages:
   ```bash
   uv sync
   ```

### ImportError: No module named 'hush'

Make sure you've synced the environment:
```bash
uv sync
```

Or if using pip:
```bash
pip install -e ".[all]"
```

### ResourceHub not finding resources

Verify that `resources.yaml` exists in the parent directory and contains valid configurations.

### Provider initialization errors

Check that you've registered the necessary plugins before using nodes:
```python
hub.register_plugin(EmbeddingPlugin)
hub.register_plugin(LLMPlugin)
hub.register_plugin(RerankPlugin)
```

### Missing dependencies for specific providers

Install the specific provider extras:
```bash
# For ONNX-based providers
uv sync --extra all

# For HuggingFace models
uv pip install "hush-providers[huggingface]"

# For Gemini
uv pip install "hush-providers[gemini]"
```

## Contributing

When adding new examples or tests:

1. Follow the existing code structure
2. Add clear documentation explaining what the example does
3. Keep examples simple and focused on specific features
4. Test your examples before committing

## License

MIT
