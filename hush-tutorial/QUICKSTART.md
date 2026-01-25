# Hush Experiment - Quick Start

Test and experiment with the Hush ecosystem.

## Install

```bash
cd hush-experiment
uv sync
```

## Verify Installation

```bash
uv run python test_import.py
```

## Run Examples

### Simple Workflow (no external dependencies)

```bash
uv run python examples/simple_workflow.py
```

### Basic Embedding (requires resources.yaml)

```bash
uv run python examples/basic_embedding.py
```

### Advanced RAG Pipeline (requires resources.yaml + API keys)

```bash
uv run python examples/advanced_workflow.py
```

## Create Your Own

```python
from hush.core import WorkflowEngine, START, END
from hush.core.nodes import code_node

@code_node
def process(text: str):
    """Process text.

    Returns:
        result (str): Processed text
    """
    return {"result": text.upper()}

async def main():
    with WorkflowEngine(name="my_workflow") as wf:
        node = process(inputs={"text": "hello"})
        START >> node >> END

    wf.compile()
    result = await wf.run(inputs={"text": "hello"})
    print(result)
```

## Troubleshooting

### Import errors
```bash
rm -rf .venv && uv sync
```

### Missing providers
```bash
uv sync --extra all
```
