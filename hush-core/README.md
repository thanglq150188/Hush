# Hush Core

A powerful async workflow orchestration framework.

## Installation

```bash
pip install hush-core
```

Or with optional dependencies:

```bash
pip install hush-core[all]  # Install all hush packages
pip install hush-core[providers,storage]  # Install specific extras
```

## Quick Start

```python
from hush.core import WorkflowEngine, START, END, BaseNode

# Define a simple workflow
with WorkflowEngine(name="my-workflow") as workflow:
    # Add your nodes here
    # node1 = YourNode(name="step1", ...)
    # node2 = YourNode(name="step2", ...)
    # START >> node1 >> node2 >> END
    pass

# Compile the workflow
workflow.compile()

# Run the workflow
import asyncio

async def main():
    result = await workflow.run(
        inputs={"query": "hello world"},
        user_id="user-123",
        session_id="session-456"
    )
    print(result)

asyncio.run(main())
```

## Package Structure

```
hush-core/
    hush/
        core/
            __init__.py          # Main exports
            workflow.py          # WorkflowEngine
            nodes/
                base.py          # BaseNode, DummyNode, START, END, etc.
                graph/
                    graph_node.py
                flow/
                    branch_node.py
                    for_loop_node.py
                    while_loop_node.py
                    stream_node.py
            states/
                workflow_state.py
                workflow_indexer.py
                state_manager.py
                state_value.py
            configs/
                node_config.py
                edge_config.py
            schema/
                __init__.py      # ParamSet, Param
            utils/
                context.py
                bimap.py
                common.py
```

## Core Components

### WorkflowEngine

The main entry point for defining and executing workflows:

```python
with WorkflowEngine(name="my-workflow", description="...") as workflow:
    # Define nodes and flow
    pass

workflow.compile()
result = await workflow.run(inputs={...})
```

### Nodes

- **BaseNode**: Abstract base class for all nodes
- **GraphNode**: Container for subgraphs
- **BranchNode**: Conditional routing
- **ForLoopNode**: Iterate over collections
- **WhileLoopNode**: Conditional iteration
- **StreamNode**: Process streaming data

### Flow Control

Use `>>` operator to define flow:

```python
START >> node1 >> node2 >> END
START >> node1 >> [node2a, node2b]  # Fork
[node2a, node2b] >> node3 >> END    # Merge
```

### State Management

- **WorkflowState**: Tracks execution context and variables
- **WorkflowIndexer**: Manages node dependencies and variable mapping
- **StateRegistry**: Singleton registry for workflow states

## Related Packages

- `hush-providers`: LLM, embeddings, reranking providers
- `hush-storage`: Vector DB, MongoDB, S3, cache
- `hush-observability`: Tracing, logging, metrics
- `hush-integrations`: Tools, MCP, document processing
- `hush-templates`: Pre-built workflow templates

## License

thanglq150188@gmail.com
