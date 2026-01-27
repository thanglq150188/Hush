# Agent Workflow

Ví dụ xây dựng AI Agent có khả năng sử dụng tools với Hush.

## Tổng quan Architecture

```
User Query → Plan → Execute Tools → Observe → (Loop) → Final Answer
```

## Cấu hình Resources

```yaml
llm:agent:
  _class: OpenAIConfig
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o
  base_url: https://api.openai.com/v1
```

## Tool-Calling Agent

```python
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT
from hush.core.nodes.iteration import WhileLoopNode
from hush.providers import LLMNode
import json

# Define tools
def calculator(expression: str) -> dict:
    try:
        result = eval(expression)  # Use safer eval in production
        return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}


def search(query: str) -> dict:
    mock_results = {
        "python": "Python is a programming language.",
        "hush": "Hush is a workflow orchestration framework.",
    }
    for key, value in mock_results.items():
        if key in query.lower():
            return {"result": value}
    return {"result": "No information found."}


TOOLS = {"calculator": calculator, "search": search}

TOOL_DESCRIPTIONS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate mathematical expressions",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    }
]


def process_response(content, tool_calls, messages, iteration):
    new_messages = messages.copy()
    new_messages.append({
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls
    })

    if tool_calls:
        for tool_call in tool_calls:
            func_name = tool_call["function"]["name"]
            args = json.loads(tool_call["function"]["arguments"])

            if func_name in TOOLS:
                result = TOOLS[func_name](**args)
            else:
                result = {"error": f"Unknown tool: {func_name}"}

            new_messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(result)
            })

        return {
            "new_messages": new_messages,
            "new_iteration": iteration + 1,
            "is_done": False,
            "answer": None
        }
    else:
        return {
            "new_messages": new_messages,
            "new_iteration": iteration + 1,
            "is_done": True,
            "answer": content
        }


with GraphNode(name="simple-agent") as graph:
    init = CodeNode(
        name="init",
        code_fn=lambda query: {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant with tools."},
                {"role": "user", "content": query}
            ],
            "iteration": 0,
            "done": False,
            "final_answer": None
        },
        inputs={"query": PARENT["query"]}
    )

    with WhileLoopNode(
        name="agent_loop",
        inputs={
            "messages": init["messages"],
            "iteration": init["iteration"],
            "done": init["done"],
            "final_answer": init["final_answer"]
        },
        stop_condition="done == True or iteration >= 5",
        max_iterations=10
    ) as loop:
        llm = LLMNode(
            name="llm",
            resource_key="llm:agent",
            inputs={
                "messages": PARENT["messages"],
                "tools": TOOL_DESCRIPTIONS
            }
        )

        process = CodeNode(
            name="process",
            code_fn=lambda content, tool_calls, messages, iteration: process_response(
                content, tool_calls, messages, iteration
            ),
            inputs={
                "content": llm["content"],
                "tool_calls": llm["tool_calls"],
                "messages": PARENT["messages"],
                "iteration": PARENT["iteration"]
            }
        )

        process["new_messages"] >> PARENT["messages"]
        process["new_iteration"] >> PARENT["iteration"]
        process["is_done"] >> PARENT["done"]
        process["answer"] >> PARENT["final_answer"]

        START >> llm >> process >> END

    loop["final_answer"] >> PARENT["answer"]
    START >> init >> loop >> END


# Usage
async def main():
    engine = Hush(graph)
    result = await engine.run(inputs={
        "query": "What is 25 * 4 + 100?"
    })
    print(f"Answer: {result['answer']}")

import asyncio
asyncio.run(main())
```

## Parallel Tool Execution

Agent có thể chạy nhiều tools song song:

```python
from hush.core.nodes.iteration import MapNode
from hush.core.nodes.iteration.base import Each

with MapNode(
    name="execute_tools",
    inputs={"tool_call": Each(llm["tool_calls"])},
    max_concurrency=5
) as tool_map:
    execute = CodeNode(
        name="execute",
        code_fn=lambda tool_call: {"result": execute_single_tool(tool_call)},
        inputs={"tool_call": PARENT["tool_call"]},
        outputs={"result": PARENT}
    )
    START >> execute >> END
```

## Best Practices

1. **Limit iterations** - Luôn set `max_iterations` để tránh infinite loops
2. **Clear tool descriptions** - Mô tả tool rõ ràng để LLM hiểu cách sử dụng
3. **Validate arguments** - Validate tool arguments trước khi execute
4. **Tracing** - Monitor agent behavior với Langfuse

## Xem thêm

- [RAG Workflow](rag-workflow.md)
- [Multi-Model Workflow](multi-model.md)
