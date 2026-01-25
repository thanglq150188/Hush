# Agent Workflow

Ví dụ này hướng dẫn xây dựng AI Agent có khả năng sử dụng tools với Hush.

## Tổng quan Architecture

```
User Query → Plan → Execute Tools → Observe → (Loop) → Final Answer
```

## Cấu hình Resources

### resources.yaml

```yaml
llm:agent:
  _class: OpenAIConfig
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o
  base_url: https://api.openai.com/v1

llm:fast:
  _class: OpenAIConfig
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o-mini
  base_url: https://api.openai.com/v1
```

## Ví dụ 1: Simple Tool-Calling Agent

Agent với một vài tools cơ bản.

```python
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT
from hush.core.nodes.iteration.while_loop_node import WhileLoopNode
from hush.providers import PromptNode, LLMNode
import json

# Define tools
def calculator(expression: str) -> dict:
    """Evaluate a math expression."""
    try:
        result = eval(expression)  # Note: Use safer eval in production
        return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}


def search(query: str) -> dict:
    """Search for information (mock)."""
    # In production, use real search API
    mock_results = {
        "python": "Python is a programming language created by Guido van Rossum.",
        "hush": "Hush is a workflow orchestration framework for AI applications.",
    }
    for key, value in mock_results.items():
        if key in query.lower():
            return {"result": value}
    return {"result": "No information found."}


def get_weather(city: str) -> dict:
    """Get weather for a city (mock)."""
    # In production, use real weather API
    return {"result": f"Weather in {city}: 25°C, Sunny"}


# Tool registry
TOOLS = {
    "calculator": calculator,
    "search": search,
    "get_weather": get_weather
}

TOOL_DESCRIPTIONS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate mathematical expressions",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g., '2 + 2 * 3'"
                    }
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
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name"
                    }
                },
                "required": ["city"]
            }
        }
    }
]


with GraphNode(name="simple-agent") as graph:
    # Initialize
    init = CodeNode(
        name="init",
        code_fn=lambda query: {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant with access to tools."},
                {"role": "user", "content": query}
            ],
            "iteration": 0,
            "max_iterations": 5,
            "done": False,
            "final_answer": None
        },
        inputs={"query": PARENT["query"]}
    )

    # Agent loop
    with WhileLoopNode(
        name="agent_loop",
        inputs={
            "messages": init["messages"],
            "iteration": init["iteration"],
            "max_iterations": init["max_iterations"],
            "done": init["done"],
            "final_answer": init["final_answer"]
        },
        stop_condition="done == True or iteration >= max_iterations",
        max_iterations=10
    ) as loop:
        # Call LLM with tools
        llm = LLMNode(
            name="llm",
            resource_key="llm:agent",
            inputs={
                "messages": PARENT["messages"],
                "tools": TOOL_DESCRIPTIONS
            }
        )

        # Process LLM response
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
    loop["messages"] >> PARENT["conversation"]

    START >> init >> loop >> END


def process_response(content, tool_calls, messages, iteration):
    """Process LLM response and execute tools if needed."""
    new_messages = messages.copy()
    new_messages.append({
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls
    })

    if tool_calls:
        # Execute each tool call
        for tool_call in tool_calls:
            func_name = tool_call["function"]["name"]
            args = json.loads(tool_call["function"]["arguments"])

            if func_name in TOOLS:
                result = TOOLS[func_name](**args)
            else:
                result = {"error": f"Unknown tool: {func_name}"}

            # Add tool result to messages
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
        # No tool calls - final answer
        return {
            "new_messages": new_messages,
            "new_iteration": iteration + 1,
            "is_done": True,
            "answer": content
        }


# Usage
async def main():
    engine = Hush(graph)

    result = await engine.run(inputs={
        "query": "What is 25 * 4 + 100? Also, what's the weather in Hanoi?"
    })

    print(f"Answer: {result['answer']}")

import asyncio
asyncio.run(main())
```

## Ví dụ 2: ReAct Agent

Agent sử dụng ReAct (Reasoning + Acting) pattern.

```python
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT
from hush.core.nodes.iteration.while_loop_node import WhileLoopNode
from hush.providers import PromptNode, LLMNode
import re

REACT_SYSTEM_PROMPT = """You are an AI assistant that follows the ReAct pattern:
1. Thought: Think about what to do
2. Action: Choose an action from available tools
3. Observation: See the result
4. ... (repeat if needed)
5. Final Answer: Provide the final answer

Available tools:
- calculator(expression): Evaluate math expressions
- search(query): Search for information
- get_weather(city): Get weather for a city

Format your response as:
Thought: <your reasoning>
Action: <tool_name>(<arguments>)

OR if you have the final answer:
Thought: <your reasoning>
Final Answer: <your answer>
"""


def parse_react_response(response: str):
    """Parse ReAct formatted response."""
    # Check for final answer
    if "Final Answer:" in response:
        answer = response.split("Final Answer:")[-1].strip()
        return {"type": "answer", "answer": answer}

    # Parse action
    action_match = re.search(r'Action:\s*(\w+)\(([^)]*)\)', response)
    if action_match:
        tool_name = action_match.group(1)
        args_str = action_match.group(2).strip()
        # Simple arg parsing
        args = args_str.strip('"\'')
        return {"type": "action", "tool": tool_name, "args": args}

    return {"type": "unknown", "response": response}


def execute_tool(tool_name: str, args: str):
    """Execute a tool and return observation."""
    tools = {
        "calculator": lambda x: str(eval(x)),
        "search": lambda q: f"Search result for '{q}': Information found.",
        "get_weather": lambda c: f"Weather in {c}: 28°C, Partly cloudy"
    }

    if tool_name in tools:
        try:
            result = tools[tool_name](args)
            return f"Observation: {result}"
        except Exception as e:
            return f"Observation: Error - {str(e)}"
    return f"Observation: Unknown tool '{tool_name}'"


with GraphNode(name="react-agent") as graph:
    # Initialize
    init = CodeNode(
        name="init",
        code_fn=lambda query: {
            "scratchpad": f"Question: {query}\n\n",
            "iteration": 0,
            "max_iterations": 5,
            "done": False,
            "final_answer": None
        },
        inputs={"query": PARENT["query"]}
    )

    # ReAct loop
    with WhileLoopNode(
        name="react_loop",
        inputs={
            "scratchpad": init["scratchpad"],
            "iteration": init["iteration"],
            "max_iterations": init["max_iterations"],
            "done": init["done"],
            "final_answer": init["final_answer"],
            "original_query": PARENT["query"]
        },
        stop_condition="done == True or iteration >= max_iterations",
        max_iterations=10
    ) as loop:
        # Build prompt
        prompt = PromptNode(
            name="prompt",
            inputs={
                "prompt": {
                    "system": REACT_SYSTEM_PROMPT,
                    "user": "{scratchpad}"
                },
                "scratchpad": PARENT["scratchpad"]
            }
        )

        # Call LLM
        llm = LLMNode(
            name="llm",
            resource_key="llm:agent",
            inputs={"messages": prompt["messages"]}
        )

        # Process response
        process = CodeNode(
            name="process",
            code_fn=lambda response, scratchpad, iteration: process_react(
                response, scratchpad, iteration
            ),
            inputs={
                "response": llm["content"],
                "scratchpad": PARENT["scratchpad"],
                "iteration": PARENT["iteration"]
            }
        )

        process["new_scratchpad"] >> PARENT["scratchpad"]
        process["new_iteration"] >> PARENT["iteration"]
        process["is_done"] >> PARENT["done"]
        process["answer"] >> PARENT["final_answer"]

        START >> prompt >> llm >> process >> END

    loop["final_answer"] >> PARENT["answer"]
    loop["scratchpad"] >> PARENT["reasoning"]

    START >> init >> loop >> END


def process_react(response: str, scratchpad: str, iteration: int):
    """Process ReAct response."""
    parsed = parse_react_response(response)
    new_scratchpad = scratchpad + response + "\n"

    if parsed["type"] == "answer":
        return {
            "new_scratchpad": new_scratchpad,
            "new_iteration": iteration + 1,
            "is_done": True,
            "answer": parsed["answer"]
        }
    elif parsed["type"] == "action":
        observation = execute_tool(parsed["tool"], parsed["args"])
        new_scratchpad += observation + "\n\n"
        return {
            "new_scratchpad": new_scratchpad,
            "new_iteration": iteration + 1,
            "is_done": False,
            "answer": None
        }
    else:
        # Force continue
        return {
            "new_scratchpad": new_scratchpad + "Please follow the ReAct format.\n\n",
            "new_iteration": iteration + 1,
            "is_done": False,
            "answer": None
        }


# Usage
async def main():
    engine = Hush(graph)

    result = await engine.run(inputs={
        "query": "What is the square root of 144, and what's the weather in Tokyo?"
    })

    print(f"Answer: {result['answer']}")
    print(f"\nReasoning:\n{result['reasoning']}")

import asyncio
asyncio.run(main())
```

## Ví dụ 3: Multi-Tool Agent với Parallel Execution

Agent có thể chạy nhiều tools song song.

```python
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT
from hush.core.nodes.iteration import MapNode
from hush.core.nodes.iteration.base import Each
from hush.core.nodes.iteration.while_loop_node import WhileLoopNode
from hush.providers import PromptNode, LLMNode
import json

with GraphNode(name="parallel-tool-agent") as graph:
    # Initialize
    init = CodeNode(
        name="init",
        code_fn=lambda query: {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. You can call multiple tools in parallel."},
                {"role": "user", "content": query}
            ],
            "iteration": 0,
            "done": False,
            "final_answer": None
        },
        inputs={"query": PARENT["query"]}
    )

    # Agent loop
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
        # Call LLM
        llm = LLMNode(
            name="llm",
            resource_key="llm:agent",
            inputs={
                "messages": PARENT["messages"],
                "tools": TOOL_DESCRIPTIONS,
                "parallel_tool_calls": True  # Allow parallel calls
            }
        )

        # Execute tools in parallel if multiple calls
        with MapNode(
            name="execute_tools",
            inputs={"tool_call": Each(llm["tool_calls"])},
            max_concurrency=5
        ) as tool_map:
            execute = CodeNode(
                name="execute",
                code_fn=lambda tool_call: {
                    "result": execute_single_tool(tool_call)
                },
                inputs={"tool_call": PARENT["tool_call"]}
            )
            execute["result"] >> PARENT["result"]
            START >> execute >> END

        # Aggregate and update messages
        aggregate = CodeNode(
            name="aggregate",
            code_fn=lambda content, tool_calls, tool_results, messages, iteration: aggregate_results(
                content, tool_calls, tool_results, messages, iteration
            ),
            inputs={
                "content": llm["content"],
                "tool_calls": llm["tool_calls"],
                "tool_results": tool_map["result"],
                "messages": PARENT["messages"],
                "iteration": PARENT["iteration"]
            }
        )

        aggregate["new_messages"] >> PARENT["messages"]
        aggregate["new_iteration"] >> PARENT["iteration"]
        aggregate["is_done"] >> PARENT["done"]
        aggregate["answer"] >> PARENT["final_answer"]

        START >> llm >> tool_map >> aggregate >> END

    loop["final_answer"] >> PARENT["answer"]

    START >> init >> loop >> END


def execute_single_tool(tool_call):
    """Execute a single tool call."""
    func_name = tool_call["function"]["name"]
    args = json.loads(tool_call["function"]["arguments"])

    if func_name in TOOLS:
        result = TOOLS[func_name](**args)
    else:
        result = {"error": f"Unknown tool: {func_name}"}

    return {
        "tool_call_id": tool_call["id"],
        "result": json.dumps(result)
    }


def aggregate_results(content, tool_calls, tool_results, messages, iteration):
    """Aggregate tool results and update messages."""
    new_messages = messages.copy()

    if tool_calls:
        # Add assistant message with tool calls
        new_messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls
        })

        # Add all tool results
        for result in tool_results:
            new_messages.append({
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "content": result["result"]
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
```

## Ví dụ 4: Agent với Memory

Agent có khả năng nhớ context từ các cuộc hội thoại trước.

```python
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT
from hush.core.nodes.iteration.while_loop_node import WhileLoopNode
from hush.providers import PromptNode, LLMNode

# Simple in-memory store (use Redis/DB in production)
class MemoryStore:
    def __init__(self):
        self.memories = {}

    def save(self, session_id: str, key: str, value: str):
        if session_id not in self.memories:
            self.memories[session_id] = {}
        self.memories[session_id][key] = value
        return {"status": "saved"}

    def recall(self, session_id: str, key: str):
        if session_id in self.memories and key in self.memories[session_id]:
            return {"value": self.memories[session_id][key]}
        return {"value": None, "error": "Not found"}

    def list_keys(self, session_id: str):
        if session_id in self.memories:
            return {"keys": list(self.memories[session_id].keys())}
        return {"keys": []}


memory_store = MemoryStore()

MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save information to memory for later recall",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Memory key"},
                    "value": {"type": "string", "description": "Information to save"}
                },
                "required": ["key", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": "Recall information from memory",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Memory key to recall"}
                },
                "required": ["key"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_memories",
            "description": "List all memory keys",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]


with GraphNode(name="memory-agent") as graph:
    # Load session context
    load_context = CodeNode(
        name="load_context",
        code_fn=lambda query, session_id: {
            "messages": [
                {"role": "system", "content": f"You are an assistant with memory. Session: {session_id}. You can save and recall information."},
                {"role": "user", "content": query}
            ],
            "session_id": session_id,
            "iteration": 0,
            "done": False,
            "final_answer": None
        },
        inputs={
            "query": PARENT["query"],
            "session_id": PARENT["session_id"]
        }
    )

    # Agent loop with memory tools
    with WhileLoopNode(
        name="agent_loop",
        inputs={
            "messages": load_context["messages"],
            "session_id": load_context["session_id"],
            "iteration": load_context["iteration"],
            "done": load_context["done"],
            "final_answer": load_context["final_answer"]
        },
        stop_condition="done == True or iteration >= 5",
        max_iterations=10
    ) as loop:
        llm = LLMNode(
            name="llm",
            resource_key="llm:agent",
            inputs={
                "messages": PARENT["messages"],
                "tools": MEMORY_TOOLS
            }
        )

        process = CodeNode(
            name="process",
            code_fn=lambda content, tool_calls, messages, session_id, iteration: process_memory_response(
                content, tool_calls, messages, session_id, iteration
            ),
            inputs={
                "content": llm["content"],
                "tool_calls": llm["tool_calls"],
                "messages": PARENT["messages"],
                "session_id": PARENT["session_id"],
                "iteration": PARENT["iteration"]
            }
        )

        process["new_messages"] >> PARENT["messages"]
        process["new_iteration"] >> PARENT["iteration"]
        process["is_done"] >> PARENT["done"]
        process["answer"] >> PARENT["final_answer"]

        START >> llm >> process >> END

    loop["final_answer"] >> PARENT["answer"]

    START >> load_context >> loop >> END


def process_memory_response(content, tool_calls, messages, session_id, iteration):
    """Process response with memory tools."""
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

            if func_name == "save_memory":
                result = memory_store.save(session_id, args["key"], args["value"])
            elif func_name == "recall_memory":
                result = memory_store.recall(session_id, args["key"])
            elif func_name == "list_memories":
                result = memory_store.list_keys(session_id)
            else:
                result = {"error": "Unknown tool"}

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


# Usage
async def main():
    engine = Hush(graph)

    # First interaction - save info
    result = await engine.run(inputs={
        "query": "Remember that my favorite color is blue.",
        "session_id": "user_123"
    })
    print(result["answer"])

    # Second interaction - recall
    result = await engine.run(inputs={
        "query": "What's my favorite color?",
        "session_id": "user_123"
    })
    print(result["answer"])  # Should mention blue

import asyncio
asyncio.run(main())
```

## Error Handling cho Tools

```python
def safe_tool_execution(tool_name: str, args: dict) -> dict:
    """Execute tool with error handling."""
    try:
        if tool_name not in TOOLS:
            return {"error": f"Unknown tool: {tool_name}", "success": False}

        result = TOOLS[tool_name](**args)
        return {"result": result, "success": True}

    except TypeError as e:
        return {"error": f"Invalid arguments: {str(e)}", "success": False}
    except Exception as e:
        return {"error": f"Tool execution failed: {str(e)}", "success": False}
```

## Best Practices

### 1. Limit Iterations

```python
# Luôn set max_iterations để tránh infinite loops
WhileLoopNode(
    ...,
    stop_condition="done == True",
    max_iterations=10  # Safety limit
)
```

### 2. Tool Descriptions Rõ Ràng

```python
# Mô tả tool rõ ràng để LLM hiểu cách sử dụng
{
    "name": "calculator",
    "description": "Evaluate mathematical expressions. Use Python syntax. Examples: '2+2', 'sqrt(16)', '10**2'",
    "parameters": {...}
}
```

### 3. Validate Tool Arguments

```python
def calculator(expression: str) -> dict:
    # Validate input
    if not expression or len(expression) > 1000:
        return {"error": "Invalid expression"}

    # Safe evaluation
    allowed_names = {"sqrt": math.sqrt, "sin": math.sin, ...}
    result = eval(expression, {"__builtins__": {}}, allowed_names)
    return {"result": str(result)}
```

### 4. Tracing

```python
from hush.observability import LangfuseTracer

tracer = LangfuseTracer(
    resource_key="langfuse:default",
    tags=["agent", "tool-calling"]
)

result = await engine.run(
    inputs={"query": "..."},
    tracer=tracer,
    session_id="user_123"
)
```

## Tiếp theo

- [Multi-Model Workflow](multi-model.md) - Sử dụng nhiều models
- [RAG Workflow](rag-workflow.md) - Kết hợp Agent với RAG
- [Xử lý lỗi](../guides/error-handling.md) - Error handling patterns
