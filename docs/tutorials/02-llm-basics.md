# Tutorial 2: Sử dụng LLM

Tutorial này hướng dẫn cách sử dụng LLM trong Hush workflows, bao gồm cấu hình providers, xây dựng prompts, và các tính năng nâng cao.

## Cấu hình LLM Provider

### OpenAI

```yaml
# resources.yaml
llm:gpt-4o:
  _class: OpenAIConfig
  api_key: ${OPENAI_API_KEY}
  api_type: openai
  base_url: https://api.openai.com/v1
  model: gpt-4o

llm:gpt-4o-mini:
  _class: OpenAIConfig
  api_key: ${OPENAI_API_KEY}
  api_type: openai
  base_url: https://api.openai.com/v1
  model: gpt-4o-mini
```

### Azure OpenAI

```yaml
llm:azure-gpt4:
  _class: OpenAIConfig
  api_key: ${AZURE_OPENAI_API_KEY}
  api_type: azure
  azure_endpoint: https://your-resource.openai.azure.com
  api_version: "2024-02-15-preview"
  model: gpt-4-deployment-name
```

### Google Gemini

```yaml
llm:gemini:
  _class: GeminiConfig
  project_id: your-gcp-project
  private_key: ${GEMINI_PRIVATE_KEY}
  client_email: your-service-account@project.iam.gserviceaccount.com
  location: us-central1
  model: gemini-2.0-flash-001
```

### vLLM / OpenAI-compatible

```yaml
llm:local-llama:
  _class: OpenAIConfig
  api_key: "not-needed"
  api_type: openai
  base_url: http://localhost:8000/v1
  model: meta-llama/Llama-3.1-8B-Instruct
```

## PromptNode - Xây dựng Messages

`PromptNode` chuyển đổi prompt templates thành messages array cho LLM.

### Cách 1: String prompt (đơn giản nhất)

```python
from hush.providers import PromptNode

prompt = PromptNode(
    name="simple_prompt",
    inputs={
        "prompt": "Tóm tắt văn bản sau: {text}",
        "text": PARENT["text"]
    }
)
# Output messages: [{"role": "user", "content": "Tóm tắt văn bản sau: ..."}]
```

### Cách 2: Dict với system/user

```python
prompt = PromptNode(
    name="chat_prompt",
    inputs={
        "prompt": {
            "system": "Bạn là assistant chuyên {task}.",
            "user": "{query}"
        },
        "task": "tóm tắt văn bản",
        "query": PARENT["query"]
    }
)
# Output messages:
# [
#   {"role": "system", "content": "Bạn là assistant chuyên tóm tắt văn bản."},
#   {"role": "user", "content": "..."}
# ]
```

### Cách 3: Full messages array (multimodal)

```python
prompt = PromptNode(
    name="multimodal_prompt",
    inputs={
        "prompt": [
            {"role": "system", "content": "Bạn là assistant phân tích hình ảnh."},
            {"role": "user", "content": [
                {"type": "text", "text": "Mô tả hình ảnh này: {query}"},
                {"type": "image_url", "image_url": {"url": "{image_url}"}}
            ]}
        ],
        "query": PARENT["query"],
        "image_url": PARENT["image_url"]
    }
)
```

### Với conversation history

```python
prompt = PromptNode(
    name="chat_with_history",
    inputs={
        "prompt": {"system": "Bạn là assistant hữu ích.", "user": "{query}"},
        "conversation_history": PARENT["history"],  # List of previous messages
        "query": PARENT["query"]
    }
)
# History được insert trước user message cuối
```

## LLMNode - Gọi LLM

`LLMNode` sử dụng ResourceHub để gọi LLM đã cấu hình.

### Basic usage

```python
import asyncio
from hush.core import Hush, GraphNode, START, END, PARENT
from hush.providers import PromptNode, LLMNode

async def main():
    with GraphNode(name="chat-workflow") as graph:
        prompt = PromptNode(
            name="prompt",
            inputs={
                "prompt": {"system": "Bạn là assistant hữu ích.", "user": "{query}"},
                "query": PARENT["query"]
            }
        )

        llm = LLMNode(
            name="llm",
            resource_key="gpt-4o",
            inputs={"messages": prompt["messages"]},
            outputs={"content": PARENT["response"]}
        )

        START >> prompt >> llm >> END

    engine = Hush(graph)
    result = await engine.run(inputs={"query": "Hello!"})
    print(result["response"])

asyncio.run(main())
```

### LLMNode outputs

| Output | Type | Mô tả |
|--------|------|-------|
| `role` | str | "assistant" |
| `content` | str | Response content |
| `model_used` | str | Model đã dùng |
| `tokens_used` | dict | `{prompt_tokens, completion_tokens, total_tokens}` |
| `tool_calls` | list | Tool calls nếu có |
| `finish_reason` | str | "stop", "tool_calls", etc. |

### Generation Parameters

```python
llm = LLMNode(
    name="llm",
    resource_key="gpt-4o",
    inputs={
        "messages": prompt["messages"],
        "temperature": 0.7,      # Randomness (0.0 = deterministic)
        "max_tokens": 1000,      # Max response length
        "top_p": 0.9,            # Nucleus sampling
        "frequency_penalty": 0.5, # Reduce repetition
        "presence_penalty": 0.5,  # Encourage new topics
        "stop": ["\n\n", "END"], # Stop sequences
    }
)
```

## LLMChainNode - All-in-one

`LLMChainNode` kết hợp PromptNode + LLMNode trong một node.

```python
from hush.providers import LLMChainNode

with GraphNode(name="simple-chain") as graph:
    chain = LLMChainNode(
        name="chain",
        resource_key="gpt-4o",
        prompt={
            "system": "Bạn là assistant hữu ích.",
            "user": "{query}"
        },
        inputs={"query": PARENT["query"]},
        outputs={"content": PARENT["response"]}
    )

    START >> chain >> END
```

## Structured Output

Force LLM trả về JSON theo schema.

```python
llm = LLMNode(
    name="llm",
    resource_key="gpt-4o",
    inputs={
        "messages": prompt["messages"],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "sentiment_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                        "confidence": {"type": "number"}
                    },
                    "required": ["sentiment", "confidence"]
                }
            }
        }
    }
)
```

## Tool Use / Function Calling

### Định nghĩa tools

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Lấy thông tin thời tiết",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Tên thành phố"
                    }
                },
                "required": ["location"]
            }
        }
    }
]
```

### Sử dụng trong workflow

```python
with GraphNode(name="tool-calling") as graph:
    prompt = PromptNode(
        name="prompt",
        inputs={
            "prompt": {"system": "Bạn có thể tra cứu thời tiết.", "user": "{query}"},
            "query": PARENT["query"]
        }
    )

    llm = LLMNode(
        name="llm",
        resource_key="gpt-4o",
        inputs={
            "messages": prompt["messages"],
            "tools": tools,
            "tool_choice": "auto"
        }
    )

    # Check if tool was called
    process = CodeNode(
        name="process",
        code_fn=lambda tool_calls, content: {
            "has_tool_call": len(tool_calls) > 0 if tool_calls else False,
            "tool_calls": tool_calls,
            "response": content
        },
        inputs={
            "tool_calls": llm["tool_calls"],
            "content": llm["content"]
        },
        outputs={"*": PARENT}
    )

    START >> prompt >> llm >> process >> END
```

## Load Balancing và Fallback

### Load Balancing

```python
llm = LLMNode(
    name="llm",
    resource_key=["gpt-4o", "gpt-4o-mini"],  # Multiple models
    ratios=[0.3, 0.7],  # 30% gpt-4o, 70% gpt-4o-mini
    inputs={"messages": prompt["messages"]}
)
```

### Fallback

```python
llm = LLMNode(
    name="llm",
    resource_key="gpt-4o",
    fallback=["azure-gpt4", "gemini"],  # Fallback chain
    inputs={"messages": prompt["messages"]}
)
# Nếu gpt-4o fails → try azure-gpt4 → try gemini
```

## Cost Tracking

### Cấu hình trong resources.yaml

```yaml
llm:gpt-4o:
  _class: OpenAIConfig
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o
  cost_per_input_token: 0.000005   # $5 per 1M input tokens
  cost_per_output_token: 0.000015  # $15 per 1M output tokens
```

### Truy cập cost

```python
result = await engine.run(inputs={"query": "Hello"}, tracer=tracer)
state = result["$state"]

for node_name, metadata in state.trace_metadata.items():
    if "cost" in metadata:
        print(f"{node_name}: ${metadata['cost']:.6f}")
```

## Ví dụ hoàn chỉnh: Multi-turn Chat

```python
import asyncio
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT
from hush.providers import PromptNode, LLMNode

async def main():
    with GraphNode(name="multi-turn-chat") as graph:
        # Build prompt với history
        prompt = PromptNode(
            name="prompt",
            inputs={
                "prompt": {
                    "system": "Bạn là assistant hữu ích. Trả lời ngắn gọn.",
                    "user": "{message}"
                },
                "conversation_history": PARENT["history"],
                "message": PARENT["message"]
            }
        )

        # Call LLM
        llm = LLMNode(
            name="llm",
            resource_key="gpt-4o",
            inputs={
                "messages": prompt["messages"],
                "temperature": 0.7,
                "max_tokens": 500
            },
            outputs={"content": PARENT["response"]}
        )

        # Update history
        update_history = CodeNode(
            name="update_history",
            code_fn=lambda history, message, response: {
                "new_history": history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": response}
                ]
            },
            inputs={
                "history": PARENT["history"],
                "message": PARENT["message"],
                "response": PARENT["response"]
            },
            outputs={"new_history": PARENT}
        )

        START >> prompt >> llm >> update_history >> END

    engine = Hush(graph)

    # Simulate multi-turn conversation
    history = []
    messages = ["Xin chào!", "Tên tôi là An.", "Tôi tên gì?"]

    for msg in messages:
        result = await engine.run(inputs={"message": msg, "history": history})
        print(f"User: {msg}")
        print(f"Bot: {result['response']}\n")
        history = result["new_history"]

asyncio.run(main())
```

## Tổng kết

| Concept | Mô tả |
|---------|-------|
| `resources.yaml` | Cấu hình LLM providers |
| `PromptNode` | Build messages từ templates |
| `LLMNode` | Gọi LLM với resource_key |
| `LLMChainNode` | All-in-one prompt + LLM |
| `tools` | Function calling support |
| `response_format` | Structured JSON output |
| `fallback` | Automatic retry với model khác |

## Tiếp theo

- [Tutorial 3: Loops và Branches](03-loops-branches.md) - Flow control
- [Guide: Tích hợp LLM](../guides/llm-integration.md) - Chi tiết đầy đủ
