# Tích hợp LLM

Hướng dẫn này sẽ giúp bạn cấu hình và sử dụng các LLM providers trong Hush workflows.

## Cấu hình LLM trong resources.yaml

### OpenAI

```yaml
# resources.yaml
llm:gpt-4:
  type: openai
  api_key: ${OPENAI_API_KEY}
  base_url: https://api.openai.com/v1
  model: gpt-4o

llm:gpt-4-mini:
  type: openai
  api_key: ${OPENAI_API_KEY}
  base_url: https://api.openai.com/v1
  model: gpt-4o-mini
```

### Azure OpenAI

```yaml
llm:azure-gpt4:
  type: azure
  api_key: ${AZURE_OPENAI_API_KEY}
  azure_endpoint: https://your-resource.openai.azure.com
  api_version: "2024-02-15-preview"
  model: gpt-4-deployment-name
```

### Google Gemini (Vertex AI)

```yaml
llm:gemini:
  type: gemini
  project_id: your-gcp-project
  private_key_id: ${GEMINI_PRIVATE_KEY_ID}
  private_key: ${GEMINI_PRIVATE_KEY}
  client_email: your-service-account@project.iam.gserviceaccount.com
  client_id: "123456789"
  client_x509_cert_url: https://www.googleapis.com/robot/v1/metadata/x509/...
  location: us-central1
  model: gemini-2.0-flash-001
```

### vLLM / OpenAI-compatible endpoints

```yaml
llm:local-llama:
  type: openai
  api_key: "not-needed"
  base_url: http://localhost:8000/v1
  model: meta-llama/Llama-3.1-8B-Instruct
```

## PromptNode - Xây dựng Messages

`PromptNode` giúp xây dựng messages array từ templates.

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
# Output: [{"role": "user", "content": "Tóm tắt văn bản sau: ..."}]
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
# Output: [
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
from hush.core import GraphNode, START, END, PARENT
from hush.providers import PromptNode, LLMNode

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
        resource_key="gpt-4",
        inputs={"messages": prompt["messages"]}
    )

    llm["content"] >> PARENT["response"]

    START >> prompt >> llm >> END
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
| `thinking_content` | str | Reasoning content (cho reasoning models) |

### Streaming

LLMNode mặc định stream responses. Chunks được push qua `STREAM_SERVICE`.

```python
llm = LLMNode(
    name="llm",
    resource_key="gpt-4",
    stream=True,  # Default
    inputs={"messages": prompt["messages"]}
)

# Subscribe to stream
from hush.core import STREAM_SERVICE

async for chunk in STREAM_SERVICE.subscribe(request_id, channel_name):
    print(chunk.choices[0].delta.content, end="")
```

### Non-streaming

```python
llm = LLMNode(
    name="llm",
    resource_key="gpt-4",
    stream=False,  # Disable streaming
    inputs={"messages": prompt["messages"]}
)
```

## Load Balancing

Phân tải requests giữa nhiều models.

```python
llm = LLMNode(
    name="llm",
    resource_key=["gpt-4", "gpt-4-mini"],  # Multiple models
    ratios=[0.3, 0.7],  # 30% gpt-4, 70% gpt-4-mini
    inputs={"messages": prompt["messages"]}
)
```

## Fallback

Tự động retry với model khác khi primary fails.

```python
llm = LLMNode(
    name="llm",
    resource_key="gpt-4",
    fallback=["azure-gpt4", "gemini"],  # Fallback chain
    inputs={"messages": prompt["messages"]}
)
# Nếu gpt-4 fails → try azure-gpt4 → try gemini
```

## Batch Mode

Sử dụng OpenAI Batch API để tiết kiệm 50% chi phí.

```python
llm = LLMNode(
    name="batch_llm",
    resource_key="gpt-4",
    batch_mode=True,  # Enable batch mode
    inputs={"messages": prompt["messages"]}
)
```

**Lưu ý**: Batch mode xử lý async, kết quả có thể mất đến 24h.

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

### Sử dụng trong LLMNode

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
        resource_key="gpt-4",
        inputs={
            "messages": prompt["messages"],
            "tools": tools,
            "tool_choice": "auto"  # or "required" or {"type": "function", "function": {"name": "get_weather"}}
        }
    )

    # Check if tool was called
    process = CodeNode(
        name="process",
        code_fn=lambda tool_calls, content: {
            "has_tool_call": len(tool_calls) > 0,
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

### Xử lý tool results

```python
# Sau khi execute tools, gửi results về LLM
prompt_with_tools = PromptNode(
    name="prompt_with_tools",
    inputs={
        "prompt": {"system": "Bạn có thể tra cứu thời tiết.", "user": "{query}"},
        "tool_results": PARENT["tool_results"],  # List of tool result messages
        "query": PARENT["query"]
    }
)
```

## Structured Output

Force LLM trả về JSON theo schema.

```python
llm = LLMNode(
    name="llm",
    resource_key="gpt-4",
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

## Generation Parameters

```python
llm = LLMNode(
    name="llm",
    resource_key="gpt-4",
    inputs={
        "messages": prompt["messages"],
        "temperature": 0.7,      # Randomness (0.0 = deterministic)
        "max_tokens": 1000,      # Max response length
        "top_p": 0.9,            # Nucleus sampling
        "frequency_penalty": 0.5, # Reduce repetition
        "presence_penalty": 0.5,  # Encourage new topics
        "stop": ["\n\n", "END"], # Stop sequences
        "seed": 42               # Reproducibility
    }
)
```

## Cost Tracking

### Cấu hình cost per token

```yaml
llm:gpt-4:
  type: openai
  api_key: ${OPENAI_API_KEY}
  base_url: https://api.openai.com/v1
  model: gpt-4o
  cost_per_input_token: 0.000005   # $5 per 1M input tokens
  cost_per_output_token: 0.000015  # $15 per 1M output tokens
```

### Access cost trong kết quả

```python
result = await engine.run(inputs={"query": "Hello"})
state = result["$state"]

# Access cost từ trace metadata
for node_name, metadata in state.trace_metadata.items():
    if "cost" in metadata:
        print(f"{node_name}: ${metadata['cost']['total']:.6f}")
```

### Với Tracer

```python
from hush.observability import LangfuseTracer

tracer = LangfuseTracer(resource_key="langfuse:default")
result = await engine.run(inputs={...}, tracer=tracer)

# Cost được gửi tự động đến Langfuse dashboard
```

## LLMChainNode - All-in-one

`LLMChainNode` kết hợp PromptNode + LLMNode trong một node.

```python
from hush.providers import LLMChainNode

with GraphNode(name="simple-chain") as graph:
    chain = LLMChainNode(
        name="chain",
        resource_key="gpt-4",
        prompt={
            "system": "Bạn là assistant hữu ích.",
            "user": "{query}"
        },
        inputs={"query": PARENT["query"]},
        outputs={"content": PARENT["response"]}
    )

    START >> chain >> END
```

## Best Practices

### 1. Separate prompt và LLM nodes

```python
# Good - dễ debug và reuse
prompt = PromptNode(...)
llm = LLMNode(...)
START >> prompt >> llm >> END

# OK for simple cases
chain = LLMChainNode(...)
```

### 2. Sử dụng ResourceHub cho production

```yaml
# resources.yaml - quản lý tập trung
llm:production:
  type: openai
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o

llm:development:
  type: openai
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o-mini
```

### 3. Cấu hình fallback cho reliability

```python
llm = LLMNode(
    resource_key="gpt-4",
    fallback=["azure-gpt4", "gemini"]  # Backup models
)
```

### 4. Monitor costs với tracing

```python
tracer = LangfuseTracer(resource_key="langfuse:default")
result = await engine.run(inputs={...}, tracer=tracer)
```

### 5. Set temperature theo use case

- `temperature=0.0`: Factual Q&A, code generation
- `temperature=0.3-0.5`: General conversation
- `temperature=0.7-1.0`: Creative writing

## Tiếp theo

- [Embeddings & Reranking](embeddings-reranking.md) - Vector embeddings và reranking
- [Xử lý lỗi](error-handling.md) - Error handling và retry strategies
- [RAG Workflow](../examples/rag-workflow.md) - Ví dụ RAG với LLM
