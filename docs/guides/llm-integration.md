# Tích hợp LLM

Hướng dẫn cấu hình và sử dụng các LLM providers trong Hush workflows.

## Cấu hình LLM trong resources.yaml

### OpenAI

```yaml
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

### String prompt

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

### Dict với system/user

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
```

### Full messages array (multimodal)

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
        "conversation_history": PARENT["history"],
        "query": PARENT["query"]
    }
)
```

## LLMNode - Gọi LLM

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
        resource_key="gpt-4o",
        inputs={"messages": prompt["messages"]},
        outputs={"content": PARENT["response"]}
    )

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

### Generation Parameters

```python
llm = LLMNode(
    name="llm",
    resource_key="gpt-4o",
    inputs={
        "messages": prompt["messages"],
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 0.9,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.5,
        "stop": ["\n\n", "END"],
        "seed": 42
    }
)
```

## Streaming

LLMNode mặc định stream responses.

```python
llm = LLMNode(
    name="llm",
    resource_key="gpt-4o",
    stream=True,  # Default
    inputs={"messages": prompt["messages"]}
)

# Subscribe to stream
from hush.core.streams import STREAM_SERVICE

async for chunk in STREAM_SERVICE.subscribe(request_id, channel_name):
    print(chunk.choices[0].delta.content, end="")
```

## Load Balancing và Fallback

### Load Balancing

```python
llm = LLMNode(
    name="llm",
    resource_key=["gpt-4o", "gpt-4o-mini"],
    ratios=[0.3, 0.7],  # 30% gpt-4o, 70% gpt-4o-mini
    inputs={"messages": prompt["messages"]}
)
```

### Fallback

```python
llm = LLMNode(
    name="llm",
    resource_key="gpt-4o",
    fallback=["azure-gpt4", "gemini"],
    inputs={"messages": prompt["messages"]}
)
# Nếu gpt-4o fails → try azure-gpt4 → try gemini
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
llm = LLMNode(
    name="llm",
    resource_key="gpt-4o",
    inputs={
        "messages": prompt["messages"],
        "tools": tools,
        "tool_choice": "auto"
    }
)
```

## Structured Output

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

## LLMChainNode - All-in-one

```python
from hush.providers import LLMChainNode

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
```

## Cost Tracking

### Cấu hình

```yaml
llm:gpt-4o:
  _class: OpenAIConfig
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o
  cost_per_input_token: 0.000005
  cost_per_output_token: 0.000015
```

### Truy cập cost

```python
result = await engine.run(inputs={"query": "Hello"}, tracer=tracer)
state = result["$state"]

for node_name, metadata in state.trace_metadata.items():
    if "cost" in metadata:
        print(f"{node_name}: ${metadata['cost']:.6f}")
```

## Best Practices

1. **Separate prompt và LLM nodes** - dễ debug và reuse
2. **Sử dụng ResourceHub** cho production
3. **Cấu hình fallback** cho reliability
4. **Monitor costs** với tracing
5. **Set temperature theo use case**:
   - `0.0`: Factual Q&A, code generation
   - `0.3-0.5`: General conversation
   - `0.7-1.0`: Creative writing

## Xem thêm

- [Tutorial: Sử dụng LLM](../tutorials/02-llm-basics.md)
- [Embeddings & Reranking](embeddings-reranking.md)
- [Xử lý lỗi](error-handling.md)
