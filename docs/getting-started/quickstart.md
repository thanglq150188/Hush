# Quickstart - 5 phút đầu tiên

Hướng dẫn này giúp bạn chạy workflow đầu tiên với Hush trong 5 phút.

## 1. Cài đặt

```bash
pip install hush-ai[standard]
```

## 2. Hello World - Workflow đơn giản

Tạo file `hello.py`:

```python
import asyncio
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT

async def main():
    # Bước 1: Định nghĩa workflow bên trong GraphNode context
    with GraphNode(name="hello-world") as graph:
        # Bước 2: Tạo node xử lý
        greet = CodeNode(
            name="greet",
            code_fn=lambda name: {"greeting": f"Xin chào, {name}!"},
            inputs={"name": PARENT["name"]},  # Lấy input từ parent
            outputs={"greeting": PARENT}       # Ghi output lên parent
        )
        # Bước 3: Kết nối nodes
        START >> greet >> END

    # Bước 4: Chạy workflow
    engine = Hush(graph)
    result = await engine.run(inputs={"name": "Hush"})

    print(result["greeting"])  # Output: Xin chào, Hush!

asyncio.run(main())
```

Chạy:
```bash
python hello.py
# Output: Xin chào, Hush!
```

## 3. Sử dụng ResourceHub với LLM

### Bước 1: Tạo file `resources.yaml`

```yaml
llm:gpt-4o:
  _class: OpenAIConfig
  api_key: ${OPENAI_API_KEY}
  api_type: openai
  base_url: https://api.openai.com/v1
  model: gpt-4o
```

### Bước 2: Đặt biến môi trường

```bash
# Đặt API key
export OPENAI_API_KEY=sk-your-api-key

# Đặt đường dẫn config (optional nếu resources.yaml ở thư mục hiện tại)
export HUSH_CONFIG=/path/to/resources.yaml
```

### Bước 3: Tạo workflow với LLM

Tạo file `chat.py`:

```python
import asyncio
from hush.core import Hush, GraphNode, START, END, PARENT
from hush.providers import PromptNode, LLMNode

async def main():
    with GraphNode(name="chat-workflow") as graph:
        # Node 1: Tạo messages từ prompt template
        prompt = PromptNode(
            name="prompt",
            inputs={
                "prompt": {
                    "system": "Bạn là trợ lý AI thân thiện.",
                    "user": "{question}"
                },
                "question": PARENT["question"]
            },
            outputs={"messages": PARENT}
        )

        # Node 2: Gọi LLM
        llm = LLMNode(
            name="llm",
            resource_key="gpt-4o",  # Tham chiếu đến llm:gpt-4o trong resources.yaml
            inputs={"messages": PARENT["messages"]},
            outputs={"content": PARENT["answer"]}
        )

        START >> prompt >> llm >> END

    engine = Hush(graph)
    result = await engine.run(inputs={"question": "Python là gì?"})

    print(f"Trả lời: {result['answer']}")

asyncio.run(main())
```

Chạy:
```bash
python chat.py
```

## 4. Multi-step Pipeline

Ví dụ pipeline với nhiều bước xử lý:

```python
import asyncio
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT

async def main():
    with GraphNode(name="data-pipeline") as graph:
        # Bước 1: Fetch data
        fetch = CodeNode(
            name="fetch",
            code_fn=lambda: {"data": [1, 2, 3, 4, 5]},
            outputs={"data": PARENT}
        )

        # Bước 2: Transform - nhân đôi mỗi phần tử
        transform = CodeNode(
            name="transform",
            code_fn=lambda data: {"transformed": [x * 2 for x in data]},
            inputs={"data": PARENT["data"]},
            outputs={"transformed": PARENT}
        )

        # Bước 3: Aggregate - tính tổng
        aggregate = CodeNode(
            name="aggregate",
            code_fn=lambda data: {"total": sum(data)},
            inputs={"data": PARENT["transformed"]},
            outputs={"total": PARENT}
        )

        START >> fetch >> transform >> aggregate >> END

    engine = Hush(graph)
    result = await engine.run(inputs={})

    print(f"Data gốc: {result['data']}")           # [1, 2, 3, 4, 5]
    print(f"Sau transform: {result['transformed']}")  # [2, 4, 6, 8, 10]
    print(f"Tổng: {result['total']}")              # 30

asyncio.run(main())
```

## Khái niệm chính

| Khái niệm | Mô tả |
|-----------|-------|
| `GraphNode` | Container chứa workflow, nodes phải được tạo bên trong context |
| `CodeNode` | Node chạy Python function |
| `PromptNode` | Node tạo messages cho LLM (từ hush-providers) |
| `LLMNode` | Node gọi LLM qua ResourceHub (từ hush-providers) |
| `START >> node >> END` | Kết nối nodes thành pipeline |
| `PARENT["key"]` | Lấy data từ state của parent graph |
| `inputs` | Mapping input variables cho node |
| `outputs` | Mapping output variables từ node |

## Tiếp theo

- [Workflow đầu tiên](first-workflow.md) - Tutorial chi tiết từng bước
- [Khái niệm cốt lõi](../concepts/overview.md) - Hiểu sâu về kiến trúc Hush
- [Tích hợp LLM](../guides/llm-integration.md) - Hướng dẫn đầy đủ về LLM
