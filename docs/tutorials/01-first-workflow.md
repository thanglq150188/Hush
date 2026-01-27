# Tutorial 1: Workflow đầu tiên

Tutorial này hướng dẫn từng bước xây dựng một workflow thực tế: **Pipeline tóm tắt văn bản với LLM**.

## Bài toán

Xây dựng pipeline:
1. Nhận văn bản đầu vào
2. Tiền xử lý (clean text)
3. Gọi LLM để tóm tắt
4. Trả về kết quả

## Chuẩn bị

### 1. Cài đặt

```bash
pip install hush-ai[standard]
```

### 2. Tạo file `resources.yaml`

```yaml
# resources.yaml
llm:gpt-4o:
  _class: OpenAIConfig
  api_key: ${OPENAI_API_KEY}
  api_type: openai
  base_url: https://api.openai.com/v1
  model: gpt-4o
```

### 3. Đặt biến môi trường

```bash
export OPENAI_API_KEY=sk-your-api-key
export HUSH_CONFIG=/path/to/resources.yaml
```

## Bước 1: Tạo GraphNode

`GraphNode` là container chứa toàn bộ workflow. Tất cả nodes phải được tạo **bên trong** context của GraphNode.

```python
from hush.core import GraphNode

# Tạo workflow container
with GraphNode(name="summarize-pipeline") as graph:
    # Nodes sẽ được định nghĩa ở đây
    pass
```

**Lưu ý quan trọng**: Nodes được tạo bên ngoài `with GraphNode` sẽ không hoạt động.

## Bước 2: Thêm CodeNode để tiền xử lý

`CodeNode` chạy một Python function. Function nhận inputs và trả về dict outputs.

```python
from hush.core import GraphNode, CodeNode, PARENT

def clean_text(text: str) -> dict:
    """Tiền xử lý văn bản: loại bỏ whitespace thừa."""
    cleaned = " ".join(text.split())
    return {"cleaned_text": cleaned}

with GraphNode(name="summarize-pipeline") as graph:
    # Node tiền xử lý
    preprocess = CodeNode(
        name="preprocess",
        code_fn=clean_text,
        inputs={"text": PARENT["text"]},      # Lấy 'text' từ input
        outputs={"cleaned_text": PARENT}       # Ghi 'cleaned_text' lên parent state
    )
```

### Giải thích inputs/outputs

- `inputs={"text": PARENT["text"]}`: Lấy giá trị `text` từ state của parent (GraphNode) và truyền vào parameter `text` của function
- `outputs={"cleaned_text": PARENT}`: Ghi giá trị `cleaned_text` từ kết quả function lên state của parent

## Bước 3: Thêm PromptNode và LLMNode

`PromptNode` tạo messages cho LLM. `LLMNode` gọi LLM từ ResourceHub.

```python
from hush.providers import PromptNode, LLMNode

with GraphNode(name="summarize-pipeline") as graph:
    preprocess = CodeNode(
        name="preprocess",
        code_fn=clean_text,
        inputs={"text": PARENT["text"]},
        outputs={"cleaned_text": PARENT}
    )

    # Node tạo prompt
    prompt = PromptNode(
        name="prompt",
        inputs={
            "prompt": {
                "system": "Bạn là chuyên gia tóm tắt văn bản. Tóm tắt ngắn gọn, súc tích.",
                "user": "Tóm tắt văn bản sau:\n\n{text}"
            },
            "text": PARENT["cleaned_text"]  # Lấy từ output của preprocess
        },
        outputs={"messages": PARENT}
    )

    # Node gọi LLM
    summarize = LLMNode(
        name="summarize",
        resource_key="gpt-4o",  # Tham chiếu đến llm:gpt-4o trong resources.yaml
        inputs={"messages": PARENT["messages"]},
        outputs={"content": PARENT["summary"]}  # Ghi output vào key 'summary'
    )
```

## Bước 4: Kết nối nodes với >> operator

Sử dụng operator `>>` để định nghĩa thứ tự thực thi. `START` và `END` là markers đặc biệt.

```python
from hush.core import START, END

with GraphNode(name="summarize-pipeline") as graph:
    preprocess = CodeNode(...)
    prompt = PromptNode(...)
    summarize = LLMNode(...)

    # Kết nối: START → preprocess → prompt → summarize → END
    START >> preprocess >> prompt >> summarize >> END
```

## Bước 5: Chạy với Hush engine

```python
import asyncio
from hush.core import Hush

async def main():
    with GraphNode(name="summarize-pipeline") as graph:
        # ... định nghĩa nodes ...
        START >> preprocess >> prompt >> summarize >> END

    # Tạo engine từ graph
    engine = Hush(graph)

    # Chạy workflow với input
    result = await engine.run(inputs={
        "text": """
        Trí tuệ nhân tạo (AI) đang thay đổi cách chúng ta sống và làm việc.
        Từ xe tự lái đến trợ lý ảo, AI đã trở thành một phần không thể thiếu
        trong cuộc sống hàng ngày. Các công ty công nghệ lớn đang đầu tư
        hàng tỷ đô la vào nghiên cứu AI, với hy vọng tạo ra những đột phá
        mới trong lĩnh vực này.
        """
    })

    print("Tóm tắt:", result["summary"])

asyncio.run(main())
```

## Bước 6: Debug với LocalTracer

`LocalTracer` giúp theo dõi quá trình thực thi workflow.

```python
from hush.core.tracers import LocalTracer

async def main():
    with GraphNode(name="summarize-pipeline") as graph:
        # ... định nghĩa nodes ...
        START >> preprocess >> prompt >> summarize >> END

    # Tạo tracer
    tracer = LocalTracer(name="summarize-debug", tags=["tutorial", "summarize"])

    engine = Hush(graph)
    result = await engine.run(
        inputs={"text": "..."},
        tracer=tracer,  # Truyền tracer vào run()
        user_id="user-123",
        session_id="session-456"
    )

    print("Tóm tắt:", result["summary"])

    # Truy cập state để debug
    state = result["$state"]
    print(f"User ID: {state.user_id}")
    print(f"Execution order: {state.execution_order}")
```

## Code hoàn chỉnh

```python
import asyncio
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT
from hush.core.tracers import LocalTracer
from hush.providers import PromptNode, LLMNode


def clean_text(text: str) -> dict:
    """Tiền xử lý văn bản."""
    cleaned = " ".join(text.split())
    return {"cleaned_text": cleaned}


async def main():
    # Định nghĩa workflow
    with GraphNode(name="summarize-pipeline") as graph:
        # Node 1: Tiền xử lý
        preprocess = CodeNode(
            name="preprocess",
            code_fn=clean_text,
            inputs={"text": PARENT["text"]},
            outputs={"cleaned_text": PARENT}
        )

        # Node 2: Tạo prompt
        prompt = PromptNode(
            name="prompt",
            inputs={
                "prompt": {
                    "system": "Bạn là chuyên gia tóm tắt văn bản. Tóm tắt ngắn gọn, súc tích.",
                    "user": "Tóm tắt văn bản sau:\n\n{text}"
                },
                "text": PARENT["cleaned_text"]
            },
            outputs={"messages": PARENT}
        )

        # Node 3: Gọi LLM
        summarize = LLMNode(
            name="summarize",
            resource_key="gpt-4o",
            inputs={"messages": PARENT["messages"]},
            outputs={"content": PARENT["summary"]}
        )

        # Kết nối nodes
        START >> preprocess >> prompt >> summarize >> END

    # Tạo tracer (optional)
    tracer = LocalTracer(name="summarize-debug", tags=["tutorial"])

    # Chạy workflow
    engine = Hush(graph)
    result = await engine.run(
        inputs={
            "text": """
            Trí tuệ nhân tạo (AI) đang thay đổi cách chúng ta sống và làm việc.
            Từ xe tự lái đến trợ lý ảo, AI đã trở thành một phần không thể thiếu
            trong cuộc sống hàng ngày. Các công ty công nghệ lớn đang đầu tư
            hàng tỷ đô la vào nghiên cứu AI, với hy vọng tạo ra những đột phá
            mới trong lĩnh vực này.
            """
        },
        tracer=tracer
    )

    # In kết quả
    print("=" * 50)
    print("KẾT QUẢ TÓM TẮT")
    print("=" * 50)
    print(result["summary"])


if __name__ == "__main__":
    asyncio.run(main())
```

## Tổng kết

Trong tutorial này, bạn đã học:

| Concept | Mô tả |
|---------|-------|
| `GraphNode` | Container chứa workflow |
| `CodeNode` | Chạy Python function |
| `PromptNode` | Tạo messages cho LLM |
| `LLMNode` | Gọi LLM qua ResourceHub |
| `PARENT["key"]` | Truy cập data từ parent state |
| `inputs` / `outputs` | Mapping data vào/ra nodes |
| `START >> node >> END` | Kết nối nodes |
| `LocalTracer` | Debug và trace workflow |
| `resources.yaml` | Cấu hình resources (LLM, etc.) |

## Tiếp theo

- [Tutorial 2: Sử dụng LLM](02-llm-basics.md) - Chi tiết về LLM integration
- [Tutorial 3: Loops và Branches](03-loops-branches.md) - Flow control
