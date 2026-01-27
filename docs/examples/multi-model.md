# Multi-Model Workflow

Ví dụ sử dụng nhiều LLM models trong một workflow.

## Use Cases

1. **Cost optimization** - Dùng model rẻ cho tasks đơn giản, model đắt cho tasks phức tạp
2. **Quality comparison** - So sánh output từ nhiều models
3. **Ensemble** - Kết hợp nhiều models để tăng accuracy
4. **Fallback** - Tự động chuyển sang model khác khi primary fails

## Cấu hình Resources

```yaml
# resources.yaml
llm:gpt-4o:
  _class: OpenAIConfig
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o
  base_url: https://api.openai.com/v1

llm:gpt-4o-mini:
  _class: OpenAIConfig
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o-mini
  base_url: https://api.openai.com/v1

llm:claude:
  _class: AnthropicConfig
  api_key: ${ANTHROPIC_API_KEY}
  model: claude-3-5-sonnet-20241022

llm:gemini:
  _class: GeminiConfig
  project_id: ${GCP_PROJECT}
  model: gemini-2.0-flash-001
```

## Ví dụ 1: Parallel Multi-Model

Gọi nhiều models song song và so sánh kết quả.

```python
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT
from hush.providers import PromptNode, LLMNode

with GraphNode(name="multi-model-parallel") as graph:
    # Build prompt
    prompt = PromptNode(
        name="prompt",
        inputs={
            "prompt": {
                "system": "You are a helpful assistant.",
                "user": "{query}"
            },
            "query": PARENT["query"]
        }
    )

    # Call multiple models in parallel
    gpt = LLMNode(
        name="gpt",
        resource_key="gpt-4o-mini",
        inputs={"messages": prompt["messages"]}
    )

    claude = LLMNode(
        name="claude",
        resource_key="claude",
        inputs={"messages": prompt["messages"]}
    )

    gemini = LLMNode(
        name="gemini",
        resource_key="gemini",
        inputs={"messages": prompt["messages"]}
    )

    # Compare results
    compare = CodeNode(
        name="compare",
        code_fn=lambda gpt_answer, claude_answer, gemini_answer: {
            "gpt": gpt_answer,
            "claude": claude_answer,
            "gemini": gemini_answer,
            "all_agree": gpt_answer == claude_answer == gemini_answer
        },
        inputs={
            "gpt_answer": gpt["content"],
            "claude_answer": claude["content"],
            "gemini_answer": gemini["content"]
        },
        outputs={"*": PARENT}
    )

    # Parallel execution
    START >> prompt >> [gpt, claude, gemini] >> compare >> END
```

## Ví dụ 2: Cost-Optimized Routing

Dùng model rẻ cho tasks đơn giản, model đắt cho tasks phức tạp.

```python
from hush.core import BranchNode

with GraphNode(name="smart-routing") as graph:
    # Classify query complexity
    classifier = LLMNode(
        name="classifier",
        resource_key="gpt-4o-mini",  # Cheap model for classification
        inputs={
            "messages": [
                {"role": "system", "content": "Classify if this query is SIMPLE or COMPLEX. Reply with just one word."},
                {"role": "user", "content": PARENT["query"]}
            ]
        }
    )

    # Route based on complexity
    router = BranchNode(
        name="router",
        cases={"'SIMPLE' in classification.upper()": "simple_path"},
        default="complex_path",
        inputs={"classification": classifier["content"]}
    )

    # Simple path - cheap model
    simple_prompt = PromptNode(
        name="simple_prompt",
        inputs={
            "prompt": {"system": "Be concise.", "user": "{query}"},
            "query": PARENT["query"]
        }
    )
    simple_llm = LLMNode(
        name="simple_llm",
        resource_key="gpt-4o-mini",
        inputs={"messages": simple_prompt["messages"]},
        outputs={"content": PARENT["answer"]}
    )

    # Complex path - powerful model
    complex_prompt = PromptNode(
        name="complex_prompt",
        inputs={
            "prompt": {"system": "Think step by step.", "user": "{query}"},
            "query": PARENT["query"]
        }
    )
    complex_llm = LLMNode(
        name="complex_llm",
        resource_key="gpt-4o",
        inputs={"messages": complex_prompt["messages"]},
        outputs={"content": PARENT["answer"]}
    )

    START >> classifier >> router
    router >> simple_prompt >> simple_llm
    router >> complex_prompt >> complex_llm
    [simple_llm, complex_llm] >> ~END
```

## Ví dụ 3: Ensemble với Voting

Kết hợp nhiều models và chọn câu trả lời tốt nhất.

```python
with GraphNode(name="ensemble") as graph:
    prompt = PromptNode(
        name="prompt",
        inputs={
            "prompt": {
                "system": "Answer the question accurately.",
                "user": "{query}"
            },
            "query": PARENT["query"]
        }
    )

    # Generate from multiple models
    model_1 = LLMNode(name="model_1", resource_key="gpt-4o-mini", inputs={"messages": prompt["messages"]})
    model_2 = LLMNode(name="model_2", resource_key="claude", inputs={"messages": prompt["messages"]})
    model_3 = LLMNode(name="model_3", resource_key="gemini", inputs={"messages": prompt["messages"]})

    # Use a judge model to select best answer
    judge_prompt = PromptNode(
        name="judge_prompt",
        inputs={
            "prompt": {
                "system": """Given a question and multiple answers, select the best one.
Reply with just the number (1, 2, or 3) of the best answer.""",
                "user": """Question: {query}

Answer 1: {answer_1}
Answer 2: {answer_2}
Answer 3: {answer_3}"""
            },
            "query": PARENT["query"],
            "answer_1": model_1["content"],
            "answer_2": model_2["content"],
            "answer_3": model_3["content"]
        }
    )

    judge = LLMNode(
        name="judge",
        resource_key="gpt-4o",
        inputs={"messages": judge_prompt["messages"]}
    )

    # Select best answer
    select = CodeNode(
        name="select",
        code_fn=lambda choice, a1, a2, a3: {
            "answer": a1 if "1" in choice else (a2 if "2" in choice else a3)
        },
        inputs={
            "choice": judge["content"],
            "a1": model_1["content"],
            "a2": model_2["content"],
            "a3": model_3["content"]
        },
        outputs={"answer": PARENT}
    )

    START >> prompt >> [model_1, model_2, model_3] >> judge_prompt >> judge >> select >> END
```

## Ví dụ 4: Automatic Fallback

Tự động chuyển sang model khác khi primary fails.

```python
llm = LLMNode(
    name="llm",
    resource_key="gpt-4o",
    fallback=["claude", "gemini"],  # Fallback chain
    inputs={"messages": prompt["messages"]}
)
# Nếu gpt-4o fails → try claude → try gemini
```

## Ví dụ 5: Load Balancing

Phân tải requests giữa nhiều models.

```python
llm = LLMNode(
    name="llm",
    resource_key=["gpt-4o", "gpt-4o-mini"],
    ratios=[0.3, 0.7],  # 30% gpt-4o, 70% gpt-4o-mini
    inputs={"messages": prompt["messages"]}
)
```

## Cost Tracking

```yaml
llm:gpt-4o:
  _class: OpenAIConfig
  model: gpt-4o
  cost_per_input_token: 0.000005
  cost_per_output_token: 0.000015

llm:gpt-4o-mini:
  _class: OpenAIConfig
  model: gpt-4o-mini
  cost_per_input_token: 0.00000015
  cost_per_output_token: 0.0000006
```

```python
# Track costs với tracing
from hush.observability import LangfuseTracer

tracer = LangfuseTracer(resource_key="langfuse:default")
result = await engine.run(inputs={...}, tracer=tracer)
```

## Best Practices

1. **Cost-aware routing** - Route simple queries to cheaper models
2. **Fallback chains** - Always have fallback for reliability
3. **Parallel calls** - Call multiple models in parallel when comparing
4. **Track costs** - Monitor costs per model với tracing

## Xem thêm

- [RAG Workflow](rag-workflow.md)
- [Agent Workflow](agent-workflow.md)
- [Tích hợp LLM](../guides/llm-integration.md)
