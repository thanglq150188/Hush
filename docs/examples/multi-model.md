# Multi-Model Workflow

Ví dụ này hướng dẫn sử dụng nhiều LLM models trong cùng một workflow để tối ưu cost và quality.

## Tổng quan Use Cases

| Pattern | Mô tả | Khi nào dùng |
|---------|-------|--------------|
| **Cascading** | Model rẻ → Model mạnh | Lọc trước khi xử lý nặng |
| **Routing** | Classify → Route to specialist | Nhiều loại tasks khác nhau |
| **Ensemble** | Multiple models → Aggregate | Cần độ tin cậy cao |
| **Fallback** | Primary → Backup | High availability |

## Cấu hình Resources

### resources.yaml

```yaml
# Fast/cheap model
llm:fast:
  _class: OpenAIConfig
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o-mini
  base_url: https://api.openai.com/v1

# Strong model
llm:strong:
  _class: OpenAIConfig
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o
  base_url: https://api.openai.com/v1

# Specialized for code
llm:code:
  _class: OpenAIConfig
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o
  base_url: https://api.openai.com/v1

# Backup (different provider)
llm:backup:
  _class: AzureConfig
  api_key: ${AZURE_API_KEY}
  azure_endpoint: ${AZURE_ENDPOINT}
  model: gpt-4

# Alternative provider
llm:gemini:
  _class: GeminiConfig
  project_id: ${GCP_PROJECT}
  model: gemini-2.0-flash-001
```

## Ví dụ 1: Cascading Pattern

Sử dụng model rẻ để filter, model mạnh để generate.

```python
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT
from hush.core.nodes.flow.branch_node import Branch
from hush.providers import PromptNode, LLMNode

with GraphNode(name="cascading-pipeline") as graph:
    # Step 1: Fast model để classify
    classify_prompt = PromptNode(
        name="classify_prompt",
        inputs={
            "prompt": {
                "system": """Classify the query complexity:
- SIMPLE: Basic questions, greetings, factual lookups
- COMPLEX: Analysis, reasoning, creative tasks

Respond with only: SIMPLE or COMPLEX""",
                "user": "{query}"
            },
            "query": PARENT["query"]
        }
    )

    classify = LLMNode(
        name="classify",
        resource_key="llm:fast",  # Cheap model
        inputs={"messages": classify_prompt["messages"]}
    )

    # Step 2: Route based on complexity
    router = (Branch("complexity_router")
        .if_(classify["content"].contains("COMPLEX"), "complex_handler")
        .otherwise("simple_handler"))

    # Simple handler - fast model
    simple_prompt = PromptNode(
        name="simple_prompt",
        inputs={
            "prompt": {"system": "Trả lời ngắn gọn.", "user": "{query}"},
            "query": PARENT["query"]
        }
    )
    simple_llm = LLMNode(
        name="simple_llm",
        resource_key="llm:fast",
        inputs={"messages": simple_prompt["messages"]}
    )

    # Complex handler - strong model
    complex_prompt = PromptNode(
        name="complex_prompt",
        inputs={
            "prompt": {
                "system": "Bạn là expert assistant. Phân tích kỹ và trả lời chi tiết.",
                "user": "{query}"
            },
            "query": PARENT["query"]
        }
    )
    complex_llm = LLMNode(
        name="complex_llm",
        resource_key="llm:strong",  # Expensive model
        inputs={"messages": complex_prompt["messages"]}
    )

    # Merge results
    merge = CodeNode(
        name="merge",
        code_fn=lambda simple=None, complex=None, classification=None: {
            "response": complex if complex else simple,
            "model_used": "strong" if complex else "fast",
            "classification": classification
        },
        inputs={
            "simple": simple_llm["content"],
            "complex": complex_llm["content"],
            "classification": classify["content"]
        },
        outputs={"*": PARENT}
    )

    # Flow
    START >> classify_prompt >> classify >> router
    router >> [simple_prompt, complex_prompt]
    simple_prompt >> simple_llm
    complex_prompt >> complex_llm
    [simple_llm, complex_llm] >> ~merge  # Soft edges
    merge >> END


# Usage
async def main():
    engine = Hush(graph)

    # Simple query - uses fast model
    result = await engine.run(inputs={"query": "What is 2+2?"})
    print(f"Response: {result['response']}")
    print(f"Model: {result['model_used']}")  # "fast"

    # Complex query - uses strong model
    result = await engine.run(inputs={
        "query": "Explain quantum entanglement and its implications for quantum computing."
    })
    print(f"Response: {result['response']}")
    print(f"Model: {result['model_used']}")  # "strong"

import asyncio
asyncio.run(main())
```

## Ví dụ 2: Task Routing

Route đến specialized models dựa trên task type.

```python
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT
from hush.core.nodes.flow.branch_node import Branch
from hush.providers import PromptNode, LLMNode

with GraphNode(name="task-routing") as graph:
    # Classify task type
    classify_prompt = PromptNode(
        name="classify_prompt",
        inputs={
            "prompt": {
                "system": """Classify the task type:
- CODE: Programming, debugging, code explanation
- MATH: Calculations, equations, statistics
- CREATIVE: Writing, storytelling, brainstorming
- GENERAL: Everything else

Respond with only the category.""",
                "user": "{query}"
            },
            "query": PARENT["query"]
        }
    )

    classify = LLMNode(
        name="classify",
        resource_key="llm:fast",
        inputs={"messages": classify_prompt["messages"]}
    )

    # Router
    router = (Branch("task_router")
        .if_(classify["content"].contains("CODE"), "code_handler")
        .if_(classify["content"].contains("MATH"), "math_handler")
        .if_(classify["content"].contains("CREATIVE"), "creative_handler")
        .otherwise("general_handler"))

    # Specialized handlers
    # Code specialist
    code_prompt = PromptNode(
        name="code_prompt",
        inputs={
            "prompt": {
                "system": "You are an expert programmer. Provide clean, well-documented code.",
                "user": "{query}"
            },
            "query": PARENT["query"]
        }
    )
    code_llm = LLMNode(
        name="code_llm",
        resource_key="llm:code",
        inputs={"messages": code_prompt["messages"]}
    )

    # Math specialist
    math_prompt = PromptNode(
        name="math_prompt",
        inputs={
            "prompt": {
                "system": "You are a math expert. Show step-by-step solutions.",
                "user": "{query}"
            },
            "query": PARENT["query"]
        }
    )
    math_llm = LLMNode(
        name="math_llm",
        resource_key="llm:strong",
        inputs={"messages": math_prompt["messages"]}
    )

    # Creative specialist
    creative_prompt = PromptNode(
        name="creative_prompt",
        inputs={
            "prompt": {
                "system": "You are a creative writer. Be imaginative and engaging.",
                "user": "{query}"
            },
            "query": PARENT["query"]
        }
    )
    creative_llm = LLMNode(
        name="creative_llm",
        resource_key="llm:gemini",  # Gemini for creative
        inputs={"messages": creative_prompt["messages"]}
    )

    # General handler
    general_prompt = PromptNode(
        name="general_prompt",
        inputs={
            "prompt": {"system": "You are a helpful assistant.", "user": "{query}"},
            "query": PARENT["query"]
        }
    )
    general_llm = LLMNode(
        name="general_llm",
        resource_key="llm:fast",
        inputs={"messages": general_prompt["messages"]}
    )

    # Merge
    merge = CodeNode(
        name="merge",
        code_fn=lambda code=None, math=None, creative=None, general=None, task_type=None: {
            "response": code or math or creative or general,
            "task_type": task_type.strip() if task_type else "GENERAL"
        },
        inputs={
            "code": code_llm["content"],
            "math": math_llm["content"],
            "creative": creative_llm["content"],
            "general": general_llm["content"],
            "task_type": classify["content"]
        },
        outputs={"*": PARENT}
    )

    # Flow
    START >> classify_prompt >> classify >> router
    router >> [code_prompt, math_prompt, creative_prompt, general_prompt]
    code_prompt >> code_llm
    math_prompt >> math_llm
    creative_prompt >> creative_llm
    general_prompt >> general_llm
    [code_llm, math_llm, creative_llm, general_llm] >> ~merge
    merge >> END
```

## Ví dụ 3: Ensemble Pattern

Gọi nhiều models song song và aggregate kết quả.

```python
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT
from hush.providers import PromptNode, LLMNode

with GraphNode(name="ensemble-pipeline") as graph:
    # Same prompt for all models
    prompt = PromptNode(
        name="prompt",
        inputs={
            "prompt": {
                "system": "Trả lời câu hỏi một cách chính xác.",
                "user": "{query}"
            },
            "query": PARENT["query"]
        }
    )

    # Call multiple models in parallel
    llm_openai = LLMNode(
        name="llm_openai",
        resource_key="llm:strong",
        inputs={"messages": prompt["messages"]}
    )

    llm_gemini = LLMNode(
        name="llm_gemini",
        resource_key="llm:gemini",
        inputs={"messages": prompt["messages"]}
    )

    llm_azure = LLMNode(
        name="llm_azure",
        resource_key="llm:backup",
        inputs={"messages": prompt["messages"]}
    )

    # Aggregate responses
    aggregate = CodeNode(
        name="aggregate",
        code_fn=lambda r1, r2, r3, query: aggregate_responses(r1, r2, r3, query),
        inputs={
            "r1": llm_openai["content"],
            "r2": llm_gemini["content"],
            "r3": llm_azure["content"],
            "query": PARENT["query"]
        },
        outputs={"*": PARENT}
    )

    # Parallel execution
    START >> prompt >> [llm_openai, llm_gemini, llm_azure] >> aggregate >> END


def aggregate_responses(r1, r2, r3, query):
    """Aggregate multiple model responses."""
    responses = [r for r in [r1, r2, r3] if r]

    if len(responses) == 0:
        return {"response": "No response available", "confidence": 0}

    # Simple aggregation: check agreement
    # In production, use more sophisticated methods
    if len(set(responses)) == 1:
        # All agree
        return {
            "response": responses[0],
            "confidence": 1.0,
            "agreement": "full"
        }
    else:
        # Return longest/most detailed response
        best = max(responses, key=len)
        return {
            "response": best,
            "confidence": 0.7,
            "agreement": "partial",
            "all_responses": responses
        }


# Alternative: Majority voting for classification
def majority_vote(responses):
    """Simple majority voting."""
    from collections import Counter
    votes = Counter(r.strip().upper() for r in responses if r)
    if votes:
        winner, count = votes.most_common(1)[0]
        return {
            "result": winner,
            "confidence": count / len(responses),
            "votes": dict(votes)
        }
    return {"result": None, "confidence": 0}
```

## Ví dụ 4: Fallback Chain

Primary model với fallback khi fail.

```python
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT
from hush.providers import PromptNode, LLMNode

with GraphNode(name="fallback-pipeline") as graph:
    prompt = PromptNode(
        name="prompt",
        inputs={
            "prompt": {"system": "You are a helpful assistant.", "user": "{query}"},
            "query": PARENT["query"]
        }
    )

    # LLM with built-in fallback
    llm = LLMNode(
        name="llm",
        resource_key="llm:strong",
        fallback=["llm:backup", "llm:gemini", "llm:fast"],  # Fallback chain
        inputs={"messages": prompt["messages"]}
    )

    llm["content"] >> PARENT["response"]
    llm["model_used"] >> PARENT["model_used"]

    START >> prompt >> llm >> END


# Usage
async def main():
    engine = Hush(graph)
    result = await engine.run(inputs={"query": "Hello"})

    print(f"Response: {result['response']}")
    print(f"Model used: {result['model_used']}")
    # If primary fails, shows which fallback was used

import asyncio
asyncio.run(main())
```

## Ví dụ 5: Quality Gate

Kiểm tra chất lượng output và retry với model mạnh hơn nếu cần.

```python
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT
from hush.core.nodes.flow.branch_node import Branch
from hush.providers import PromptNode, LLMNode

with GraphNode(name="quality-gate") as graph:
    # Initial prompt
    prompt = PromptNode(
        name="prompt",
        inputs={
            "prompt": {"system": "Answer the question.", "user": "{query}"},
            "query": PARENT["query"]
        }
    )

    # First attempt with fast model
    fast_llm = LLMNode(
        name="fast_llm",
        resource_key="llm:fast",
        inputs={"messages": prompt["messages"]}
    )

    # Quality check
    quality_prompt = PromptNode(
        name="quality_prompt",
        inputs={
            "prompt": {
                "system": """Rate the quality of this answer on a scale of 1-10.
Consider: accuracy, completeness, clarity.
Respond with only a number.""",
                "user": "Question: {query}\n\nAnswer: {answer}"
            },
            "query": PARENT["query"],
            "answer": fast_llm["content"]
        }
    )

    quality_check = LLMNode(
        name="quality_check",
        resource_key="llm:fast",
        inputs={"messages": quality_prompt["messages"]}
    )

    # Parse quality score
    parse_score = CodeNode(
        name="parse_score",
        code_fn=lambda score_str: {
            "score": int(''.join(filter(str.isdigit, score_str)) or 5)
        },
        inputs={"score_str": quality_check["content"]}
    )

    # Route based on quality
    quality_router = (Branch("quality_router")
        .if_(parse_score["score"] < 7, "retry_strong")
        .otherwise("use_fast"))

    # Use fast result
    use_fast = CodeNode(
        name="use_fast",
        code_fn=lambda answer, score: {
            "response": answer,
            "quality_score": score,
            "model_used": "fast"
        },
        inputs={
            "answer": fast_llm["content"],
            "score": parse_score["score"]
        }
    )

    # Retry with strong model
    retry_prompt = PromptNode(
        name="retry_prompt",
        inputs={
            "prompt": {
                "system": "Provide a detailed, high-quality answer.",
                "user": "{query}"
            },
            "query": PARENT["query"]
        }
    )

    strong_llm = LLMNode(
        name="strong_llm",
        resource_key="llm:strong",
        inputs={"messages": retry_prompt["messages"]}
    )

    retry_result = CodeNode(
        name="retry_result",
        code_fn=lambda answer, original_score: {
            "response": answer,
            "quality_score": original_score,
            "model_used": "strong (retry)"
        },
        inputs={
            "answer": strong_llm["content"],
            "original_score": parse_score["score"]
        }
    )

    # Merge
    merge = CodeNode(
        name="merge",
        code_fn=lambda fast=None, strong=None: fast or strong,
        inputs={
            "fast": use_fast["response"],
            "strong": retry_result["response"]
        },
        outputs={"*": PARENT}
    )

    # Flow
    START >> prompt >> fast_llm >> quality_prompt >> quality_check >> parse_score
    parse_score >> quality_router
    quality_router >> [use_fast, retry_prompt]
    retry_prompt >> strong_llm >> retry_result
    [use_fast, retry_result] >> ~merge
    merge >> END
```

## Cost Comparison

### Model Pricing (Approximate)

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| gpt-4o-mini | $0.15 | $0.60 |
| gpt-4o | $2.50 | $10.00 |
| gemini-2.0-flash | $0.075 | $0.30 |

### Cascading Cost Savings

```python
# Without cascading: All queries to gpt-4o
# 1000 queries × 1000 tokens avg = 1M tokens
# Cost: $2.50 + $10.00 = $12.50

# With cascading (80% simple, 20% complex):
# 800 simple × gpt-4o-mini: 0.8M tokens × ($0.15 + $0.60) = $0.60
# 200 complex × gpt-4o: 0.2M tokens × ($2.50 + $10.00) = $2.50
# Total: $3.10 (75% savings!)
```

## Best Practices

### 1. Choose Models Wisely

```yaml
# Fast/cheap for: classification, simple Q&A, filtering
llm:fast:
  model: gpt-4o-mini

# Strong for: complex reasoning, code, analysis
llm:strong:
  model: gpt-4o

# Specialized: consider fine-tuned models for specific tasks
```

### 2. Monitor Usage

```python
from hush.observability import LangfuseTracer

tracer = LangfuseTracer(
    resource_key="langfuse:default",
    tags=["multi-model"]
)

result = await engine.run(inputs={...}, tracer=tracer)

# Langfuse tracks:
# - Token usage per model
# - Cost per request
# - Latency per model
```

### 3. A/B Testing

```python
import random

# Randomly route 10% to new model for testing
router = (Branch("ab_test")
    .if_(random.random() < 0.1, "new_model")
    .otherwise("current_model"))
```

### 4. Graceful Degradation

```python
# Always have fallback
llm = LLMNode(
    resource_key="llm:primary",
    fallback=["llm:secondary", "llm:tertiary"],
    ...
)
```

## Tiếp theo

- [RAG Workflow](rag-workflow.md) - Multi-model với RAG
- [Agent Workflow](agent-workflow.md) - Agents với specialized models
- [Deploy Production](../guides/production-deployment.md) - Production multi-model setup
