"""Generate Langfuse traces for complex workflows with nested loops and tags.

This script demonstrates two ways to use LangfuseTracer:

1. **ResourceHub (production)**: Uses HUSH_CONFIG to load config from resources.yaml
   ```python
   tracer = LangfuseTracer(resource_key="langfuse:vpbank")
   ```

2. **Direct config (simple)**: Uses LangfuseConfig directly, no ResourceHub needed
   ```python
   tracer = LangfuseTracer(config=LangfuseConfig.from_env())
   ```

The script creates traces using:
- MapNode workflows
- Nested ForLoopNode workflows
- WhileLoopNode workflows
- Complex pipelines combining all loop types
- Both static tags (set on tracer) and dynamic tags (from node output)

Run with: cd hush-experiment && uv run python examples/langfuse_complex_traces.py
"""

import asyncio
import os
from pathlib import Path
from time import sleep

# Load .env file from hush-experiment root
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)
print(f"Loaded .env from: {env_path}")
print(f"HUSH_CONFIG: {os.environ.get('HUSH_CONFIG', 'Not set')}")

from hush.core import (
    Hush,
    GraphNode,
    START, END, PARENT,
)
from hush.core.nodes.iteration.for_loop_node import ForLoopNode
from hush.core.nodes.iteration.map_node import MapNode
from hush.core.nodes.iteration.while_loop_node import WhileLoopNode
from hush.core.nodes.iteration.base import Each
from hush.core.nodes.transform.code_node import code_node
from hush.core.tracers import BaseTracer
from hush.observability import LangfuseTracer, LangfuseConfig


# ============================================================================
# Code Nodes with Dynamic Tags
# ============================================================================

@code_node
def preprocess(text: str):
    """Preprocess input text with dynamic tags."""
    sleep(0.01)
    cleaned = text.strip().lower()
    tags = ["preprocessed"]
    if len(cleaned) > 30:
        tags.append("long-text")
    return {"cleaned": cleaned, "$tags": tags}


@code_node
def tokenize(text: str):
    """Split text into tokens."""
    sleep(0.02)
    tokens = text.split()
    count = len(tokens)
    tags = ["tokenized"]
    if count > 5:
        tags.append("many-tokens")
    return {"tokens": tokens, "count": count, "$tags": tags}


@code_node
def analyze_token(token: str, multiplier: int):
    """Analyze a single token."""
    sleep(0.005)
    score = len(token) * multiplier
    return {"score": score}


@code_node
def aggregate_scores(scores: list):
    """Aggregate all scores with dynamic tags."""
    sleep(0.01)
    total = sum(scores) if scores else 0
    avg = total / len(scores) if scores else 0
    tags = ["aggregated"]
    if avg > 20:
        tags.append("high-avg-score")
    return {"total": total, "average": avg, "$tags": tags}


@code_node
def classify(score: float):
    """Classify based on score with dynamic category tag."""
    sleep(0.01)
    if score > 50:
        category, confidence = "high", 0.9
    elif score > 25:
        category, confidence = "medium", 0.7
    else:
        category, confidence = "low", 0.5
    return {"category": category, "confidence": confidence, "$tags": [f"category:{category}"]}


@code_node
def format_output(category: str, confidence: float, total: int):
    """Format final output."""
    sleep(0.01)
    return {
        "result": f"{category} (confidence: {confidence:.0%})",
        "details": {"category": category, "confidence": confidence, "total_score": total}
    }


@code_node
def halve_value(value: int):
    """Halve a value (for while loop demo)."""
    sleep(0.005)
    new_val = value // 2
    tags = []
    if new_val < 10:
        tags.append("small-value")
    return {"new_value": new_val, "$tags": tags} if tags else {"new_value": new_val}


@code_node
def square(x: int):
    """Square a number."""
    sleep(0.005)
    return {"squared": x * x}


@code_node
def validate(x: int):
    """Validate input."""
    sleep(0.005)
    return {"validated_x": x, "$tags": ["validated"]}


@code_node
def multiply(x: int, y: int):
    """Multiply two numbers."""
    sleep(0.005)
    product = x * y
    tags = ["multiplied"]
    if product > 50:
        tags.append("large-product")
    return {"product": product, "$tags": tags}


@code_node
def summarize(products: list):
    """Summarize products."""
    sleep(0.006)
    total = sum(products) if products else 0
    return {"total": total}


# ============================================================================
# Build Workflows
# ============================================================================

def build_map_node_workflow():
    """Build a workflow with MapNode for parallel processing."""
    with GraphNode(name="map-node-workflow") as graph:
        with MapNode(
            name="square_items",
            inputs={"x": Each(PARENT["items"])}
        ) as map_node:
            square_node = square(
                name="square",
                inputs={"x": PARENT["x"]},
                outputs=PARENT,
            )
            START >> square_node >> END

        map_node["squared"] >> PARENT["results"]
        START >> map_node >> END

    return graph


def build_text_analysis_workflow():
    """Build text analysis workflow with MapNode."""
    with GraphNode(name="text-analysis-pipeline") as graph:
        preprocess_node = preprocess(
            name="preprocess",
            inputs={"text": PARENT["input_text"]},
        )

        tokenize_node = tokenize(
            name="tokenize",
            inputs={"text": preprocess_node["cleaned"]},
        )

        with MapNode(
            name="analyze_tokens",
            inputs={
                "token": Each(tokenize_node["tokens"]),
                "multiplier": PARENT["multiplier"],
            }
        ) as map_node:
            analyze = analyze_token(
                name="analyze",
                inputs={
                    "token": PARENT["token"],
                    "multiplier": PARENT["multiplier"],
                },
                outputs=PARENT,
            )
            START >> analyze >> END

        aggregate_node = aggregate_scores(
            name="aggregate",
            inputs={"scores": map_node["score"]},
        )

        classify_node = classify(
            name="classify",
            inputs={"score": aggregate_node["average"]},
        )

        format_node = format_output(
            name="format",
            inputs={
                "category": classify_node["category"],
                "confidence": classify_node["confidence"],
                "total": aggregate_node["total"],
            },
            outputs=PARENT,
        )

        START >> preprocess_node >> tokenize_node >> map_node >> aggregate_node >> classify_node >> format_node >> END

    return graph


def build_nested_loop_workflow():
    """Build workflow with nested ForLoops."""
    with GraphNode(name="nested-loop-workflow") as graph:
        with ForLoopNode(
            name="outer_loop",
            inputs={"x": Each([2, 3, 4])}
        ) as outer:
            validate_node = validate(
                name="validate",
                inputs={"x": PARENT["x"]},
            )

            with ForLoopNode(
                name="inner_loop",
                inputs={
                    "y": Each([10, 20, 30]),
                    "x": validate_node["validated_x"],
                }
            ) as inner:
                mult_node = multiply(
                    name="multiply",
                    inputs={"x": PARENT["x"], "y": PARENT["y"]},
                    outputs=PARENT,
                )
                START >> mult_node >> END

            summarize_node = summarize(
                name="summarize",
                inputs={"products": inner["product"]},
                outputs=PARENT,
            )

            START >> validate_node >> inner >> summarize_node >> END

        outer["total"] >> PARENT["results"]
        START >> outer >> END

    return graph


def build_while_loop_workflow():
    """Build workflow with WhileLoopNode."""
    with GraphNode(name="while-loop-workflow") as graph:
        with WhileLoopNode(
            name="halve_loop",
            inputs={"value": PARENT["start_value"]},
            stop_condition="value < 5",
            max_iterations=10,
        ) as while_loop:
            halve_node = halve_value(
                name="halve",
                inputs={"value": PARENT["value"]},
            )
            halve_node["new_value"] >> PARENT["value"]
            START >> halve_node >> END

        while_loop["value"] >> PARENT["final_value"]
        START >> while_loop >> END

    return graph


# ============================================================================
# Run Demo
# ============================================================================

async def run_langfuse_traces():
    """Run workflows and generate Langfuse traces."""
    import time

    # Generate unique suffix for request IDs to avoid duplicates in Langfuse
    run_id = int(time.time())

    print("=" * 70)
    print("Langfuse Complex Workflow Traces Generator")
    print(f"Run ID: {run_id}")
    print("=" * 70)
    print()

    # ========================================================================
    # Two ways to create LangfuseTracer
    # ========================================================================
    print("Creating tracers using two approaches:")
    print()

    # Approach 1: ResourceHub (production) - uses HUSH_CONFIG
    print("  [ResourceHub] Using resource_key='langfuse:vpbank'")
    print(f"                Config loaded from: {os.environ.get('HUSH_CONFIG')}")
    tracer_resource_hub = LangfuseTracer(resource_key="langfuse:vpbank", tags=["resource-hub"])
    print(f"                Tracer: {tracer_resource_hub}")
    print()

    # Approach 2: Direct config (simple) - no ResourceHub needed
    print("  [Direct Config] Using LangfuseConfig.from_env()")
    direct_config = LangfuseConfig.from_env()
    tracer_direct = LangfuseTracer(config=direct_config, tags=["direct-config"])
    print(f"                  Config: host={direct_config.host}")
    print(f"                  Tracer: {tracer_direct}")
    print()

    # Build workflows
    map_workflow = build_map_node_workflow()
    text_workflow = build_text_analysis_workflow()
    nested_workflow = build_nested_loop_workflow()
    while_workflow = build_while_loop_workflow()

    # Create engines
    map_engine = Hush(map_workflow)
    text_engine = Hush(text_workflow)
    nested_engine = Hush(nested_workflow)
    while_engine = Hush(while_workflow)

    # ========================================================================
    # Trace 1: Using ResourceHub tracer
    # ========================================================================
    print("1. MapNode workflow [ResourceHub] (tags: resource-hub, production)")
    tracer1 = LangfuseTracer(resource_key="langfuse:vpbank", tags=["resource-hub", "production"])

    result = await map_engine.run(
        inputs={"items": [2, 3, 4, 5]},
        request_id=f"langfuse-map-{run_id}",
        user_id="user-alice",
        session_id=f"session-prod-{run_id}",
        tracer=tracer1,
    )
    print(f"   Results: {result.get('results')}")

    # ========================================================================
    # Trace 2: Using Direct Config tracer
    # ========================================================================
    print("2. Text analysis pipeline [Direct Config] (tags: direct-config, staging)")
    tracer2 = LangfuseTracer(config=LangfuseConfig.from_env(), tags=["direct-config", "staging"])

    result = await text_engine.run(
        inputs={"input_text": "Machine learning transforms how we process data", "multiplier": 3},
        request_id=f"langfuse-text-{run_id}-1",
        user_id="user-bob",
        session_id=f"session-staging-{run_id}",
        tracer=tracer2,
    )
    print(f"   Result: {result.get('result')}")

    # ========================================================================
    # Trace 3: Using ResourceHub tracer
    # ========================================================================
    print("3. Text analysis - long text [ResourceHub] (tags: resource-hub, development)")
    tracer3 = LangfuseTracer(resource_key="langfuse:vpbank", tags=["resource-hub", "development"])

    result = await text_engine.run(
        inputs={
            "input_text": "The quick brown fox jumps over the lazy dog and runs away into the forest",
            "multiplier": 4
        },
        request_id=f"langfuse-text-{run_id}-2",
        user_id="user-charlie",
        session_id=f"session-dev-{run_id}",
        tracer=tracer3,
    )
    print(f"   Result: {result.get('result')}")

    # ========================================================================
    # Trace 4: Using Direct Config tracer
    # ========================================================================
    print("4. Nested ForLoop workflow [Direct Config] (tags: direct-config, experiment)")
    tracer4 = LangfuseTracer(config=LangfuseConfig.from_env(), tags=["direct-config", "experiment"])

    result = await nested_engine.run(
        inputs={},
        request_id=f"langfuse-nested-{run_id}",
        user_id="user-alice",
        session_id=f"session-experiment-{run_id}",
        tracer=tracer4,
    )
    print(f"   Results: {result.get('results')}")

    # ========================================================================
    # Trace 5: Using ResourceHub tracer
    # ========================================================================
    print("5. WhileLoop - high start value [ResourceHub] (tags: resource-hub, performance)")
    tracer5 = LangfuseTracer(resource_key="langfuse:vpbank", tags=["resource-hub", "performance"])

    result = await while_engine.run(
        inputs={"start_value": 256},
        request_id=f"langfuse-while-{run_id}-1",
        user_id="user-bob",
        session_id=f"session-perf-{run_id}",
        tracer=tracer5,
    )
    print(f"   256 -> {result.get('final_value')}")

    # ========================================================================
    # Trace 6: Using Direct Config tracer
    # ========================================================================
    print("6. WhileLoop - low start value [Direct Config] (tags: direct-config, iteration)")
    tracer6 = LangfuseTracer(config=LangfuseConfig.from_env(), tags=["direct-config", "iteration"])

    result = await while_engine.run(
        inputs={"start_value": 20},
        request_id=f"langfuse-while-{run_id}-2",
        user_id="user-charlie",
        session_id=f"session-perf-{run_id}",
        tracer=tracer6,
    )
    print(f"   20 -> {result.get('final_value')}")

    # ========================================================================
    # Trace 7: Using ResourceHub tracer
    # ========================================================================
    print("7. Text analysis - high scores [ResourceHub] (tags: resource-hub, monitoring)")
    tracer7 = LangfuseTracer(resource_key="langfuse:vpbank", tags=["resource-hub", "monitoring"])

    result = await text_engine.run(
        inputs={
            "input_text": "Artificial intelligence deep learning neural networks powerful computing",
            "multiplier": 5
        },
        request_id=f"langfuse-text-{run_id}-3",
        user_id="user-alice",
        session_id=f"session-monitor-{run_id}",
        tracer=tracer7,
    )
    print(f"   Result: {result.get('result')}")

    # ========================================================================
    # Trace 8: Using Direct Config tracer
    # ========================================================================
    print("8. Same workflow, different user [Direct Config] (tags: direct-config, user-tracking)")
    tracer8 = LangfuseTracer(config=LangfuseConfig.from_env(), tags=["direct-config", "user-tracking"])

    result = await text_engine.run(
        inputs={"input_text": "Cloud computing scalability microservices", "multiplier": 2},
        request_id=f"langfuse-text-{run_id}-4",
        user_id="user-david",
        session_id=f"session-tracking-{run_id}",
        tracer=tracer8,
    )
    print(f"   Result: {result.get('result')}")

    # ========================================================================
    # Flush and cleanup
    # ========================================================================
    print()
    print("=" * 70)
    print("All workflows completed! Waiting for traces to flush to Langfuse...")
    print("=" * 70)

    # Wait for background flush to complete
    sleep(5)

    # Shutdown the executor to ensure all traces are flushed
    BaseTracer.shutdown_executor()

    print()
    print("Traces should now be visible at: https://langfuse.aws.coreai.vpbank.dev")
    print()
    print(f"Generated traces (run_id={run_id}):")
    print(f"  [ResourceHub]   langfuse-map-{run_id}: MapNode workflow")
    print(f"  [Direct Config] langfuse-text-{run_id}-1: Text analysis (staging)")
    print(f"  [ResourceHub]   langfuse-text-{run_id}-2: Text analysis (long text)")
    print(f"  [Direct Config] langfuse-nested-{run_id}: Nested ForLoops")
    print(f"  [ResourceHub]   langfuse-while-{run_id}-1: WhileLoop (256 start)")
    print(f"  [Direct Config] langfuse-while-{run_id}-2: WhileLoop (20 start)")
    print(f"  [ResourceHub]   langfuse-text-{run_id}-3: Text analysis (high scores)")
    print(f"  [Direct Config] langfuse-text-{run_id}-4: Text analysis (user tracking)")
    print()
    print("Filter by tags in Langfuse:")
    print("  - 'resource-hub': traces created via ResourceHub")
    print("  - 'direct-config': traces created via direct LangfuseConfig")


def main():
    """Main entry point."""
    asyncio.run(run_langfuse_traces())


if __name__ == "__main__":
    main()
