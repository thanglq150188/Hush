"""Demo script to generate traces for the Hush Trace Viewer UI.

This script creates a workflow with various node types:
- CodeNode (simple transform)
- ForLoopNode (sequential iteration)
- MapNode (parallel iteration)
- WhileLoopNode (conditional iteration)
- Nested graphs
- BranchNode (conditional routing)

Then runs the workflow 100 times with LocalTracer to populate ~/.hush/traces.db
"""

import asyncio
import random
from time import sleep

from hush.core import (
    Hush,
    GraphNode,
    CodeNode,
    START, END, PARENT,
)
from hush.core.nodes.iteration.for_loop_node import ForLoopNode
from hush.core.nodes.iteration.map_node import MapNode
from hush.core.nodes.iteration.while_loop_node import WhileLoopNode
from hush.core.nodes.iteration.base import Each
from hush.core.nodes.flow.branch_node import BranchNode
from hush.core.nodes.transform.code_node import code_node
from hush.core.tracers import LocalTracer
from hush.core.background import shutdown_background


# ============================================================================
# Code Nodes (simple transforms)
# ============================================================================

@code_node
def preprocess(text: str):
    """Preprocess input text."""
    sleep(0.01)  # Simulate some work
    return {"cleaned": text.strip().lower()}


@code_node
def tokenize(text: str):
    """Split text into tokens."""
    sleep(0.02)
    tokens = text.split()
    return {"tokens": tokens, "count": len(tokens)}


@code_node
def analyze_token(token: str, multiplier: int):
    """Analyze a single token."""
    sleep(0.005)
    score = len(token) * multiplier
    return {"score": score}


@code_node
def aggregate_scores(scores: list):
    """Aggregate all scores."""
    sleep(0.01)
    total = sum(scores) if scores else 0
    avg = total / len(scores) if scores else 0
    return {"total": total, "average": avg}


@code_node
def classify(score: float):
    """Classify based on score."""
    sleep(0.01)
    if score > 50:
        return {"category": "high", "confidence": 0.9}
    elif score > 25:
        return {"category": "medium", "confidence": 0.7}
    else:
        return {"category": "low", "confidence": 0.5}


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
    return {"new_value": value // 2}


# ============================================================================
# Build the Demo Workflow
# ============================================================================

def build_demo_workflow():
    """Build a complex workflow demonstrating various node types."""

    with GraphNode(name="text-analysis-pipeline") as graph:
        # Step 1: Preprocess input
        preprocess_node = preprocess(
            name="preprocess",
            inputs={"text": PARENT["input_text"]},
        )

        # Step 2: Tokenize
        tokenize_node = tokenize(
            name="tokenize",
            inputs={"text": preprocess_node["cleaned"]},
        )

        # Step 3: MapNode - analyze each token in parallel
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

        # Step 4: Aggregate scores
        aggregate_node = aggregate_scores(
            name="aggregate",
            inputs={"scores": map_node["score"]},
        )

        # Step 5: Classify
        classify_node = classify(
            name="classify",
            inputs={"score": aggregate_node["average"]},
        )

        # Step 6: Format output
        format_node = format_output(
            name="format",
            inputs={
                "category": classify_node["category"],
                "confidence": classify_node["confidence"],
                "total": aggregate_node["total"],
            },
            outputs=PARENT,
        )

        # Connect the flow
        START >> preprocess_node >> tokenize_node >> map_node >> aggregate_node >> classify_node >> format_node >> END

    return graph


def build_nested_loop_workflow():
    """Build a workflow with nested loops, each containing 4+ nodes."""

    # Define code nodes for the nested workflow
    @code_node
    def validate_input(x: int):
        """Validate input value."""
        sleep(0.005)
        is_valid = x > 0
        return {"validated_x": x, "is_valid": is_valid}

    @code_node
    def prepare_outer(validated_x: int):
        """Prepare value for inner loop."""
        sleep(0.008)
        prepared = validated_x * 10
        return {"prepared_value": prepared}

    @code_node
    def scale_y(y: int):
        """Scale y value."""
        sleep(0.004)
        scaled = y * 2
        return {"scaled_y": scaled}

    @code_node
    def multiply(x: int, y: int):
        """Multiply x and y."""
        sleep(0.005)
        return {"product": x * y}

    @code_node
    def add_bonus(product: int):
        """Add bonus to product."""
        sleep(0.003)
        bonus = product // 10
        return {"with_bonus": product + bonus}

    @code_node
    def format_result(with_bonus: int, x: int, y: int):
        """Format the result."""
        sleep(0.004)
        return {"formatted": f"{x}*{y}={with_bonus}", "value": with_bonus}

    @code_node
    def summarize_inner(products: list):
        """Summarize inner loop results."""
        sleep(0.006)
        total = sum(products) if products else 0
        return {"inner_total": total, "count": len(products)}

    @code_node
    def finalize_outer(inner_total: int, prepared_value: int):
        """Finalize outer loop iteration."""
        sleep(0.005)
        final = inner_total + prepared_value
        return {"outer_result": final}

    with GraphNode(name="nested-loop-demo") as graph:
        # Outer ForLoop with 4 nodes
        with ForLoopNode(
            name="outer_loop",
            inputs={"x": Each([1, 2, 3])}
        ) as outer:
            # Node 1: Validate input
            validate_node = validate_input(
                name="validate",
                inputs={"x": PARENT["x"]},
            )

            # Node 2: Prepare for inner loop
            prepare_node = prepare_outer(
                name="prepare",
                inputs={"validated_x": validate_node["validated_x"]},
            )

            # Inner ForLoop with 4 nodes
            with ForLoopNode(
                name="inner_loop",
                inputs={
                    "y": Each([10, 20, 30]),
                    "x": validate_node["validated_x"],
                }
            ) as inner:
                # Inner Node 1: Scale y
                scale_node = scale_y(
                    name="scale",
                    inputs={"y": PARENT["y"]},
                )

                # Inner Node 2: Multiply
                mult_node = multiply(
                    name="multiply",
                    inputs={"x": PARENT["x"], "y": scale_node["scaled_y"]},
                )

                # Inner Node 3: Add bonus
                bonus_node = add_bonus(
                    name="bonus",
                    inputs={"product": mult_node["product"]},
                )

                # Inner Node 4: Format result
                format_node = format_result(
                    name="format",
                    inputs={
                        "with_bonus": bonus_node["with_bonus"],
                        "x": PARENT["x"],
                        "y": scale_node["scaled_y"],
                    },
                    outputs=PARENT,
                )

                START >> scale_node >> mult_node >> bonus_node >> format_node >> END

            # Node 3: Summarize inner results
            summarize_node = summarize_inner(
                name="summarize",
                inputs={"products": inner["value"]},
            )

            # Node 4: Finalize outer iteration
            finalize_node = finalize_outer(
                name="finalize",
                inputs={
                    "inner_total": summarize_node["inner_total"],
                    "prepared_value": prepare_node["prepared_value"],
                },
                outputs=PARENT,
            )

            START >> validate_node >> prepare_node >> inner >> summarize_node >> finalize_node >> END

        outer["outer_result"] >> PARENT["results"]
        START >> outer >> END

    return graph


def build_while_loop_workflow():
    """Build a workflow with while loop."""

    with GraphNode(name="while-loop-demo") as graph:
        # WhileLoop: halve until < threshold
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

async def run_demo(num_traces: int = 100):
    """Run the demo workflow multiple times to generate traces."""

    print(f"Generating {num_traces} traces...")
    print("=" * 60)

    # Build workflows
    main_workflow = build_demo_workflow()
    nested_workflow = build_nested_loop_workflow()
    while_workflow = build_while_loop_workflow()

    # Create engines
    main_engine = Hush(main_workflow)
    nested_engine = Hush(nested_workflow)
    while_engine = Hush(while_workflow)

    # Create tracer
    tracer = LocalTracer(name="demo-tracer")

    # Sample inputs
    sample_texts = [
        "Hello world this is a test",
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming industries",
        "Python is a great programming language",
        "Data science combines statistics and programming",
        "Natural language processing is fascinating",
        "Deep learning neural networks are powerful",
        "Artificial intelligence is the future",
        "Cloud computing enables scalability",
        "Software engineering best practices matter",
    ]

    # Run main workflow
    main_count = int(num_traces * 0.6)  # 60% main workflow
    print(f"\nRunning text-analysis-pipeline {main_count} times...")

    for i in range(main_count):
        text = random.choice(sample_texts)
        multiplier = random.randint(1, 5)

        result = await main_engine.run(
            inputs={
                "input_text": text,
                "multiplier": multiplier,
            },
            request_id=f"main-{i:04d}",
            user_id=f"user-{random.randint(1, 5)}",
            session_id=f"session-{random.randint(1, 10)}",
            tracer=tracer,
        )

        if i % 20 == 0:
            print(f"  Completed {i + 1}/{main_count} - {result.get('result', 'N/A')}")

    # Run nested loop workflow
    nested_count = int(num_traces * 0.2)  # 20% nested loops
    print(f"\nRunning nested-loop-demo {nested_count} times...")

    for i in range(nested_count):
        result = await nested_engine.run(
            inputs={},
            request_id=f"nested-{i:04d}",
            user_id=f"user-{random.randint(1, 5)}",
            session_id=f"session-{random.randint(1, 10)}",
            tracer=tracer,
        )

        if i % 10 == 0:
            print(f"  Completed {i + 1}/{nested_count} - results: {result.get('results', 'N/A')}")

    # Run while loop workflow
    while_count = num_traces - main_count - nested_count  # Remaining
    print(f"\nRunning while-loop-demo {while_count} times...")

    for i in range(while_count):
        start_value = random.randint(50, 500)

        result = await while_engine.run(
            inputs={"start_value": start_value},
            request_id=f"while-{i:04d}",
            user_id=f"user-{random.randint(1, 5)}",
            session_id=f"session-{random.randint(1, 10)}",
            tracer=tracer,
        )

        if i % 10 == 0:
            print(f"  Completed {i + 1}/{while_count} - {start_value} -> {result.get('final_value', 'N/A')}")

    print("\n" + "=" * 60)
    print(f"Generated {num_traces} traces!")
    print("Waiting for background process to flush...")

    # Wait for background to write all traces
    sleep(3)

    # Shutdown background process
    shutdown_background()

    print("\nTraces saved to ~/.hush/traces.db")
    print("Open hush/core/ui/index.html in your browser to view them!")


if __name__ == "__main__":
    asyncio.run(run_demo(100))
