"""Example demonstrating nested ForLoop with WhileLoop inside.

This example shows:
1. ForLoopNode iterating over a list of items
2. WhileLoopNode processing each item until a condition is met
3. Using >> operator for output mapping
4. Nested graph structures with PARENT references
"""

import asyncio

from hush.core import (
    GraphNode,
    code_node,
    START, END, PARENT,
    StateSchema,
)
from hush.core.nodes.iteration.for_loop_node import ForLoopNode
from hush.core.nodes.iteration.while_loop_node import WhileLoopNode
from hush.core.nodes.iteration.base import Each


# ============================================================================
# Example: ForLoop with WhileLoop Inside
# ============================================================================

@code_node
def halve(value: int):
    """Divide value by 2."""
    return {"new_value": value // 2}


async def run_nested_loop_example():
    """ForLoop iterating items, with WhileLoop processing each item.

    ForLoop: iterate over [10, 20, 30]
    WhileLoop: for each item, divide by 2 until < 5

    Expected results:
    - 10 -> 5 -> 2 (stops at 2)
    - 20 -> 10 -> 5 -> 2 (stops at 2)
    - 30 -> 15 -> 7 -> 3 (stops at 3)
    """
    print("=" * 60)
    print("Nested ForLoop + WhileLoop Example")
    print("=" * 60)
    print()

    # Build the graph
    with GraphNode(name="forloop_whileloop_graph") as graph:
        # Outer ForLoop: iterate over [10, 20, 30]
        with ForLoopNode(
            name="outer_for",
            inputs={"item": Each([10, 20, 30])}
        ) as for_loop:
            # Inner WhileLoop: divide by 2 until value < 5
            with WhileLoopNode(
                name="inner_while",
                inputs={"value": PARENT["item"]},
                stop_condition="value < 5",
                max_iterations=5
            ) as while_loop:
                # Halve the value each iteration
                halve_node = halve(
                    inputs={"value": PARENT["value"]}
                )
                # Map output back to parent's value
                halve_node["new_value"] >> PARENT["value"]
                START >> halve_node >> END

            # Map while_loop output to for_loop
            while_loop["value"] >> PARENT["value"]
            START >> while_loop >> END

        # Map for_loop results to graph output
        for_loop["value"] >> PARENT["results"]
        START >> for_loop >> END

    # Build and run
    graph.build()
    schema = StateSchema(graph)
    state = schema.create_state(inputs={})

    print("Input: [10, 20, 30]")
    print("Operation: Divide each by 2 repeatedly until < 5")
    print()

    result = await graph.run(state)

    print("Processing steps:")
    print("  10 -> 5 -> 2 (stops at 2, since 2 < 5)")
    print("  20 -> 10 -> 5 -> 2 (stops at 2)")
    print("  30 -> 15 -> 7 -> 3 (stops at 3, since 3 < 5)")
    print()
    print(f"Results: {result.get('results', 'N/A')}")
    print(f"Expected: [2, 2, 3]")

    return result


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    result = asyncio.run(run_nested_loop_example())
