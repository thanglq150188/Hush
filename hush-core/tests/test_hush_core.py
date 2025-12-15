"""Comprehensive test suite for hush-core to verify compatibility with beeflow."""

import asyncio
import sys
import os

# Add hush-core to path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List

from hush.core import (
    WorkflowEngine,
    CodeNode,
    code_node,
    LambdaNode,
    ParserNode,
    BranchNode,
    ForLoopNode,
    GraphNode,
    START, END, INPUT, OUTPUT,
    WorkflowState,
    WorkflowIndexer,
)


print("✓ All imports successful")



# ============================================================
# Test 1: Basic Code Node
# ============================================================

async def test_code_node_basic():
    """Test 1: Basic CodeNode with function."""
    print("\n" + "="*60)
    print("Test 1: Basic CodeNode")
    print("="*60)

    def add_numbers(a: int, b: int) -> Dict[str, Any]:
        """Add two numbers.

        Returns:
            sum (int): The sum of a and b
        """
        return {"sum": a + b}

    # Use a GraphNode to properly set up inputs
    with GraphNode(name="test1") as graph:
        node = CodeNode(
            name="add_node",
            code_fn=add_numbers,
            return_keys=["sum: int"],
            inputs={"a": INPUT, "b": INPUT},
            outputs={"sum": OUTPUT}
        )
        START >> node >> END

    graph.build()

    print(f"Node name: {node.name}")
    print(f"Input schema: {list(node.input_schema.keys())}")
    print(f"Output schema: {list(node.output_schema.keys())}")

    # Build and run - use same name for both!
    indexer = WorkflowIndexer("test1").add_node(graph).build()
    state = WorkflowState(indexer=indexer, inputs={"a": 5, "b": 3})

    result = await graph.run(state)

    print(f"Result: {result}")
    assert result.get("sum") == 8, f"Expected sum=8, got {result.get('sum')}"

    print("✓ Test 1 PASSED\n")


# ============================================================
# Test 2: Lambda Node
# ============================================================

async def test_lambda_node():
    """Test 2: LambdaNode."""
    print("\n" + "="*60)
    print("Test 2: LambdaNode")
    print("="*60)

    with GraphNode(name="test2") as graph:
        node = LambdaNode(
            name="lambda_multiply",
            code_fn=lambda x, y: {"product": x * y},
            return_keys=["product: int"],
            inputs={"x": INPUT, "y": INPUT},
            outputs={"product": OUTPUT}
        )
        START >> node >> END

    graph.build()

    print(f"Node name: {node.name}")
    print(f"Input schema: {list(node.input_schema.keys())}")
    print(f"Output schema: {list(node.output_schema.keys())}")

    indexer = WorkflowIndexer("test2").add_node(graph).build()
    state = WorkflowState(indexer=indexer, inputs={"x": 4, "y": 7})

    result = await graph.run(state)

    print(f"Result: {result}")
    assert result.get("product") == 28, f"Expected product=28, got {result.get('product')}"

    print("✓ Test 2 PASSED\n")


# ============================================================
# Test 3: Parser Node
# ============================================================

async def test_parser_node():
    """Test 3: ParserNode with XML."""
    print("\n" + "="*60)
    print("Test 3: ParserNode (XML)")
    print("="*60)

    xml_text = """
    <person>
        <name>John Doe</name>
        <age>30</age>
    </person>
    """

    with GraphNode(name="test3") as graph:
        node = ParserNode(
            name="xml_parser",
            format="xml",
            extract_schema=["person.name: str", "person.age: int"],
            inputs={"text": INPUT},
            outputs={"name": OUTPUT, "age": OUTPUT}
        )
        START >> node >> END

    graph.build()

    print(f"Node name: {node.name}")
    print(f"Input schema: {list(node.input_schema.keys())}")
    print(f"Output schema: {list(node.output_schema.keys())}")

    indexer = WorkflowIndexer("test3").add_node(graph).build()
    state = WorkflowState(indexer=indexer, inputs={"text": xml_text})

    result = await graph.run(state)

    print(f"Result: {result}")
    assert result.get("name") == "John Doe", f"Expected name='John Doe', got {result.get('name')}"
    assert result.get("age") == 30, f"Expected age=30, got {result.get('age')}"

    print("✓ Test 3 PASSED\n")


# ============================================================
# Test 4: Simple Workflow
# ============================================================

async def test_simple_workflow():
    """Test 4: Simple workflow with multiple nodes."""
    print("\n" + "="*60)
    print("Test 4: Simple Workflow")
    print("="*60)

    def double(n: int) -> Dict[str, Any]:
        """Double a number.
        Returns:
            doubled (int)
        """
        return {"doubled": n * 2}

    def add_ten(doubled: int) -> Dict[str, Any]:
        """Add ten.
        Returns:
            result (int)
        """
        return {"result": doubled + 10}

    with WorkflowEngine(name="simple_workflow") as workflow:
        node1 = CodeNode(
            name="double",
            code_fn=double,
            return_keys=["doubled: int"],
            inputs={"n": INPUT}
        )

        node2 = CodeNode(
            name="add_ten",
            code_fn=add_ten,
            return_keys=["result: int"],
            inputs={"doubled": node1},
            outputs={"result": OUTPUT}
        )

        START >> node1 >> node2 >> END

    workflow.compile()

    result = await workflow.run(inputs={"n": 5})

    print(f"Input: n=5")
    print(f"Result: {result}")
    # 5 * 2 = 10, 10 + 10 = 20
    assert result.get("result") == 20, f"Expected result=20, got {result.get('result')}"

    print("✓ Test 4 PASSED\n")


# ============================================================
# Test 5: Branch Node
# ============================================================

async def test_branch_workflow():
    """Test 5: Branch node workflow."""
    print("\n" + "="*60)
    print("Test 5: Branch Workflow")
    print("="*60)

    def positive_path(value: int) -> Dict[str, Any]:
        return {"result": f"Positive: {value}"}

    def negative_path(value: int) -> Dict[str, Any]:
        return {"result": f"Negative or Zero: {value}"}

    with WorkflowEngine(name="branch_workflow") as workflow:
        branch = BranchNode(
            name="check_sign",
            cases={"value > 0": "positive"},
            default="negative",
            inputs={"value": INPUT}
        )

        pos_node = CodeNode(
            name="positive",
            code_fn=positive_path,
            return_keys=["result: str"],
            inputs={"value": INPUT},
            outputs={"result": OUTPUT}
        )

        neg_node = CodeNode(
            name="negative",
            code_fn=negative_path,
            return_keys=["result: str"],
            inputs={"value": INPUT},
            outputs={"result": OUTPUT}
        )

        START >> branch >> [pos_node, neg_node]
        [pos_node, neg_node] >> END

    workflow.compile()

    # Test positive
    result1 = await workflow.run(inputs={"value": 5})
    print(f"value=5: {result1}")
    assert "Positive" in result1.get("result", ""), f"Expected Positive, got {result1}"

    # Test negative
    result2 = await workflow.run(inputs={"value": -3})
    print(f"value=-3: {result2}")
    assert "Negative" in result2.get("result", ""), f"Expected Negative, got {result2}"

    print("✓ Test 5 PASSED\n")


# ============================================================
# Test 6: ForLoop Node
# ============================================================

async def test_for_loop_workflow():
    """Test 6: ForLoop node workflow."""
    print("\n" + "="*60)
    print("Test 6: ForLoop Workflow")
    print("="*60)

    def process_item(item: str) -> Dict[str, Any]:
        """Process an item.
        Returns:
            processed (str)
        """
        return {"processed": f"processed_{item}"}

    with WorkflowEngine(name="for_loop_workflow") as workflow:
        with ForLoopNode(
            name="batch_processor",
            inputs={"batch_data": INPUT},
            outputs={"batch_result": OUTPUT}  # Export batch_result to OUTPUT
        ) as for_loop:
            processor = CodeNode(
                name="processor",
                code_fn=process_item,
                return_keys=["processed: str"],
                inputs={"item": INPUT},
                outputs={"processed": OUTPUT}
            )

            START >> processor >> END

        START >> for_loop >> END

    workflow.compile()

    batch_input = [
        {"item": "a"},
        {"item": "b"},
        {"item": "c"}
    ]

    result = await workflow.run(inputs={"batch_data": batch_input})

    print(f"Batch result: {result}")
    batch_result = result.get("batch_result", [])
    assert len(batch_result) == 3, f"Expected 3 results, got {len(batch_result)}"

    print("✓ Test 6 PASSED\n")


# ============================================================
# Test 7: @code_node Decorator
# ============================================================

async def test_code_node_decorator():
    """Test 7: @code_node decorator."""
    print("\n" + "="*60)
    print("Test 7: @code_node Decorator")
    print("="*60)

    @code_node
    def calculate_area(width: int, height: int) -> Dict[str, Any]:
        """Calculate rectangle area.
        Returns:
            area (int)
        """
        return {"area": width * height}

    with WorkflowEngine(name="decorator_workflow") as workflow:
        area_node = calculate_area(
            inputs={"width": INPUT, "height": INPUT},
            outputs={"area": OUTPUT}
        )

        START >> area_node >> END

    workflow.compile()

    result = await workflow.run(inputs={"width": 5, "height": 8})

    print(f"Result: {result}")
    assert result.get("area") == 40, f"Expected area=40, got {result.get('area')}"

    print("✓ Test 7 PASSED\n")


# ============================================================
# Test 8: Complex Nested Graph
# ============================================================

async def test_complex_nested():
    """Test 8: Complex nested graphs."""
    print("\n" + "="*60)
    print("Test 8: Complex Nested Graphs")
    print("="*60)

    def square(x: int) -> Dict[str, Any]:
        return {"squared": x * x}

    def cube(x: int) -> Dict[str, Any]:
        return {"cubed": x * x * x}

    def sum_all(squared: int, cubed: int) -> Dict[str, Any]:
        return {"total": squared + cubed}

    
    # Create outer graph using inner
    with WorkflowEngine(name="nested_workflow") as workflow:
        cb = CodeNode(
            name="cube",
            code_fn=cube,
            return_keys=["cubed: int"],
            inputs={"x": INPUT}
        )

        # Create inner graph
        with GraphNode(name="inner", inputs={"x": INPUT}) as inner:
            sq = CodeNode(
                name="square",
                code_fn=square,
                return_keys=["squared: int"],
                inputs={"x": INPUT},
                outputs={"squared": OUTPUT}
            )
            START >> sq >> END
        
        sm = CodeNode(
            name="sum",
            code_fn=sum_all,
            return_keys=["total: int"],
            inputs={"squared": inner, "cubed": cb},
            outputs={"total": OUTPUT}
        )

        START >> [inner, cb] >> sm >> END

    workflow.compile()

    result = await workflow.run(inputs={"x": 3})

    print(f"Input: x=3")
    print(f"Result: {result}")
    # 3^2 = 9, 3^3 = 27, 9 + 27 = 36
    assert result.get("total") == 36, f"Expected total=36, got {result.get('total')}"

    print("✓ Test 8 PASSED\n")


# ============================================================
# Run All Tests
# ============================================================

async def run_all_tests():
    """Run all test cases."""
    print("\n" + "="*70)
    print(" RUNNING ALL HUSH-CORE TESTS ".center(70, "="))
    print("="*70)

    tests = [
        ("Basic CodeNode", test_code_node_basic),
        ("LambdaNode", test_lambda_node),
        ("ParserNode", test_parser_node),
        ("Simple Workflow", test_simple_workflow),
        ("Branch Workflow", test_branch_workflow),
        ("ForLoop Workflow", test_for_loop_workflow),
        ("@code_node Decorator", test_code_node_decorator),
        ("Complex Nested Graphs", test_complex_nested),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in tests:
        try:
            await test_fn()
            passed += 1
        except Exception as e:
            print(f"\n✗ Test '{name}' FAILED: {e}")
            import traceback
            error_trace = traceback.format_exc()
            print(error_trace)
            errors.append((name, str(e), error_trace))
            failed += 1

    print("\n" + "="*70)
    print(f" TEST RESULTS: {passed} passed, {failed} failed ".center(70))
    print("="*70)

    if errors:
        print("\nFailed tests:")
        for name, error, _ in errors:
            print(f"  - {name}: {error}")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
