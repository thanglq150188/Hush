"""Test to reproduce the chain comparison issue with `a > b >> c`."""

import pytest
from hush.core.nodes.flow.branch_node import BranchNode
from hush.core.nodes.transform.code_node import CodeNode
from hush.core.nodes.graph.graph_node import GraphNode
from hush.core.nodes.base import START, END, PARENT
from hush.core.states import StateSchema, MemoryState


class TestChainComparisonIssue:
    """Test the chain comparison issue with soft edge > followed by hard edge >>."""

    def test_soft_then_hard_edge_chain(self):
        """
        Test: a > b >> c

        Problem: Python interprets `a > b >> c` as `(a > b) and (b >> c)` due to
        comparison chaining, NOT as `(a > b) >> c`.

        Expected behavior: a connects to b with soft edge, b connects to c with hard edge.
        """
        with GraphNode(name="chain_test") as graph:
            a = CodeNode(name="a", code_fn=lambda: {"x": 1}, inputs={})
            b = CodeNode(name="b", code_fn=lambda x: {"y": x}, inputs={"x": a["x"]})
            c = CodeNode(name="c", code_fn=lambda y: {"z": y}, inputs={"y": b["y"]})

            START >> a

            # This is the problematic line - what happens?
            result = a > b >> c
            print(f"Result of 'a > b >> c': {result}, type: {type(result)}")

            c >> END

        graph.build()

        # Check edges - what do we actually get?
        print(f"Graph edges: {graph._edges}")

        # We expect:
        # - a -> b (soft edge)
        # - b -> c (hard edge)
        # But due to chain comparison, we might get something different

        # Let's check if both edges exist
        a_to_b = any(e for e in graph._edges if e.from_node == "a" and e.to_node == "b")
        b_to_c = any(e for e in graph._edges if e.from_node == "b" and e.to_node == "c")
        a_to_c = any(e for e in graph._edges if e.from_node == "a" and e.to_node == "c")

        print(f"a->b edge exists: {a_to_b}")
        print(f"b->c edge exists: {b_to_c}")
        print(f"a->c edge exists (BUG!): {a_to_c}")

        # Due to operator precedence: a > b >> c  parses as  a > (b >> c)
        # So we get: b->c and a->c, NOT a->b and b->c
        assert a_to_b, "Edge a->b should exist but doesn't due to precedence bug"
        assert b_to_c, "Edge b->c should exist"

    def test_soft_edge_only(self):
        """Test that soft edge alone works: a > b"""
        with GraphNode(name="soft_only") as graph:
            a = CodeNode(name="a", code_fn=lambda: {"x": 1}, inputs={})
            b = CodeNode(name="b", code_fn=lambda x: {"y": x}, inputs={"x": a["x"]})

            START >> a
            result = a > b
            print(f"Result of 'a > b': {result}, type: {type(result)}")
            b >> END

        graph.build()

        a_to_b = any(e for e in graph._edges if e.from_node == "a" and e.to_node == "b")
        assert a_to_b, "Edge a->b should exist"

    def test_hard_edge_only(self):
        """Test that hard edge alone works: a >> b"""
        with GraphNode(name="hard_only") as graph:
            a = CodeNode(name="a", code_fn=lambda: {"x": 1}, inputs={})
            b = CodeNode(name="b", code_fn=lambda x: {"y": x}, inputs={"x": a["x"]})

            START >> a
            result = a >> b
            print(f"Result of 'a >> b': {result}, type: {type(result)}")
            b >> END

        graph.build()

        a_to_b = any(e for e in graph._edges if e.from_node == "a" and e.to_node == "b")
        assert a_to_b, "Edge a->b should exist"

    def test_workaround_with_parentheses(self):
        """Test workaround: (a > b) >> c"""
        with GraphNode(name="parentheses_test") as graph:
            a = CodeNode(name="a", code_fn=lambda: {"x": 1}, inputs={})
            b = CodeNode(name="b", code_fn=lambda x: {"y": x}, inputs={"x": a["x"]})
            c = CodeNode(name="c", code_fn=lambda y: {"z": y}, inputs={"y": b["y"]})

            START >> a

            # Workaround with explicit parentheses
            (a > b) >> c

            c >> END

        graph.build()

        a_to_b = any(e for e in graph._edges if e.from_node == "a" and e.to_node == "b")
        b_to_c = any(e for e in graph._edges if e.from_node == "b" and e.to_node == "c")

        print(f"With parentheses - a->b exists: {a_to_b}, b->c exists: {b_to_c}")

        assert a_to_b, "Edge a->b should exist"
        assert b_to_c, "Edge b->c should exist"

    def test_what_python_actually_does(self):
        """Demonstrate what Python actually does with chain comparison."""

        class FakeNode:
            def __init__(self, name):
                self.name = name
                self.gt_called = False
                self.rshift_called = False

            def __gt__(self, other):
                print(f"{self.name}.__gt__({other.name}) called")
                self.gt_called = True
                return other  # Return other to allow chaining

            def __rshift__(self, other):
                print(f"{self.name}.__rshift__({other.name}) called")
                self.rshift_called = True
                return other

        a = FakeNode("a")
        b = FakeNode("b")
        c = FakeNode("c")

        print("\n--- Testing: a > b >> c ---")
        # Python parses this as: (a > b) and (b >> c)
        # Due to comparison chaining!
        result = a > b >> c

        print(f"\na.__gt__ called: {a.gt_called}")
        print(f"b.__rshift__ called: {b.rshift_called}")
        print(f"Result type: {type(result)}")
        print(f"Result value: {result}")

        # The issue: if a > b returns truthy, Python evaluates b >> c
        # But the overall expression becomes a boolean AND result!
