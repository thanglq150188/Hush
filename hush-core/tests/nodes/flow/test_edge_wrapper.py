"""Test if a wrapper class can fix the precedence issue."""


class EdgeResult:
    """Wrapper returned by soft edge to capture the chain."""

    def __init__(self, source, target, graph):
        self.source = source
        self.target = target
        self.graph = graph

    def __rshift__(self, other):
        """EdgeResult >> other: connect target to other with hard edge."""
        print(f"EdgeResult.__rshift__: {self.target.name} >> {other.name}")
        self.graph.add_edge(self.target.name, other.name, soft=False)
        return other

    def __gt__(self, other):
        """EdgeResult > other: connect target to other with soft edge."""
        print(f"EdgeResult.__gt__: {self.target.name} > {other.name}")
        self.graph.add_edge(self.target.name, other.name, soft=True)
        return EdgeResult(self.target, other, self.graph)


class FakeGraph:
    def __init__(self):
        self.edges = []

    def add_edge(self, src, dst, soft=False):
        self.edges.append((src, dst, "soft" if soft else "hard"))
        print(f"  Added edge: {src} -> {dst} ({'soft' if soft else 'hard'})")


class FakeNode:
    def __init__(self, name, graph):
        self.name = name
        self.graph = graph

    def __gt__(self, other):
        """a > b: soft edge, returns EdgeResult to capture chain."""
        print(f"FakeNode.__gt__: {self.name} > {other.name}")
        self.graph.add_edge(self.name, other.name, soft=True)
        # Return EdgeResult wrapping 'other' so >> chains from 'other'
        return EdgeResult(self, other, self.graph)

    def __rshift__(self, other):
        """a >> b: hard edge."""
        print(f"FakeNode.__rshift__: {self.name} >> {other.name}")
        self.graph.add_edge(self.name, other.name, soft=False)
        return other


def test_wrapper_approach():
    """Test if returning EdgeResult from > fixes the chain."""
    graph = FakeGraph()
    a = FakeNode("a", graph)
    b = FakeNode("b", graph)
    c = FakeNode("c", graph)

    print("\n=== Testing: a > b >> c ===")
    print("(We want: a->b soft, b->c hard)")
    print()

    result = a > b >> c

    print()
    print(f"Result: {result}")
    print(f"Edges created: {graph.edges}")

    # Check if we got the right edges
    assert ("a", "b", "soft") in graph.edges, "Missing a->b soft edge"
    assert ("b", "c", "hard") in graph.edges, "Missing b->c hard edge"
    assert ("a", "c", "soft") not in graph.edges, "Unexpected a->c edge"

    print("\nSUCCESS! The wrapper approach works!")


def test_longer_chain():
    """Test a > b >> c > d >> e."""
    graph = FakeGraph()
    a = FakeNode("a", graph)
    b = FakeNode("b", graph)
    c = FakeNode("c", graph)
    d = FakeNode("d", graph)
    e = FakeNode("e", graph)

    print("\n=== Testing: a > b >> c > d >> e ===")
    print("(We want: a->b soft, b->c hard, c->d soft, d->e hard)")
    print()

    result = a > b >> c > d >> e

    print()
    print(f"Edges created: {graph.edges}")

    assert ("a", "b", "soft") in graph.edges
    assert ("b", "c", "hard") in graph.edges
    assert ("c", "d", "soft") in graph.edges
    assert ("d", "e", "hard") in graph.edges


if __name__ == "__main__":
    test_wrapper_approach()
    test_longer_chain()
