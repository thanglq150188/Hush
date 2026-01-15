"""Tests for Ref - variable reference with chainable operations."""

import pytest
from hush.core.states.ref import Ref


# ============================================================
# Helper Classes
# ============================================================

class MockNode:
    """Mock node for testing."""
    def __init__(self, name: str):
        self.full_name = name


# ============================================================
# Test 1: Basic Creation
# ============================================================

class TestBasicCreation:
    """Test basic Ref creation."""

    def test_string_node(self):
        """Test Ref creation with string node."""
        ref = Ref("graph.node", "output")

        assert ref.node == "graph.node"
        assert ref.var == "output"
        assert ref.has_ops is False

    def test_mock_node(self):
        """Test Ref creation with MockNode."""
        mock = MockNode("graph.mock")
        ref = Ref(mock, "result")

        assert ref.node == "graph.mock"
        assert ref.raw_node is mock

    def test_as_tuple(self):
        """Test as_tuple method."""
        ref = Ref("graph.node", "output")

        assert ref.as_tuple() == ("graph.node", "output")


# ============================================================
# Test 2: Getitem Operations
# ============================================================

class TestGetitemOperations:
    """Test getitem operations."""

    def test_dict_access(self):
        """Test dict key access."""
        ref = Ref("n", "data")

        assert ref['key'].execute({"key": "value"}) == "value"

    def test_list_access(self):
        """Test list index access."""
        ref = Ref("n", "data")

        assert ref[0].execute([10, 20]) == 10
        assert ref[-1].execute([10, 20, 30]) == 30

    def test_nested_access(self):
        """Test nested dict access."""
        ref = Ref("n", "data")

        assert ref['a']['b'].execute({"a": {"b": 42}}) == 42

    def test_slice_access(self):
        """Test slice access."""
        ref = Ref("n", "data")

        assert ref[1:3].execute([0, 1, 2, 3]) == [1, 2]


# ============================================================
# Test 3: Getattr Operations
# ============================================================

class TestGetattrOperations:
    """Test getattr operations."""

    def test_simple_attribute(self):
        """Test simple attribute access."""
        class Obj:
            name = "test"
            value = 100

        ref = Ref("n", "obj")

        assert ref.name.execute(Obj()) == "test"
        assert ref.value.execute(Obj()) == 100


# ============================================================
# Test 4: Method Call Operations
# ============================================================

class TestMethodCallOperations:
    """Test method call operations."""

    def test_upper(self):
        """Test string upper() method."""
        ref = Ref("n", "data")

        assert ref.upper().execute("hello") == "HELLO"

    def test_split(self):
        """Test string split() method."""
        ref = Ref("n", "data")

        assert ref.split(',').execute("a,b,c") == ["a", "b", "c"]

    def test_replace(self):
        """Test string replace() method."""
        ref = Ref("n", "data")

        assert ref.replace('a', 'x').execute("banana") == "bxnxnx"

    def test_chained_methods(self):
        """Test chained method calls."""
        ref = Ref("n", "data")

        assert ref.strip().lower().execute("  HELLO  ") == "hello"


# ============================================================
# Test 5: Arithmetic Operations
# ============================================================

class TestArithmeticOperations:
    """Test arithmetic operations."""

    def test_add(self):
        """Test addition."""
        ref = Ref("n", "num")
        assert (ref + 5).execute(10) == 15
        assert (5 + ref).execute(10) == 15

    def test_sub(self):
        """Test subtraction."""
        ref = Ref("n", "num")
        assert (ref - 3).execute(10) == 7
        assert (20 - ref).execute(8) == 12

    def test_mul(self):
        """Test multiplication."""
        ref = Ref("n", "num")
        assert (ref * 4).execute(5) == 20
        assert (4 * ref).execute(5) == 20

    def test_truediv(self):
        """Test true division."""
        ref = Ref("n", "num")
        assert (ref / 2).execute(10) == 5.0
        assert (100 / ref).execute(4) == 25.0

    def test_floordiv(self):
        """Test floor division."""
        ref = Ref("n", "num")
        assert (ref // 3).execute(10) == 3
        assert (10 // ref).execute(3) == 3

    def test_mod(self):
        """Test modulo."""
        ref = Ref("n", "num")
        assert (ref % 3).execute(10) == 1
        assert (10 % ref).execute(3) == 1

    def test_pow(self):
        """Test power."""
        ref = Ref("n", "num")
        assert (ref ** 2).execute(5) == 25
        assert (2 ** ref).execute(3) == 8


# ============================================================
# Test 6: Unary Operations
# ============================================================

class TestUnaryOperations:
    """Test unary operations."""

    def test_neg(self):
        """Test negation."""
        ref = Ref("n", "num")
        assert (-ref).execute(5) == -5

    def test_pos(self):
        """Test positive."""
        ref = Ref("n", "num")
        assert (+ref).execute(-5) == -5

    def test_abs(self):
        """Test absolute value."""
        ref = Ref("n", "num")
        assert abs(ref).execute(-5) == 5


# ============================================================
# Test 7: Comparison Operations
# ============================================================

class TestComparisonOperations:
    """Test comparison operations."""

    def test_lt(self):
        """Test less than."""
        ref = Ref("n", "val")
        assert (ref < 10).execute(5) is True
        assert (ref < 10).execute(15) is False

    def test_le(self):
        """Test less than or equal."""
        ref = Ref("n", "val")
        assert (ref <= 10).execute(10) is True

    def test_gt(self):
        """Test greater than."""
        ref = Ref("n", "val")
        assert (ref > 10).execute(15) is True

    def test_ge(self):
        """Test greater than or equal."""
        ref = Ref("n", "val")
        assert (ref >= 10).execute(10) is True

    def test_eq(self):
        """Test equality returns Ref."""
        ref = Ref("n", "val")
        ref_eq = ref == 10
        assert isinstance(ref_eq, Ref)
        assert ref_eq.execute(10) is True

    def test_ne(self):
        """Test not equal."""
        ref = Ref("n", "val")
        assert (ref != 10).execute(5) is True


# ============================================================
# Test 8: Contains Operation
# ============================================================

class TestContainsOperation:
    """Test contains operation."""

    def test_contains_in_list(self):
        """Test contains in list."""
        ref = Ref("n", "container")
        ref_contains = ref.__contains__("x")

        assert ref_contains.execute(["a", "x", "b"]) is True
        assert ref_contains.execute(["a", "b"]) is False

    def test_contains_in_string(self):
        """Test contains in string."""
        ref = Ref("n", "container")
        ref_contains = ref.__contains__("x")

        assert ref_contains.execute("text") is True


# ============================================================
# Test 9: Apply Operation
# ============================================================

class TestApplyOperation:
    """Test apply operation."""

    def test_apply_len(self):
        """Test apply(len)."""
        ref = Ref("n", "data")
        assert ref.apply(len).execute([1, 2, 3]) == 3

    def test_apply_sum(self):
        """Test apply(sum)."""
        ref = Ref("n", "data")
        assert ref.apply(sum).execute([1, 2, 3]) == 6

    def test_apply_sorted(self):
        """Test apply(sorted)."""
        ref = Ref("n", "data")
        assert ref.apply(sorted).execute([3, 1, 2]) == [1, 2, 3]

    def test_apply_with_kwargs(self):
        """Test apply with kwargs."""
        ref = Ref("n", "data")
        assert ref.apply(sorted, reverse=True).execute([3, 1, 2]) == [3, 2, 1]

    def test_apply_lambda(self):
        """Test apply with lambda."""
        ref = Ref("n", "data")
        assert ref.apply(lambda x: x * 2).execute(21) == 42

    def test_apply_chained(self):
        """Test chained getitem then apply."""
        ref = Ref("n", "data")
        assert ref['items'].apply(len).execute({"items": [1, 2, 3]}) == 3


# ============================================================
# Test 10: Complex Chains
# ============================================================

class TestComplexChains:
    """Test complex operation chains."""

    def test_nested_access_and_arithmetic(self):
        """Test nested access followed by arithmetic."""
        ref = Ref("n", "data")
        result = (ref['users'][0]['score'] * 2 + 10).execute({"users": [{"score": 15}]})
        assert result == 40

    def test_division_chain(self):
        """Test chain with division."""
        ref = Ref("n", "data")
        result = ((ref['value'] + 100) / 2).execute({"value": 50})
        assert result == 75.0

    def test_string_concat(self):
        """Test string concatenation."""
        ref = Ref("n", "data")
        result = ("Hello, " + ref['name'] + "!").execute({"name": "World"})
        assert result == "Hello, World!"


# ============================================================
# Test 11: Immutability
# ============================================================

class TestImmutability:
    """Test that operations return new Refs."""

    def test_operations_create_new_refs(self):
        """Test that operations don't mutate original."""
        ref = Ref("n", "x")
        ref_add = ref + 5
        ref_mul = ref * 3

        assert ref.has_ops is False
        assert ref_add is not ref
        assert ref_mul is not ref


# ============================================================
# Test 12: Hash and is_same_ref
# ============================================================

class TestHashAndIsSameRef:
    """Test hash and is_same_ref methods."""

    def test_same_refs_same_hash(self):
        """Test refs with same node/var have same hash."""
        ref1 = Ref("n", "x")
        ref2 = Ref("n", "x")
        assert hash(ref1) == hash(ref2)

    def test_different_ops_different_hash(self):
        """Test refs with different ops have different hash."""
        ref1 = Ref("n", "x")
        ref3 = Ref("n", "x")['key']
        assert hash(ref1) != hash(ref3)

    def test_is_same_ref_true(self):
        """Test is_same_ref returns True for equivalent refs."""
        ref1 = Ref("n", "x")
        ref2 = Ref("n", "x")
        assert ref1.is_same_ref(ref2) is True

    def test_is_same_ref_with_ops(self):
        """Test is_same_ref with same ops."""
        ref3 = Ref("n", "x")['key']
        ref4 = Ref("n", "x")['key']
        assert ref3.is_same_ref(ref4) is True

    def test_is_same_ref_false(self):
        """Test is_same_ref returns False for different refs."""
        ref1 = Ref("n", "x")
        ref3 = Ref("n", "x")['key']
        assert ref1.is_same_ref(ref3) is False

    def test_is_same_ref_non_ref(self):
        """Test is_same_ref returns False for non-Ref."""
        ref1 = Ref("n", "x")
        assert ref1.is_same_ref("not a ref") is False


# ============================================================
# Test 13: Clone
# ============================================================

class TestClone:
    """Test _clone method."""

    def test_clone_is_new_object(self):
        """Test clone creates new object."""
        ref = Ref("n", "x")['key'] + 5
        ref_clone = ref._clone()

        assert ref_clone is not ref

    def test_clone_preserves_properties(self):
        """Test clone preserves node, ops."""
        ref = Ref("n", "x")['key'] + 5
        ref_clone = ref._clone()

        assert ref_clone.node == ref.node
        assert ref_clone.ops == ref.ops

    def test_clone_executes_same(self):
        """Test clone produces same result."""
        ref = Ref("n", "x")['key'] + 5
        ref_clone = ref._clone()

        assert ref_clone.execute({"key": 10}) == ref.execute({"key": 10})


# ============================================================
# Test 14: Deserialization
# ============================================================

class TestDeserialization:
    """Test rebuilding Ref from ops."""

    def test_rebuild_from_ops(self):
        """Test Ref can be rebuilt from serialized ops."""
        ops = [('getitem', ('key',)), ('add', (5,))]
        ref = Ref("n", "x", _ops=ops)

        assert ref.execute({"key": 10}) == 15


# ============================================================
# Test 15: Error Handling
# ============================================================

class TestErrorHandling:
    """Test error handling."""

    def test_private_attr_raises(self):
        """Test accessing private attr raises AttributeError."""
        ref = Ref("n", "x")

        with pytest.raises(AttributeError):
            _ = ref._private


# ============================================================
# Test 16: Repr
# ============================================================

class TestRepr:
    """Test __repr__ method."""

    def test_repr_basic(self):
        """Test repr for basic ref."""
        ref = Ref("graph.node", "out")
        assert repr(ref) == "Ref('graph.node', 'out')"

    def test_repr_with_ops(self):
        """Test repr shows ops count."""
        ref = Ref("graph.node", "out")['key'] + 5
        assert "ops=2" in repr(ref)
