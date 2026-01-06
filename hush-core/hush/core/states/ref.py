"""Reference type for zero-copy variable linking with operation chaining."""

from typing import Union, Tuple, List, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hush.core.nodes.base import BaseNode

__all__ = ["Ref"]


# Operation types
OP_GETITEM = 'getitem'
OP_GETATTR = 'getattr'
OP_ADD = 'add'
OP_RADD = 'radd'
OP_SUB = 'sub'
OP_RSUB = 'rsub'
OP_MUL = 'mul'
OP_RMUL = 'rmul'
OP_TRUEDIV = 'truediv'
OP_RTRUEDIV = 'rtruediv'
OP_FLOORDIV = 'floordiv'
OP_RFLOORDIV = 'rfloordiv'
OP_MOD = 'mod'
OP_RMOD = 'rmod'
OP_POW = 'pow'
OP_RPOW = 'rpow'
OP_NEG = 'neg'
OP_POS = 'pos'
OP_ABS = 'abs'
OP_INVERT = 'invert'
OP_AND = 'and'
OP_RAND = 'rand'
OP_OR = 'or'
OP_ROR = 'ror'
OP_XOR = 'xor'
OP_RXOR = 'rxor'
OP_LSHIFT = 'lshift'
OP_RLSHIFT = 'rlshift'
OP_RSHIFT = 'rshift'
OP_RRSHIFT = 'rrshift'
OP_EQ = 'eq'
OP_NE = 'ne'
OP_LT = 'lt'
OP_LE = 'le'
OP_GT = 'gt'
OP_GE = 'ge'
OP_CONTAINS = 'contains'
OP_CALL = 'call'
OP_MATMUL = 'matmul'
OP_RMATMUL = 'rmatmul'
OP_APPLY = 'apply'  # Apply a function: func(x, *args, **kwargs)


class Ref:
    """Reference to another variable (zero-copy link) with operation chaining.

    Stores a reference to a node's variable and records operations performed on it.
    Operations are applied lazily when the reference is resolved.

    The node can be either:
    - A BaseNode object (during graph building)
    - A string full_name (after serialization or in schema)

    The `node` property always returns the string full_name.

    Example:
        # During graph building
        ref = Ref(some_node, "output")
        ref.node  # Returns "graph.some_node" (full_name)

        # Operation chaining
        ref = Ref(node_a, "data")
        ref['key']              # Records: x['key']
        ref['key'][0]           # Records: x['key'][0]
        ref.attribute           # Records: x.attribute
        ref + 10                # Records: x + 10
        ref['items'].count()    # Records: x['items'].count()
        ref.apply(len)          # Records: len(x)
        ref.apply(sorted, reverse=True)  # Records: sorted(x, reverse=True)

        # Execute operations on actual value
        result = ref.execute(actual_value)

        # Usage in node connections
        node_b = CodeNode(inputs={"x": Ref(node_a, "result")['key']})
    """

    __slots__ = ("_node", "var", "_ops")

    # Attributes that should not be proxied
    _RESERVED_ATTRS = frozenset({
        '_node', 'var', '_ops', 'node', 'raw_node', 'ops',
        'as_tuple', 'apply', 'execute', 'has_ops', '_with_op', '_clone'
    })

    def __init__(
        self,
        node: Union["BaseNode", str],
        var: str,
        _ops: Optional[List[Tuple[str, Any]]] = None
    ) -> None:
        """Create a reference to another variable.

        Args:
            node: Source node (BaseNode object or full_name string)
            var: Source variable name
            _ops: Internal list of recorded operations (do not use directly)
        """
        object.__setattr__(self, '_node', node)
        object.__setattr__(self, 'var', var)
        object.__setattr__(self, '_ops', _ops if _ops is not None else [])

    def _clone(self) -> "Ref":
        """Create a shallow clone of this Ref."""
        return Ref(self._node, self.var, list(self._ops))

    def _with_op(self, op: str, *args: Any) -> "Ref":
        """Return a new Ref with an additional operation."""
        new_ops = self._ops + [(op, args)]
        return Ref(self._node, self.var, new_ops)

    @property
    def node(self) -> str:
        """Get the node's full_name (always returns string)."""
        if hasattr(self._node, 'full_name'):
            return self._node.full_name
        return self._node

    @property
    def raw_node(self) -> Union["BaseNode", str]:
        """Get the raw node reference (BaseNode or str)."""
        return self._node

    @property
    def ops(self) -> List[Tuple[str, Tuple[Any, ...]]]:
        """Get the list of recorded operations."""
        return self._ops

    @property
    def has_ops(self) -> bool:
        """Check if this Ref has any recorded operations."""
        return len(self._ops) > 0

    def as_tuple(self) -> Tuple[str, str]:
        """Return as (node_full_name, var) tuple."""
        return (self.node, self.var)

    def apply(self, func: callable, *args: Any, **kwargs: Any) -> "Ref":
        """Record a function application: func(x, *args, **kwargs).

        Args:
            func: The function to apply
            *args: Additional positional arguments to pass after the value
            **kwargs: Keyword arguments to pass to the function

        Returns:
            A new Ref with the function application recorded

        Example:
            ref = Ref("node", "data")
            ref.apply(len)                    # Records: len(x)
            ref.apply(json.dumps, indent=2)   # Records: json.dumps(x, indent=2)
            ref['items'].apply(sorted, reverse=True)  # Records: sorted(x['items'], reverse=True)
        """
        return self._with_op(OP_APPLY, func, args, kwargs)

    def execute(self, value: Any) -> Any:
        """Execute all recorded operations on a value.

        Args:
            value: The initial value to transform

        Returns:
            The value after all operations have been applied
        """
        for op, args in self._ops:
            value = self._execute_op(value, op, args)
        return value

    def _execute_op(self, value: Any, op: str, args: Tuple[Any, ...]) -> Any:
        """Execute a single operation on a value."""
        # Resolve any Ref in args first
        resolved_args = tuple(
            arg.execute(value) if isinstance(arg, Ref) and arg._node == self._node and arg.var == self.var
            else arg
            for arg in args
        )

        if op == OP_GETITEM:
            return value[resolved_args[0]]
        elif op == OP_GETATTR:
            return getattr(value, resolved_args[0])
        elif op == OP_CALL:
            call_args, call_kwargs = resolved_args
            return value(*call_args, **call_kwargs)
        # Arithmetic
        elif op == OP_ADD:
            return value + resolved_args[0]
        elif op == OP_RADD:
            return resolved_args[0] + value
        elif op == OP_SUB:
            return value - resolved_args[0]
        elif op == OP_RSUB:
            return resolved_args[0] - value
        elif op == OP_MUL:
            return value * resolved_args[0]
        elif op == OP_RMUL:
            return resolved_args[0] * value
        elif op == OP_TRUEDIV:
            return value / resolved_args[0]
        elif op == OP_RTRUEDIV:
            return resolved_args[0] / value
        elif op == OP_FLOORDIV:
            return value // resolved_args[0]
        elif op == OP_RFLOORDIV:
            return resolved_args[0] // value
        elif op == OP_MOD:
            return value % resolved_args[0]
        elif op == OP_RMOD:
            return resolved_args[0] % value
        elif op == OP_POW:
            return value ** resolved_args[0]
        elif op == OP_RPOW:
            return resolved_args[0] ** value
        elif op == OP_MATMUL:
            return value @ resolved_args[0]
        elif op == OP_RMATMUL:
            return resolved_args[0] @ value
        # Unary
        elif op == OP_NEG:
            return -value
        elif op == OP_POS:
            return +value
        elif op == OP_ABS:
            return abs(value)
        elif op == OP_INVERT:
            return ~value
        # Bitwise
        elif op == OP_AND:
            return value & resolved_args[0]
        elif op == OP_RAND:
            return resolved_args[0] & value
        elif op == OP_OR:
            return value | resolved_args[0]
        elif op == OP_ROR:
            return resolved_args[0] | value
        elif op == OP_XOR:
            return value ^ resolved_args[0]
        elif op == OP_RXOR:
            return resolved_args[0] ^ value
        elif op == OP_LSHIFT:
            return value << resolved_args[0]
        elif op == OP_RLSHIFT:
            return resolved_args[0] << value
        elif op == OP_RSHIFT:
            return value >> resolved_args[0]
        elif op == OP_RRSHIFT:
            return resolved_args[0] >> value
        # Comparison
        elif op == OP_EQ:
            return value == resolved_args[0]
        elif op == OP_NE:
            return value != resolved_args[0]
        elif op == OP_LT:
            return value < resolved_args[0]
        elif op == OP_LE:
            return value <= resolved_args[0]
        elif op == OP_GT:
            return value > resolved_args[0]
        elif op == OP_GE:
            return value >= resolved_args[0]
        elif op == OP_CONTAINS:
            return resolved_args[0] in value
        # Function application
        elif op == OP_APPLY:
            func, func_args, func_kwargs = resolved_args
            return func(value, *func_args, **func_kwargs)
        else:
            raise ValueError(f"Unknown operation: {op}")

    # --- Item/Attribute access ---

    def __getitem__(self, key: Any) -> "Ref":
        return self._with_op(OP_GETITEM, key)

    def __getattr__(self, name: str) -> "Ref":
        # This is called for non-existent attributes
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return self._with_op(OP_GETATTR, name)

    def __call__(self, *args: Any, **kwargs: Any) -> "Ref":
        return self._with_op(OP_CALL, args, kwargs)

    # --- Arithmetic operators ---

    def __add__(self, other: Any) -> "Ref":
        return self._with_op(OP_ADD, other)

    def __radd__(self, other: Any) -> "Ref":
        return self._with_op(OP_RADD, other)

    def __sub__(self, other: Any) -> "Ref":
        return self._with_op(OP_SUB, other)

    def __rsub__(self, other: Any) -> "Ref":
        return self._with_op(OP_RSUB, other)

    def __mul__(self, other: Any) -> "Ref":
        return self._with_op(OP_MUL, other)

    def __rmul__(self, other: Any) -> "Ref":
        return self._with_op(OP_RMUL, other)

    def __truediv__(self, other: Any) -> "Ref":
        return self._with_op(OP_TRUEDIV, other)

    def __rtruediv__(self, other: Any) -> "Ref":
        return self._with_op(OP_RTRUEDIV, other)

    def __floordiv__(self, other: Any) -> "Ref":
        return self._with_op(OP_FLOORDIV, other)

    def __rfloordiv__(self, other: Any) -> "Ref":
        return self._with_op(OP_RFLOORDIV, other)

    def __mod__(self, other: Any) -> "Ref":
        return self._with_op(OP_MOD, other)

    def __rmod__(self, other: Any) -> "Ref":
        return self._with_op(OP_RMOD, other)

    def __pow__(self, other: Any) -> "Ref":
        return self._with_op(OP_POW, other)

    def __rpow__(self, other: Any) -> "Ref":
        return self._with_op(OP_RPOW, other)

    def __matmul__(self, other: Any) -> "Ref":
        return self._with_op(OP_MATMUL, other)

    def __rmatmul__(self, other: Any) -> "Ref":
        return self._with_op(OP_RMATMUL, other)

    # --- Unary operators ---

    def __neg__(self) -> "Ref":
        return self._with_op(OP_NEG)

    def __pos__(self) -> "Ref":
        return self._with_op(OP_POS)

    def __abs__(self) -> "Ref":
        return self._with_op(OP_ABS)

    def __invert__(self) -> "Ref":
        return self._with_op(OP_INVERT)

    # --- Bitwise operators ---

    def __and__(self, other: Any) -> "Ref":
        return self._with_op(OP_AND, other)

    def __rand__(self, other: Any) -> "Ref":
        return self._with_op(OP_RAND, other)

    def __or__(self, other: Any) -> "Ref":
        return self._with_op(OP_OR, other)

    def __ror__(self, other: Any) -> "Ref":
        return self._with_op(OP_ROR, other)

    def __xor__(self, other: Any) -> "Ref":
        return self._with_op(OP_XOR, other)

    def __rxor__(self, other: Any) -> "Ref":
        return self._with_op(OP_RXOR, other)

    def __lshift__(self, other: Any) -> "Ref":
        return self._with_op(OP_LSHIFT, other)

    def __rlshift__(self, other: Any) -> "Ref":
        return self._with_op(OP_RLSHIFT, other)

    def __rshift__(self, other: Any) -> "Ref":
        return self._with_op(OP_RSHIFT, other)

    def __rrshift__(self, other: Any) -> "Ref":
        return self._with_op(OP_RRSHIFT, other)

    # --- Comparison operators (return Ref, not bool) ---

    def __lt__(self, other: Any) -> "Ref":
        return self._with_op(OP_LT, other)

    def __le__(self, other: Any) -> "Ref":
        return self._with_op(OP_LE, other)

    def __gt__(self, other: Any) -> "Ref":
        return self._with_op(OP_GT, other)

    def __ge__(self, other: Any) -> "Ref":
        return self._with_op(OP_GE, other)

    def __contains__(self, item: Any) -> "Ref":
        return self._with_op(OP_CONTAINS, item)

    # --- Representation ---

    def __repr__(self) -> str:
        if not self._ops:
            return f"Ref({self.node!r}, {self.var!r})"
        ops_str = self._ops_to_str()
        return f"Ref({self.node!r}, {self.var!r}){ops_str}"

    def _ops_to_str(self) -> str:
        """Convert operations to a readable string."""
        result = ""
        for op, args in self._ops:
            if op == OP_GETITEM:
                result += f"[{args[0]!r}]"
            elif op == OP_GETATTR:
                result += f".{args[0]}"
            elif op == OP_CALL:
                call_args, call_kwargs = args
                parts = [repr(a) for a in call_args]
                parts += [f"{k}={v!r}" for k, v in call_kwargs.items()]
                result += f"({', '.join(parts)})"
            elif op == OP_ADD:
                result += f" + {args[0]!r}"
            elif op == OP_RADD:
                result = f"{args[0]!r} + " + result
            elif op == OP_SUB:
                result += f" - {args[0]!r}"
            elif op == OP_RSUB:
                result = f"{args[0]!r} - " + result
            elif op == OP_MUL:
                result += f" * {args[0]!r}"
            elif op == OP_RMUL:
                result = f"{args[0]!r} * " + result
            elif op == OP_TRUEDIV:
                result += f" / {args[0]!r}"
            elif op == OP_FLOORDIV:
                result += f" // {args[0]!r}"
            elif op == OP_MOD:
                result += f" % {args[0]!r}"
            elif op == OP_POW:
                result += f" ** {args[0]!r}"
            elif op == OP_NEG:
                result = "-" + result
            elif op == OP_LT:
                result += f" < {args[0]!r}"
            elif op == OP_LE:
                result += f" <= {args[0]!r}"
            elif op == OP_GT:
                result += f" > {args[0]!r}"
            elif op == OP_GE:
                result += f" >= {args[0]!r}"
            elif op == OP_EQ:
                result += f" == {args[0]!r}"
            elif op == OP_NE:
                result += f" != {args[0]!r}"
            elif op == OP_APPLY:
                func, func_args, func_kwargs = args
                func_name = getattr(func, '__name__', repr(func))
                parts = [f"{func_name}(x"]
                parts.extend(repr(a) for a in func_args)
                parts.extend(f"{k}={v!r}" for k, v in func_kwargs.items())
                result = ", ".join(parts) + ")"
            else:
                result += f".{op}({args})"
        return result

    # --- Equality and hashing (based on node, var, and ops) ---

    def __eq__(self, other: Any) -> "Ref":
        # For Ref equality comparison in workflow context, return a new Ref
        # For actual equality check, use ref.node == other.node etc.
        return self._with_op(OP_EQ, other)

    def __ne__(self, other: Any) -> "Ref":
        return self._with_op(OP_NE, other)

    def __hash__(self) -> int:
        # Convert ops to hashable form
        def make_hashable(item):
            if isinstance(item, list):
                return tuple(make_hashable(i) for i in item)
            elif isinstance(item, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in item.items()))
            elif isinstance(item, tuple):
                return tuple(make_hashable(i) for i in item)
            return item

        ops_hashable = make_hashable(self._ops)
        return hash((self.node, self.var, ops_hashable))

    def is_same_ref(self, other: "Ref") -> bool:
        """Check if two Refs point to the same node/var with same operations.

        Use this instead of == for actual equality checking, since == returns a Ref.
        """
        if not isinstance(other, Ref):
            return False
        return (
            self.node == other.node
            and self.var == other.var
            and self._ops == other._ops
        )


def main():
    """Comprehensive test cases for the Ref class."""
    import numpy as np

    print("=" * 60)
    print("Ref Class Test Suite")
    print("=" * 60)

    # Helper for test results
    def test(name: str, condition: bool):
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}")
        if not condition:
            raise AssertionError(f"Test failed: {name}")

    # Mock node class for testing
    class MockNode:
        def __init__(self, name: str):
            self.full_name = name

    # ==========================================================================
    # 1. Basic Ref Creation and Properties
    # ==========================================================================
    print("\n1. Basic Ref Creation and Properties")
    print("-" * 40)

    # Test with string node
    ref1 = Ref("graph.node_a", "output")
    test("Create Ref with string node", ref1.node == "graph.node_a")
    test("Ref var property", ref1.var == "output")
    test("Ref raw_node returns string", ref1.raw_node == "graph.node_a")
    test("Ref has no ops initially", ref1.has_ops == False)
    test("Ref ops is empty list", ref1.ops == [])
    test("as_tuple returns correct tuple", ref1.as_tuple() == ("graph.node_a", "output"))

    # Test with MockNode (simulating BaseNode)
    mock_node = MockNode("graph.mock_node")
    ref2 = Ref(mock_node, "result")
    test("Create Ref with MockNode", ref2.node == "graph.mock_node")
    test("raw_node returns MockNode object", ref2.raw_node is mock_node)

    # ==========================================================================
    # 2. __getitem__ - Indexing Operations
    # ==========================================================================
    print("\n2. __getitem__ - Indexing Operations")
    print("-" * 40)

    ref = Ref("node", "data")

    # Single key access
    ref_key = ref['key']
    test("getitem returns new Ref", isinstance(ref_key, Ref))
    test("getitem records operation", ref_key.has_ops == True)
    test("getitem execute works", ref_key.execute({"key": "value"}) == "value")

    # Nested key access
    ref_nested = ref['a']['b']['c']
    test("nested getitem chains", len(ref_nested.ops) == 3)
    test("nested getitem apply", ref_nested.execute({"a": {"b": {"c": 42}}}) == 42)

    # Integer index
    ref_idx = ref[0]
    test("integer index works", ref_idx.execute([10, 20, 30]) == 10)

    # Negative index
    ref_neg = ref[-1]
    test("negative index works", ref_neg.execute([10, 20, 30]) == 30)

    # Slice
    ref_slice = ref[1:3]
    test("slice works", ref_slice.execute([0, 1, 2, 3, 4]) == [1, 2])

    # ==========================================================================
    # 3. __getattr__ - Attribute Access
    # ==========================================================================
    print("\n3. __getattr__ - Attribute Access")
    print("-" * 40)

    class Obj:
        def __init__(self):
            self.name = "test"
            self.value = 100
            self.nested = type('Nested', (), {'inner': 'deep'})()

    ref = Ref("node", "obj")

    # Simple attribute
    ref_name = ref.name
    test("getattr returns Ref", isinstance(ref_name, Ref))
    test("getattr execute works", ref_name.execute(Obj()) == "test")

    # Nested attributes
    ref_inner = ref.nested.inner
    test("nested getattr works", ref_inner.execute(Obj()) == "deep")

    # Combined with getitem
    ref_combined = ref.items[0].name
    data = type('Data', (), {'items': [type('Item', (), {'name': 'first'})()]})()
    test("getattr + getitem combined", ref_combined.execute(data) == "first")

    # ==========================================================================
    # 4. __call__ - Method Calls
    # ==========================================================================
    print("\n4. __call__ - Method Calls")
    print("-" * 40)

    ref = Ref("node", "data")

    # No-arg method call
    ref_upper = ref.upper()
    test("method call no args", ref_upper.execute("hello") == "HELLO")

    # Method with args
    ref_split = ref.split(',')
    test("method call with args", ref_split.execute("a,b,c") == ["a", "b", "c"])

    # Method with kwargs
    ref_replace = ref.replace('a', 'x')
    test("method call with multiple args", ref_replace.execute("banana") == "bxnxnx")

    # Chained method calls
    ref_chain = ref.strip().lower().replace(' ', '_')
    test("chained method calls", ref_chain.execute("  Hello World  ") == "hello_world")

    # ==========================================================================
    # 5. Arithmetic Operators
    # ==========================================================================
    print("\n5. Arithmetic Operators")
    print("-" * 40)

    ref = Ref("node", "num")

    # Addition
    test("add", (ref + 5).execute(10) == 15)
    test("radd", (5 + ref).execute(10) == 15)

    # Subtraction
    test("sub", (ref - 3).execute(10) == 7)
    test("rsub", (20 - ref).execute(8) == 12)

    # Multiplication
    test("mul", (ref * 4).execute(5) == 20)
    test("rmul", (4 * ref).execute(5) == 20)

    # Division
    test("truediv", (ref / 2).execute(10) == 5.0)
    test("rtruediv", (100 / ref).execute(4) == 25.0)

    # Floor division
    test("floordiv", (ref // 3).execute(10) == 3)
    test("rfloordiv", (10 // ref).execute(3) == 3)

    # Modulo
    test("mod", (ref % 3).execute(10) == 1)
    test("rmod", (10 % ref).execute(3) == 1)

    # Power
    test("pow", (ref ** 2).execute(5) == 25)
    test("rpow", (2 ** ref).execute(3) == 8)

    # Matmul (with numpy)
    ref_mat = Ref("node", "matrix")
    mat_a = np.array([[1, 2], [3, 4]])
    mat_b = np.array([[5, 6], [7, 8]])
    test("matmul", np.array_equal((ref_mat @ mat_b).execute(mat_a), mat_a @ mat_b))
    # rmatmul: numpy arrays don't defer to __rmatmul__, so we test via _with_op directly
    ref_rmat = ref_mat._with_op(OP_RMATMUL, mat_b)
    test("rmatmul", np.array_equal(ref_rmat.execute(mat_a), mat_b @ mat_a))

    # ==========================================================================
    # 6. Unary Operators
    # ==========================================================================
    print("\n6. Unary Operators")
    print("-" * 40)

    ref = Ref("node", "num")

    # Negation
    test("neg", (-ref).execute(5) == -5)
    test("neg negative", (-ref).execute(-3) == 3)

    # Positive
    test("pos", (+ref).execute(-5) == -5)

    # Absolute value
    test("abs positive", abs(ref).execute(5) == 5)
    test("abs negative", abs(ref).execute(-5) == 5)

    # Bitwise invert
    test("invert", (~ref).execute(5) == ~5)

    # ==========================================================================
    # 7. Bitwise Operators
    # ==========================================================================
    print("\n7. Bitwise Operators")
    print("-" * 40)

    ref = Ref("node", "bits")

    # AND
    test("and", (ref & 0b1100).execute(0b1010) == 0b1000)
    test("rand", (0b1100 & ref).execute(0b1010) == 0b1000)

    # OR
    test("or", (ref | 0b1100).execute(0b1010) == 0b1110)
    test("ror", (0b1100 | ref).execute(0b1010) == 0b1110)

    # XOR
    test("xor", (ref ^ 0b1100).execute(0b1010) == 0b0110)
    test("rxor", (0b1100 ^ ref).execute(0b1010) == 0b0110)

    # Left shift
    test("lshift", (ref << 2).execute(0b0011) == 0b1100)
    test("rlshift", (2 << ref).execute(3) == 16)

    # Right shift
    test("rshift", (ref >> 2).execute(0b1100) == 0b0011)
    test("rrshift", (16 >> ref).execute(2) == 4)

    # ==========================================================================
    # 8. Comparison Operators
    # ==========================================================================
    print("\n8. Comparison Operators")
    print("-" * 40)

    ref = Ref("node", "val")

    # Less than
    test("lt true", (ref < 10).execute(5) == True)
    test("lt false", (ref < 10).execute(15) == False)

    # Less than or equal
    test("le true equal", (ref <= 10).execute(10) == True)
    test("le true less", (ref <= 10).execute(5) == True)
    test("le false", (ref <= 10).execute(15) == False)

    # Greater than
    test("gt true", (ref > 10).execute(15) == True)
    test("gt false", (ref > 10).execute(5) == False)

    # Greater than or equal
    test("ge true equal", (ref >= 10).execute(10) == True)
    test("ge true greater", (ref >= 10).execute(15) == True)
    test("ge false", (ref >= 10).execute(5) == False)

    # Equal (returns Ref with eq operation)
    ref_eq = ref == 10
    test("eq returns Ref", isinstance(ref_eq, Ref))
    test("eq true", ref_eq.execute(10) == True)
    test("eq false", ref_eq.execute(5) == False)

    # Not equal
    ref_ne = ref != 10
    test("ne returns Ref", isinstance(ref_ne, Ref))
    test("ne true", ref_ne.execute(5) == True)
    test("ne false", ref_ne.execute(10) == False)

    # ==========================================================================
    # 9. Contains Operation
    # ==========================================================================
    print("\n9. Contains Operation")
    print("-" * 40)

    ref = Ref("node", "container")

    # Note: 'in' operator uses __contains__ which must return bool in Python
    # So we test it differently - by checking the recorded operation
    ref_contains = ref.__contains__("x")
    test("contains records op", ref_contains.ops[-1][0] == OP_CONTAINS)
    test("contains apply list", ref_contains.execute(["a", "x", "b"]) == True)
    test("contains apply string", ref_contains.execute("text") == True)
    test("contains apply not found", ref_contains.execute(["a", "b"]) == False)

    # ==========================================================================
    # 10. Complex Chained Operations
    # ==========================================================================
    print("\n10. Complex Chained Operations")
    print("-" * 40)

    # Example: Extract, transform, compute
    ref = Ref("node", "data")

    # data['users'][0]['score'] * 2 + 10
    ref_complex = ref['users'][0]['score'] * 2 + 10
    data = {"users": [{"score": 15}, {"score": 20}]}
    test("complex chain 1", ref_complex.execute(data) == 40)  # 15 * 2 + 10

    # (data['value'] + 100) / 2 - 25
    ref_math = (ref['value'] + 100) / 2 - 25
    test("complex math chain", ref_math.execute({"value": 50}) == 50.0)  # (50+100)/2-25

    # data.name.upper().replace('A', 'X')
    ref_str = ref.name.upper().replace('A', 'X')
    obj = type('Obj', (), {'name': 'banana'})()
    test("complex string chain", ref_str.execute(obj) == "BXNXNX")

    # ==========================================================================
    # 10b. apply() - Function Application
    # ==========================================================================
    print("\n10b. apply() - Function Application")
    print("-" * 40)

    ref = Ref("node", "data")

    # Built-in functions
    ref_len = ref.apply(len)
    test("apply len", ref_len.execute([1, 2, 3, 4, 5]) == 5)

    ref_sum = ref.apply(sum)
    test("apply sum", ref_sum.execute([1, 2, 3, 4]) == 10)

    ref_sorted = ref.apply(sorted)
    test("apply sorted", ref_sorted.execute([3, 1, 2]) == [1, 2, 3])

    ref_sorted_rev = ref.apply(sorted, reverse=True)
    test("apply sorted with kwargs", ref_sorted_rev.execute([3, 1, 2]) == [3, 2, 1])

    # Custom function
    def double(x):
        return x * 2
    ref_double = ref.apply(double)
    test("apply custom func", ref_double.execute(21) == 42)

    # Function with extra args
    def add_prefix(x, prefix):
        return prefix + x
    ref_prefix = ref.apply(add_prefix, "Hello, ")
    test("apply with extra args", ref_prefix.execute("World") == "Hello, World")

    # Chained with other operations
    ref_chain = ref['items'].apply(len) * 2
    test("apply chained", ref_chain.execute({"items": [1, 2, 3]}) == 6)

    # Lambda
    ref_lambda = ref.apply(lambda x: x.upper())
    test("apply lambda", ref_lambda.execute("hello") == "HELLO")

    # json.dumps style (import not needed, just test structure)
    import json
    ref_json = ref.apply(json.dumps, indent=2)
    test("apply json.dumps", '"key"' in ref_json.execute({"key": "value"}))

    # ==========================================================================
    # 11. Immutability - Operations Return New Refs
    # ==========================================================================
    print("\n11. Immutability - Operations Return New Refs")
    print("-" * 40)

    ref = Ref("node", "x")
    ref_add = ref + 5
    ref_mul = ref * 3

    test("original unchanged after add", ref.has_ops == False)
    test("add creates new ref", ref_add is not ref)
    test("mul creates new ref", ref_mul is not ref)
    test("add and mul are different", ref_add is not ref_mul)
    test("add has 1 op", len(ref_add.ops) == 1)
    test("mul has 1 op", len(ref_mul.ops) == 1)

    # ==========================================================================
    # 12. __repr__ - String Representation
    # ==========================================================================
    print("\n12. __repr__ - String Representation")
    print("-" * 40)

    ref = Ref("graph.node", "out")
    test("repr basic", repr(ref) == "Ref('graph.node', 'out')")

    ref_item = ref['key']
    test("repr with getitem", "['key']" in repr(ref_item))

    ref_attr = ref.name
    test("repr with getattr", ".name" in repr(ref_attr))

    ref_add = ref + 10
    test("repr with add", "+ 10" in repr(ref_add))

    ref_call = ref.method(1, 2)
    test("repr with call", "(1, 2)" in repr(ref_call))

    ref_neg = -ref
    test("repr with neg", repr(ref_neg).startswith("Ref") and "-" in repr(ref_neg))

    # ==========================================================================
    # 13. __hash__ - Hashability
    # ==========================================================================
    print("\n13. __hash__ - Hashability")
    print("-" * 40)

    ref1 = Ref("node", "x")
    ref2 = Ref("node", "x")
    ref3 = Ref("node", "x")['key']
    ref4 = Ref("node", "x")['key']

    test("same refs have same hash", hash(ref1) == hash(ref2))
    test("different ops have different hash", hash(ref1) != hash(ref3))
    test("same ops have same hash", hash(ref3) == hash(ref4))

    # Can be used in set
    ref_set = {ref1, ref3}
    test("refs can be in set", len(ref_set) == 2)

    # Can be used as dict key
    ref_dict = {ref1: "value1", ref3: "value2"}
    test("refs can be dict keys", len(ref_dict) == 2)

    # ==========================================================================
    # 14. is_same_ref - True Equality Check
    # ==========================================================================
    print("\n14. is_same_ref - True Equality Check")
    print("-" * 40)

    ref1 = Ref("node", "x")
    ref2 = Ref("node", "x")
    ref3 = Ref("node", "y")
    ref4 = Ref("other", "x")
    ref5 = Ref("node", "x")['key']
    ref6 = Ref("node", "x")['key']

    test("same node/var is_same_ref", ref1.is_same_ref(ref2))
    test("different var not same", ref1.is_same_ref(ref3) == False)
    test("different node not same", ref1.is_same_ref(ref4) == False)
    test("with ops is_same_ref", ref5.is_same_ref(ref6))
    test("no ops vs ops not same", ref1.is_same_ref(ref5) == False)
    test("non-Ref returns False", ref1.is_same_ref("not a ref") == False)

    # ==========================================================================
    # 15. _clone - Cloning
    # ==========================================================================
    print("\n15. _clone - Cloning")
    print("-" * 40)

    ref = Ref("node", "x")['key'] + 5
    ref_clone = ref._clone()

    test("clone is new object", ref_clone is not ref)
    test("clone has same node", ref_clone.node == ref.node)
    test("clone has same var", ref_clone.var == ref.var)
    test("clone has same ops", ref_clone.ops == ref.ops)
    test("clone ops is different list", ref_clone.ops is not ref.ops)

    # ==========================================================================
    # 16. Error Handling
    # ==========================================================================
    print("\n16. Error Handling")
    print("-" * 40)

    # Private attribute access should raise
    ref = Ref("node", "x")
    try:
        _ = ref._private
        test("private attr raises", False)
    except AttributeError:
        test("private attr raises", True)

    # Unknown operation (shouldn't happen normally)
    ref_bad = Ref("node", "x", [("unknown_op", (1,))])
    try:
        ref_bad.execute(10)
        test("unknown op raises", False)
    except ValueError as e:
        test("unknown op raises", "Unknown operation" in str(e))

    # ==========================================================================
    # 17. Edge Cases
    # ==========================================================================
    print("\n17. Edge Cases")
    print("-" * 40)

    # Empty string node
    ref_empty = Ref("", "var")
    test("empty string node", ref_empty.node == "")

    # Special characters in var
    ref_special = Ref("node", "my-var.name")
    test("special chars in var", ref_special.var == "my-var.name")

    # None as key
    ref_none = Ref("node", "data")[None]
    test("None as key", ref_none.execute({None: "null_value"}) == "null_value")

    # Tuple as key
    ref_tuple = Ref("node", "data")[(1, 2)]
    test("tuple as key", ref_tuple.execute({(1, 2): "tuple_value"}) == "tuple_value")

    # Very long chain
    ref_long = Ref("node", "x")
    for i in range(100):
        ref_long = ref_long + 1
    test("long chain (100 ops)", ref_long.execute(0) == 100)

    # ==========================================================================
    # 18. Real-world Workflow Scenarios
    # ==========================================================================
    print("\n18. Real-world Workflow Scenarios")
    print("-" * 40)

    # Scenario 1: API response processing
    ref = Ref("api_node", "response")
    ref_users = ref['data']['users'][0]['email'].lower()
    api_response = {
        "data": {
            "users": [
                {"email": "ADMIN@EXAMPLE.COM", "role": "admin"},
                {"email": "user@example.com", "role": "user"}
            ]
        }
    }
    test("API response extraction", ref_users.execute(api_response) == "admin@example.com")

    # Scenario 2: Numeric computation
    ref = Ref("compute_node", "metrics")
    ref_scaled = ref['value'] * 100
    metrics = {"value": 0.75, "min": 0, "max": 1}
    test("numeric scaling", ref_scaled.execute(metrics) == 75.0)

    # Scenario 3: Conditional-like (comparison result)
    ref = Ref("check_node", "status")
    ref_is_active = ref['is_active'] == True
    test("boolean comparison", ref_is_active.execute({"is_active": True}) == True)
    test("boolean comparison false", ref_is_active.execute({"is_active": False}) == False)

    # Scenario 4: String template building
    ref = Ref("format_node", "data")
    ref_greeting = "Hello, " + ref['name'] + "!"
    test("string concatenation", ref_greeting.execute({"name": "World"}) == "Hello, World!")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
