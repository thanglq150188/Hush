"""Kiểu Ref cho liên kết biến zero-copy với khả năng chain operation."""

from typing import Union, Tuple, List, Any, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from hush.core.nodes.base import BaseNode

__all__ = ["Ref"]


class Ref:
    """Tham chiếu đến biến khác với khả năng chain các operation.

    Các operation được ghi lại và compile thành callable để thực thi nhanh.
    Ref cho phép truy cập dữ liệu từ node khác mà không cần copy,
    đồng thời hỗ trợ transform dữ liệu thông qua các operation như
    getitem, getattr, arithmetic, comparison, v.v.

    Attributes:
        _node: Node nguồn (có thể là string hoặc BaseNode)
        var: Tên biến nguồn
        _ops: Danh sách các operation đã ghi
        _fn: Function đã compile từ ops
        idx: Index trong schema (được set bởi StateSchema._build())
        is_output: True nếu đây là output ref (đẩy giá trị ra ngoài)
    """

    __slots__ = ("_node", "var", "_ops", "_fn", "idx", "is_output")

    _RESERVED_ATTRS = frozenset({
        '_node', 'var', '_ops', '_fn', 'idx', 'is_output', 'node', 'raw_node', 'ops',
        'as_tuple', 'apply', 'execute', 'has_ops', '_with_op', '_clone'
    })

    def __init__(
        self,
        node: Union["BaseNode", str],
        var: str,
        _ops: Optional[List[Tuple[str, Any]]] = None,
        _fn: Optional[Callable] = None,
        is_output: bool = False
    ) -> None:
        """Khởi tạo Ref.

        Args:
            node: Node nguồn (BaseNode hoặc string tên node)
            var: Tên biến nguồn
            _ops: Danh sách operation (dùng cho deserialization)
            _fn: Function đã compile (dùng cho clone)
            is_output: True nếu là output ref
        """
        object.__setattr__(self, '_node', node)
        object.__setattr__(self, 'var', var)
        object.__setattr__(self, '_ops', _ops or [])
        object.__setattr__(self, 'idx', -1)  # Được set bởi StateSchema._build()
        object.__setattr__(self, 'is_output', is_output)  # True cho output ref
        # Nếu có ops nhưng không có fn, rebuild từ ops (trường hợp deserialization)
        if _fn is None and _ops:
            _fn = lambda x: x
            for op, args in _ops:
                _fn = self._wrap(_fn, op, args)
        object.__setattr__(self, '_fn', _fn or (lambda x: x))

    @property
    def node(self) -> str:
        """Tên đầy đủ của node nguồn."""
        return self._node.full_name if hasattr(self._node, 'full_name') else self._node

    @property
    def raw_node(self) -> Union["BaseNode", str]:
        """Node nguồn gốc (có thể là object hoặc string)."""
        return self._node

    @property
    def ops(self) -> List[Tuple[str, Tuple[Any, ...]]]:
        """Danh sách các operation đã ghi."""
        return self._ops

    @property
    def has_ops(self) -> bool:
        """Kiểm tra có operation nào không."""
        return len(self._ops) > 0

    def as_tuple(self) -> Tuple[str, str]:
        """Trả về tuple (node_name, var_name)."""
        return (self.node, self.var)

    def _clone(self) -> "Ref":
        """Tạo bản sao của Ref."""
        return Ref(self._node, self.var, list(self._ops), self._fn, self.is_output)

    def _with_op(self, op: str, *args: Any) -> "Ref":
        """Tạo Ref mới với thêm một operation."""
        new_ops = self._ops + [(op, args)]
        new_fn = self._wrap(self._fn, op, args)
        return Ref(self._node, self.var, new_ops, new_fn)

    @staticmethod
    def _wrap(fn: Callable, op: str, args: Tuple) -> Callable:
        """Wrap function với thêm một operation."""
        a = args[0] if args else None

        match op:
            # Truy cập
            case 'getitem': return lambda x, f=fn, k=a: f(x)[k]
            case 'getattr': return lambda x, f=fn, k=a: getattr(f(x), k)
            case 'call':
                ca, kw = args
                return lambda x, f=fn, a=ca, k=kw: f(x)(*a, **k)
            # Số học
            case 'add': return lambda x, f=fn, v=a: f(x) + v
            case 'radd': return lambda x, f=fn, v=a: v + f(x)
            case 'sub': return lambda x, f=fn, v=a: f(x) - v
            case 'rsub': return lambda x, f=fn, v=a: v - f(x)
            case 'mul': return lambda x, f=fn, v=a: f(x) * v
            case 'rmul': return lambda x, f=fn, v=a: v * f(x)
            case 'truediv': return lambda x, f=fn, v=a: f(x) / v
            case 'rtruediv': return lambda x, f=fn, v=a: v / f(x)
            case 'floordiv': return lambda x, f=fn, v=a: f(x) // v
            case 'rfloordiv': return lambda x, f=fn, v=a: v // f(x)
            case 'mod': return lambda x, f=fn, v=a: f(x) % v
            case 'rmod': return lambda x, f=fn, v=a: v % f(x)
            case 'pow': return lambda x, f=fn, v=a: f(x) ** v
            case 'rpow': return lambda x, f=fn, v=a: v ** f(x)
            case 'matmul': return lambda x, f=fn, v=a: f(x) @ v
            case 'rmatmul': return lambda x, f=fn, v=a: v @ f(x)
            # Một ngôi
            case 'neg': return lambda x, f=fn: -f(x)
            case 'pos': return lambda x, f=fn: +f(x)
            case 'abs': return lambda x, f=fn: abs(f(x))
            case 'invert': return lambda x, f=fn: ~f(x)
            # Bitwise
            case 'and': return lambda x, f=fn, v=a: f(x) & v
            case 'rand': return lambda x, f=fn, v=a: v & f(x)
            case 'or': return lambda x, f=fn, v=a: f(x) | v
            case 'ror': return lambda x, f=fn, v=a: v | f(x)
            case 'xor': return lambda x, f=fn, v=a: f(x) ^ v
            case 'rxor': return lambda x, f=fn, v=a: v ^ f(x)
            case 'lshift': return lambda x, f=fn, v=a: f(x) << v
            case 'rlshift': return lambda x, f=fn, v=a: v << f(x)
            case 'rshift': return lambda x, f=fn, v=a: f(x) >> v
            case 'rrshift': return lambda x, f=fn, v=a: v >> f(x)
            # So sánh
            case 'eq': return lambda x, f=fn, v=a: f(x) == v
            case 'ne': return lambda x, f=fn, v=a: f(x) != v
            case 'lt': return lambda x, f=fn, v=a: f(x) < v
            case 'le': return lambda x, f=fn, v=a: f(x) <= v
            case 'gt': return lambda x, f=fn, v=a: f(x) > v
            case 'ge': return lambda x, f=fn, v=a: f(x) >= v
            case 'contains': return lambda x, f=fn, v=a: v in f(x)
            # Áp dụng function
            case 'apply':
                func, fa, kw = args
                return lambda x, f=fn, func=func, a=fa, k=kw: func(f(x), *a, **k)
            case _:
                raise ValueError(f"Operation không xác định: {op}")

    def execute(self, value: Any) -> Any:
        """Thực thi tất cả operation trên giá trị đầu vào.

        Args:
            value: Giá trị nguồn để transform

        Returns:
            Giá trị sau khi áp dụng tất cả operation
        """
        return self._fn(value)

    def apply(self, func: Callable, *args: Any, **kwargs: Any) -> "Ref":
        """Áp dụng một function tùy chỉnh lên giá trị.

        Args:
            func: Function cần áp dụng
            *args: Các argument bổ sung cho func
            **kwargs: Các keyword argument bổ sung cho func

        Returns:
            Ref mới với operation apply
        """
        return self._with_op('apply', func, args, kwargs)

    # =========================================================================
    # Truy cập
    # =========================================================================
    def __getitem__(self, key: Any) -> "Ref": return self._with_op('getitem', key)
    def __getattr__(self, name: str) -> "Ref":
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' không có attribute '{name}'")
        return self._with_op('getattr', name)
    def __call__(self, *args: Any, **kwargs: Any) -> "Ref": return self._with_op('call', args, kwargs)

    # =========================================================================
    # Số học
    # =========================================================================
    def __add__(self, other): return self._with_op('add', other)
    def __radd__(self, other): return self._with_op('radd', other)
    def __sub__(self, other): return self._with_op('sub', other)
    def __rsub__(self, other): return self._with_op('rsub', other)
    def __mul__(self, other): return self._with_op('mul', other)
    def __rmul__(self, other): return self._with_op('rmul', other)
    def __truediv__(self, other): return self._with_op('truediv', other)
    def __rtruediv__(self, other): return self._with_op('rtruediv', other)
    def __floordiv__(self, other): return self._with_op('floordiv', other)
    def __rfloordiv__(self, other): return self._with_op('rfloordiv', other)
    def __mod__(self, other): return self._with_op('mod', other)
    def __rmod__(self, other): return self._with_op('rmod', other)
    def __pow__(self, other): return self._with_op('pow', other)
    def __rpow__(self, other): return self._with_op('rpow', other)
    def __matmul__(self, other): return self._with_op('matmul', other)
    def __rmatmul__(self, other): return self._with_op('rmatmul', other)

    # =========================================================================
    # Một ngôi
    # =========================================================================
    def __neg__(self): return self._with_op('neg')
    def __pos__(self): return self._with_op('pos')
    def __abs__(self): return self._with_op('abs')
    def __invert__(self): return self._with_op('invert')

    # =========================================================================
    # Bitwise
    # =========================================================================
    def __and__(self, other): return self._with_op('and', other)
    def __rand__(self, other): return self._with_op('rand', other)
    def __or__(self, other): return self._with_op('or', other)
    def __ror__(self, other): return self._with_op('ror', other)
    def __xor__(self, other): return self._with_op('xor', other)
    def __rxor__(self, other): return self._with_op('rxor', other)
    def __lshift__(self, other): return self._with_op('lshift', other)
    def __rlshift__(self, other): return self._with_op('rlshift', other)
    def __rshift__(self, other): return self._with_op('rshift', other)
    def __rrshift__(self, other): return self._with_op('rrshift', other)

    # =========================================================================
    # So sánh
    # =========================================================================
    def __lt__(self, other): return self._with_op('lt', other)
    def __le__(self, other): return self._with_op('le', other)
    def __gt__(self, other): return self._with_op('gt', other)
    def __ge__(self, other): return self._with_op('ge', other)
    def __eq__(self, other): return self._with_op('eq', other)
    def __ne__(self, other): return self._with_op('ne', other)
    def __contains__(self, item): return self._with_op('contains', item)

    # =========================================================================
    # Tiện ích
    # =========================================================================
    def __repr__(self) -> str:
        if not self._ops:
            return f"Ref({self.node!r}, {self.var!r})"
        return f"Ref({self.node!r}, {self.var!r}, ops={len(self._ops)})"

    def __hash__(self) -> int:
        """Tính hash của Ref dựa trên node, var và ops."""
        def hashable(item):
            if isinstance(item, (list, tuple)):
                return tuple(hashable(i) for i in item)
            if isinstance(item, dict):
                return tuple(sorted((k, hashable(v)) for k, v in item.items()))
            return item
        return hash((self.node, self.var, hashable(self._ops)))

    def is_same_ref(self, other: "Ref") -> bool:
        """Kiểm tra hai Ref có giống nhau không (cùng node, var và ops)."""
        if not isinstance(other, Ref):
            return False
        return self.node == other.node and self.var == other.var and self._ops == other._ops


def main():
    """Các test case cho class Ref."""

    def test(name: str, condition: bool):
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}")
        if not condition:
            raise AssertionError(f"Test failed: {name}")

    class MockNode:
        def __init__(self, name: str):
            self.full_name = name

    # 1. Basic Creation
    print("\n1. Basic Creation")
    ref = Ref("graph.node", "output")
    test("string node", ref.node == "graph.node")
    test("var property", ref.var == "output")
    test("no ops initially", ref.has_ops == False)
    test("as_tuple", ref.as_tuple() == ("graph.node", "output"))

    mock = MockNode("graph.mock")
    ref2 = Ref(mock, "result")
    test("MockNode node", ref2.node == "graph.mock")
    test("raw_node returns object", ref2.raw_node is mock)

    # 2. Getitem
    print("\n2. Getitem")
    ref = Ref("n", "data")
    test("getitem dict", ref['key'].execute({"key": "value"}) == "value")
    test("getitem list", ref[0].execute([10, 20]) == 10)
    test("getitem negative", ref[-1].execute([10, 20, 30]) == 30)
    test("getitem nested", ref['a']['b'].execute({"a": {"b": 42}}) == 42)
    test("getitem slice", ref[1:3].execute([0, 1, 2, 3]) == [1, 2])

    # 3. Getattr
    print("\n3. Getattr")
    class Obj:
        name = "test"
        value = 100
    ref = Ref("n", "obj")
    test("getattr simple", ref.name.execute(Obj()) == "test")
    test("getattr value", ref.value.execute(Obj()) == 100)

    # 4. Call
    print("\n4. Call")
    ref = Ref("n", "data")
    test("call upper", ref.upper().execute("hello") == "HELLO")
    test("call split", ref.split(',').execute("a,b,c") == ["a", "b", "c"])
    test("call replace", ref.replace('a', 'x').execute("banana") == "bxnxnx")
    test("call chain", ref.strip().lower().execute("  HELLO  ") == "hello")

    # 5. Arithmetic
    print("\n5. Arithmetic")
    ref = Ref("n", "num")
    test("add", (ref + 5).execute(10) == 15)
    test("radd", (5 + ref).execute(10) == 15)
    test("sub", (ref - 3).execute(10) == 7)
    test("rsub", (20 - ref).execute(8) == 12)
    test("mul", (ref * 4).execute(5) == 20)
    test("rmul", (4 * ref).execute(5) == 20)
    test("truediv", (ref / 2).execute(10) == 5.0)
    test("rtruediv", (100 / ref).execute(4) == 25.0)
    test("floordiv", (ref // 3).execute(10) == 3)
    test("rfloordiv", (10 // ref).execute(3) == 3)
    test("mod", (ref % 3).execute(10) == 1)
    test("rmod", (10 % ref).execute(3) == 1)
    test("pow", (ref ** 2).execute(5) == 25)
    test("rpow", (2 ** ref).execute(3) == 8)

    # 6. Unary
    print("\n6. Unary")
    ref = Ref("n", "num")
    test("neg", (-ref).execute(5) == -5)
    test("pos", (+ref).execute(-5) == -5)
    test("abs", abs(ref).execute(-5) == 5)
    test("invert", (~ref).execute(5) == ~5)

    # 7. Bitwise
    print("\n7. Bitwise")
    ref = Ref("n", "bits")
    test("and", (ref & 0b1100).execute(0b1010) == 0b1000)
    test("rand", (0b1100 & ref).execute(0b1010) == 0b1000)
    test("or", (ref | 0b1100).execute(0b1010) == 0b1110)
    test("ror", (0b1100 | ref).execute(0b1010) == 0b1110)
    test("xor", (ref ^ 0b1100).execute(0b1010) == 0b0110)
    test("rxor", (0b1100 ^ ref).execute(0b1010) == 0b0110)
    test("lshift", (ref << 2).execute(0b0011) == 0b1100)
    test("rlshift", (2 << ref).execute(3) == 16)
    test("rshift", (ref >> 2).execute(0b1100) == 0b0011)
    test("rrshift", (16 >> ref).execute(2) == 4)

    # 8. Comparison
    print("\n8. Comparison")
    ref = Ref("n", "val")
    test("lt true", (ref < 10).execute(5) == True)
    test("lt false", (ref < 10).execute(15) == False)
    test("le true", (ref <= 10).execute(10) == True)
    test("gt true", (ref > 10).execute(15) == True)
    test("ge true", (ref >= 10).execute(10) == True)
    ref_eq = ref == 10
    test("eq returns Ref", isinstance(ref_eq, Ref))
    test("eq true", ref_eq.execute(10) == True)
    test("ne true", (ref != 10).execute(5) == True)

    # 9. Contains
    print("\n9. Contains")
    ref = Ref("n", "container")
    ref_contains = ref.__contains__("x")
    test("contains list", ref_contains.execute(["a", "x", "b"]) == True)
    test("contains string", ref_contains.execute("text") == True)
    test("contains not found", ref_contains.execute(["a", "b"]) == False)

    # 10. Apply
    print("\n10. Apply")
    ref = Ref("n", "data")
    test("apply len", ref.apply(len).execute([1, 2, 3]) == 3)
    test("apply sum", ref.apply(sum).execute([1, 2, 3]) == 6)
    test("apply sorted", ref.apply(sorted).execute([3, 1, 2]) == [1, 2, 3])
    test("apply sorted reverse", ref.apply(sorted, reverse=True).execute([3, 1, 2]) == [3, 2, 1])
    test("apply lambda", ref.apply(lambda x: x * 2).execute(21) == 42)
    test("apply chained", ref['items'].apply(len).execute({"items": [1, 2, 3]}) == 3)

    # 11. Complex Chains
    print("\n11. Complex Chains")
    ref = Ref("n", "data")
    test("chain 1", (ref['users'][0]['score'] * 2 + 10).execute({"users": [{"score": 15}]}) == 40)
    test("chain 2", ((ref['value'] + 100) / 2).execute({"value": 50}) == 75.0)
    test("string concat", ("Hello, " + ref['name'] + "!").execute({"name": "World"}) == "Hello, World!")

    # 12. Immutability
    print("\n12. Immutability")
    ref = Ref("n", "x")
    ref_add = ref + 5
    ref_mul = ref * 3
    test("original unchanged", ref.has_ops == False)
    test("add is new ref", ref_add is not ref)
    test("mul is new ref", ref_mul is not ref)

    # 13. Hash and is_same_ref
    print("\n13. Hash and is_same_ref")
    ref1 = Ref("n", "x")
    ref2 = Ref("n", "x")
    ref3 = Ref("n", "x")['key']
    ref4 = Ref("n", "x")['key']
    test("same refs same hash", hash(ref1) == hash(ref2))
    test("diff ops diff hash", hash(ref1) != hash(ref3))
    test("same ops same hash", hash(ref3) == hash(ref4))
    test("is_same_ref true", ref1.is_same_ref(ref2))
    test("is_same_ref with ops", ref3.is_same_ref(ref4))
    test("is_same_ref false", ref1.is_same_ref(ref3) == False)
    test("is_same_ref non-Ref", ref1.is_same_ref("not a ref") == False)

    # 14. Clone
    print("\n14. Clone")
    ref = Ref("n", "x")['key'] + 5
    ref_clone = ref._clone()
    test("clone is new object", ref_clone is not ref)
    test("clone same node", ref_clone.node == ref.node)
    test("clone same ops", ref_clone.ops == ref.ops)
    test("clone executes same", ref_clone.execute({"key": 10}) == ref.execute({"key": 10}))

    # 15. Deserialization (ops without fn)
    print("\n15. Deserialization")
    ops = [('getitem', ('key',)), ('add', (5,))]
    ref = Ref("n", "x", _ops=ops)
    test("rebuilt from ops", ref.execute({"key": 10}) == 15)

    # 16. Error Handling
    print("\n16. Error Handling")
    ref = Ref("n", "x")
    try:
        _ = ref._private
        test("private attr raises", False)
    except AttributeError:
        test("private attr raises", True)

    # 17. Repr
    print("\n17. Repr")
    ref = Ref("graph.node", "out")
    test("repr basic", repr(ref) == "Ref('graph.node', 'out')")
    ref2 = ref['key'] + 5
    test("repr with ops", "ops=2" in repr(ref2))

    print("\n" + "=" * 40)
    print("All tests passed!")
    print("=" * 40)


if __name__ == "__main__":
    main()
