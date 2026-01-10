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
