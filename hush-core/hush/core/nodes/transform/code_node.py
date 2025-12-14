"""Code execution node for running Python functions in workflows."""

from typing import Dict, Callable, Any, Optional, List
import asyncio
import inspect
import ast
from functools import wraps

from hush.core.nodes.base import BaseNode
from hush.core.configs.node_config import NodeType
from hush.core.utils.common import Param


def code_node(func):
    """
    Decorator that converts a function into a CodeNode factory.

    Usage:
        @code_node
        def my_function(arg1, arg2):
            return {"result": value}

        node = my_function(inputs={"arg1": INPUT, "arg2": "value"})
    """
    @wraps(func)
    def wrapper(inputs=None, name=None, **kwargs):
        node_name = name or func.__name__
        if node_name.endswith("_fn"):
            node_name = node_name[:-3]

        return CodeNode(
            name=node_name,
            code_fn=func,
            inputs=inputs or {},
            **kwargs
        )

    wrapper.__wrapped__ = func
    return wrapper


def ensure_async(func: Callable) -> Callable:
    """Ensure a function is async."""
    if asyncio.iscoroutinefunction(func):
        return func

    async def async_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return async_wrapper


TYPE_MAP = {
    "str": str, "string": str,
    "int": int, "integer": int,
    "float": float,
    "bool": bool, "boolean": bool,
    "list": list, "List": list,
    "dict": dict, "Dict": dict,
    "any": Any, "Any": Any,
}


def parse_comment(comment: str) -> tuple:
    """Parse comment like '(type) description' into (type, description).

    Examples:
        '(str) greeting message' -> (str, 'greeting message')
        '(int) the count' -> (int, 'the count')
        'just a description' -> (Any, 'just a description')
    """
    comment = comment.strip()

    if comment.startswith("(") and ")" in comment:
        type_part, description = comment.split(")", 1)
        type_str = type_part[1:].strip()  # remove leading '('
        param_type = TYPE_MAP.get(type_str, Any)
        description = description.strip()
    else:
        param_type = Any
        description = comment

    return param_type, description


def extract_return_schema(func: Callable) -> Dict[str, Param]:
    """Extract return schema from function source code using AST.

    Supports:
        return {"key": value}
        return {"key": value,  # description}
        return {"key": value,  # (type) description}
    """
    try:
        source = inspect.getsource(func)
        source_lines = source.splitlines()
        source = inspect.cleandoc(source)
        tree = ast.parse(source)

        schema = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Return) and node.value:
                if isinstance(node.value, ast.Dict):
                    for key_node in node.value.keys:
                        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                            key_name = key_node.value

                            # Extract type and description from inline comment
                            param_type = Any
                            description = ""
                            for src_line in source_lines:
                                if "#" in src_line and f'"{key_name}"' in src_line:
                                    comment = src_line.split("#", 1)[1].strip()
                                    param_type, description = parse_comment(comment)
                                    break

                            schema[key_name] = Param(
                                type=param_type,
                                description=description
                            )
        return schema
    except Exception:
        return {}


class CodeNode(BaseNode):
    """Node that executes a Python function."""

    type: NodeType = "code"

    __slots__ = ["code_fn", "source"]

    def __init__(
        self,
        code_fn: Optional[Callable] = None,
        return_keys: Optional[List[str]] = None,
        **kwargs
    ):
        # Build schemas before super().__init__
        input_schema, output_schema = self._build_schemas(code_fn, return_keys)

        super().__init__(
            input_schema=input_schema,
            output_schema=output_schema,
            **kwargs
        )

        self.code_fn = code_fn
        self.core = ensure_async(code_fn) if code_fn else None

        # Get source code
        try:
            self.source = inspect.getsource(code_fn) if code_fn else ""
        except:
            self.source = str(code_fn) if code_fn else ""

        # Set description from docstring if not provided
        if not self.description and code_fn and code_fn.__doc__:
            self.description = code_fn.__doc__.strip().split('\n')[0]

    def _build_schemas(
        self,
        code_fn: Optional[Callable],
        return_keys: Optional[List[str]]
    ) -> tuple:
        """Build input/output schemas from function signature and source."""
        if code_fn is None:
            return {}, {}

        # Build input schema from function parameters
        input_schema = {}
        sig = inspect.signature(code_fn)

        for param_name, param in sig.parameters.items():
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else Any
            has_default = param.default != inspect.Parameter.empty
            default_val = param.default if has_default else None

            input_schema[param_name] = Param(
                type=param_type,
                required=not has_default,
                default=default_val
            )

        # Build output schema: explicit return_keys > AST parsing
        if return_keys:
            output_schema = {key: Param(type=Any) for key in return_keys}
        else:
            # Parse return schema from source code (with type hints and descriptions)
            output_schema = extract_return_schema(code_fn)

        return input_schema, output_schema

    def specific_metadata(self) -> Dict[str, Any]:
        """Return subclass-specific metadata."""
        return {
            "code_fn": self.source[:200] + "..." if len(self.source) > 200 else self.source,
            "function_name": self.code_fn.__name__ if self.code_fn else "unknown"
        }


if __name__ == "__main__":
    import asyncio
    from hush.core.states import StateSchema

    @code_node
    def add_numbers(a: int, b: int = 10):
        """Add two numbers."""
        return {"result": a + b}  # (int) the sum

    @code_node
    def process_data(name: str, count: int):
        """Process data with type hints and descriptions."""
        return {
            "message": f"Hello, {name}!",  # (str) greeting message
            "total": count * 2,  # (int) doubled count
            "status": "success",  # just a description, type defaults to Any
        }

    async def main():
        # Test 1: Single-line return with type hint in comment
        node = add_numbers(inputs={"a": 5, "b": 3})
        print(f"Name: {node.name}")
        print(f"Input schema: {node.input_schema}")
        print(f"Output schema: {node.output_schema}")

        schema = StateSchema("test")
        state = schema.create_state()
        result = await node.run(state)
        print(f"Result: {result}")

        # Test 2: Multi-line return with type hints and descriptions
        proc = process_data(inputs={"name": "World", "count": 5})
        print(f"\nProcess node output schema:")
        for key, param in proc.output_schema.items():
            print(f"  {key}: type={param.type}, desc='{param.description}'")
        result2 = await proc.run(state)
        print(f"Process result: {result2}")

        # Test 3: Quick test using __call__
        print("\n" + "=" * 50)
        print("Test 3: Quick test using __call__")
        print("=" * 50)

        node3 = add_numbers()
        result3 = node3(a=10, b=20)
        print(f"node3(a=10, b=20) = {result3}")

        node4 = process_data()
        result4 = node4(name="Alice", count=3)
        print(f"node4(name='Alice', count=3) = {result4}")

    asyncio.run(main())
