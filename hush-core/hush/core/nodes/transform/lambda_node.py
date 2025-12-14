"""Lambda execution node for running lambda functions in workflows."""

from typing import Dict, Callable, Any, Optional
import asyncio
import inspect
import textwrap
import ast

from hush.core.nodes.base import BaseNode
from hush.core.configs.node_config import NodeType
from hush.core.utils.common import Param


TYPE_MAP = {
    "str": str, "string": str,
    "int": int, "integer": int,
    "float": float,
    "bool": bool, "boolean": bool,
    "list": list, "List": list,
    "dict": dict, "Dict": dict,
    "any": Any, "Any": Any,
}


def ensure_async(func: Callable) -> Callable:
    """Ensure a function is async."""
    if asyncio.iscoroutinefunction(func):
        return func

    async def async_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return async_wrapper


def get_lambda_source(lambda_func):
    """Extract source code of a lambda function."""
    try:
        source = inspect.getsource(lambda_func)
        return textwrap.dedent(source)
    except:
        return str(lambda_func)


def parse_comment(comment: str) -> tuple:
    """Parse comment like '(type) description' into (type, description)."""
    comment = comment.strip()

    if comment.startswith("(") and ")" in comment:
        type_part, description = comment.split(")", 1)
        type_str = type_part[1:].strip()
        param_type = TYPE_MAP.get(type_str, Any)
        description = description.strip()
    else:
        param_type = Any
        description = comment

    return param_type, description


def extract_lambda_schema(func: Callable) -> Dict[str, Param]:
    """Extract return schema from lambda function source code using AST.

    Supports:
        lambda x: {"key": value}
        lambda x: {"key": value,  # (type) description}
    """
    try:
        source = inspect.getsource(func)
        source_lines = source.splitlines()
        source = inspect.cleandoc(source)
        tree = ast.parse(source)

        schema = {}
        for node in ast.walk(tree):
            # Lambda body is the return value (no explicit return statement)
            if isinstance(node, ast.Lambda):
                if isinstance(node.body, ast.Dict):
                    for key_node in node.body.keys:
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


class LambdaNode(BaseNode):
    """Node that executes a lambda function.

    Output schema is parsed from inline comments in the return dict.

    Usage:
        node = LambdaNode(
            name="double",
            code_fn=lambda x: {
                "result": x * 2,  # (int) doubled value
            },
            inputs={"x": some_node["value"]}
        )
    """

    type: NodeType = "lambda"

    __slots__ = ["code_fn", "source"]

    def __init__(
        self,
        code_fn: Optional[Callable] = None,
        **kwargs
    ):
        if code_fn and not self._is_lambda_function(code_fn):
            raise ValueError(f"{code_fn} is not a lambda function")

        # Build schemas before super().__init__
        input_schema, output_schema = self._build_schemas(code_fn)

        super().__init__(
            input_schema=input_schema,
            output_schema=output_schema,
            **kwargs
        )

        self.code_fn = code_fn
        self.core = ensure_async(code_fn) if code_fn else None
        self.source = get_lambda_source(code_fn) if code_fn else ""

        if not self.description and code_fn:
            self.description = f"Lambda function: lambda_{id(code_fn)}"

    def _is_lambda_function(self, func) -> bool:
        """Check if the function is a lambda function."""
        if not callable(func):
            return False
        return getattr(func, '__name__', None) == '<lambda>'

    def _build_schemas(self, code_fn: Optional[Callable]) -> tuple:
        """Build input/output schemas."""
        if code_fn is None:
            return {}, {}

        # Input schema from lambda signature
        input_schema = {}
        try:
            sig = inspect.signature(code_fn)
            for param_name, param in sig.parameters.items():
                param_type = Any
                if param.annotation != inspect.Parameter.empty:
                    param_type = param.annotation

                has_default = param.default != inspect.Parameter.empty
                default_val = param.default if has_default else None

                input_schema[param_name] = Param(
                    type=param_type,
                    required=not has_default,
                    default=default_val
                )
        except Exception:
            pass

        # Output schema from lambda source code
        output_schema = extract_lambda_schema(code_fn)

        return input_schema, output_schema

    def specific_metadata(self) -> Dict[str, Any]:
        """Return subclass-specific metadata."""
        return {
            "code_fn": self.source[:200] + "..." if len(self.source) > 200 else self.source,
            "is_lambda": True,
        }


if __name__ == "__main__":
    import asyncio
    from hush.core.states import StateSchema

    async def main():
        # Test 1: Simple lambda with type hint in comment
        double_node = LambdaNode(
            name="double",
            code_fn=lambda x: {"result": x * 2},  # (int) doubled value
            inputs={"x": 5}
        )

        print(f"Name: {double_node.name}")
        print(f"Input schema: {double_node.input_schema}")
        print(f"Output schema: {double_node.output_schema}")

        schema = StateSchema("test")
        state = schema.create_state()
        result = await double_node.run(state)
        print(f"Result: {result}")

        # Test 2: Lambda with multiple return keys (multi-line)
        process_node = LambdaNode(
            name="process",
            code_fn=lambda name, count: {
                "greeting": f"Hello, {name}!",  # (str) the greeting message
                "total": count * 2,  # (int) doubled count
                "status": "ok",  # no type hint, defaults to Any
            },
            inputs={"name": "World", "count": 10}
        )

        print(f"\nProcess node output schema:")
        for key, param in process_node.output_schema.items():
            print(f"  {key}: type={param.type}, desc='{param.description}'")

        result2 = await process_node.run(state)
        print(f"Process result: {result2}")

        # Test 3: Verify non-lambda raises error
        try:
            def regular_func(x):
                return {"result": x}

            LambdaNode(name="bad", code_fn=regular_func)
        except ValueError as e:
            print(f"\nCorrectly rejected non-lambda: {e}")

        # Test 4: Quick test using __call__
        print("\n" + "=" * 50)
        print("Test 4: Quick test using __call__")
        print("=" * 50)

        triple_node = LambdaNode(
            name="triple",
            code_fn=lambda x: {"result": x * 3}  # (int) tripled value
        )
        result4 = triple_node(x=7)
        print(f"triple_node(x=7) = {result4}")

    asyncio.run(main())
