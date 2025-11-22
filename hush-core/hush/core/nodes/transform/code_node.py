"""Code execution node for running Python functions in workflows."""

from typing import Dict, Callable, Any, Optional, List
import asyncio
import inspect
from functools import wraps

from hush.core.nodes.base import BaseNode
from hush.core.configs.node_config import NodeType
from hush.core.schema import ParamSet


def code_node(func):
    """
    Decorator that converts a function into a CodeNode factory.

    Usage:
        @code_node
        def my_function(arg1, arg2):
            return result

        # In workflow:
        node = my_function(inputs={"arg1": INPUT, "arg2": "value"})
        START >> node >> END
    """
    @wraps(func)
    def wrapper(inputs=None, name=None, **kwargs):
        """Create a CodeNode instance."""
        node_name = name or func.__name__

        if node_name.endswith("_fn"):
            node_name = node_name[:-3]

        node_inputs = inputs or {}

        return CodeNode(
            name=node_name,
            code_fn=func,
            inputs=node_inputs,
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


class CodeNode(BaseNode):
    """
    A node that executes custom Python code within the graph flow.

    This node allows for running arbitrary code functions as part of the flow,
    supporting both synchronous and asynchronous functions.
    """
    type: NodeType = "code"

    __slots__ = ["code_fn", "source"]

    def __init__(
        self,
        code_fn: Optional[Callable] = None,
        return_keys: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize CodeNode.

        Args:
            code_fn: Python function to execute
            return_keys: List of output variable definitions
            **kwargs: Additional keyword arguments for BaseNode
        """
        super().__init__(**kwargs)

        self.code_fn = code_fn
        self.core = ensure_async(self.code_fn)
        self.source = ""
        self._build_schemas_from_function(return_keys)

    def _build_schemas_from_function(self, return_keys: Optional[List[str]]):
        """Build input/output schemas from function signature."""
        if self.code_fn is None:
            self.input_schema = ParamSet.new().build()
            self.output_schema = ParamSet.new().build()
            return

        # Get function source
        try:
            self.source = inspect.getsource(self.code_fn)
        except:
            self.source = str(self.code_fn)

        # Set description from docstring if not provided
        if not self.description and self.code_fn.__doc__:
            self.description = self.code_fn.__doc__.strip().split('\n')[0]

        # Set name from function name if not provided
        if not self.name:
            self.name = self.code_fn.__name__

        # Build input schema from function parameters
        sig = inspect.signature(self.code_fn)
        input_schema = ParamSet.new()

        for param_name, param in sig.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                type_name = param.annotation.__name__ if hasattr(param.annotation, '__name__') else str(param.annotation)
            else:
                type_name = "Any"

            if param.default != inspect.Parameter.empty:
                default_repr = repr(param.default)
                input_schema = input_schema.var(f"{param_name}: {type_name} = {default_repr}")
            else:
                input_schema = input_schema.var(f"{param_name}: {type_name}", required=True)

        self.input_schema = input_schema.build()

        # Build output schema
        self.output_schema = self._build_output_schema(return_keys)

    def _build_output_schema(self, provided_schema: Optional[List[str]]) -> ParamSet:
        """Build output schema from provided list or docstring."""
        output_schema = ParamSet.new()

        if provided_schema:
            for var_def in provided_schema:
                if ":" not in var_def:
                    var_def += ": Any"
                output_schema = output_schema.var(var_def.strip())
            return output_schema.build()

        # Try to parse from docstring
        if self.code_fn and self.code_fn.__doc__:
            doc = self.code_fn.__doc__
            if "Returns:" in doc:
                returns_section = doc.split("Returns:")[1].strip()
                for line in returns_section.split("\n"):
                    line = line.strip()
                    if line and not line.startswith("-"):
                        # Parse "name (type)" or "name: type"
                        if "(" in line:
                            name = line.split("(")[0].strip()
                            type_hint = line.split("(")[1].split(")")[0].strip() if ")" in line else "Any"
                        elif ":" in line:
                            parts = line.split(":")
                            name = parts[0].strip()
                            type_hint = parts[1].strip().split()[0] if len(parts) > 1 else "Any"
                        else:
                            name = line.split()[0].strip()
                            type_hint = "Any"

                        if name and name[0].isalpha():
                            output_schema = output_schema.var(f"{name}: {type_hint}")

        return output_schema.build()

    def specific_metadata(self) -> Dict[str, Any]:
        """Return subclass-specific metadata dictionary."""
        return {
            "code_fn": self.source[:200] + "..." if len(self.source) > 200 else self.source,
            "function_name": self.code_fn.__name__ if self.code_fn else "unknown"
        }
