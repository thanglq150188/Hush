"""Lambda execution node for running lambda functions in workflows."""

from typing import Dict, Callable, Any, Optional, List
import asyncio
import inspect
import textwrap

from hush.core.nodes.base import BaseNode
from hush.core.configs.node_config import NodeType
from hush.core.schema import ParamSet


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


class LambdaNode(BaseNode):
    """
    A node that executes lambda functions within the graph flow.

    This node is specifically for lambda functions, requiring explicit
    output schema definition.
    """
    type: NodeType = "lambda"

    __slots__ = ["code_fn", "source"]

    def __init__(
        self,
        code_fn: Optional[Callable] = None,
        return_keys: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize LambdaNode.

        Args:
            code_fn: Lambda function to execute
            return_keys: List of output variable definitions (required)
            **kwargs: Additional keyword arguments for BaseNode
        """
        super().__init__(**kwargs)

        self.code_fn = code_fn
        self.core = ensure_async(self.code_fn)
        self.source = ""

        if not self._is_lambda_function(self.code_fn):
            raise ValueError(f"{self.code_fn} is not a lambda function")

        self._build_schemas_for_lambda(return_keys)

    def _is_lambda_function(self, func) -> bool:
        """Check if the function is a lambda function."""
        if not callable(func):
            return False
        return getattr(func, '__name__', None) == '<lambda>'

    def _build_schemas_for_lambda(self, return_keys: Optional[List[str]]):
        """Build schemas for lambda function."""
        if not return_keys:
            raise TypeError(f"Lambda functions require explicit return_keys")

        self.source = get_lambda_source(self.code_fn)

        # Set basic properties for lambda
        code_function_name = f"lambda_{id(self.code_fn)}"
        if not self.description:
            self.description = f"Lambda function: {code_function_name}"

        if not self.name:
            self.name = code_function_name

        # Build input schema from lambda signature
        try:
            sig = inspect.signature(self.code_fn)
            input_schema = ParamSet.new()

            for param_name, param in sig.parameters.items():
                param_type = "Any"
                if param.annotation != inspect.Parameter.empty:
                    param_type = getattr(param.annotation, '__name__', str(param.annotation))

                if param.default != inspect.Parameter.empty:
                    input_schema = input_schema.var(f"{param_name}: {param_type} = {repr(param.default)}")
                else:
                    input_schema = input_schema.var(f"{param_name}: {param_type}", required=True)

            self.input_schema = input_schema.build()

        except Exception:
            self.input_schema = ParamSet.new().build()

        # Build output schema from provided variables
        self.output_schema = self._build_output_schema(return_keys)

    def _build_output_schema(self, provided_schema: Optional[List[str]]) -> ParamSet:
        """Build output schema from provided list."""
        if not provided_schema:
            raise TypeError(f"{self.name}'s output_schema has not been provided")

        output_schema = ParamSet.new()
        for var_def in provided_schema:
            if ":" not in var_def:
                var_def += ": Any"
            output_schema = output_schema.var(var_def.strip())

        return output_schema.build()

    def specific_metadata(self) -> Dict[str, Any]:
        """Return subclass-specific metadata dictionary."""
        return {
            "code_fn": self.source[:200] + "..." if len(self.source) > 200 else self.source,
            "is_lambda": True,
        }
