"""Base node class for all workflow nodes."""

from abc import ABC
from typing import Dict, Any, Callable, Optional, List, TYPE_CHECKING
from datetime import datetime
import traceback
import asyncio
import uuid
import sys

from hush.core.configs.node_config import NodeType
from hush.core.utils.context import get_current, _current_graph
from hush.core.utils.common import unique_name
from hush.core.loggings import LOGGER

if TYPE_CHECKING:
    from hush.core.states.workflow_state import WorkflowState



class BaseNode(ABC):
    """Base class for all nodes - extended to support Airflow-style operators."""

    TYPE_MAP = {"int": int, "float": float, "str": str, "bool": bool}
    NULL_VALUES = ('None', None, 'none', 'null', 'Null')

    INNER_PROCESS = "__inner__"

    __slots__ = [
        'id',
        'name',
        'description',
        'status_message',
        'type',
        'stream',
        'start',
        'end',
        'verbose',
        'sources',
        'targets',
        'inputs',
        'outputs',
        'input_schema',
        'output_schema',
        'hard_input_schema',
        'hard_output_schema',
        'input_indexes',
        'output_indexes',
        'metrics',
        'default_inputs',
        'core',
        'father',
        'contain_generation',
        'continue_loop'
    ]

    def __init_subclass__(cls, **kwargs):
        """Hook that runs when a subclass is created."""
        super().__init_subclass__(**kwargs)

        original_init = cls.__init__

        def wrapped_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if type(self) is cls and getattr(self, 'type', None) not in ["dummy", "graph"]:
                self._post_init()

        cls.__init__ = wrapped_init

    def __init__(
        self,
        id: str = None,
        name: str = None,
        description: str = "",
        status_message: str = "",
        inputs: Dict[str, Any] = None,
        outputs: Dict[str, Any] = None,
        input_schema: Any = None,
        output_schema: Any = None,
        sources: List[str] = None,
        targets: List[str] = None,
        stream: bool = False,
        start: bool = False,
        end: bool = False,
        contain_generation: bool = False,
        verbose: bool = True
    ):
        """Create a node instance."""
        self.id = id or uuid.uuid4().hex
        self.name = name or unique_name()
        self.description = description
        self.status_message = status_message

        self.stream = stream
        self.start = start
        self.end = end
        self.verbose = verbose

        self.sources: List[str] = sources or []
        self.targets: List[str] = targets or []

        self.inputs = inputs or {}
        self.outputs = outputs or {}

        self.hard_input_schema = input_schema
        self.hard_output_schema = output_schema

        self.input_indexes: Dict[str, int] = {}
        self.output_indexes: Dict[str, int] = {}

        self.metrics: Dict[str, int] = {}

        self.core: Optional[Callable] = None

        self.contain_generation = contain_generation

        self.continue_loop = False

        self.father = get_current()

        if self.father:
            if hasattr(self.father, "add_node"):
                self.father.add_node(self)

        if self.name and not self.name.replace('_', '').replace('-', '').isalnum():
            raise ValueError(f"Node name '{self.name}' must be alphanumeric with underscores/hyphens")

    def _post_init(self):
        """Process and normalize input/output configuration after initialization."""
        if self.hard_input_schema:
            self.input_schema = self.hard_input_schema
        if self.hard_output_schema:
            self.output_schema = self.hard_output_schema

        if hasattr(self, "input_schema") and hasattr(self, "output_schema"):
            if self.input_schema and self.output_schema:
                input_keys = set(self.input_schema.keys())
                output_keys = set(self.output_schema.keys())

                overlap = input_keys & output_keys
                if overlap:
                    raise ValueError(
                        f"Node '{self.name}' has duplicate parameter names between input and output schemas: {sorted(overlap)}."
                    )

        if isinstance(self.inputs, dict):
            expanded_inputs = {}
            for k, v in (self.inputs or {}).items():

                if isinstance(k, tuple):
                    for key in k:
                        processed_v = {v: key} if hasattr(v, "name") else v
                        expanded_inputs[key] = processed_v
                else:
                    processed_v = {v: k} if hasattr(v, "name") else v
                    expanded_inputs[k] = processed_v

            self.inputs = expanded_inputs

        elif hasattr(self.inputs, "name"):
            if self.input_schema:
                self.inputs = {
                    k: {self.inputs: k} for k in self.input_schema
                }

        if isinstance(self.outputs, dict):
            expanded_outputs = {}
            for k, v in (self.outputs or {}).items():

                if isinstance(k, tuple):
                    for key in k:
                        processed_v = {v: key} if hasattr(v, "name") else v
                        expanded_outputs[key] = processed_v
                else:
                    processed_v = {v: k} if hasattr(v, "name") else v
                    expanded_outputs[k] = processed_v

            self.outputs = expanded_outputs

        elif hasattr(self.outputs, "name"):
            if self.output_schema:
                self.outputs = {
                    k: {self.outputs: k} for k in self.output_schema
                }

    @property
    def full_name(self) -> str:
        """Returns the full hierarchical path of this node."""
        if self.father:
            return f"{self.father.full_name}.{self.name}"
        else:
            return self.name

    def identity(self, context_id: str) -> str:
        """Returns the full hierarchical path of this node, plus context_id."""
        return f"{self.full_name}[{context_id or 'main'}]"

    def __getitem__(self, item: str):
        return {self: item}

    def __rshift__(self, other):
        type = "normal"
        if self.type == "branch":
            type = "condition"

        if isinstance(other, list):
            for node in other:
                if hasattr(self.father, "add_edge"):
                    self.father.add_edge(self.name, node.name, type)
            return other
        elif hasattr(other, 'name'):
            if hasattr(self.father, "add_edge"):
                self.father.add_edge(self.name, other.name, type)
            return other
        else:
            return NotImplemented

    def __lshift__(self, other):
        type = "normal"
        if self.type == "branch":
            type = "condition"

        if isinstance(other, list):
            for node in other:
                if hasattr(self.father, "add_edge"):
                    self.father.add_edge(node.name, self.name, type)
            return other
        elif hasattr(other, 'name'):
            if hasattr(self.father, "add_edge"):
                self.father.add_edge(other.name, self.name, type)
            return self
        else:
            return NotImplemented

    def __rrshift__(self, other):
        """Called for [n1, n2] >> self because lists don't have __rshift__"""
        self.__lshift__(other)
        return self

    def __rlshift__(self, other):
        """Called for [n1, n2] << self because lists don't have __lshift__"""
        self.__rshift__(other)
        return self

    def is_base_node(self) -> bool:
        return True

    async def run(
        self,
        state: 'WorkflowState',
        context_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run the node with the given state."""

        parent_name = self.father.full_name if self.father else None
        state.record_execution(self.full_name, parent_name, context_id)

        request_id = state.request_id

        start_time = datetime.now()

        _outputs = {}

        try:
            _inputs = self.get_inputs(state, context_id=context_id)

            if self.verbose:
                LOGGER.info("request[%s] - running NODE: %s[%s] (%s), inputs = %s",
                    request_id, self.name, context_id, str(self.type).upper(), str(_inputs)[:200])

            if asyncio.iscoroutinefunction(self.core):
                _outputs = await self.core(**_inputs)
            else:
                _outputs = self.core(**_inputs)

            self.store_result(
                state=state,
                result=_outputs,
                context_id=context_id
            )

            if self.verbose:
                LOGGER.info("request[%s] - running NODE: %s[%s] (%s), _outputs = %s",
                    request_id, self.name, context_id, str(self.type).upper(), str(_outputs)[:200])

        except Exception as e:
            error_msg = traceback.format_exc()
            LOGGER.error(f"Error in node {self.name}: {str(e)}")
            LOGGER.error(error_msg)
            state[self.full_name, "error", context_id] = error_msg

        finally:
            await asyncio.sleep(0.00001)
            end_time = datetime.now()
            state.set_by_index(self.metrics['start_time'], start_time, context_id=context_id)
            state.set_by_index(self.metrics['end_time'], end_time, context_id=context_id)
            return _outputs

    def cast_value(self, value: Any, type_str: str):
        if value in self.NULL_VALUES:
            return None

        if value is not None and type_str in self.TYPE_MAP:
            target_type = self.TYPE_MAP[type_str]

            if target_type is bool:
                if isinstance(value, str):
                    val_lower = value.strip().lower()
                    if val_lower in ('true', '1', 'yes', 'y', 't'):
                        return True
                    elif val_lower in ('false', '0', 'no', 'n', 'f', ''):
                        return False
                    else:
                        raise ValueError(f"Cannot cast string '{value}' to bool")
                return bool(value)

            return target_type(value)

        return value

    def get_input_variables(self) -> List[str]:
        """Return the list of input variables in input schema."""
        return list(self.input_schema.keys()) if self.input_schema else []

    def get_output_variables(self) -> List[str]:
        """Return the list of output variables in output schema."""
        return list(self.output_schema.keys()) if self.output_schema else []

    def get_inputs(self, state: 'WorkflowState', context_id: str) -> Dict[str, Any]:
        """Retrieve all input values from state using pre-registered indices."""
        return {
            k: state.get_by_index(
                index=idx,
                context_id=context_id
            )
            for k, idx in self.input_indexes.items()
        }

    def get_outputs(self, state: 'WorkflowState', context_id: str) -> Dict[str, Any]:
        """Retrieve all output values from state using pre-registered indices."""
        return {
            k: state.get_by_index(
                index=idx,
                context_id=context_id
            )
            for k, idx in self.output_indexes.items()
        }

    def get_result(self, state: 'WorkflowState') -> Dict[str, Any]:
        return self.get_outputs(state, None)

    def store_result(
        self,
        state: 'WorkflowState',
        result: Dict[str, Any],
        context_id: str
    ) -> None:
        """Store the result dict in the state."""
        if not result or not self.output_schema:
            return

        for key, value in result.items():
            _type = self.output_schema[key].type

            cast_val = self.cast_value(value, _type)

            state.set_by_index(
                index=self.output_indexes[key],
                value=cast_val,
                context_id=context_id
            )

    def specific_metadata(self) -> Dict[str, Any]:
        """Return subclass-specific metadata dictionary."""
        return {}

    def metadata(self) -> Dict[str, Any]:
        """Generate a complete metadata dictionary for the node."""
        result = {
            "id": self.id,
            "name": self.full_name,
            "type": str(self.type).replace('NodeType.', '').lower()
        }

        def get_connect_key(value):
            if isinstance(value, dict) and value:
                node, var = next(iter(value.items()))
                node_iden = node.full_name if hasattr(node, 'full_name') else node
                return {node_iden: var}
            return value

        optional_fields = {
            "description": self.description,
            "input_connects": {k: get_connect_key(v) for k, v in self.inputs.items()},
            "output_connects": {k: get_connect_key(v) for k, v in self.outputs.items()},
        }

        if self.sources:
            optional_fields["sources"] = f"<- ({','.join(self.sources)})"

        if self.targets:
            optional_fields["targets"] = f"-> ({','.join(self.targets)})"

        for flag in ["stream", "start", "end"]:
            if getattr(self, flag, False):
                optional_fields[flag] = True

        result.update({k: v for k, v in optional_fields.items() if v})

        specific_data = {k: v for k, v in self.specific_metadata().items() if v}
        result.update(specific_data)

        return result

    def _sanitize_trace_data(self, data: Dict[str, Any], threshold_bytes: int) -> Dict[str, Any]:
        """Check and replace values that are too large in dictionary."""
        if not isinstance(data, dict):
            return data

        sanitized = {}
        for key, value in data.items():
            try:
                if isinstance(value, str):
                    size = len(value.encode('utf-8'))
                elif hasattr(value, 'nbytes'):
                    size = value.nbytes
                else:
                    size = sys.getsizeof(value)

                if size > threshold_bytes:
                    size_mb = size / (1024 * 1024)
                    sanitized[key] = f"maximum size exceeded ({size_mb:.2f} MB)"
                else:
                    sanitized[key] = value
            except:
                sanitized[key] = value

        return sanitized

    def trace_data(self, state: 'WorkflowState', context_id: str, size_threshold_bytes: int = 100000) -> Dict[str, Any]:
        """Generate trace data for observability."""
        full_name = self.full_name
        name = self.name

        if context_id:
            name = f"{name}:{context_id}"

        raw_input = self.get_inputs(state, context_id)
        raw_output = self.get_outputs(state, context_id)

        _data = {
            "name": name,
            "start_time": state[full_name, "start_time", context_id],
            "end_time": state[full_name, "end_time", context_id],
            "input": self._sanitize_trace_data(raw_input, size_threshold_bytes),
            "output": self._sanitize_trace_data(raw_output, size_threshold_bytes),
            "contain_generation": self.contain_generation,
            "metadata": self.metadata()
        }

        error_msg = state[full_name, "error", context_id]
        if not error_msg:
            _data.update({"level": "DEFAULT"})

            if self.type == "llm":
                _data.update({
                    "model": getattr(self, 'resource_key', None),
                    "usage": state[full_name, "usage", context_id],
                    "completion_start_time": state[full_name, "completion_start_time", context_id]
                })
        else:
            _data.update({
                "output": error_msg,
                "level": "ERROR"
            })

        return _data


class DummyNode(BaseNode):
    """Dummy node for START/END markers."""

    type: NodeType = "dummy"

    def __init__(self, name: str):
        super().__init__(name=name)

    def __rshift__(self, other):
        """Handle START >> node or START >> [nodes] case."""
        if self == START:
            current_graph = get_current()
            if current_graph and hasattr(current_graph, 'add_edge'):
                if isinstance(other, list):
                    for node in other:
                        current_graph.add_edge(self.name, node.name)
                    return other
                elif hasattr(other, 'name'):
                    current_graph.add_edge(self.name, other.name)
                    return other

        return super().__rshift__(other)

    def __rrshift__(self, other):
        """Handle [nodes] >> START or [nodes] >> END case."""
        current_graph = get_current()
        if current_graph and hasattr(current_graph, 'add_edge'):
            if self == START:
                if isinstance(other, list):
                    for node in other:
                        current_graph.add_edge(self.name, node.name)
                elif hasattr(other, 'name'):
                    current_graph.add_edge(self.name, other.name)
                return self

            elif self == END:
                if isinstance(other, list):
                    for node in other:
                        current_graph.add_edge(node.name, self.name)
                elif hasattr(other, 'name'):
                    current_graph.add_edge(other.name, self.name)
                return self

        return self

    def __rlshift__(self, other):
        """Handle node >> END or END << node case."""
        if self == END:
            current_graph = get_current()
            if current_graph and hasattr(current_graph, 'add_edge'):
                if isinstance(other, list):
                    for node in other:
                        current_graph.add_edge(node.name, self.name)
                    return self
                elif hasattr(other, 'name'):
                    current_graph.add_edge(other.name, self.name)
                    return self

        return self


# Global dummy nodes
START = DummyNode("__START__")
END = DummyNode("__END__")
CONTINUE = DummyNode("__CONTINUE__")
INPUT = DummyNode("__INPUT__")
OUTPUT = DummyNode("__OUTPUT__")
