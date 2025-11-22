"""Workflow state management for tracking execution context."""

from typing import Dict, Any, List, Optional, Tuple
import uuid

from hush.core.states.workflow_indexer import WorkflowIndexer
from hush.core.states.state_value import VariableCell
from hush.core.loggings import LOGGER



class WorkflowState:
    """Workflow state for tracking execution context and variables."""

    __slots__ = ["_indexer", "_values", '_execution_order', '_metadata']

    def __init__(self,
                 indexer: WorkflowIndexer,
                 inputs: Dict[str, Any] = None,
                 **metadata) -> None:
        self._indexer = indexer
        self._values = [VariableCell(v) for v in indexer._values]
        self._execution_order: List[Dict[str, str]] = []
        self._metadata = {
            'user_id': str(uuid.uuid4()),
            'session_id': str(uuid.uuid4()),
            'request_id': str(uuid.uuid4()),
            **metadata
        }

        if inputs:
            for var_name, value in inputs.items():
                self[indexer._name, var_name, None] = value

    def inject_inputs(self,
                      node: str,
                      inputs: Dict[str, Any],
                      context_id: str = None) -> None:
        """Inject inputs into a node's context."""
        if inputs:
            for var_name, value in inputs.items():
                try:
                    self[node, var_name, context_id] = value
                except ValueError as e:
                    LOGGER.error(str(e))

    def __setitem__(self, key: Tuple[str, str, str], value: Any) -> None:
        if len(key) == 3:
            node, var, context = key
            index = self._indexer[node, var]
            self._values[index][context] = value
        else:
            raise ValueError(f"key must be (node, var, context)")

    def __getitem__(self, key: Tuple[str, str, str]) -> Any:
        if len(key) == 3:
            node, var, context = key
            index = self._indexer[node, var]
            return self._values[index][context]
        else:
            raise ValueError(f"key must be (node, var, context)")

    def get(self, node: str, var: str) -> VariableCell:
        index = self._indexer[node, var]
        return self._values[index]

    def get_by_index(self, index: int, context_id: str) -> Any:
        """Direct array access by index."""
        if 0 <= index < len(self._values):
            return self._values[index][context_id]
        raise IndexError(f"Index {index} out of range")

    def set_by_index(self, index: int, value: Any, context_id: str) -> None:
        """Direct array assignment by index."""
        if 0 <= index < len(self._values):
            self._values[index][context_id] = value
        else:
            raise IndexError(f"Index {index} out of range")

    def record_execution(self, node_name: str, parent: str, context_id: str) -> None:
        """Record node execution for observability tracking."""
        self._execution_order.append({"node": node_name, "parent": parent, "context_id": context_id})

    @property
    def name(self) -> str:
        """Workflow name."""
        return self._indexer._name

    @property
    def execution_order(self) -> List[str]:
        """Execution history (copy)."""
        return self._execution_order.copy()

    @property
    def metadata(self) -> Dict[str, Any]:
        """State metadata (copy)."""
        return self._metadata.copy()

    @property
    def user_id(self) -> str:
        """Get user identifier."""
        return self._metadata["user_id"]

    @property
    def session_id(self) -> str:
        """Get session identifier."""
        return self._metadata["session_id"]

    @property
    def request_id(self) -> str:
        """Get request identifier."""
        return self._metadata["request_id"]

    def __contains__(self, key: Tuple[str, str]) -> bool:
        """Check if (node, variable) mapping exists."""
        return key in self._indexer

    def __len__(self) -> int:
        """Number of storage slots."""
        return len(self._values)

    def __enter__(self) -> 'WorkflowState':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        pass

    def __repr__(self) -> str:
        return (f"WorkflowState(name='{self.name}', "
                f"variables={len(self)}, "
                f"executions={len(self._execution_order)})")

    def show(self):
        """Display current state values."""
        for (node, var) in self._indexer:
            variable_cell = self.get(node, var)

            if not variable_cell.contexts:
                print(f"{node}.{var} → {variable_cell.DEFAULT_VALUE}")
            elif len(variable_cell.contexts) == 1:
                context_id = variable_cell.versions[0]
                value = variable_cell.contexts[context_id]
                print(f"{node}.{var} → [{context_id}] → {value}")
            else:
                print(f"{node}.{var}:")
                for context_id in variable_cell.versions:
                    value = variable_cell.contexts[context_id]
                    print(f"  [{context_id}] → {value}")
