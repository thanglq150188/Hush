"""Graph node for managing subgraphs of workflow nodes."""

from datetime import datetime
from typing import Dict, List, Literal, Any, Optional
from collections import defaultdict
import asyncio

from hush.core.configs.edge_config import EdgeConfig, EdgeType
from hush.core.configs.node_config import NodeType
from hush.core.nodes.base import BaseNode, END
from hush.core.states import BaseState, StateSchema
from hush.core.utils.context import _current_graph
from hush.core.utils.bimap import BiMap
from hush.core.loggings import LOGGER


NodeFlowType = Literal["MERGE", "FORK", "BLOOM", "BRANCH", "NORMAL", "OTHER"]


class GraphNode(BaseNode):
    """
    Node that contains and manages a subgraph of nodes.

    Allows organizing nodes into reusable subgraphs with parallel branch
    execution and proper flow control.
    """

    __slots__ = [
        '_token',
        '_nodes',
        'entries',
        'exits',
        'prevs',
        'nexts',
        'ready_count',
        'flowtype_map',
        '_edges',
        '_edges_lookup',
        '_is_building'
    ]

    type: NodeType = "graph"

    def __init__(self, **kwargs):
        """Initialize GraphNode."""
        super().__init__(**kwargs)
        self._token = None
        self._is_building = True
        self._nodes: Dict[str, BaseNode] = {}
        self._edges = []
        self._edges_lookup = {}
        self.entries = []
        self.exits = []
        self.prevs = defaultdict(list)
        self.nexts = defaultdict(list)
        self.flowtype_map = BiMap[str, NodeFlowType]()

    def __enter__(self):
        """Enter context manager mode."""
        self._token = _current_graph.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager mode."""
        _current_graph.reset(self._token)

    def _setup_endpoints(self):
        """Initialize entry/exit nodes."""
        LOGGER.info(f"Graph: '{self.name}': initializing endpoints...")

        if not self.entries:
            self.entries = [node for node in self._nodes if not self.prevs[node]]

        if not self.exits:
            self.exits = [node for node in self._nodes if not self.nexts[node]]

        if not self.entries:
            raise ValueError("Graph must have at least one entry node.")
        if not self.exits:
            raise ValueError("Graph must have at least one exit node.")

    def _setup_schema(self):
        """Initialize input/output schema."""
        from hush.core.utils.common import Param

        LOGGER.info(f"Graph '{self.name}': Creating configs...")
        input_schema = {}
        output_schema = {"continue_loop": Param(type=bool, default=False)}

        for _, node in self._nodes.items():
            # Check inputs: if ref_key is self (father), it's a graph input
            for var, ref in node.inputs.items():
                if isinstance(ref, dict) and len(ref) > 0:
                    ref_key, ref_var = next(iter(ref.items()))
                    if ref_key is self:  # INPUT["x"] resolved to father
                        if var not in node.input_schema:
                            raise KeyError(
                                f"Variable not found in input schema: "
                                f"{self.name}:{ref_var} <-- {node.name}.{var}"
                            )
                        input_schema[ref_var] = node.input_schema[var]

            # Check outputs: if ref_key is self (father), it's a graph output
            for var, ref in node.outputs.items():
                if isinstance(ref, dict) and len(ref) > 0:
                    ref_key, ref_var = next(iter(ref.items()))
                    if ref_key is self:  # OUTPUT["x"] resolved to father
                        if var not in node.output_schema:
                            raise KeyError(
                                f"Variable not found in output schema: "
                                f"{self.name}:{ref_var} <-- {node.name}.{var}"
                            )
                        output_schema[ref_var] = node.output_schema[var]

        self.input_schema = input_schema
        self.output_schema = output_schema

    def _build_flow_type(self):
        """Determine flow type of each node based on connection pattern."""
        LOGGER.info(f"Graph '{self.name}': determining node flow types...")
        self.flowtype_map = BiMap[str, NodeFlowType]()

        for name, node in self._nodes.items():
            prev_count = len(self.prevs[name])
            next_count = len(self.nexts[name])

            flow_type: NodeFlowType = "OTHER"

            if node.type == "branch":
                flow_type = "BRANCH"
                for target in node.candidates:
                    if target in self._nodes and len(self.nexts[target]) == 1:
                        self.flowtype_map[target] = "NORMAL"

            elif prev_count > 1 and next_count > 1:
                flow_type = "BLOOM"
            elif prev_count > 1 and next_count == 1:
                flow_type = "MERGE"
            elif prev_count == 1 and next_count > 1:
                flow_type = "FORK"
            elif prev_count == 1 and next_count == 1:
                flow_type = "NORMAL"

            self.flowtype_map[name] = flow_type

    def build(self):
        """Build graph by building child nodes then this graph."""
        for node in self._nodes.values():
            if hasattr(node, 'build'):
                node.build()

        self._setup_schema()
        self._build_flow_type()
        self._setup_endpoints()

        self.ready_count = {
            name: len(preds) for name, preds in self.prevs.items()
        }

        self._is_building = False

    @staticmethod
    def get_current_graph() -> Optional['GraphNode']:
        """Get current graph from context."""
        try:
            return _current_graph.get()
        except LookupError:
            return None

    def add_node(self, node: BaseNode) -> BaseNode:
        """Add a node to the graph."""
        from hush.core.nodes.base import START, END

        if not self._is_building:
            raise RuntimeError("Cannot add nodes after graph has been built")

        if node in [START, END]:
            return node

        self._nodes[node.name] = node

        if hasattr(node, 'start') and node.start:
            if node.name not in self.entries:
                self.entries.append(node.name)

        if hasattr(node, 'end') and node.end:
            if node.name not in self.exits:
                self.exits.append(node.name)

        return node

    def add_edge(self, source: str, target: str, type: EdgeType = "normal"):
        """Add an edge between two nodes."""
        from hush.core.nodes.base import START, END, INPUT, OUTPUT, CONTINUE

        if not self._is_building:
            raise RuntimeError("Cannot add edges after graph has been built!")

        if source == START.name:
            if target not in self._nodes:
                raise ValueError(f"Target node '{target}' not found")

            target_node = self._nodes[target]
            target_node.start = True

            if target not in self.entries:
                self.entries.append(target)

            return

        if target == END.name:
            if source not in self._nodes:
                raise ValueError(f"Source node '{source}' not found")

            source_node = self._nodes[source]
            source_node.end = True

            if source not in self.exits:
                self.exits.append(source)

            return

        if target == CONTINUE.name:
            source_node = self._nodes[source]
            source_node.continue_loop = True
            return

        if target in [INPUT.name, OUTPUT.name]:
            return

        if source not in self._nodes:
            raise ValueError(f"Source node '{source}' not found")
        if target not in self._nodes:
            raise ValueError(f"Target node '{target}' not found")

        new_edge = EdgeConfig(from_node=source, to_node=target, type=type)
        if (source, target) not in self._edges_lookup:
            self._edges.append(new_edge)
            self._edges_lookup[source, target] = new_edge
            self.nexts[source].append(target)
            self.prevs[target].append(source)

    def show(self, indent=0):
        """Display graph structure (debug)."""
        prefix = "  " * indent
        print(f"{prefix}Graph: {self.name}")
        print(f"{prefix}Nodes:", list(self._nodes.keys()))
        print(f"{prefix}Edges:")
        for edge in self._edges:
            print(f"{prefix}  {edge.from_node} -> {edge.to_node}: {edge.type}")

        for node in self._nodes.values():
            if isinstance(node, GraphNode):
                node.show(indent + 1)

    async def run(
        self,
        state: BaseState,
        context_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute graph by running all nodes in dependency order."""

        parent_name = self.father.full_name if self.father else None
        state.record_execution(self.full_name, parent_name, context_id)

        request_id = state.request_id
        start_time = datetime.now()
        _outputs = {}

        try:
            _inputs = self.get_inputs(state, context_id=context_id)

            if self.name != BaseNode.INNER_PROCESS:
                if self.verbose:
                    LOGGER.info("request[%s] - running NODE: %s[%s] (%s), inputs = %s",
                        request_id, self.name, context_id, str(self.type).upper(), str(_inputs)[:200])

            if self._is_building:
                raise ValueError(
                    f"Graph {self.name} not built. "
                    "Must call graph.build() before execution!!"
                )

            active_tasks: Dict[str, asyncio.Task] = {}

            ready_count: Dict[str, int] = self.ready_count.copy()

            for entry in self.entries:
                task = asyncio.create_task(
                    name=entry,
                    coro=self._nodes[entry].run(state, context_id)
                )
                active_tasks[entry] = task

            while active_tasks:
                done_tasks, _ = await asyncio.wait(
                    active_tasks.values(),
                    return_when=asyncio.FIRST_COMPLETED
                )

                for task in done_tasks:
                    node_name = task.get_name()
                    active_tasks.pop(node_name)

                    node = self._nodes[node_name]

                    if node.type == "branch":
                        branch_target = node.get_target(state, context_id)
                        if branch_target != END.name:
                            next_nodes = [branch_target]
                        else:
                            next_nodes = []
                    else:
                        next_nodes = self.nexts[node_name]

                    for next_node in next_nodes:
                        ready_count[next_node] -= 1

                        if ready_count[next_node] == 0:
                            task = asyncio.create_task(
                                name=next_node,
                                coro=self._nodes[next_node].run(state, context_id)
                            )
                            active_tasks[next_node] = task

                    if node.continue_loop:
                        state[self.full_name, "continue_loop", context_id] = True

            _outputs = self.get_outputs(state, context_id=context_id)
            if self.name != BaseNode.INNER_PROCESS:
                if self.verbose:
                    LOGGER.info("request[%s] - running NODE: %s[%s] (%s), outputs = %s",
                        request_id, self.name, context_id, str(self.type).upper(), str(_outputs)[:200])

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            LOGGER.error(f"Error in node {self.name}: {str(e)}")
            LOGGER.error(error_msg)

            state[self.full_name, "error", context_id] = error_msg

        finally:
            end_time = datetime.now()
            state[self.full_name, "start_time", context_id] = start_time
            state[self.full_name, "end_time", context_id] = end_time
            return _outputs


if __name__ == "__main__":
    from hush.core.nodes.base import START, END, INPUT, OUTPUT
    from hush.core.nodes.transform.code_node import CodeNode

    async def main():
        print("=" * 60)
        print("Test 1: Simple linear graph (A -> B -> C)")
        print("=" * 60)

        with GraphNode(name="linear_graph") as graph:
            node_a = CodeNode(
                name="node_a",
                code_fn=lambda x: {"result": x + 10},
                inputs={"x": INPUT["x"]}
            )

            node_b = CodeNode(
                name="node_b",
                code_fn=lambda x: {"result": x * 2},
                inputs={"x": node_a["result"]}
            )

            node_c = CodeNode(
                name="node_c",
                code_fn=lambda x: {"result": x - 5},
                inputs={"x": node_b["result"]},
                outputs=OUTPUT  # graph output = node output
            )

            START >> node_a >> node_b >> node_c >> END

        graph.build()
        graph.show()

        # Create schema from graph (simple!)
        schema = StateSchema(graph)
        schema.show()

        # Create state and run
        state = schema.create_state(inputs={"x": 5})
        result = await graph.run(state)
        print(f"\nInput: x=5")
        print(f"Expected: ((5 + 10) * 2) - 5 = 25")
        print(f"Result: {result}")

        print("\n" + "=" * 60)
        print("Test 2: Parallel graph (A -> [B, C] -> D)")
        print("=" * 60)

        with GraphNode(name="parallel_graph") as graph2:
            node_a = CodeNode(
                name="start_node",
                code_fn=lambda x: {"value": x},
                inputs={"x": INPUT["x"]}
            )

            node_b = CodeNode(
                name="branch_1",
                code_fn=lambda x: {"result": x * 2},
                inputs={"x": node_a["value"]}
            )

            node_c = CodeNode(
                name="branch_2",
                code_fn=lambda x: {"result": x * 3},
                inputs={"x": node_a["value"]}
            )

            node_d = CodeNode(
                name="merge_node",
                code_fn=lambda a, b: {"final": a + b},
                inputs={"a": node_b["result"], "b": node_c["result"]},
                outputs=OUTPUT  # graph output = node output
            )

            START >> node_a >> [node_b, node_c] >> node_d >> END

        graph2.build()
        graph2.show()

        schema2 = StateSchema(graph2)
        schema2.show()

        state2 = schema2.create_state(inputs={"x": 10})
        
        result2 = await graph2.run(state2)
        print(f"\nInput: x=10")
        print(f"Expected: (10*2) + (10*3) = 50")
        print(f"Result: {result2}")

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)

    asyncio.run(main())
