"""Graph node for managing subgraphs of workflow nodes."""

from datetime import datetime
from time import perf_counter
from typing import Dict, List, Literal, Any, Optional
from collections import defaultdict
import asyncio
import traceback

from hush.core.configs.edge_config import EdgeConfig, EdgeType
from hush.core.configs.node_config import NodeType
from hush.core.nodes.base import BaseNode, START, END, PARENT
from hush.core.states import BaseState, StateSchema, Ref
from hush.core.utils.context import _current_graph
from hush.core.utils.bimap import BiMap
from hush.core.utils.common import Param
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
        LOGGER.debug(f"Graph '{self.name}': initializing endpoints...")

        if not self.entries:
            self.entries = [node for node in self._nodes if not self.prevs[node]]

        if not self.exits:
            self.exits = [node for node in self._nodes if not self.nexts[node]]

        if not self.entries:
            LOGGER.error(f"Graph '{self.name}': no entry nodes found. Check START >> node connections.")
            raise ValueError("Graph must have at least one entry node.")
        if not self.exits:
            LOGGER.error(f"Graph '{self.name}': no exit nodes found. Check node >> END connections.")
            raise ValueError("Graph must have at least one exit node.")

    def _setup_schema(self):
        """Initialize input/output schema and inputs dictionary."""
        LOGGER.debug(f"Graph '{self.name}': creating schema...")
        input_schema = {}
        output_schema = {}

        for _, node in self._nodes.items():
            # Check inputs: if ref points to self (father), it's a graph input
            for var, ref in node.inputs.items():
                if isinstance(ref, Ref) and ref.raw_node is self:
                    # PARENT["x"] resolved to father - this is a graph input
                    if var not in node.input_schema:
                        LOGGER.error(
                            f"Graph '{self.name}': variable '{var}' not found in input schema of node '{node.name}'"
                        )
                        raise KeyError(
                            f"Variable not found in input schema: "
                            f"{self.name}:{ref.var} <-- {node.name}.{var}"
                        )
                    input_schema[ref.var] = node.input_schema[var]

            # Check outputs: if ref points to self (father), it's a graph output
            for var, ref in node.outputs.items():
                if isinstance(ref, Ref) and ref.raw_node is self:
                    # PARENT["x"] resolved to father - this is a graph output
                    if var not in node.output_schema:
                        LOGGER.error(
                            f"Graph '{self.name}': variable '{var}' not found in output schema of node '{node.name}'"
                        )
                        raise KeyError(
                            f"Variable not found in output schema: "
                            f"{self.name}:{ref.var} <-- {node.name}.{var}"
                        )
                    output_schema[ref.var] = node.output_schema[var]

        self.input_schema = input_schema
        self.output_schema = output_schema
        # Graph inputs are direct values from caller, not references
        self.inputs = {}

    def get_inputs(self, state: 'BaseState', context_id: str) -> Dict[str, Any]:
        """Get graph inputs directly from state.

        Unlike regular nodes that read from referenced nodes, graph inputs
        are stored directly at (graph.full_name, var_name, context_id).
        """
        return {
            var_name: state._get_value(self.full_name, var_name, context_id)
            for var_name in self.input_schema
        }

    def get_outputs(self, state: 'BaseState', context_id: str) -> Dict[str, Any]:
        """Get graph outputs using schema links.

        Graph outputs are aliases to inner node outputs (e.g., graph.result -> inner_node.result).
        We MUST follow schema links here to read from the inner node's output location.
        """
        return {
            var_name: state[self.full_name, var_name, context_id]
            for var_name in self.output_schema
        }

    def _build_flow_type(self):
        """Determine flow type of each node based on connection pattern."""
        LOGGER.debug(f"Graph '{self.name}': determining node flow types...")
        self.flowtype_map = BiMap[str, NodeFlowType]()

        # Detect orphan nodes (no connections at all)
        orphan_nodes = []

        for name, node in self._nodes.items():
            prev_count = len(self.prevs[name])
            next_count = len(self.nexts[name])

            # Check for orphan nodes (not start/end, not inner graph, and no connections)
            if prev_count == 0 and next_count == 0 and not node.start and not node.end and name != BaseNode.INNER_PROCESS:
                orphan_nodes.append(name)

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

        # Warn about orphan nodes
        if orphan_nodes:
            LOGGER.warning(
                f"Graph '{self.full_name}': orphan nodes detected (no edges): {orphan_nodes}. "
                "These nodes will never be executed."
            )

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
        if not self._is_building:
            raise RuntimeError("Cannot add nodes after graph has been built")

        if node in [START, END]:
            return node

        # Warn if node with same name already exists (will be overwritten)
        if node.name in self._nodes:
            LOGGER.warning(
                f"Graph '{self.name}': node '{node.name}' already exists and will be overwritten"
            )

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

        if target == PARENT.name:
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
        perf_start = perf_counter()
        _inputs = {}
        _outputs = {}

        try:
            _inputs = self.get_inputs(state, context_id=context_id)

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

            _outputs = self.get_outputs(state, context_id=context_id)

        except Exception as e:
            error_msg = traceback.format_exc()
            LOGGER.error(f"Error in node {self.name}: {str(e)}")
            LOGGER.error(error_msg)
            state[self.full_name, "error", context_id] = error_msg

        finally:
            end_time = datetime.now()
            duration_ms = (perf_counter() - perf_start) * 1000
            self._log(request_id, context_id, _inputs, _outputs, duration_ms)
            state[self.full_name, "start_time", context_id] = start_time
            state[self.full_name, "end_time", context_id] = end_time
            return _outputs


if __name__ == "__main__":
    from hush.core.nodes.transform.code_node import CodeNode

    async def main():
        print("=" * 60)
        print("Test 1: Simple linear graph (A -> B -> C)")
        print("=" * 60)

        with GraphNode(name="linear_graph") as graph:
            node_a = CodeNode(
                name="node_a",
                code_fn=lambda x: {"result": x + 10},
                inputs={"x": PARENT["x"]}
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
                outputs={"result": PARENT["result"]}  # graph output = node output
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

        state.show()

        # =================================================================
        # Test 2: Ref with operations (new Ref feature)
        # =================================================================
        print("\n" + "=" * 60)
        print("Test 2: Ref with operations (getitem, apply)")
        print("=" * 60)

        with GraphNode(name="ref_ops_graph") as graph2:
            # Node A returns a dict with nested data
            node_a = CodeNode(
                name="data_source",
                code_fn=lambda: {"data": {"items": [1, 2, 3, 4, 5], "name": "test"}},
                inputs={}
            )

            # Node B uses Ref with getitem to extract nested value
            node_b = CodeNode(
                name="extract_items",
                code_fn=lambda items: {"count": len(items), "sum": sum(items)},
                inputs={"items": node_a["data"]["items"]}  # Ref with chained getitem
            )

            # Node C uses Ref with apply to transform data
            node_c = CodeNode(
                name="transform_name",
                code_fn=lambda name: {"upper_name": name},
                inputs={"name": node_a["data"]["name"].upper()}  # Ref with method call
            )

            # Node D merges results
            node_d = CodeNode(
                name="merge_results",
                code_fn=lambda count, total, name: {"result": f"{name}: {count} items, sum={total}"},
                inputs={
                    "count": node_b["count"],
                    "total": node_b["sum"],
                    "name": node_c["upper_name"]
                },
                outputs={"result": PARENT["result"]}
            )

            START >> node_a >> [node_b, node_c] >> node_d >> END

        graph2.build()

        schema2 = StateSchema(graph2)
        state2 = schema2.create_state()

        result2 = await graph2.run(state2)
        print(f"Result: {result2}")
        print(f"Expected: {{'result': 'TEST: 5 items, sum=15'}}")
        assert result2["result"] == "TEST: 5 items, sum=15", f"Got: {result2}"

        # =================================================================
        # Test 3: Ref with apply() for function application
        # =================================================================
        print("\n" + "=" * 60)
        print("Test 3: Ref with apply() for function application")
        print("=" * 60)

        with GraphNode(name="ref_apply_graph") as graph3:
            # Node A returns a list
            node_a = CodeNode(
                name="list_source",
                code_fn=lambda: {"numbers": [5, 2, 8, 1, 9, 3]},
                inputs={}
            )

            # Node B uses Ref.apply(len) to get length
            node_b = CodeNode(
                name="get_length",
                code_fn=lambda length: {"length": length},
                inputs={"length": node_a["numbers"].apply(len)}
            )

            # Node C uses Ref.apply(sorted) to sort
            node_c = CodeNode(
                name="sort_numbers",
                code_fn=lambda sorted_nums: {"sorted": sorted_nums},
                inputs={"sorted_nums": node_a["numbers"].apply(sorted)}
            )

            # Node D uses Ref.apply(sum) to get sum
            node_d = CodeNode(
                name="sum_numbers",
                code_fn=lambda total: {"total": total},
                inputs={"total": node_a["numbers"].apply(sum)}
            )

            # Node E merges all results
            node_e = CodeNode(
                name="merge_all",
                code_fn=lambda length, sorted_nums, total: {
                    "length": length,
                    "sorted": sorted_nums,
                    "total": total
                },
                inputs={
                    "length": node_b["length"],
                    "sorted_nums": node_c["sorted"],
                    "total": node_d["total"]
                },
                outputs=PARENT
            )

            START >> node_a >> [node_b, node_c, node_d] >> node_e >> END

        graph3.build()

        schema3 = StateSchema(graph3)
        state3 = schema3.create_state()

        result3 = await graph3.run(state3)
        print(f"Result: {result3}")
        print(f"Expected length: 6, sorted: [1, 2, 3, 5, 8, 9], total: 28")
        assert result3["length"] == 6
        assert result3["sorted"] == [1, 2, 3, 5, 8, 9]
        assert result3["total"] == 28

        # =================================================================
        # Test 4: Ref with chained operations
        # =================================================================
        print("\n" + "=" * 60)
        print("Test 4: Ref with chained operations")
        print("=" * 60)

        with GraphNode(name="ref_chain_graph") as graph4:
            node_a = CodeNode(
                name="data_source",
                code_fn=lambda: {"data": {"users": [{"name": "alice"}, {"name": "bob"}]}},
                inputs={}
            )

            # Chain: data["users"][0]["name"].upper()
            node_b = CodeNode(
                name="get_first_user_upper",
                code_fn=lambda name: {"first_user": name},
                inputs={"name": node_a["data"]["users"][0]["name"].upper()},
                outputs=PARENT
            )

            START >> node_a >> node_b >> END

        graph4.build()

        schema4 = StateSchema(graph4)
        state4 = schema4.create_state()

        result4 = await graph4.run(state4)
        print(f"Result: {result4}")
        print(f"Expected: {{'first_user': 'ALICE'}}")
        assert result4["first_user"] == "ALICE"

        # =================================================================
        # Test 5: Ref with arithmetic operations
        # =================================================================
        print("\n" + "=" * 60)
        print("Test 5: Ref with arithmetic operations")
        print("=" * 60)

        with GraphNode(name="ref_arithmetic_graph") as graph5:
            node_a = CodeNode(
                name="number_source",
                code_fn=lambda: {"value": 10},
                inputs={}
            )

            # Use arithmetic: (value + 5) * 2
            node_b = CodeNode(
                name="compute",
                code_fn=lambda result: {"result": result},
                inputs={"result": (node_a["value"] + 5) * 2},
                outputs=PARENT
            )

            START >> node_a >> node_b >> END

        graph5.build()

        schema5 = StateSchema(graph5)
        state5 = schema5.create_state()

        result5 = await graph5.run(state5)
        print(f"Result: {result5}")
        print(f"Expected: {{'result': 30}}  # (10 + 5) * 2")
        assert result5["result"] == 30

        print("\n" + "=" * 60)
        print("All Ref operation tests passed!")
        print("=" * 60)

        # print("\n" + "=" * 60)
        # print("Test 2: Parallel graph (A -> [B, C] -> D)")
        # print("=" * 60)

        # with GraphNode(name="parallel_graph") as graph2:
        #     node_a = CodeNode(
        #         name="start_node",
        #         code_fn=lambda x: {"value": x},
        #         inputs={"x": PARENT["x"]}
        #     )

        #     node_b = CodeNode(
        #         name="branch_1",
        #         code_fn=lambda x: {"result": x * 2},
        #         inputs={"x": node_a["value"]}
        #     )

        #     node_c = CodeNode(
        #         name="branch_2",
        #         code_fn=lambda x: {"result": x * 3},
        #         inputs={"x": node_a["value"]}
        #     )

        #     node_d = CodeNode(
        #         name="merge_node",
        #         code_fn=lambda a, b: {"final": a + b},
        #         inputs={"a": node_b["result"], "b": node_c["result"]},
        #         outputs=PARENT  # graph output = node output
        #     )

        #     START >> node_a >> [node_b, node_c] >> node_d >> END

        # graph2.build()
        # graph2.show()

        # schema2 = StateSchema(graph2)
        # schema2.show()

        # state2 = schema2.create_state(inputs={"x": 10})
        
        # result2 = await graph2.run(state2)
        # print(f"\nInput: x=10")
        # print(f"Expected: (10*2) + (10*3) = 50")
        # print(f"Result: {result2}")

        # print("\n" + "=" * 60)
        # print("All tests passed!")
        # print("=" * 60)

    asyncio.run(main())
