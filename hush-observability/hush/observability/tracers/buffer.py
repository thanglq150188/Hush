"""Async trace buffer for hierarchical trace management.

Based on legacy AsyncTraceBuffer but made backend-agnostic.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Protocol, Set


@dataclass
class TraceItem:
    """Represents a single trace item (span, generation, or event)."""
    type: str  # "span", "generation", "event"
    parent: Optional[str]  # Parent item name, None for root
    data: Dict[str, Any]  # Item-specific data
    trace_metadata: Dict[str, Any]  # Trace-level metadata (user_id, session_id, etc.)


class BackendAdapter(Protocol):
    """Protocol for backend-specific trace creation.

    Each tracer backend (Langfuse, Phoenix, etc.) implements this protocol
    to handle the actual creation of traces in their system.
    """

    def create_span(self, item: TraceItem, parent_obj: Optional[Any]) -> Any:
        """Create a span in the backend system."""
        ...

    def create_generation(self, item: TraceItem, parent_obj: Optional[Any]) -> Any:
        """Create a generation in the backend system."""
        ...

    def end_item(self, obj: Any) -> None:
        """End/finalize an item in the backend system."""
        ...

    def flush_backend(self) -> None:
        """Flush any pending data to the backend."""
        ...


class AsyncTraceBuffer:
    """Async buffer for collecting hierarchical traces before flushing.

    Uses parent-child relationships with name-based keys for hierarchy management.
    Separates trace-level metadata from item data.

    This is backend-agnostic and works with any observability system via
    the BackendAdapter protocol.
    """

    # Trace-level metadata keys that should be handled separately
    TRACE_METADATA_KEYS = {
        "user_id", "session_id", "request_id", "tags",
        "metadata", "release", "version"
    }

    def __init__(self, backend: BackendAdapter):
        """Initialize AsyncTraceBuffer.

        Args:
            backend: Backend adapter for creating traces
        """
        self._backend = backend
        # Structure: {request_id: {item_name: TraceItem}}
        self._buffer: Dict[str, Dict[str, TraceItem]] = {}
        self._lock = asyncio.Lock()

    def _separate_trace_metadata(
        self,
        kwargs: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Separate trace-level metadata from item data.

        Args:
            kwargs: Combined parameters

        Returns:
            Tuple of (item_data, trace_metadata)
        """
        item_data = {}
        trace_metadata = {}

        for key, value in kwargs.items():
            if key in self.TRACE_METADATA_KEYS:
                trace_metadata[key] = value
            else:
                item_data[key] = value

        return item_data, trace_metadata

    def add_span(
        self,
        request_id: str,
        name: str,
        parent: Optional[str] = None,
        **kwargs
    ) -> None:
        """Add a span to the buffer."""
        item_data, trace_metadata = self._separate_trace_metadata(kwargs)
        self._add_item(request_id, name, "span", parent, item_data, trace_metadata)

    def add_generation(
        self,
        request_id: str,
        name: str,
        parent: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> None:
        """Add a generation to the buffer."""
        item_data, trace_metadata = self._separate_trace_metadata(kwargs)
        if model:
            item_data["model"] = model

        self._add_item(request_id, name, "generation", parent, item_data, trace_metadata)

    def add_event(
        self,
        request_id: str,
        name: str,
        parent: Optional[str] = None,
        **kwargs
    ) -> None:
        """Add an event to the buffer."""
        item_data, trace_metadata = self._separate_trace_metadata(kwargs)
        self._add_item(request_id, name, "event", parent, item_data, trace_metadata)

    def _add_item(
        self,
        request_id: str,
        name: str,
        item_type: str,
        parent: Optional[str],
        item_data: Dict[str, Any],
        trace_metadata: Dict[str, Any]
    ) -> None:
        """Internal method to add any type of item."""
        if request_id not in self._buffer:
            self._buffer[request_id] = {}

        # Ensure name is included in item data
        if "name" not in item_data:
            item_data["name"] = name

        self._buffer[request_id][name] = TraceItem(
            type=item_type,
            parent=parent,
            data=item_data,
            trace_metadata=trace_metadata
        )

    def update_item(
        self,
        request_id: str,
        name: str,
        **kwargs
    ) -> bool:
        """Update an existing item in the buffer."""
        if request_id in self._buffer and name in self._buffer[request_id]:
            item_data, trace_metadata = self._separate_trace_metadata(kwargs)

            # Update item data
            self._buffer[request_id][name].data.update(item_data)

            # Update trace metadata
            self._buffer[request_id][name].trace_metadata.update(trace_metadata)

            return True
        return False

    def _build_hierarchy_order(self, items: Dict[str, TraceItem]) -> List[str]:
        """Build processing order based on parent-child dependencies.

        Parents must be processed before their children.

        Args:
            items: Dictionary of items for a request

        Returns:
            List of item names in dependency order

        Raises:
            ValueError: If circular dependencies or orphaned items detected
        """
        # Find all root items (no parent)
        roots = [name for name, item in items.items() if item.parent is None]

        if not roots:
            raise ValueError("No root items found - at least one item must have parent=None")

        ordered = []
        visited = set()
        visiting: Set[str] = set()  # For cycle detection

        def visit(name: str) -> None:
            if name in visiting:
                raise ValueError(f"Circular dependency detected involving '{name}'")
            if name in visited:
                return
            if name not in items:
                raise ValueError(f"Parent '{name}' referenced but not found in items")

            visiting.add(name)

            # Visit all children of this item
            children = [
                child_name for child_name, child_item in items.items()
                if child_item.parent == name
            ]

            for child in children:
                visit(child)

            visiting.remove(name)
            visited.add(name)
            ordered.append(name)

        # Process all roots and their descendants
        for root in roots:
            visit(root)

        # Ensure all items were processed
        unprocessed = set(items.keys()) - visited
        if unprocessed:
            raise ValueError(f"Orphaned items detected (no path from root): {unprocessed}")

        # Reverse to get parents-first order
        return ordered[::-1]

    def clear_request(self, request_id: str) -> bool:
        """Clear all data for a specific request without flushing."""
        if request_id in self._buffer:
            del self._buffer[request_id]
            return True
        return False

    async def flush(self, request_id: str) -> bool:
        """Flush all buffered items for a request to the backend."""
        async with self._lock:
            if request_id not in self._buffer:
                return False

            items = self._buffer[request_id].copy()

            # Validate and get processing order
            try:
                ordered_names = self._build_hierarchy_order(items)
            except ValueError as e:
                raise ValueError(f"Cannot flush request {request_id}: {e}")

            # Track created backend objects for parent-child linking
            backend_objects = {}  # name -> backend object

            try:
                # Create all items in order
                for name in ordered_names:
                    item = items[name]

                    # Get parent object if exists
                    parent_obj = None
                    if item.parent and item.parent in backend_objects:
                        parent_obj = backend_objects[item.parent]

                    # Create item using backend adapter
                    if item.type == "span":
                        backend_obj = self._backend.create_span(item, parent_obj)
                    elif item.type == "generation":
                        backend_obj = self._backend.create_generation(item, parent_obj)
                    elif item.type == "event":
                        # Events might not return objects in all backends
                        backend_obj = None
                    else:
                        raise ValueError(f"Unknown item type: {item.type}")

                    if backend_obj:
                        backend_objects[name] = backend_obj

                # End all items (children first)
                for name in reversed(ordered_names):
                    if name in backend_objects:
                        self._backend.end_item(backend_objects[name])

                # Flush backend
                self._backend.flush_backend()

                # Remove from buffer after successful flush
                del self._buffer[request_id]
                return True

            except Exception as e:
                # If flush fails, keep items in buffer for potential retry
                raise Exception(f"Failed to flush request {request_id}: {e}")

    async def flush_all(self) -> Dict[str, bool]:
        """Flush all buffered requests to the backend."""
        results = {}

        async with self._lock:
            request_ids = list(self._buffer.keys())

        for request_id in request_ids:
            try:
                results[request_id] = await self.flush(request_id)
            except Exception as e:
                print(f"Failed to flush request {request_id}: {e}")
                results[request_id] = False

        return results
