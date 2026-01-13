"""Abstract base tracer for workflow tracing.

This module provides the BaseTracer abstract class that all concrete tracer
implementations must inherit from. Tracers are responsible for collecting
and exporting workflow execution traces to observability platforms.

Example:
    ```python
    from hush.core.tracers import BaseTracer, register_tracer

    @register_tracer
    class MyTracer(BaseTracer):
        def _get_tracer_config(self) -> Dict[str, Any]:
            return {"api_key": self.api_key}

        @staticmethod
        def flush(flush_data: Dict[str, Any]) -> None:
            # Send traces to your platform
            pass
    ```
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TYPE_CHECKING
import multiprocessing

if TYPE_CHECKING:
    from hush.core.states import MemoryState


# Global worker process and queue
_flush_queue: Optional[multiprocessing.Queue] = None
_worker_process: Optional[multiprocessing.Process] = None


def _flush_worker(queue: multiprocessing.Queue, config_path: Optional[str], dotenv_path: Optional[str] = None) -> None:
    """Worker process that initializes ResourceHub and processes flush requests.

    This worker stays alive and processes flush requests from the queue.
    ResourceHub is initialized once when the worker starts.

    Args:
        queue: Queue to receive flush requests
        config_path: Absolute path to resources.yaml (from main process)
        dotenv_path: Absolute path to .env file (from main process)
    """
    import signal
    import os

    # Ignore SIGINT in worker - let main process handle it
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Load .env file if provided
    if dotenv_path:
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path)
        except ImportError:
            pass

    # Set HUSH_CONFIG env var so get_hub() finds the right config
    if config_path:
        os.environ['HUSH_CONFIG'] = config_path

    # Initialize ResourceHub once in this process
    try:
        from hush.core.registry import get_hub
        get_hub()  # This loads resources.yaml from HUSH_CONFIG
    except Exception as e:
        print(f"[FlushWorker] Failed to initialize ResourceHub: {e}")

    while True:
        try:
            flush_data = queue.get()

            # Sentinel value to stop worker
            if flush_data is None:
                break

            _dispatch_flush(flush_data)

        except Exception as e:
            import traceback
            print(f"[FlushWorker] Error processing flush: {e}\n{traceback.format_exc()}")


class BaseTracer(ABC):
    """Abstract base class for workflow tracers.

    Subclasses must implement:
        - flush(): The actual tracing logic that runs in a subprocess
        - _get_tracer_config(): Returns tracer-specific config for serialization

    The tracer uses a persistent worker process with ResourceHub initialized.
    Flush requests are sent via a queue and processed asynchronously.
    """

    @classmethod
    def _ensure_worker(cls) -> multiprocessing.Queue:
        """Ensure the flush worker process is running and return the queue."""
        global _flush_queue, _worker_process

        if _worker_process is None or not _worker_process.is_alive():
            import os
            from pathlib import Path

            # Get config path from current hub's storage
            config_path = None
            try:
                from hush.core.registry import get_hub
                hub = get_hub()
                if hasattr(hub._storage, '_file_path'):
                    config_path = str(hub._storage._file_path.resolve())
            except Exception:
                pass

            # Find .env file near config or in current directory
            dotenv_path = None
            if config_path:
                env_file = Path(config_path).parent / ".env"
                if env_file.exists():
                    dotenv_path = str(env_file.resolve())
            if not dotenv_path:
                cwd_env = Path.cwd() / ".env"
                if cwd_env.exists():
                    dotenv_path = str(cwd_env.resolve())

            ctx = multiprocessing.get_context("spawn")
            _flush_queue = ctx.Queue()
            _worker_process = ctx.Process(
                target=_flush_worker,
                args=(_flush_queue, config_path, dotenv_path),
                daemon=True,
            )
            _worker_process.start()

        return _flush_queue

    @classmethod
    def shutdown_executor(cls) -> None:
        """Shutdown the flush worker process."""
        global _flush_queue, _worker_process

        if _worker_process is not None and _worker_process.is_alive():
            # Send sentinel to stop worker
            _flush_queue.put(None)
            _worker_process.join(timeout=5)

            if _worker_process.is_alive():
                _worker_process.terminate()

        _flush_queue = None
        _worker_process = None

    @abstractmethod
    def _get_tracer_config(self) -> Dict[str, Any]:
        """Return tracer-specific configuration for serialization.

        This config will be passed to the subprocess flush function.

        Returns:
            Dictionary containing tracer configuration
        """
        pass

    def prepare_flush_data(
        self,
        workflow_name: str,
        state: 'MemoryState',
    ) -> Dict[str, Any]:
        """Prepare serializable data for the subprocess.

        This method reads trace metadata and values from state to build
        the flush_data dictionary. Input/output values are read directly
        from state cells (no duplication in trace_metadata).

        Args:
            workflow_name: Name of the workflow
            state: MemoryState object containing execution data

        Returns:
            Dictionary containing all data needed for flushing
        """
        flush_data = {
            "tracer_type": self.__class__.__name__,
            "tracer_config": self._get_tracer_config(),
            "workflow_name": workflow_name,
            "request_id": state.request_id,
            "user_id": state.user_id,
            "session_id": state.session_id,
            "execution_order": [],
            "nodes_trace_data": {},
        }

        for execution in state.execution_order:
            node_name = execution["node"]
            parent_name = execution["parent"]
            context_id = execution["context_id"]

            # Build key for trace_metadata lookup
            key = f"{node_name}:{context_id}" if context_id else node_name
            metadata = state._trace_metadata.get(key, {})

            # Build input dict from state cells
            input_data = {}
            for var in metadata.get("input_vars", []):
                input_data[var] = state.get(node_name, var, context_id)

            # Build output dict from state cells
            output_data = {}
            for var in metadata.get("output_vars", []):
                output_data[var] = state.get(node_name, var, context_id)

            # Get start/end times from state
            start_time = state.get(node_name, "start_time", context_id)
            end_time = state.get(node_name, "end_time", context_id)

            # Build trace_data for this node
            trace_data = {
                "name": metadata.get("name", node_name),
                "start_time": start_time,
                "end_time": end_time,
                "input": input_data,
                "output": output_data,
                "model": metadata.get("model"),
                "usage": metadata.get("usage"),
                "cost": metadata.get("cost"),
                "metadata": metadata.get("metadata", {}),
            }

            flush_data["nodes_trace_data"][key] = trace_data
            flush_data["execution_order"].append({
                "node": node_name,
                "parent": parent_name,
                "context_id": context_id,
                "contain_generation": metadata.get("contain_generation", False),
            })

        return flush_data

    def flush_in_background(self, workflow_name: str, state: 'MemoryState') -> None:
        """Submit flush task to worker process without blocking the main flow.

        Args:
            workflow_name: Name of the workflow
            state: MemoryState object containing execution data
        """
        from hush.core.loggings import LOGGER

        try:
            flush_data = self.prepare_flush_data(workflow_name, state)
            queue = self._ensure_worker()
            queue.put(flush_data)

        except Exception as e:
            import traceback
            LOGGER.error(
                "Workflow: %s, Request ID: %s, Failed to submit flush task: %s\nTraceback:\n%s",
                workflow_name,
                state.request_id,
                str(e),
                traceback.format_exc()
            )

    @staticmethod
    @abstractmethod
    def flush(flush_data: Dict[str, Any]) -> None:
        """Execute the flush logic in a subprocess.

        This method runs in a separate process, so it must:
        - Re-import all dependencies
        - Use only the data provided in flush_data
        - Not access any instance attributes

        Args:
            flush_data: Dictionary containing all data needed for flushing
        """
        pass


# Registry of tracer types for subprocess dispatch
_TRACER_REGISTRY: Dict[str, type] = {}


def register_tracer(tracer_cls: type) -> type:
    """Decorator to register a tracer class for subprocess dispatch.

    Args:
        tracer_cls: The tracer class to register

    Returns:
        The registered tracer class

    Example:
        ```python
        @register_tracer
        class MyTracer(BaseTracer):
            ...
        ```
    """
    _TRACER_REGISTRY[tracer_cls.__name__] = tracer_cls
    return tracer_cls


def get_registered_tracers() -> Dict[str, type]:
    """Get all registered tracer classes.

    Returns:
        Dictionary mapping tracer names to their classes
    """
    return _TRACER_REGISTRY.copy()


def _dispatch_flush(flush_data: Dict[str, Any]) -> None:
    """Dispatch flush to the appropriate tracer based on tracer_type.

    This function runs in the subprocess and routes the flush_data
    to the correct tracer implementation.

    Args:
        flush_data: Dictionary containing tracer_type and flush data
    """
    from hush.core.loggings import LOGGER

    tracer_type = flush_data.get("tracer_type")

    if tracer_type not in _TRACER_REGISTRY:
        # Try to import common tracer implementations to populate registry
        try:
            from hush.observability.langfuse import LangfuseTracer  # noqa: F401
        except ImportError:
            pass

    tracer_cls = _TRACER_REGISTRY.get(tracer_type)
    if tracer_cls is None:
        LOGGER.error(f"Unknown tracer type: {tracer_type}")
        return

    tracer_cls.flush(flush_data)