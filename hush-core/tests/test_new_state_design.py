"""Test new StateSchema + BaseState design."""

import time
import concurrent.futures
from hush.core.states import StateSchema, MemoryState, RedisState, BaseState


def test_schema_basic():
    """Test basic schema operations."""
    schema = StateSchema("test_workflow")

    # Chain methods
    schema.link("llm", "messages", "prompt", "output")
    schema.link("parser", "text", "llm", "content")
    schema.set("llm", "temperature", 0.7)
    schema.set("llm", "model", "gpt-4o")

    # Check contains
    assert ("llm", "messages") in schema
    assert ("llm", "temperature") in schema

    # Check resolve
    assert schema.resolve("llm", "messages") == ("prompt", "output")
    assert schema.resolve("prompt", "output") == ("prompt", "output")  # No redirect

    # Check get default
    assert schema.get("llm", "temperature") == 0.7

    print("✓ Schema basic operations work")


def test_memory_state():
    """Test MemoryState with schema."""
    schema = StateSchema("test_workflow")
    schema.link("llm", "messages", "prompt", "output")
    schema.set("llm", "temperature", 0.7)

    state = schema.create_state(inputs={"query": "hello"})

    # Check type
    assert isinstance(state, MemoryState)
    assert isinstance(state, BaseState)

    # Check defaults applied
    assert state["llm", "temperature", None] == 0.7

    # Check inputs applied
    assert state["test_workflow", "query", None] == "hello"

    # Check redirect works
    state["prompt", "output", None] = "Hello World"
    assert state["llm", "messages", None] == "Hello World"  # Via redirect

    # Check properties
    assert state.name == "test_workflow"
    assert state.user_id is not None
    assert state.session_id is not None
    assert state.request_id is not None

    print("✓ MemoryState works correctly")


def test_multiple_contexts():
    """Test state with multiple contexts (loop iterations)."""
    schema = StateSchema("test_workflow")
    schema.link("llm", "messages", "prompt", "output")

    state = schema.create_state()

    # Set values in different contexts
    state["prompt", "output", "loop_1"] = "First"
    state["prompt", "output", "loop_2"] = "Second"
    state["prompt", "output", "loop_3"] = "Third"

    # Check values via redirect
    assert state["llm", "messages", "loop_1"] == "First"
    assert state["llm", "messages", "loop_2"] == "Second"
    assert state["llm", "messages", "loop_3"] == "Third"

    print("✓ Multiple contexts work correctly")


def test_schema_create_state_backends():
    """Test create_state with different backends."""
    schema = StateSchema("test_workflow")
    schema.link("llm", "messages", "prompt", "output")

    # Default (MemoryState)
    state1 = schema.create_state()
    assert isinstance(state1, MemoryState)

    # Explicit MemoryState
    state2 = schema.create_state(state_class=MemoryState)
    assert isinstance(state2, MemoryState)

    # Different request_ids
    assert state1.request_id != state2.request_id

    print("✓ create_state with different backends works")


def test_performance():
    """Benchmark state creation."""
    schema = StateSchema("benchmark_workflow")

    # Setup schema with 1000 vars
    for i in range(1000):
        node = f"node_{i}"
        schema.set(node, "output", None)
        if i > 0 and i % 2 == 0:
            schema.link(node, "input", f"node_{i-1}", "output")

    # Warmup
    for _ in range(10):
        _ = schema.create_state()

    # Benchmark
    iterations = 1000
    start = time.perf_counter()
    for _ in range(iterations):
        state = schema.create_state(inputs={"query": "test"})
    elapsed = time.perf_counter() - start

    avg_us = (elapsed / iterations) * 1_000_000
    print(f"✓ Performance: {avg_us:.2f} µs per state creation (1000 vars)")


def test_concurrent_creation():
    """Test concurrent state creation."""
    schema = StateSchema("benchmark_workflow")

    for i in range(1000):
        node = f"node_{i}"
        schema.set(node, "output", None)
        if i > 0 and i % 2 == 0:
            schema.link(node, "input", f"node_{i-1}", "output")

    n_ccu = 1000

    def do_create(request_id):
        start = time.perf_counter()
        state = schema.create_state(inputs={"query": f"request_{request_id}"})
        elapsed = time.perf_counter() - start
        return elapsed

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(do_create, i) for i in range(n_ccu)]
        times = [f.result() for f in concurrent.futures.as_completed(futures)]
    concurrent_time = time.perf_counter() - start

    avg_time = sum(times) / len(times) * 1_000_000
    print(f"✓ Concurrent: {n_ccu} states in {concurrent_time*1000:.2f} ms (avg {avg_time:.2f} µs)")


def test_show_methods():
    """Test debug show methods."""
    schema = StateSchema("test_workflow")
    schema.link("llm", "messages", "prompt", "output")
    schema.link("parser", "text", "llm", "content")
    schema.set("llm", "temperature", 0.7)
    schema.set("prompt", "template", "Hello {name}")

    print("\n--- Schema Debug ---")
    schema.show()
    schema.show_links()
    schema.show_defaults()

    state = schema.create_state(inputs={"query": "hello"})
    state["prompt", "output", None] = "Hello World"
    state["llm", "content", None] = "Response"

    print("\n--- State Debug ---")
    state.show()

    print("✓ Show methods work")


def test_read_write_performance():
    """Benchmark read/write operations with millions of keys."""
    schema = StateSchema("benchmark_rw")
    state = schema.create_state()

    # Write 1 million keys
    n_keys = 1_000_000
    print(f"\n--- Write {n_keys:,} keys ---")

    start = time.perf_counter()
    for i in range(n_keys):
        state._data[(f"node_{i % 1000}", f"var_{i % 100}", "main")] = i
    write_time = time.perf_counter() - start
    write_ops_per_sec = n_keys / write_time

    print(f"Write: {write_time*1000:.2f} ms ({write_ops_per_sec/1_000_000:.2f}M ops/sec)")

    # Read 1 million keys (existing)
    print(f"\n--- Read {n_keys:,} existing keys ---")

    start = time.perf_counter()
    for i in range(n_keys):
        _ = state._data.get((f"node_{i % 1000}", f"var_{i % 100}", "main"))
    read_time = time.perf_counter() - start
    read_ops_per_sec = n_keys / read_time

    print(f"Read: {read_time*1000:.2f} ms ({read_ops_per_sec/1_000_000:.2f}M ops/sec)")

    # Read via __getitem__ (with redirect)
    print(f"\n--- Read {n_keys:,} via __getitem__ (with redirect) ---")

    # Add some links to test redirect
    for i in range(100):
        schema.link(f"alias_{i}", "value", f"node_{i}", f"var_{i}")

    start = time.perf_counter()
    for i in range(n_keys):
        _ = state[f"node_{i % 1000}", f"var_{i % 100}", "main"]
    read_redirect_time = time.perf_counter() - start
    read_redirect_ops_per_sec = n_keys / read_redirect_time

    print(f"Read (redirect): {read_redirect_time*1000:.2f} ms ({read_redirect_ops_per_sec/1_000_000:.2f}M ops/sec)")

    # Write via __setitem__ (with redirect)
    print(f"\n--- Write {n_keys:,} via __setitem__ (with redirect) ---")

    start = time.perf_counter()
    for i in range(n_keys):
        state[f"node_{i % 1000}", f"var_{i % 100}", "main"] = i * 2
    write_redirect_time = time.perf_counter() - start
    write_redirect_ops_per_sec = n_keys / write_redirect_time

    print(f"Write (redirect): {write_redirect_time*1000:.2f} ms ({write_redirect_ops_per_sec/1_000_000:.2f}M ops/sec)")

    print(f"\n✓ Read/Write performance test complete")
    print(f"  Direct dict: {read_ops_per_sec/1_000_000:.2f}M read, {write_ops_per_sec/1_000_000:.2f}M write ops/sec")
    print(f"  Via state[]: {read_redirect_ops_per_sec/1_000_000:.2f}M read, {write_redirect_ops_per_sec/1_000_000:.2f}M write ops/sec")


if __name__ == "__main__":
    print("=" * 60)
    print("New State Design Test")
    print("=" * 60)

    test_schema_basic()
    test_memory_state()
    test_multiple_contexts()
    test_schema_create_state_backends()
    test_performance()
    test_concurrent_creation()
    test_show_methods()
    test_read_write_performance()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
