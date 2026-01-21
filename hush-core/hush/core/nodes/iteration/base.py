"""Base utilities cho các iteration nodes."""

from typing import Dict, Any, List


def calculate_iteration_metrics(latencies_ms: List[float]) -> Dict[str, Any]:
    """Tính aggregate metrics từ iteration latencies.

    Standalone utility function để các iteration nodes có thể import và sử dụng.

    Args:
        latencies_ms: List các giá trị latency tính bằng milliseconds

    Returns:
        Dictionary chứa các thống kê latency
    """
    if not latencies_ms:
        return {
            "latency_avg_ms": 0.0,
            "latency_min_ms": 0.0,
            "latency_max_ms": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_p99_ms": 0.0,
        }

    sorted_latencies = sorted(latencies_ms)
    n = len(sorted_latencies)

    def percentile(p: float) -> float:
        """Tính giá trị percentile."""
        if n == 1:
            return sorted_latencies[0]
        k = (n - 1) * p
        f = int(k)
        c = f + 1 if f + 1 < n else f
        return sorted_latencies[f] + (k - f) * (sorted_latencies[c] - sorted_latencies[f])

    return {
        "latency_avg_ms": round(sum(latencies_ms) / n, 2),
        "latency_min_ms": round(sorted_latencies[0], 2),
        "latency_max_ms": round(sorted_latencies[-1], 2),
        "latency_p50_ms": round(percentile(0.50), 2),
        "latency_p95_ms": round(percentile(0.95), 2),
        "latency_p99_ms": round(percentile(0.99), 2),
    }


class Each:
    """Marker wrapper đánh dấu nguồn iteration trong unified inputs.

    Dùng để đánh dấu input nào sẽ được iterate (mỗi iteration nhận một value),
    phân biệt với broadcast inputs (cùng value cho tất cả iterations).

    Example:
        with ForLoopNode(
            inputs={
                "item": Each(items_node["items"]),  # iterate qua list này
                "multiplier": PARENT["multiplier"]   # broadcast cho tất cả iterations
            }
        ) as loop:
            ...

        with AsyncIterNode(
            inputs={
                "chunk": Each(stream_source["stream"]),  # iterate qua async iterable
                "prefix": config_node["prefix"]          # broadcast
            }
        ) as stream:
            ...
    """

    __slots__ = ['source']

    def __init__(self, source: Any):
        """Khởi tạo Each wrapper.

        Args:
            source: Nguồn iterable (list, Ref đến list, async iterable, etc.)
        """
        self.source = source

    def __repr__(self) -> str:
        return f"Each({self.source!r})"
