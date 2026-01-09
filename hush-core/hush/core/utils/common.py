"""Các hàm tiện ích dùng chung cho hush-core."""

import asyncio
import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Type


@dataclass
class Param:
    """Định nghĩa parameter cho input/output của node.

    Example:
        input_schema = {
            "query": Param(str, required=True, description="Search query"),
            "limit": Param(int, default=10, description="Max results"),
        }
    """
    type: Type = str
    required: bool = False
    default: Any = None
    description: str = ""


def unique_name() -> str:
    """Tạo tên duy nhất sử dụng UUID."""
    return uuid.uuid4().hex[:8]


def verify_data(data: Any) -> bool:
    """Xác thực dữ liệu hợp lệ (không chứa các type không hợp lệ)."""
    # Validation cơ bản - có thể mở rộng
    return True


def raise_error(message: str) -> None:
    """Raise ValueError với message được cung cấp."""
    raise ValueError(message)


def extract_condition_variables(condition: str) -> Dict[str, str]:
    """Trích xuất tên biến và type suy luận từ condition string.

    Args:
        condition: Chuỗi condition như "a > 10 and b == 'hello'"

    Returns:
        Dict ánh xạ tên biến sang type suy luận
    """
    # Pattern để match tên biến (identifier không theo sau bởi dấu ngoặc mở)
    # Loại trừ Python keyword và built-in name
    keywords = {'and', 'or', 'not', 'in', 'is', 'True', 'False', 'None', 'if', 'else'}

    # Tìm tất cả identifier
    identifiers = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', condition)

    variables = {}
    for var in identifiers:
        if var not in keywords and var not in variables:
            # Thử suy luận type từ context
            variables[var] = "Any"

    return variables


def fake_chunk_from(content: str, model: str = "default") -> Dict[str, Any]:
    """Tạo fake chunk response cho xử lý lỗi."""
    return {
        "choices": [{
            "delta": {
                "content": content
            }
        }],
        "model": model
    }


def ensure_async(func: Callable) -> Callable:
    """Đảm bảo function tương thích với async.

    Nếu function đã là async, trả về nguyên vẹn.
    Nếu là synchronous, wrap để chạy trong thread pool executor.

    Args:
        func: Function cần làm cho tương thích async

    Returns:
        Async function có thể await

    Example:
        def sync_func(x):
            return x * 2

        async def async_func(x):
            return x * 2

        # Cả hai giờ có thể await:
        func = ensure_async(sync_func)
        result = await func(10)
    """
    if asyncio.iscoroutinefunction(func):
        return func

    async def async_wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    return async_wrapper
