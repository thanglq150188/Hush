"""Các tiện ích format log."""

from typing import Any

# Indent cho log nhiều dòng
# Rich tự động indent các dòng tiếp theo để căn chỉnh với phần đầu message
LOG_INDENT = "  "  # Indent nhỏ để phân cấp trực quan


def log_break(text: str = "") -> str:
    """Thêm ngắt dòng với indent phù hợp cho log nhiều dòng.

    Sử dụng khi bạn muốn chia log message dài thành nhiều dòng.
    Rich tự động căn chỉnh các dòng tiếp theo với phần đầu message.

    Args:
        text: Nội dung đặt trên dòng mới (tùy chọn)

    Returns:
        Text với tiền tố newline và indent

    Example:
        LOGGER.info(f"Processing request{log_break('user_id: 123')}{log_break('action: update')}")
        # Output:
        # [12/12/25 00:59:32] INFO     [hush.core] Processing request
        #                                           user_id: 123
        #                                           action: update
    """
    return f"\n{LOG_INDENT}{text}"


def format_log_data(data: Any, max_length: int = 400, max_items: int = 3) -> str:
    """Format thông minh cho dữ liệu log, xử lý các kiểu dữ liệu khác nhau.

    Args:
        data: Dữ liệu cần format (dict, list, str, v.v.)
        max_length: Độ dài ký tự tối đa cho output
        max_items: Số lượng item tối đa hiển thị trong collection

    Returns:
        String đã format phù hợp cho logging

    Example:
        >>> format_log_data({"key": "value", "list": [1, 2, 3]})
        "{key='value', list=<list len=3>}"
    """
    if data is None:
        return "None"

    if isinstance(data, dict):
        if not data:
            return "{}"

        items = []
        for i, (k, v) in enumerate(data.items()):
            if i >= max_items:
                items.append(f"... +{len(data) - max_items} more")
                break

            if isinstance(v, str):
                val_str = f"'{v[:50]}...'" if len(v) > 50 else f"'{v}'"
            elif isinstance(v, (list, dict)):
                val_str = f"<{type(v).__name__} len={len(v)}>"
            elif isinstance(v, bytes):
                val_str = f"<bytes len={len(v)}>"
            else:
                val_str = str(v)[:50]

            items.append(f"{k}={val_str}")

        result = "{" + ", ".join(items) + "}"

    elif isinstance(data, (list, tuple)):
        if not data:
            return "[]" if isinstance(data, list) else "()"

        items = []
        for i, item in enumerate(data):
            if i >= max_items:
                items.append(f"... +{len(data) - max_items} more")
                break
            items.append(str(item)[:50])

        bracket = ("[]" if isinstance(data, list) else "()")
        result = bracket[0] + ", ".join(items) + bracket[1]

    elif isinstance(data, str):
        if len(data) > max_length:
            result = f"'{data[:max_length]}...' (len={len(data)})"
        else:
            result = f"'{data}'"

    elif isinstance(data, bytes):
        result = f"<bytes len={len(data)}>"

    else:
        result = str(data)

    if len(result) > max_length:
        result = result[:max_length] + "..."

    return result
