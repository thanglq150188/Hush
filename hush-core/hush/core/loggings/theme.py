"""Theme và highlighter cho logging."""

from rich.theme import Theme


# Theme logging sạch - tối ưu cho khả năng đọc
# INFO là trung tính (mặc định), chỉ các vấn đề mới nổi bật
LOGGING_THEME = Theme({
    # Log level - chỉ highlight các vấn đề
    "logging.level.debug": "#6e7681",           # Xám (mờ vào background)
    "logging.level.info": "white",              # Trắng
    "logging.level.warning": "#d29922",         # Vàng (chú ý)
    "logging.level.error": "#f85149",           # Đỏ (có vấn đề)
    "logging.level.critical": "bold reverse #b81c1c",  # NGUY HIỂM - trắng trên nền đỏ đậm

    # Các thành phần log - metadata mờ
    "log.time": "dim white",                    # Timestamp trắng mờ
    "log.message": "white",                     # Nội dung trắng
    "log.path": "#6e7681",                      # Path xám

    # Màu markup tùy chỉnh cho inline highlighting (phiên bản nhạt/sáng hơn)
    "info": "#a5d6ff",                          # Xanh dương nhạt
    "warning": "#f0c674",                       # Vàng nhạt
    "error": "#ffa198",                         # Đỏ nhạt
    "critical": "#ffa198",                      # Đỏ nhạt
    "success": "#a5d6a7",                       # Xanh lá nhạt
    "muted": "#b0b8c1",                         # Xám nhạt
    "highlight": "#ffb86c",                     # Cam nhạt
    "special": "#d8a9ff",                       # Tím nhạt
})
