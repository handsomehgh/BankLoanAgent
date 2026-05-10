"""
统一日志配置模块
提供 setup_logging() 初始化根日志器，支持上下文注入（user_id/thread_id）
"""
import logging
import threading
from typing import Optional


class ContextFilter(logging.Filter):
    """向 LogRecord 注入 user_id 和 thread_id（线程安全）"""

    def __init__(self):
        super().__init__()
        self._local = threading.local()

    def set_context(self, user_id: Optional[str] = None, thread_id: Optional[str] = None):
        """在当前线程设置上下文标识"""
        self._local.user_id = user_id or "-"
        self._local.thread_id = thread_id or "-"

    def filter(self, record: logging.LogRecord) -> bool:
        record.user_id = getattr(self._local, "user_id", "-")
        record.thread_id = getattr(self._local, "thread_id", "-")
        return True


_context_filter = ContextFilter()


def get_context_filter() -> ContextFilter:
    """获取全局唯一的上下文过滤器"""
    return _context_filter


def setup_logging(log_level: str = "INFO") -> None:
    """
    初始化根日志系统：
    - 移除已有 handlers，防止第三方库抢占
    - 控制台输出（指定级别）
    - 注入上下文过滤器（用户 / 会话标识）
    - 安静化部分第三方库
    """
    root = logging.getLogger()
    # 先清空任何已有 handler（包括第三方库提前设置的）
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # 控制台 handler
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-5s] %(name)s "
        "[user:%(user_id)s][tid:%(thread_id)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console.setFormatter(fmt)
    console.addFilter(_context_filter)
    root.addHandler(console)

    # 根日志器设为 DEBUG，让所有级别都可以通过 handler 过滤（实际输出级别由 handler 控制）
    root.setLevel(logging.DEBUG)

    # 安静化非常啰嗦的第三方库
    for lib in ("httpx", "urllib3", "watchdog", "pymilvus", "transformers"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def set_log_context(user_id: Optional[str] = None, thread_id: Optional[str] = None) -> None:
    """在业务线程中设置当前请求的用户 / 会话标识"""
    _context_filter.set_context(user_id=user_id, thread_id=thread_id)