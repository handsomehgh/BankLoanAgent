"""
Unified Logging Configuration Module
Provides setup_logging() to initialize the root logger, supporting context injection (user_id/thread_id)
"""
import logging
import os
import threading
from logging.handlers import RotatingFileHandler
from typing import Optional


class ContextFilter(logging.Filter):
    """Inject user_id and thread_id into LogRecord (thread-safe)"""

    def __init__(self):
        super().__init__()
        self._local = threading.local()

    def set_context(self, user_id: Optional[str] = None, thread_id: Optional[str] = None):
        """Set the context identifier in the current thread"""
        self._local.user_id = user_id or "-"
        self._local.thread_id = thread_id or "-"

    def filter(self, record: logging.LogRecord) -> bool:
        record.user_id = getattr(self._local, "user_id", "-")
        record.thread_id = getattr(self._local, "thread_id", "-")
        return True


_context_filter = ContextFilter()


def get_context_filter() -> ContextFilter:
    """Get a globally unique context filter"""
    return _context_filter


def setup_logging(log_level: str = "INFO") -> None:
    """
    Initialize the root logging system:
    - Remove existing handlers to prevent third-party libraries from taking over
    - Console output (specify level)
    - Inject context filter (user/session identification)
    - Quiet some third-party libraries
    """
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Console handler
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

    log_dir = "D:\\logs\\bank_agent"
    os.makedirs(log_dir, exist_ok=True)
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "app.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5
    )
    file_handler.terminator = "\n"
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    file_handler.addFilter(_context_filter)
    root.addHandler(file_handler)

    # Set the root logger to DEBUG, allowing all levels to pass through the handler filter (the actual output level is controlled by the handler)
    root.setLevel(logging.DEBUG)

    for lib in ("httpx", "urllib3", "watchdog", "pymilvus", "transformers"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def set_log_context(user_id: Optional[str] = None, thread_id: Optional[str] = None) -> None:
    _context_filter.set_context(user_id=user_id, thread_id=thread_id)