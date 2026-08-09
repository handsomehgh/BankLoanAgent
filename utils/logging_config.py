"""
Unified Logging Configuration Module
Provides setup_logging() to initialize the root logger, supporting context injection (user_id/thread_id)
Uses contextvars.ContextVar for async-safe context propagation across tasks.
"""
import contextvars
import logging
from typing import Optional

# Async-safe context variables (replaces threading.local)
_user_id_var: contextvars.ContextVar[str] = contextvars.ContextVar('user_id', default='-')
_thread_id_var: contextvars.ContextVar[str] = contextvars.ContextVar('thread_id', default='-')
_trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar('trace_id', default='-')


class ContextFilter(logging.Filter):
    """Inject user_id, thread_id, trace_id into LogRecord (async-safe via ContextVar)"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.user_id = _user_id_var.get()
        record.thread_id = _thread_id_var.get()
        record.trace_id = _trace_id_var.get()
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

    # Set the root logger to DEBUG, allowing all levels to pass through the handler filter (the actual output level is controlled by the handler)
    root.setLevel(logging.DEBUG)

    for lib in ("httpx", "urllib3", "watchdog", "pymilvus", "transformers","dashscope","urllib3","sshtunnel", "paramiko"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def set_log_context(user_id: Optional[str] = None, thread_id: Optional[str] = None, trace_id: Optional[str] = None) -> None:
    """Set logging context for the current task/thread (async-safe via ContextVar)"""
    if user_id is not None:
        _user_id_var.set(user_id)
    if thread_id is not None:
        _thread_id_var.set(thread_id)
    if trace_id is not None:
        _trace_id_var.set(trace_id)