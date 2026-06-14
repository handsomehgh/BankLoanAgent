# author hgh
# version 1.0
import logging
import threading
import time
from typing import Callable, Any

from exceptions.exception import CircuitBreakerOpenError, ToolExecutionException
from modules.tools.base_tool import ToolErrorType

logger = logging.getLogger(__name__)


class CircuitBreaker():
    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"
        self._lock = threading.Lock()

    def call(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        with self._lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    logger.info(f"[CircuitBreaker] {self.name} enter half-open state and attempt to detect recovery")
                else:
                    logger.warning(f"[CircuitBreaker] {self.name} circuit breaker triggered,directly rejected")
                    raise CircuitBreakerOpenError(f"断路器 {self.name} 已打开")

            try:
                result = func(*args, **kwargs)
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    logger.info(f"[CircuitBreaker] {self.name} detect successfully，has been recovered")
                self.failure_count = 0
                return result
            except ToolExecutionException as e:
                if e.error_type in (ToolErrorType.TEMPORARY_ERROR, ToolErrorType.EXTERNAL_ERROR):
                    self.failure_count += 1
                    self.last_failure_time = time.time()
                    if self.failure_count >= self.failure_threshold:
                        self.state = "OPEN"
                        logger.error(
                            f"[CircuitBreaker] {self.name} continuous failure {self.failure_count} time，already tripped")
                raise e
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.error(
                        f"[CircuitBreaker] {self.name} continuous failure {self.failure_count} time，already tripped")
                raise e
