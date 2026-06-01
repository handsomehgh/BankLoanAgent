# author hgh
# version 1.0
"""
retry tool module
provides an exponential backoff retry decorator,suitable for idempotent oprations
"""
import functools
import logging
import time
from typing import Tuple, Type, Callable, Any

logger = logging.getLogger(__name__)


def retry_on_failure(
        max_retries: int = 3,
        initial_delay: float = 0.5,
        backoff_factor: float = 2.0,
        exceptions: Tuple[Type[Exception], ...] = (Exception, ),
        on_retry: Callable[[Exception, int], None] = None
) -> Callable:
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries - 1:
                        logger.error(
                            f"Operation failed after {max_retries} retries."
                            f"Final error: {last_exception}"
                        )
                        raise
                    logger.warning(
                        f"Operation '{func.__name__}' failed (attempt {attempt + 1}/{max_retries})'"
                        f"Retrying in {delay:.2f}s Error: {e}"
                    )
                    if on_retry:
                        on_retry(e, attempt + 1)

                    time.sleep(delay)
                    delay *= backoff_factor

            if last_exception:
                raise last_exception
            return None

        return wrapper

    return decorator
