# author hgh
# version 1.0
import functools
import logging

from exceptions.exception import ToolExecutionException
from modules.tools.base_tool import ToolErrorType

logger = logging.getLogger(__name__)

def with_tool_error_handling(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ToolExecutionException:
            raise
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.warning(f"工具 {func.__name__} 临时性故障: {e}")
            raise ToolExecutionException(str(e), ToolErrorType.TEMPORARY_ERROR)
        except (KeyError, AttributeError) as e:
            logger.error(f"工具 {func.__name__} 数据/配置异常: {e}")
            raise ToolExecutionException(str(e), ToolErrorType.BUSINESS_ERROR)
        except Exception as e:
            logger.exception(f"工具 {func.__name__} 未知异常")
            raise ToolExecutionException(str(e), ToolErrorType.EXTERNAL_ERROR)
    return wrapper