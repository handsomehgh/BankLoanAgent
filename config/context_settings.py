# author hgh
# version 1.0
import contextvars

_strict_enum_validation = contextvars.ContextVar(
    "strict_enum_validation",
    default=True
)

def set_enum_strictness(strict: bool):
    """在应用启动时调用一次，将配置注入上下文"""
    _strict_enum_validation.set(strict)

def get_enum_strictness() -> bool:
    """模型内部通过此函数获取当前严格度"""
    return _strict_enum_validation.get()
