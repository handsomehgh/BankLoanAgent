# author hgh
# version 1.0
"""
全局缓存注册表
提供缓存管理器的注册与获取，避免循环依赖
"""
from typing import Dict, Optional
from infra.cache.cache_manager import CacheManager

_registry: Dict[str, CacheManager] = {}
_default: Optional[CacheManager] = None

def cache_register(namespace: str, manager: CacheManager) -> None:
    """注册缓存管理器"""
    _registry[namespace] = manager

def set_default(manager: CacheManager) -> None:
    """设置默认缓存管理器"""
    global _default
    _default = manager

def get_registry_manager(namespace: Optional[str] = None) -> Optional[CacheManager]:
    """根据命名空间获取缓存管理器，若不指定则返回默认"""
    if namespace and namespace in _registry:
        return _registry[namespace]
    return _default
