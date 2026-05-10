# author hgh
# version 1.0
from typing import Dict, Optional

from infra.cache.cache_manager import CacheManager

_registry: Dict[str,CacheManager] = {}
_default: Optional[CacheManager] = None

def cache_register(namespace: str,manager: CacheManager) -> None:
    _registry[namespace] = manager

def set_default(manager: CacheManager) -> None:
    global _default
    _default = manager

def get_registry_manager(namespace: Optional[str] = None) -> Optional[CacheManager]:
    if namespace and namespace in _registry:
        return _registry[namespace]
    return _default