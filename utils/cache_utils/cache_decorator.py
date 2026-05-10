# author hgh
# version 1.0
import asyncio
import functools
import hashlib
import json
import logging
import threading
from typing import Dict, Optional, Tuple, Callable, List, Any, TypeVar, Union

from infra.cache.cache_manager import CacheManager
from infra.cache.cache_registry import get_registry_manager

logger = logging.getLogger(__name__)

T = TypeVar("T")

_locks: Dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()


def _get_lock(key: str) -> threading.Lock:
    with _locks_lock:
        if key not in _locks:
            _locks[key] = threading.Lock()
        return _locks[key]


def _build_cache_key(args: Tuple,kwargs: Dict,ignore_args: Optional[List[int]] = None) -> str:
    filtered_args = list(args)
    if ignore_args:
        for idx in sorted(ignore_args,reverse=True):
            if idx < len(filtered_args):
                filtered_args.pop(idx)
    params = {"args": filtered_args, "kwargs": kwargs}
    raw = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.md5(raw.encode()).hexdigest()

    params = {"args": args,"kwargs": kwargs}
    raw = json.dumps(params,sort_keys=True,ensure_ascii=False,default=str)
    return hashlib.md5(raw.encode('utf-8')).hexdigest()

def custom_cached(
        namespace: Optional[str] = None,
        cache_manager: Optional[CacheManager] = None,
        ttl: Optional[int] = None,
        null_ttl: int = 60,
        converter: Optional[Callable[[Any],Any]] = None,
        empty_result_factory: Optional[Callable[[], Any]] = None,
        ignore_args: Optional[List[int]] = None):
    """general cache decorator factory"""

    def _resolve_manager(args) -> Optional[CacheManager]:
        if cache_manager:
            return cache_manager
        return get_registry_manager(namespace)

    def decorator(func: Callable[...,T]) -> Callable[...,Union[T,Any]]:
        @functools.wraps(func)
        def sync_wrapper(*args,**kwargs):
            mgr = _resolve_manager(namespace)
            if not mgr:
                return func(*args,**kwargs)

            param_key = _build_cache_key(args,kwargs,ignore_args)
            full_key = mgr.build_key(func.__name__,param_key)

            def _read_cache():
                try:
                    raw = mgr.get(full_key)
                    if raw is None:
                        return (False,None)
                    if isinstance(raw,bytes) and raw ==  b"__NULL__":
                        logger.debug(f"[Cache] null hit: {full_key}")
                        empty = empty_result_factory() if empty_result_factory else None
                        return (True, empty)
                    logger.debug(f"[Cache] hit: {full_key}")
                    return (True,_safe_convert(converter,raw))
                except Exception as e:
                    logger.warning(f"[Cache] read failed: {e}")
                    return (False, None)

            hit,result = _read_cache()
            if hit:
                return result

            lock = _get_lock(full_key)
            acquired = lock.acquire(blocking=False)
            if not acquired:
                with lock:
                    pass
                hit,result = _read_cache()
                if hit:
                    return result
                return func(*args, **kwargs)

            try:
                result = func(*args, **kwargs)
                try:
                    if result:
                        to_cache = _serialize_result(result)
                        mgr.set(full_key,to_cache,ttl=ttl)
                    else:
                        mgr.set_null(full_key, ttl=null_ttl)
                except Exception as e:
                    logger.warning(f"[Cache] write failed: {e}")
                return result
            finally:
                lock.release()

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            mgr = _resolve_manager(args)
            if not mgr:
                return await func(*args, **kwargs)

            param_key = _build_cache_key(args, kwargs, ignore_args)
            full_key = mgr.build_key(func.__name__, param_key)

            def _read_cache():
                try:
                    raw = mgr.get(full_key)
                    if raw is None:
                        return (False, None)
                    if isinstance(raw, bytes) and raw == b"__NULL__":
                        logger.debug(f"[Cache] null hit: {full_key}")
                        empty = empty_result_factory() if empty_result_factory else None
                        return (True, empty)
                    logger.debug(f"[Cache] hit: {full_key}")
                    return (True, _safe_convert(converter, raw))
                except Exception as e:
                    logger.warning(f"[Cache] read failed: {e}")
                    return (False, None)

            hit, result = _read_cache()
            if hit:
                return result

            lock = _get_lock(full_key)
            acquired = lock.acquire(blocking=False)
            if not acquired:
                with lock:
                    pass
                hit, result = _read_cache()
                if hit:
                    return result
                return await func(*args, **kwargs)

            try:
                result = await func(*args, **kwargs)
                try:
                    if result:
                        to_cache = _serialize_result(result)
                        mgr.set(full_key, to_cache, ttl=ttl)
                    else:
                        mgr.set_null(full_key, ttl=null_ttl)
                except Exception as e:
                    logger.warning(f"[Cache] write failed: {e}")
                return result
            finally:
                lock.release()
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

def _make_json_safe(obj):
    """递归将 Pydantic 模型转换为字典，其他不可序列化对象转为字符串"""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_safe(item) for item in obj]
    elif hasattr(obj, "model_dump"):          # Pydantic v2
        return _make_json_safe(obj.model_dump(mode="json"))
    elif hasattr(obj, "dict"):                # Pydantic v1
        return _make_json_safe(obj.dict())
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)

def _serialize_result(result: Any) -> Any:
    return _make_json_safe(result)

def _safe_convert(converter,data):
    if converter is None:
        return data
    try:
        return converter(data)
    except Exception as e:
        logger.warning(f"[Cache] Converter failed, returning None. Data: {str(data)[:200]}, Error: {e}")
        return None