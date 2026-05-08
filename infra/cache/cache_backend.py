# author hgh
# version 1.0
from abc import ABC, abstractmethod
from typing import Optional


class CacheBackend(ABC):
    @abstractmethod
    def get(self,key: str) -> Optional[bytes]:
        """return original bytes,return None if not hit"""
        pass

    @abstractmethod
    def set(self,key: str,value: bytes,ttl: Optional[int] = None) -> None:
        """set cache"""
        pass

    @abstractmethod
    def delete(self,key: str) -> None:
        """delete cache"""
        pass

    @abstractmethod
    def exists(self,key: str) -> bool:
        """check existence of cache"""
        pass
