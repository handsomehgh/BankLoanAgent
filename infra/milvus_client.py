# author hgh
# version 1.0
import logging
from typing import Dict

from pymilvus import Collection, connections
from pymilvus.orm import utility

logger = logging.getLogger(__name__)


class MilvusClientManager:
    """initialize the milvus client(singleton)"""
    _instance = None

    def __new__(cls, uri: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, uri: str = None):
        # only initialize once
        if self._initialized:
            return
        if uri is None:
            raise ValueError("MilvusClientManager uri cannot be None")

        self.uri = uri
        self._collections: Dict[str, Collection] = {}
        self._connect()
        self._initialized = True

    def _connect(self):
        try:
            if not connections.has_connection("default"):
                connections.connect(alias="default", uri=self.uri, timeout=30)
                logger.info(f"Milvus successfully connected: {self.uri}")
        except Exception as e:
            logger.error(f"Milvus connection error: {e}")
            raise

    def get_collection(self, name: str) -> Collection:
        """get the loaded collection by name"""
        if name not in self._collections:
            if not utility.has_collection(name):
                raise RuntimeError(f"Collection {name} doesn't exist，Please execute initialization scripts first")
            col = Collection(name)
            col.load()
            self._collections[name] = col
            logger.info(f"Collection '{name}' has been loaded!")
        return self._collections[name]

    def delete_collection(self, name: str) -> bool:
        """delete collection by name"""
        if utility.has_collection(name):
            try:
                utility.drop_collection(name)
                if name in self._collections:
                    self._collections.pop(name)
                return True
            except Exception as e:
                logger.error(f"Failed to delete collection {name}: {e}")
        else:
            logger.error(f"Collection '{name}' doesn't exist!")
            return False


    def has_collection(self, name: str) -> bool:
        return utility.has_collection(name)
