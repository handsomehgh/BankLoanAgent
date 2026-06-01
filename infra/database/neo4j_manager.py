# author hgh
# version 1.0
import atexit
import logging
import threading
from typing import Optional

from neo4j import Driver, GraphDatabase, basic_auth

from config.models.datasource_config import GraphConfig

logger = logging.getLogger(__name__)


class Neo4jManager:
    _instance: Optional['Neo4jManager'] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize = False

        return cls._instance

    def __init__(self, config: Optional[GraphConfig] = None):
        if self._initialize:
            return
        self._driver: Optional[Driver] = None
        if config and config.enabled:
            try:
                self._driver = GraphDatabase.driver(
                    uri=config.neo4j_uri,
                    auth=basic_auth(config.neo4j_user, config.neo4j_password),
                    max_connection_lifetime=config.max_connection_lifetime,
                )
                self._driver.verify_connectivity()
                logger.info("Neo4j driver initialized successfully")
            except Exception as e:
                logger.error("Failed to initialize Neo4j driver: %s", e)
                self._driver = None

        self._initialize = True
        atexit.register(self.close)

    @classmethod
    def from_config(cls, config: GraphConfig):
        return cls(config)

    def get_driver(self) -> Optional[Driver]:
        return self._driver

    @property
    def is_available(self) -> bool:
        return self._driver is not None

    def close(self):
        if self._driver:
            self._driver.close()
            logger.info("Neo4j driver closed")
            self._driver = None
