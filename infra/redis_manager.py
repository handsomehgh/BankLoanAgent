# author hgh
# version 1.0
import logging
import threading
from typing import Optional

import redis
from redis import ConnectionPool

from config.models.redis_config import RedisConfig

logger = logging.getLogger(__name__)

class RedisManager:
    _instance: Optional['RedisManager'] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
            self,
            host: str = "localhost",
            port: int = 6379,
            db: int = 0,
            password: str = None,
            max_connections: int = 50,
            socket_timeout: float = 0.5,
            socket_connect_timeout: float = 1.0,
            retry_on_timeout: bool = True,
            health_check_interval: int = 30
    ):
        if self._initialized:
            return
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.max_connections = max_connections
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.retry_on_timeout = retry_on_timeout
        self.health_check_interval = health_check_interval

        self._pool: Optional[ConnectionPool] = None
        self._init_pool()
        self._initialized = True

    def _init_pool(self):
        try:
            self._pool = ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_connect_timeout,
                retry_on_timeout=self.retry_on_timeout,
                health_check_interval=self.health_check_interval,
                decode_responses=False
            )
            test_client = redis.Redis(connection_pool=self._pool)
            test_client.ping()
            logger.info(
                "Redis connection pool established: %s:%d/%d (max_connections=%d)",
                self.host, self.port, self.db, self.max_connections
            )
        except Exception as e:
            logger.error(
                "Failed to create Redis connection pool (host=%s:%d): %s. "
                "All Redis-dependent features will be degraded.",
                self.host, self.port, e
            )
            self._pool = None

    @classmethod
    def from_config(cls, config: RedisConfig):
        return cls(
            host=config.host,
            port=config.port,
            db=config.db,
            password=config.password,
            max_connections=config.max_connections,
            socket_timeout=config.socket_timeout,
            socket_connect_timeout=config.socket_connect_timeout,
            retry_on_timeout=config.retry_on_timeout,
            health_check_interval=config.health_check_interval,
        )

    def get_client(self) -> Optional[redis.Redis]:
        if self._pool is None:
            return None
        try:
            return redis.Redis(connection_pool=self._pool)
        except Exception as e:
            logger.warning(f"Failed to get Redis client from pool: {e}")
            return None

    @property
    def is_available(self) -> bool:
        if self._pool is None:
            return False
        try:
            client = self.get_client()
            client.ping()
            return True
        except Exception as e:
            return False

    def close(self):
        if self._pool:
            try:
                self._pool.disconnect()
                logger.info("Redis connection pool closed")
            except Exception as e:
                logger.warning(f"Error closing Redis pool: {e}")
            finally:
                self._pool = None


