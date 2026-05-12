# author hgh
# version 1.0
import logging
import threading

from infra.redis_manager import RedisManager

logger = logging.getLogger(__name__)

class SequenceGenerator:
    def __init__(self,key_prefix: str = "user_msg_seq"):
        self.redis_manager = RedisManager
        self._key_prefix = key_prefix

        self._local_counters: dict[str, int] = {}
        self._local_lock = threading.Lock()

    def _build_key(self,extra: str) -> str:
        return f"{self._key_prefix}:{extra}"

    def next_seq(self,keyword: str) -> int:
        client = self.redis_manager.get_client()
        key = self._build_key(keyword)

        if client is not None:
            try:
                seq = client.incr(key)
                logger.debug(f"Redis INCR: keyword={keyword}, seq={seq}")
                return seq
            except Exception as e:
                logger.error(
                    f"Redis INCR failed for keyword={keyword}: {e}. Falling back to local counter."
                )

        with self._local_lock:
            current = self._local_counters.get(keyword,0) + 1
            self._local_counters[keyword] = current
            logger.warning(f"Local INCR (degraded): keyword={keyword}, seq={current}")
            return current

    def get_current(self,keyword: str) -> int:
        client = self.redis_manager.get_client()
        key = self._build_key(keyword)

        if client is not None:
            try:
                val = client.get(key)
                return int(val) if val else 0
            except Exception as e:
                logger.warning(f"Redis GET failed: {e}")
        with self._local_lock:
            return self._local_counters.get(keyword, 0)





