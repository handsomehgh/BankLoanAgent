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

    def _build_key(self,user_id: str,session_id: str) -> str:
        return f"{self._key_prefix}:{user_id}:{session_id}"

    def next_seq(self,user_id: str,session_id: str) -> int:
        client = self.redis_manager.get_client()
        key = self._build_key(user_id,session_id)

        if client is not None:
            try:
                seq = client.incr(key)
                logger.debug(f"Redis INCR: user_id={user_id},session_id={session_id}, seq={seq}")
                return seq
            except Exception as e:
                logger.error(
                    f"Redis INCR failed for user_id={user_id},session_id={session_id}: {e}. Falling back to local counter."
                )

        with self._local_lock:
            current = self._local_counters.get(key,0) + 1
            self._local_counters[key] = current
            logger.warning(f"Local INCR (degraded): keyword={key}, seq={current}")
            return current

    def get_current(self,user_id: str,session_id: str) -> int:
        client = self.redis_manager.get_client()
        key = self._build_key(user_id,session_id)

        if client is not None:
            try:
                val = client.get(key)
                return int(val) if val else 0
            except Exception as e:
                logger.warning(f"Redis GET failed: {e}")
        with self._local_lock:
            return self._local_counters.get(key, 0)





