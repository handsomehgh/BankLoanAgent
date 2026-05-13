# author hgh
# version 1.0
import logging
from typing import Set

from infra.redis_manager import RedisManager

logger = logging.getLogger(__name__)


class CursorManager:
    """cursor manager:store and query the collection of sequence numbers of processed messages"""

    def __init__(self, key_prefix: str = "process_at"):
        self.key_prefix = key_prefix
        self._redis_manager = RedisManager()

        self._local_sets: dict[str, set[int]] = {}

    def _build_key(self, user_id: str, cursor_type: str) -> str:
        return f"{self.key_prefix}:{cursor_type}:{user_id}"

    def get_process_at(self, user_id: str, cursor_type: str) -> Set[int]:
        """get the set of processed sequence numbers"""
        client = self._redis_manager.get_client()
        key = self._build_key(user_id, cursor_type)

        if client is not None:
            try:
                members = client.smembers(key)
                result = {int(m) for m in members}
                logger.debug(
                    f"Redis SMEMBERS: user={user_id}, type={cursor_type}, count={len(result)}"
                )
                return result
            except Exception as e:
                logger.warning(
                    f"Redis SMEMBERS failed for user={user_id}, type={cursor_type}: {e}. "
                    "Falling back to local set."
                )

        return self._local_sets.get(key, set())

    def add_to_process_set(self, user_id: str, cursor_type: str, seq: int):
        """add sequence number to the processed set"""
        client = self._redis_manager.get_client()
        key = self._build_key(user_id, cursor_type)

        if client is not None:
            try:
                client.sadd(key, seq)
                logger.debug(f"Redis SADD: user={user_id}, type={cursor_type}, seq={seq}")
                return
            except Exception as e:
                logger.warning(
                    f"Redis SADD failed for user={user_id}, type={cursor_type}, seq={seq}: {e}. "
                    "Falling back to local set."
                )

        # memory downgrade
        if key not in self._local_sets:
            self._local_sets[key] = set()
        self._local_sets[key].add(seq)

    def add_batch_to_processed_set(self, user_id: str, cursor_type: str, seqs: Set[int]):
        """batch add processed sequence numbers"""
        if not seqs:
            return
        client = self._redis_manager.get_client()
        key = self._build_key(user_id, cursor_type)

        if client is not None:
            try:
                client.sadd(key, *[str(s) for s in seqs])
                logger.debug(
                    f"Redis SADD batch: user={user_id}, type={cursor_type}, count={len(seqs)}"
                )
                return
            except Exception as e:
                logger.warning(f"Redis SADD batch failed: {e}")

        # memory downgrade
        if key not in self._local_sets:
            self._local_sets[key] = set()
        self._local_sets[key].update(seqs)

    def is_processed(self, user_id: str, cursor_type: str, seq: int) -> bool:
        """check whether a certain sequence number has already been processed"""
        client = self._redis_manager.get_client()
        key = self._build_key(user_id, cursor_type)

        if client is not None:
            try:
                return bool(client.sismember(key, str(seq)))
            except Exception as e:
                logger.warning(f"Redis SISMEMBER failed: {e}")

        return seq in self._local_sets.get(key, set())

    def remove_old_entries(self, user_id: str, cursor_type: str, keep_last_n: int = 10000):
        """clear outdated sequence numbers to prevent the collection from growing indefinitely"""
        processed = self.get_process_at(user_id, cursor_type)
        if len(processed) <= keep_last_n:
            return

        # after sorting,keep the largest items
        sorted_seq = sorted(processed, reverse=True)
        to_keep = sorted_seq[:keep_last_n]
        to_remove = processed - to_keep

        if not to_remove:
            return

        client = self._redis_manager.get_client()
        key = self._build_key(user_id, cursor_type)
        if client is not None:
            try:
                client.srem(key, *[str(s) for s in to_remove])
                logger.info(
                    f"Cleaned {len(to_remove)} old cursor entries for user={user_id}, type={cursor_type}"
                )
            except Exception as e:
                logger.warning(f"Redis SREM during cleanup failed: {e}")
        else:
            if key in self._local_sets:
                self._local_sets[key] = to_keep

    def clean_processed_set(self, user_id: str, cursor_type: str):
        """clean out processed sequence numbers"""
        client = self._redis_manager.get_client()
        key = self._build_key(user_id, cursor_type)

        if client is not None:
            try:
                client.delete(key)
            except Exception as e:
                logger.warning(f"Redis DEL failed: {e}")

        if key in self._local_sets:
            del self._local_sets[key]
