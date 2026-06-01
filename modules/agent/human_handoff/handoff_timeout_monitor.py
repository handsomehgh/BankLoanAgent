# author hgh
# version 1.0
"""
humanHandoff timeout degradation monitor,
regular scans pending work order,automatically invoking graph to recover and inject degradation messages when time out
"""
import logging
import threading
from datetime import datetime, timezone
from typing import Optional

import redis
from langgraph.types import Command

from config.global_constant.constants import ConfigFields
from infra.database.redis_manager import RedisManager
from utils.monitor_utils.metrics import handoff_task_timeout_total

logger = logging.getLogger(__name__)

HANDOFF_PENDING_KEY = "human_handoff:pending"
DEFAULT_TIMEOUT_SECONDS = 90
DEGRADE_MESSAGE = "当前人工坐席繁忙，请稍后重试或拨打我行客服热线 95333。"

class HandoffTimeoutMonitor:
    def __init__(self,graph,redis_manager: RedisManager,timeout_seconds=DEFAULT_TIMEOUT_SECONDS):
        self.graph = graph
        self.redis_manager = redis_manager
        self.timeout_seconds = timeout_seconds

    def start(self):
        def _loop():
            while True:
                try:
                    self._scan_and_process()
                except Exception as e:
                    logger.error("[HandoffTimeoutMonitor] scan exception: %s", e, exc_info=True)
        thread = threading.Thread(target=_loop,daemon=True,name="HandoffTimeoutMonitor")
        thread.start()
        logger.info("[HandoffTimeoutMonitor] has been started，timeout threshold=%d seconds", self.timeout_seconds)

    def _scan_and_process(self):
        client = self.redis_manager.get_client()
        if not client:
            return

        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - self.timeout_seconds
        try:
            expire_ids = client.zrangebyscore(HANDOFF_PENDING_KEY,0,cutoff)
        except Exception as e:
            logger.error("[HandoffTimeoutMonitor] redis failed scan pending work order: %s", e)
            return

        for thread_id_bytes in expire_ids:
            thread_id = thread_id_bytes.decode() if isinstance(thread_id_bytes, bytes) else thread_id_bytes
            logger.warning("[HandoffTimeoutMonitor] work order timeout，automatically degrade: thread_id=%s", thread_id)
            try:
                self._degrade(thread_id)
            except Exception as e:
                logger.error("[HandoffTimeoutMonitor] failed degrade recover (thread_id=%s): %s", thread_id, e, exc_info=True)
            finally:
                client.zrem(HANDOFF_PENDING_KEY,thread_id)

    def _degrade(self,thread_id):
        resume_value = {"action": "reply","content": DEGRADE_MESSAGE}
        self.graph.invoke(Command(resume=resume_value),config={ConfigFields.CONFIGURABLE.value: {ConfigFields.THREAD_ID.value:thread_id}})
        handoff_task_timeout_total.inc()
        logger.info("[HandoffTimeoutMonitor] timeout degrade complete: thread_id=%s", thread_id)

    @staticmethod
    def add_pending_task(redis_manager: RedisManager, thread_id: str, timestamp: Optional[float] = None):
        client = redis_manager.get_client()
        if not client:
            return
        score = timestamp or datetime.now(timezone.utc).timestamp()
        try:
            client.zadd(HANDOFF_PENDING_KEY,{thread_id:score})
        except redis.RedisError as e:
            logger.error("[HandoffTimeoutMonitor] failed add work order to pending set: %s", e)

    @staticmethod
    def remove_pending_task(redis_manager: RedisManager, thread_id: str):
        client = redis_manager.get_client()
        if not client:
            return
        try:
            client.zrem(HANDOFF_PENDING_KEY, thread_id)
        except redis.RedisError as e:
            logger.error("移除挂起工单失败: %s", e)

