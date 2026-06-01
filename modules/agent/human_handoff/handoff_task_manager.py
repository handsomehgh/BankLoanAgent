# author hgh
# version 1.0
"""
人工转接工单管理器（供 Streamlit 工作台使用）
"""
import logging
from typing import List, Dict
from langgraph.types import Command

from config.global_constant.constants import ConfigFields
from infra.database.redis_manager import RedisManager

logger = logging.getLogger(__name__)

HANDOFF_PENDING_KEY = "human_handoff:pending"
HANDOFF_TASK_KEY = "handoff_task"

class HandoffTaskManager:
    def __init__(self, graph, redis_manager: RedisManager):
        self.graph = graph
        self.redis = redis_manager

    def get_pending_tasks(self) -> List[Dict]:
        """Get all pending work orders"""
        logger.info(f"[HandoffTaskManager] start getting pending tasks")
        client = self.redis.get_client()
        if not client:
            return []
        tasks = []
        pending_ids = client.zrange(HANDOFF_PENDING_KEY, 0, -1)
        logger.info(f"[HandoffTaskManager] pending ids: {pending_ids}")
        for tid in pending_ids:
            thread_id = tid.decode() if isinstance(tid, bytes) else tid
            task_data = client.hgetall(f"{HANDOFF_TASK_KEY}:{thread_id}")
            logger.info(f"[HandoffTaskManager] pending data: {task_data}")
            if task_data:
                tasks.append({
                    "thread_id": thread_id,
                    "trace_id": task_data.get(b"trace_id", b"").decode(),
                    "user_id": task_data.get(b"user_id", b"").decode(),
                    "handoff_summary": task_data.get(b"handoff_summary", b"").decode(),
                    "timestamp": task_data.get(b"timestamp", b"").decode()
                })
        return tasks

    def recover_task(self, thread_id: str, action: str, content: str = ""):
        """recover the specified work order"""
        resume = {"action": action, "content": content} if action == "reply" else {"action": action}
        self.graph.invoke(
            Command(resume=resume),
            config={ConfigFields.CONFIGURABLE.value: {ConfigFields.THREAD_ID.value: thread_id}}
        )
        # clear pending collections and work order cache
        client = self.redis.get_client()
        if client:
            client.zrem(HANDOFF_PENDING_KEY, thread_id)
            client.delete(f"{HANDOFF_TASK_KEY}:{thread_id}")
        logger.info("[HandoffTaskManager] work order recover completed: thread_id=%s", thread_id)
