# author hgh
# version 1.0
"""
message encapsulation based on redis streams,
provided basic capabilities for producer,consumers and dead-letter queues
"""
import json
import logging
from datetime import timezone, datetime
from typing import Dict, Any, Optional, List

import redis
from pydantic import BaseModel, Field
from redis import RedisError
from infra.database.redis_manager import RedisManager

logger = logging.getLogger(__name__)


# ================== message structure =====================
class Message(BaseModel):
    event_type: str = Field(..., description="event type (knowledge_miss, routing_detail, ...)")
    payload: Dict[str, Any] = Field(default_factory=dict, description="event data load")
    trace_id: str = Field(default="", description="all link trace id")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="timestamp of message"
    )


# ================ message producers ====================
class MessageProducer:
    """write messages to redis stream"""

    def __init__(self, redis_manager: RedisManager):
        self._redis = redis_manager

    def publish(self, stream_name: str, event_type: str, payload: Dict[str, Any], trace_id: str) -> Optional[str]:
        """publish a message to specific stream"""
        client = self._redis.get_client()
        if client is None:
            logger.warning("Redis is unavailable,message production downgrade and skipped")
            return None

        msg = Message(event_type=event_type, payload=payload, trace_id=trace_id)
        try:
            msg_id = client.xadd(stream_name, {"data": msg.model_dump_json()})
            logger.debug("The message has been written Stream=%s, id=%s", stream_name, msg_id)
            return msg_id
        except Exception as e:
            logger.error("Failed to write message (Stream=%s): %s", stream_name, e)
            raise


# ================== message consumer ========================
class MessageConsumer:
    """consume messages from redis streams,supporting consumer groups and dead letter queues"""

    def __init__(self, redis_manager: RedisManager, stream_name: str, group_name: str, consumer_name: str):
        self._client = redis_manager.get_long_timeout_client()
        if self._client is None:
            raise RuntimeError("[MessageConsumer] cannot to create redis consumer connector")
        self.stream_name = stream_name
        self.group_name = group_name
        self.consumer_name = consumer_name
        self.dlq_stream = f"{stream_name}:dlq"

        self._ensure_group()

    def _ensure_group(self):
        try:
            self._client.xgroup_create(self.stream_name, self.group_name, id="0", mkstream=True)
            logger.info("Consumer group has been built: stream=%s, group=%s", self.stream_name, self.group_name)
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.debug("Consumer group already exists : %s", self.group_name)
            else:
                logger.warning("Failed to create consumer group: %s", e)
        except RedisError as e:
            logger.warning("Redis operation exception: %s", e)

    def consume(self, count: int = 1, block_ms: int = 5000) -> List[Dict[str, Any]]:
        """
        consume message
        """
        try:
            result = self._client.xreadgroup(
                self.group_name,
                self.consumer_name,
                {self.stream_name: ">"},
                count=count,
                block=block_ms
            )
        except RedisError as e:
            logger.error("Failed to consume message (Stream=%s): %s", self.stream_name, e)
            return []

        messages = []
        for stream, entries in result:
            for msg_id, fields in entries:
                raw = fields.get(b"data") or fields.get("data")
                if raw is None:
                    continue
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                try:
                    data = json.loads(raw)
                    messages.append({"id": msg_id, "data": data})
                except json.JSONDecodeError:
                    logger.warning("无法解析消息数据: %s", raw)
        return messages

    def ack(self, msg_id: str):
        """confirm message processing completed"""
        try:
            self._client.xack(self.stream_name, self.group_name, msg_id)
        except RedisError as e:
            logger.warning("Failed to confirm message (msg_id=%s): %s", msg_id, e)

    def send_to_dlq(self, msg_data: Dict[str, Any]):
        """
        write message to dlq
        """
        payload = {
            "data": json.dumps(msg_data),
            "error_time": datetime.now(timezone.utc).isoformat()
        }
        try:
            self._client.xadd(self.dlq_stream, payload)
            logger.warning("Message have been moved to dlq : %s", self.dlq_stream)
        except RedisError as e:
            logger.error("Failed to write to the dead-letter queue: %s", e)