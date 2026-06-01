# author hgh
# version 1.0
import logging
from abc import ABC, abstractmethod

from config.global_constant.fields import CommonFields
from infra.database.redis_manager import RedisManager
from infra.message_queue import MessageConsumer, Message

logger = logging.getLogger(__name__)

class BaseMessageConsumer(ABC):
    def __init__(self, redis_manager: RedisManager,stream_name: str,group_name: str,consumer_name: str):
        self.consumer = MessageConsumer(redis_manager,stream_name,group_name,consumer_name)

    def process(self,block_ms: int = 5000):
        while True:
            try:
                messages = self.consumer.consume(count=1, block_ms=block_ms)
                for msg in messages:
                    try:
                        msg_str = msg[CommonFields.DATA]
                        self.handle(Message(**msg_str))
                        self.consumer.ack(msg[CommonFields.ID])
                    except Exception as e:
                        logger.error(f"Failed to process message: {e}, transfer to dlq")
                        msg_str = msg[CommonFields.DATA]
                        self.consumer.send_to_dlq(Message(**msg_str).payload)
                        self.consumer.ack(msg[CommonFields.ID])
            except Exception as e:
                logger.error(f"Consumer loop error: {e}", exc_info=True)
                # 短暂休眠避免快速重试，然后继续
                import time
                time.sleep(1)

    @abstractmethod
    def handle(self,data: Message):
        pass



