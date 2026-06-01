# author hgh
# version 1.0
import logging

from langchain_core.messages import messages_from_dict

from config.global_constant.constants import MemoryType
from config.global_constant.fields import CommonFields
from infra.database.redis_manager import RedisManager
from infra.message_queue import Message
from modules.agent.constants import StreamName, StateFields
from modules.consumer.base_message_consumer import BaseMessageConsumer
from modules.memory.memory_business_store.base_memory_store import BaseMemoryStore
from modules.memory.memory_constant.constants import MemorySource, MemoryStatus, InteractionEventType, \
    InteractionSentiment
from modules.module_services.SummaryGenerator import SummaryGenerator
from modules.module_services.sentiment_analyser import SentimentAnalyzer

logger = logging.getLogger(__name__)

class InteractionLogConsumer(BaseMessageConsumer):
    def __init__(self, stream_name: str,group_name: str,redis_manager: RedisManager,memory_store: BaseMemoryStore,summary_generator: SummaryGenerator,sentiment_analyzer: SentimentAnalyzer):
        super().__init__(redis_manager,stream_name,group_name,"worker-1")
        self.memory_store = memory_store
        self.summary_generator = summary_generator
        self.sentiment_analyzer = sentiment_analyzer

    def handle(self,data: Message):
        trace_id = data.trace_id
        session_id = data.payload.get(CommonFields.SESSION_ID)
        event_type = data.event_type
        user_id = data.payload.get(CommonFields.USER_ID)
        content = data.payload.get(CommonFields.CONTENT)
        agent_name = data.payload.get(CommonFields.AGENT_NAME)
        message_dict = data.payload.get(StateFields.MESSAGES.value,{})
        timestamp = data.timestamp
        if not content:
            logging.info(f"[InteractionLogConsumer] received no conversation summary")
            return
        logging.info(f"[InteractionLogConsumer] start processing interaction log extract,user_id={user_id},session_id={session_id},content={content}")

        logger.info("Generating interaction summary for session_id=%s", session_id)
        summary = self.summary_generator.generate(content, messages_from_dict(message_dict) if not agent_name else None)
        logger.info("Interaction summary generated: '%.60s...'", summary)

        # detect sentiment
        logger.debug("Analyzing sentiment for summary")
        sentiment = self.sentiment_analyzer.analyze(summary)
        logger.info("Detected sentiment: %s", sentiment)

        # build log memory data
        metadata = {
            CommonFields.SOURCE: MemorySource.AUTO_SUMMARY if event_type == StreamName.INTERACTION_LOG  else agent_name,
            CommonFields.STATUS: MemoryStatus.ACTIVE,
            CommonFields.CONFIDENCE: 1.0,
            CommonFields.EVENT_TYPE: InteractionEventType.INQUIRY,
            CommonFields.SESSION_ID: session_id,
            CommonFields.SENTIMENT: InteractionSentiment(sentiment),
            CommonFields.KEY_ENTITIES: [],
            CommonFields.TIMESTAMP: timestamp,
        }

        # add to memory
        try:
            self.memory_store.add_memory(
                user_id=user_id,
                content=summary,
                memory_type=MemoryType.INTERACTION_LOG,
                metadata=metadata
            )
            logger.info("Logged interaction for session %s", session_id)
        except Exception as e:
            logger.error(
                "Failed to write interaction log for session %s: %s",
                session_id, e, exc_info=True
            )
            raise









