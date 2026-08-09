# author hgh
# version 1.0
import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from config.global_constant.constants import MemoryType, RegistryModules, ConfigFields
from config.registry import ConfigRegistry
from modules.agent.constants import StateFields
from modules.agent.multi_agent_state import SupervisorState
from modules.memory.memory_utils.base_memory_utils import get_message_index, format_messages
from modules.module_services.chat_models import RobustLLM
from modules.tools.common_utils import assign_message_index
from utils.monitor_utils.metrics import record_llm_metrics
from utils.serialize_utils.seq_generator import SequenceGenerator

logger = logging.getLogger(__name__)


class DirectReplyNode:
    def __init__(self, llm_client: RobustLLM, registry: ConfigRegistry,seq_generator: SequenceGenerator):
        self.llm_client = llm_client
        self.registry = registry
        self.seq_generator = seq_generator

    async def __call__(self, state: SupervisorState, config: RunnableConfig):
        user_id = state.get(StateFields.USER_ID.value)
        session_id = config.get(ConfigFields.CONFIGURABLE.value,{}).get(ConfigFields.THREAD_ID.value)
        if self.llm_client is None:
            logger.warning("[DirectReply] LLM client not injected, unable to perform direct reply judgment")
            return {StateFields.SHOULD_SKIP_SUPERVISOR.value: False}

        # 1. obtain user messages
        messages = state.get(StateFields.MESSAGES.value, [])
        if not messages:
            return {StateFields.SHOULD_SKIP_SUPERVISOR.value: False}
        user_query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content.strip()
                break
        if not user_query:
            return {StateFields.SHOULD_SKIP_SUPERVISOR.value: False}

        # 2. organize context
        formatted = state.get(StateFields.FORMATTED_CONTEXT.value, {})
        user_profile = formatted.get(MemoryType.USER_PROFILE.value, "")
        interaction_log = formatted.get(MemoryType.INTERACTION_LOG.value, "")
        recent_conversation = self._extract_recent_conversation(state)

        # 3. organize prompt
        reply_config = self.registry.get_config(RegistryModules.DIRECT_REPLY.value)
        sys_prompt = reply_config.system_prompt
        prompt = sys_prompt.format(
            user_query=user_query,
            user_profile=user_profile or "暂无相关信息",
            recent_conversation=recent_conversation or "暂无相关信息",
            interaction_log=interaction_log or "暂无相关信息",
        )

        # 3. call llm
        try:
            total_start = time.monotonic()
            response = await self.llm_client.ainvoke([
                SystemMessage(content="你是一个有帮助的银行客服助手。"),
                HumanMessage(content=prompt)
            ])
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                record_llm_metrics(provider=self.llm_client.provider,
                                   total_tokens=response.usage_metadata.get("total_tokens", 0),
                                   duration_ms=(time.monotonic() - total_start) * 1000)
            llm_output = response.content.strip()
        except Exception as e:
            logger.error("[DirectReply] direct reply failed call LLM: %s", e)
            return {StateFields.SHOULD_SKIP_SUPERVISOR.value: False}

        # 4. deal result
        if "ROUTE" in llm_output:
            logger.info("[DirectReply] LLM decide release to Supervisor")
            return {StateFields.SHOULD_SKIP_SUPERVISOR.value: False}
        else:
            logger.info("[DirectReply] LLM direct reply: %s", llm_output[:50])
            res_message = AIMessage(content=llm_output)
            assign_message_index(res_message,user_id,session_id,self.seq_generator)
            return {
                StateFields.MESSAGES.value: [res_message],
                StateFields.SHOULD_SKIP_SUPERVISOR.value: True,
            }

    def _extract_recent_conversation(self, state: SupervisorState) -> str:
        messages = state.get(StateFields.MESSAGES.value, [])
        last_logged_index = state.get(StateFields.LAST_LOGGED_MESSAGE_INDEX.value)

        if last_logged_index is None:
            recent = messages[-5:] if len(messages) > 5 else messages
        else:
            recent = [m for m in messages
                      if get_message_index(m) is not None and get_message_index(m) > last_logged_index]
            if not recent:
                recent = messages[-5:] if len(messages) > 5 else messages

        return format_messages(recent)