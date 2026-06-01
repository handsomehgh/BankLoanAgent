# author hgh
# version 1.0
"""
Message INDEX completion node
"""
import logging
from typing import Dict, Any

from langchain_core.runnables import RunnableConfig

from config.global_constant.constants import ConfigFields
from modules.agent.constants import StateFields, MessageCommonFields
from modules.agent.multi_agent_state import SupervisorState
from utils.serialize_utils.seq_generator import SequenceGenerator

logger = logging.getLogger(__name__)

def ensure_message_indexes_node(
    state: SupervisorState,
    config: RunnableConfig,
    seq_generator: SequenceGenerator
) -> Dict[str, Any]:
    messages = state.get(StateFields.MESSAGES.value, [])
    user_id = state.get(StateFields.USER_ID.value, "unknown")
    configurable = config.get(ConfigFields.CONFIGURABLE.value, {})
    session_id = configurable.get(ConfigFields.THREAD_ID.value, "unknown")

    updated = False
    for msg in messages:
        if not hasattr(msg, MessageCommonFields.ADDITIONAL_KWARGS) or msg.additional_kwargs is None:
            msg.additional_kwargs = {}
        if MessageCommonFields.MESSAGE_INDEX not in msg.additional_kwargs:
            idx = seq_generator.next_seq(user_id, session_id)
            msg.additional_kwargs[MessageCommonFields.MESSAGE_INDEX] = idx
            updated = True

    if updated:
        logger.debug("已为缺失消息补全全局序号")

    return {}
