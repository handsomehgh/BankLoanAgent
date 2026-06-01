# author hgh
# version 1.0
from typing import Dict, Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from modules.agent.constants import StateFields
from modules.agent.multi_agent_state import SupervisorState


class HumanHandoffResponseNode:
    def __call__(self, state: SupervisorState,config: RunnableConfig) -> Dict[str, Any]:
        user_message = "您的问题已转接至人工客服，请稍候。"
        return {
            StateFields.MESSAGES.value: [AIMessage(content=user_message)],
        }

