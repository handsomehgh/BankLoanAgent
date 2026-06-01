# author hgh
# version 1.0
"""
result aggregator
responsibility: combines responses from multiple agents
"""
import logging
from typing import Dict, Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from modules.agent.constants import StateFields, AgentName
from modules.agent.multi_agent_state import SupervisorState, AgentResponse

logger = logging.getLogger(__name__)

AGENT_PRIORITY = [
    AgentName.LOAN_ADVISOR,
    AgentName.RISK_ASSESSMENT,
    AgentName.AFTER_LOAN,
]

FALLBACK_MESSAGE = "抱歉，我暂时无法处理您的问题。您可以拨打我行客服热线 95333 获取帮助。"


class ResultAggregatorAgent:
    """
    Result aggregation node: merge the responses of multiple sub-Agents into a final reply
    """

    def __call__(self, state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
        agent_responses: Dict[str, AgentResponse] = state.get(StateFields.AGENT_RESPONSES.value, {})
        if not agent_responses:
            return {StateFields.MESSAGES.value: [AIMessage(content=FALLBACK_MESSAGE)]}

        # sorted by priority
        sorted_resps = sorted(
            agent_responses.items(),
            key=lambda x: AGENT_PRIORITY.index(x[0]) if x[0] in AGENT_PRIORITY else 99,
        )

        combined = ""
        for _, resp in sorted_resps:
            if resp.content:
                combined += f"{resp.content}\n\n"

        if not combined.strip():
            combined = FALLBACK_MESSAGE

        return {
            StateFields.MESSAGES.value: [AIMessage(content=combined.strip())],
            StateFields.AGENT_RESPONSES.value: {},
            StateFields.AGENT_CONTEXT.value: None,
            StateFields.NEXT_AGENTS.value: [],
            StateFields.TRIGGER_HUMAN_HANDOFF.value: False
        }
