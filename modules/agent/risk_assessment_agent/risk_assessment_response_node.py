# author hgh
# version 1.0
import logging
from typing import Dict, Any

from langchain_core.runnables import RunnableConfig

from config.global_constant.constants import RegistryModules
from config.registry import ConfigRegistry
from modules.agent.constants import StateFields, AgentName
from modules.agent.multi_agent_state import RiskAssessmentState
from modules.agent.nodes.agent_decision_node import AgentDecisionNode
from modules.agent.nodes.agent_reply_node import AgentReplyNode
from modules.module_services.chat_models import RobustLLM
from modules.module_services.classifier.risk_assessment_classifier import RiskAssessmentClassifier
from modules.skills.skill_executor import SkillExecutor
from modules.skills.skill_registry import SkillRegistry
from modules.tools import ToolExecutor
from modules.tools.tool_selector import ToolSelector
from utils.serialize_utils.seq_generator import SequenceGenerator

logger = logging.getLogger(__name__)


def _build_decision_node(
        registry: ConfigRegistry,
        llm_client: RobustLLM,
        tool_executor: ToolExecutor,
        tool_selector: ToolSelector,
        classifier: RiskAssessmentClassifier,
        skill_executor: SkillExecutor,
        skill_selector: SkillRegistry
) -> AgentDecisionNode:
    return AgentDecisionNode(
        agent_module=RegistryModules.RISK_ASSESSMENT,
        agent_name=AgentName.RISK_ASSESSMENT.value,
        registry=registry,
        llm_client=llm_client,
        tool_executor=tool_executor,
        tool_selector=tool_selector,
        classifier=classifier,
        skill_executor=skill_executor,
        skill_selector=skill_selector
    )


def _build_reply_node(
        registry: ConfigRegistry,
        llm_client: RobustLLM,
        seq_generator: SequenceGenerator
) -> AgentReplyNode:
    return AgentReplyNode(
        agent_module=RegistryModules.RISK_ASSESSMENT,
        agent_name=AgentName.RISK_ASSESSMENT.value,
        registry=registry,
        llm_client=llm_client,
        seq_generator=seq_generator,
        post_process=_risk_post_process
    )


async def risk_assessment_decision_node(
        state: RiskAssessmentState,
        config: RunnableConfig,
        registry: ConfigRegistry,
        llm_client: RobustLLM,
        tool_executor: ToolExecutor,
        tool_selector: ToolSelector,
        classifier: RiskAssessmentClassifier,
        skill_executor: SkillExecutor,
        skill_selector: SkillRegistry
) -> Dict[str, Any]:
    decision_node = _build_decision_node(registry, llm_client, tool_executor, tool_selector,
                                         classifier, skill_executor, skill_selector)
    return await decision_node.decide(state, config)


async def risk_assessment_response_node(
        state: RiskAssessmentState,
        config: RunnableConfig,
        registry: ConfigRegistry,
        llm_client: RobustLLM,
        seq_generator: SequenceGenerator
) -> Dict[str, Any]:
    reply_node = _build_reply_node(registry, llm_client, seq_generator)
    return await reply_node.reply(state, config)


def _detect_severe_risk(query: str, profile: str) -> bool:
    """简单规则检测严重风险（后续可升级为更复杂逻辑）"""
    # 征信“连三累六”关键词
    severe_keywords = ["连三累六", "连续三个月逾期", "累计六次逾期"]
    combined = query + profile
    for kw in severe_keywords:
        if kw in combined:
            return True
    return False


def _risk_post_process(final_msg, context, messages):
    """检测严重风险，触发转人工"""
    trigger_handoff = _detect_severe_risk(final_msg.content, context.user_profile_summary if context else "")
    if trigger_handoff and "转接人工" not in final_msg.content:
        final_msg.content += "\n\n您的情况较为严重，建议转接人工客服进一步沟通。"
    return {StateFields.TRIGGER_HUMAN_HANDOFF.value: trigger_handoff}
