# author hgh
# version 1.0
import logging
from typing import Dict, Any

from langchain_core.runnables import RunnableConfig

from config.global_constant.constants import RegistryModules
from config.registry import ConfigRegistry
from modules.agent.constants import StateFields, AgentName
from modules.agent.multi_agent_state import RiskAssessmentState
from modules.agent.nodes.agent_node_executor import AgentNodeExecutor
from modules.module_services.chat_models import RobustLLM
from modules.module_services.classifier.risk_assessment_classifier import RiskAssessmentClassifier
from modules.tools import ToolExecutor
from modules.tools.tool_selector import ToolSelector
from utils.serialize_utils.seq_generator import SequenceGenerator

logger = logging.getLogger(__name__)


def risk_assessment_response_node(
        state: RiskAssessmentState,
        config: RunnableConfig,
        registry: ConfigRegistry,
        llm_client: RobustLLM,
        tool_executor: ToolExecutor,
        seq_generator: SequenceGenerator,
        tool_selector: ToolSelector,
        classifier: RiskAssessmentClassifier
) -> Dict[str, Any]:
    executor = AgentNodeExecutor(
        agent_module=RegistryModules.RISK_ASSESSMENT,
        agent_name=AgentName.RISK_ASSESSMENT.value,
        registry=registry,
        llm_client=llm_client,
        tool_executor=tool_executor,
        seq_generator=seq_generator,
        tool_selector=tool_selector,
        post_process=_risk_post_process,
        classifier=classifier
    )
    return executor.execute(state, config)


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
