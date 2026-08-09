# author hgh
# version 1.0
import logging
from typing import Dict, Any

from langchain_core.runnables import RunnableConfig

from config.registry import ConfigRegistry
from infra.cache.cache_manager import CacheManager
from infra.database.mysql_manager import DatabaseManager
from modules.agent.nodes.agent_decision_node import AgentDecisionNode
from modules.agent.nodes.agent_reply_node import AgentReplyNode
from modules.agent.nodes.proactive_suggestion_gate import ProactiveSuggestionGate
from modules.module_services.chat_models import RobustLLM
from modules.module_services.classifier.loan_advisor_classifier import LoanAdvisorClassifier
from modules.module_services.suggestion_timing_classifier import SuggestionTimingClassifier
from modules.skills.skill_executor import SkillExecutor
from modules.skills.skill_registry import SkillRegistry
from modules.tools import ToolExecutor
from config.global_constant.constants import RegistryModules
from modules.agent.constants import AgentName
from modules.agent.multi_agent_state import LoanAdvisorState
from modules.tools.tool_selector import ToolSelector
from utils.serialize_utils.seq_generator import SequenceGenerator

logger = logging.getLogger(__name__)


def _build_decision_node(
        registry: ConfigRegistry,
        llm_client: RobustLLM,
        tool_executor: ToolExecutor,
        tool_selector: ToolSelector,
        classifier: LoanAdvisorClassifier,
        skill_executor: SkillExecutor,
        skill_selector: SkillRegistry
) -> AgentDecisionNode:
    return AgentDecisionNode(
        agent_module=RegistryModules.LOAN_ADVISOR,
        agent_name=AgentName.LOAN_ADVISOR.value,
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
        agent_module=RegistryModules.LOAN_ADVISOR,
        agent_name=AgentName.LOAN_ADVISOR.value,
        registry=registry,
        llm_client=llm_client,
        seq_generator=seq_generator
    )


async def loan_advisor_decision_node(
        state: LoanAdvisorState,
        config: RunnableConfig,
        registry: ConfigRegistry,
        llm_client: RobustLLM,
        tool_executor: ToolExecutor,
        tool_selector: ToolSelector,
        classifier: LoanAdvisorClassifier,
        skill_executor: SkillExecutor,
        skill_selector: SkillRegistry
) -> Dict[str, Any]:
    decision_node = _build_decision_node(registry, llm_client, tool_executor, tool_selector,
                                         classifier, skill_executor, skill_selector)
    return await decision_node.decide(state, config)


async def loan_advisor_response_node(
        state: LoanAdvisorState,
        config: RunnableConfig,
        registry: ConfigRegistry,
        llm_client: RobustLLM,
        seq_generator: SequenceGenerator
) -> Dict[str, Any]:
    reply_node = _build_reply_node(registry, llm_client, seq_generator)
    return await reply_node.reply(state, config)


async def loan_advisor_proactive_gate_node(
        state: LoanAdvisorState,
        config: RunnableConfig,
        suggestion_classifier: SuggestionTimingClassifier,
        suggestion_cache: CacheManager,
        db_manager: DatabaseManager
) -> Dict[str, Any]:
    """gate节点：只写 proactive_hint，永不生成用户可见文本，内部任何异常都降级为不邀请"""
    gate = ProactiveSuggestionGate(
        classifier=suggestion_classifier,
        cache_manager=suggestion_cache,
        db_manager=db_manager
    )
    return await gate.check(state, config)
