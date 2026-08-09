# author hgh
# version 1.0
import logging
from typing import Dict, Any

from langchain_core.runnables import RunnableConfig

from config.global_constant.constants import RegistryModules
from config.registry import ConfigRegistry
from modules.agent.constants import AgentName
from modules.agent.multi_agent_state import AfterLoanState
from modules.agent.nodes.agent_node_executor import AgentNodeExecutor
from modules.module_services.chat_models import RobustLLM
from modules.module_services.classifier.after_loan_classifier import AfterLoanClassifier
from modules.skills.skill_executor import SkillExecutor
from modules.skills.skill_registry import SkillRegistry
from modules.tools import ToolExecutor
from modules.tools.tool_selector import ToolSelector
from utils.serialize_utils.seq_generator import SequenceGenerator

logger = logging.getLogger(__name__)


def _build_executor(
        registry: ConfigRegistry,
        llm_client: RobustLLM,
        tool_executor: ToolExecutor,
        seq_generator: SequenceGenerator,
        tool_selector: ToolSelector,
        classifier: AfterLoanClassifier,
        skill_executor: SkillExecutor,
        skill_selector: SkillRegistry
) -> AgentNodeExecutor:
    return AgentNodeExecutor(
        agent_module=RegistryModules.AFTER_LOAN,
        agent_name=AgentName.AFTER_LOAN.value,
        registry=registry,
        llm_client=llm_client,
        tool_executor=tool_executor,
        seq_generator=seq_generator,
        tool_selector=tool_selector,
        classifier=classifier,
        skill_executor=skill_executor,
        skill_selector=skill_selector
    )


async def after_loan_decision_node(
        state: AfterLoanState,
        config: RunnableConfig,
        registry: ConfigRegistry,
        llm_client: RobustLLM,
        tool_executor: ToolExecutor,
        seq_generator: SequenceGenerator,
        tool_selector: ToolSelector,
        classifier: AfterLoanClassifier,
        skill_executor: SkillExecutor,
        skill_selector: SkillRegistry

) -> Dict[str, Any]:
    executor = _build_executor(registry, llm_client, tool_executor, seq_generator, tool_selector,
                                classifier, skill_executor, skill_selector)
    return await executor.decide(state, config)


async def after_loan_response_node(
        state: AfterLoanState,
        config: RunnableConfig,
        registry: ConfigRegistry,
        llm_client: RobustLLM,
        tool_executor: ToolExecutor,
        seq_generator: SequenceGenerator,
        tool_selector: ToolSelector,
        classifier: AfterLoanClassifier,
        skill_executor: SkillExecutor,
        skill_selector: SkillRegistry

) -> Dict[str, Any]:
    executor = _build_executor(registry, llm_client, tool_executor, seq_generator, tool_selector,
                                classifier, skill_executor, skill_selector)
    return await executor.reply(state, config)
