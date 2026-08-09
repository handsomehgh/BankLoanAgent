# author hgh
# version 1.0
import logging
from typing import Dict, Any

from langchain_core.runnables import RunnableConfig

from config.registry import ConfigRegistry
from modules.agent.nodes.agent_node_executor import AgentNodeExecutor
from modules.module_services.chat_models import RobustLLM
from modules.module_services.classifier.loan_advisor_classifier import LoanAdvisorClassifier
from modules.skills.skill_executor import SkillExecutor
from modules.skills.skill_registry import SkillRegistry
from modules.tools import ToolExecutor
from config.global_constant.constants import RegistryModules
from modules.agent.constants import AgentName
from modules.agent.multi_agent_state import LoanAdvisorState
from modules.tools.tool_selector import ToolSelector
from utils.serialize_utils.seq_generator import SequenceGenerator

logger = logging.getLogger(__name__)


async def loan_advisor_response_node(
        state: LoanAdvisorState,
        config: RunnableConfig,
        registry: ConfigRegistry,
        llm_client: RobustLLM,
        tool_executor: ToolExecutor,
        seq_generator: SequenceGenerator,
        tool_selector: ToolSelector,
        classifier: LoanAdvisorClassifier,
        skill_executor: SkillExecutor,
        skill_selector: SkillRegistry
) -> Dict[str, Any]:
    executor = AgentNodeExecutor(
        agent_module=RegistryModules.LOAN_ADVISOR,
        agent_name=AgentName.LOAN_ADVISOR.value,
        registry=registry,
        llm_client=llm_client,
        tool_executor=tool_executor,
        seq_generator=seq_generator,
        tool_selector=tool_selector,
        classifier=classifier,
        skill_executor=skill_executor,
        skill_selector=skill_selector
    )
    return await executor.execute(state, config)
