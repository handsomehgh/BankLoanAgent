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
from modules.tools import ToolExecutor
from modules.tools.tool_selector import ToolSelector
from utils.serialize_utils.seq_generator import SequenceGenerator

logger = logging.getLogger(__name__)


def after_loan_response_node(
        state: AfterLoanState,
        config: RunnableConfig,
        registry: ConfigRegistry,
        llm_client: RobustLLM,
        tool_executor: ToolExecutor,
        seq_generator: SequenceGenerator,
        tool_selector: ToolSelector,
        classifier: AfterLoanClassifier
) -> Dict[str, Any]:
    executor = AgentNodeExecutor(
        agent_module=RegistryModules.AFTER_LOAN,
        agent_name=AgentName.AFTER_LOAN.value,
        registry=registry,
        llm_client=llm_client,
        tool_executor=tool_executor,
        seq_generator=seq_generator,
        tool_selector=tool_selector,
        classifier=classifier
    )
    return executor.execute(state, config)
