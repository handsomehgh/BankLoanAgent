# author hgh
# version 1.0
"""
LoanAdvisor Agent-Loan consultation expert(independent subgraph)
Responsibilities: product consultation,application guidance,credit limit calculation,interest inquiry

Design Principles:
- Independent StateGraph, communicates with Supervisor through AgentContext
- All external dependencies are injected via constructor
- Reserved interfaces for tool calls, currently using LLM for direct generation mode
- Strict error handling, do not throw exceptions that affect the main process
"""
import logging
from functools import partial

from langgraph.graph import StateGraph

from config.registry import ConfigRegistry
from modules.agent.constants import AgentNodeName
from modules.agent.loan_advisor_agent.loan_advisor_response_node import loan_advisor_response_node
from modules.agent.multi_agent_state import LoanAdvisorState
from modules.module_services.chat_models import RobustLLM
from modules.module_services.classifier.loan_advisor_classifier import LoanAdvisorClassifier
from modules.skills.skill_executor import SkillExecutor
from modules.skills.skill_registry import SkillRegistry
from modules.tools import ToolExecutor
from modules.tools.tool_selector import ToolSelector
from utils.serialize_utils.seq_generator import SequenceGenerator

logger = logging.getLogger(__name__)


class LoanAdvisorAgent:
    """implementation of loanAdvisor agent"""

    def __init__(
            self,
            llm_client: RobustLLM,
            registry: ConfigRegistry,
            tool_executor: ToolExecutor,
            seq_generator: SequenceGenerator,
            tool_selector: ToolSelector,
            classifier: LoanAdvisorClassifier,
            skill_executor: SkillExecutor,
            skill_selector: SkillRegistry
    ):
        self.llm_client = llm_client
        self.registry = registry
        self.tool_executor = tool_executor
        self.seq_generator = seq_generator
        self.tool_selector = tool_selector
        self.classifier = classifier
        self.skill_executor = skill_executor
        self.skill_selector = skill_selector

    def build_graph(self) -> StateGraph:
        """build loanAdvisor subgraph"""
        graph = StateGraph(LoanAdvisorState)
        graph.add_node(AgentNodeName.LOAN_ADVISOR_RESPONSE.value,
                       partial(
                           loan_advisor_response_node,
                           llm_client=self.llm_client,
                           registry=self.registry,
                           tool_executor=self.tool_executor,
                           seq_generator=self.seq_generator,
                           tool_selector=self.tool_selector,
                           classifier=self.classifier,
                           skill_executor=self.skill_executor,
                           skill_selector=self.skill_selector
                       )
        )
        graph.set_entry_point(AgentNodeName.LOAN_ADVISOR_RESPONSE.value)
        graph.set_finish_point(AgentNodeName.LOAN_ADVISOR_RESPONSE.value)
        return graph.compile()
