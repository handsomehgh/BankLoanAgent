# author hgh
# version 1.0
import logging
from functools import partial

from langgraph.graph import StateGraph

from config.registry import ConfigRegistry
from modules.agent.constants import AgentNodeName
from modules.agent.multi_agent_state import RiskAssessmentState
from modules.agent.risk_assessment_agent.risk_assessment_response_node import risk_assessment_response_node, \
    risk_assessment_decision_node
from modules.module_services.chat_models import RobustLLM
from modules.module_services.classifier.risk_assessment_classifier import RiskAssessmentClassifier
from modules.skills.skill_executor import SkillExecutor
from modules.skills.skill_registry import SkillRegistry
from modules.tools import ToolExecutor
from modules.tools.tool_selector import ToolSelector
from utils.serialize_utils.seq_generator import SequenceGenerator

logger = logging.getLogger(__name__)

class RiskAssessmentAgent:
    def __init__(
            self,
            llm_client: RobustLLM,
            registry: ConfigRegistry,
            tool_executor: ToolExecutor,
            seq_generator: SequenceGenerator,
            tool_selector: ToolSelector,
            classifier: RiskAssessmentClassifier,
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
        graph = StateGraph(RiskAssessmentState)
        graph.add_node(AgentNodeName.RISK_ASSESSMENT_DECISION.value,
                       partial(risk_assessment_decision_node,
                               registry=self.registry,
                               llm_client=self.llm_client,
                               tool_executor=self.tool_executor,
                               tool_selector=self.tool_selector,
                               classifier=self.classifier,
                               skill_executor=self.skill_executor,
                               skill_selector=self.skill_selector))
        graph.add_node(AgentNodeName.RISK_ASSESSMENT_RESPONSE.value,
                       partial(risk_assessment_response_node,
                               registry=self.registry,
                               llm_client=self.llm_client,
                               seq_generator=self.seq_generator))

        graph.add_edge(AgentNodeName.RISK_ASSESSMENT_DECISION.value, AgentNodeName.RISK_ASSESSMENT_RESPONSE.value)
        graph.set_entry_point(AgentNodeName.RISK_ASSESSMENT_DECISION.value)
        graph.set_finish_point(AgentNodeName.RISK_ASSESSMENT_RESPONSE.value)
        return graph.compile()
