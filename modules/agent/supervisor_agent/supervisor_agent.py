# author hgh
# version 1.0
"""
Core implementation of supervisor agent
responsibilities: intent recognition,routing distribution,result integration,fallback response
"""
import logging

from langgraph.graph import StateGraph

from config.models.memory_config import MemorySystemConfig
from config.registry import ConfigRegistry
from modules.agent.constants import AgentNodeName
from modules.agent.multi_agent_state import SupervisorState

from modules.agent.supervisor_agent.supervisor_route_node import SupervisorRouteNode
from modules.memory.base import BaseRetriever
from modules.module_services.chat_models import RobustLLM
from modules.retrieval.retrieval_service import RetrievalService
from utils.serialize_utils.seq_generator import SequenceGenerator

logger = logging.getLogger(__name__)


class SupervisorAgent:
    """
    supervisor agent core class

    encapsulate all logic such as memory retrieval,rule routing,LLM dynamic routing,result integration,and fallback response,
    instantiate it as a callable object and use it directly as a Langgraph node
    """

    def __init__(
            self,
            memory_retriever: BaseRetriever,
            seq_generator: SequenceGenerator,
            registry: ConfigRegistry,
            llm_client: RobustLLM,
            memory_config: MemorySystemConfig,
            knowledge_retrieve: RetrievalService

    ):
        self.memory_retriever = memory_retriever
        self.seq_generator = seq_generator
        self.registry = registry
        self.llm_client = llm_client
        self.memory_config = memory_config
        self.knowledge_retrieve = knowledge_retrieve

    def build_graph(self) -> StateGraph:
        """build loanAdvisor subgraph"""
        graph = StateGraph(SupervisorState)

        supervisor_route_node = SupervisorRouteNode(self.registry, self.llm_client, self.knowledge_retrieve,self.seq_generator)

        graph.add_node(AgentNodeName.SUPERVISOR_ROUTE_NODE.value, supervisor_route_node)

        graph.set_entry_point(AgentNodeName.SUPERVISOR_ROUTE_NODE.value)
        graph.set_finish_point(AgentNodeName.SUPERVISOR_ROUTE_NODE.value)

        return graph.compile()


