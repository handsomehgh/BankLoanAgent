# author hgh
# version 1.0
from functools import partial

from langgraph.constants import END
from langgraph.graph import StateGraph

from config.global_constant.constants import RegistryModules
from config.registry import ConfigRegistry
from modules.agent.constants import AgentNodeName, StateFields
from modules.agent.checkpointer import get_checkpointer
from modules.agent.nodes.call_llm_node import call_model_node
from modules.agent.nodes.compliance_guard_node import compliance_guard_node
from modules.agent.nodes.extract_profile_node import extract_profile_node
from modules.agent.nodes.retrieve_memory_node import retrieve_memory_node
from modules.agent.nodes.summary_interaction_node import log_interaction_node
from modules.agent.state import AgentState
from modules.memory.base import BaseRetriever
from modules.memory.memory_business_store.base_memory_store import BaseMemoryStore
from modules.memory.memory_utils.profile_gate_util import ProfileGate


def build_graph(memory_store: BaseMemoryStore, retriever: BaseRetriever,registry: ConfigRegistry):
    """
    build and compile langgraph workflow

    Args:
        memory_store (BaseMemoryStore): store instance of long term memory
        retriever (BaseRetriever): retrieval instance

    Returns:
        compiled langgraph app with checkpointer
    """
    workflow = StateGraph(AgentState)

    memory_cfg = registry.get_config(RegistryModules.MEMORY_SYSTEM)
    retrieval_cfg = registry.get_config(RegistryModules.RETRIEVAL)
    profile_gate = ProfileGate(memory_cfg.memory_gate)

    # =========================== register node =================================
    # retrieve node:get long term memory for vector store
    workflow.add_node(AgentNodeName.RETRIEVE.value, partial(retrieve_memory_node, retrieval=retriever))

    # compliance interception node: scan user input,and if high-risk rules are hit,directly return interception phrasing
    workflow.add_node(AgentNodeName.COMPLIANCE_GUARD.value, partial(compliance_guard_node, memory_store=memory_store,memory_config=memory_cfg))

    # generate answer node,inject context and invoke llm
    workflow.add_node(AgentNodeName.CALL_MODEL.value, partial(call_model_node,memory_config=memory_cfg))

    # profile extract node,extract user profiles from conversation and store them in long term memory
    workflow.add_node(AgentNodeName.EXTRACT_PROFILE.value, partial(extract_profile_node, memory_store=memory_store,profile_gate=profile_gate,memory_config=memory_cfg))

    # interaction log node,generate,generate a conversation summary and store it in the interaction trajectory memory
    workflow.add_node(AgentNodeName.LOG_INTERACTION.value, partial(log_interaction_node, memory_store=memory_store,memory_config=memory_cfg))

    # =========================== define entry =================================
    workflow.set_entry_point(AgentNodeName.RETRIEVE.value)

    # ============================ define edge ===================================
    # enter compliance check after the search is completed
    workflow.add_edge(AgentNodeName.RETRIEVE.value, AgentNodeName.COMPLIANCE_GUARD.value)
    # conditional edge after compliance check,if intercepted,it ends directly,otherwise,it enters generation node
    workflow.add_conditional_edges(
        AgentNodeName.COMPLIANCE_GUARD,
        lambda s: END if s.get(StateFields.SHOULD_SKIP_LLM.value) else AgentNodeName.CALL_MODEL.value,
        {
            AgentNodeName.CALL_MODEL.value: AgentNodeName.CALL_MODEL.value,
            END: END
        }
    )

    # after generating the answer, proceed to profile extraction
    workflow.add_edge(AgentNodeName.CALL_MODEL.value, AgentNodeName.EXTRACT_PROFILE.value)

    # after the profile is extracted, it enters the interactive log record
    workflow.add_edge(AgentNodeName.EXTRACT_PROFILE.value, AgentNodeName.LOG_INTERACTION.value)

    # end after logging interaction
    workflow.add_edge(AgentNodeName.LOG_INTERACTION, END)

    # ============================= compile and inject checkpointer ========================
    checkpointer = get_checkpointer(retrieval_cfg)
    return workflow.compile(checkpointer)
