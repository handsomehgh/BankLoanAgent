# author hgh
# version 1.0
from functools import partial

from langgraph.constants import END
from langgraph.graph import StateGraph

from agent.checkpointer import get_checkpointer
from agent.nodes import retrieve_memory_node, compliance_guard_node, call_model_node, extract_profile_node, \
    log_interaction_node
from agent.state import AgentState
from memory.base_memory_store import BaseMemoryStore
from config.constants import AgentNodeName, StateFields
from retriever.base import BaseRetriever


def build_graph(memory_store: BaseMemoryStore, retriever: BaseRetriever):
    """
    build and compile langgraph workflow

    Args:
        memory_store (BaseMemoryStore): store instance of long term memory
        retriever (BaseRetriever): retriever instance

    Returns:
        compiled langgraph app with checkpointer
    """
    workflow = StateGraph(AgentState)

    # =========================== register node =================================
    # retrieve node:get long term memory for vector store
    workflow.add_node(AgentNodeName.RETRIEVE.value, partial(retrieve_memory_node, retriever))

    # compliance interception node: scan user input,and if high-risk rules are hit,directly return interception phrasing
    workflow.add_node(AgentNodeName.COMPLIANCE_GUARD.value, partial(compliance_guard_node, memory_store))

    # generate answer node,inject context and invoke llm
    workflow.add_node(AgentNodeName.CALL_MODEL.value, call_model_node)

    # profile extract node,extract user profiles from conversation and store them in long term memory
    workflow.add_node(AgentNodeName.EXTRACT_PROFILE.value, partial(extract_profile_node, memory_store))

    # interaction log node,generate,generate a conversation summary and store it in the interaction trajectory memory
    workflow.add_node(AgentNodeName.LOG_INTERACTION.value, partial(log_interaction_node, memory_store))

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
    checkpointer = get_checkpointer()
    return workflow.compile(checkpointer)
