# author hgh
# version 1.0
from functools import partial

from langchain_core.messages import HumanMessage
from langgraph.constants import END
from langgraph.graph import StateGraph

from config.global_constant.constants import RegistryModules
from modules.agent.constants import AgentNodeName, StateFields
from modules.agent.checkpointer import get_checkpointer
from modules.agent.nodes.call_llm_node import call_model_node
from modules.agent.nodes.compliance_guard_node import compliance_guard_node
from modules.agent.nodes.extract_profile_node import extract_profile_node
from modules.agent.nodes.retrieval_knowledge_node import retrieval_knowledge_node
from modules.agent.nodes.retrieve_memory_node import retrieve_memory_node
from modules.agent.nodes.summary_interaction_node import log_interaction_node
from modules.agent.state import AgentState
from modules.module_services.agent_services import AgentServices


def build_graph(services: AgentServices):
    """
    build and compile langgraph workflow

    Args:
        services: all services and configuration

    Returns:
        compiled langgraph app with checkpointer
    """
    workflow = StateGraph(AgentState)
    memory_cfg = services.registry.get_config(RegistryModules.MEMORY_SYSTEM)
    retrieval_cfg = services.registry.get_config(RegistryModules.RETRIEVAL)

    # =========================== register node =================================
    # retrieve node:get long term memory for vector store
    workflow.add_node(AgentNodeName.RETRIEVE_MEMORY, partial(retrieve_memory_node, retrieval=services.memory_retriever))

    # retrieve knowledge node: get relevant content from knowledge base
    workflow.add_node(AgentNodeName.RETRIEVE_KNOWLEDGE,
                      partial(retrieval_knowledge_node, retrieval_service=services.knowledge_retriever,
                              retrieval_config=retrieval_cfg))

    # compliance interception node: scan user input,and if high-risk rules are hit,directly return interception phrasing
    workflow.add_node(AgentNodeName.COMPLIANCE_GUARD, partial(compliance_guard_node, memory_config=memory_cfg))

    # generate answer node,inject context and invoke llm
    workflow.add_node(AgentNodeName.CALL_MODEL,
                      partial(call_model_node, memory_config=memory_cfg, llm_client=services.creative_llm))

    # profile extract node,extract user profiles from conversation and store them in long term memory
    workflow.add_node(AgentNodeName.EXTRACT_PROFILE,
                      partial(extract_profile_node, memory_store=services.memory_store, profile_gate=services.profile_gate,
                              memory_config=memory_cfg,evidence_infer=services.evidence_infer, profile_extractor=services.profile_extractor))

    # interaction log node,generate,generate a conversation summary and store it in the interaction trajectory memory
    workflow.add_node(AgentNodeName.LOG_INTERACTION,
                      partial(log_interaction_node, memory_store=services.memory_store, memory_config=memory_cfg,
                              summary_generator=services.summary_generator,sentiment_analyzer=services.sentiment_analyzer))

    # =========================== define entry =================================
    workflow.set_entry_point(AgentNodeName.RETRIEVE_MEMORY)

    # ============================ define edge ===================================
    # enter compliance check after the search is completed
    workflow.add_edge(AgentNodeName.RETRIEVE_MEMORY, AgentNodeName.COMPLIANCE_GUARD)

    # retrieval knowledge router
    retrieve_router = services.retrieval_router
    def routing_decision(state: AgentState) -> str:
        messages = state.get(StateFields.MESSAGES.value,[])
        query = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                query = msg.content
                break
        return (AgentNodeName.RETRIEVE_KNOWLEDGE if retrieve_router.should_retrieve(query) else AgentNodeName.CALL_MODEL)


    # conditional edge after compliance check,if intercepted,it ends directly,otherwise,it enters retrieval knowledge node
    workflow.add_conditional_edges(
        AgentNodeName.COMPLIANCE_GUARD,
        lambda s: END if s.get(StateFields.SHOULD_SKIP_LLM) else routing_decision(s),
        {
            AgentNodeName.RETRIEVE_KNOWLEDGE: AgentNodeName.RETRIEVE_KNOWLEDGE,
            AgentNodeName.CALL_MODEL: AgentNodeName.CALL_MODEL,
            END: END
        }
    )

    # after retrieval node,proceed to call llm node
    workflow.add_edge(AgentNodeName.RETRIEVE_KNOWLEDGE, AgentNodeName.CALL_MODEL)

    # after generating the answer, proceed to profile extraction
    workflow.add_edge(AgentNodeName.CALL_MODEL, AgentNodeName.EXTRACT_PROFILE)

    # after the profile is extracted, it enters the interactive log record
    workflow.add_edge(AgentNodeName.EXTRACT_PROFILE, AgentNodeName.LOG_INTERACTION)

    # end after logging interaction
    workflow.add_edge(AgentNodeName.LOG_INTERACTION, END)

    # ============================= compile and inject checkpointer ========================
    checkpointer = get_checkpointer(retrieval_cfg)
    return workflow.compile(checkpointer)
