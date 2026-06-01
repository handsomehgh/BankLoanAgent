# modules/agent/graph.py
"""
多 Agent 图构建器
"""
import logging
from functools import partial

from langchain_core.runnables import RunnableConfig
from langgraph.constants import END
from langgraph.graph import StateGraph

from config.container import ApplicationContainer
from config.global_constant.constants import RegistryModules
from modules.agent.checkpointer import get_checkpointer
from modules.agent.constants import AgentNodeName, AgentName, StateFields, RouteTarget
from modules.agent.multi_agent_state import SupervisorState, AgentResponse
from modules.agent.nodes.ensure_message_index_node import ensure_message_indexes_node
from modules.agent.nodes.fanout_dispatcher import FanoutDispatcher

logger = logging.getLogger(__name__)


class MultiAgentGraphBuilder:
    """
    Multi-agent system diagram builder,responsible for registering nodes,defining edges,and compiling the final workflow。
    """

    def __init__(self, container: ApplicationContainer):
        self.container = container
        self.registry = container.config_registry()
        self.memory_config = self.registry.get_config(RegistryModules.MEMORY_SYSTEM)
        self.retrieval_config = self.registry.get_config(RegistryModules.RETRIEVAL)

        # compile subgraph
        self.supervisor_graph = container.supervisor_graph()
        self.loan_advisor_graph = container.loan_advisor_graph()
        self.risk_assessment_graph = container.risk_assessment_graph()
        self.after_loan_graph = container.after_loan_graph()
        self.compliance_prefilter_node = container.compliance_prefilter_node()
        self.summary_interaction_node = container.summary_interaction_node()
        self.extract_profile_node = container.extract_profile_node()
        self.result_aggregator_node = container.result_aggregator_node()
        self.human_handoff_notify_node = container.human_handoff_notify_node()
        self.human_handoff_interrupt_node = container.human_handoff_interrupt_node()
        self.direct_reply_node = container.direct_reply_node()
        self.memory_retrieve_node = container.memory_retrieve_node()

    def build(self) -> StateGraph:
        """build and compile the final stategraph"""
        workflow = StateGraph(SupervisorState)

        self._register_nodes(workflow)
        # entry node
        workflow.set_entry_point(AgentNodeName.COMPLIANCE_PREFILTER.value)
        self._define_edges(workflow)

        checkpointer = get_checkpointer(self.retrieval_config)
        return workflow.compile(checkpointer)

    def _register_nodes(self, workflow: StateGraph) -> None:
        """register node to workflow"""

        # pre compliance filter
        workflow.add_node(
            AgentNodeName.COMPLIANCE_PREFILTER.value,
            self.compliance_prefilter_node
        )

        # memory retrieve node
        workflow.add_node(
            AgentNodeName.MEMORY_RETRIEVE.value,
            self.memory_retrieve_node
        )

        # direct reply node
        workflow.add_node(
            AgentNodeName.DIRECT_REPLY.value,
            self.direct_reply_node
        )

        # Supervisor 路由
        workflow.add_node(
            AgentName.SUPERVISOR.value,
            self.supervisor_graph
        )

        # business node
        workflow.add_node(
            AgentName.LOAN_ADVISOR.value,
            self._dispatch_loan_advisor
        )
        workflow.add_node(
            AgentName.RISK_ASSESSMENT.value,
            self._dispatch_risk_assessment
        )
        workflow.add_node(
            AgentName.AFTER_LOAN.value,
            self._dispatch_after_loan
        )
        workflow.add_node(
            AgentNodeName.HUMAN_HANDOFF_NOTIFY.value,
            self.human_handoff_notify_node
        )
        workflow.add_node(
            AgentNodeName.HUMAN_HANDOFF_INTERRUPT.value,
            self.human_handoff_interrupt_node
        )

        # Fanout Distributor (used for parallel invocation of multiple Agents)
        agent_graphs = {
            AgentName.LOAN_ADVISOR.value: self.loan_advisor_graph,
            AgentName.RISK_ASSESSMENT.value: self.risk_assessment_graph,
            AgentName.AFTER_LOAN.value: self.after_loan_graph,
        }
        supervisor_cfg = self.registry.get_config(RegistryModules.SUPERVISOR)
        fanout_dispatcher = FanoutDispatcher(
            supervisor_cfg.agent_time_out,
            agent_graphs
        )
        workflow.add_node(AgentNodeName.FANOUT_DISPATCHER.value, fanout_dispatcher)

        # message sequence manager
        workflow.add_node(
            AgentNodeName.ENSURE_MESSAGE_INDEX.value,
            partial(
                ensure_message_indexes_node,
                seq_generator=self.container.seq_generator()
            )
        )

        # extract user profile node
        workflow.add_node(
            AgentNodeName.EXTRACT_PROFILE.value,
            self.extract_profile_node
        )

        # record interaction node
        workflow.add_node(
            AgentNodeName.LOG_INTERACTION.value,
            self.summary_interaction_node
        )

        # result aggregator node
        workflow.add_node(
            AgentNodeName.RESULT_AGGREGATOR.value,
            self.result_aggregator_node
        )

    def _define_edges(self, workflow: StateGraph) -> None:
        """Define all edges and conditional edges in the workflow"""

        # pre compliance → memory retrieve or END
        workflow.add_conditional_edges(
            AgentNodeName.COMPLIANCE_PREFILTER.value,
            lambda s: END if s.get(StateFields.SHOULD_SKIP_SUPERVISOR.value) else AgentNodeName.MEMORY_RETRIEVE.value,
            {AgentNodeName.MEMORY_RETRIEVE.value: AgentNodeName.MEMORY_RETRIEVE.value, END: END}
        )

        # memory retrieve -> direct reply
        workflow.add_edge(
            AgentNodeName.MEMORY_RETRIEVE.value,
            AgentNodeName.DIRECT_REPLY.value
        )

        # direct reply -> END or supervisor
        workflow.add_conditional_edges(
            AgentNodeName.DIRECT_REPLY.value,
            lambda s: END if s.get(StateFields.SHOULD_SKIP_SUPERVISOR.value) else AgentName.SUPERVISOR.value,
            {AgentName.SUPERVISOR.value: AgentName.SUPERVISOR.value, END: END}
        )

        # Supervisor → sub Agent or Fanout or End
        def route_supervisor(state: SupervisorState) -> str:
            next_agents = state.get(StateFields.NEXT_AGENTS.value, [])
            if len(next_agents) > 1:
                return AgentNodeName.FANOUT_DISPATCHER.value
            return next_agents[0]

        workflow.add_conditional_edges(
            AgentName.SUPERVISOR.value,
            route_supervisor,
            {
                END: END,
                AgentName.LOAN_ADVISOR.value: AgentName.LOAN_ADVISOR.value,
                AgentName.RISK_ASSESSMENT.value: AgentName.RISK_ASSESSMENT.value,
                AgentName.AFTER_LOAN.value: AgentName.AFTER_LOAN.value,
                AgentNodeName.FANOUT_DISPATCHER.value: AgentNodeName.FANOUT_DISPATCHER.value,
                AgentNodeName.HUMAN_HANDOFF_NOTIFY.value: AgentNodeName.HUMAN_HANDOFF_NOTIFY.value,
            }
        )

        # Fanout → Human_handoff or result aggregator
        workflow.add_conditional_edges(
            AgentNodeName.FANOUT_DISPATCHER.value,
            lambda s: AgentNodeName.HUMAN_HANDOFF_NOTIFY.value if s.get(
                StateFields.TRIGGER_HUMAN_HANDOFF.value) else AgentNodeName.RESULT_AGGREGATOR.value,
            {AgentNodeName.HUMAN_HANDOFF_NOTIFY.value: AgentNodeName.HUMAN_HANDOFF_NOTIFY.value,
             AgentNodeName.RESULT_AGGREGATOR.value: AgentNodeName.RESULT_AGGREGATOR.value}
        )

        # risk assessment → human_handoff or result aggregator
        workflow.add_conditional_edges(
            AgentName.RISK_ASSESSMENT.value,
            lambda s: AgentNodeName.HUMAN_HANDOFF_NOTIFY.value if s.get(
                StateFields.TRIGGER_HUMAN_HANDOFF.value) else AgentNodeName.RESULT_AGGREGATOR.value,
            {AgentNodeName.HUMAN_HANDOFF_NOTIFY.value: AgentNodeName.HUMAN_HANDOFF_NOTIFY.value,
             AgentNodeName.RESULT_AGGREGATOR.value: AgentNodeName.RESULT_AGGREGATOR.value}
        )

        # business Agent -> result aggregator
        workflow.add_edge(AgentName.LOAN_ADVISOR.value, AgentNodeName.RESULT_AGGREGATOR.value)
        workflow.add_edge(AgentName.AFTER_LOAN.value, AgentNodeName.RESULT_AGGREGATOR.value)

        # human_handoff → assign message index → extract profile → record interaction log → END
        workflow.add_edge(AgentNodeName.HUMAN_HANDOFF_NOTIFY.value, AgentNodeName.HUMAN_HANDOFF_INTERRUPT.value)
        workflow.add_edge(AgentNodeName.HUMAN_HANDOFF_INTERRUPT.value, AgentNodeName.ENSURE_MESSAGE_INDEX.value)
        workflow.add_edge(AgentNodeName.RESULT_AGGREGATOR.value, AgentNodeName.ENSURE_MESSAGE_INDEX.value)
        workflow.add_edge(AgentNodeName.ENSURE_MESSAGE_INDEX.value, AgentNodeName.EXTRACT_PROFILE.value)
        workflow.add_edge(AgentNodeName.EXTRACT_PROFILE.value, AgentNodeName.LOG_INTERACTION.value)
        workflow.add_edge(AgentNodeName.LOG_INTERACTION.value, END)

    # ================================================================
    # 子Subgraph Scheduling Wrapping Method (Manual State Mapping)
    # ================================================================
    def _dispatch_loan_advisor(self, state: SupervisorState, config: RunnableConfig) -> dict:
        ctx = state.get(StateFields.AGENT_CONTEXT.value, {}).get(AgentName.LOAN_ADVISOR.value)
        if not ctx:
            logger.warning(f"[LoanAdvisor] miss agent context")
            return {
                StateFields.AGENT_RESPONSES.value: {},
                StateFields.SUB_MESSAGES.value: {},
            }
        try:
            sub_state = {StateFields.AGENT_CONTEXT.value: ctx}
            result = self.loan_advisor_graph.invoke(sub_state, config=config)
            return {
                StateFields.AGENT_RESPONSES.value: {
                    AgentName.LOAN_ADVISOR.value: result.get(StateFields.FINAL_RESPONSE.value)
                },
                StateFields.SUB_MESSAGES.value: {
                    AgentName.LOAN_ADVISOR.value: result.get(StateFields.INTERNAL_MESSAGES.value, [])
                },
            }
        except Exception as e:
            logger.exception("[LoanAdvisor] subgraph invocation failed: %s", e)
            return {
                StateFields.AGENT_RESPONSES.value: {
                    AgentName.LOAN_ADVISOR.value: AgentResponse(content="抱歉，贷款咨询服务暂时不可用，请稍后重试。")
                },
                StateFields.SUB_MESSAGES.value: {
                    AgentName.LOAN_ADVISOR.value: []
                },
            }

    def _dispatch_risk_assessment(self, state: SupervisorState, config: RunnableConfig) -> dict:
        ctx = state.get(StateFields.AGENT_CONTEXT.value, {}).get(AgentName.RISK_ASSESSMENT.value)
        if not ctx:
            logger.warning(f"[RiskAssessment] miss agent context")
            return {
                StateFields.AGENT_RESPONSES.value: {},
                StateFields.SUB_MESSAGES.value: {},
                StateFields.TRIGGER_HUMAN_HANDOFF.value: False
            }
        try:
            sub_state = {StateFields.AGENT_CONTEXT.value: ctx}
            result = self.risk_assessment_graph.invoke(sub_state, config=config)
            return {
                StateFields.AGENT_RESPONSES.value: {
                    AgentName.RISK_ASSESSMENT.value: result.get(StateFields.FINAL_RESPONSE.value)
                },
                StateFields.SUB_MESSAGES.value: {
                    AgentName.RISK_ASSESSMENT.value: result.get(StateFields.INTERNAL_MESSAGES.value, [])
                },
                StateFields.TRIGGER_HUMAN_HANDOFF.value: result.get(StateFields.TRIGGER_HUMAN_HANDOFF.value, False)
            }
        except Exception as e:
            logger.exception("[RiskAssessment] subgraph invocation failed: %s", e)
            return {
                StateFields.AGENT_RESPONSES.value: {
                    AgentName.RISK_ASSESSMENT.value: AgentResponse(content="抱歉，风控服务暂时不可用，请稍后重试。")
                },
                StateFields.SUB_MESSAGES.value: {
                    AgentName.RISK_ASSESSMENT.value: []
                },
                StateFields.TRIGGER_HUMAN_HANDOFF.value: False
            }

    def _dispatch_after_loan(self, state: SupervisorState, config: RunnableConfig) -> dict:
        ctx = state.get(StateFields.AGENT_CONTEXT.value, {}).get(AgentName.AFTER_LOAN.value)
        if not ctx:
            logger.warning(f"[AfterLoan] miss agent context")
            return {
                StateFields.AGENT_RESPONSES.value: {},
                StateFields.SUB_MESSAGES.value: {},
            }
        try:
            sub_state = {StateFields.AGENT_CONTEXT.value: ctx}
            result = self.after_loan_graph.invoke(sub_state, config=config)
            return {
                StateFields.AGENT_RESPONSES.value: {
                    AgentName.AFTER_LOAN.value: result.get(StateFields.FINAL_RESPONSE.value)
                },
                StateFields.SUB_MESSAGES.value: {
                    AgentName.AFTER_LOAN.value: result.get(StateFields.INTERNAL_MESSAGES.value, [])
                },
            }
        except Exception as e:
            logger.exception("[AfterLoan] subgraph invocation failed: %s", e)
            return {
                StateFields.AGENT_RESPONSES.value: {
                    AgentName.AFTER_LOAN.value: AgentResponse(content="抱歉，贷后服务暂时不可用，请稍后重试。")
                },
                StateFields.SUB_MESSAGES.value: {
                    AgentName.AFTER_LOAN.value: []
                },
            }
