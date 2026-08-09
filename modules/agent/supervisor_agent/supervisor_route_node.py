# author hgh
# version 1.1
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.constants import END

from config.global_constant.constants import RegistryModules, MemoryType, ConfigFields, KnowledgeFileSourceType
from config.prompt_hub import PromptHub
from config.registry import ConfigRegistry
from modules.agent.constants import StateFields, RouteTarget, AgentContextFields, AgentName, RouteDecision, \
    AgentNodeName
from modules.agent.multi_agent_state import SupervisorState, AgentContext
from modules.memory.memory_utils.base_memory_utils import get_message_index, format_messages
from modules.module_services.chat_models import RobustLLM
from modules.retrieval.retrieval_service import RetrievalService
from modules.tools.common_utils import assign_message_index
from utils.monitor_utils.metrics import record_llm_metrics, supervisor_routing_total, negative_feedback_total, \
    supervisor_routing_duration_seconds
from utils.query_utils.milvus_query_builder import MilvusQueryBuilder
from utils.query_utils.query_model import Condition, Query
from utils.serialize_utils.seq_generator import SequenceGenerator

logger = logging.getLogger(__name__)

GREETING_REPLY = "您好！我是银行贷款顾问助手，请问有什么可以帮您？"
CLARIFICATION_REPLY = "抱歉，我没有完全理解您的需求。您是想咨询贷款产品、评估贷款风险，还是办理贷后业务呢？"
AMBIGUOUS_HANDOFF_REPLY = "我暂时无法理解您的问题，正在为您转接人工客服..."
ERROR_MESSAGE = "抱歉，系统正忙，暂时无法处理您的问题。请稍后再试，或拨打我行客服热线 95333。"

BUSINESS_AGENTS = [
    AgentName.LOAN_ADVISOR.value,
    AgentName.RISK_ASSESSMENT.value,
    AgentName.AFTER_LOAN.value,
]


class SupervisorRouteNode:
    def __init__(self, registry: ConfigRegistry, llm_client: RobustLLM, knowledge_retriever: RetrievalService,
                 seq_generator: SequenceGenerator):
        self.registry = registry
        self.prompt_hub = PromptHub(registry)
        self.llm_client = llm_client
        self.knowledge_retriever = knowledge_retriever
        self.seq_generator = seq_generator

    async def __call__(self, state: SupervisorState, config: RunnableConfig):
        logger.debug("Enter Supervisor route node")

        # 1. obtain user memory
        formatted = state.get(StateFields.FORMATTED_CONTEXT.value, {})

        # 2. obtain basic information
        messages = state.get(StateFields.MESSAGES.value, [])
        user_query = self._get_latest_human_query(messages)

        # 3. LLM dynamic route
        start = time.monotonic()
        recent_conversations = self._extract_recent_conversation(state)
        decision = await self._llm_route(user_query, recent_conversations, formatted.get(MemoryType.USER_PROFILE.value, ""),
                                         formatted.get(MemoryType.INTERACTION_LOG.value, ""))
        logger.info("Llm dynamic routing result: %s", decision)
        supervisor_routing_duration_seconds.labels("llm").observe(time.monotonic() - start)
        for agent in decision.target_agents:
            supervisor_routing_total.labels(target_agent=agent, route_method="llm").inc()

        # 4. Processing routing results
        response = await self._handle_llm_decision(decision, state, config, formatted, user_query)
        logger.info("Supervisor agent process completed,result: %s", response)
        return response

    def _get_latest_human_query(self, messages: List) -> str:
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return msg.content.strip()
        return ""

    async def _llm_route(
            self,
            user_query: str,
            recent_conversations: str,
            user_profile: Optional[str],
            interaction_log: Optional[str]
    ) -> RouteDecision:
        system_prompt = self.prompt_hub.get_text("supervisor_router")

        # build human message
        human_msg_parts = []
        if recent_conversations:
            human_msg_parts.append(f"最近几轮对话:\n{recent_conversations}")
        if user_profile and user_profile != "暂无相关记录":
            human_msg_parts.append(f"用户画像摘要: {user_profile}")
        if interaction_log and interaction_log != "暂无相关记录":
            human_msg_parts.append(f"对话历史摘要: {interaction_log}")
        human_msg_parts.append(f"用户当前问题: {user_query}")
        human_msg_parts.append("请决定路由目标:")

        human_msg = "\n".join(human_msg_parts)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_msg),
        ]

        try:
            total_start = time.monotonic()
            response = await self.llm_client.ainvoke(messages)
            decision = response.content.strip()
            logger.info("LLM routing decision: %s", decision)
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                record_llm_metrics(provider=self.llm_client.provider,
                                   total_tokens=response.usage_metadata.get("total_tokens", 0),
                                   duration_ms=(time.monotonic() - total_start) * 1000)
        except Exception as e:
            logger.error("LLM route call failed: %s", e)
            return RouteDecision(special=RouteTarget.LLM_ERROR.value)

        # parse llm decision
        decision_lower = decision.lower()
        if RouteTarget.HUMAN_HANDOFF_NOTIFY.value in decision_lower:
            logger.info("LLM decision contains human_handoff, force handoff")
            return RouteDecision(special=RouteTarget.HUMAN_HANDOFF_NOTIFY.value)

        if RouteTarget.DIRECT.value in decision_lower:
            logger.info("LLM decision contains direct, force direct reply")
            return RouteDecision(special=RouteTarget.DIRECT.value)

        if RouteTarget.UNKNOWN.value in decision_lower:
            logger.info("LLM decision contains unknown, treat as ambiguous")
            return RouteDecision(special=RouteTarget.UNKNOWN.value)

        targets = [t.strip() for t in decision.split(",") if t.strip() in BUSINESS_AGENTS]
        if not targets:
            logger.warning("LLM returned unrecognizable target: %s", decision)
            return RouteDecision(special=RouteTarget.UNKNOWN.value)

        return RouteDecision(target_agents=targets)

    async def _handle_llm_decision(
            self,
            decision: RouteDecision,
            state: SupervisorState,
            config: RunnableConfig,
            formatted: Dict,
            user_query: str
    ) -> Dict[str, Any]:
        # handle with special route
        if decision.special:
            if decision.special == RouteTarget.HUMAN_HANDOFF_NOTIFY.value:
                ctx = await self._build_agent_context(state, config, formatted, user_query,
                                                [RouteTarget.HUMAN_HANDOFF_NOTIFY.value])
                return self._make_response(next_agents=[RouteTarget.HUMAN_HANDOFF_NOTIFY.value], agent_context=ctx)
            elif decision.special == RouteTarget.LLM_ERROR.value:
                return self._make_response(next_agents=[], messages=[AIMessage(content=ERROR_MESSAGE)])
            elif decision.special == RouteTarget.UNKNOWN.value:
                return await self._handle_ambiguous(state, config, formatted, user_query)
            else:
                return await self._handle_ambiguous(state, config, formatted, user_query)

        # handle with business route
        targets = decision.target_agents
        if len(targets) > 1:
            ctx = await self._build_agent_context(state, config, formatted, user_query, targets)
            return self._make_response(next_agents=targets, agent_context=ctx)
        else:
            ctx = await self._build_agent_context(state, config, formatted, user_query, targets)
            return self._make_response(next_agents=targets, agent_context=ctx)

    async def _handle_ambiguous(
            self,
            state: SupervisorState,
            config: RunnableConfig,
            formatted: Dict,
            user_query: str
    ) -> Dict[str, Any]:
        user_id = state.get(StateFields.USER_ID.value)
        session_id = config.get(ConfigFields.CONFIGURABLE.value, {}).get(ConfigFields.THREAD_ID.value)
        count = state.get(StateFields.CLARIFICATION_COUNT.value, 0) + 1
        if count >= 2:
            negative_feedback_total.labels(reason="ambiguous").inc()
            logger.warning("Unclear intent for 2 consecutive times, transferring to a official custom worker")
            ctx = await self._build_agent_context(state, config, formatted, user_query,
                                            RouteTarget.HUMAN_HANDOFF_NOTIFY.value)
            return self._make_response(
                next_agents=[AgentNodeName.HUMAN_HANDOFF_NOTIFY.value],
                agent_context=ctx,
                messages=[AIMessage(content=AMBIGUOUS_HANDOFF_REPLY)],
                clarification_count=0,
            )
        else:
            logger.info("Intent unclear, asking user for clarification (time %d)", count)
            res_message = AIMessage(content=CLARIFICATION_REPLY)
            assign_message_index(res_message, user_id, session_id, self.seq_generator)
            return self._make_response(
                next_agents=[END],
                messages=[res_message],
                clarification_count=count,
            )

    async def _build_agent_context(
            self,
            state: SupervisorState,
            config: RunnableConfig,
            formatted: Dict[str, str],
            user_query: str,
            agent_name: List[str]
    ) -> Dict[str, AgentContext]:
        user_id = state.get(StateFields.USER_ID.value, "")
        configurable = config.get(ConfigFields.CONFIGURABLE.value, {})
        session_id = configurable.get(ConfigFields.THREAD_ID.value, "unknown")
        trace_id = state.get(AgentContextFields.TRACE_ID.value, "")
        audit_logger = state.get(AgentContextFields.AUDIT_LOG.value)
        compliance_warnings = state.get(StateFields.COMPLIANCE_WARNINGS.value, [])
        conversation_summary = formatted.get(MemoryType.INTERACTION_LOG.value, "暂无相关信息")
        sub_conversation_summary = formatted.get(MemoryType.SUB_INTERACTION_LOG.value, {})
        user_profile_summary = formatted.get(MemoryType.USER_PROFILE.value, "暂无相关信息")
        recent_conversations = self._extract_recent_conversation(state) or "暂无相关信息"

        # Directional Knowledge Retrieval
        agent_contexts = {}
        retrieve_agent_name = [n for n in agent_name if n != AgentNodeName.HUMAN_HANDOFF_NOTIFY.value]
        supervisor_cfg = self.registry.get_config(RegistryModules.SUPERVISOR.value)
        if self.knowledge_retriever and supervisor_cfg and getattr(supervisor_cfg, 'enable_directed_retrieval', False):
            human_handoff = [n for n in agent_name if n == AgentNodeName.HUMAN_HANDOFF_NOTIFY.value]
            if human_handoff:
                agent_contexts[AgentNodeName.HUMAN_HANDOFF_NOTIFY.value] = AgentContext.from_state(
                    user_id, session_id, trace_id, user_query,
                    user_profile_summary, compliance_warnings,
                    conversation_summary, "",
                    f"请以 {AgentNodeName.HUMAN_HANDOFF_NOTIFY.value} 的身份回答以下用户问题。",
                    audit_logger, "", recent_conversations)

            if len(agent_name) > 1:
                logger.info("Fan-out to agents: %s", agent_name)
                # aretrieve() 是 async 方法，直接用 asyncio.gather 并行调用
                knowledge_results = await asyncio.gather(*[
                    self._retrieve_knowledge_for_agent(
                        agt_name, user_query, state, conversation_summary
                    )
                    for agt_name in retrieve_agent_name
                ], return_exceptions=True)

                for agt_name, knowledge in zip(retrieve_agent_name, knowledge_results):
                    if isinstance(knowledge, Exception):
                        logger.error("Fan-out 知识检索异常 (%s): %s", agt_name, knowledge)
                        knowledge = ""

                    sub_conversation = sub_conversation_summary.get(agt_name, "")
                    recent_sub = state.get(StateFields.SUB_MESSAGES.value, {}).get(agt_name, [])
                    if recent_sub:
                        recent_str = format_messages(recent_sub)
                        sub_conversation = (recent_str + "\n" + sub_conversation).strip()
                    if not sub_conversation:
                        sub_conversation = "暂无相关信息"

                    agent_contexts[agt_name] = AgentContext.from_state(
                        user_id, session_id, trace_id, user_query,
                        user_profile_summary, compliance_warnings,
                        conversation_summary, knowledge,
                        f"请以 {agt_name} 的身份回答以下用户问题。",
                        audit_logger, sub_conversation, recent_conversations)

            else:
                if retrieve_agent_name:
                    knowledge = await self._retrieve_knowledge_for_agent(
                        retrieve_agent_name[0], user_query, state, conversation_summary
                    )

                    sub_conversation = sub_conversation_summary.get(retrieve_agent_name[0], "")
                    recent_sub = state.get(StateFields.SUB_MESSAGES.value, {}).get(retrieve_agent_name[0], [])
                    if recent_sub:
                        recent_str = format_messages(recent_sub)
                        sub_conversation = (recent_str + "\n" + sub_conversation).strip()
                    if not sub_conversation:
                        sub_conversation = "暂无相关信息"

                    agent_contexts[agent_name[0]] = AgentContext.from_state(
                        user_id, session_id, trace_id, user_query,
                        user_profile_summary, compliance_warnings,
                        conversation_summary, knowledge,
                        f"请以 {retrieve_agent_name[0]} 的身份回答以下用户问题。",
                        audit_logger, sub_conversation, recent_conversations)
        else:
            for agent in agent_name:
                agent_instruction = f"请以 {agent} 的身份回答以下用户问题。"

                sub_conversation = sub_conversation_summary.get(agent, "")
                recent_sub = state.get(StateFields.SUB_MESSAGES.value, {}).get(agent, [])
                if recent_sub:
                    recent_str = format_messages(recent_sub)
                    sub_conversation = (recent_str + "\n" + sub_conversation).strip()
                if not sub_conversation:
                    sub_conversation = "暂无相关信息"

                agent_contexts[agent] = AgentContext.from_state(
                    user_id, session_id, trace_id, user_query,
                    user_profile_summary, compliance_warnings,
                    conversation_summary, "", agent_instruction,
                    audit_logger, sub_conversation, recent_conversations)

        return agent_contexts

    def _make_response(
            self,
            next_agents: List[str],
            messages: Optional[List] = None,
            agent_context: Optional[Dict[str, AgentContext]] = None,
            clarification_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """统一组装状态更新字典，避免重复字段"""
        result = {
            StateFields.NEXT_AGENTS.value: next_agents,
            StateFields.AGENT_CONTEXT.value: agent_context,
        }
        if messages is not None:
            result[StateFields.MESSAGES.value] = messages
        if clarification_count is not None:
            result[StateFields.CLARIFICATION_COUNT.value] = clarification_count
        return result

    async def _retrieve_knowledge_for_agent(
            self,
            agent_name: str,
            user_query: str,
            state: SupervisorState,
            conversation_summary: Optional[str]
    ) -> str:
        """根据目标 Agent 执行定向知识检索并精炼为短文本（async，直接 await aretrieve）"""
        # 根据 Agent 类型确定过滤条件
        parts = []
        if agent_name == AgentName.LOAN_ADVISOR.value:
            parts.append(Condition(field="source_type", op="in", value=[KnowledgeFileSourceType.FAQ.value,
                                                                        KnowledgeFileSourceType.PRODUCT_MANUAL.value]))
        elif agent_name == AgentName.RISK_ASSESSMENT.value:
            parts.append(Condition(field="source_type", op="==", value=KnowledgeFileSourceType.REGULATION.value))
        elif agent_name == AgentName.AFTER_LOAN.value:
            parts.append(Condition(field="source_type", op="in", value=[KnowledgeFileSourceType.PROCESS_GUIDE.value,
                                                                        KnowledgeFileSourceType.FAQ.value]))

        filter = MilvusQueryBuilder().build(Query(conditions=parts, logic="AND"))

        last_summary = self._build_last_summary_for_entry_retrieval(state)
        parts = []
        if conversation_summary:
            parts.append(f"相关对话历史:\n{conversation_summary}")
        if last_summary:
            parts.append(f"最近对话:\n{last_summary}")
        context = "\n".join(parts) if parts else ""

        try:
            logger.info(f"Supervisor directly retrieving knowledge for agent {agent_name}")
            docs = await self.knowledge_retriever.aretrieve(
                query=user_query,
                context=context,
                filter_expr=filter,
            )
            if not docs:
                return ""
            # 格式化并限制长度
            config = self.registry.get_config(RegistryModules.SUPERVISOR.value)
            from modules.retrieval.knowledge_utils.knowledge_formatter import format_context
            max_length = getattr(config, 'directed_retrieval_max_length', 500)
            return format_context(docs, max_context_length=max_length)
        except Exception as e:
            logger.error(f"Supervisor directly retrieving failed for agent {agent_name},e:{e}")
            return ""

    def _extract_recent_conversation(self, state: SupervisorState) -> str:
        messages = state.get(StateFields.MESSAGES.value, [])
        if not messages:
            return ""

        candidate_messages = messages[:-1] if len(messages) > 1 else []

        last_logged_index = state.get(StateFields.LAST_LOGGED_MESSAGE_INDEX.value)
        if last_logged_index is not None:
            recent = [
                m for m in candidate_messages
                if get_message_index(m) is not None and get_message_index(m) > last_logged_index
            ]
        else:
            recent = candidate_messages[-5:] if len(candidate_messages) > 5 else candidate_messages

        return format_messages(recent) if recent else ""

    def _build_last_summary_for_entry_retrieval(self, state: SupervisorState) -> str:
        formatted = state.get(StateFields.FORMATTED_CONTEXT.value, {})
        parts = []

        # 1. 已摘要的全局对话
        interaction_log = formatted.get(MemoryType.INTERACTION_LOG.value, "")
        if interaction_log and interaction_log != "暂无相关信息":
            parts.append(f"对话摘要：{interaction_log}")

        # 2. 最新的未摘要主图消息
        messages = state.get(StateFields.MESSAGES.value, [])
        last_logged_index = state.get(StateFields.LAST_LOGGED_MESSAGE_INDEX.value)
        candidate_messages = messages[:-1] if len(messages) > 1 else []
        if last_logged_index is not None:
            recent = [
                m for m in candidate_messages
                if get_message_index(m) is not None and get_message_index(m) > last_logged_index
            ]
        else:
            recent = candidate_messages[-5:] if len(candidate_messages) > 5 else candidate_messages

        if recent:
            recent_str = format_messages(recent)
            parts.append(f"最新对话：{recent_str}")

        return "；".join(parts) if parts else ""
