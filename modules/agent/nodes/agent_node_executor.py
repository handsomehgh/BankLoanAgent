import logging
import time
from typing import Dict, Any, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from config.global_constant.constants import RegistryModules
from config.prompts.system_prompt import SYSTEM_PROMPT
from config.registry import ConfigRegistry
from exceptions.exception import ToolExecutionError, CircuitBreakerOpenError, \
    ToolExecutionException
from infra.circuit_breaker import CircuitBreaker
from modules.agent.constants import StateFields
from modules.agent.multi_agent_state import AgentContext, AgentResponse
from modules.module_services.chat_models import RobustLLM
from modules.tools import ToolResult
from modules.tools.base_tool import ToolExecutor, ToolErrorType
from modules.tools.common_utils import assign_message_index, get_agent_tools, get_tools_metadata, build_text_a_for_bert
from modules.tools.response_handler.upsert_response_handler import UpsertLoanInterestHandler
from modules.tools.tool_selector import ToolSelector
from utils.monitor_utils.metrics import record_llm_metrics, agent_executor_errors_total, circuit_breaker_state, \
    agent_select_tool_total
from utils.serialize_utils.seq_generator import SequenceGenerator

logger = logging.getLogger(__name__)


class AgentNodeExecutor:
    """
    General Agent node executor, encapsulating hierarchical decision Function Calling loop
    """

    def __init__(
            self,
            agent_module: str,
            agent_name: str,
            registry: ConfigRegistry,
            llm_client: RobustLLM,
            tool_executor: ToolExecutor,
            seq_generator: SequenceGenerator,
            tool_selector: ToolSelector,
            post_process: Optional[callable] = None,
            classifier: Optional[Any] = None
    ):
        self.agent_module = agent_module
        self.agent_name = agent_name
        self.registry = registry
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.seq_generator = seq_generator
        self.tool_selector = tool_selector
        self.post_process = post_process
        self.classifier = classifier

        self.cb_config = registry.get_config(RegistryModules.AGENT_EXECUTOR.value).circuit_breaker
        self.fallback_msgs = registry.get_config(RegistryModules.AGENT_EXECUTOR.value).fallback_messages
        self.tool_fallbacks = registry.get_config(RegistryModules.AGENT_EXECUTOR.value).tool_fallbacks
        self.response_handlers_config = registry.get_config(
            RegistryModules.AGENT_EXECUTOR.value
        ).response_handlers

        self._circuit_breakers: Dict[str, CircuitBreaker] = {}

    def _get_cb(self, tool_name: str) -> CircuitBreaker:
        if tool_name not in self._circuit_breakers:
            if self.cb_config.enabled:
                self._circuit_breakers[tool_name] = CircuitBreaker(
                    name=f"{self.agent_name}:{tool_name}",
                    failure_threshold=self.cb_config.failure_threshold,
                    recovery_timeout=self.cb_config.recovery_timeout_sec
                )
        return self._circuit_breakers[tool_name]

    def _execute_tool_safe(self, tool_call: dict, trace_id: str, context: AgentContext) -> ToolResult:
        tool_name = tool_call["name"]
        cb = self._get_cb(tool_name)

        def do_execute():
            return self.tool_executor.execute(
                tool_name=tool_name,
                args=tool_call["args"],
                caller_agent=self.agent_name,
                trace_id=trace_id,
                user_id=context.user_id,
                conversation_summary=context.conversation_summary,
                profile_summary=context.user_profile_summary,
            )

        try:
            result = cb.call(do_execute)
            circuit_breaker_state.labels(tool_name=tool_name).set(0)
            return result
        except CircuitBreakerOpenError:
            circuit_breaker_state.labels(tool_name=tool_name).set(1)  # OPEN
            logger.warning(f"[{self.agent_name}] 断路器打开，工具 {tool_name} 降级")
            return ToolResult(
                success=False,
                error=f"工具 {tool_name} 暂时不可用，请稍后重试",
                error_type=ToolErrorType.CIRCUIT_OPEN
            )
        except ToolExecutionException as e:
            logger.error(f"[{self.agent_name}] 工具 {tool_name} 执行异常: {e}")
            return ToolResult(success=False, error=str(e), error_type=e.error_type)
        except Exception as e:
            logger.error(f"[{self.agent_name}] 工具 {tool_name} 执行异常: {e}")
            return ToolResult(success=False, error=str(e), error_type=ToolErrorType.EXTERNAL_ERROR)

    def execute(self, state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
        context: AgentContext = state.get(StateFields.AGENT_CONTEXT.value)
        user_query = context.current_query
        trace_id = context.trace_id
        user_id = context.user_id
        session_id = context.session_id

        # 1. read agent config
        agent_cfg = self.registry.get_config(self.agent_module)

        # 2. build prompt
        context_vars = {
            "user_profile": context.user_profile_summary or "暂无相关信息",
            "compliance_rule": context.compliance_warnings or "暂无相关信息",
            "interaction_log": context.conversation_summary or "暂无相关信息",
            "business_knowledge": context.retrieved_knowledge or "暂无相关信息",
            "tool_conversation": context.sub_conversation or "暂无相关信息",
            "recent_conversation": context.recent_conversation or "暂无相关信息",
        }
        messages = []

        # 3. round 1: LLM chooses the tool name
        stage = "judge"
        try:
            if agent_cfg.use_bert_classifier:
                text_a = build_text_a_for_bert(context)
                text_b = user_query
                tool_name = self.classifier.predict(text_a, text_b)
                agent_select_tool_total.labels(tool_name=tool_name, agent_name=self.agent_name).inc()
                logger.info("[%s] BERT selected tool: %s", self.agent_name, tool_name)
            else:
                # obtain tools metadata
                tools_metadata = get_tools_metadata(agent_cfg, self.agent_name, self.tool_selector)
                metadata_str = "\n".join([f"- {m['name']}: {m['description']}" for m in tools_metadata])

                # build judge messages
                judge_role = agent_cfg.judge_prompt.format(tools_metadata=metadata_str)
                judge_prompt = SYSTEM_PROMPT.format(agent_role=judge_role, **context_vars)
                judge_messages = [SystemMessage(content=judge_prompt), HumanMessage(content=user_query)]

                # call llm to judge
                total_start = time.monotonic()
                first_response = self.llm_client.invoke(judge_messages)
                tool_name = first_response.content.strip()
                if hasattr(first_response, "usage_metadata") and first_response.usage_metadata:
                    record_llm_metrics(provider=self.llm_client.provider,
                                       total_tokens=first_response.usage_metadata.get("total_tokens", 0),
                                       duration_ms=(time.monotonic() - total_start) * 1000)
                logger.info("[%s] LLM selected tool: %s", self.agent_name, tool_name)
        except Exception as e:
            agent_executor_errors_total.labels(
                agent_name=self.agent_name, stage=stage, error_type=type(e).__name__
            ).inc()
            logger.exception("[%s] First round LLM failed: %s", self.agent_name, e)
            error_response = AIMessage(content="抱歉，我暂时无法处理您的问题，请稍后再试。")
            assign_message_index(error_response, user_id, session_id, self.seq_generator)
            return self._final_response(error_response, messages, context)

        # 4. the agent cannot directly respond by selecting a tool based on the user's request.
        if tool_name.upper() == "CLARIFY":
            stage = "clarify"
            try:
                # build clarify messages
                clarify_prompt = SYSTEM_PROMPT.format(agent_role=agent_cfg.clarify_prompt, **context_vars)
                clarify_messages = [SystemMessage(content=clarify_prompt), HumanMessage(content=user_query)]

                # call llm
                clarify_response = self.llm_client.invoke(clarify_messages, tool_choice="none")

                # assign message index
                assign_message_index(clarify_response, user_id, session_id, self.seq_generator)
                logger.info("[%s] LLM clarify reply with: %s", self.agent_name, clarify_response.content.strip())

                # return result
                return self._final_response(clarify_response, messages, context)
            except Exception as e:
                agent_executor_errors_total.labels(
                    agent_name=self.agent_name, stage=stage, error_type=type(e).__name__
                ).inc()
                logger.exception("[%s] LLM clarify reply failed: %s", self.agent_name, e)
                clarify_response = AIMessage(content="抱歉，我没太理解您的需求，可以再具体描述一下吗？")
                assign_message_index(clarify_response, user_id, session_id, self.seq_generator)
                return self._final_response(clarify_response, messages, context)

        if tool_name.upper() == "DIRECT_REPLY":
            stage = "direct"
            try:
                # build direct messages
                direct_prompt = SYSTEM_PROMPT.format(agent_role=agent_cfg.direct_prompt, **context_vars)
                direct_messages = [SystemMessage(content=direct_prompt), HumanMessage(content=user_query)]

                # call llm
                total_start = time.monotonic()
                direct_res = self.llm_client.invoke(direct_messages, tool_choice="none")
                if hasattr(direct_res, "usage_metadata") and direct_res.usage_metadata:
                    record_llm_metrics(provider=self.llm_client.provider,
                                       total_tokens=direct_res.usage_metadata.get("total_tokens", 0),
                                       duration_ms=(time.monotonic() - total_start) * 1000)
                logger.info("[%s] LLM direct reply with: %s", self.agent_name, direct_res.content.strip()[:50])

                # assign message index
                assign_message_index(direct_res, user_id, session_id, self.seq_generator)

                # retrun result
                return self._final_response(direct_res, messages, context)
            except Exception as e:
                agent_executor_errors_total.labels(
                    agent_name=self.agent_name, stage=stage, error_type=type(e).__name__
                ).inc()
                logger.exception("[%s] Second round LLM failed: %s", self.agent_name, e)
                error_response = AIMessage(content="抱歉，我暂时无法处理您的问题，请稍后再试。")
                assign_message_index(error_response, user_id, session_id, self.seq_generator)
                return self._final_response(error_response, messages)

        # 6. round 2: Obtain the full schema of the selected tool
        stage = "execute"
        try:
            # messages.append(first_response)
            # get tool
            full_tools = get_agent_tools(agent_cfg, self.tool_selector, self.agent_name)
            selected_tool = next((t for t in full_tools if t.name == tool_name), None)
            if not selected_tool:
                raise ToolExecutionError(f"工具 {tool_name} 未找到")
            selected_tools = [selected_tool] if selected_tool else full_tools

            # build messages
            tool_role = agent_cfg.execute_prompt.format(tool_name=tool_name)
            tool_prompt = SYSTEM_PROMPT.format(agent_role=tool_role, **context_vars)
            tool_messages = [SystemMessage(content=tool_prompt), HumanMessage(content=user_query)]
            tool_messages.extend(messages)

            # cal llm
            total_start = time.monotonic()
            sec_response = self.llm_client.invoke(tool_messages, tools=selected_tools)
            logger.info("[%s] LLM call tool: %s", self.agent_name, sec_response.tool_calls)
            if hasattr(sec_response, "usage_metadata") and sec_response.usage_metadata:
                record_llm_metrics(provider=self.llm_client.provider,
                                   total_tokens=sec_response.usage_metadata.get("total_tokens", 0),
                                   duration_ms=(time.monotonic() - total_start) * 1000)
        except Exception as e:
            agent_executor_errors_total.labels(
                agent_name=self.agent_name, stage=stage, error_type=type(e).__name__
            ).inc()
            logger.exception("[%s] Second round LLM failed: %s", self.agent_name, e)
            error_response = AIMessage(content="抱歉，我暂时无法处理您的问题，请稍后再试。")
            assign_message_index(error_response, user_id, session_id, self.seq_generator)
            return self._final_response(error_response, messages)

        # 7. handle tool call
        if sec_response.tool_calls:
            messages.append(sec_response)
            for tool_call in sec_response.tool_calls:
                if self.cb_config.enabled:
                    tool_result = self._execute_tool_safe(tool_call, trace_id, context)
                else:
                    tool_result = self.tool_executor.execute(
                        tool_name=tool_call["name"],
                        args=tool_call["args"],
                        caller_agent=self.agent_name,
                        trace_id=trace_id,
                        user_id=context.user_id,
                        conversation_summary=context.conversation_summary,
                        profile_summary=context.user_profile_summary,
                    )

                if not tool_result.success:
                    error_response = self._handle_tool_error(user_query, tool_call["name"], tool_result, user_id,
                                                             session_id, context_vars, messages)
                    messages = [m for m in messages if m != sec_response]
                    if error_response:
                        return self._final_response(error_response, messages, context)

                if tool_result.success:
                    suggested_reply = self._handle_tool_result(tool_call['name'], tool_result)
                    if suggested_reply:
                        tool_result.data['suggested_reply'] = suggested_reply

                tool_msg = ToolMessage(
                    content=tool_result.to_message_content(),
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                )
                messages.append(tool_msg)

            # generate final reply
            stage = "final"
            try:
                # build messages
                res_prompt = SYSTEM_PROMPT.format(agent_role=agent_cfg.res_prompt, **context_vars)
                res_messages = [SystemMessage(content=res_prompt), HumanMessage(content=user_query)]
                res_messages.extend(messages)

                # call llm
                total_start = time.monotonic()
                final_response = self.llm_client.invoke(res_messages, tool_choice="none")
                if hasattr(final_response, "usage_metadata") and final_response.usage_metadata:
                    record_llm_metrics(provider=self.llm_client.provider,
                                       total_tokens=final_response.usage_metadata.get("total_tokens", 0),
                                       duration_ms=(time.monotonic() - total_start) * 1000)
                logger.info("[%s] Final reply with tools: %s", self.agent_name,
                            final_response.content.strip()[:100])

                # assign message index
                assign_message_index(final_response, user_id, session_id, self.seq_generator)

                # return result
                return self._final_response(final_response, messages, context)
            except Exception as e:
                agent_executor_errors_total.labels(
                    agent_name=self.agent_name, stage=stage, error_type=type(e).__name__
                ).inc()
                logger.exception("[%s] Final generation failed: %s", self.agent_name, e)
                error_response = AIMessage(content="抱歉，我暂时无法生成回复，请稍后再试。")
                assign_message_index(error_response, user_id, session_id, self.seq_generator)
                return self._final_response(error_response, messages)
        else:
            logger.info("[%s] LLM asks for more info: %s", self.agent_name,
                        sec_response.content.strip()[:100])
            assign_message_index(sec_response, user_id, session_id, self.seq_generator)
            return self._final_response(sec_response, messages, context)

    def _final_response(
            self,
            final_msg: AIMessage,
            all_messages: List,
            context: Optional[AgentContext] = None,
    ) -> Dict[str, Any]:
        """build final response"""
        agent_response = AgentResponse(content=final_msg.content.strip(), metadata={})

        # post process
        result = {
            StateFields.FINAL_RESPONSE.value: agent_response,
            StateFields.INTERNAL_MESSAGES.value: [
                m for m in all_messages
                if not isinstance(m, (SystemMessage, HumanMessage))
            ],
        }
        if self.post_process:
            extra = self.post_process(final_msg, context, all_messages)
            if extra:
                result.update(extra)
        return result

    def _handle_tool_error(self, user_query: str, tool_name: str, tool_result: ToolResult, user_id: str,
                           session_id: str, context_vars: Dict[str, Any], all_messages: list) -> Optional[AIMessage]:
        """根据工具错误类型生成差异化用户引导"""
        error_type = tool_result.error_type
        error_msg = tool_result.error or ""

        if error_type == ToolErrorType.PARAMETER_ERROR:
            agent_cfg = self.registry.get_config(self.agent_module)
            param_error_role = agent_cfg.param_error_prompt.format(error_msg=error_msg)
            system_prompt = SYSTEM_PROMPT.format(agent_role=param_error_role, **context_vars)
            messages = [SystemMessage(content=system_prompt),
                        HumanMessage(content=user_query)]
            messages.extend(all_messages)

            try:
                response = self.llm_client.invoke(messages, tool_choice="none")
                reply = AIMessage(content=response.content.strip())
            except Exception:
                reply = AIMessage(content=f"抱歉，参数似乎有误：{error_msg}，请您重新提供正确的信息。")
            assign_message_index(reply, user_id, session_id, self.seq_generator)
            return reply
        else:
            fallback = self.tool_fallbacks.get(tool_name, {}).get("message", "该服务暂时不可用。")
            reply = AIMessage(content=fallback)
            assign_message_index(reply, user_id, session_id, self.seq_generator)
            return reply

    def _handle_tool_result(self, tool_name: str, tool_result: ToolResult) -> Optional[str]:
        """工具执行成功后，如果配置了响应处理器，生成建议回复"""
        handler_config = self.response_handlers_config.get(tool_name)
        if not handler_config:
            return None  # 无配置，LLM 自行处理

        if tool_name == 'upsert_loan_interest':
            handler = UpsertLoanInterestHandler(tool_result.data, handler_config)
            return handler.generate()

        return None
