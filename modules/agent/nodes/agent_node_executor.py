import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from config.global_constant.constants import RegistryModules
from config.prompts.system_prompt import SYSTEM_PROMPT
from config.registry import ConfigRegistry
from exceptions.exception import ToolExecutionError, CircuitBreakerOpenError, \
    ToolExecutionException
from infra.circuit_breaker import CircuitBreaker
from modules.agent.constants import StateFields, ReplyStage
from modules.agent.multi_agent_state import AgentContext, AgentResponse
from modules.memory.memory_utils.base_memory_utils import format_messages
from modules.module_services.chat_models import RobustLLM
from modules.skills.skill_executor import SkillExecutor
from modules.skills.skill_registry import SkillRegistry
from modules.tools import ToolResult
from modules.tools.base_tool import ToolExecutor, ToolErrorType
from modules.tools.common_utils import assign_message_index, get_agent_tools, get_tools_metadata, build_text_a_for_bert
from modules.tools.response_handler.upsert_response_handler import UpsertLoanInterestHandler
from modules.tools.tool_selector import ToolSelector
from utils.monitor_utils.metrics import record_llm_metrics, agent_executor_errors_total, circuit_breaker_state, \
    agent_select_tool_total
from utils.serialize_utils.seq_generator import SequenceGenerator

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


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
            classifier: Optional[Any] = None,
            skill_executor: SkillExecutor = None,
            skill_selector: SkillRegistry = None
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
        self.skill_executor = skill_executor
        self.skill_selector = skill_selector

        self.cb_config = registry.get_config(RegistryModules.AGENT_EXECUTOR.value).circuit_breaker
        self.fallback_msgs = registry.get_config(RegistryModules.AGENT_EXECUTOR.value).fallback_messages
        self.tool_fallbacks = registry.get_config(RegistryModules.AGENT_EXECUTOR.value).tool_fallbacks
        self.response_handlers_config = registry.get_config(
            RegistryModules.AGENT_EXECUTOR.value
        ).response_handlers

        self._circuit_breakers: Dict[str, CircuitBreaker] = {}

    def _build_context_vars(self, context: AgentContext) -> Dict[str, Any]:
        return {
            "user_profile": context.user_profile_summary or "暂无相关信息",
            "compliance_rule": context.compliance_warnings or "暂无相关信息",
            "interaction_log": context.conversation_summary or "暂无相关信息",
            "business_knowledge": context.retrieved_knowledge or "暂无相关信息",
            "tool_conversation": context.sub_conversation or "暂无相关信息",
            "recent_conversation": context.recent_conversation or "暂无相关信息",
        }

    @staticmethod
    def _route_to(stage: ReplyStage, payload: Dict[str, Any], **extra) -> Dict[str, Any]:
        """decide()统一走这个函数把交接信息写进state,交给reply()消费"""
        return {
            StateFields.REPLY_STAGE.value: stage.value,
            StateFields.REPLY_PAYLOAD.value: payload,
            **extra,
        }

    async def decide(self, state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
        """决策节点：意图分类+参数提取+工具调用,永不生成用户可见文本,只把结果路由给reply()"""
        context: AgentContext = state.get(StateFields.AGENT_CONTEXT.value)
        user_query = context.current_query
        trace_id = context.trace_id

        # 1. read agent config
        agent_cfg = self.registry.get_config(self.agent_module)
        context_vars = self._build_context_vars(context)
        messages = []

        # 2. round 1: LLM chooses the tool name
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
                format_kwargs = {
                    **context_vars,
                    "tool_facts": "暂无",
                    "agent_role": judge_role
                }
                judge_prompt = SYSTEM_PROMPT.format(**format_kwargs)
                judge_messages = [SystemMessage(content=judge_prompt), HumanMessage(content=user_query)]

                # call llm to judge
                total_start = time.monotonic()
                first_response = await self.llm_client.ainvoke(judge_messages)
                tool_name = first_response.content.strip()
                if hasattr(first_response, "usage_metadata") and first_response.usage_metadata:
                    record_llm_metrics(provider=self.llm_client.provider,
                                       total_tokens=first_response.usage_metadata.get("total_tokens", 0),
                                       duration_ms=(time.monotonic() - total_start) * 1000)

                # write to file
                if tool_name:
                    file_path = PROJECT_ROOT / "data" / "wheel" / self.agent_name / "train.jsonl"
                    try:
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(str(file_path), "a", encoding="utf-8") as f:
                            tool_context = build_text_a_for_bert(context)
                            tool_query = user_query
                            content = {"text_a": tool_context, "text_b": tool_query, "label": tool_name}
                            f.write(json.dumps(content, ensure_ascii=False) + "\n")
                    except Exception as write_error:
                        logger.error(f"Write to {file_path} failed: {write_error}")
                logger.info("[%s] LLM selected tool: %s", self.agent_name, tool_name)
        except Exception as e:
            agent_executor_errors_total.labels(
                agent_name=self.agent_name, stage=stage, error_type=type(e).__name__
            ).inc()
            logger.exception("[%s] First round LLM failed: %s", self.agent_name, e)
            return self._route_to(ReplyStage.CANNED_FALLBACK, {"text": "抱歉，我暂时无法处理您的问题，请稍后再试。"})

        # 3. the agent cannot directly respond by selecting a tool based on the user's request.
        if tool_name.upper() == "CLARIFY":
            return self._route_to(ReplyStage.CLARIFY, {})

        if tool_name.upper() == "DIRECT_REPLY":
            return self._route_to(ReplyStage.DIRECT, {})

        # 4. round 2: Obtain the full schema of the selected tool
        stage = "execute"
        try:
            # get tool
            full_tools = get_agent_tools(agent_cfg, self.tool_selector, self.agent_name)
            selected_tool = next((t for t in full_tools if t.name == tool_name), None)
            if not selected_tool:
                raise ToolExecutionError(f"工具 {tool_name} 未找到")
            selected_tools = [selected_tool] if selected_tool else full_tools

            # build messages
            tool_role = agent_cfg.execute_prompt.format(tool_name=tool_name)
            format_kwargs = {
                **context_vars,
                "tool_facts": "暂无",
                "agent_role": tool_role
            }
            tool_prompt = SYSTEM_PROMPT.format(**format_kwargs)
            tool_messages = [SystemMessage(content=tool_prompt), HumanMessage(content=user_query)]
            tool_messages.extend(messages)

            # cal llm
            total_start = time.monotonic()
            sec_response = await self.llm_client.ainvoke(tool_messages, tools=selected_tools)
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
            return self._route_to(ReplyStage.CANNED_FALLBACK, {"text": "抱歉，我暂时无法处理您的问题，请稍后再试。"})

        # 5. handle tool call
        if not sec_response.tool_calls:
            logger.info("[%s] LLM asks for more info: %s", self.agent_name,
                        sec_response.content.strip()[:100])
            return self._route_to(ReplyStage.PASSTHROUGH, {"text": sec_response.content})

        messages.append(sec_response)
        for tool_call in sec_response.tool_calls:
            if self.cb_config.enabled:
                if self.skill_selector and self.skill_selector.get(tool_call["name"]):
                    logger.info(f"[%s] Start execute skill %s", self.agent_name, tool_call["name"])
                    tool_result = await self._execute_skill_safe(
                        skill_name=tool_call["name"],
                        input_data=tool_call["args"],
                        trace_id=trace_id,
                        context=context
                    )
                else:
                    logger.info(f"[%s] Start execute tool %s", self.agent_name, tool_call["name"])
                    tool_result = await self._execute_tool_safe(tool_call, trace_id, context)
            else:
                tool_result = await asyncio.to_thread(
                    self.tool_executor.execute,
                    tool_name=tool_call["name"],
                    args=tool_call["args"],
                    caller_agent=self.agent_name,
                    trace_id=trace_id,
                    user_id=context.user_id,
                    conversation_summary=context.conversation_summary,
                    profile_summary=context.user_profile_summary,
                )

            if not tool_result.success:
                return self._route_tool_error(tool_call["name"], tool_result, messages, sec_response)

            suggested_reply = self._handle_tool_result(tool_call['name'], tool_result)
            if suggested_reply:
                tool_result.data['suggested_reply'] = suggested_reply

            tool_msg = ToolMessage(
                content=tool_result.to_message_content(),
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
            )
            messages.append(tool_msg)

        # all tool calls succeeded,leave the reply text generation entirely to reply()
        tool_facts_text = format_messages(messages)
        return self._route_to(
            ReplyStage.FINAL,
            {"tool_facts_text": tool_facts_text, "all_messages": messages},
        )

    def _route_tool_error(self, tool_name: str, tool_result: ToolResult, messages: list,
                           sec_response: AIMessage) -> Dict[str, Any]:
        """根据工具错误类型把差异化引导信息路由给reply(),decide()自身不生成任何用户可见文本"""
        # 与原逻辑保持一致：出错时把本轮尚未确认成功的tool_call消息从messages中剔除
        messages_before_error = [m for m in messages if m != sec_response]
        if tool_result.error_type == ToolErrorType.PARAMETER_ERROR:
            return self._route_to(
                ReplyStage.PARAM_ERROR,
                {
                    "error_msg": tool_result.error or "",
                    "tool_facts_text": format_messages(messages_before_error),
                    "all_messages": messages_before_error,
                },
            )
        fallback = self.tool_fallbacks.get(tool_name, {}).get("message", "该服务暂时不可用。")
        return self._route_to(ReplyStage.CANNED_FALLBACK, {"text": fallback, "all_messages": messages_before_error})

    async def _gen_with_prompt(self, agent_role: str, user_query: str, extra_vars: Dict[str, Any],
                                fallback: str, stage: str, tool_choice: str = "none") -> str:
        """reply()专用：拼prompt→调LLM→异常兜底,把原来clarify/direct/param_error/final四处重复逻辑抽成一处"""
        format_kwargs = {"tool_facts": "暂无", **extra_vars, "agent_role": agent_role}
        prompt = SYSTEM_PROMPT.format(**format_kwargs)
        messages = [SystemMessage(content=prompt), HumanMessage(content=user_query)]
        try:
            total_start = time.monotonic()
            response = await self.llm_client.ainvoke(messages, tool_choice=tool_choice)
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                record_llm_metrics(provider=self.llm_client.provider,
                                   total_tokens=response.usage_metadata.get("total_tokens", 0),
                                   duration_ms=(time.monotonic() - total_start) * 1000)
            logger.info("[%s] LLM %s reply with: %s", self.agent_name, stage, response.content.strip()[:100])
            return response.content.strip()
        except Exception as e:
            agent_executor_errors_total.labels(
                agent_name=self.agent_name, stage=stage, error_type=type(e).__name__
            ).inc()
            logger.exception("[%s] LLM %s reply failed: %s", self.agent_name, stage, e)
            return fallback

    async def reply(self, state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
        """回复节点：唯一生成用户可见AIMessage的地方,读decide()留下的reply_stage/reply_payload完成生成"""
        context: AgentContext = state.get(StateFields.AGENT_CONTEXT.value)
        user_id = context.user_id
        session_id = context.session_id
        user_query = context.current_query

        stage = state.get(StateFields.REPLY_STAGE.value)
        payload = state.get(StateFields.REPLY_PAYLOAD.value) or {}
        agent_cfg = self.registry.get_config(self.agent_module)
        context_vars = self._build_context_vars(context)
        all_messages = payload.get("all_messages", [])

        if stage in (ReplyStage.CANNED_FALLBACK.value, ReplyStage.PASSTHROUGH.value):
            content = payload["text"]
        elif stage == ReplyStage.CLARIFY.value:
            content = await self._gen_with_prompt(
                agent_cfg.clarify_prompt, user_query, context_vars,
                fallback="抱歉，我没太理解您的需求，可以再具体描述一下吗？", stage="clarify")
        elif stage == ReplyStage.DIRECT.value:
            content = await self._gen_with_prompt(
                agent_cfg.direct_prompt, user_query, context_vars,
                fallback="抱歉，我暂时无法处理您的问题，请稍后再试。", stage="direct")
        elif stage == ReplyStage.PARAM_ERROR.value:
            param_error_role = agent_cfg.param_error_prompt.format(error_msg=payload["error_msg"])
            content = await self._gen_with_prompt(
                param_error_role, user_query, {**context_vars, "tool_facts": payload["tool_facts_text"]},
                fallback=f"抱歉，参数似乎有误：{payload['error_msg']}，请您重新提供正确的信息。", stage="param_error")
        elif stage == ReplyStage.FINAL.value:
            content = await self._gen_with_prompt(
                agent_cfg.res_prompt, user_query, {**context_vars, "tool_facts": payload["tool_facts_text"]},
                fallback="抱歉，我暂时无法生成回复，请稍后再试。", stage="final")
        else:
            logger.warning("[%s] unknown reply_stage=%s,fallback to generic message", self.agent_name, stage)
            content = "抱歉，我暂时无法处理您的问题，请稍后再试。"

        final_msg = AIMessage(content=content)
        assign_message_index(final_msg, user_id, session_id, self.seq_generator)
        return self._final_response(final_msg, all_messages, context)

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

    def _get_cb(self, tool_name: str) -> CircuitBreaker:
        if tool_name not in self._circuit_breakers:
            if self.cb_config.enabled:
                self._circuit_breakers[tool_name] = CircuitBreaker(
                    name=f"{self.agent_name}:{tool_name}",
                    failure_threshold=self.cb_config.failure_threshold,
                    recovery_timeout=self.cb_config.recovery_timeout_sec
                )
        return self._circuit_breakers[tool_name]

    async def _execute_tool_safe(self, tool_call: dict, trace_id: str, context: AgentContext) -> ToolResult:
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
            result = await asyncio.to_thread(cb.call, do_execute)
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

    async def _execute_skill_safe(self, skill_name: str, input_data: Dict[str, Any], trace_id: str,
                            context: AgentContext) -> ToolResult:
        cb = self._get_cb(skill_name)
        skill_config = self.skill_selector.get(skill_name)
        if not skill_config:
            return ToolResult(success=False, error=f"Skill {skill_name} 未找到配置",
                              error_type=ToolErrorType.BUSINESS_ERROR)

        skill_context = {
            "user_id": context.user_id,
            "session_id": context.session_id,
            "conversation_summary": context.conversation_summary,
            "profile_summary": context.user_profile_summary,
            "recent_conversation": context.recent_conversation,
        }

        def do_execute():
            result = self.skill_executor.execute(
                skill=skill_config,
                input_data=input_data,
                trace_id=trace_id,
                caller_agent=self.agent_name,
                **skill_context
            )
            return result

        try:
            result = await asyncio.to_thread(cb.call, do_execute)
            circuit_breaker_state.labels(tool_name=skill_name).set(0)
            data = result.get("data", str(result))
            return ToolResult(success=True, data=data, summary=data[:100] if isinstance(data, str) else "")
        except CircuitBreakerOpenError:
            circuit_breaker_state.labels(tool_name=skill_name).set(1)
            return ToolResult(success=False, error=f"Skill {skill_name} 暂时不可用",
                              error_type=ToolErrorType.CIRCUIT_OPEN)
        except ToolExecutionException as e:
            return ToolResult(success=False, error=str(e), error_type=e.error_type)
        except Exception as e:
            logger.exception(f"Skill {skill_name} 执行异常")
            return ToolResult(success=False, error=str(e), error_type=ToolErrorType.EXTERNAL_ERROR)

    def _handle_tool_result(self, tool_name: str, tool_result: ToolResult) -> Optional[str]:
        handler_config = self.response_handlers_config.get(tool_name)
        if not handler_config:
            return None

        if tool_name == 'upsert_loan_interest':
            handler = UpsertLoanInterestHandler(tool_result.data, handler_config)
            return handler.generate()

        return None
