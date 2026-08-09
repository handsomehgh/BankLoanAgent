import logging
import time
from typing import Dict, Any, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from config.prompt_hub import PromptHub
from config.registry import ConfigRegistry
from modules.agent.constants import StateFields, ReplyStage
from modules.agent.multi_agent_state import AgentContext, AgentResponse
from modules.agent.nodes.node_common import build_context_vars
from modules.module_services.chat_models import RobustLLM
from modules.tools.common_utils import assign_message_index
from utils.monitor_utils.metrics import record_llm_metrics, agent_executor_errors_total
from utils.serialize_utils.seq_generator import SequenceGenerator

logger = logging.getLogger(__name__)


class AgentReplyNode:
    """
    reply节点执行器:唯一生成用户可见AIMessage的地方,
    读decision节点留下的reply_stage/reply_payload完成最终回复生成
    """

    def __init__(
            self,
            agent_module: str,
            agent_name: str,
            registry: ConfigRegistry,
            llm_client: RobustLLM,
            seq_generator: SequenceGenerator,
            post_process: Optional[callable] = None,
            prompt_hub: Optional[PromptHub] = None
    ):
        self.agent_module = agent_module
        self.agent_module_key = getattr(agent_module, "value", agent_module)
        self.agent_name = agent_name
        self.registry = registry
        self.prompt_hub = prompt_hub or PromptHub(registry)
        self.llm_client = llm_client
        self.seq_generator = seq_generator
        self.post_process = post_process

    async def _gen_with_prompt(self, agent_role: str, user_query: str, extra_vars: Dict[str, Any],
                               fallback: str, stage: str, tool_choice: str = "none") -> str:
        """reply()专用：拼prompt→调LLM→异常兜底,把原来clarify/direct/param_error/final四处重复逻辑抽成一处"""
        format_kwargs = {"tool_facts": "暂无", "proactive_hint": "无", **extra_vars, "agent_role": agent_role}
        prompt = self.prompt_hub.render_text("system_prompt", **format_kwargs)
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
        """回复节点：唯一生成用户可见AIMessage的地方,读decision节点留下的reply_stage/reply_payload完成生成"""
        context: AgentContext = state.get(StateFields.AGENT_CONTEXT.value)
        user_id = context.user_id
        session_id = context.session_id
        user_query = context.current_query

        stage = state.get(StateFields.REPLY_STAGE.value)
        payload = state.get(StateFields.REPLY_PAYLOAD.value) or {}
        context_vars = build_context_vars(context)
        # written by proactive_suggestion_gate(loan_advisor only),other agents always get "无"
        proactive_hint = state.get(StateFields.PROACTIVE_HINT.value) or "无"
        all_messages = payload.get("all_messages", [])

        if stage in (ReplyStage.CANNED_FALLBACK.value, ReplyStage.PASSTHROUGH.value):
            content = payload["text"]
        elif stage == ReplyStage.CLARIFY.value:
            content = await self._gen_with_prompt(
                self.prompt_hub.get_text(f"{self.agent_module_key}_clarify"), user_query, context_vars,
                fallback="抱歉，我没太理解您的需求，可以再具体描述一下吗？", stage="clarify")
        elif stage == ReplyStage.DIRECT.value:
            content = await self._gen_with_prompt(
                self.prompt_hub.get_text(f"{self.agent_module_key}_direct"), user_query, context_vars,
                fallback="抱歉，我暂时无法处理您的问题，请稍后再试。", stage="direct")
        elif stage == ReplyStage.PARAM_ERROR.value:
            param_error_role = self.prompt_hub.render_text(
                f"{self.agent_module_key}_param_error", error_msg=payload["error_msg"])
            content = await self._gen_with_prompt(
                param_error_role, user_query, {**context_vars, "tool_facts": payload["tool_facts_text"]},
                fallback=f"抱歉，参数似乎有误：{payload['error_msg']}，请您重新提供正确的信息。", stage="param_error")
        elif stage == ReplyStage.FINAL.value:
            content = await self._gen_with_prompt(
                self.prompt_hub.get_text(f"{self.agent_module_key}_res"), user_query,
                {**context_vars, "tool_facts": payload["tool_facts_text"], "proactive_hint": proactive_hint},
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
