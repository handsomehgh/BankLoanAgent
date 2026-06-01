# author hgh
# version 1.0
"""
HumanHandoff Node: Generate human handoff summary and immediately downgrade
"""
import logging
from datetime import datetime, timezone
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from config.global_constant.constants import ConfigFields
from config.global_constant.fields import CommonFields
from infra.database.redis_manager import RedisManager
from modules.agent.constants import StateFields, AgentContextFields, AgentNodeName
from modules.agent.human_handoff.handoff_timeout_monitor import HandoffTimeoutMonitor
from modules.agent.multi_agent_state import AgentContext, SupervisorState
from utils.monitor_utils.metrics import handoff_task_created_total, handoff_task_completed_total

logger = logging.getLogger(__name__)

HANDOFF_TASK_KEY = "handoff_task"
DEGRADE_MESSAGE = "当前人工坐席繁忙，请稍后重试或拨打我行客服热线 95333。"

class HumanHandoffInterruptNode:
    def __init__(self, redis_manager: RedisManager):
        self.redis_manager = redis_manager

    def __call__(self, state: SupervisorState, config: RunnableConfig):
        logger.info("[HumanHandoffInterruptNone] starting process interrupt by human with state %s",
                    state)

        # 1. obtain context
        ctx = state.get(StateFields.AGENT_CONTEXT.value, {}).get(AgentNodeName.HUMAN_HANDOFF_NOTIFY.value)
        user_id = state.get(StateFields.USER_ID.value, "")
        trace_id = ctx.trace_id
        thread_id = config.get(ConfigFields.CONFIGURABLE.value, {}).get(ConfigFields.THREAD_ID.value, "")
        user_query = ctx.current_query if ctx else ""
        user_profile = ctx.user_profile_summary if ctx else ""
        conversation_summary = ctx.conversation_summary if ctx else ""
        compliance_warnings = ctx.compliance_warnings if ctx else []

        # 2. generate transfer summary
        handoff_summary = "\n".join([
            f"用户问题: {user_query}",
            f"用户画像: {user_profile}",
            f"对话历史: {conversation_summary}",
            f"合规警告: {', '.join(compliance_warnings) if compliance_warnings else '无'}"
        ])
        logger.info("[HumanHandoffInterruptNone] interrupt suspend, trace_id=%s", trace_id)

        # 3. send the work order to redis
        timestamp = datetime.now(timezone.utc)
        client = self.redis_manager.get_client()
        if client:
            # add to pending ordered set(for timeout monitoring)
            HandoffTimeoutMonitor.add_pending_task(self.redis_manager, thread_id, timestamp.timestamp())

            # store work order to set
            task_info = {
                AgentContextFields.TRACE_ID.value: trace_id,
                CommonFields.USER_ID: user_id,
                StateFields.HANDOFF_SUMMARY.value: handoff_summary,
                CommonFields.TIMESTAMP: timestamp.isoformat()
            }
            client.hset(f"{HANDOFF_TASK_KEY}:{thread_id}", mapping=task_info)
            client.expire(f"{HANDOFF_TASK_KEY}:{thread_id}", 3600)
            logger.info("[HumanHandoffInterruptNone] work order has been stored: thread_id=%s", thread_id)

        # 4. execute interrupt
        resume_value = interrupt({
            "type": "human_handoff",
            "handoff_summary": handoff_summary
        })
        handoff_task_created_total.inc()

        # 5. organize result
        action = resume_value.get("action", "close")
        content = resume_value.get("content", DEGRADE_MESSAGE) if action == "reply" else DEGRADE_MESSAGE
        handoff_task_completed_total.labels(result=action).inc()

        # 6. clean work order
        if client:
            HandoffTimeoutMonitor.remove_pending_task(self.redis_manager, thread_id)
            client.delete(f"{HANDOFF_TASK_KEY}:{thread_id}")
            logger.info("[HumanHandoffInterruptNone] work order has been clean: thread_id=%s", thread_id)

        return {
            StateFields.MESSAGES.value: [AIMessage(content=f"工号001:{content}")],
            StateFields.HANDOFF_SUMMARY.value: handoff_summary,
            StateFields.NEXT_AGENTS.value: [],
            StateFields.AGENT_RESPONSES.value: {},
            StateFields.AGENT_CONTEXT.value: None,
            StateFields.TRIGGER_HUMAN_HANDOFF.value: False
        }
