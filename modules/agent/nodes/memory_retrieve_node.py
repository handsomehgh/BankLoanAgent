# author hgh
# version 1.0
# author hgh
# version 1.0
"""
public memory retrieval layer
responsibilities: called uniformly by the supervisor to obtain user profile,compliance rules,interaction log and assemble them into a formatted context,
it also completes the global message sequence number assignment
"""
import logging
import time
from typing import Dict, List

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from config.global_constant.constants import ConfigFields, MemoryType
from config.global_constant.fields import CommonFields
from config.models.memory_config import MemorySystemConfig
from modules.agent.constants import StateFields, MessageCommonFields, AgentName
from modules.agent.multi_agent_state import SupervisorState
from modules.memory.base import BaseRetriever
from modules.memory.memory_constant.fields import MemoryFields
from utils.monitor_utils.metrics import memory_read_duration_seconds, memory_hit_total
from utils.serialize_utils.seq_generator import SequenceGenerator

logger = logging.getLogger(__name__)


class MemoryRetrieveNode:
    def __init__(self, retriever: BaseRetriever, seq_generator: SequenceGenerator, memory_config: MemorySystemConfig):
        self.retriever = retriever
        self.seq_generator = seq_generator
        self.memory_config = memory_config

    def __call__(self, state: SupervisorState, config: RunnableConfig):
        logger.info("[MemoryRetrieveNode] starting retrieving with state, state=%s", state)

        user_id = state.get(StateFields.USER_ID.value, "unknown")
        logger.info("[MemoryRetrieveNode] retrieving memory for user_id=%s", user_id)
        messages = state.get(StateFields.MESSAGES.value, [])

        # 1. assign global message sequence number
        self._assign_message_indexes(messages, user_id, config)

        # 2. obtain user query
        user_query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break
        if user_query:
            logger.info("[MemoryRetrieveNode] memory retrieval for user_id=%s, query='%.60s...'", user_id, user_query)
        else:
            logger.warning("[MemoryRetrieveNode] no user query found, returning empty context")

        # 3. execute memory retrieval
        memory_types = [
            MemoryType.USER_PROFILE,
            MemoryType.INTERACTION_LOG,
            MemoryType.SUB_INTERACTION_LOG
        ]
        try:
            start_time = time.monotonic()
            retrieved = self.retriever.retrieve(query=user_query, user_id=user_id, memory_types=memory_types)
            memory_read_duration_seconds.labels(user=user_id).observe(time.monotonic() - start_time)

            profile_count = len(retrieved.get(MemoryType.USER_PROFILE, []))
            sub_count = len(retrieved.get(MemoryType.SUB_INTERACTION_LOG, []))
            interaction_count = len(retrieved.get(MemoryType.INTERACTION_LOG, []))
            memory_hit_total.labels(user=user_id, type=MemoryType.SUB_INTERACTION_LOG.value).inc(sub_count)
            memory_hit_total.labels(user=user_id, type=MemoryType.USER_PROFILE.value).inc(profile_count)
            memory_hit_total.labels(user=user_id, type=MemoryType.INTERACTION_LOG.value).inc(interaction_count)

            logger.info(
                "Memory retrieval succeeded: user_profile=%d, sub_interaction=%d, interaction_logs=%d",
                profile_count, sub_count, interaction_count
            )
        except Exception as e:
            logger.error("Memory retrieval failed: %s", e, exc_info=True)
            empty_formatted = {
                MemoryType.USER_PROFILE.value: "暂无相关记录",
                MemoryType.INTERACTION_LOG.value: "暂无相关记录",
                MemoryType.SUB_INTERACTION_LOG.value: {}
            }
            return {
                StateFields.RETRIEVED_CONTEXT.value: {
                    MemoryType.USER_PROFILE.value: [],
                    MemoryType.INTERACTION_LOG.value: [],
                    MemoryType.SUB_INTERACTION_LOG.value: {}
                },
                StateFields.FORMATTED_CONTEXT.value: empty_formatted,
                StateFields.ERROR.value: f"Memory retrieval error: {e}",
            }

        # 4. formatted context
        def fmt_memories(mems: List[Dict]) -> str:
            if not mems:
                return "暂无相关记录"
            return "\n".join(f"- {m[CommonFields.TEXT]}" for m in mems)

        # assemble sub logs
        sub_log_dict = {}
        sub_log = retrieved.get(MemoryType.SUB_INTERACTION_LOG)
        if sub_log:
            loan_advisor_log = [l for l in sub_log if
                                l.get(CommonFields.METADATA, {}).get(
                                    CommonFields.SOURCE) == AgentName.LOAN_ADVISOR.value]
            sub_log_dict[AgentName.LOAN_ADVISOR.value] = fmt_memories(
                loan_advisor_log) if loan_advisor_log else "暂无相关记录"

            after_loan_log = [l for l in sub_log if
                              l.get(CommonFields.METADATA, {}).get(CommonFields.SOURCE) == AgentName.AFTER_LOAN.value]
            sub_log_dict[AgentName.AFTER_LOAN.value] = fmt_memories(after_loan_log) if after_loan_log else "暂无相关记录"

            risk_assessment_log = [l for l in sub_log if l.get(CommonFields.METADATA, {}).get(
                CommonFields.SOURCE) == AgentName.RISK_ASSESSMENT.value]
            sub_log_dict[AgentName.RISK_ASSESSMENT.value] = fmt_memories(
                risk_assessment_log) if risk_assessment_log else "暂无相关记录"

        formatted = {
            MemoryType.USER_PROFILE.value: self._fmt_profile_memories(retrieved.get(MemoryType.USER_PROFILE, [])),
            MemoryType.INTERACTION_LOG.value: fmt_memories(retrieved.get(MemoryType.INTERACTION_LOG, [])),
            MemoryType.SUB_INTERACTION_LOG.value: sub_log_dict,
        }

        return {
            StateFields.RETRIEVED_CONTEXT.value: retrieved,
            StateFields.FORMATTED_CONTEXT.value: formatted,
            StateFields.ERROR.value: None,
        }

    def _fmt_profile_memories(self, mems: List[Dict]) -> str:
        if not mems:
            return "暂无相关记录"
        lines = []
        for m in mems:
            metadata = m.get(MemoryFields.METADATA, {})
            entity_key = metadata.get(MemoryFields.ENTITY_KEY, 'unknown')
            content = m.get(MemoryFields.TEXT, '')
            confidence = metadata.get(MemoryFields.CONFIDENCE, 0.0)
            if entity_key in self.memory_config.sensitive_keys:
                if len(content) > 4:
                    content = content[:2] + "*" * (len(content) - 4) + content[-2:]
                else:
                    content = "***"

            if confidence >= self.memory_config.high_conf_threshold:
                # 高置信度：正常标注
                lines.append(f"- {entity_key}: {content}")
                continue
            elif confidence >= self.memory_config.medium_conf_threshold:
                # 中置信度：加注提示
                lines.append(f"- {entity_key}: {content}（该信息置信度中等，仅供参考,建议结合上下文综合判断）")
                continue
            else:
                # 低置信度：默认不展示，或加强烈提示
                lines.append(f"- {entity_key}: {content}（该信息置信度较低，不应用于关键决策，如有疑问应向用户确认）")
        return "\n".join(lines)

    def _assign_message_indexes(
            self,
            messages: List[BaseMessage],
            user_id: str,
            config: RunnableConfig
    ) -> None:
        """
        Assign numbers to all user/assistant messages that do not have a global sequence number.
        """
        configurable = config.get(ConfigFields.CONFIGURABLE.value, {})
        session_id = configurable.get(ConfigFields.THREAD_ID.value, "unknown")

        for msg in messages:
            if not hasattr(msg, MessageCommonFields.ADDITIONAL_KWARGS) or msg.additional_kwargs is None:
                msg.additional_kwargs = {}
            if MessageCommonFields.MESSAGE_INDEX not in msg.additional_kwargs:
                idx = self.seq_generator.next_seq(user_id, session_id)
                msg.additional_kwargs[MessageCommonFields.MESSAGE_INDEX] = idx
