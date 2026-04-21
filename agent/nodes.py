# author hgh
# version 1.0
import logging
import re
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

from agent.state import AgentState
from config import config
from exception import MemoryWriteFailedError
from memory.base import BaseMemoryStore
from memory.constant.constants import MemoryType, StateFields, MetadataFields, MemoryModelFields, MemoryStatus, \
    MemorySource, ComplianceSeverity, ComplianceAction, ComplianceRuleFields, PromptKeys, ConfigFields, EvidenceType, \
    InteractionEventType, InteractionSentiment, ProfileEntityKey
from prompt.extract_prompt import EXTRACT_PROMPT
from prompt.system_prompt import SYSTEM_TEMPLATE
from retriever.base import BaseRetriever
from utils.llm import get_llm
from utils.parser import safe_parse_extraction_output

logger = logging.getLogger(__name__)
llm = get_llm()


def call_model_node(state: AgentState, agent_config: RunnableConfig) -> dict:
    """call llm"""
    formatted = state.get(StateFields.FORMATTED_CONTEXT.value, {})
    system = SYSTEM_TEMPLATE.format(
        user_profile=formatted.get(MemoryType.USER_PROFILE.value, "暂无"),
        compliance_rule=formatted.get(MemoryType.COMPLIANCE_RULE.value, "暂无"),
        interaction_log=formatted.get(MemoryType.INTERACTION_LOG.value, "暂无")
    )
    recent = state[StateFields.MESSAGES.value][-config.max_context_messages:]
    messages = [SystemMessage(content=system)] + recent
    response = llm.invoke_with_fallback(messages,fallback_response="Sorry, I am temporarily unable to handle your request. Please try again later.")
    return {StateFields.MESSAGES.value: [response]}


def extract_profile_node(state: AgentState, agent_config: RunnableConfig, memory_store: BaseMemoryStore) -> dict:
    """extract user profile and save to store"""
    # get user id
    user_id = state.get(MetadataFields.USER_ID.value)
    if not user_id:
        logger.warning("No user_id found in state, skipping profile extraction")
        return {StateFields.PROFILE_UPDATED.value: False}

    #get history message
    messages = state.get(StateFields.MESSAGES.value, [])
    if not messages:
        logger.debug("No messages to extract profile from")
        return {StateFields.PROFILE_UPDATED.value: False}
    recent = messages[-6:]
    conversation = "\n".join(f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}" for m in recent)

    try:
        chain = EXTRACT_PROMPT | llm | StrOutputParser()
        extract_str = chain.invoke({PromptKeys.CONVERSATION.value: conversation})
    except Exception as e:
        logger.error(f"LLM profile extraction failed: {e}")
        return {StateFields.PROFILE_UPDATED.value: False}

    # analyze the information that extracted by llm
    items = safe_parse_extraction_output(extract_str)
    allowed_entity_keys = {e.value for e in ProfileEntityKey}

    updated = False
    for item in items:
        if item.get(MemoryModelFields.CONTENT.value) and item.get(MetadataFields.ENTITY_KEY.value):
            content = item.get(MemoryModelFields.CONTENT.value)
            entity_key_raw = item.get(MetadataFields.ENTITY_KEY.value)

            # 校验必要字段及 entity_key 合法性
            if not content or not entity_key_raw:
                continue
            if entity_key_raw not in allowed_entity_keys:
                logger.warning(f"Ignored invalid entity_key '{entity_key_raw}' from LLM extraction")
                continue

            metadata = {
                MetadataFields.TYPE.value: MemoryType.USER_PROFILE.value,
                MetadataFields.SOURCE.value: MemorySource.CHAT_EXTRACTION.value,
                MetadataFields.CONFIDENCE.value: item.get(MetadataFields.CONFIDENCE.value, 0.7),
                MetadataFields.STATUS.value: MemoryStatus.ACTIVE.value,
                MetadataFields.EVIDENCE_TYPE.value: EvidenceType.EXPLICIT_STATEMENT.value,
                MetadataFields.EFFECTIVE_DATE.value: datetime.now().isoformat(),
                MetadataFields.EXPIRES_AT.value: None,
            }
            try:
                memory_store.add_memory(
                    user_id=user_id,
                    content=item.get(MemoryModelFields.CONTENT.value),
                    memory_type=MemoryType.USER_PROFILE,
                    entity_key=item.get(MetadataFields.ENTITY_KEY.value),
                    metadata=metadata
                )
                updated = True
            except MemoryWriteFailedError as e:
                logger.error(f"Memory write failed (DLQ): {e}")
            except Exception as e:
                logger.error(f"Unexpected error during profile extraction: {e}")
    return {StateFields.PROFILE_UPDATED.value: updated}


def retrieve_memory_node(state: AgentState, retrieval: BaseRetriever) -> dict:
    """retrieve memory and add content to the context"""
    user_id = state.get(StateFields.USER_ID.value, "unknown")
    messages = state.get(StateFields.MESSAGES.value, [])

    user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    memory_types = [
        MemoryType.USER_PROFILE,
        MemoryType.COMPLIANCE_RULE,
        MemoryType.INTERACTION_LOG
    ]
    try:
        context = retrieval.retrieve(query=user_query, user_id=user_id,memory_types=memory_types)
    except Exception as e:
        logger.error(f"Retrieval failed, using empty context: {e}")
        context = {MemoryType.USER_PROFILE.value: [], MemoryType.COMPLIANCE_RULE.value: [],
                   MemoryType.INTERACTION_LOG.value: []}
        return {
            StateFields.RETRIEVED_CONTEXT.value: context,
            StateFields.FORMATTED_CONTEXT.value: {
                MemoryType.USER_PROFILE.value: "暂无相关信息",
                MemoryType.COMPLIANCE_RULE.value: "暂无相关信息",
                MemoryType.INTERACTION_LOG.value: "暂无相关信息"
            },
            StateFields.ERROR.value: f"Retrieval error: {e}"
        }

    def fmt(mems):
        return "\n".join(f"- {m[MemoryModelFields.CONTENT.value]}" for m in mems) if mems else "暂无相关信息"

    formatted = {
        MemoryType.USER_PROFILE.value: fmt(context.get(MemoryType.USER_PROFILE.value, [])),
        MemoryType.COMPLIANCE_RULE.value: fmt(context.get(MemoryType.COMPLIANCE_RULE.value, [])),
        MemoryType.INTERACTION_LOG.value: fmt(context.get(MemoryType.INTERACTION_LOG.value, []))
    }
    return {StateFields.RETRIEVED_CONTEXT.value: context, StateFields.FORMATTED_CONTEXT.value: formatted, StateFields.ERROR.value: None}


def log_interaction_node(state: AgentState, agent_config: RunnableConfig, memory_store: BaseMemoryStore):
    """generate a conversation summary and store it in the interaction memory"""
    # 获取 session_id
    configurable = agent_config.get(ConfigFields.CONFIGURABLE.value, {})
    session_id = configurable.get(ConfigFields.THREAD_ID.value, "unknown")

    recent = state.get(StateFields.MESSAGES.value,[])
    if len(recent) > config.interaction_recent_num:
        recent = recent[config.interaction_recent_num:]
    conversation = "\n".join(
        f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
        for m in recent
    )

    summary_prompt = f"请用一句话总结以下对话核心内容，不要包含冗余信息:\n{conversation}\n\n摘要:"
    summary = llm.invoke_with_fallback([HumanMessage(content=summary_prompt)],fallback_response="Failed to generate conversation summary").content

    metadata = {
        MetadataFields.TYPE.value: MemoryType.INTERACTION_LOG.value,
        MetadataFields.SOURCE.value: MemorySource.AUTO_SUMMARY.value,
        MetadataFields.STATUS.value: MemoryStatus.ACTIVE.value,
        MetadataFields.CONFIDENCE.value: 1.0,
        MetadataFields.EVENT_TYPE.value: InteractionEventType.INQUIRY.value,
        MetadataFields.SESSION_ID.value: session_id,
        MetadataFields.SENTIMENT.value: InteractionSentiment.NEUTRAL.value,
        MetadataFields.KEY_ENTITIES.value: [],  # 暂留空，后续可 NLP 提取
        MetadataFields.TIMESTAMP.value: datetime.now().isoformat(),
    }

    try:
        memory_store.add_memory(
            user_id=state[MetadataFields.USER_ID.value],
            content=summary,
            memory_type=MemoryType.INTERACTION_LOG,
            metadata=metadata
        )
        logger.info(f"Logged interaction for session {session_id}")
    except Exception as e:
        logger.error(f"Failed to write interaction log: {e}")
    return {"interaction_logged": True}


def compliance_guard_node(state: AgentState, agent_config: RunnableConfig, memory_store: BaseMemoryStore) -> dict:
    """
    compliance check node:
    scan user input and draft response before generating answers to intercept non-compliant content
    """
    # get rules that retrieved at retrieval_memory_node from context
    rules = state.get(StateFields.RETRIEVED_CONTEXT.value, {}).get(MemoryType.COMPLIANCE_RULE.value, [])

    # find user input
    user_query = ""
    for msg in reversed(state.get(StateFields.MESSAGES.value,[])):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    blocked = False
    block_reason = ""
    warnings = []
    mandatory_appends = []

    for rule in rules:
        rule_meta = rule.get(MemoryModelFields.METADATA.value, rule)
        action = rule_meta.get(ComplianceRuleFields.ACTION.value, ComplianceAction.WARN.value)
        pattern = rule_meta.get(ComplianceRuleFields.PATTERN.value, "")
        severity = rule_meta.get(ComplianceRuleFields.SEVERITY.value, ComplianceSeverity.LOW.value)

        if pattern:
            try:
                if re.search(pattern, user_query, re.IGNORECASE):
                    if action == ComplianceAction.BLOCK.value and severity in [ComplianceSeverity.CRITICAL.value,
                                                                               ComplianceSeverity.HIGH.value]:
                        blocked = True
                        block_reason = rule.get(ComplianceRuleFields.RULE_NAME.value, "合规规则")
                        logger.error(f"Compliance guard blocked {block_reason} - {user_query[:50]}")
                        break
                    elif action == ComplianceAction.WARN.value:
                        warnings.append(rule_meta.get(ComplianceRuleFields.DESCRIPTION.value, ""))
            except Exception as e:
                logger.warning(f"Rule {rule_meta.get(ComplianceRuleFields.RULE_ID.value)}'s regular expression invalid")

    if blocked:
        compliance_response = AIMessage(
            content="Sorry,your question involves non-compliant content,and i cannot answer it,\n\nif you have any questions,please contact our official customer service"
        )
        return {
            StateFields.MESSAGES.value: [compliance_response],
            StateFields.COMPLIANCE_BLOCKED.value: True,
            StateFields.BLOCK_REASON.value: block_reason,
            StateFields.SHOULD_SKIP_LLM.value: True
        }

    for rule in rules:
        rule_meta = rule.get(MemoryModelFields.METADATA.value, rule)
        if rule_meta.get(ComplianceRuleFields.ACTION.value) == ComplianceAction.APPEND.value and rule_meta.get(
                ComplianceRuleFields.TEMPLATE.value):
            mandatory_appends.append(rule_meta[ComplianceRuleFields.TEMPLATE.value])

    return {
        StateFields.COMPLIANCE_BLOCKED.value: False,
        StateFields.COMPLIANCE_WARNINGS.value: warnings,
        StateFields.MANDATORY_APPENDS.value: mandatory_appends,
        StateFields.SHOULD_SKIP_LLM.value: False
    }
