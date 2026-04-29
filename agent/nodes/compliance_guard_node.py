# author hgh
# version 1.0
import logging
import re

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from agent.state import AgentState
from config.constants import MemoryType, StateFields, GeneralFieldNames, ComplianceAction, ComplianceSeverity
from memory.base_memory_store import BaseMemoryStore
from memory.classifiers.rules.rules_loader import get_compliance_loader

logger = logging.getLogger(__name__)

def compliance_guard_node(state: AgentState, config: RunnableConfig, memory_store: BaseMemoryStore) -> dict:
    """
    compliance check node:
    scan user input and draft response before generating answers to intercept non-compliant content
    """
    # get rules that retrieved at retrieval_memory_node from context
    rules = state.get(StateFields.RETRIEVED_CONTEXT.value, {}).get(MemoryType.COMPLIANCE_RULE.value, [])

    # find user input
    messages = state.get(StateFields.MESSAGES.value, [])
    user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    # match all rules that are hit
    hit_rules = []
    for rule in rules:
        meta = rule.get(GeneralFieldNames.METADATA, rule)
        pattern = meta.get(GeneralFieldNames.PATTERN)
        if not pattern:
            continue
        try:
            if re.search(pattern, user_query, re.IGNORECASE):
                hit_rules.append(rule)
        except re.error as e:
            logger.warning(f"Invalid regex in rule {meta.get(GeneralFieldNames.RULE_ID)}")

    # load rule severity level from yaml
    loader = get_compliance_loader()
    severity_order = loader.get_compliance_severity()
    hit_rules.sort(key=lambda r: (severity_order.get(r.get(GeneralFieldNames.SEVERITY), 4),
                                  r.get(GeneralFieldNames.PRIORITY, 100)))

    blocked = False
    block_reason = ""
    warnings = []
    mandatory_appends = []

    # handle according to different rules
    for rule in hit_rules:
        action = rule.get(GeneralFieldNames.ACTION, ComplianceAction.WARN.value)
        severity = rule.get(GeneralFieldNames.SEVERITY)

        if action == ComplianceAction.BLOCK.value and severity in [ComplianceSeverity.CRITICAL.value,
                                                                   ComplianceSeverity.HIGH.value]:
            blocked = True
            block_reason = rule.get(GeneralFieldNames.RULE_NAME, "合规规则")
            break
        elif action == ComplianceAction.WARN.value:
            warnings.append(rule.get(GeneralFieldNames.DESCRIPTION, ""))
        elif action == ComplianceAction.APPEND.value:
            template = rule.get(GeneralFieldNames.TEMPLATE)
            if template:
                mandatory_appends.append(template)

    # if action equals blocking,return immediately
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

    return {
        StateFields.COMPLIANCE_BLOCKED.value: False,
        StateFields.COMPLIANCE_WARNINGS.value: warnings,
        StateFields.MANDATORY_APPENDS.value: mandatory_appends,
        StateFields.SHOULD_SKIP_LLM.value: False
    }

