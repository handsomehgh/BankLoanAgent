# author hgh
# version 1.0
import logging
import re

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from config.global_constant.fields import CommonFields
from config.global_constant.constants import MemoryType, ComplianceAction, ComplianceSeverity
from config.models.memory_config import MemorySystemConfig
from modules.agent.constants import StateFields
from modules.agent.state import AgentState
from modules.memory.memory_business_store.base_memory_store import BaseMemoryStore

logger = logging.getLogger(__name__)


def compliance_guard_node(state: AgentState, config: RunnableConfig, memory_store: BaseMemoryStore,
                          memory_config: MemorySystemConfig) -> dict:
    """
    compliance check node:
    scan user input and draft response before generating answers to intercept non-compliant content
    """
    # get rules that retrieved at retrieval_memory_node from context
    rules = state.get(StateFields.RETRIEVED_CONTEXT, {}).get(MemoryType.COMPLIANCE_RULE, [])

    # find user input
    messages = state.get(StateFields.MESSAGES, [])
    user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    # match all rules that are hit
    hit_rules = []
    for rule in rules:
        meta = rule.get(CommonFields.METADATA, rule)
        pattern = meta.get(CommonFields.PATTERN)
        if not pattern:
            continue
        try:
            if re.search(pattern, user_query, re.IGNORECASE):
                hit_rules.append(rule)
        except re.error as e:
            logger.warning(f"Invalid regex in rule {meta.get(CommonFields.RULE_ID)}")

    # load rule severity level from yaml
    severity_model = memory_config.compliance_severity
    hit_rules.sort(key=lambda r: (
        getattr(severity_model, r.get(CommonFields.SEVERITY, "medium"), 4),
        r.get(CommonFields.PRIORITY, 100)
    ))

    blocked = False
    block_reason = ""
    warnings = []
    mandatory_appends = []

    # handle according to different rules
    for rule in hit_rules:
        action = rule.get(CommonFields.ACTION, ComplianceAction.WARN)
        severity = rule.get(CommonFields.SEVERITY)

        if action == ComplianceAction.BLOCK and severity in [ComplianceSeverity.CRITICAL,
                                                             ComplianceSeverity.HIGH]:
            blocked = True
            block_reason = rule.get(CommonFields.RULE_NAME, "合规规则")
            break
        elif action == ComplianceAction.WARN:
            warnings.append(rule.get(CommonFields.DESCRIPTION, ""))
        elif action == ComplianceAction.APPEND:
            template = rule.get(CommonFields.TEMPLATE)
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
