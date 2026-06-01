# author hgh
# version 1.0
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class AgentNodeName(str, Enum):
    RETRIEVE_MEMORY = "retrieve_memory"
    RETRIEVE_KNOWLEDGE = "retrieve_knowledge"
    COMPLIANCE_GUARD = "compliance_guard"
    CALL_MODEL = "call_model"
    EXTRACT_PROFILE = "extract_profile"
    LOG_INTERACTION = "log_interaction"

    LOAN_ADVISOR_RESPONSE = "loan_advisor_response"
    SUPERVISOR_ROUTE_NODE = "supervisor_route_node"
    AFTER_LOAN_RESPONSE = "after_loan_response"
    RISK_ASSESSMENT_RESPONSE = "risk_assessment_response"

    HUMAN_HANDOFF_NOTIFY = "human_handoff_notify"
    HUMAN_HANDOFF_INTERRUPT = "human_handoff_interrupt"

    COMPLIANCE_PREFILTER = "compliance_prefilter"
    MEMORY_RETRIEVE = "memory_retrieve"
    DIRECT_REPLY = "direct_reply"
    RESULT_AGGREGATOR = "result_aggregator"
    FANOUT_DISPATCHER = "fanout_dispatcher"
    ENSURE_MESSAGE_INDEX = "ensure_message_index"

class AgentName(str, Enum):
    SUPERVISOR = "supervisor"
    HUMAN_HANDOFF = "human_handoff"
    LOAN_ADVISOR = "LoanAdvisor"
    RISK_ASSESSMENT = "RiskAssessment"
    AFTER_LOAN = "AfterLoan"

class RouteTarget(str, Enum):
    DIRECT = "direct"
    HUMAN_HANDOFF = "human_handoff"
    HUMAN_HANDOFF_NOTIFY = "human_handoff_notify"
    LOAN_ADVISOR = "LoanAdvisor"
    RISK_ASSESSMENT = "RiskAssessment"
    AFTER_LOAN = "AfterLoan"
    UNKNOWN = "unknown"
    LLM_ERROR = "LLM_error"

@dataclass
class RouteDecision:
    target_agents: List[str] = field(default_factory=list)
    special: Optional[str] = None

class StateFields(str, Enum):
    USER_ID = "user_id"
    MESSAGES = "messages"
    INTERNAL_MESSAGES = "internal_messages"
    SUB_MESSAGES = "sub_messages"
    AGENT_CONTEXT = "agent_context"
    RETRIEVED_CONTEXT = "retrieved_context"
    FORMATTED_CONTEXT = "formatted_context"
    PROFILE_UPDATED = "profile_updated"
    INTERACTION_LOGGED = "interaction_logged"
    COMPLIANCE_BLOCKED = "compliance_blocked"
    COMPLIANCE_WARNINGS = "compliance_warnings"
    BLOCK_REASON = "block_reason"
    MANDATORY_APPENDS = "mandatory_appends"
    SHOULD_SKIP_LLM = "should_skip_llm"
    ERROR = "error"
    LAST_EXTRACTED_MESSAGE_INDEX = "last_extracted_message_index"
    NEXT_MESSAGE_INDEX = "next_message_index"
    LAST_LOGGED_MESSAGE_INDEX = "last_logged_message_index"
    NEXT_AGENTS = "next_agents"
    CLARIFICATION_COUNT = "clarification_count"
    AGENT_RESPONSES = "agent_responses"
    FINAL_RESPONSE = "final_response"
    HANDOFF_SUMMARY = "handoff_summary"
    TRIGGER_HUMAN_HANDOFF = "trigger_handoff"
    SHOULD_SKIP_SUPERVISOR = "should_skip_supervisor"

class AgentContextFields(str, Enum):
    TRACE_ID = "trace_id"
    AUDIT_LOG = "audit_log"

class PromptKeys(str, Enum):
    CONVERSATION = "conversation"
    KNOWN_PROFILE = "known_profile"

class MessageCommonFields(str, Enum):
    ADDITIONAL_KWARGS = "additional_kwargs"
    MESSAGE_INDEX = "message_index"

class StreamName(str,Enum):
    INTERACTION_LOG = "interaction_log"
    SUB_INTERACTION = "sub_interaction"

class ConsumerGroupName(str,Enum):
    PROFILE_GROUP_FIRST = "profile_group_first"
    INTERACTION_GROUP_FIRST = "interaction_group_first"

class ToolMode(str,Enum):
    SKILL = "skill"
    HYBRID = "hybrid"




