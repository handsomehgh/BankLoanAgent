# author hgh
# version 1.0
from enum import Enum

class AgentNodeName(str, Enum):
    RETRIEVE = "retrieve"
    COMPLIANCE_GUARD = "compliance_guard"
    CALL_MODEL = "call_model"
    EXTRACT_PROFILE = "extract_profile"
    LOG_INTERACTION = "log_interaction"

class StateFields(str, Enum):
    USER_ID = "user_id"
    MESSAGES = "messages"
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

class PromptKeys(str, Enum):
    CONVERSATION = "conversation"
    KNOWN_PROFILE = "known_profile"

class MessageCommonFields(str, Enum):
    ADDITIONAL_KWARGS = "additional_kwargs"
    MESSAGE_INDEX = "message_index"
