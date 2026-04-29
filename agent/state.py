# author hgh
# version 1.0
from typing import TypedDict, Annotated, List, Dict, Any, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage],add_messages]
    user_id: str
    #global sequence number of the last processed message
    last_extracted_message_index: Optional[int]
    #global message sequence number counter,new messages take this value and then increment
    next_message_index: int
    retrieved_context: Dict[str, Any]
    formatted_context: Dict[str, Any]
    profile_updated: bool
    interaction_logged: bool
    error: Optional[str]
    compliance_blocked: bool
    compliance_warnings: List[str]
    mandatory_appends: List[str]
    should_skip_llm: bool

