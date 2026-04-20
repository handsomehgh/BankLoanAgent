# author hgh
# version 1.0
from typing import TypedDict, Annotated, List, Dict, Any, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage],add_messages]
    user_id: str
    retrieved_context: Dict[str, Any]
    formatted_context: Dict[str, Any]
    profile_updated: bool
    interaction_logged: bool
    error: Optional[str]
    compliance_blocked: bool  # 新增
    compliance_warnings: List[str]  # 新增
    mandatory_appends: List[str]  # 新增
    should_skip_llm: bool

