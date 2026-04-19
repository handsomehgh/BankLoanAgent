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
    eval_score: Optional[float]
    eval_feed_back: Optional[str]
    needs_rewrite: bool
    profile_updated: bool
    error: Optional[str]

