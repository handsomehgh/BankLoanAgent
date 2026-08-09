# author hgh
# version 1.0
"""
multi-agent system state and context definition
includes the state of supervisor and each sub-agent,unified agentContext and agentResponse
"""
from typing import List, Optional, Any, Dict, TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field

#=================== agent context ==================
class AgentContext(BaseModel):
    """
    unified context passed from supervisor to child agent
    """
    user_id: str = Field(...,description="unique id of the user")
    session_id: str = Field(...,description="unique id of the session")
    trace_id: str = Field(...,description="full link trace id")
    current_query: str = Field(...,description="current user question")
    compliance_warnings: List[str] = Field(default_factory=list,description="compliance warning list")
    retrieved_knowledge: Optional[str] = Field(None,description="content of retrieved knowledge")
    agent_instruction: str = Field("",description="instructions from the supervisor to the agent")
    audit_logger: Any = Field(default=None,description="audit log recoder"),
    sub_conversation: str = Field("暂无相关信息",description="audit log recoder")
    conversation_summary: str = Field("暂无相关信息", description="summary of recent N rounds of conversation")
    recent_conversation: Optional[str] = Field(None, description="从上次日志记录点到当前的最新对话原文")
    user_profile_summary: str = Field("暂无相关信息", description="desensitized profile of user")
    @classmethod
    def from_state(
            cls,
            user_id: str,
            session_id: str,
            trace_id: str,
            current_query: str,
            user_profile_summary: str,
            compliance_warnings: List[str],
            conversation_summary: str,
            retrieved_knowledge: Optional[str],
            agent_instruction: str,
            audit_logger: Any,
            sub_conversation: str,
            recent_conversation: Optional[str]
    ) -> "AgentContext":
        return cls(
            user_id=user_id,
            session_id=session_id,
            trace_id=trace_id,
            current_query=current_query,
            user_profile_summary=user_profile_summary,
            compliance_warnings=compliance_warnings,
            conversation_summary=conversation_summary,
            retrieved_knowledge=retrieved_knowledge,
            agent_instruction=agent_instruction,
            audit_logger=audit_logger,
            sub_conversation=sub_conversation,
            recent_conversation=recent_conversation
        )

    class Config:
        arbitrary_types_allowed = True

#==================== agent response ======================
class AgentResponse(BaseModel):
    """standard response returned by sub agent"""
    content: str = Field(...,description="content of the response")
    metadata: Dict[str,Any] = Field(default_factory=dict,description="metadata of the response")

#=================== supervisor state =======================
def _merge_audit_log(old: Dict[str, List[BaseMessage]], new: Dict[str, List[BaseMessage]]) -> Dict[str, List[BaseMessage]]:
    """合并两个审计日志字典，新值覆盖旧值"""
    return {**old, **new}

class SupervisorState(TypedDict):
    """the state of the supervisor"""
    messages: Annotated[List[BaseMessage],add_messages]
    user_id: str
    trace_id: str
    #cursor
    last_extracted_message_index: Optional[int]
    last_logged_message_index: Optional[int]
    #context
    retrieved_context: Dict[str,Any]
    formatted_context: Dict[str,Any]
    #result:
    profile_updated: bool
    error: Optional[str]
    compliance_blocked: bool
    compliance_warnings: List[str]
    mandatory_appends: List[str]
    #specialized fields
    next_agents: Optional[List[str]]
    agent_context: Optional[Dict[str,AgentContext]]
    agent_responses: Dict[str, AgentResponse]
    #human handoff summary
    handoff_summary: str
    trigger_human_handoff: bool
    #intermnal messages
    sub_messages: Annotated[Dict[str, List[BaseMessage]], _merge_audit_log]
    #skip supervisor
    should_skip_supervisor: bool

#================== sub agent state =======================
class LoanAdvisorState(TypedDict):
    agent_context: AgentContext
    internal_messages: Annotated[List[BaseMessage],add_messages]
    tool_results: List[Dict[str,Any]]
    final_response: Optional[AgentResponse]
    # handoff from decision node to reply node,see modules.agent.constants.ReplyStage
    reply_stage: Optional[str]
    reply_payload: Optional[Dict[str, Any]]

class RiskAssessmentState(TypedDict):
    agent_context: AgentContext
    internal_messages: Annotated[List[BaseMessage], add_messages]
    tool_results: List[Dict[str, Any]]
    risk_level: Optional[str]
    trigger_human_handoff: bool
    final_response: Optional[AgentResponse]
    reply_stage: Optional[str]
    reply_payload: Optional[Dict[str, Any]]

class AfterLoanState(TypedDict):
    agent_context: AgentContext
    internal_messages: Annotated[List[BaseMessage], add_messages]
    tool_results: List[Dict[str, Any]]
    final_response: Optional[AgentResponse]
    reply_stage: Optional[str]
    reply_payload: Optional[Dict[str, Any]]