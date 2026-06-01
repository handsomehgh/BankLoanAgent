# author hgh
# version 1.0
"""
multi_agent configuration model
define the config structure of each specialized agent
"""
from typing import List, Dict, Any

from pydantic import BaseModel, Field


class ToolDependency(BaseModel):
    name: str = Field(..., description="工具名称")
    version_range: str = Field(">=1.0.0", description="语义化版本范围，如 >=1.0.0,<2.0.0")


class RoutingRule(BaseModel):
    """单条静态路由规则"""
    pattern: str = Field(..., description="正则表达式")
    target: str = Field(..., description="路由目标: 'direct' 或 Agent 名称")


class SupervisorConfig(BaseModel):
    system_prompt: str = Field(
        default="",
        description="Supervisor 系统提示词，包含路由规则和各Agent能力描述"
    )
    enable_compliance_llm_fallback: bool = Field(
        default=True,
        description="是否启用合规 LLM 二审（当正则无命中时）"
    )
    agent_time_out: int = Field(
        default=60,
        description="并行调用subgraph超时时间"
    )
    enable_directed_retrieval: bool = Field(True, description="是否启用定向知识检索")
    directed_retrieval_max_length: int = Field(800, description="定向检索结果的最大字符数")


class DirectReplyConfig(BaseModel):
    fall_back_res: str = Field(..., description="降级回复")
    system_prompt: str = Field(
        default="",
        description="Direct reply 系统提示词"
    )


class AgentConfig(BaseModel):
    system_prompt: str = Field(default="", description="Agent 系统提示词")
    execute_prompt: str
    res_prompt: str
    direct_prompt: str
    clarify_prompt: str

    use_bert_classifier: bool


class LoanAdvisorConfig(AgentConfig):
    """贷款咨询 Agent 配置"""
    tool_exposure: str = Field(default="skills", description="工具选择策略")


class RiskAssessmentConfig(AgentConfig):
    """风险评估 Agent 配置"""
    tool_exposure: str = Field(default="skills", description="工具选择策略")


class AfterLoanConfig(AgentConfig):
    """贷后管理 Agent 配置"""
    tool_exposure: str = Field(default="skills", description="工具选择策略")

class CircuitBreakerConfig(BaseModel):
    enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout_sec: int = 60

class AgentExecutorConfig(BaseModel):
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    fallback_messages: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    tool_fallbacks: Dict[str, Any] = Field(default_factory=dict)