# author hgh
# version 1.0
"""
multi_agent configuration model
define the config structure of each specialized agent
"""
from typing import Dict, Any

from pydantic import BaseModel, Field


class ToolDependency(BaseModel):
    name: str = Field(..., description="工具名称")
    version_range: str = Field(">=1.0.0", description="语义化版本范围，如 >=1.0.0,<2.0.0")


class RoutingRule(BaseModel):
    """单条静态路由规则"""
    pattern: str = Field(..., description="正则表达式")
    target: str = Field(..., description="路由目标: 'direct' 或 Agent 名称")


class SupervisorConfig(BaseModel):
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


class AgentConfig(BaseModel):
    """worker agent 行为配置；所有提示词已迁移至 prompts_agent.yaml 提示词库"""
    use_bert_classifier: bool = Field(default=False, description="工具选择是否走 BERT 分类器")
    tool_exposure: str = Field(default="skills", description="工具选择策略")


class AgentsConfig(BaseModel):
    """全部 worker agent 的统一配置：default 全局默认 + overrides 按 agent 名差异化覆盖"""
    default: AgentConfig = Field(default_factory=AgentConfig)
    overrides: Dict[str, AgentConfig] = Field(
        default_factory=dict,
        description="按 agent 模块键（loan_advisor/risk_assessment/after_loan）覆盖默认值"
    )

    def resolve(self, agent_key: str) -> AgentConfig:
        """override 优先，缺省回落 default；override 只覆盖其显式声明的字段"""
        override = self.overrides.get(agent_key)
        if not override:
            return self.default
        merged = self.default.model_dump()
        merged.update(override.model_dump(exclude_unset=True))
        return AgentConfig(**merged)


class CircuitBreakerConfig(BaseModel):
    enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout_sec: int = 60

class AgentExecutorConfig(BaseModel):
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    fallback_messages: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    tool_fallbacks: Dict[str, Any] = Field(default_factory=dict)
    response_handlers: Dict[str,Any] = Field(default_factory=dict)