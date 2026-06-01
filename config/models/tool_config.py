# author hgh
# version 1.0
from typing import Optional, List

from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    name: str = Field(..., description="工具唯一名称")
    version: Optional[str] = Field(None, description="语义化版本，可选")
    allowed_agents: List[str] = Field(default_factory=list, description="允许调用的Agent列表，可选")

class ToolRegistryConfig(BaseModel):
    tools: List[ToolDefinition] = Field(
        default_factory=list,
        description="已注册的工具清单"
    )

