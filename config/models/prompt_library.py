# author hgh
# version 1.0
"""
提示词库配置模型：三个领域库（prompts_retrieval/prompts_memory/prompts_agent）共用。
每条提示词带独立版本号，PromptHub 渲染时记录版本归因日志。
"""
from typing import Dict, Optional

from pydantic import BaseModel, Field, model_validator


class PromptEntry(BaseModel):
    """单条提示词条目。
    两种形态二选一：
    - text：单段文本（配合 str.format 渲染，如 system_prompt、各 agent 角色提示词）
    - system + human：对话式模板（配合 ChatPromptTemplate 渲染，human 可省略）
    """
    version: str = Field(default="1.0", description="提示词版本号，内容变更时手动递增")
    description: str = Field(default="", description="用途说明")
    text: Optional[str] = Field(default=None, description="单段文本模板")
    system: Optional[str] = Field(default=None, description="对话式模板的 system 段")
    human: Optional[str] = Field(default=None, description="对话式模板的 human 段")

    @model_validator(mode="after")
    def check_shape(self):
        has_text = self.text is not None
        has_chat = self.system is not None or self.human is not None
        if has_text == has_chat:
            raise ValueError(
                "prompt entry 必须且只能选择一种形态：text 或 system(+human)"
            )
        return self


class PromptLibrary(BaseModel):
    """一个领域的提示词库"""
    version: str = Field(default="1.0", description="库整体版本")
    prompts: Dict[str, PromptEntry] = Field(default_factory=dict)
