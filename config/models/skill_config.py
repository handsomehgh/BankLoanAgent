# author hgh
# version 1.0
"""
skills config model
"""
from typing import Literal, Any, Dict, List, Union, Optional

from pydantic import BaseModel, Field

class SkillValidation(BaseModel):
    ge: Optional[float] = None   # 大于等于
    le: Optional[float] = None   # 小于等于
    gt: Optional[float] = None   # 大于
    lt: Optional[float] = None   # 小于

class SkillInputParam(BaseModel):
    """input parameter"""
    name: str = Field(...,description="param name")
    type: Literal["string","integer","float","boolean","json"] = Field(...,description="param type")
    description: str = Field("",description="param description")
    required: bool = Field(default=True,description="param required")
    default: Any = Field(None,description="default value")
    validation: Optional[SkillValidation] = None

class SkillArg(BaseModel):
    """step parameter definition"""
    value: str = Field(...,description="param value,support jinja2 template placeholder:{{input.loan_amount}}")
    type: Literal["string", "integer", "float", "boolean", "json"] = Field(
        default="string",
        description="期望的参数类型，用于渲染后的类型转换"
    )
    description: str = Field(default="", description="参数说明，便于维护")


class SkillStep(BaseModel):
    """skill process step definition"""
    name: str = Field(...,description="unique id of step")
    tool: str = Field(...,description="name of the called tool")
    args: Dict[str, Union[str, SkillArg]] = Field(
        default_factory=dict,
        description="参数映射，支持纯字符串（模板）或结构化 SkillArg"
    )
    output_key: str = Field(...,description="the storage key name of the output of this step")
    optional: bool = Field(default=False,description="whether it is an optional step,and whether to continue execution if it is fails")
    on_failure: Literal["skip","abort"] = Field(default="skip",description="handling strategy when optional steps fail")

class SkillConfig(BaseModel):
    """complete skill configuration definition"""
    name: str = Field(...,description="the unique name of skill")
    version: str = Field("1.0.0",description="version of skill")
    description: str = Field("",description="description of skill")
    capability_tags: List[str] = Field(default_factory=list,description="list of capability label,used to dynamically filter tools by intent")
    sop_description: str = Field("",description="standard operate process")
    tags: List[str] = Field(default_factory=list,description="list of agent names allowed to be called,an empty list means all agent are available")
    baseline: bool = Field(default=False,description="whether it is a baseline tool,always loaded")
    input_schema: List[SkillInputParam] = Field(
        default_factory=list,
        description="输入参数定义列表"
    )
    steps: List[SkillStep] = Field(default_factory=list,description="execution step list,execute in order")
    output_template: str = Field(default="",description="output template")

class SkillRegistryConfig(BaseModel):
    """main configuration"""
    skills: List[SkillConfig] = Field(default_factory=list, description="skills configuration list")


