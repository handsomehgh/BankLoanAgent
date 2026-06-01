# author hgh
# version 1.0
"""
skill tool factory
"""
import logging

from langchain_core.tools import tool
from pydantic import Field, create_model

from config.models.skill_config import SkillConfig
from modules.skills.skill_executor import SkillExecutor

logger = logging.getLogger(__name__)

def create_tool_from_skill(skill_config: SkillConfig,skill_executor: SkillExecutor):
    fields = {}
    for param in skill_config.input_schema:
        py_type = _map_type(param.type)
        if param.required:
            field = Field(description=param.description)
        else:
            field = Field(default=param.default, description=param.description)
        fields[param.name] = (py_type, field)

    DynamicInputModel = create_model(f"{skill_config.name}_input", **fields)

    @tool(
        skill_config.name,
        description=skill_config.description,
        args_schema=DynamicInputModel,
        extras={
            "version": skill_config.version,
            "tags": skill_config.tags,
            "baseline": skill_config.baseline,
            "capability_tags": skill_config.capability_tags
        }
    )
    def skill_func(input: DynamicInputModel) -> dict:
        result = skill_executor.execute(
            skill=skill_config,
            input_data=input.model_dump(),
            trace_id="",
            caller_agent=skill_config.tags[0]
        )
        if result["success"]:
            return result["data"]
        else:
            return f"Skill 执行失败: {result.get('error', '未知错误')}"
    return skill_func



def _map_type(type_str: str) -> type:
    mapping = {
        "string": str,
        "integer": int,
        "float": float,
        "boolean": bool,
        "json": str,
    }
    return mapping.get(type_str, str)

