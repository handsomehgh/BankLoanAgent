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

    full_description = skill_config.description
    if skill_config.sop_description:
        full_description = f"{skill_config.description}\n\n执行流程：{skill_config.sop_description}"

    @tool(
        skill_config.name,
        description=full_description,
        args_schema=DynamicInputModel,
        extras={
            "version": skill_config.version,
            "tags": skill_config.tags,
            "baseline": skill_config.baseline,
            "capability_tags": skill_config.capability_tags
        }
    )
    def skill_func(input: DynamicInputModel) -> dict:
       pass
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

