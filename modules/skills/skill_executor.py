# author hgh
# version 1.0
"""
skill execute engine
responsible for executing steps in order according to skill config,parsing param placeholder,calling tool executor,rendering output templates
"""
import logging
import time
from typing import Dict, Any,Union

from jinja2 import Template, Undefined

from config.models.skill_config import SkillConfig, SkillArg
from modules.tools import ToolExecutor, ToolResult
from modules.tools.base_tool import ToolErrorType
from utils.monitor_utils.metrics import skill_execution_total, skill_execution_duration_seconds

logger = logging.getLogger(__name__)


class SilentUndefined(Undefined):
    def __fail_with_undefined_error(self, *args, **kwargs):
        return None


class SkillExecutor:
    """skill executor"""

    def __init__(self, tool_executor: ToolExecutor):
        self.tool_executor = tool_executor

    def execute(
            self,
            skill: SkillConfig,
            input_data: Dict[str, Any],
            trace_id: str = "",
            caller_agent: str = ""
    ) -> Dict[str, Any]:
        """
        execute skill

        Args:
            skill: skill config model
            input_data: the parameters passed by the LLM correspond to Skill.input_schema
            trace_id: trace id
            caller_agent: agent name

        Returns:
            result,contain fields such as success/data/error etc.
        """
        # store execute state
        start_time = time.monotonic()
        logger.info(
            "Start process Skill '%s' (version=%s), trace_id=%s, input=%s",
            skill.name, skill.version, trace_id, input_data
        )

        for param_def in skill.input_schema:
            value = input_data.get(param_def.name)
            if value is None:
                if param_def.required:
                    logger.warning("Skill '%s' 缺少必填参数: %s", skill.name, param_def.name)
                    return {
                        "success": False,
                        "error": f"缺少必填参数: {param_def.name}",
                        "error_type": ToolErrorType.PARAMETER_ERROR.value
                    }
                continue
            if param_def.validation:
                v = param_def.validation
                if v.ge is not None and value < v.ge:
                    return {
                        "success": False,
                        "error": f"参数 {param_def.name} 需要 >= {v.ge}，当前值: {value}",
                        "error_type": ToolErrorType.PARAMETER_ERROR.value
                    }
                if v.le is not None and value > v.le:
                    return {
                        "success": False,
                        "error": f"参数 {param_def.name} 需要 <= {v.le}，当前值: {value}",
                        "error_type": ToolErrorType.PARAMETER_ERROR.value
                    }
                if v.gt is not None and value <= v.gt:
                    return {
                        "success": False,
                        "error": f"参数 {param_def.name} 需要 > {v.gt}，当前值: {value}",
                        "error_type": ToolErrorType.PARAMETER_ERROR.value
                    }
                if v.lt is not None and value >= v.lt:
                    return {
                        "success": False,
                        "error": f"参数 {param_def.name} 需要 < {v.lt}，当前值: {value}",
                        "error_type": ToolErrorType.PARAMETER_ERROR.value
                    }

        state = {"input": input_data}

        for step in skill.steps:
            state[step.output_key] = None

        # excute skill steps
        for step in skill.steps:
            logger.debug("Skill '%s' strat processing step '%s'", skill.name, step.name)
            try:
                # parsing step args
                resolved_args = self._resolve_args(step.args, state)
            except Exception as e:
                logger.error("failed to parsing param (step=%s): %s", step.name, e)

                if step.optional:
                    state[step.output_key] = step.fallback_value if hasattr(step, 'fallback_value') else None
                    continue

                skill_execution_total.labels(
                    skill_name=skill.name, status="error"
                ).inc()

                return {
                    "success": False,
                    "error": f"步骤 '{step.name}' 参数解析失败: {e}",
                    "error_type": ToolErrorType.PARAMETER_ERROR.value,
                    "failed_step": step.name
                }

            # call atomic tool
            logger.info("Skill '%s' execute step '%s' calling tool '%s'", skill.name, step.name, step.tool)
            result: ToolResult = self.tool_executor.execute(
                tool_name=step.tool,
                args=resolved_args,
                caller_agent=caller_agent,
                trace_id=trace_id
            )

            # handling result
            if result.success:
                logger.debug("Step '%s' process successfully", step.name)
                state[step.output_key] = result.data
                continue

            logger.warning("步骤 '%s' 执行失败: error_type=%s, error=%s", step.name, result.error_type, result.error)
            skill_execution_total.labels(
                skill_name=skill.name, status="error"
            ).inc()
            on_failure = step.on_failure if hasattr(step, 'on_failure') else "abort"
            if on_failure == "skip":
                state[step.output_key] = step.fallback_value if hasattr(step, 'fallback_value') else None
                continue
            elif on_failure == "use_fallback":
                state[step.output_key] = step.fallback_value if hasattr(step, 'fallback_value') else None
                continue
            elif on_failure == "abort":
                return {
                    "success": False,
                    "error": result.error,
                    "error_type": result.error_type.value if result.error_type else ToolErrorType.EXTERNAL_ERROR.value,
                    "failed_step": step.name
                }

        # render output template
        try:
            output = self._render_template(skill.output_template, state)
        except Exception as e:
            logger.error("Failed to render output template: %s", e)
            skill_execution_total.labels(
                skill_name=skill.name, status="error"
            ).inc()
            return {
                "success": False,
                "error": f"输出模板渲染失败: {e}",
                "error_type": ToolErrorType.EXTERNAL_ERROR.value
            }

        duration = time.monotonic() - start_time
        skill_execution_total.labels(
            skill_name=skill.name, status="success"
        ).inc()
        skill_execution_duration_seconds.labels(skill_name=skill.name).observe(duration)
        logger.info(
            "Skill '%s' process successfully, cost time %.3f second", skill.name, duration
        )

        return {
            "success": True,
            "data": output,
            "raw_state": state
        }

    def _build_render_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        render_ctx = {"input": state.get("input", {})}
        for k, v in state.items():
            if k != "input" and not k.startswith("_"):
                render_ctx[k] = v
        return render_ctx

    def _resolve_args(self, args: Dict[str, Union[str, SkillArg]], state: Dict[str, Any]) -> Dict[str, Any]:
        """parsing param placeholder and replace them with actual value"""
        resolved = {}
        render_ctx = self._build_render_context(state)

        for key, arg_def in args.items():
            if isinstance(arg_def, SkillArg):
                template_str = arg_def.value
                expected_type = arg_def.type
            else:
                template_str = arg_def
                expected_type = "string"

            try:
                rendered = self._render_single_template(template_str, render_ctx)
            except Exception as e:
                logger.error("Param %s failed to render: %s", key, e)
                raise

            try:
                resolved[key] = self._cast_value(rendered, expected_type)
            except Exception as e:
                logger.error(
                    "Param %s failed cast type (type=%s, value=%s): %s",
                    key, expected_type, rendered, e
                )
                resolved[key] = rendered
        resolved = {k: v for k, v in resolved.items() if v is not None}
        return resolved

    def _render_single_template(self, template_str: str, render_ctx: Dict[str, Any]) -> str:
        if not isinstance(template_str, str):
            return str(template_str)
        template = Template(template_str, undefined=SilentUndefined)
        result = template.render(**render_ctx)
        # 如果渲染结果是 None，返回空字符串以便后续过滤
        return result if result is not None else ""

    def _cast_value(self, value: str, target_type: str) -> Any:
        """根据目标类型转换渲染后的字符串值"""
        if value.strip() == "" or value.strip().lower() == "null":
            return None
        if target_type == "string":
            return value.strip()
        if target_type == "integer":
            return int(value.strip())
        if target_type == "float":
            return float(value.strip())
        if target_type == "boolean":
            cleaned = value.strip().lower()
            if cleaned in ("true", "1", "yes"):
                return True
            elif cleaned in ("false", "0", "no"):
                return False
            else:
                raise ValueError(f"无法转换为布尔值: {value}")
        if target_type == "json":
            import json
            return json.loads(value.strip())
        # 默认原样返回
        return value

    def _render_template(self, template_str: str, state: Dict[str, Any]) -> str:
        if not template_str:
            return ""
        template = Template(template_str)
        render_ctx = self._build_render_context(state)
        return template.render(**render_ctx)
