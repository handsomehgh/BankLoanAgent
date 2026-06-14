# author hgh
# version 1.0
"""
skill execute engine
responsible for executing steps in order according to skill config,parsing param placeholder,calling tool executor,rendering output templates
"""
import logging
import time
from typing import Dict, Any, Union

from jinja2 import Template, Undefined
from config.models.skill_config import SkillConfig, SkillArg, CompensateConfig
from modules.tools import ToolRegistry
from modules.tools.base_tool import ToolErrorType, ToolExecutionException
from utils.monitor_utils.metrics import skill_execution_total, skill_execution_duration_seconds

logger = logging.getLogger(__name__)


class SilentUndefined(Undefined):
    def __fail_with_undefined_error(self, *args, **kwargs):
        return None


class SkillExecutor:
    """技能执行器（重构版）"""

    def __init__(self, registry: ToolRegistry, audit_logger=None):
        self.registry = registry
        self.audit_logger = audit_logger

    def execute(
            self,
            skill: SkillConfig,
            input_data: Dict[str, Any],
            trace_id: str = "",
            caller_agent: str = "",
            **context
    ) -> Dict[str, Any]:
        start_time = time.monotonic()
        logger.info("Skill '%s' (v%s) start, trace=%s", skill.name, skill.version, trace_id)

        # ===== 1. build base dict =====
        injected_base = self._build_injected_base(context, trace_id)

        # ===== 2. validate input =====
        self._validate_input(skill, input_data)

        state = {"input": input_data}
        for step in skill.steps:
            state[step.output_key] = None

        compensation_stack = []
        # ===== 3. execute step =====
        try:
            for step in skill.steps:
                logger.debug("Skill '%s' step '%s'", skill.name, step.name)

                # 3.1 validate step param
                try:
                    resolved_args = self._resolve_args(step.args, state)
                except Exception as e:
                    logger.error("步骤参数解析失败 step=%s: %s", step.name, e)
                    if step.optional:
                        state[step.output_key] = getattr(step, 'fallback_value', None)
                        continue
                    raise ToolExecutionException(
                        f"步骤 '{step.name}' 参数解析失败: {e}",
                        ToolErrorType.PARAMETER_ERROR
                    )

                # 3.2 get tool
                tool = self.registry.get_tool(step.tool, ">=1.0.0", caller_agent)
                if not tool:
                    raise ToolExecutionException(
                        f"工具 {step.tool} 未注册或无权调用",
                        ToolErrorType.BUSINESS_ERROR
                    )

                # 3.3 injected param
                injected = getattr(tool, '_injected_kwargs', {}).copy()
                for param_name in getattr(tool, '_injected_params', []):
                    if param_name == 'trace_id':
                        injected[param_name] = trace_id
                    elif param_name in injected_base:
                        injected[param_name] = injected_base[param_name]

                # 3.4 execute tools
                validated = tool.args_schema(**resolved_args) if tool.args_schema else resolved_args
                max_step_retries = 2
                for attempt in range(max_step_retries + 1):
                    try:
                        result = tool.func(validated, **injected)
                        if step.compensate:
                            compensation_stack.append(step.compensate)
                        state[step.output_key] = self._normalize_result(result)
                        break
                    except ToolExecutionException as e:
                        if e.error_type == ToolErrorType.TEMPORARY_ERROR and attempt < max_step_retries:
                            time.sleep(0.5 * (2 ** attempt))
                            continue
                        self._handle_failure(skill.name, step, state)
                        break
                    except Exception:
                        self._handle_failure(skill.name, step, state)
                        break

            # ===== 4. success/commit transaction =====
            duration = time.monotonic() - start_time
            logger.info("Skill '%s' success, time=%.3fs", skill.name, duration)
            skill_execution_total.labels(skill_name=skill.name, status="success").inc()
            skill_execution_duration_seconds.labels(skill_name=skill.name).observe(duration)
        except Exception as e:
            while compensation_stack:
                compensate_config = compensation_stack.pop()
                self._execute_compensate(trace_id, caller_agent, compensate_config, state, context)
            raise

        # ===== 5. render output =====
        try:
            output = self._render_template(skill.output_template, state)
            return {"success": True, "data": output, "raw_state": state}
        except Exception as e:
            raise ToolExecutionException(
                f"Skill '{skill.name}' 结果渲染失败,但业务操作已完成",
                ToolErrorType.EXTERNAL_ERROR
            )

    def _execute_compensate(self, trace_id: str, caller_agent: str, compensate_config: CompensateConfig, state: dict,
                            context):
        try:
            resolved_args = self._resolve_args(compensate_config.args, state)
            injected_base = self._build_injected_base(context, trace_id)
            tool = self.registry.get_tool(compensate_config.tool, ">=1.0.0", caller_agent)
            if not tool:
                raise ToolExecutionException(
                    f"工具 {compensate_config.tool} 未注册或无权调用",
                    ToolErrorType.BUSINESS_ERROR
                )

            injected = getattr(tool, '_injected_kwargs', {}).copy()
            for param_name in getattr(tool, '_injected_params', []):
                if param_name == 'trace_id':
                    injected[param_name] = trace_id
                elif param_name in injected_base:
                    injected[param_name] = injected_base[param_name]

            tool = self.registry.get_tool(compensate_config.tool, ">=1.0.0", context.get("caller_agent", ""))
            validated = tool.args_schema(**resolved_args) if tool.args_schema else resolved_args
            tool.func(validated, **context)
        except Exception as comp_e:
            logger.error(f"补偿 {compensate_config.tool} 失败: {comp_e}")
            self._save_compensation_task(compensate_config, resolved_args)

    def _save_compensation_task(self, compensate_config: CompensateConfig, resolved_args: Dict[str, Any]):
        pass

    def _build_injected_base(self, context: dict, trace_id: str) -> dict:
        return {
            "user_id": context.get("user_id", ""),
            "trace_id": trace_id,
            "conversation_summary": context.get("conversation_summary", ""),
            "profile_summary": context.get("profile_summary", "")
        }

    def _build_injected(self, tool, base_injected: dict) -> dict:
        needed = set(getattr(tool, '_injected_params', []))
        return {k: v for k, v in base_injected.items() if k in needed}

    def _handle_failure(self, skill_name: str, step, state: dict):
        on_failure = getattr(step, 'on_failure', 'abort')
        skill_execution_total.labels(skill_name=skill_name, status="error").inc()

        if on_failure == "skip":
            state[step.output_key] = getattr(step, 'fallback_value', None)
        elif on_failure == "use_fallback":
            state[step.output_key] = getattr(step, 'fallback_value', None)
        elif on_failure == "abort":
            raise ToolExecutionException(
                f"步骤 '{step.name}' 失败且策略为 abort",
                ToolErrorType.EXTERNAL_ERROR
            )

    def _validate_input(self, skill: SkillConfig, input_data: Dict[str, Any]):
        for param_def in skill.input_schema:
            value = input_data.get(param_def.name)
            if value is None:
                if param_def.required:
                    raise ToolExecutionException(
                        f"缺少必填参数: {param_def.name}",
                        ToolErrorType.PARAMETER_ERROR
                    )
                continue
            if param_def.validation:
                v = param_def.validation
                if v.ge is not None and value < v.ge:
                    raise ToolExecutionException(f"参数 {param_def.name} 需要 >= {v.ge}, 当前 {value}",
                                                 ToolErrorType.PARAMETER_ERROR)
                if v.le is not None and value > v.le:
                    raise ToolExecutionException(f"参数 {param_def.name} 需要 <= {v.le}, 当前 {value}",
                                                 ToolErrorType.PARAMETER_ERROR)
                if v.gt is not None and value <= v.gt:
                    raise ToolExecutionException(f"参数 {param_def.name} 需要 > {v.gt}, 当前 {value}",
                                                 ToolErrorType.PARAMETER_ERROR)
                if v.lt is not None and value >= v.lt:
                    raise ToolExecutionException(f"参数 {param_def.name} 需要 < {v.lt}, 当前 {value}",
                                                 ToolErrorType.PARAMETER_ERROR)

    def _normalize_result(self, result):
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            return result
        else:
            return str(result)

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
        return result if result is not None else ""

    def _cast_value(self, value: str, target_type: str) -> Any:
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
        return value

    def _render_template(self, template_str: str, state: Dict[str, Any]) -> str:
        if not template_str:
            return ""
        template = Template(template_str)
        render_ctx = self._build_render_context(state)
        return template.render(**render_ctx)
