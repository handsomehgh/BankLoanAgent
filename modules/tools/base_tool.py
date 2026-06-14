# author hgh
# version 1.0
"""
tool system abstract base class
define standard interfaces for tools and tool outputs
"""
import json
import logging
import sys
import time
from typing import Any, Optional, Dict

from packaging.version import Version
from langchain_core.tools import BaseTool as LangChainBaseTool

from config.models.tool_config import ToolRegistryConfig
from exceptions.exception import ToolExecutionException, ToolErrorType
from infra.database.mysql_manager import DatabaseManager
from infra.repository.LoanInterestRepository import LoanInterestRepository
from modules.tools.tool_constatnt import ToolResult
from utils.monitor_utils.metrics import tool_call_total, tool_duration_seconds

logger = logging.getLogger(__name__)


# ================= Tool Registration Assistance ======================
class ToolRegistry:
    """
    tool registry center
    At startup, scan all @tool functions in the tools/ directory and register them in the internal dictionary.
    At the same time, validate the tool list from the YAML configuration to ensure consistency between the code and the configuration (Fast-Fail).
    """

    def __init__(self, config: ToolRegistryConfig):
        self._tools: Dict[str, Dict[str, LangChainBaseTool]] = {}
        self._config = config
        self._scanned = False

    def register(self, tool: LangChainBaseTool):
        """register a tool"""
        name = tool.name
        extras = getattr(tool, 'extras', {}) or {}
        version = extras.get("version", "0.0.0")
        tags = extras.get("tags", [])

        if name not in self._tools:
            self._tools[name] = {}
        if version in self._tools[name]:
            logger.warning("Tool %s v%s already exists，will be covered", name, version)
        self._tools[name][version] = tool
        logger.debug("Tool has been registered: %s v%s (agents=%s)", name, version, tags)

    def scan_and_register(self, tools_path: Optional[str] = None) -> None:
        import importlib
        import pkgutil
        from pathlib import Path

        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir.parent))

        for _, module_name, _ in pkgutil.walk_packages(
                [str(current_dir)], prefix="tools."
        ):
            if module_name in ("tools.base_tool", "tools.common_utils"):
                continue
            try:
                module = importlib.import_module(module_name)
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, LangChainBaseTool):
                        if not hasattr(attr, 'args_schema') or attr.args_schema is None:
                            logger.warning("工具 %s 缺少有效的 args_schema，跳过注册", attr.name)
                            continue
                        self.register(attr)
            except Exception as e:
                logger.error("Scanning module %s Failed: %s", module_name, e)
        self._scanned = True

    def validate_against_config(self) -> None:
        """Verify whether the YAML tool list is fully registered in the code"""
        if not self._scanned:
            raise RuntimeError("Please execute scan_and_register() first before verification")
        for tool_entry in self._config.tools:
            name = tool_entry.name
            if name not in self._tools:
                raise RuntimeError(f"Tool '{name}' doesn't register in code，Please check tools content")
        logger.info("Tool configuration check passed, a total of %d tools", len(self._config.tools))

    def get_tool(self, name: str, version_range: str, caller_agent: str) -> Optional[LangChainBaseTool]:
        """Obtain a tool instance based on name, version range, and caller permissions"""
        if name not in self._tools:
            return None
        versions = self._tools[name]
        if not versions:
            return None
        sorted_versions = sorted(versions.keys(), key=lambda v: Version(v))
        latest_version = sorted_versions[-1]
        tool = versions[latest_version]
        if not self._check_permission(tool, caller_agent):
            return None
        return tool

    def _check_permission(self, tool: LangChainBaseTool, caller_agent: str) -> bool:
        allowed = (getattr(tool, 'extras', {}) or {}).get("tags", [])
        if not allowed:
            return True
        return caller_agent in allowed


class ToolExecutor:
    """
    tool executor
    Process: Permission check → Parameter validation → Audit log (before and after) → Execution → Exception handling
    """

    def __init__(self, registry: ToolRegistry, audit_logger=None, db_manager: DatabaseManager=None):
        self.registry = registry
        self.audit_logger = audit_logger
        self.db_manager = db_manager

    def execute(
            self,
            tool_name: str,
            args: Dict[str, Any],
            caller_agent: str,
            trace_id: str,
            version_range: str = ">=1.0.0",
            **runtime_context
    ) -> ToolResult:
        # 1. permission validate + obtain tool
        session = self.db_manager.create_session()
        tool = self.registry.get_tool(tool_name, version_range, caller_agent)
        if tool is None:
            msg = f"Tool {tool_name} is unavailable or insufficient permissions"
            logger.warning(msg)
            self._audit(trace_id, tool_name, None, caller_agent, args, success=False, error=msg)
            return ToolResult(success=False, error=msg, error_type=ToolErrorType.BUSINESS_ERROR)

        extras = getattr(tool, 'extras', {}) or {}
        tool_version = extras.get("version", "0.0.0")

        # 2. parameter validation
        logger.info(f"Tool runtime context: {runtime_context}")
        try:
            validated = tool.args_schema(**args) if tool.args_schema else args
            injected_base = self._build_injected_base(session, runtime_context, trace_id)
            logger.info(f"Tool injected_base: {injected_base}")
            injected = getattr(tool, '_injected_kwargs', {}).copy()
            logger.info(f"Tool injected: {getattr(tool, '_injected_params', [])}")
            for param_name in getattr(tool, '_injected_params', []):
                if param_name == 'trace_id':
                    injected[param_name] = trace_id
                elif param_name in injected_base:
                    injected[param_name] = injected_base[param_name]
        except Exception as e:
            logger.warning(f"工具 {tool_name} 参数校验失败: {e}")
            return ToolResult(
                success=False,
                error=f"参数校验失败: {e}",
                error_type=ToolErrorType.PARAMETER_ERROR
            )
        logger.info(f"Tool injected parameters: {injected}")

        max_retries = 2
        retry_delay = 0.5
        last_result = None
        # 4. execute tool
        try:
            # 4. execute tool
            start = time.monotonic()
            for attempt in range(max_retries + 1):
                logger.info(f"Start calling tool: {tool_name}, args: {validated}")
                try:
                    result = tool.func(validated, **injected)
                    if isinstance(result, str):
                        final_data = result
                    elif isinstance(result, dict):
                        final_data = result
                    else:
                        final_data = str(result)
                    duration = time.monotonic() - start
                    self._record_metrics(tool_name, caller_agent, success=True, duration=duration)
                    return ToolResult(
                        success=True,
                        data=final_data,
                        summary=self._build_summary(tool_name, final_data)
                    )
                except ToolExecutionException as e:
                    last_result = ToolResult(success=False, error=str(e), error_type=e.error_type)
                    if e.error_type != ToolErrorType.TEMPORARY_ERROR:
                        break
                    logger.warning(f"工具 {tool_name} 临时性故障 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
                    if session:
                        session.rollback()
                except Exception as e:
                    logger.exception(f"工具 {tool_name} 未捕获异常")
                    last_result = ToolResult(success=False, error=str(e), error_type=ToolErrorType.EXTERNAL_ERROR)
                    break

                if attempt < max_retries:
                    time.sleep(retry_delay * (2 ** attempt))

            # 5. monitor metrics record
            duration = time.monotonic() - start
            self._record_metrics(tool_name, caller_agent, success=False, duration=duration)
            return last_result if last_result else ToolResult(
                success=False,
                error="未知错误",
                error_type=ToolErrorType.EXTERNAL_ERROR
            )
        except Exception as e:
            if session:
                session.rollback()
        finally:
            # 确保会话正确关闭，释放连接回连接池
            if session:
                session.close()

    def _build_injected_base(self, session, context: dict, trace_id: str) -> dict:
        return {
            "user_id": context.get("user_id", ""),
            "trace_id": trace_id,
            "conversation_summary": context.get("conversation_summary", ""),
            "profile_summary": context.get("profile_summary", ""),
            "repository": LoanInterestRepository(session),
        }

    def _record_metrics(self, tool_name, caller_agent, success, duration):
        status = "success" if success else "error"
        tool_call_total.labels(tool_name=tool_name, status=status, caller_agent=caller_agent).inc()
        tool_duration_seconds.labels(caller_agent=caller_agent, tool_name=tool_name).observe(duration)

    def _audit(self, trace_id, tool_name, tool_version, caller_agent, args, success,
               error=None, result=None, pre_call=False):
        if self.audit_logger:
            try:
                self.audit_logger.record(
                    event_type="tool_call",
                    trace_id=trace_id,
                    tool_name=tool_name,
                    tool_version=tool_version,
                    caller_agent=caller_agent,
                    args=json.dumps(args, ensure_ascii=False)[:200],
                    success=success,
                    error=error,
                    result=result,
                )
            except Exception as e:
                logger.warning("Audit log recording failed: %s", e)

    @staticmethod
    def _build_summary(tool_name: str, data: Any) -> str:
        if isinstance(data, dict):
            items = list(data.items())[:3]
            return ", ".join(f"{k}={v}" for k, v in items)
        return str(data)[:100]
