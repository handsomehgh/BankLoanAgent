# author hgh
# version 1.0
"""
tool selector
supports filter tools based on exposure mode and capability tags,keeping baseline tools
"""
import logging
from typing import Optional, List

from langchain_core.tools import BaseTool

from modules.agent.constants import ToolMode
from modules.tools import ToolRegistry

logger = logging.getLogger(__name__)


class ToolSelector:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def get_tools(
            self,
            agent_name: str,
            exposure_mode: str = "skills",
            capability_tags: Optional[List[str]] = None
    ) -> List[BaseTool]:
        """
        Get the list of tools currently available to the Agent

        Args:
            agent_name: Agent name (如 "LoanAdvisor")
            exposure_mode: exposure mode，optional "skills" or "hybrid"
            capability_tags: capability list，used for filter Skills,don't filter if is none。

        Returns:
            tool list
        """
        #1. get all tools that the Agent has permissions for
        all_tools = self._get_allowed_tools(agent_name)
        if exposure_mode == ToolMode.HYBRID.value:
            return all_tools

        #2. skill mode: by default, only Skills and baseline tools are retained
        selected_tools = []
        for tool in all_tools:
            meta = tool.extras or {}
            if meta.get("baseline",False):
                selected_tools.append(tool)
                continue

            caps = meta.get("capability_tags")
            if caps:
                if capability_tags is None:
                    selected_tools.append(tool)

                else:
                    if any(tag in caps for tag in capability_tags):
                        selected_tools.append(tool)

        return selected_tools

    def _get_allowed_tools(self, agent_name: str) -> List[BaseTool]:
        allowed_tools = []
        for tool_name,versions in self.registry._tools.items():
            latest_version = sorted(versions.keys())[-1]
            tool = versions[latest_version]
            tags = tool.extras.get("tags", []) if tool.extras else []
            if not tags or agent_name in tags:
                allowed_tools.append(tool)
        return allowed_tools




