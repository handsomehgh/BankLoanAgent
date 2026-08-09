# author hgh
# version 1.0
"""
PromptHub：提示词唯一访问入口。
- 每次调用都从 ConfigRegistry 实时取配置，天然支持 yaml 热更新
- 渲染时记录"提示词名+版本"归因日志，线上问题可回溯到具体提示词版本
- text 条目走 str.format；system(+human) 条目走 ChatPromptTemplate
"""
import logging
from typing import Dict, Any, List

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from config.global_constant.constants import RegistryModules
from config.models.prompt_library import PromptEntry
from config.registry import ConfigRegistry

logger = logging.getLogger(__name__)

_LIBRARY_MODULES = (
    RegistryModules.PROMPTS_RETRIEVAL,
    RegistryModules.PROMPTS_MEMORY,
    RegistryModules.PROMPTS_AGENT,
)


class PromptHub:
    """提示词统一入口，薄封装 ConfigRegistry，无自身缓存"""

    def __init__(self, registry: ConfigRegistry):
        self.registry = registry

    def _find(self, name: str) -> PromptEntry:
        for module in _LIBRARY_MODULES:
            library = self.registry.get_config(module)
            entry = library.prompts.get(name)
            if entry is not None:
                return entry
        raise KeyError(f"提示词 [{name}] 在三个提示词库中均不存在")

    def version_of(self, name: str) -> str:
        return self._find(name).version

    def get_text(self, name: str) -> str:
        """取无占位符的纯文本提示词（如 supervisor_router）"""
        entry = self._find(name)
        if entry.text is None:
            raise ValueError(f"提示词 [{name}] 不是 text 形态，请用 render_messages/as_chat_template")
        logger.info("[PromptHub] get prompt=%s version=%s", name, entry.version)
        return entry.text

    def render_text(self, name: str, **variables: Any) -> str:
        """渲染 text 形态条目（str.format 规则，字面大括号需 {{}} 转义）"""
        entry = self._find(name)
        if entry.text is None:
            raise ValueError(f"提示词 [{name}] 不是 text 形态，请用 render_messages/as_chat_template")
        rendered = entry.text.format(**variables)
        logger.info("[PromptHub] render prompt=%s version=%s vars=%s", name, entry.version, sorted(variables))
        return rendered

    def as_chat_template(self, name: str) -> ChatPromptTemplate:
        """把 system(+human) 条目构建成 ChatPromptTemplate（langchain f-string 规则）"""
        entry = self._find(name)
        if entry.text is not None:
            raise ValueError(f"提示词 [{name}] 是 text 形态，请用 render_text")
        messages = [("system", entry.system)]
        if entry.human:
            messages.append(("human", entry.human))
        logger.info("[PromptHub] build template prompt=%s version=%s", name, entry.version)
        return ChatPromptTemplate.from_messages(messages)

    def render_messages(self, name: str, **variables: Any) -> List[BaseMessage]:
        """渲染 system(+human) 条目为可直接送入 LLM 的消息列表"""
        return self.as_chat_template(name).invoke(variables).to_messages()
