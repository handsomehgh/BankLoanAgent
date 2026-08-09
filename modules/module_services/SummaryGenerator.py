# author hgh
# version 1.1
import logging
from typing import List, Optional

from langchain_core.messages import HumanMessage, BaseMessage

from config.prompt_hub import PromptHub
from modules.module_services.chat_models import RobustLLM

logger = logging.getLogger(__name__)


class SummaryGenerator:
    """Dialogue Summary Generator"""

    def __init__(
            self,
            llm_client: RobustLLM,
            prompt_hub: PromptHub,
            prompt_name: str,
            max_summary_length: int = 200,
            max_interaction_length: int  = 1000
    ):
        """
        Args:
            llm_client: LLM client used for generating summaries (usually with low temperature precise)
            prompt_hub: 提示词统一入口，渲染时实时取库（支持热更新）
            prompt_name: 提示词库条目名（含 {conversation} 占位符）
            max_summary_length: Maximum summary length (number of characters), used for post-processing truncation
            max_interaction_length: Maximum interaction length
        """
        self.llm_client = llm_client
        self.prompt_hub = prompt_hub
        self.prompt_name = prompt_name
        self.max_summary_length = max_summary_length
        self.max_interaction_length = max_interaction_length

    def generate(self, conversation: str,context: Optional[List[BaseMessage]]) -> str:
        """Generate a one-sentence summary based on the content of the conversation"""
        try:
            messages = self.prompt_hub.render_messages(
                self.prompt_name,
                conversation=conversation[:self.max_interaction_length],
                max_chars=self.max_summary_length
            )

            summary = self.llm_client.invoke(messages).content.strip()
            return summary
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            if context:
                user_parts = [m.content for m in context if isinstance(m, HumanMessage)]
                return f"用户询问：{'；'.join(user_parts[:2])}" if user_parts else "对话摘要生成失败"
            else:
                return conversation[:self.max_interaction_length]