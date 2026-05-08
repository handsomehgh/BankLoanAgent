# author hgh
# version 1.0
import logging
from typing import List

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from modules.module_services.chat_models import RobustLLM

logger = logging.getLogger(__name__)


class SummaryGenerator:
    """Dialogue Summary Generator"""

    def __init__(
            self,
            llm_client: RobustLLM,
            prompt: ChatPromptTemplate,
            max_summary_length: int = 200,
            max_interaction_length: int  = 1000
    ):
        """
        Args:
            llm_client: LLM client used for generating summaries (usually with low temperature precise)
            prompt: Summary prompt template, including the {conversation} placeholder
            max_summary_length: Maximum summary length (number of characters), used for post-processing truncation
            max_interaction_length: Maximum interaction length
        """
        self.llm_client = llm_client
        self.prompt_template = prompt
        self.max_summary_length = max_summary_length
        self.max_interaction_length = max_interaction_length

    def generate(self, conversation: str,context: List[BaseMessage]) -> str:
        """Generate a one-sentence summary based on the content of the conversation"""
        if not conversation:
            return "对话内容为空"
        try:
            messages = self.prompt_template.invoke({
                "conversation": conversation[:self.max_interaction_length],
                "max_chars": self.max_summary_length
            }).to_messages()

            summary = self.llm_client.invoke(messages).content.strip()
            return summary
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            user_parts = [m.content for m in context if isinstance(m, HumanMessage)]
            return f"用户询问：{'；'.join(user_parts[:2])}" if user_parts else "对话摘要生成失败"