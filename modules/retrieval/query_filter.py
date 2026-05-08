# author hgh
# version 1.0
"""
self_query filter
extract metadata filter expression from the user query,return milvus expr string
"""
import json
import logging
from typing import Optional

from config.models.retrieval_config import FilterConfig
from config.prompts.extract_filter_prompt import EXTRACT_FILTER_PROMPT
from modules.module_services.chat_models import RobustLLM
from utils.query_utils.milvus_query_builder import MilvusQueryBuilder
from utils.query_utils.query_model import Condition, Query

logger = logging.getLogger(__name__)


class QueryFilter:
    def __init__(self, config: FilterConfig,llm_client: RobustLLM):
        self.config = config
        self.llm = llm_client

    def extract(self,query: str) -> Optional[str]:
        if not self.config.enabled:
            return None
        try:
            prompt = EXTRACT_FILTER_PROMPT + f"\n\n用户问题：{query}"
            response = self.llm.invoke(prompt)
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            conditions = json.loads(raw)
            if not conditions:
                return None

            parts = []
            if "product_type" in conditions and conditions["product_type"]:
                parts.append(Condition(field="product_type", value=conditions["product_type"],op="=="))
            if "topics" in conditions and isinstance(conditions["topics"], list):
                for topic in conditions["topics"]:
                    parts.append(Condition(field="topics", value=topic,op="array_contains"))
            return MilvusQueryBuilder().build(Query(conditions=parts,logic="AND"))
        except Exception as e:
            logger.warning(f"Filter extraction failed: {e}")
            return None





