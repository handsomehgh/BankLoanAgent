# author hgh
# version 1.0
import logging
from typing import List, Dict

from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from config.global_constant.fields import CommonFields
from config.models.retrieval_config import CompressorConfig
from config.prompts.context_compress import CONTEXT_COMPRESS_PROMPT
from modules.module_services.chat_models import RobustLLM

logger = logging.getLogger(__name__)


class RerankResult(BaseModel):
    sorted_ids: list[int] = Field(description="按相关性从高到低排序的文档ID列表")


"""
split the original document content into sentences,
and user a cross encoder to rank and select the most appropriate content
"""


class ContextCompressor:
    def __init__(self, config: CompressorConfig, llm_client: RobustLLM):
        self.config = config
        self.llm_client = llm_client
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    def compress(self, query: str, documents: List[Dict]) -> List[Dict]:
        """
        compress the text of each document,retaining key sentences,and if it fails,keep the original text
        """
        if not self.config.enabled:
            logger.debug("Context compression is disabled, returning %d documents as-is", len(documents))
            return documents

        # doc_text = [item.get(CommonFields.TEXT, "") for item in documents if item.get(CommonFields.TEXT, "")]
        # total_token = sum(len(self.tokenizer.encode(item)) for item in doc_text)
        # if total_token > self.config.max_context_tokens:
        #     logger.info("[ContextCompressor] Total tokens %d within limit, skip compression", total_tokens)
        #     return documents

        try:
            formatted_docs = []
            for doc in documents:
                text = doc['text']
                formatted_docs.append(f"[id:{doc['id']}] {text}")
            docs_str = "\n".join(formatted_docs)
            messages = CONTEXT_COMPRESS_PROMPT.invoke({"query": query, "docs": docs_str}).to_messages()
            res = self.llm_client.invoke(messages, schema=RerankResult)
            sorted_ids = res.sorted_ids if hasattr(res, 'sorted_ids') else []
            if not sorted_ids:
                logger.warning("[ContextCompressor] LLM returned empty sorted_ids, fallback to top-3")
                return documents[:3]

            doc_map = {doc.get('id'): doc for doc in documents}
            compressed = []
            for doc_id in sorted_ids:
                if doc_id in doc_map:
                    compressed.append(doc_map[doc_id])
                else:
                    logger.warning("Document id %s not found in original documents, skip", doc_id)

            logger.info("Compressed %d docs -> %d docs", len(documents), len(compressed))
            return compressed[:self.config.compress_top_k]

        except Exception as e:
            logger.error(f"[ContextCompressor] Failed to compress {query}: {e}")
            return documents[:3]
