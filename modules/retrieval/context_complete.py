# author hgh
# version 1.0
import json
import logging
import os
from pathlib import Path

import requests

from config.models.retrieval_config import RetrievalConfig
from config.prompts.context_rewrite_prompt import CONTEXT_REWRITE_PROMPT
from modules.module_services.chat_models import RobustLLM

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class ContextComplete:
    def __init__(self, config: RetrievalConfig, llm_client: RobustLLM):
        self.api_url = config.context_complete_uri
        self.llm = llm_client

    def complete(self, query: str, context: str) -> str:
        try:
            resp = requests.post(
                f"{self.api_url}/predict/context",
                json={"text_a": context, "text_b": query},
                timeout=30
            )
            resp.raise_for_status()
            result = resp.json()["label"]
            result_probability = resp.json()["probability"]
            logger.info(f"Context complete result: {result}, probability: {result_probability}")

            if result == "COMPLETE" and result_probability <= 0.6:
                low_data_dir = PROJECT_ROOT / "data" / "wheel" / "complete"
                os.makedirs(low_data_dir, exist_ok=True)
                low_data_file = low_data_dir / "context_complete.jsonl"

                with open(str(low_data_file), "a", encoding="utf-8") as f:
                    record = {"text_a": context, "text_b": query, "label": result}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if result == "COMPLETE" and result_probability > 0.6:
                return query

            logger.debug("Context-aware completion with summary: %.50s...", context)
            messages = CONTEXT_REWRITE_PROMPT.invoke({"last_summary": context, "query": query}).to_messages()
            rewritten = self.llm.invoke(messages).content.strip()
            logger.info(f"RAG Context-aware complete: '{query}' -> '{rewritten[:50]}'")
            if rewritten and len(rewritten) > 0:
                logger.info("Context-aware complete: '%s' -> '%s'", query[:50], rewritten[:50])
                return rewritten
        except Exception as e:
            logger.warning("Context-aware completion failed: %s", e, exc_info=True)
        return query
