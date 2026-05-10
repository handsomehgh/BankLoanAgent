# author hgh
# version 1.0
import logging
from typing import List

from modules.retrieval.knowledge_model import BusinessKnowledge

logger = logging.getLogger(__name__)

def format_context(docs: List[BusinessKnowledge],max_context_length: int = 2000) -> str:
    """format the knowledge text"""
    if not docs:
        return "无。"

    blocks = []
    total_chars = 0
    for idx,doc in enumerate(docs):
        source_info = f"{doc.source_type.value if hasattr(doc.source_type, 'value') else doc.source_type}"
        product_info = f" - {doc.product_type}" if doc.product_type else ""
        header = f"【来源 {idx} | {source_info}{product_info}】\n"
        body = doc.text.strip()
        block = header + body

        if total_chars + len(block) > max_context_length:
            remaining_chars = max_context_length - total_chars
            if remaining_chars > len(header):
                block = header + body[:remaining_chars - len(header)] + "..."
            else:
                break

        blocks.append(block)
        total_chars += len(block)

        if total_chars >= max_context_length:
            break

    return "\n\n".join(blocks)








