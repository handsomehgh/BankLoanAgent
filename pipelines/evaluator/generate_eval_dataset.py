# author hgh
# version 1.0
import json
import logging
import random
import re
from pathlib import Path
from typing import List, Dict

from config.global_constant.constants import RegistryModules, MemoryType
from config.prompts.question_gen_prompt import QUESTION_GEN_PROMPT
from infra.collections import CollectionNames
from infra.milvus_client import MilvusClientManager
from modules.module_services.chat_models import RobustLLM
from utils.config_utils.get_config import get_config
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

def generate_questions(
        llm: RobustLLM,
        chunks: List[Dict],
        num_samples: int = 100,
        output_path: Path = Path("data/eval/auto_generated.jsonl")
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    random.shuffle(chunks)
    sampled = chunks[:num_samples]

    with open(output_path,"w",encoding="utf-8") as f:
        for i,chunk in enumerate(sampled):
            chunk_text = chunk.get("text","")
            chunk_id = chunk.get("id","")
            if not chunk_text or not chunk_id:
                continue

            messages = QUESTION_GEN_PROMPT.invoke({"chunk_text":chunk_text}).to_messages()
            try:
                response = llm.invoke(messages)
                question = response.content.strip()
                question = re.sub(r'^\d+[\.\、\s]+', '', question)
            except Exception as e:
                logger.warning(f"生成问题失败 (chunk {chunk_id}): {e}")
                continue

            record = {
                "query": question,
                "relevant_doc_ids": [chunk_id],
                "ground_truth_answer": chunk_text
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info(f"已生成 {i + 1}/{len(sampled)}: {question[:50]}...")
    logger.info(f"候选数据集已保存至 {output_path}")

def main():
    registry = get_config()
    llm_config = registry.get_config(RegistryModules.LLM)
    retrieval_config = registry.get_config(RegistryModules.RETRIEVAL)

    llm = RobustLLM(
        temperature=llm_config.creative_temperature,
        api_key=llm_config.deepseek_api_key,
        base_url=llm_config.deepseek_base_url,
        model=llm_config.deepseek_llm_name,
        provider=llm_config.openai_provider
    )

    milvus_client = MilvusClientManager(retrieval_config.milvus_uri)
    collection = milvus_client.get_collection(
        CollectionNames.for_type(MemoryType.BUSINESS_KNOWLEDGE)
    )
    results = collection.query(
        expr="status == 'active'",
        output_fields=["id", "text", "source_type"],
        limit=5000
    )

    logger.info(f"从知识库获取到 {len(results)} 个活跃 chunk")
    generate_questions(
        llm=llm,
        chunks=results,
        num_samples=100,
        output_path=Path(__file__).resolve().parent.parent / "data/eval/auto_generated.jsonl"
    )

if __name__ == "__main__":
    setup_logging(log_level="INFO")
    main()
