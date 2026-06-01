import json
import logging
import random
import re
from pathlib import Path
from typing import List, Dict

from config.global_constant.constants import RegistryModules, MemoryType
from infra.database.collections_type import CollectionNames
from infra.database.milvus_client import MilvusClientManager
from modules.module_services.chat_models import RobustLLM
from utils.config_utils.get_config import get_config
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

# 新的 Prompt：只给标签，不给原文
TOPIC_BASED_QUESTION_PROMPT = """
你是一位正在银行智能客服对话框中咨询贷款业务的客户。请根据下面的【专业知识片段】，转化成一个真实、口语化的用户问题。

【专业知识片段】
{chunk_text}

【转化要求】
1. **禁止直接引用原文**：不允许使用原文中的完整句子、专业术语堆砌或固定表达。
2. **必须用你自己的话**：想象你是一个对银行业务不太熟悉的普通客户，用你理解后的、最直白的语言来表达。
3. **虚构一个具体场景**：根据片段内容，虚构一个合理的咨询场景（如首次购房、个体户周转、想提前还款等）。
4. **包含具体的数字或条件**：如贷款金额、期限、月收入、年龄等（请合理虚构）。
5. **句式和篇幅必须随机多样**：
   - 有时可以很长（交代一堆背景后问一个小问题）。
   - 有时可以很短（直接一句话抛出核心问题）。
6. **禁止使用固定客套开头**：如“您好”、“经理”、“咨询一下”。

【正确示范】
专业知识片段：首套房利率最低可至LPR-20BP，具体以银行审批为准...
转化后的问题：“我最近在看房子，算了下首付还差一点，想贷个200万左右，分20年还。我这是买的第一套，你们这边现在最低能给我多少的利率啊？”

【错误示范（请避免）】
专业知识片段：首套房利率最低可至LPR-20BP...
转化后的问题（错误）：“请问首套房利率最低能到LPR-20BP吗？”（直接引用原文术语）

现在请处理以下专业知识片段，只输出一个转化后的口语化问题：
"""


def generate_questions(
    llm: RobustLLM,
    chunks: List[Dict],
    num_samples: int = 100,
    output_path: Path = Path("data/eval/auto_generated.jsonl")
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    random.shuffle(chunks)
    sampled = chunks[:num_samples]

    with open(output_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(sampled):
            chunk_text = chunk.get("text", "")
            chunk_id = chunk.get("id", "")
            if not chunk_text or not chunk_id:
                continue

            # 用基于原文的转化式 Prompt 生成口语化问题
            prompt = TOPIC_BASED_QUESTION_PROMPT.format(chunk_text=chunk_text[:800])
            try:
                response = llm.invoke(prompt)
                question = response.content.strip() if hasattr(response, 'content') else str(response).strip()
                question = re.sub(r'^\d+[\.\、\s]+', '', question)
            except Exception as e:
                logger.warning(f"生成问题失败 (chunk {chunk_id}): {e}")
                continue

            # 直接标记原 chunk 为正确答案
            record = {
                "query": question,
                "relevant_doc_ids": [chunk_id],
                "ground_truth_answer": chunk_text
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info(f"已生成 {i + 1}/{len(sampled)}: {question[:50]}...")

    logger.info(f"候选数据集已保存至 {output_path}")

def main():
    setup_logging(log_level="INFO")
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
    # 注意：需要额外查询 topics 和 product_type 字段
    results = collection.query(
        expr="status == 'active'",
        output_fields=["id", "text", "source_type", "topics", "product_type"],
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
    main()