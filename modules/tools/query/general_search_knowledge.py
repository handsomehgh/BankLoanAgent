# author hgh
# version 1.0
"""
general knowledge query tool
"""
import logging
from typing import List, Optional, Annotated

from langchain_core.tools import tool, InjectedToolArg
from pydantic import BaseModel, Field

from config.global_constant.constants import KnowledgeFileSourceType
from modules.agent.constants import AgentName
from modules.retrieval.knowledge_utils.knowledge_formatter import format_context
from modules.retrieval.retrieval_service import RetrievalService
from modules.tools.error_handler import with_tool_error_handling
from utils.query_utils.milvus_query_builder import MilvusQueryBuilder
from utils.query_utils.query_model import Condition, Query

logger = logging.getLogger(__name__)


class GeneralSearchKnowledgeInput(BaseModel):
    query: str = Field(..., description="检索关键词或完整的自然语言问题，如 '装修贷款的申请材料有哪些")
    source_type: Optional[List[KnowledgeFileSourceType]] = Field(None,
                                                                 description="知识类型过滤列表，可选值：faq, product_manual, process_guide, regulation, glossary。"
                                                                             "可同时指定多个，如 ['faq', 'product_manual']。留空不过滤。")
    product_type: Optional[str] = Field(
        None,
        description="产品类型过滤：住房贷款、消费贷款、经营贷款、特色贷款。留空不过滤。"
    )
    top_k: int = Field(10, ge=1, le=20, description="返回结果数量，默认10")


@tool(
    "general_search_knowledge",
    description="从银行知识库中检索相关文档。"
                "query 应为完整的自然语言问题（如 '住房贷款的利率是多少'），避免过于简短的词。"
                "可通过 source_type 列表和 product_type 精准过滤。"
                "返回格式化文本及匹配文档数量。",
    args_schema=GeneralSearchKnowledgeInput,
    extras={"version": "1.0.0",
            "tags": [AgentName.AFTER_LOAN.value, AgentName.LOAN_ADVISOR.value, AgentName.RISK_ASSESSMENT.value],
            "baseline": True}
)
@with_tool_error_handling
def general_search_knowledge(input: GeneralSearchKnowledgeInput,
                             retriever: Annotated[RetrievalService, InjectedToolArg]) -> dict:
    parts = []
    if input.source_type:
        source_values = [source.value for source in input.source_type]
        parts.append(Condition(field="source_type", value=source_values, op="in"))
    if input.product_type:
        parts.append(Condition(field="product_type", value=input.product_type, op="=="))
    filter_expr = MilvusQueryBuilder().build(Query(conditions=parts, logic="AND"))

    # execute retrieve
    docs = retriever.retrieve(query=input.query, context=None, filter_expr=filter_expr)

    if not docs:
        logger.info("SearchKnowledge 未找到结果: query='%s', filter=%s", input.query, filter_expr)
        return {
            "query": input.query,
            "documents": "",
            "total_found": 0,
            "message": "未找到相关知识，请尝试更换关键词或放宽过滤条件。"
        }

    knowledge_text = format_context(docs, max_context_length=3000)
    return {
        "query": input.query,
        "documents": knowledge_text,
        "total_found": len(docs)
    }
