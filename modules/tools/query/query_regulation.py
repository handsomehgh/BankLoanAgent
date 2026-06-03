# author hgh
# version 1.0
"""
Regulatory Compliance Search Tool
"""
import logging
from typing import Annotated

from langchain_core.tools import tool, InjectedToolArg
from pydantic import BaseModel, Field

from modules.agent.constants import AgentName
from modules.retrieval.retrieval_service import RetrievalService
from modules.tools.error_handler import with_tool_error_handling

logger = logging.getLogger(__name__)


class QueryRegulationInput(BaseModel):
    keyword: str = Field(..., description="检索关键词，如 '个人贷款管理办法'、'征信管理条例'、'逾期催收' 等")


@tool(
    "query_regulation",
    description="检索银行监管法规条文。根据关键词从知识库中查找相关法规原文或摘要，为风险评估提供法律/政策依据。",
    args_schema=QueryRegulationInput,
    extras={"version": "1.0.0", "tags": [AgentName.RISK_ASSESSMENT.value]}
)
@with_tool_error_handling
def query_regulation(
    input: QueryRegulationInput,
    knowledge_retriever: Annotated[RetrievalService, InjectedToolArg],
) -> dict:
    # 1. execute retrieval
    docs = knowledge_retriever.retrieve(input.keyword)
    if not docs:
        return {
            "keyword": input.keyword,
            "regulations": [],
            "message": "未找到相关法规"
        }

    # 2. format result
    regulations = []
    for doc in docs[:5]:
        regulations.append({
            "title": doc.regulation_names or doc.source_file or "未知法规",
            "content": doc.text[:300] + "..." if len(doc.text) > 300 else doc.page_content,
            "source": doc.source_type or "位置来源"
        })

    return {
        "keyword": input.keyword,
        "regulations": regulations,
        "total_found": len(docs),
        "disclaimer": "以上法规内容仅供参考，具体以官方发布的最新版本为准。"
    }
