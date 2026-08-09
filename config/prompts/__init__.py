# author hgh
# version 1.0
"""提示词统一出口，按领域分组。业务代码优先从这里导入。"""

# ---------- 检索（查询改写 / 过滤 / 重排） ----------
from config.prompts.context_rewrite_prompt import CONTEXT_REWRITE_PROMPT
from config.prompts.decompose import DECOMPOSE_PROMPT
from config.prompts.extract_filter_prompt import EXTRACT_FILTER_PROMPT
from config.prompts.faq_similar_prompt import FAQ_SIMILAR_PROMPT_TEMPLATE
from config.prompts.hyde_query_prompt import HYDE_QUERY_PROMPT
from config.prompts.llm_rerank_prompt import LLM_RERANK_PROMPT
from config.prompts.multi_query_prompt import MULTI_QUERY_PROMPT
from config.prompts.stepback_query_prompt import STEPBACK_QUERY_PROMPT

# ---------- 记忆（画像提取 / 摘要 / 情感 / 证据） ----------
from config.prompts.detect_evidence_prompt import EVIDENCE_PROMPT
from config.prompts.detect_sentiment_prompt import DETECT_SENTIMENT_PROMPT
from config.prompts.extract_prompt import EXTRACT_PROMPT
from config.prompts.summary_interaction_prompt import (
    SUB_SUMMARY_INTERACTION_PROMPT,
    SUMMARY_INTERACTION_PROMPT,
)

# ---------- Agent（系统提示词 / 合规预检） ----------
from config.prompts.compliance_fallback_prompt import COMPLIANCE_FALLBACK_PROMPT
from config.prompts.system_prompt import SYSTEM_PROMPT

__all__ = [
    "CONTEXT_REWRITE_PROMPT",
    "DECOMPOSE_PROMPT",
    "EXTRACT_FILTER_PROMPT",
    "FAQ_SIMILAR_PROMPT_TEMPLATE",
    "HYDE_QUERY_PROMPT",
    "LLM_RERANK_PROMPT",
    "MULTI_QUERY_PROMPT",
    "STEPBACK_QUERY_PROMPT",
    "EVIDENCE_PROMPT",
    "DETECT_SENTIMENT_PROMPT",
    "EXTRACT_PROMPT",
    "SUB_SUMMARY_INTERACTION_PROMPT",
    "SUMMARY_INTERACTION_PROMPT",
    "COMPLIANCE_FALLBACK_PROMPT",
    "SYSTEM_PROMPT",
]
