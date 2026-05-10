# author hgh
# version 1.0
"""
query rewriter module - dynamic strategy selection
function:
 - automatically select the optimal rewriting strategy based on the semantic features of the user's query
 - support three rewriting modes:Multi-query,Step-back and Hyde
 - automatically downgrade to the original query in case of failure,ensuring high availability
"""
import logging
from typing import Optional, List, Dict

from config.models.retrieval_config import RewriterConfig
from config.prompts.context_rewrite_prompt import CONTEXT_REWRITE_PROMPT
from config.prompts.hyde_query_prompt import HYDE_QUERY_PROMPT
from config.prompts.multi_query_prompt import MULTI_QUERY_PROMPT
from config.prompts.stepback_query_prompt import STEPBACK_QUERY_PROMPT
from modules.module_services.chat_models import RobustLLM
from modules.retrieval.knowledge_constant import RewritingStrategy

logger = logging.getLogger(__name__)


class DynamicStrategySelector:
    """
    automatically select the rewriting strategy based on query feature
    """

    @staticmethod
    def select(query: str) -> Optional[str]:
        if len(query) <= 5 or any(c.isdigit() for c in query) or any(kw in query for kw in ["LPR", "BP", "DTI"]):
            return RewritingStrategy.MULTI_QUERY

        if any(kw in query for kw in
               ["介绍", "哪些", "种类", "分类", "有哪些", "怎么选", "如何选择", "哪个好", "哪个更好", "有什么区别",
                "区别", "比较", "对比", "优缺点", "选哪个"]):
            return RewritingStrategy.STEP_BACK

        if len(query) > 20 and any(
                kw in query for kw in ["贷款", "利率", "额度", "还款", "征信", "经营", "消费", "住房"]):
            return RewritingStrategy.HYDE

        return None


class QueryRewriter:
    def __init__(self, config: RewriterConfig, llm_client: RobustLLM):
        self.config = config
        self.selector = DynamicStrategySelector()
        self.llm = llm_client

    def _needs_context_complete(self,query: str) -> bool:
        if len(query) <= 10:
            return True
        if any(p in query for p in ["它", "那", "这个", "那个", "这", "其"]):
            return True
        return False

    def _complete_context(self, query: str, last_summary: str) -> str:
        if not last_summary and not last_summary.strip():
            return query
        try:
            logger.info(f"RAG Context-aware: {last_summary[:50]}")
            messages = CONTEXT_REWRITE_PROMPT.invoke({"last_summary": last_summary, "query": query}).to_messages()
            rewritten = self.llm.invoke(messages).content.strip()
            logger.info(f"RAG Context-aware complete: '{query}' -> '{rewritten[:50]}'")
            if rewritten and len(rewritten) > 0:
                return rewritten
        except Exception as e:
            logger.warning(f"Context-aware completion failed: {e}")
        return query

    def rewrite(self, query: str, context: Optional[Dict] = None) -> List[str]:
        """
        main entrance: return the rewritten query according to the selected strategy
        return a list containing the original query when the failure or strategy is none
        """

        last_summary = context.get("last_summary") if context else None
        if last_summary and self._needs_context_complete(query):
            query = self._complete_context(query, last_summary)

        if self.config.enable_dynamic and not self.config.override_strategy:
            strategy = self.selector.select(query)
        else:
            strategy = self.config.override_strategy
        logger.info(f"rewriting strategy selected: {strategy} for query: {query[:60]}")

        try:
            if strategy == RewritingStrategy.MULTI_QUERY:
                return self._multi_query(query)
            elif strategy == RewritingStrategy.STEP_BACK:
                return self._stepback(query)
            elif strategy == RewritingStrategy.HYDE:
                return self._hyde(query)
            else:
                return [query]
        except Exception as e:
            logger.warning(f"Query rewrite failed (strategy={strategy}), fallback to original. Error: {e}")
            if self.config.fallback_to_original:
                return [query]
            else:
                raise

    def _multi_query(self, query: str) -> List[str]:
        #prompt
        messages = MULTI_QUERY_PROMPT.invoke({
            "num_variants": self.config.num_variants,
            "query": query
        }).to_messages()

        # invoke llm
        response = self.llm.invoke(messages)

        # parse result
        text = response.content.strip()
        logger.info(f"Multi_query rewriting llm generate result: {text[:100]}")

        variants = []
        for line in text.split('\n'):
            line = line.strip()
            if line and not line.startswith("查询变体") and not line.startswith("原问题"):
                # 去除可能的编号
                if line[0].isdigit() and (line[1] == '.' or line[1] == '、'):
                    line = line[2:].strip()
                variants.append(line)
        if not variants:
            raise ValueError("Multi-query generate result is null")

        # remove duplicates
        seen = {query}
        final = []
        for v in variants:
            if v not in seen:
                seen.add(v)
                final.append(v)
                if len(final) >= self.config.num_variants:
                    break
        final.append(query)
        logger.info(f"Multi_query rewriting final result: {text} for query: {text[:100]}")
        return final

    def _hyde(self, query: str) -> List[str]:
        messages = HYDE_QUERY_PROMPT.invoke({"query": query}).to_messages()

        response = self.llm.invoke(messages)
        logger.info(f"HYDE rewriting llm generate result: {response[:100]}")

        text = response.content.strip()
        if not text:
            raise ValueError("HyDE generate result is null")
        return [text]

    def _stepback(self, query: str) -> List[str]:
        messages = STEPBACK_QUERY_PROMPT.invoke({"query": query}).to_messages()

        response = self.llm.invoke(messages)
        logger.info(f"step_back rewriting llm generate result: {response[:100]}")

        text = response.content.strip()
        if not text:
            raise ValueError("Step-back generate result is null")
        return [text]
