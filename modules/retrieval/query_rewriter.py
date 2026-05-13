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
        @staticmethod
        def select(query: str) -> Optional[str]:
            # ---------- 1. HYDE：仅用于极少数场景 ----------
            # 当用户问的是非常宽泛的大概念、且查询本身很简短时，用 HyDE 生成一段假设文档来增强检索
            # 例如："住房贷款"、"经营贷政策"
            if len(query) <= 10 and not any(kw in query for kw in ["怎么", "如何", "什么", "为什么"]):
                return RewritingStrategy.HYDE

            # ---------- 2. STEP_BACK：需要抽象化概括的场景 ----------
            # 用户在进行对比、选择、了解全貌时，把具体问题抽象为更通用的概念问题
            step_back_kw = [
                "有什么区别", "有什么不同", "哪个好", "哪个更好", "哪个划算",
                "怎么选", "如何选择", "选哪个", "有哪些", "哪些种类", "种类",
                "分类", "对比", "比较", "优缺点", "分别是什么"
            ]
            if any(kw in query for kw in step_back_kw) and len(query) > 8:
                return RewritingStrategy.STEP_BACK

            # ---------- 3. MULTI_QUERY：需要多角度表达的场景 ----------
            # 短查询或意图明确的精确查询，用多查询扩展来提升召回
            if len(query) <= 12:
                return RewritingStrategy.MULTI_QUERY

            # ---------- 4. 默认：不改写 ----------
            # 大多数自然语言查询已经足够完整，不需要额外改写
            return None


class QueryRewriter:
    def __init__(self, config: RewriterConfig, llm_client: RobustLLM):
        self.config = config
        self.selector = DynamicStrategySelector()
        self.llm = llm_client
        logger.info("QueryRewriter initialized with strategy=%s",
                    "dynamic" if config.enable_dynamic else config.override_strategy)

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
            logger.debug("Context-aware completion with summary: %.50s...", last_summary)
            messages = CONTEXT_REWRITE_PROMPT.invoke({"last_summary": last_summary, "query": query}).to_messages()
            rewritten = self.llm.invoke(messages).content.strip()
            logger.info(f"RAG Context-aware complete: '{query}' -> '{rewritten[:50]}'")
            if rewritten and len(rewritten) > 0:
                logger.info("Context-aware complete: '%s' -> '%s'", query[:50], rewritten[:50])
                return rewritten
        except Exception as e:
            logger.warning("Context-aware completion failed: %s", e, exc_info=True)
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
        logger.info("Rewrite strategy selected: %s for query: '%s...'", strategy, query[:60])

        try:
            if strategy == RewritingStrategy.MULTI_QUERY:
                results = self._multi_query(query)
            elif strategy == RewritingStrategy.STEP_BACK:
                results = self._stepback(query)
            elif strategy == RewritingStrategy.HYDE:
                results = self._hyde(query)
            else:
                results = [query]
            logger.debug("Rewrite produced %d queries", len(results))
            return results
        except Exception as e:
            logger.warning("Query rewrite failed (strategy=%s), fallback to original. Error: %s", strategy, e,exc_info=True)
            if self.config.fallback_to_original:
                return [query]
            else:
                raise

    def _multi_query(self, query: str) -> List[str]:
        logger.debug("Executing Multi-Query rewrite")
        #prompt
        messages = MULTI_QUERY_PROMPT.invoke({
            "num_variants": self.config.num_variants,
            "query": query
        }).to_messages()

        # invoke llm
        response = self.llm.invoke(messages)

        # parse result
        text = response.content.strip()
        logger.debug("Multi-query LLM raw output: %.100s...", text)

        variants = []
        for line in text.split('\n'):
            line = line.strip()
            if line and not line.startswith("查询变体") and not line.startswith("原问题"):
                # 去除可能的编号
                if line[0].isdigit() and (line[1] == '.' or line[1] == '、'):
                    line = line[2:].strip()
                variants.append(line)
        if not variants:
            logger.warning("Multi-query generated empty variants, falling back")
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
        logger.info("Multi-query generated %d variants (including original query)", len(final))
        return final

    def _hyde(self, query: str) -> List[str]:
        logger.debug("Executing HyDE rewrite")
        messages = HYDE_QUERY_PROMPT.invoke({"query": query}).to_messages()

        response = self.llm.invoke(messages)

        text = response.content.strip()
        logger.info("HyDE LLM raw output: %.100s...", text)
        if not text:
            logger.warning("HyDE generated empty output")
            raise ValueError("HyDE generate result is null")
        return [text]

    def _stepback(self, query: str) -> List[str]:
        logger.debug("Executing Step-back rewrite")
        messages = STEPBACK_QUERY_PROMPT.invoke({"query": query}).to_messages()

        response = self.llm.invoke(messages)

        text = response.content.strip()
        logger.debug("Step-back LLM raw output: %.100s...", text)
        if not text:
            logger.warning("Step-back generated empty output")
            raise ValueError("Step-back generate result is null")
        return [text]
