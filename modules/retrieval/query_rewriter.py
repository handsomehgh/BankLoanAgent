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
from typing import Optional, List

from config.models.retrieval_config import RewriterConfig
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

        if any(kw in query for kw in [ "介绍", "哪些", "种类", "分类", "有哪些","怎么选", "如何选择", "哪个好", "哪个更好","有什么区别", "区别", "比较", "对比","优缺点", "选哪个"]):
            return RewritingStrategy.STEP_BACK

        if len(query) > 20 and any(
                kw in query for kw in ["贷款", "利率", "额度", "还款", "征信", "经营", "消费", "住房"]):
            return RewritingStrategy.HYDE

        return None


class QueryRewriter:
    def __init__(self, config: RewriterConfig,llm_client: RobustLLM):
        self.config = config
        self.selector = DynamicStrategySelector()
        self.llm = llm_client

    def rewrite(self, query: str) -> List[str]:
        """
        main entrance: return the rewritten query according to the selected strategy
        return a list containing the original query when the failure or strategy is none
        """
        if self.config.enable_dynamic and not self.config.override_strategy:
            strategy = self.selector.select(query)
        else:
            strategy = self.config.override_strategy

        logger.info(f"rewriting strategy selected: {strategy} for query: {query[60:]}")

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
        prompt = MULTI_QUERY_PROMPT.format(num_variants=self.config.num_variants)
        full_prompt = f"{prompt}\n\n原问题：{query}\n查询变体"

        # invoke llm
        response = self.llm.invoke(full_prompt)

        # parse result
        text = response.content.strip()
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
        return final

    def _hyde(self, query: str) -> List[str]:
        prompt = HYDE_QUERY_PROMPT + f"\n\n用户问题：{query}\n文档片段："
        response = self.llm.invoke(prompt)
        text = response.content.strip()
        if not text:
            raise ValueError("HyDE generate result is null")
        return [text]

    def _stepback(self, query: str) -> List[str]:
        prompt = STEPBACK_QUERY_PROMPT + f"\n\n用户问题：{query}\n抽象问题："
        response = self.llm.invoke(prompt)
        text = response.content.strip()
        if not text:
            raise ValueError("Step-back generate result is null")
        return [text]
