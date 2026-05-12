# author hgh
# version 1.0
from typing import Dict, Any

from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge, generate_latest

#================= rag metrics ======================
rag_requests_total = Counter(
    'rag_requests_total',
    'Total number of retrieval requests.',
    ['status', 'cache_hit', 'route_skipped'],
)
rag_retrieval_duration_seconds = Histogram(
    'rag_retrieval_duration_seconds',
    'Retrieval pipeline total duration.',
    buckets=(0.1, 0.5, 1, 2, 5, 10, 20, 30, 60),
)
rag_dense_hits = Gauge('rag_dense_hits', 'Hits from dense vector search.')
rag_sparse_hits = Gauge('rag_sparse_hits', 'Hits from sparse keyword search.')
rag_term_hits = Gauge('rag_term_hits', 'Hits from term vector search.')
rag_fused_candidates = Gauge('rag_fused_candidates', 'Candidates after RRF fusion.')
rag_rerank_candidates = Gauge('rag_rerank_candidates', 'Candidates sent to reranker.')
rag_compression_ratio = Gauge('rag_compression_ratio', 'Average compression ratio.')


#=================== llm metrics ===================
llm_requests_total = Counter(
    'llm_requests_total',
    'Total number of LLM calls.',
    ['provider']
)
llm_tokens_total = Gauge('llm_tokens_total', 'Total tokens consumed by last LLM call.')

#=============== add memory metrics ====================
memory_write_total = Counter(
    'memory_write_total',
    'Total number of memory writes.',
    ['type', 'user']
)

#============= compliance metrics =========================
compliance_block_total = Counter(
    'compliance_block_total',
    'Total number of compliance blocks.',
    ['action', 'reason']
)

#================== metrics record function ====================
def record_retrieval_metrics(stats: Dict[str, Any], route_skipped: bool = False, cache_hit: bool = False):
    """根据检索统计字典写入 Prometheus 指标"""
    status = "success"
    if route_skipped:
        status = "route_skipped"
    elif cache_hit:
        status = "cache_hit"

    rag_requests_total.labels(
        status=status,
        cache_hit=str(cache_hit),
        route_skipped=str(route_skipped),
    ).inc()

    if not route_skipped and not cache_hit:
        # 正常检索，记录详细指标
        if 'duration_ms' in stats:
            rag_retrieval_duration_seconds.observe(stats['duration_ms'] / 1000.0)
        if 'dense' in stats:
            rag_dense_hits.set(stats['dense'])
        if 'sparse' in stats:
            rag_sparse_hits.set(stats['sparse'])
        if 'term' in stats:
            rag_term_hits.set(stats['term'])
        if 'fused' in stats:
            rag_fused_candidates.set(stats['fused'])
        if 'rerank' in stats:
            rag_rerank_candidates.set(stats['rerank'])
        if 'comp_ratio' in stats:
            rag_compression_ratio.set(stats['comp_ratio'])


def record_llm_metrics(provider: str, total_tokens: int):
    """记录 LLM 调用指标"""
    llm_requests_total.labels(provider=provider).inc()
    llm_tokens_total.set(total_tokens)


def record_memory_write_metrics(memory_type: str, user: str):
    """记录记忆写入指标"""
    memory_write_total.labels(type=memory_type, user=user).inc()


def record_compliance_block_metrics(reason: str):
    """记录合规拦截指标"""
    compliance_block_total.labels(action='block', reason=reason).inc()