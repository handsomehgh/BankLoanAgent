# author hgh
# version 1.0
"""
Prometheus metrics
使用安全创建函数，避免多页面或重加载时的重复注册错误
"""
from typing import Dict, Any

from prometheus_client import Counter, Histogram, Gauge, REGISTRY

# ==================== 安全创建函数 ====================

def _get_or_create_counter(name: str, documentation: str, labels: list = None) -> Counter:
    """创建或获取已存在的 Counter，避免重复注册"""
    try:
        return Counter(name, documentation, labels or [])
    except ValueError:
        return REGISTRY._names_to_collectors[name]


def _get_or_create_histogram(name: str, documentation: str, labels: list = None,
                             buckets: list = None) -> Histogram:
    """创建或获取已存在的 Histogram"""
    try:
        return Histogram(name, documentation, labels or [], buckets=buckets or Histogram.DEFAULT_BUCKETS)
    except ValueError:
        return REGISTRY._names_to_collectors[name]


def _get_or_create_gauge(name: str, documentation: str, labels: list = None) -> Gauge:
    """创建或获取已存在的 Gauge"""
    try:
        return Gauge(name, documentation, labels or [])
    except ValueError:
        return REGISTRY._names_to_collectors[name]


# ==================== Supervisor & 路由 ====================

supervisor_routing_total = _get_or_create_counter(
    'supervisor_routing_total',
    'Total number of Supervisor routing decisions.',
    ['target_agent', 'route_method']          # route_method: llm / rule
)

supervisor_routing_duration_seconds = _get_or_create_histogram(
    'supervisor_routing_duration_seconds',
    'Duration of Supervisor routing decisions.',
    ['route_method'],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30]
)

fanout_dispatch_total = _get_or_create_counter(
    'fanout_dispatch_total',
    'Total number of Fan-out dispatches.',
    ['agent_count']
)

fanout_dispatch_duration_seconds = _get_or_create_histogram(
    'fanout_dispatch_duration_seconds',
    'Duration of Fan-out dispatches.',
    [],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30]
)


# ==================== Agent 执行 ====================

agent_duration_seconds = _get_or_create_histogram(
    'agent_duration_seconds',
    'Duration of each agent invocation (end-to-end).',
    ['agent_name'],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 30, 60]
)

agent_response_total = _get_or_create_counter(
    'agent_response_total',
    'Total number of agent responses.',
    ['agent_name', 'status']
)

agent_executor_errors_total = _get_or_create_counter(
    'agent_executor_errors_total',
    'Total number of agent executor errors.',
    ['agent_name','stage','error_type'],
)

agent_select_tool_total = _get_or_create_counter(
    'agent_select_tool_total',
    'Total number of agent select tool.',
    ['agent_name','tool_name']
)


# ==================== 工具系统 ====================

tool_call_total = _get_or_create_counter(
    'tool_call_total',
    'Total number of tool calls.',
    ['tool_name', 'status', 'caller_agent']   # status: success / error
)

tool_duration_seconds = _get_or_create_histogram(
    'tool_duration_seconds',
    'Duration of individual tool invocations.',
    ['tool_name', 'caller_agent'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
)

tool_permission_denied_total = _get_or_create_counter(
    'tool_permission_denied_total',
    'Total number of tool calls denied due to permission.',
    ['caller_agent', 'tool_name']
)

circuit_breaker_state = _get_or_create_gauge(
    'circuit_breaker_state',
    'Current state of circuit breaker (0=CLOSED, 1=OPEN, 2=HALF_OPEN)',
    ['tool_name']
)

# ==================== 知识检索 (RAG) ====================

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

knowledge_miss_total = Counter(
    'knowledge_miss_total',
    'Total number of knowledge retrieval empty results.',
    ['agent_name'],
)

def record_retrieval_metrics(stats: Dict[str, Any], route_skipped: bool = False, cache_hit: bool = False):
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


# ==================== 合规与安全 ====================

compliance_block_total = _get_or_create_counter(
    'compliance_block_total',
    'Total number of compliance blocks.',
    ['action', 'reason']
)

compliance_fallback_total = _get_or_create_counter(
    'compliance_fallback_total',
    'Total number of compliance LLM fallback invocations.'
)


# ==================== 人机协同 (HumanHandoff) ====================

handoff_task_created_total = _get_or_create_counter(
    'handoff_task_created_total',
    'Total number of human handoff tasks created.'
)

handoff_task_completed_total = _get_or_create_counter(
    'handoff_task_completed_total',
    'Total number of human handoff tasks completed.',
    ['result']                                  # result: replied / closed / timeout
)

handoff_task_timeout_total = _get_or_create_counter(
    'handoff_task_timeout_total',
    'Total number of human handoff task timeouts.'
)

handoff_task_pending_gauge = _get_or_create_gauge(
    'handoff_task_pending_gauge',
    'Number of pending human handoff tasks.'
)


# ==================== 记忆与画像 ====================

memory_write_total = _get_or_create_counter(
    'memory_write_total',
    'Total number of memory writes.',
    ['type', 'user']
)

memory_hit_total = _get_or_create_counter(
    'memory_hit_total',
    'Total number of memory hits.',
    ['type', 'user']
)

memory_write_duration_seconds = _get_or_create_histogram(
    "memory_write_duration_seconds",
    "Duration of memory writes",
    ['user'],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30]
)

memory_read_duration_seconds = _get_or_create_histogram(
    "memory_read_duration_seconds",
    "Duration of memory reads",
    ['user'],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30]
)

profile_extraction_total = _get_or_create_counter(
    'profile_extraction_total',
    'Total number of profile extractions.',
    ['method']                                  # method: sync / async
)

profile_extraction_duration_seconds = _get_or_create_histogram(
    'profile_extraction_duration_seconds',
    'Duration of profile extraction.',
    [],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30]
)


# ==================== 消息队列 ====================

mq_consumer_pending_total = _get_or_create_gauge(
    'mq_consumer_pending_total',
    'Number of pending messages in consumer streams.',
    ['stream_name']
)

mq_consumer_dead_letter_total = _get_or_create_counter(
    'mq_consumer_dead_letter_total',
    'Total number of dead letter messages.',
    ['stream_name']
)


# ==================== LLM 调用 ====================

llm_requests_total = _get_or_create_counter(
    'llm_requests_total',
    'Total number of LLM calls.',
    ['provider']
)

llm_tokens_total = _get_or_create_gauge(
    'llm_tokens_total',
    'Total tokens consumed by last LLM call.'
)

llm_request_duration_seconds = _get_or_create_histogram(
    'llm_request_duration_seconds',
    'Duration of LLM requests.',
    buckets=[0.5, 1, 2, 5, 10, 20, 30, 60]
)

def record_llm_metrics(provider: str, total_tokens: int,duration_ms: float):
    """记录 LLM 调用指标"""
    llm_requests_total.labels(provider=provider).inc()
    llm_tokens_total.set(total_tokens)
    llm_request_duration_seconds.observe(duration_ms / 1000.0)


# ==================== 反馈 ====================

negative_feedback_total = _get_or_create_counter(
    'negative_feedback_total',
    'Total number of negative feedback events.',
    ['reason']
)

#======================= skill ==================
skill_execution_total = _get_or_create_counter(
    'skill_execution_total',
    'Total number of skills executed.',
    ['skill_name','status']
)

skill_execution_duration_seconds = _get_or_create_histogram(
    'skill_execution_duration_seconds',
    'Duration of skills executed.',
    ['skill_name'],
    buckets=[0.5, 1, 2, 5, 10, 20, 30, 60]
)