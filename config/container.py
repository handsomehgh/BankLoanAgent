# author hgh
# version 1.0
"""
application service container
"""
import logging
import os
from pathlib import Path

from infra.database.mysql_manager import DatabaseManager
from infra.message_queue import MessageProducer
from infra.repository.LoanInterestRepository import LoanInterestRepository
from modules.agent.after_loan_agent.after_loan_agent import AfterLoanAgent
from modules.agent.constants import StreamName
from modules.agent.loan_advisor_agent.loan_advisor_agent import LoanAdvisorAgent
from modules.agent.nodes.direct_reply_node import DirectReplyNode
from modules.agent.nodes.extract_profile_node import ExtractProfileNode
from modules.agent.nodes.result_aggregator_agent import ResultAggregatorAgent
from modules.agent.nodes.summary_interaction_node import SummaryInteractionNode
from modules.agent.risk_assessment_agent.risk_assessment_agent import RiskAssessmentAgent
from modules.agent.nodes.compliance_prefilter import CompliancePrefilter
from modules.agent.nodes.human_handoff_interrupt_node import HumanHandoffInterruptNode
from modules.agent.nodes.human_handoff_notify_node import HumanHandoffResponseNode
from modules.agent.supervisor_agent.supervisor_agent import SupervisorAgent
from modules.agent.nodes.memory_retrieve_node import MemoryRetrieveNode
from modules.consumer.interaction_log_consumer import InteractionLogConsumer
from modules.consumer.user_profile_consumer import UserProfileConsumer
from modules.memory.memory_utils.cursor_manager import CursorManager
from modules.module_services.classifier.after_loan_classifier import AfterLoanClassifier
from modules.module_services.classifier.loan_advisor_classifier import LoanAdvisorClassifier
from modules.module_services.classifier.risk_assessment_classifier import RiskAssessmentClassifier
from modules.module_services.lpr_data_service import LPRDataService
from modules.retrieval.context_complete import ContextComplete
from modules.skills.skill_executor import SkillExecutor
from modules.skills.skill_registry import SkillRegistry
from modules.tools import ToolRegistry, ToolExecutor
from dependency_injector import containers, providers

from config.global_constant.constants import RegistryModules, CacheNamespace
from config.prompts.detect_evidence_prompt import EVIDENCE_PROMPT
from config.prompts.detect_setiment_prompt import DETECT_SENTIMENT_PROMPT
from config.prompts.extract_prompt import EXTRACT_PROMPT
from config.prompts.summary_interaction_prompt import SUMMARY_INTERACTION_PROMPT, SUB_SUMMARY_INTERACTION_PROMPT
from config.registry import ConfigRegistry
from infra.cache.cache_factory import CacheFactory
from infra.database.milvus_client import MilvusClientManager
from modules.memory.memory_business_store.long_term_memory_store import LongTermMemoryStore
from modules.memory.memory_retriever import MemoryVectorRetriever
from modules.memory.memory_vector_store.milvus_memory_vector_store import MilvusMemoryVectorStore
from modules.memory.memory_utils.profile_gate_util import ProfileGate
from modules.module_services.SummaryGenerator import SummaryGenerator
from modules.module_services.chat_models import RobustLLM
from modules.module_services.embeddings import RobustEmbeddings, RobustLocalEmbeder
from modules.module_services.evidence_infer import EvidenceTypeInfer
from modules.module_services.profile_extractor import ProfileExtractor
from modules.module_services.sentiment_analyser import SentimentAnalyzer
from modules.module_services.suggestion_timing_classifier import SuggestionTimingClassifier
from modules.retrieval.context_compressor import ContextCompressor
from modules.retrieval.knowledge_vector_store.knowledge_search_engine import KnowledgeSearchEngine
from modules.retrieval.query_filter import QueryFilter
from modules.retrieval.query_rewriter import QueryRewriter
from modules.retrieval.rereanker import Reranker
from modules.retrieval.retrieval_service import RetrievalService
from modules.tools.tool_selector import ToolSelector
from utils.serialize_utils.seq_generator import SequenceGenerator

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =====================================================================
# 工厂函数：纯粹的"如何创建对象"定义，不包含任何执行逻辑
# =====================================================================

def _get_llm_config(registry: ConfigRegistry):
    return registry.get_config(RegistryModules.LLM)


def _get_memory_config(registry: ConfigRegistry):
    return registry.get_config(RegistryModules.MEMORY_SYSTEM)


def _get_retrieval_config(registry: ConfigRegistry):
    return registry.get_config(RegistryModules.RETRIEVAL)


def _get_cache_config(registry: ConfigRegistry):
    return registry.get_config(RegistryModules.CACHE)


def _get_datasource_config(registry: ConfigRegistry):
    return registry.get_config(RegistryModules.DATASOURCE)


def _get_tool_registry_config(registry: ConfigRegistry):
    return registry.get_config(RegistryModules.TOOL_REGISTRY)


def _get_bank_global_config(registry: ConfigRegistry):
    return registry.get_config(RegistryModules.BANK_GLOBAL_CONFIG)


def _get_supervisor_config(registry: ConfigRegistry):
    return registry.get_config(RegistryModules.SUPERVISOR)


def _create_cache_factory(cache_config, redis_manager):
    return CacheFactory(config=cache_config, redis_manager=redis_manager)


def _create_redis_manager(datasource_config):
    from infra.database.redis_manager import RedisManager
    return RedisManager.from_config(datasource_config.redis)


def _create_seq_generator(redis_manager):
    return SequenceGenerator(redis_manager)


def _create_creative_llm(registry: ConfigRegistry):
    cfg = registry.get_config(RegistryModules.LLM)
    return RobustLLM(
        temperature=cfg.creative_temperature,
        api_key=cfg.deepseek_api_key,
        base_url=cfg.deepseek_base_url,
        model=cfg.deepseek_llm_name,
        provider=cfg.openai_provider
    )


def _create_precise_llm(registry: ConfigRegistry):
    cfg = registry.get_config(RegistryModules.LLM)
    return RobustLLM(
        temperature=cfg.precise_temperature,
        api_key=cfg.deepseek_api_key,
        base_url=cfg.deepseek_base_url,
        model=cfg.deepseek_llm_name,
        provider=cfg.openai_provider
    )

def _create_local_llm(registry: ConfigRegistry):
    cfg = registry.get_config(RegistryModules.LLM)
    return RobustLLM(
        temperature=cfg.precise_temperature,
        api_key="not_need",
        base_url=cfg.local_qwen_url,
        model=cfg.local_qwen_name,
        provider=cfg.openai_provider
    )

def _create_local_creative_llm(registry: ConfigRegistry):
    cfg = registry.get_config(RegistryModules.LLM)
    return RobustLLM(
        temperature=cfg.creative_temperature,
        api_key="not_need",
        base_url=cfg.local_qwen_url,
        model=cfg.local_qwen_name,
        provider=cfg.openai_provider
    )


def _create_embedder(registry: ConfigRegistry):
    cfg = registry.get_config(RegistryModules.LLM)
    return RobustLocalEmbeder(
        model_name=cfg.loan_official_embeder_name,
        base_url=cfg.loan_official_embeder_url,
        dimensions=cfg.loan_embeder_dimension
    )


def _create_local_embeder(registry: ConfigRegistry):
    cfg = registry.get_config(RegistryModules.LLM)
    return RobustLocalEmbeder(
        base_url=cfg.loan_custom_embeder_url,
        model_name=cfg.loan_custom_embeder_name,
        dimensions=cfg.loan_embeder_dimension
    )


def _create_milvus_client(registry: ConfigRegistry):
    cfg = registry.get_config(RegistryModules.RETRIEVAL)
    return MilvusClientManager(uri=os.getenv("MILVUS_URI", cfg.milvus_uri))


def _create_vector_store(registry: ConfigRegistry, embedder, milvus_client):
    mem_cfg = registry.get_config(RegistryModules.MEMORY_SYSTEM)
    if mem_cfg.vector_backend == "chroma":
        from modules.memory.memory_vector_store.chroma_memory_vector_store import ChromaVectorStore
        return ChromaVectorStore(persist_dir=mem_cfg.chroma_persist_dir)
    else:
        return MilvusMemoryVectorStore(
            milvus_client=milvus_client,
            embeder=embedder,
            config=mem_cfg
        )


def _create_cursor_manager(redis_manager):
    return CursorManager(redis_manager=redis_manager)


def _create_memory_store(vector_store, registry, cursor_manager):
    mem_cfg = registry.get_config(RegistryModules.MEMORY_SYSTEM)
    return LongTermMemoryStore(vector_store=vector_store, config=mem_cfg, cursor_manager=cursor_manager)


def _create_memory_retriever(memory_store, registry):
    mem_cfg = registry.get_config(RegistryModules.MEMORY_SYSTEM)
    return MemoryVectorRetriever(memory_store=memory_store, memory_config=mem_cfg)


def _create_knowledge_engine(milvus_client, embedder, registry):
    cfg = registry.get_config(RegistryModules.RETRIEVAL)
    return KnowledgeSearchEngine(milvus_client=milvus_client, embedder=embedder, config=cfg)


def _create_query_rewriter(registry, creative_llm):
    cfg = registry.get_config(RegistryModules.RETRIEVAL)
    return QueryRewriter(config=cfg.rewriter, llm_client=creative_llm)


def _create_query_filter(registry, precise_llm):
    cfg = registry.get_config(RegistryModules.RETRIEVAL)
    return QueryFilter(config=cfg.filter, llm_client=precise_llm)


def _create_reranker(registry):
    cfg = registry.get_config(RegistryModules.RETRIEVAL)
    return Reranker(config=cfg.reranker)


def _create_compressor(registry,llm_client):
    cfg = registry.get_config(RegistryModules.RETRIEVAL)
    return ContextCompressor(config=cfg.compressor,llm_client=llm_client)


def _create_retrieval_complete(registry, llm_client):
    cfg = registry.get_config(RegistryModules.RETRIEVAL)
    return ContextComplete(cfg, llm_client)


def _create_loan_advisor_classifier():
    return LoanAdvisorClassifier()


def _create_after_loan_classifier():
    return AfterLoanClassifier()


def _create_risk_assessment_classifier():
    return RiskAssessmentClassifier()


def _create_knowledge_retriever(knowledge_engine, query_rewriter, query_filter, reranker, compressor, context_complete,
                                registry):
    cfg = registry.get_config(RegistryModules.RETRIEVAL)
    return RetrievalService(
        engine=knowledge_engine,
        rewriter=query_rewriter,
        filter=query_filter,
        reranker=reranker,
        compressor=compressor,
        config=cfg,
        context_complete=context_complete
    )


def _create_summary_generator(creative_llm, registry):
    mem_cfg = registry.get_config(RegistryModules.MEMORY_SYSTEM)
    return SummaryGenerator(
        llm_client=creative_llm,
        prompt=SUMMARY_INTERACTION_PROMPT,
        max_summary_length=mem_cfg.max_summary_length,
        max_interaction_length=mem_cfg.interaction_log_max_length
    )


def _create_sub_summary_generator(creative_llm, registry):
    mem_cfg = registry.get_config(RegistryModules.MEMORY_SYSTEM)
    return SummaryGenerator(
        llm_client=creative_llm,
        prompt=SUB_SUMMARY_INTERACTION_PROMPT,
        max_summary_length=mem_cfg.max_summary_length,
        max_interaction_length=mem_cfg.interaction_log_max_length
    )


def _create_sentiment_analyzer(precise_llm, registry):
    mem_cfg = registry.get_config(RegistryModules.MEMORY_SYSTEM)
    return SentimentAnalyzer(
        llm_client=precise_llm,
        strong_keywords=mem_cfg.sentiment_rules.strong_keywords,
        prompt=DETECT_SENTIMENT_PROMPT
    )


def _create_suggestion_timing_classifier(local_llm, registry):
    """主动邀请时机分类器:prompt来自loan_advisor.yaml,复用local_llm不占主链路额度"""
    cfg = registry.get_config(RegistryModules.LOAN_ADVISOR)
    return SuggestionTimingClassifier(llm_client=local_llm, prompt_template=cfg.suggestion_gate_prompt)


def _create_evidence_infer(precise_llm, registry):
    mem_cfg = registry.get_config(RegistryModules.MEMORY_SYSTEM)
    return EvidenceTypeInfer(
        llm_client=precise_llm,
        strong_keywords=mem_cfg.evidence_rules.strong_keywords,
        prompt=EVIDENCE_PROMPT
    )


def _create_profile_extractor(precise_llm, registry):
    return ProfileExtractor(
        llm_client=precise_llm,
        extract_prompt=EXTRACT_PROMPT
    )


def _create_profile_gate(registry):
    mem_cfg = registry.get_config(RegistryModules.MEMORY_SYSTEM)
    return ProfileGate(rules=mem_cfg.memory_gate)


def _create_message_producer(redis_manager):
    return MessageProducer(redis_manager=redis_manager)


def _create_interaction_consumer(redis_manager, memory_store, summary_generator, sentiment_analyzer):
    return InteractionLogConsumer(
        StreamName.INTERACTION_LOG.value,
        StreamName.INTERACTION_LOG.value,
        redis_manager, memory_store, summary_generator, sentiment_analyzer
    )


def _create_sub_interaction_consumer(redis_manager, memory_store, summary_generator, sentiment_analyzer):
    return InteractionLogConsumer(
        StreamName.SUB_INTERACTION.value,
        StreamName.SUB_INTERACTION.value,
        redis_manager, memory_store, summary_generator, sentiment_analyzer
    )


def _create_user_profile_consumer(redis_manager, memory_store, evidence_infer, profile_extractor):
    return UserProfileConsumer(
        StreamName.USER_PROFILE.value,
        StreamName.USER_PROFILE.value,
        redis_manager, memory_store, evidence_infer, profile_extractor
    )


def _create_lpr_service(registry, cache):
    interest_config = registry.get_config(RegistryModules.BANK_GLOBAL_CONFIG)
    return LPRDataService(config=interest_config, cache=cache)


def _create_tool_registry(registry: ConfigRegistry):
    """
    创建工具注册表（仅创建对象，不执行扫描和验证）。
    扫描和验证在 bootstrap.py 阶段3显式调用，保证启动顺序可观测。
    """
    cfg = registry.get_config(RegistryModules.TOOL_REGISTRY)
    return ToolRegistry(cfg)


def _create_tool_executor(tool_registry, audit_logger):
    return ToolExecutor(registry=tool_registry, audit_logger=audit_logger)


def _create_skill_executor(tool_registry, audit_logger):
    return SkillExecutor(tool_registry, audit_logger)


def _build_supervisor_graph(memory_retriever, seq_generator, registry, llm_client, memory_config, knowledge_retrieve):
    agent = SupervisorAgent(memory_retriever, seq_generator, registry, llm_client, memory_config, knowledge_retrieve)
    return agent.build_graph()


def _build_loan_advisor_graph(llm_client, registry, tool_executor, seq_generator, tool_selector, classifier,
                              skill_executor, skill_selector, suggestion_classifier, suggestion_cache, db_manager):
    agent = LoanAdvisorAgent(
        llm_client=llm_client,
        registry=registry,
        tool_executor=tool_executor,
        seq_generator=seq_generator,
        tool_selector=tool_selector,
        classifier=classifier,
        skill_executor=skill_executor,
        skill_selector=skill_selector,
        suggestion_classifier=suggestion_classifier,
        suggestion_cache=suggestion_cache,
        db_manager=db_manager
    )
    return agent.build_graph()


def _build_risk_assessment_graph(llm_client, registry, tool_executor, seq_generator, tool_selector, classifier,
                                 skill_executor, skill_selector):
    agent = RiskAssessmentAgent(llm_client, registry, tool_executor, seq_generator, tool_selector, classifier,
                                skill_executor, skill_selector)
    return agent.build_graph()


def _build_after_loan_graph(llm_client, registry, tool_executor, seq_generator, tool_selector, classifier,
                            skill_executor, skill_selector):
    agent = AfterLoanAgent(llm_client, registry, tool_executor, seq_generator, tool_selector, classifier,
                           skill_executor, skill_selector)
    return agent.build_graph()


def _build_human_handoff_interrupt_node(redis_manager):
    return HumanHandoffInterruptNode(redis_manager)


def _build_extract_profile_node(memory_store, profile_gate, memory_config, evidence_infer, profile_extractor,
                                message_producer, cursor_manager):
    return ExtractProfileNode(memory_store, profile_gate, memory_config, evidence_infer, profile_extractor,
                              message_producer, cursor_manager)


def _build_interaction_node(memory_store, memory_config, summary_generator, sentiment_analyzer, message_producer,
                            cursor_manager):
    return SummaryInteractionNode(memory_store, memory_config, summary_generator, sentiment_analyzer,
                                  message_producer, cursor_manager)


def _create_database_manager(datasource_config):
    return DatabaseManager(datasource_config.mysql)


# =====================================================================
# 应用容器：纯粹的依赖定义（无执行逻辑）
# =====================================================================

class ApplicationContainer(containers.DeclarativeContainer):
    """Main Application Container —— 只定义'如何创建'，不执行任何初始化"""

    # ---------- Config Register Center ----------
    config_registry = providers.Singleton(ConfigRegistry)

    # ---------- Configuration ----------
    llm_config = providers.Callable(_get_llm_config, config_registry)
    memory_config = providers.Callable(_get_memory_config, config_registry)
    retrieval_config = providers.Callable(_get_retrieval_config, config_registry)
    cache_config = providers.Callable(_get_cache_config, config_registry)
    datasource_config = providers.Callable(_get_datasource_config, config_registry)
    tool_registry_config = providers.Callable(_get_tool_registry_config, config_registry)
    bank_global_config = providers.Callable(_get_bank_global_config, config_registry)

    # ---------- Infrastructure ----------
    redis_manager = providers.Singleton(_create_redis_manager, datasource_config)
    db_manager = providers.Singleton(_create_database_manager, datasource_config)

    # ---------- Cache ----------
    cache_factory = providers.Singleton(_create_cache_factory, cache_config, redis_manager)

    rag_cache = providers.Singleton(
        cache_factory.provided.create.call(namespace=CacheNamespace.RAG.value)
    )
    compliance_cache = providers.Singleton(
        cache_factory.provided.create.call(namespace=CacheNamespace.COMPLIANCE.value)
    )
    profile_summary_cache = providers.Singleton(
        cache_factory.provided.create.call(namespace=CacheNamespace.PROFILE_SUMMARY.value)
    )
    interaction_cache = providers.Singleton(
        cache_factory.provided.create.call(namespace=CacheNamespace.RECENT_INTERACTION.value)
    )
    lpr_cache = providers.Singleton(
        cache_factory.provided.create.call(namespace=CacheNamespace.LPR.value)
    )
    suggestion_cooldown_cache = providers.Singleton(
        cache_factory.provided.create.call(namespace=CacheNamespace.SUGGESTION_COOLDOWN.value)
    )

    # ---------- Sequence Generator ----------
    seq_generator = providers.Singleton(_create_seq_generator, redis_manager)

    # ---------- LLM Clients ----------
    creative_llm = providers.Singleton(_create_creative_llm, config_registry)
    precise_llm = providers.Singleton(_create_precise_llm, config_registry)
    local_llm = providers.Singleton(_create_local_llm, config_registry)
    local_creative_llm = providers.Singleton(_create_local_creative_llm, config_registry)

    # ---------- Embedding Services ----------
    embedder = providers.Singleton(_create_embedder, config_registry)
    local_embedder = providers.Singleton(_create_local_embeder, config_registry)

    # ---------- Milvus ----------
    milvus_client = providers.Singleton(_create_milvus_client, config_registry)

    # ---------- Memory System ----------
    vector_store = providers.Singleton(
        _create_vector_store, config_registry, embedder, milvus_client
    )
    cursor_manager = providers.Singleton(_create_cursor_manager, redis_manager)
    memory_store = providers.Singleton(_create_memory_store, vector_store, config_registry, cursor_manager)
    memory_retriever = providers.Singleton(_create_memory_retriever, memory_store, config_registry)

    # ---------- Knowledge Retrieval ----------
    knowledge_engine = providers.Singleton(
        _create_knowledge_engine, milvus_client, local_embedder, config_registry
    )
    query_rewriter = providers.Singleton(_create_query_rewriter, config_registry, local_creative_llm)
    query_filter = providers.Singleton(_create_query_filter, config_registry, precise_llm)
    reranker = providers.Singleton(_create_reranker, config_registry)
    compressor = providers.Singleton(_create_compressor, config_registry,local_llm)
    context_complete = providers.Singleton(_create_retrieval_complete, config_registry, local_llm)
    knowledge_retriever = providers.Singleton(
        _create_knowledge_retriever,
        knowledge_engine, query_rewriter, query_filter, reranker, compressor,
        context_complete, config_registry
    )

    # ---------- Domain Services ----------
    summary_generator = providers.Singleton(_create_summary_generator, local_llm, config_registry)
    sub_summary_generator = providers.Singleton(_create_sub_summary_generator, local_llm, config_registry)
    sentiment_analyzer = providers.Singleton(_create_sentiment_analyzer, local_llm, config_registry)
    evidence_infer = providers.Singleton(_create_evidence_infer, local_llm, config_registry)
    profile_extractor = providers.Singleton(_create_profile_extractor, precise_llm, config_registry)
    profile_gate = providers.Singleton(_create_profile_gate, config_registry)
    message_producer = providers.Singleton(_create_message_producer, redis_manager)
    interaction_log_consumer = providers.Singleton(
        _create_interaction_consumer, redis_manager, memory_store,
        summary_generator, sentiment_analyzer
    )
    sub_interaction_log_consumer = providers.Singleton(
        _create_sub_interaction_consumer, redis_manager, memory_store,
        sub_summary_generator, sentiment_analyzer
    )
    user_profile_consumer = providers.Singleton(
        _create_user_profile_consumer, redis_manager, memory_store,
        evidence_infer, profile_extractor
    )

    # ---------- LPR Service ----------
    lpr_service = providers.Singleton(_create_lpr_service, config_registry, lpr_cache)

    # ---------- Repository ----------
    loan_interest_repository = providers.Factory(
        LoanInterestRepository,
        db_session=None
    )

    # ---------- Tool System ----------
    tool_registry = providers.Singleton(_create_tool_registry, config_registry)
    tool_selector = providers.Singleton(ToolSelector, registry=tool_registry)
    tool_executor = providers.Singleton(_create_tool_executor, tool_registry, None)
    skill_executor = providers.Singleton(_create_skill_executor, tool_registry, None)
    skill_registry = providers.Singleton(SkillRegistry)

    # ---------- Classifiers ----------
    loan_advisor_classifier = _create_loan_advisor_classifier()
    after_loan_classifier = _create_after_loan_classifier()
    risk_assessment_classifier = _create_risk_assessment_classifier()
    suggestion_timing_classifier = providers.Singleton(
        _create_suggestion_timing_classifier, local_llm, config_registry
    )

    # ---------- Agent Nodes ----------
    compliance_prefilter_node = providers.Singleton(
        CompliancePrefilter,
        memory_store=memory_store,
        memory_config=memory_config,
        registry=config_registry,
        llm_client=precise_llm,
        seq_generator=seq_generator
    )

    direct_reply_node = providers.Singleton(
        DirectReplyNode,
        llm_client=local_creative_llm,
        registry=config_registry,
        seq_generator=seq_generator
    )

    memory_retrieve_node = providers.Singleton(
        MemoryRetrieveNode,
        retriever=memory_retriever,
        seq_generator=seq_generator,
        memory_config=memory_config
    )

    # ---------- Subgraphs ----------
    supervisor_graph = providers.Singleton(
        _build_supervisor_graph,
        memory_retriever=memory_retriever,
        seq_generator=seq_generator,
        registry=config_registry,
        llm_client=local_creative_llm,
        memory_config=memory_config,
        knowledge_retrieve=knowledge_retriever
    )

    loan_advisor_graph = providers.Singleton(
        _build_loan_advisor_graph,
        llm_client=local_creative_llm,
        registry=config_registry,
        tool_executor=tool_executor,
        seq_generator=seq_generator,
        tool_selector=tool_selector,
        classifier=loan_advisor_classifier,
        skill_executor=skill_executor,
        skill_selector=skill_registry,
        suggestion_classifier=suggestion_timing_classifier,
        suggestion_cache=suggestion_cooldown_cache,
        db_manager=db_manager,
    )

    risk_assessment_graph = providers.Singleton(
        _build_risk_assessment_graph,
        llm_client=local_creative_llm,
        registry=config_registry,
        tool_executor=tool_executor,
        seq_generator=seq_generator,
        tool_selector=tool_selector,
        classifier=risk_assessment_classifier,
        skill_executor=skill_executor,
        skill_selector=skill_registry,
    )

    after_loan_graph = providers.Singleton(
        _build_after_loan_graph,
        llm_client=local_creative_llm,
        registry=config_registry,
        tool_executor=tool_executor,
        seq_generator=seq_generator,
        tool_selector=tool_selector,
        classifier=after_loan_classifier,
        skill_executor=skill_executor,
        skill_selector=skill_registry,
    )

    # ---------- Other Nodes ----------
    result_aggregator_node = providers.Singleton(ResultAggregatorAgent)
    human_handoff_notify_node = providers.Singleton(HumanHandoffResponseNode)
    human_handoff_interrupt_node = providers.Singleton(_build_human_handoff_interrupt_node, redis_manager)

    extract_profile_node = providers.Singleton(
        _build_extract_profile_node,
        memory_store=memory_store,
        profile_gate=profile_gate,
        memory_config=memory_config,
        evidence_infer=evidence_infer,
        profile_extractor=profile_extractor,
        message_producer=message_producer,
        cursor_manager=cursor_manager
    )

    summary_interaction_node = providers.Singleton(
        _build_interaction_node,
        memory_store,
        memory_config,
        summary_generator,
        sentiment_analyzer,
        message_producer,
        cursor_manager
    )
