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
from modules.memory.memory_utils.cursor_manager import CursorManager
from modules.module_services.classifier.after_loan_classifier import AfterLoanClassifier
from modules.module_services.classifier.loan_advisor_classifier import LoanAdvisorClassifier
from modules.module_services.classifier.risk_assessment_classifier import RiskAssessmentClassifier
from modules.module_services.lpr_data_service import LPRDataService
from modules.retrieval.context_complete import ContextComplete
from modules.skills.skill_executor import SkillExecutor
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
from modules.retrieval.context_compressor import ContextCompressor
from modules.retrieval.knowledge_vector_store.knowledge_search_engine import KnowledgeSearchEngine
from modules.retrieval.query_filter import QueryFilter
from modules.retrieval.query_rewriter import QueryRewriter
from modules.retrieval.rereanker import Reranker
from modules.retrieval.retrieval_service import RetrievalService
from modules.retrieval.router.retrieval_rule_router import RuleBaseRetrievalRouter
from modules.tools.tool_selector import ToolSelector
from utils.serialize_utils.seq_generator import SequenceGenerator

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------- Factory Function ----------
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


# ---------- 工厂函数：创建带配置依赖的服务 ----------
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


def _create_embedder(registry: ConfigRegistry):
    cfg = registry.get_config(RegistryModules.LLM)
    return RobustEmbeddings(
        api_key=cfg.alibaba_api_key,
        model_name=cfg.alibaba_emb_name,
        backup_model_name=cfg.alibaba_emb_backup,
        dimensions=cfg.dimension
    )

def _create_local_embeder(registry: ConfigRegistry):
    cfg = registry.get_config(RegistryModules.LLM)
    return RobustLocalEmbeder(
        base_url=os.getenv("EMBEDDING_API_URL", cfg.loan_embeder_url),
        model_name=cfg.loan_embeder_name,
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


def _create_compressor(registry):
    cfg = registry.get_config(RegistryModules.RETRIEVAL)
    return ContextCompressor(config=cfg.compressor)


def _create_retrieval_complete(registry,llm_client):
    cfg = registry.get_config(RegistryModules.RETRIEVAL)
    return ContextComplete(cfg,llm_client)

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
    return InteractionLogConsumer(StreamName.INTERACTION_LOG.value,StreamName.INTERACTION_LOG.value,redis_manager, memory_store, summary_generator, sentiment_analyzer)

def _create_sub_interaction_consumer(redis_manager, memory_store, summary_generator, sentiment_analyzer):
    return InteractionLogConsumer(StreamName.SUB_INTERACTION.value,StreamName.SUB_INTERACTION.value,redis_manager, memory_store, summary_generator, sentiment_analyzer)


def _create_lpr_service(registry, cache):
    interest_config = registry.get_config(RegistryModules.BANK_GLOBAL_CONFIG)
    return LPRDataService(config=interest_config, cache=cache)


def _create_tool_registry(registry: ConfigRegistry):
    cfg = registry.get_config(RegistryModules.TOOL_REGISTRY)
    reg = ToolRegistry(cfg)
    reg.scan_and_register()
    reg.validate_against_config()
    return reg


def _create_tool_executor(tool_registry, audit_logger,db_manager):
    return ToolExecutor(registry=tool_registry, audit_logger=audit_logger,session_factory=db_manager.session_factory)


def _build_supervisor_graph(memory_retriever, seq_generator, registry, llm_client, memory_config, knowledge_retrieve):
    agent = SupervisorAgent(memory_retriever, seq_generator, registry, llm_client, memory_config, knowledge_retrieve)
    return agent.build_graph()


def _build_loan_advisor_graph(llm_client, registry, tool_executor, seq_generator,tool_selector,classifier):
    agent = LoanAdvisorAgent(
        llm_client=llm_client,
        registry=registry,
        tool_executor=tool_executor,
        seq_generator=seq_generator,
        tool_selector=tool_selector,
        classifier = classifier

    )
    return agent.build_graph()


def _build_risk_assessment_graph(llm_client, registry, tool_executor, seq_generator,tool_selector,classifier):
    agent = RiskAssessmentAgent(llm_client, registry, tool_executor, seq_generator,tool_selector,classifier)
    return agent.build_graph()


def _build_after_loan_graph(llm_client, registry, tool_executor, seq_generator,tool_selector,classifier):
    agent = AfterLoanAgent(llm_client, registry, tool_executor, seq_generator,tool_selector,classifier)
    return agent.build_graph()


def _build_human_handoff_interrupt_node(redis_manager):
    return HumanHandoffInterruptNode(redis_manager)

def _build_extract_profile_node(memory_store,profile_gate,memory_config,evidence_infer, profile_extractor,message_producer):
    return ExtractProfileNode(memory_store,profile_gate,memory_config,evidence_infer,profile_extractor,message_producer)

def _build_interaction_node(memory_store, memory_config, summary_generator, sentiment_analyzer, message_producer):
    return SummaryInteractionNode(memory_store, memory_config, summary_generator, sentiment_analyzer, message_producer)

def _register_skills(tool_registry, skill_executor):
    from config.skills_loader import load_skill_configs
    from modules.skills.skill_factory import create_tool_from_skill
    import logging
    logger = logging.getLogger(__name__)

    skill_configs = load_skill_configs(PROJECT_ROOT / "config/skills")
    logger.info("Loaded %d skill configs", len(skill_configs))
    for cfg in skill_configs:
        tool = create_tool_from_skill(cfg, skill_executor)
        tool_registry.register(tool)
        logger.info("Registered skill: %s v%s", cfg.name, cfg.version)

def _create_database_manager(datasource_config):
    return DatabaseManager(datasource_config.mysql)

class ApplicationContainer(containers.DeclarativeContainer):
    """Main Application Container"""

    # Config Register Center
    config_registry = providers.Singleton(ConfigRegistry)

    # Configuration
    llm_config = providers.Callable(_get_llm_config, config_registry)
    memory_config = providers.Callable(_get_memory_config, config_registry)
    retrieval_config = providers.Callable(_get_retrieval_config, config_registry)
    cache_config = providers.Callable(_get_cache_config, config_registry)
    datasource_config = providers.Callable(_get_datasource_config, config_registry)
    tool_registry_config = providers.Callable(_get_tool_registry_config, config_registry)
    bank_global_config = providers.Callable(_get_bank_global_config, config_registry)

    # Redis Manager
    redis_manager = providers.Singleton(_create_redis_manager, datasource_config)

    #mysql manager
    db_manager = providers.Singleton(_create_database_manager, datasource_config)

    # Cache Factory
    cache_factory = providers.Singleton(_create_cache_factory, cache_config, redis_manager)

    # Cache Manager
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

    # Sequence Generator
    seq_generator = providers.Singleton(_create_seq_generator, redis_manager)

    # LLM client
    creative_llm = providers.Singleton(_create_creative_llm, config_registry)
    precise_llm = providers.Singleton(_create_precise_llm, config_registry)

    # Embedding Service
    embedder = providers.Singleton(_create_embedder, config_registry)
    local_embedder = providers.Singleton(_create_local_embeder,config_registry)

    # Milvus client
    milvus_client = providers.Singleton(_create_milvus_client, config_registry)

    # Memory Vector Store
    vector_store = providers.Singleton(
        _create_vector_store, config_registry, embedder, milvus_client
    )

    # cursor manager
    cursor_manager = providers.Singleton(
        _create_cursor_manager,
        redis_manager
    )

    # Memory Store
    memory_store = providers.Singleton(_create_memory_store, vector_store, config_registry, cursor_manager)

    # Memory Retriever
    memory_retriever = providers.Singleton(_create_memory_retriever, memory_store, config_registry)

    # Knowledge Engine
    knowledge_engine = providers.Singleton(
        _create_knowledge_engine, milvus_client, local_embedder, config_registry
    )

    # Retrieve Component
    query_rewriter = providers.Singleton(_create_query_rewriter, config_registry, creative_llm)
    query_filter = providers.Singleton(_create_query_filter, config_registry, precise_llm)
    reranker = providers.Singleton(_create_reranker, config_registry)
    compressor = providers.Singleton(_create_compressor, config_registry)
    context_complete = providers.Singleton(_create_retrieval_complete, config_registry,precise_llm)

    # Knowledge Retrieve
    knowledge_retriever = providers.Singleton(
        _create_knowledge_retriever,
        knowledge_engine, query_rewriter, query_filter, reranker, compressor,
        context_complete, config_registry
    )

    # Domain Service
    summary_generator = providers.Singleton(_create_summary_generator, creative_llm, config_registry)
    sub_summary_generator = providers.Singleton(_create_sub_summary_generator,creative_llm,config_registry)
    sentiment_analyzer = providers.Singleton(_create_sentiment_analyzer, precise_llm, config_registry)
    evidence_infer = providers.Singleton(_create_evidence_infer, precise_llm, config_registry)
    profile_extractor = providers.Singleton(_create_profile_extractor, precise_llm, config_registry)
    profile_gate = providers.Singleton(_create_profile_gate, config_registry)
    message_producer = providers.Singleton(_create_message_producer, redis_manager)
    interaction_log_consumer = providers.Singleton(_create_interaction_consumer, redis_manager, memory_store,
                                                   summary_generator, sentiment_analyzer)
    sub_interaction_log_consumer = providers.Singleton(_create_sub_interaction_consumer, redis_manager, memory_store,
                                                   sub_summary_generator, sentiment_analyzer)

    # lpr service
    lpr_service = providers.Singleton(_create_lpr_service, config_registry, lpr_cache)

    # Tool System
    tool_registry = providers.Singleton(_create_tool_registry, config_registry)
    tool_selector = providers.Singleton(
        ToolSelector,
        registry=tool_registry
    )
    tool_executor = providers.Singleton(_create_tool_executor, tool_registry, None,db_manager)
    skill_executor = providers.Singleton(SkillExecutor, tool_executor=tool_executor)

    skills_init = providers.Resource(
        _register_skills,
        tool_registry=tool_registry,
        skill_executor=skill_executor
    )

    def _init_skills(self):
        """初始化Skills注册"""
        self._register_skills()

    loan_advisor_classifier = _create_loan_advisor_classifier()
    after_loan_classifier = _create_after_loan_classifier()
    risk_assessment_classifier = _create_risk_assessment_classifier()
    # pre compliance filter
    compliance_prefilter_node = providers.Singleton(
        CompliancePrefilter,
        memory_store=memory_store,
        memory_config=memory_config,
        registry=config_registry,
        llm_client=precise_llm,
        seq_generator=seq_generator
    )

    #direct reply
    direct_reply_node = providers.Singleton(
        DirectReplyNode,
        llm_client=creative_llm,
        registry=config_registry,
        seq_generator=seq_generator
    )

    #memory retrieve node
    memory_retrieve_node = providers.Singleton(
        MemoryRetrieveNode,
        retriever=memory_retriever,
        seq_generator=seq_generator,
        memory_config=memory_config
    )

    # supervisor graph
    supervisor_graph = providers.Singleton(
        _build_supervisor_graph,
        memory_retriever=memory_retriever,
        seq_generator=seq_generator,
        registry=config_registry,
        llm_client=creative_llm,
        memory_config=memory_config,
        knowledge_retrieve=knowledge_retriever
    )

    # loanAdvisor graph
    loan_advisor_graph = providers.Singleton(
        _build_loan_advisor_graph,
        llm_client=creative_llm,
        registry=config_registry,
        tool_executor=tool_executor,
        seq_generator=seq_generator,
        tool_selector=tool_selector,
        classifier = loan_advisor_classifier
    )

    # risk assessment graph
    risk_assessment_graph = providers.Singleton(
        _build_risk_assessment_graph,
        llm_client=creative_llm,
        registry=config_registry,
        tool_executor=tool_executor,
        seq_generator=seq_generator,
        tool_selector=tool_selector,
        classifier=risk_assessment_classifier
    )

    # after loan graph
    after_loan_graph = providers.Singleton(
        _build_after_loan_graph,
        llm_client=creative_llm,
        registry=config_registry,
        tool_executor=tool_executor,
        seq_generator=seq_generator,
        tool_selector=tool_selector,
        classifier = after_loan_classifier
    )

    #result aggregator node
    result_aggregator_node = providers.Singleton(ResultAggregatorAgent)

    # human handoff notify node
    human_handoff_notify_node = providers.Singleton(HumanHandoffResponseNode)

    #huamn handoff interrupt node
    human_handoff_interrupt_node=providers.Singleton(_build_human_handoff_interrupt_node,redis_manager)

    # extract profile node
    extract_profile_node = providers.Singleton(
        _build_extract_profile_node,
        memory_store=memory_store,
        profile_gate=profile_gate,
        memory_config=memory_config,
        evidence_infer=evidence_infer,
        profile_extractor=profile_extractor,
        message_producer=message_producer
    )

    # interaction log
    summary_interaction_node = providers.Singleton(
        _build_interaction_node,
        memory_store,
        memory_config,
        summary_generator,
        sentiment_analyzer,
        message_producer
    )