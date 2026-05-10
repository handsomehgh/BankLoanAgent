# app.py
import os

from infra.cache.cache_registry import cache_register
from utils.logging_config import setup_logging, set_log_context

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from modules.retrieval.router.retrieval_rule_router import RuleBaseRetrievalRouter
from infra.cache.cache_factory import CacheFactory

import streamlit as st
import logging
import uuid
from langchain_core.messages import HumanMessage, AIMessage

from config.global_constant.constants import RegistryModules, MemoryType, CacheNamespace
from config.prompts.detect_evidence_prompt import EVIDENCE_PROMPT
from config.prompts.detect_setiment_prompt import DETECT_SENTIMENT_PROMPT
from config.prompts.summary_interaction_prompt import SUMMARY_INTERACTION_PROMPT
from config.registry import ConfigRegistry
from infra.milvus_client import MilvusClientManager
from main import load_config
from modules.agent.constants import StateFields
from modules.agent.graph import build_graph
from modules.memory.memory_business_store.long_term_memory_store import LongTermMemoryStore
from modules.memory.memory_constant.constants import MemoryStatus
from modules.memory.memory_retriever import MemoryVectorRetriever
from modules.memory.memory_vector_store.milvus_memory_vector_store import MilvusMemoryVectorStore
from modules.memory.memory_vector_store.chroma_memory_vector_store import ChromaVectorStore
from modules.memory.memory_utils.profile_gate_util import ProfileGate
from modules.module_services.SummaryGenerator import SummaryGenerator
from modules.module_services.agent_services import AgentServices
from modules.module_services.chat_models import RobustLLM
from modules.module_services.embeddings import RobustEmbeddings
from modules.module_services.evidence_infer import EvidenceTypeInfer
from modules.module_services.profile_extractor import ProfileExtractor
from modules.module_services.sentiment_analyser import SentimentAnalyzer
from modules.retrieval.retrieval_service import RetrievalService
from modules.retrieval.knowledge_vector_store.knowledge_search_engine import KnowledgeSearchEngine
from modules.retrieval.query_rewriter import QueryRewriter
from modules.retrieval.query_filter import QueryFilter
from modules.retrieval.rereanker import Reranker
from modules.retrieval.context_compressor import ContextCompressor
from config.prompts.extract_prompt import EXTRACT_PROMPT
from utils.query_utils.chroma_query_builder import ChromaQueryBuilder
from utils.query_utils.milvus_query_builder import MilvusQueryBuilder
import streamlit.watcher.local_sources_watcher as watcher

watcher.MODULE_IGNORE_LIST = ["transformers"]

# ===================== 加载配置 =====================
try:
    load_config()
except NameError:
    raise RuntimeError("load_config() not defined. Please initialize your configuration manually.")

# ===================== 获取配置 =====================
registry = ConfigRegistry()
llm_config = registry.get_config(RegistryModules.LLM)
print(f"=================={llm_config}")
memory_config = registry.get_config(RegistryModules.MEMORY_SYSTEM)
print(f"=================={memory_config}")
retrieval_config = registry.get_config(RegistryModules.RETRIEVAL)
print(f"=================={retrieval_config}")
cache_config = registry.get_config(RegistryModules.CACHE)
print(f"=================={cache_config}")

# =================== log config ===================
setup_logging(log_level=llm_config.log_level)
logger = logging.getLogger(__name__)

# ============== registry cache manager ===================
factory = CacheFactory(cache_config)
rag_cache = factory.create(CacheNamespace.RAG)
compliance_cache = factory.create(CacheNamespace.COMPLIANCE)
profile_sum_cache = factory.create(CacheNamespace.PROFILE_SUMMARY)
log_cache = factory.create(CacheNamespace.RECENT_INTERACTION)

cache_register(CacheNamespace.RAG, rag_cache)
cache_register(CacheNamespace.COMPLIANCE, compliance_cache)
cache_register(CacheNamespace.PROFILE_SUMMARY, profile_sum_cache)
cache_register(CacheNamespace.RECENT_INTERACTION, log_cache)
# ===================== 创建 LLM 客户端 =====================
creative_llm = RobustLLM(
    temperature=llm_config.creative_temperature,
    api_key=llm_config.deepseek_api_key,
    base_url=llm_config.deepseek_base_url,
    model=llm_config.deepseek_llm_name,
    provider=llm_config.openai_provider
)
precise_llm = RobustLLM(
    temperature=llm_config.precise_temperature,
    api_key=llm_config.deepseek_api_key,
    base_url=llm_config.deepseek_base_url,
    model=llm_config.deepseek_llm_name,
    provider=llm_config.openai_provider
)

# ===================== 创建 Embedding 客户端 =====================
embedder = RobustEmbeddings(
    api_key=llm_config.alibaba_api_key,
    model_name=llm_config.alibaba_emb_name,
    backup_model_name=llm_config.alibaba_emb_backup,
    dimensions=llm_config.dimension
)

# ===================== 初始化记忆系统 =====================
if memory_config.vector_backend == "chroma":
    query_builder = ChromaQueryBuilder()
    vector_store = ChromaVectorStore(persist_dir=memory_config.chroma_persist_dir)
else:
    query_builder = MilvusQueryBuilder()
    milvus_client = MilvusClientManager(retrieval_config.milvus_uri)
    vector_store = MilvusMemoryVectorStore(
        milvus_client=milvus_client, embed=embedder, config=memory_config
    )

st.set_page_config(page_title="银行贷款助手", page_icon="🏦")
st.title("🏦 银行贷款顾问助手")

# ===================== 会话状态初始化 =====================
if "memory_store" not in st.session_state:
    try:
        st.session_state.memory_store = LongTermMemoryStore(
            vector_store=vector_store, config=memory_config
        )
    except Exception as e:
        st.error(f"记忆存储初始化失败: {e}")
        st.stop()

if "memory_retriever" not in st.session_state:
    st.session_state.memory_retriever = MemoryVectorRetriever(
        st.session_state.memory_store, memory_config
    )

if "user_id" not in st.session_state:
    st.session_state.user_id = "test_user_001"
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "agent" not in st.session_state:
    try:
        # ---- 构建知识检索服务 ----
        knowledge_client = MilvusClientManager(retrieval_config.milvus_uri)
        knowledge_engine = KnowledgeSearchEngine(
            knowledge_client, embedder, retrieval_config
        )
        rewriter = QueryRewriter(retrieval_config.rewriter, llm_client=creative_llm)
        query_filter = QueryFilter(retrieval_config.filter, llm_client=precise_llm)
        reranker = Reranker(retrieval_config.reranker)
        compressor = ContextCompressor(retrieval_config.compressor)

        router = RuleBaseRetrievalRouter(retrieval_config.retrieval_routing.rule_based)

        cache_factory = CacheFactory(cache_config)
        rag_cache_manager = cache_factory.create(CacheNamespace.RAG)

        knowledge_retriever = RetrievalService(
            engine=knowledge_engine,
            rewriter=rewriter,
            filter=query_filter,
            reranker=reranker,
            compressor=compressor,
            config=retrieval_config,
            cache_manager=rag_cache_manager,
            retrieve_router=router
        )

        # ---- 创建领域服务 ----
        summary_generator = SummaryGenerator(
            llm_client=creative_llm,
            prompt=SUMMARY_INTERACTION_PROMPT,
            max_summary_length=memory_config.max_summary_length,
            max_interaction_length=memory_config.interaction_log_max_length

        )

        sentiment_analyzer = SentimentAnalyzer(
            llm_client=precise_llm,
            strong_keywords=memory_config.sentiment_rules.strong_keywords,
            prompt=DETECT_SENTIMENT_PROMPT
        )

        evidence_infer = EvidenceTypeInfer(
            llm_client=precise_llm,
            strong_keywords=memory_config.evidence_rules.strong_keywords,
            prompt=EVIDENCE_PROMPT
        )

        profile_extractor = ProfileExtractor(
            llm_client=precise_llm,
            extract_prompt=EXTRACT_PROMPT
        )
        profile_gate = ProfileGate(memory_config.memory_gate)

        # ---- 组装服务容器 ----
        services = AgentServices(
            creative_llm=creative_llm,
            precise_llm=precise_llm,
            memory_store=st.session_state.memory_store,
            memory_retriever=st.session_state.memory_retriever,
            knowledge_retriever=knowledge_retriever,
            summary_generator=summary_generator,
            sentiment_analyzer=sentiment_analyzer,
            evidence_infer=evidence_infer,
            profile_extractor=profile_extractor,
            profile_gate=profile_gate,
            registry=ConfigRegistry()
        )

        # ---- 构建 Agent 图 ----
        st.session_state.agent = build_graph(services)
    except Exception as e:
        st.error(f"Agent 初始化失败: {e}")
        st.stop()

# ==================== 侧边栏 ====================
with st.sidebar:
    st.subheader("👤 用户管理")
    new_user = st.text_input("用户ID", value=st.session_state.user_id)
    if new_user != st.session_state.user_id:
        st.session_state.user_id = new_user
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.caption(f"会话ID: `{st.session_state.thread_id[:8]}...`")
    if st.button("🔄 新建会话"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.subheader("📝 长期画像记忆")

    try:
        profile_memories = st.session_state.memory_store.get_all_user_profile_memories(
            user_id=st.session_state.user_id,
            status=MemoryStatus.ACTIVE
        )
    except Exception as e:
        logger.error(f"获取用户画像失败: {e}")
        profile_memories = []

    if profile_memories:
        for mem in profile_memories:
            content = mem.get("content", "")
            metadata = mem.get("metadata", {})
            with st.expander(f"{content[:40]}...", expanded=False):
                st.write(content)
                st.caption(
                    f"置信度: {metadata.get('confidence', 'N/A')} | "
                    f"实体: {metadata.get('entity_key', 'N/A')}"
                )
    else:
        st.info("暂无画像记忆，开始对话后自动提取。")

    if st.button("🧹 执行遗忘清理", type="secondary"):
        try:
            count = st.session_state.memory_store.apply_forgetting(
                memory_type=MemoryType.USER_PROFILE,
                user_id=st.session_state.user_id
            )
            st.success(f"已遗忘 {count} 条画像记忆")
        except Exception as e:
            st.error(f"遗忘清理失败: {e}")
        st.rerun()

    if st.button("🗑️ 清空当前用户记忆", type="secondary"):
        try:
            success = st.session_state.memory_store.delete_user_memories(
                user_id=st.session_state.user_id,
                memory_type=None
            )
            if success:
                st.success("已清空")
            else:
                st.error("清空失败")
        except Exception as e:
            st.error(f"清空失败: {e}")
        st.rerun()

# ==================== 加载并显示对话历史 ====================
config_dict = {"configurable": {"thread_id": st.session_state.thread_id}}
try:
    current_state = st.session_state.agent.get_state(config_dict)
    if current_state and current_state.values:
        messages = current_state.values.get("messages", [])
    else:
        messages = []
except Exception:
    messages = []

for msg in messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# ==================== 对话逻辑 ====================
if prompt := st.chat_input("请输入您的问题..."):
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            input_state = {
                "messages": [HumanMessage(content=prompt)],
                "user_id": st.session_state.user_id,
            }

            set_log_context(
                user_id=st.session_state.user_id,
                thread_id=st.session_state.thread_id,
            )

            try:
                result = st.session_state.agent.invoke(
                    input_state,
                    config={"configurable": {"thread_id": st.session_state.thread_id}}
                )
                assistant_reply = result["messages"][-1].content
                st.write(assistant_reply)

                # 展示检索到的知识片段
                knowledge_list = result.get(StateFields.RETRIEVED_CONTEXT, {}).get(
                    MemoryType.BUSINESS_KNOWLEDGE, []
                )
                if knowledge_list:
                    with st.expander("🔍 查看本次检索到的知识片段（共 {} 条）".format(len(knowledge_list))):
                        for i, item in enumerate(knowledge_list, 1):
                            source = f"{item.source_type.value if hasattr(item.source_type, 'value') else item.source_type}"
                            product = f" - {item.product_type}" if item.product_type else ""
                            confidence = getattr(item, "confidence", None)
                            header = f"**{i}. [{source}{product}]**"
                            if confidence is not None:
                                header += f" | 置信度: {confidence:.2f}"
                            st.markdown(header)
                            st.text(item.text)
                            st.divider()
                else:
                    st.caption("本次对话未检索到相关业务知识。")

                if result.get("error"):
                    st.caption(f"⚠️ 处理过程中出现非致命错误: {result['error']}")
                if result.get("profile_updated"):
                    st.caption("✅ 已更新长期画像记忆")
            except Exception as e:
                logger.exception("Agent invocation failed")
                st.error(f"系统错误: {e}")
    st.rerun()
