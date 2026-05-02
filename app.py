# author hgh
# version 1.0
# author hgh
# version 1.0
import streamlit as st
import logging
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from agent.graph import build_graph
from config.settings import agentConfig
from query.chroma_query_builder import ChromaQueryBuilder
from query.milvus_query_builder import MilvusQueryBuilder
from memory.long_term_memory_store import LongTermMemoryStore
from memory.memory_vector_store.chroma_vector_store import ChromaVectorStore
from memory.memory_vector_store.milvus_vector_store import MilvusVectorStore
from retriever.memory_retriever import VectorRetriever
from config.constants import MemoryType

import streamlit.watcher.local_sources_watcher as watcher
watcher.MODULE_IGNORE_LIST = ["transformers"]  # 忽略 transformers 模块的路径检查

logging.basicConfig(level=agentConfig.log_level)
logger = logging.getLogger(__name__)


if agentConfig.vector_backend == "chroma":
    query_builder = ChromaQueryBuilder()
    vector_store = ChromaVectorStore(persist_dir=agentConfig.chroma_persist_dir)
else:
    # 预留 Milvus
    query_builder = MilvusQueryBuilder()
    vector_store = MilvusVectorStore(uri=agentConfig.milvus_uri)

st.set_page_config(page_title="银行贷款助手", page_icon="🏦")
st.title("🏦 银行贷款顾问助手")

# ==================== 初始化 ====================
if "memory_store" not in st.session_state:
    try:
        st.session_state.memory_store = LongTermMemoryStore(vector_store=vector_store)
    except Exception as e:
        st.error(f"记忆存储初始化失败: {e}")
        st.stop()

if "retriever" not in st.session_state:
    st.session_state.retriever = VectorRetriever(st.session_state.memory_store)

if "user_id" not in st.session_state:
    st.session_state.user_id = "test_user_004"
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "agent" not in st.session_state:
    try:
        st.session_state.agent = build_graph(
            st.session_state.memory_store,
            st.session_state.retriever
        )
    except Exception as e:
        st.error(f"Agent 初始化失败: {e}")
        st.stop()

# ==================== 侧边栏 ====================
with st.sidebar:
    st.subheader("👤 用户管理")
    new_user = st.text_input("用户ID", value=st.session_state.user_id)
    if new_user != st.session_state.user_id:
        st.session_state.user_id = new_user
        st.session_state.thread_id = str(uuid.uuid4())  # 切换用户时新建会话
        st.rerun()

    st.caption(f"会话ID: `{st.session_state.thread_id[:8]}...`")
    if st.button("🔄 新建会话"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.subheader("📝 长期画像记忆")

    # 获取当前用户的所有用户画像记忆（仅 USER_PROFILE 类型）
    try:
        # 使用新的抽象方法获取用户画像
        profile_memories = st.session_state.memory_store.get_all_user_profile_memories(
            user_id=st.session_state.user_id,
            status="active"
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

    # 遗忘清理按钮（仅对用户画像生效）
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

    # 清空记忆按钮（清空该用户的所有记忆，包括画像和交互日志）
    if st.button("🗑️ 清空当前用户记忆", type="secondary"):
        try:
            # 可选择只清空某类型，此处清空全部用户相关记忆
            success = st.session_state.memory_store.delete_user_memories(
                user_id=st.session_state.user_id,
                memory_type=None  # None 表示清空所有类型
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
            try:
                result = st.session_state.agent.invoke(
                    input_state,
                    config={"configurable": {"thread_id": st.session_state.thread_id}}
                )
                assistant_reply = result["messages"][-1].content
                st.write(assistant_reply)

                if result.get("error"):
                    st.caption(f"⚠️ 处理过程中出现非致命错误: {result['error']}")

                # 注意：评估节点已移除，eval_score 不再存在，可删除或保留为占位
                if result.get("profile_updated"):
                    st.caption("✅ 已更新长期画像记忆")
            except Exception as e:
                st.error(f"系统错误: {e}")
                logger.exception("Agent invocation failed")

    st.rerun()