# author hgh
# version 1.0
import streamlit as st
import logging
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from agent.graph import build_graph
from memory.chroma_store import ChromaMemoryStore
from retriever.vector_retriever import VectorRetriever
from config import config

logging.basicConfig(level=config.log_level)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="银行贷款助手", page_icon="🏦")
st.title("🏦 银行贷款顾问助手（生产级错误处理）")

# 初始化
if "memory_store" not in st.session_state:
    try:
        st.session_state.memory_store = ChromaMemoryStore(persist_dir=config.chroma_persist_dir)
    except Exception as e:
        st.error(f"记忆存储初始化失败: {e}")
        st.stop()

if "retriever" not in st.session_state:
    st.session_state.retriever = VectorRetriever(st.session_state.memory_store)

if "user_id" not in st.session_state:
    st.session_state.user_id = "default_user"
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "agent" not in st.session_state:
    try:
        st.session_state.agent = build_graph(st.session_state.memory_store, st.session_state.retriever)
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

    # 获取当前用户的所有 active 记忆
    try:
        all_memories = []
        results = st.session_state.memory_store.collection.get(
            where={"user_id": st.session_state.user_id, "status": "active"},
            include=["documents", "metadatas"]
        )
        for i, doc in enumerate(results["documents"]):
            all_memories.append({
                "content": doc,
                "metadata": results["metadatas"][i]
            })
    except Exception:
        all_memories = []

    if all_memories:
        for mem in all_memories:
            with st.expander(f"{mem['content'][:40]}...", expanded=False):
                st.write(mem["content"])
                meta = mem["metadata"]
                st.caption(f"置信度: {meta.get('confidence', 'N/A')} | 实体: {meta.get('entity_key', 'N/A')}")
    else:
        st.info("暂无画像记忆，开始对话后自动提取。")

    # 遗忘清理按钮
    if st.button("🧹 执行遗忘清理", type="secondary"):
        count = st.session_state.memory_store.apply_forgetting(user_id=st.session_state.user_id)
        st.success(f"已遗忘 {count} 条记忆")
        st.rerun()

    # 清空记忆按钮
    if st.button("🗑️ 清空当前用户记忆", type="secondary"):
        st.session_state.memory_store.delete_user_memories(st.session_state.user_id)
        st.success("已清空")
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

# 对话逻辑
if prompt := st.chat_input("请输入您的问题..."):
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            input_state = {"messages": [HumanMessage(content=prompt)], "user_id": st.session_state.user_id}
            try:
                result = st.session_state.agent.invoke(input_state, config={"configurable": {"thread_id": st.session_state.thread_id}})
                assistant_reply = result["messages"][-1].content
                st.write(assistant_reply)
                if result.get("error"):
                    st.caption(f"⚠️ 处理过程中出现非致命错误: {result['error']}")
                if result.get("eval_score"):
                    score = result["eval_score"]
                    color = "green" if score >= 0.8 else "orange" if score >= 0.6 else "red"
                    st.caption(f"自评: :{color}[{score:.2f}]")
                if result.get("profile_updated"):
                    st.caption("✅ 已更新长期画像记忆")
            except Exception as e:
                st.error(f"系统错误: {e}")
                logger.exception("Agent invocation failed")