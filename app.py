# app.py
import logging
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.errors import GraphInterrupt

from config.bootstrap import get_bootstrapper
from config.global_constant.constants import MemoryType
from modules.memory.memory_constant.constants import MemoryStatus
from utils.logging_config import set_log_context
import streamlit.watcher.local_sources_watcher as watcher

watcher.MODULE_IGNORE_LIST = ["transformers"]

# ==================== 初始化 ====================
boot = get_bootstrapper()
boot.start()

container = boot.container
memory_store = boot.memory_store
agent = boot.graph

logger = logging.getLogger(__name__)

# ===================== Streamlit UI =====================
st.set_page_config(page_title="银行贷款助手", page_icon="🏦")
st.title("🏦 银行贷款顾问助手")

# 会话状态初始化
if "user_id" not in st.session_state:
    st.session_state.user_id = "test_user_011"
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "waiting_for_human" not in st.session_state:
    st.session_state.waiting_for_human = False

# ==================== 侧边栏 ====================
with st.sidebar:
    st.subheader("👤 用户管理")
    new_user = st.text_input("用户ID", value=st.session_state.user_id, key="sidebar_user_id_input")
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
        profile_memories = memory_store.get_all_user_profile_memories(
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
            count = memory_store.apply_forgetting(
                memory_type=MemoryType.USER_PROFILE,
                user_id=st.session_state.user_id
            )
            st.success(f"已遗忘 {count} 条画像记忆")
        except Exception as e:
            st.error(f"遗忘清理失败: {e}")
        st.rerun()

    if st.button("🗑️ 清空当前用户记忆", type="secondary"):
        try:
            success = memory_store.delete_user_memories(
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
    current_state = agent.get_state(config_dict)
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

if st.session_state.waiting_for_human:
    # 如果最后一条助手消息不是转接提示，说明人工已回复
    last_ai_msgs = [m for m in messages if isinstance(m, AIMessage)]
    if last_ai_msgs:
        last_content = last_ai_msgs[-1].content
        if last_content != "您的问题已转接至人工客服，请稍候。":
            st.session_state.waiting_for_human = False
            st.rerun()

# ==================== 对话逻辑 ====================
if st.session_state.waiting_for_human:
    st.warning("您的请求正在等待人工客服处理中，请稍候...")
    # 不显示输入框，直接停止继续执行
    st.stop()
else:
    prompt = st.chat_input("请输入您的问题...")
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                trace_id = str(uuid.uuid4())
                input_state = {
                    "messages": [HumanMessage(content=prompt)],
                    "user_id": st.session_state.user_id,
                    "trace_id": trace_id
                }

                set_log_context(
                    user_id=st.session_state.user_id,
                    thread_id=st.session_state.thread_id,
                    trace_id=trace_id,
                )

                config = {"configurable": {"thread_id": st.session_state.thread_id}}

                try:
                    final_state = input_state.copy()
                    for chunk in agent.stream(input_state, config, stream_mode="updates"):
                        if "__interrupt__" in chunk:
                            interrupt_info = chunk["__interrupt__"]
                            logger.info("Human-in-the-Loop 中断挂起, trace_id=%s", trace_id)
                            st.session_state.waiting_for_human = True
                            messages = final_state.get("messages", [])
                            for msg in messages:
                                if isinstance(msg, AIMessage):
                                    st.write(msg.content)
                            st.caption("⏳ 等待人工客服处理中...")
                            break

                        for node_name, node_output in chunk.items():
                            if isinstance(node_output, dict):
                                if "messages" in node_output:
                                    final_state["messages"] = node_output["messages"]
                                for key, value in node_output.items():
                                    if key != "messages":
                                        final_state[key] = value
                            else:
                                logger.warning("Unexpected node output type from %s: %s", node_name, type(node_output))
                    else:
                        messages = final_state.get("messages", [])
                        assistant_reply = messages[-1].content if messages else "系统无响应"
                        st.write(assistant_reply)
                        if final_state.get("error"):
                            st.caption(f"⚠️ 处理过程中出现非致命错误: {final_state['error']}")
                        if final_state.get("profile_updated"):
                            st.caption("✅ 已更新长期画像记忆")
                        if st.session_state.waiting_for_human:
                            st.session_state.waiting_for_human = False

                except GraphInterrupt as e:
                    logger.info("Human-in-the-Loop 中断挂起 (GraphInterrupt), trace_id=%s", trace_id)
                    st.session_state.waiting_for_human = True
                    messages = e.state.get("messages", [])
                    for msg in messages:
                        if isinstance(msg, AIMessage):
                            st.write(msg.content)
                    st.caption("⏳ 等待人工客服处理中...")
                except Exception as e:
                    logger.exception("Agent invocation failed")
                    st.error(f"系统错误: {e}")
        st.rerun()