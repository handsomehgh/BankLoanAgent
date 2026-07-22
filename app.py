import logging
import time
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.errors import GraphInterrupt

from config.bootstrap import get_bootstrapper
from config.global_constant.constants import MemoryType
from modules.memory.memory_constant.constants import MemoryStatus
from modules.agent.constants import StateFields
from utils.logging_config import set_log_context
import streamlit.watcher.local_sources_watcher as watcher

watcher.MODULE_IGNORE_LIST = ["transformers"]

# ==================== 初始化 ====================
boot = get_bootstrapper()
runtime = boot.start()

agent = runtime.graph
memory_store = runtime.memory_store

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
if "processing" not in st.session_state:
    st.session_state.processing = False

# 侧边栏禁用的条件：处理中 或 等待人工
sidebar_disabled = st.session_state.processing or st.session_state.waiting_for_human

# ==================== 侧边栏 ====================
with st.sidebar:
    st.subheader("👤 用户管理")
    new_user = st.text_input(
        "用户ID",
        value=st.session_state.user_id,
        disabled=sidebar_disabled,
        key="sidebar_user_id_input"
    )
    if not sidebar_disabled and new_user != st.session_state.user_id:
        st.session_state.user_id = new_user
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.caption(f"会话ID: `{st.session_state.thread_id[:8]}...`")
    if st.button("🔄 新建会话", disabled=sidebar_disabled):
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

    if st.button("🧹 执行遗忘清理", type="secondary", disabled=sidebar_disabled):
        try:
            count = memory_store.apply_forgetting(
                memory_type=MemoryType.USER_PROFILE,
                user_id=st.session_state.user_id
            )
            st.success(f"已遗忘 {count} 条画像记忆")
        except Exception as e:
            st.error(f"遗忘清理失败: {e}")
        st.rerun()

    if st.button("🗑️ 清空当前用户记忆", type="secondary", disabled=sidebar_disabled):
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
    last_ai_msgs = [m for m in messages if isinstance(m, AIMessage)]
    if last_ai_msgs:
        last_content = last_ai_msgs[-1].content
        if last_content != "您的问题已转接至人工客服，请稍候。":
            st.session_state.waiting_for_human = False
            st.rerun()

# ==================== 对话逻辑 ====================
if st.session_state.waiting_for_human:
    st.warning("您的请求正在等待人工客服处理中，请稍候...")
    st.stop()
elif st.session_state.processing:
    st.info("正在处理您的上一个请求，请稍候…")
    st.stop()
else:
    prompt = st.chat_input("请输入您的问题...")
    if prompt:
        st.session_state.processing = True
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            response_placeholder = st.empty()

            trace_id = str(uuid.uuid4())
            input_state = {
                "messages": [HumanMessage(content=prompt)],
                "user_id": st.session_state.user_id,
                "trace_id": trace_id
            }
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            set_log_context(
                user_id=st.session_state.user_id,
                thread_id=st.session_state.thread_id,
                trace_id=trace_id,
            )

            handoff_occurred = False

            try:
                # 同步流式执行
                for chunk in agent.stream(input_state, config, stream_mode="updates"):
                    if "__interrupt__" in chunk:
                        logger.info("Human-in-the-Loop 中断挂起")
                        st.session_state.waiting_for_human = True
                        handoff_occurred = True
                        # 从当前已获取的状态中提取最后一条助手消息
                        try:
                            current = agent.get_state(config)
                            if current and current.values:
                                msgs = current.values.get("messages", [])
                                if msgs and isinstance(msgs[-1], AIMessage):
                                    response_placeholder.markdown(msgs[-1].content)
                                else:
                                    response_placeholder.markdown("您的问题已转接至人工客服，请稍候。")
                            else:
                                response_placeholder.markdown("您的问题已转接至人工客服，请稍候。")
                        except Exception:
                            response_placeholder.markdown("您的问题已转接至人工客服，请稍候。")
                        break

                # 如果未发生转接，获取最终回复
                if not handoff_occurred:
                    final_state = agent.get_state(config)
                    if final_state and final_state.values:
                        final_messages = final_state.values.get("messages", [])
                        if final_messages:
                            last_msg = final_messages[-1]
                            if isinstance(last_msg, AIMessage):
                                final_response = last_msg.content
                                # 前端模拟打字机
                                def char_gen(text):
                                    for c in text:
                                        yield c
                                        time.sleep(0.02)
                                st.write_stream(char_gen(final_response))
                            else:
                                response_placeholder.markdown("系统无响应")
                        else:
                            response_placeholder.markdown("系统无响应")
                    else:
                        response_placeholder.markdown("系统无响应")

            except GraphInterrupt as e:
                logger.info("Human-in-the-Loop 中断挂起 (GraphInterrupt)")
                st.session_state.waiting_for_human = True
                handoff_occurred = True
                try:
                    current = agent.get_state(config)
                    if current and current.values:
                        msgs = current.values.get("messages", [])
                        if msgs and isinstance(msgs[-1], AIMessage):
                            response_placeholder.markdown(msgs[-1].content)
                        else:
                            response_placeholder.markdown("您的问题已转接至人工客服，请稍候。")
                    else:
                        response_placeholder.markdown("您的问题已转接至人工客服，请稍候。")
                except Exception:
                    response_placeholder.markdown("您的问题已转接至人工客服，请稍候。")
            except Exception as e:
                logger.exception("Agent invocation failed")
                st.error(f"系统错误: {e}")

            # 显示非致命错误和画像更新提示
            try:
                final_state = agent.get_state(config)
                if final_state and final_state.values:
                    if final_state.values.get("error"):
                        st.caption(f"⚠️ 处理过程中出现非致命错误: {final_state.values['error']}")
                    if final_state.values.get("profile_updated"):
                        st.caption("✅ 已更新长期画像记忆")
            except Exception:
                pass

        st.session_state.processing = False
        st.rerun()