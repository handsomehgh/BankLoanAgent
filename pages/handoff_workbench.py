"""
Manual Agent Workstation
访问地址：http://localhost:8501/handoff_workbench
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.bootstrap import get_bootstrapper
from modules.agent.human_handoff.handoff_task_manager import HandoffTaskManager

st.set_page_config(page_title="人工坐席工作台", page_icon="🎧")

# 隐藏 Streamlit 默认 header
hide_header_style = """
    <style>
    header[data-testid="stHeader"] { display: none; }
    </style>
"""
st.markdown(hide_header_style, unsafe_allow_html=True)

st.title("🎧 人工坐席工作台")

boot = get_bootstrapper()
boot.start()  # 如果已启动则无操作
container = boot.container
graph = boot.graph
redis_manager = container.redis_manager()
task_manager = HandoffTaskManager(graph, redis_manager)


if st.button("🔄 刷新工单列表"):
    st.rerun()

tasks = task_manager.get_pending_tasks()

if not tasks:
    st.info("当前没有待处理的工单")
else:
    st.success(f"待处理工单：{len(tasks)} 个")
    for task in tasks:
        with st.expander(f"工单 - {task['user_id']} ({task['timestamp'][:19]})"):
            st.text("转接摘要：")
            st.text(task["handoff_summary"])

            col1, col2 = st.columns(2)
            with col1:
                action = st.radio("操作", ["reply", "close"], key=f"action_{task['thread_id']}")
            with col2:
                content = ""
                if action == "reply":
                    content = st.text_area("回复内容", key=f"content_{task['thread_id']}")

            if st.button("提交", key=f"submit_{task['thread_id']}"):
                task_manager.recover_task(task["thread_id"], action, content)
                st.success("工单处理完成")
                st.rerun()