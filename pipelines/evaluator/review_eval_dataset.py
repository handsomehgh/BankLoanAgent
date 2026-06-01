# author hgh
# version 1.0
# scripts/review_eval_dataset.py
"""
人工校验自动生成的评估数据集
通过 Streamlit 界面逐条审核、修改或删除
输出：data/eval/annotated_qa.jsonl
"""

import streamlit as st
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data/eval"
AUTO_FILE = DATA_DIR / "auto_generated.jsonl"
ANNOTATED_FILE = DATA_DIR / "annotated_qa.jsonl"


def load_jsonl(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


st.set_page_config(page_title="RAG评估数据集校验", page_icon="📋")
st.title("📋 RAG 评估数据集人工校验")

auto_data = load_jsonl(AUTO_FILE)
annotated_data = load_jsonl(ANNOTATED_FILE)

if not auto_data:
    st.warning("未找到自动生成的数据集，请先运行 scripts/generate_eval_dataset.py")
    st.stop()

# 计算进度
total_auto = len(auto_data)
annotated_ids = {item["query"]: item for item in annotated_data}
st.caption(f"自动生成总数：{total_auto}，已确认：{len(annotated_data)}")

# 选择当前审核的样本索引
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

if st.session_state.current_index >= total_auto:
    st.success("所有样本已审核完毕！")
    st.balloons()
    st.stop()

current = auto_data[st.session_state.current_index]

st.subheader(f"样本 #{st.session_state.current_index + 1} / {total_auto}")

# 显示并允许修改 query
query = st.text_area("用户问题", value=current["query"], height=100)

# 显示原始 chunk 内容
st.subheader("相关文档片段（标准答案）")
with st.expander("点击展开文档片段"):
    st.text(current.get("ground_truth_answer", "无")[:2000])

# 操作按钮
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("✅ 确认 / 保存", key="approve"):
        # 更新 query（如果用户修改了）
        current["query"] = query
        # 追加到已标注数据集（去重：如果已存在相同query则覆盖）
        existing = next((item for item in annotated_data if item["query"] == query), None)
        if existing:
            existing.update(current)
        else:
            annotated_data.append(current)
        save_jsonl(annotated_data, ANNOTATED_FILE)
        st.session_state.current_index += 1
        st.rerun()

with col2:
    if st.button("❌ 删除此条", key="delete"):
        st.session_state.current_index += 1
        st.rerun()

with col3:
    if st.button("⏭️ 跳过（保留未处理）", key="skip"):
        st.session_state.current_index += 1
        st.rerun()

st.divider()
st.caption("💡 提示：当问题与文档完全无关、表述不清或答案错位时，请删除；否则可确认或修改问题后保存。")