# 🏦 Loan Advisor Agent — 生产级银行贷款智能助手

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

一个面向金融领域、具备**短期记忆**、**多类型长期记忆**（用户画像、交互轨迹、合规规则）、**自我评估机制**和**生产级工程架构**的 AI Agent 实现。

## ✨ 核心特性

### 🧠 记忆系统
- **短期记忆管理**：基于 LangGraph 的消息窗口与状态持久化（SQLite Checkpointer），支持会话恢复。
- **长期记忆（三种类型）**：
  - **用户画像记忆**：自动从对话中提取收入、职业、偏好等信息，支持**置信度冲突解决**（高置信度覆盖低置信度）、**时间衰减遗忘**、永久记忆保护。
  - **交互轨迹记忆**：每轮对话自动生成摘要并存储，支持按时间戳检索，携带情绪标签与事件类型。
  - **合规规则记忆**：离线导入监管规则（正则表达式），生成回答前进行拦截或追加免责声明。
- **分 Collection 存储**：三种记忆独立存储于 Chroma 不同 Collection，便于差异化管理和未来迁移。
- **Pydantic 强类型模型**：所有记忆元数据均通过 Pydantic 模型校验，杜绝字段错误与数据污染。

### 🤖 Agent 能力
- **自我评估闭环**：生成回答后自动评分，不达标触发重写（可选，当前已简化）。
- **合规拦截节点**：独立节点扫描用户输入，命中高危规则直接阻断，确保回答合规。
- **多用户隔离**：通过 `user_id` 严格隔离数据，支持多用户并发使用。

### 🏗️ 生产级工程
- **分层架构**：`BaseVectorStore` 抽象向量操作，`BaseMemoryStore` 定义记忆契约，业务与存储彻底解耦，可无缝切换至 Milvus。
- **统一常量管理**：所有魔法字符串集中在 `constants.py` 枚举中，IDE 友好，消除拼写错误。
- **健壮错误处理**：LLM 调用与向量操作具备指数退避重试；写入失败进入死信队列；检索失败优雅降级。
- **配置校验**：Pydantic Settings 启动时校验环境变量，杜绝配置错误。
- **类型安全**：全项目使用 Pydantic 模型 + 枚举，配合 IDE 智能提示。

## 🚀 快速开始

### 环境要求
- Python 3.11+
- Git

### 安装

```bash
git clone https://github.com/yourusername/BankLoanAgent.git
cd BankLoanAgent
pip install -r requirements.txt
配置
复制环境变量模板并填入你的 API Key：

bash
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY 等信息（支持 DeepSeek / 通义千问）
运行
bash
make run   # 或 streamlit run app.py
访问 http://localhost:8501 即可开始对话。

📁 项目结构
text
BankLoanAgent/
├── agent/                    # LangGraph 智能体核心
│   ├── state.py              # AgentState 定义
│   ├── nodes.py              # 节点实现（检索、合规、生成、提取、日志）
│   ├── checkpointer.py       # SQLite Checkpointer 封装
│   └── graph.py              # 图编排
├── memory/                   # 长期记忆存储抽象与实现
│   ├── base.py               # BaseMemoryStore 抽象接口
│   ├── vector_store.py       # BaseVectorStore 抽象接口
│   ├── chroma_vector_store.py# Chroma 向量存储实现
│   ├── chroma_store.py       # ChromaMemoryStore（业务策略层）
│   ├── models.py             # 记忆元数据基类
│   └── schemas.py            # 三种记忆专属 Pydantic 模型
├── retriever/                # RAG 检索器抽象
│   ├── base.py               # BaseRetriever 接口
│   └── vector_retriever.py   # VectorRetriever 实现
├── utils/                    # 工具模块
│   ├── llm.py                # LLM 工厂（支持多 Provider，带重试）
│   └── retry.py              # 指数退避重试装饰器
├── scripts/                  # 数据管理脚本
│   ├── seed_profiles.py      # 用户画像测试数据导入
│   └── import_compliance.py  # 合规规则导入
├── data/                     # 数据文件
│   ├── test/                 # 测试数据
│   └── rules/                # 合规规则 JSON
├── config.py                 # Pydantic 配置管理（带校验）
├── constants.py              # 全局枚举常量
├── exceptions.py             # 自定义异常
├── app.py                    # Streamlit 入口
├── requirements.txt
├── Makefile
└── .env.example
🏗️ 架构演进
项目已从最初的单体设计重构为分层、可插拔的生产级架构：














核心设计原则
依赖倒置：上层仅依赖抽象接口，不绑定具体数据库。

单一职责：向量操作、业务策略、检索逻辑分层解耦。

开闭原则：新增向量数据库只需实现 BaseVectorStore，无需修改业务代码。

🔮 扩展路线
v1.0 — 短期记忆 + 三种长期记忆 + 冲突解决 + 衰减遗忘 + 合规拦截

v1.5 — Pydantic 强类型模型 + 分 Collection 存储 + 向量存储抽象层

v2.0 — 接入 Milvus 向量数据库，支持十亿级向量检索

v2.5 — 集成 RAG 知识库，接入银行产品文档与 FAQ

v3.0 — 多 Agent 协作架构（Supervisor + 专业子 Agent）

v3.5 — 工具调用（利率计算器、申请表推送等）

v4.0 — 可观测性（LangSmith / OpenTelemetry 集成）

📄 许可证
本项目采用 MIT License。

注：本项目为个人作品，用于展示 AI Agent 工程化与架构设计能力，非商业用途。