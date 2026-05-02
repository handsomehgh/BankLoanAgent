# 🏦 Loan Advisor Agent — 生产级银行贷款智能助手

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

一个面向金融领域、具备**增量画像提取**、**多类型长期记忆**（用户画像、交互轨迹、合规规则）、**生产级门控系统**、**合规前置拦截**和**分层可扩展架构**的 AI Agent 实现。

## ✨ 核心特性

### 🧠 记忆系统
- **短期记忆管理**：基于 LangGraph 的消息窗口与状态持久化（SQLite Checkpointer），支持会话恢复。
- **长期记忆（三种类型）**：
  - **用户画像记忆**：基于全局消息序号游标的增量提取，配合**生产级门控**（强信号触发 + 弱信号累积 + 显式指令感知）大幅减少无效 LLM 调用；已知画像脱敏摘要注入，避免重复提取；支持**置信度冲突解决**（高置信度/高证据权重覆盖）、**时间衰减遗忘**、永久记忆保护。
  - **交互轨迹记忆**：交互日志增量游标，防止重复生成摘要；每轮触发仅处理新增对话区间，携带情绪标签与事件类型，按时间戳检索。
  - **合规规则记忆**：带 TTL 缓存的活跃规则检索，支持离线导入监管正则，生成回答前匹配并拦截或追加声明。
- **Pydantic 强类型模型**：所有记忆元数据经校验，杜绝字段错误与数据污染。

### 🛡️ 生产级门控系统
- **三级门控策略**：显式更新指令（无条件触发）→ 强信号正则（金额、手机号、修正词等）→ 弱信号累积（职业、收入、资产等关键词计分，阈值可配置）。
- **配置化**：门控规则外置到 `profile_gate.yaml`，支持运行时调整；加载失败自动回退内置默认规则，确保可用性。
- **可观测**：门控通过日志反馈过滤漏斗，便于持续优化。

### 🤖 Agent 能力
- **合规拦截节点**：独立节点扫描用户输入，命中高危规则直接阻断，确保金融合规。
- **多用户隔离**：通过 `user_id` 严格隔离数据，支持多用户并发。
- **画像完整性引导**：根据业务场景动态判断是否需向用户询问补充信息（如贷款申请时缺失关键字段），引导自然流畅。

### 🏗️ 生产级工程
- **分层抽象**：`BaseVectorStore` 抽象向量操作，`BaseMemoryStore` 定义记忆契约，业务与存储彻底解耦，已实现 Milvus 与 Chroma 双后端。
- **Milvus 深度集成**：支持混合检索（稠密+稀疏）、关键词检索、MMR 多样化检索，动态策略选择。
- **统一常量枚举**：所有魔法字符串集中管理，IDE 友好。
- **健壮错误处理**：LLM 与向量操作装备指数退避重试；写入失败进入死信队列；检索失败优雅降级。
- **配置校验**：Pydantic Settings 启动时校验环境变量。
- **可扩展游标接口**：预留跨会话游标（`get/set_extraction_cursor`），为异步 Worker 阶段奠基。

## 🚀 快速开始

### 环境要求
- Python 3.11+
- Milvus（可选，默认使用 Milvus，也可切换为 Chroma）

### 安装

```bash
git clone https://github.com/yourusername/BankLoanAgent.git
cd BankLoanAgent
pip install -r requirements.txt

配置
复制环境变量模板并填入你的 API Key 与向量存储配置：

bash
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY、MILVUS_URI 等信息
运行
bash
make run   # 或 streamlit run app.py
访问 http://localhost:8501 即可开始对话。

📁 项目结构
text
BankLoanAgent/
├── agent/                        # LangGraph 智能体核心
│   ├── state.py                  # AgentState 定义（含游标）
│   ├── nodes/                    # 节点实现
│   │   ├── retrieve_memory_node.py   # 记忆检索 + 全局消息序号分配
│   │   ├── compliance_guard_node.py  # 合规拦截
│   │   ├── call_model_node.py        # LLM 生成 + 序号分配
│   │   ├── extract_profile_node.py   # 增量画像提取（含三级门控）
│   │   └── log_interaction_node.py   # 增量交互摘要与存储
│   ├── checkpointer.py           # SQLite Checkpointer 封装
│   └── graph.py                  # 图编排
├── memory/                       # 长期记忆存储抽象与实现
│   ├── base_memory_store.py      # BaseMemoryStore 抽象接口
│   ├── long_term_memory_store.py # 业务策略实现（冲突解决、遗忘、脱敏摘要、合规缓存）
│   ├── memory_vector_store/
│   │   ├── base_vector_store.py      # BaseVectorStore 抽象
│   │   ├── milvus_vector_store.py    # Milvus 实现（混合检索、MMR）
│   │   └── chroma_vector_store.py    # Chroma 实现（备选）
│   ├── models/                   # Pydantic 模型与映射器
│   │   ├── memory_base.py
│   │   ├── memory_schema.py     # UserProfile/InteractionLog/ComplianceRule
│   │   └── mappers.py           # 序列化/反序列化器
│   └── classifiers/             # 证据类型、情绪检测
├── retriever/                    # 检索器
│   ├── base.py                   # BaseRetriever 接口
│   └── memory_retriever.py       # VectorRetriever 实现（含相似度过滤）
├── config/                       # 配置与常量
│   ├── settings.py               # Pydantic 配置管理（含环境变量覆盖）
│   ├── constants.py              # 全局枚举（StateFields、MemoryType、ProfileEntityKey等）
│   └── profile_gate.yaml         # 画像提取门控规则（可热调整）
├── exceptions/                   # 自定义异常
├── llm/                          # LLM 工厂与重试
├── prompt/                       # 提示模板
│   ├── extract_prompt.py         # 画像提取提示（含 known_profile）
│   └── system_prompt.py          # Agent 系统提示
├── app.py                        # Streamlit 主入口
├── requirements.txt
├── Makefile
└── .env.example
🏗️ 架构演进
项目从单体设计重构为分层、可插拔的生产级架构。近期关键里程碑：

v1.0 — 短期记忆 + 三种长期记忆 + 冲突解决 + 衰减遗忘 + 合规拦截

v1.5 — Pydantic 强类型模型 + 分 Collection 存储 + 向量存储抽象层（Chroma）

v2.0 — Milvus 接入，支持混合检索、动态策略、MMR

v2.5（当前阶段一） — 同步门控优化：增量游标、三级门控（配置文件化）、已知画像脱敏注入、交互日志增量区间摘要、合规缓存、监控日志

核心设计原则

依赖倒置：上层仅依赖抽象接口，不绑定具体数据库。

单一职责：向量操作、业务策略、检索逻辑分层解耦。

开闭原则：新增向量数据库只需实现 BaseVectorStore，无需修改业务代码。

🔮 扩展路线
v3.0（阶段二） — RAG 知识库：接入银行产品文档与 FAQ，为生成和提取提供外部知识增强

v3.5（阶段三） — 异步 Worker 化：画像提取与交互日志从同步链路解耦，独立伸缩

v4.0（阶段四） — 多 Agent 协作：Supervisor + 专业子 Agent，记忆与 RAG 工具化共享

v4.2 — 图数据库集成：构建用户画像关系图谱，支持深层关联推理与缺失属性推断

v4.5 — 工具调用（利率计算器、申请表推送等）

v5.0 — 全链路可观测性（OpenTelemetry / LangSmith 集成）

📄 许可证
本项目采用 MIT License。

注：本项目为个人作品，用于展示 AI Agent 工程化与架构设计能力，非商业用途。