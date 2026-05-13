# 🏦 Loan Advisor Agent — 生产级银行贷款智能助手

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

一个面向金融领域的 AI Agent 实现，具备 **RAG 知识增强**、**三层长期记忆**（用户画像、交互日志、合规规则）、**生产级门控与合规拦截**，以及**分层可扩展的架构**。项目基于 LangGraph 构建有状态工作流，集成 Milvus 向量数据库，支持多用户会话隔离与配置热更新。

## ✨ 核心特性

### 🧠 记忆系统
- **状态持久化**：基于 LangGraph 的 SQLite Checkpointer 实现对话状态持久化，支持会话恢复与多会话管理。
- **三层长期记忆**：
  - **用户画像记忆**：支持 20+ 种实体类型的增量提取；**三级门控机制**（显式指令 → 强信号正则 → 弱信号得分累积）大幅减少无效 LLM 调用；已知画像脱敏摘要注入，避免重复提取；基于证据权重与置信度的**冲突自动解决**；时间衰减遗忘（指数衰减）与永久记忆保护。
  - **交互日志记忆**：增量生成的对话摘要，携带情感标签与事件类型，按时间戳检索，避免重复生成。
  - **合规规则记忆**：带 TTL 缓存的活跃规则检索，支持正则匹配，生成回答前可拦截或追加声明。
- **Pydantic 强类型模型**：所有记忆实体均经过 Pydantic 校验，杜绝字段错误与数据污染。
- **死信队列**：写入失败的记忆自动落入 JSONL 文件，保证数据不丢失。

### 🛡️ 生产级门控与合规
- **三级门控策略**（配置文件化）：
  1. 显式更新指令（无条件触发）
  2. 强信号正则（金额、手机号、身份证、修正词等）
  3. 弱信号累积（职业、收入、资产、婚姻等关键词计分，可配置阈值）
- **合规前置拦截**：独立节点扫描用户输入，命中高危规则可 **BLOCK / WARN / APPEND**，确保金融合规。
- **情感感知**：支持 `positive` / `neutral` / `anxious` / `frustrated` 四分类，规则优先 + LLM 兜底。
- **证据类型推断**：区分口头陈述、银行流水、征信报告、税务文件等，用于画像冲突解决。

### 🤖 智能对话与知识增强
- **RAG 全链路**：查询动态改写（Multi-Query / Step-Back / HyDE 三策略自动选择）→ 元数据过滤 → 三路并行召回（稠密 + 稀疏 BM25 + 术语向量）→ RRF 融合 → Cross-Encoder 重排序 → 上下文压缩。
- **双层缓存**：L1 内存 + L2 Redis，带防击穿锁，显著降低 LLM 与检索延迟。
- **多用户隔离**：通过 `user_id` 严格隔离记忆、画像与交互日志。
- **合规红线注入**：系统提示中动态注入用户画像、合规红线、近期交互与知识库内容，生成个性化、合规的回答。
- **多策略路由**：规则引擎判断用户意图是否需触发知识检索，闲聊或问候直接跳过。

### 🏗️ 生产级工程
- **分层抽象**：`BaseVectorStore` 抽象向量操作，`BaseMemoryStore` 定义记忆契约，业务与存储彻底解耦，已实现 Milvus 与 Chroma 双后端。
- **Milvus 深度集成**：支持混合检索、关键词检索、MMR 多样化检索，策略可动态选择。
- **健壮错误处理**：LLM 与 Embedding 调用装备指数退避重试；检索失败优雅降级；主备 Embedding 模型自动切换。
- **配置热更新**：基于 Pydantic + YAML 的层次化配置，支持 watchdog 热重载，敏感信息由环境变量注入。
- **全链路可观测**：Logging 注入 `user_id` / `thread_id`；Prometheus 指标上报（LLM 用量、合规拦截、检索性能等）。
- **离线数据处理管道**：文档预处理（按源类型加载）→ 智能分块（策略模式）→ Milvus 索引构建（多向量）。

## 🚀 快速开始

### 环境要求
- Python 3.11+
- Milvus 向量数据库（推荐 v2.4+），或使用 Chroma 作为轻量替代
- Redis（可选，用于 L2 缓存）

### 安装

```bash
git clone https://github.com/yourusername/BankLoanAgent.git
cd BankLoanAgent
pip install -r requirements.txt
配置
复制环境变量模板并编辑：

bash
cp .env.example .env
在 .env 中填入必要配置：

ini
DEEPSEEK_API_KEY=sk-xxxx
ALIBABA_API_KEY=sk-xxxx
MILVUS_URI=http://localhost:19530
LOG_LEVEL=INFO
（可选）根据实际情况修改 config/rules/ 下的 YAML 配置文件，如记忆系统参数、检索策略、门控规则等。

初始化向量库与索引
按顺序执行脚本（或使用 Makefile 目标）：

bash
# 初始化记忆存储集合（user_profile_memories / interaction_logs / compliance_rules）
python scripts/init_milvus_collections.py

# 运行文档预处理管道（Stage 1：加载与元数据提取）
python scripts/run_preprocess.py

# 运行文档分块管道（Stage 2：智能分块）
python scripts/run_chunking.py

# 构建知识库索引（Stage 3：多向量化并写入 Milvus）
python scripts/run_indexing.py
运行
bash
streamlit run app.py
访问 http://localhost:8501 即可开始对话。侧边栏支持切换用户、新建会话、查看长期画像及执行遗忘清理。

📁 项目结构
text
BankLoanAgent/
├── agent/                          # LangGraph 智能体核心
│   ├── state.py                    # AgentState 定义（含游标）
│   ├── nodes/                      # 节点实现
│   │   ├── retrieve_memory_node.py     # 记忆检索 + 全局消息序号分配
│   │   ├── retrieval_knowledge_node.py # RAG 知识检索
│   │   ├── compliance_guard_node.py    # 合规拦截
│   │   ├── call_model_node.py          # LLM 生成
│   │   ├── extract_profile_node.py     # 增量画像提取（三级门控）
│   │   └── log_interaction_node.py     # 增量交互摘要与存储
│   ├── checkpointer.py             # SQLite Checkpointer 封装
│   ├── graph.py                    # 图编排（边定义、条件路由）
│   └── constants.py                # 节点名、状态字段常量
├── config/                         # 配置与常量
│   ├── settings.py                 # 环境变量定义（Pydantic Settings）
│   ├── registry.py                 # 配置注册中心（支持热更新）
│   ├── global_constant/            # 全局枚举与字段名常量
│   ├── models/                     # 配置模型定义（LLM、记忆、检索等）
│   ├── prompts/                    # 所有提示词模板
│   └── rules/                      # YAML 配置文件（含门控规则）
├── memory/                         # 长期记忆系统
│   ├── base.py                     # BaseRetriever 抽象
│   ├── memory_business_store/
│   │   ├── base_memory_store.py    # BaseMemoryStore 接口
│   │   └── long_term_memory_store.py # 核心实现（冲突解决、遗忘、脱敏、缓存）
│   ├── memory_vector_store/
│   │   ├── base_vector_store.py    # BaseVectorStore 抽象
│   │   ├── milvus_memory_vector_store.py # Milvus 实现（混合检索/MMR）
│   │   └── chroma_memory_vector_store.py # Chroma 实现（备选）
│   ├── memory_retriever.py         # MemoryVectorRetriever
│   ├── models/                     # Pydantic 记忆模型与映射器
│   ├── memory_constant/            # 记忆相关枚举与常量
│   └── memory_utils/               # 门控、消息格式化、解析工具
├── modules/
│   ├── retrieval/                  # RAG 检索管道
│   │   ├── retrieval_service.py    # 检索服务编排（路由、重写、召回、融合、重排序、压缩）
│   │   ├── query_rewriter.py       # 查询改写（动态策略选择）
│   │   ├── query_filter.py         # 元数据过滤条件提取
│   │   ├── knowledge_search_engine.py # Milvus 搜索引擎（稠密/稀疏/术语）
│   │   ├── rrf_fusion.py           # RRF 融合算法
│   │   ├── reranker.py             # Cross-Encoder 重排序
│   │   ├── context_compressor.py   # 上下文压缩
│   │   └── router/                 # 检索路由（规则/模型）
│   └── module_services/            # 领域服务
│       ├── chat_models.py          # RobustLLM（重试+降级）
│       ├── embeddings.py           # RobustEmbeddings（主备切换+降级）
│       ├── profile_extractor.py    # 画像提取调用
│       ├── sentiment_analyser.py   # 情感分析
│       ├── evidence_infer.py       # 证据类型推断
│       └── SummaryGenerator.py     # 对话摘要生成
├── infra/                          # 基础设施
│   ├── milvus_client.py            # Milvus 连接管理（单例）
│   ├── cache/                      # 双层缓存（L1 Memory + L2 Redis）
│   └── collections.py              # 集合名称映射
├── pipelines/                      # 离线数据处理管道
│   ├── scripts/
│   │   ├── file_scripts/           # 文件加载、预处理、分块
│   │   └── init_knowledge_index.py # 知识库索引构建
│   └── constant.py
├── app.py                          # Streamlit 主入口
├── main.py                         # load_config 入口
├── Makefile
├── requirements.txt
└── .env.example
🏗️ 架构设计
项目严格遵循以下设计原则：

依赖倒置：上层模块仅依赖抽象接口（BaseMemoryStore、BaseVectorStore、BaseRetriever），不绑定具体数据库或框架。

单一职责：向量存储操作、业务策略、检索逻辑分层解耦；LangGraph 节点各自专注单一任务。

开闭原则：新增向量数据库只需实现 BaseVectorStore，无需修改业务代码；配置 YAML 化支持运行时调整。

游标增量机制：画像提取与交互日志均基于全局消息序号游标，避免重复处理已处理消息，大幅节省 LLM 开销。

🔮 扩展路线
v3.0 — 知识库增强：已实现完整的 RAG 管道，后续可接入更多银行产品文档与 FAQ 数据。

v3.5 — 异步 Worker 化：将画像提取与交互日志从同步链路解耦，交由独立消费队列处理，提升对话响应速度。

v4.0 — 多 Agent 协作：Supervisor + 专业子 Agent（贷款咨询、风控、合规），记忆与 RAG 工具化共享。

v4.2 — 图数据库集成：构建用户画像关系图谱，支持深层关联推理与缺失属性推断。

v4.5 — 工具调用：利率计算器、贷款申请表推送等外部工具集成。

v5.0 — 全链路可观测性增强：OpenTelemetry 深度集成，LangSmith 追踪调试。

📄 许可证
本项目采用 MIT License。

注：本项目为个人作品，用于展示 AI Agent 工程化与架构设计能力，非商业用途。