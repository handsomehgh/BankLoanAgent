# 多 Agent 银行贷款助手系统 — 开发计划（按实现顺序）

本文档基于《多 Agent 银行贷款助手系统设计方案 v2.3》制定，严格按照实现顺序列出所有任务，不包含时间预估。所有任务按阶段划分，阶段内按依赖关系从前到后排列。

---

## Phase 0：基础设施准备

**目标**：搭建多 Agent 开发所需的基础设施，确保后续工作可并行推进。

- **任务 0.1**：配置管理体系初始化
  - 创建 `config/rules/supervisor.yaml`、`loan_advisor.yaml`、`risk_assessment.yaml`、`after_loan.yaml`、`tool_registry.yaml`
  - 注册到 `ConfigRegistry`，确保支持热加载
  - 依赖：现有 `ConfigRegistry` 实现
- **任务 0.2**：抽象基类与核心数据结构定义
  - 定义 `BaseAgent` 抽象类
  - 定义 `BaseTool` 抽象类
  - 定义 `AgentContext`、`AgentResponse`、`ToolResult` 数据结构
  - 定义各子 Agent 的 `State`（如 `LoanAdvisorState`、`RiskAssessmentState` 等）
  - 依赖：任务 0.1
- **任务 0.3**：消息队列基础搭建
  - 封装 Redis Streams 消息生产者和消费者
  - 定义消息格式与消费者组注册机制
  - 实现死信队列（写入独立的 Redis key）
  - 依赖：现有 `RedisManager`
- **任务 0.4**：监控与日志基础设施升级
  - 在 `ContextFilter` 中增加 `trace_id` 字段
  - 在请求入口生成全局唯一 `trace_id`（UUID7）
  - 新增 Prometheus 指标注册（`supervisor_routing_total`、`tool_call_total` 等）
  - 依赖：现有 `logging_config` 和 `metrics` 模块

---

## Phase 1：骨架搭建

**目标**：实现 Supervisor + LoanAdvisor Agent 完整链路，用户可进行贷款咨询并获得回复。

- **任务 1.1**：公共合规过滤器实现
  - 将现有 `compliance_guard_node` 逻辑提升为独立前置过滤器节点
  - 保留正则匹配 + 增加 LLM 二审兜底
  - 返回 `BLOCK` / `WARN` / `PASS` 三种结果
  - 依赖：Phase 0 全部完成
- **任务 1.2**：公共 Memory 检索层重构
  - 将现有 `retrieve_memory_node` 调整为公共函数，由 Supervisor 直接调用
  - 返回画像摘要、合规规则、交互日志，统一组装
  - 依赖：任务 1.1
- **任务 1.3**：Supervisor Agent 核心实现
  - 实现 `SupervisorState` 定义
  - 实现 Supervisor 节点：调用 Memory 检索，组装 `AgentContext`
  - 实现 LLM 动态路由：系统提示注入所有子 Agent 能力描述，LLM 输出目标 Agent
  - 实现静态规则兜底：覆盖问候、感谢、告别等直接回复场景
  - 实现结果整合节点（ResultAggregator）
  - 实现意图澄清与兜底（连续两次无法理解转人工）
  - 依赖：任务 0.2，任务 1.2
- **任务 1.4**：LoanAdvisor Agent 实现
  - 实现 `LoanAdvisorState` 定义
  - 构建 `LoanAdvisor` 子图（`StateGraph`），至少包含生成回复节点
  - 接收 `AgentContext`，按需调用公共 RAG 层检索知识
  - 生成回答（复用现有 `RobustLLM`）
  - 编译子图，集成到 Supervisor 主图中
  - 依赖：任务 1.3
- **任务 1.5**：公共 Tool Registry 框架 + 首批工具实现
  - 实现 `ToolRegistry`：启动时扫描 `tools/` 目录下所有 `@tool` 函数并注册
  - 实现 `ToolExecutor`：权限校验、参数验证、审计日志、执行、异常处理
  - 实现首批工具（`calculate_monthly_payment`、`query_interest_rate`）
  - 配置 Fast-Fail：YAML 声明的工具不存在或版本不兼容时启动报错
  - 依赖：任务 0.2（`BaseTool` 定义），任务 1.4 完成后可集成工具调用
- **任务 1.6**：审计日志与反馈采集埋点
  - 实现审计日志记录函数：`log_routing`、`log_compliance_action`、`log_tool_call`
  - 在 Supervisor 路由后记录 `routing_detail` 事件
  - 在合规过滤后记录 `compliance_action` 事件
  - 在工具调用前后记录审计日志
  - 实现 `knowledge_miss`、`negative_feedback` 事件写入消息队列
  - 依赖：任务 0.3，任务 0.4
- **任务 1.7**：ProfileManager 同步提取路径
  - 将现有 `extract_profile_node` 逻辑拆分为 `ProfileManager.sync_extract()`
  - 同步路径提取关键实体（月收入、职业、贷款用途、征信、负债、银行关系）
  - 写入 Milvus 后立即调用 `cache.invalidate` 删除画像摘要缓存
  - 依赖：任务 0.3（消息队列）、现有 Memory 层

---

## Phase 2：业务扩展

**目标**：扩展风险评估和贷后管理 Agent，实现多 Agent 编排和人工转接机制。

- **任务 2.1**：RiskAssessment Agent 实现
  - 实现 `RiskAssessmentState` 定义
  - 构建 `RiskAssessment` 子图
  - 接收 `AgentContext`，结合画像和合规规则生成风险评估
  - 实现触发人工断点的逻辑（如“连三累六”）
  - 编译子图，集成到 Supervisor 主图
  - 依赖：Phase 1 全部完成
- **任务 2.2**：AfterLoan Agent 实现
  - 实现 `AfterLoanState` 定义
  - 构建 `AfterLoan` 子图
  - 接收 `AgentContext`，处理贷后操作类问题
  - 编译子图，集成到 Supervisor 主图
  - 依赖：Phase 1 全部完成
- **任务 2.3**：多 Agent 编排逻辑
  - Supervisor 支持并行调用多个子 Agent（Fan-out）
  - 结果整合节点完善：处理并行回复的拼接、超时兜底
  - 实现 Agent 调度超时处理
  - 依赖：任务 2.1，任务 2.2
- **任务 2.4**：Human-in-the-Loop 断点实现
  - 实现 `HumanHandoff` 子图（生成转接摘要）
  - 在 Supervisor 图中插入 `human_handoff` 节点
  - 实现 7 个断点触发条件检测
  - 实现超时降级策略（5 分钟超时自动执行预设动作）
  - 依赖：任务 1.3（Supervisor 图），任务 1.1（合规过滤）
- **任务 2.5**：补充工具实现
  - 实现 `calculate_dti`、`calculate_max_loan_amount`、`generate_risk_report`
  - 实现 `calculate_prepayment`、`generate_repayment_schedule`、`generate_settlement_certificate`
  - 实现 `generate_material_checklist`、`query_regulation`
  - 所有工具必须遵守 `@tool` 装饰器规范（`version`、`tags`、类型注解、docstring）
  - 在 `tools/` 目录下按 finance/query/generation 分类存放
  - 依赖：任务 1.5（ToolRegistry 框架）
- **任务 2.6**：Agent 提示词完善与路由规则调优
  - 完善 `supervisor.yaml`，明确各 Agent 的职责和排除边界
  - 完善各子 Agent 的 YAML 提示词，补充输出格式要求和行为约束
  - 更新 Supervisor 的 LLM 动态路由提示词，确保路由准确率
  - 依赖：Phase 2 所有 Agent 实现完成

---

## Phase 3：异步化与闭环

**目标**：画像异步提取、反馈闭环、监控告警完善。

- **任务 3.1**：ProfileManager 异步化
  - 实现画像异步消费者：从 Redis Streams 读取消息，提取非关键实体
  - 遵循“先写后删”缓存一致性策略
  - 消费失败消息进入死信队列
  - 依赖：任务 0.3（消息队列），任务 1.7（同步 ProfileManager）
- **任务 3.2**：交互日志异步化
  - 将现有 `log_interaction_node` 逻辑移入消息队列消费者
  - 保留摘要生成、情感分析逻辑，改为异步执行
  - 依赖：任务 0.3（消息队列）
- **任务 3.3**：FeedbackLoop 消费者实现
  - 消费消息队列中的 `knowledge_miss`、`routing_detail`、`negative_feedback` 事件
  - 统计知识库盲区、路由准确率、用户满意度
  - 设置阈值自动预警（如知识命中率 < 70% 连续 3 天告警）
  - 依赖：任务 0.3，任务 1.6（埋点数据已就绪）
- **任务 3.4**：监控告警完善与压测
  - 配置 Prometheus 告警规则（按 v2.3 第 13 章阈值）
  - 导入 Grafana 仪表盘模板
  - 使用 Locust 进行压力测试，记录 QPS、P50/P99 延迟基线
  - 验证缓存一致性、人工断点超时降级等边缘场景
  - 依赖：Phase 1、Phase 2、Phase 3 全部功能完成

---

## 依赖关系总览
Phase 0
├── 0.1 配置管理
├── 0.2 抽象基类 + 数据结构
├── 0.3 消息队列
└── 0.4 监控日志升级

Phase 1
├── 1.1 合规过滤器 ─────────────── 依赖 Phase 0
├── 1.2 Memory 检索层重构 ─────── 依赖 1.1
├── 1.3 Supervisor 核心 ───────── 依赖 0.2, 1.2
├── 1.4 LoanAdvisor Agent ─────── 依赖 1.3
├── 1.5 Tool Registry + 首批工具 ─ 依赖 0.2
├── 1.6 审计与反馈埋点 ────────── 依赖 0.3, 0.4
└── 1.7 ProfileManager 同步路径 ─ 依赖 0.3

Phase 2
├── 2.1 RiskAssessment Agent ──── 依赖 Phase 1
├── 2.2 AfterLoan Agent ───────── 依赖 Phase 1
├── 2.3 多 Agent 编排 ─────────── 依赖 2.1, 2.2
├── 2.4 Human-in-the-Loop ─────── 依赖 1.3, 1.1
├── 2.5 补充工具 ──────────────── 依赖 1.5
└── 2.6 提示词与路由调优 ──────── 依赖 Phase 2 所有 Agent 实现

Phase 3
├── 3.1 ProfileManager 异步化 ─── 依赖 0.3, 1.7
├── 3.2 交互日志异步化 ────────── 依赖 0.3
├── 3.3 FeedbackLoop ──────────── 依赖 0.3, 1.6
└── 3.4 监控告警与压测 ────────── 依赖 Phase 1,2,3 全部功能

Phase 2 收尾：业务扩展完善（当前阶段）
目标：完成工具调用集成、转人工闭环，确保核心业务链完整。

工具调用集成（补充任务 2.5 升级）

任务：在 LoanAdvisor、AfterLoan 等 Agent 的生成节点中实现 Function Calling 循环，接入 ToolExecutor。

内容：Agent 检测 LLM 返回的 tool_calls → 通过 ToolExecutor 执行（权限、审计、监控） → 构造 ToolMessage 并分配 message_index → 再次调用 LLM 生成最终回复。

依赖：ToolRegistry、ToolExecutor、SeqGenerator 已就绪。

产出：Agent 能真正调用计算工具（月供、利率），并返回精确结果。

Fan-out 分发器完善（原任务 2.3 补充）

任务：确保并行调用时工具结果正确聚合，转人工信号不丢失，超时降级稳定。

内容：在 FanoutDispatcher 中正确提取各子图返回的 final_response 和 trigger_human_handoff，统一设置清理标记；完善超时异常处理。

产出：多 Agent 并行调用（如 LoanAdvisor + RiskAssessment）稳定可靠。

HumanHandoff 节点优化（原任务 2.4 增强）

任务：保留当前占位降级模式，增加交互日志记录，为未来 interrupt 预留接口。

内容：确保 HumanHandoff 节点后正确进入 log_interaction，记录转接事件；不实现挂起/恢复，但代码结构允许后续扩展。

产出：转接流程闭环，日志可追溯。

状态清理与冗余字段移除

任务：按状态清理协议，彻底清除 interaction_logged 等无用字段，确保所有临时状态在消费后重置。

内容：从 SupervisorState 定义、节点返回中移除 interaction_logged；确认 profile_updated 仍被前端使用，暂保留。

产出：状态精简，无跨轮次污染。

Phase 3：异步化与质量闭环
目标：解耦画像提取、交互日志，实现反馈分析，提升系统性能与可观测性。

画像提取异步化（原任务 3.1）

任务：将 extract_profile_node 从主链路移入消息队列消费者。

内容：利用现有 MessageProducer 发送待提取事件；ProfileManager 消费者消费消息，执行门控→提取→写入 Milvus；同步路径保留或移除（建议先全部异步，观察性能）。

产出：主链路延迟降低，画像提取不影响对话响应。

交互日志异步化（原任务 3.2）

任务：将 log_interaction_node 改为异步消费模式。

内容：主链路中发送日志事件到 Redis Streams，消费者负责调用 SummaryGenerator 和 SentimentAnalyzer 并写入 Milvus。

产出：主链路进一步缩短，日志记录解耦。

FeedbackLoop 消费者实现（原任务 3.3）

任务：开发 feedback_events 消费者，分析知识缺失、路由错误、负反馈。

内容：消费 knowledge_miss、negative_feedback 等事件，生成统计报告（日志/仪表板）；为知识库优化提供依据。

产出：系统具备自我改进的数据基础。

监控告警完善（原任务 3.4）

任务：为关键指标绑定 Prometheus 告警规则。

内容：设置 tool_call_failure_rate、knowledge_miss_rate、human_handoff_timeout_total 等阈值。

产出：生产环境实时监控和通知。

Phase 4+：长期优化
HumanHandoff 真实中断（Interrupt）

任务：将当前占位回复升级为 LangGraph interrupt 挂起，支持人工坐席接管和恢复。

内容：对接人工坐席工作台，实现挂起→分配→坐席回复/转回 AI→超时降级。

产出：真正实现 Human‑in‑the‑Loop。

消息序号工厂模式改造

任务：将消息序号从补全节点统一分配改为在消息创建时自动赋予。

内容：实现 create_user_message、create_ai_message 工厂函数，内部调用 SeqGenerator；逐步替换所有直接构造消息的代码；移除 ensure_message_indexes 节点。

产出：彻底消除遗漏消息序号的风险。

子图结果缓存

任务：对相同查询在会话内缓存子图输出（RAG 检索、LLM 生成），减少重复调用。

内容：利用现有 CacheManager，在 Agent 节点入口检查缓存，命中则直接返回。

产出：提升响应速度，降低成本。

端到端测试与评估

任务：构建评估数据集，进行自动化回归测试。

内容：基于知识库生成测试问题，验证路由准确率、回答质量、转人工正确性。

产出：可量化系统质量，保证迭代不引入回归。

🔴 急需补充（面试会重点问）
Agent评估框架 (Harness)：至少搭建一个简单的对话质量评估脚本，展示你考虑到了Agent的可测试性。

Token经济性管理：增加Token预算控制、上下文压缩策略的文档化说明。

🟡 加分项（提升竞争力）
Skills化改造：将工具注册中心包装成Skills概念，展示你对2025年技术趋势的跟进。

可观测性增强：补充Agent全链路追踪的架构设计（即使代码未完整实现）。

🟢 可选（根据面试方向）
MCP集成：如果面试公司看重系统互操作性，可以考虑集成MCP。