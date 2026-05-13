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