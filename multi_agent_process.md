# 多 Agent 银行贷款助手系统 — 详细开发计划 v1.0

## 概述

本文档基于《多 Agent 银行贷款助手系统设计方案 v2.1》制定详细的开发实施计划，涵盖 Phase 0 至 Phase 3 的全部任务，明确每个任务的输入、输出、依赖关系、验收标准和预计工期。灰度相关功能不纳入本次开发范围。

---

## Phase 0：开发环境与基础设施准备

**目标**：搭建多 Agent 开发所需的基础设施，确保后续工作可并行推进。

### 任务 0.1：配置管理体系初始化
- **输入**：现有 `config/rules/` 目录结构
- **输出**：按 Agent 拆分的配置文件骨架
- **描述**：
  - 创建 `config/rules/supervisor.yaml`，定义 Supervisor 提示词、路由规则
  - 创建 `config/rules/loan_advisor.yaml`，定义 LoanAdvisor 提示词、工具列表及版本范围
  - 创建 `config/rules/risk_assessment.yaml`，定义 RiskAssessment 提示词、工具列表
  - 创建 `config/rules/after_loan.yaml`，定义 AfterLoan 提示词、工具列表
  - 创建 `config/rules/tool_registry.yaml`，定义工具元数据（名称、版本、权限、输入输出 schema）
  - 更新 `ConfigRegistry` 注册新模块，支持热加载
- **验收标准**：
  - 所有配置文件可被 `ConfigRegistry` 正常加载
  - 修改 YAML 文件后系统能自动重载
- **预计工期**：1 天

### 任务 0.2：公共基础层接口抽象
- **输入**：现有 `BaseMemoryStore`、`BaseVectorStore`、`BaseRetriever`
- **输出**：新增 `BaseAgent`、`BaseTool` 抽象基类
- **描述**：
  - 定义 `BaseAgent`：`invoke(AgentContext) -> AgentResponse`
  - 定义 `BaseTool`：`execute(params: Dict) -> ToolResult`，含 `version`、`allowed_agents` 属性
  - 定义 `AgentContext`、`AgentResponse`、`ToolResult` 数据结构
- **验收标准**：
  - 所有抽象基类定义清晰，包含类型注解
  - 后续 Agent 和 Tool 实现必须继承这些基类
- **预计工期**：1 天

### 任务 0.3：监控与日志基础设施升级
- **输入**：现有 `utils/logging_config.py`、`utils/monitor_utils/metrics.py`
- **输出**：支持 `trace_id` 全链路追踪的日志上下文
- **描述**：
  - 在 `ContextFilter` 中增加 `trace_id` 字段
  - 在请求入口生成全局唯一 `trace_id`（UUID7 或雪花算法），注入上下文
  - 为审计事件定义日志格式（JSON 结构化日志）
  - 新增 Prometheus 指标注册（`supervisor_routing_total`、`tool_call_total` 等）
- **验收标准**：
  - 每条日志自动携带 `trace_id`
  - 新增指标在 `/metrics` 端点可见
- **预计工期**：1 天

### 任务 0.4：消息队列基础搭建
- **输入**：现有 Redis 基础设施（`RedisManager`）
- **输出**：`infra/message_queue.py`
- **描述**：
  - 基于 Redis Streams 封装简单的消息生产者和消费者
  - 定义消息格式：`{event_type, payload, trace_id, timestamp}`
  - 实现消费者组注册和消息确认机制
  - 实现死信队列（写入独立的 Redis key）
- **验收标准**：
  - 生产者可写入消息，消费者可正常消费并确认
  - 消费失败的消息自动进入死信队列
- **预计工期**：1 天

---

## Phase 1：骨架搭建（P0）

**目标**：实现 Supervisor + LoanAdvisor Agent 完整链路，用户可进行贷款咨询并获得回复。

### 任务 1.1：公共合规过滤器实现
- **输入**：现有 `compliance_guard_node.py`
- **输出**：`modules/agent/nodes/compliance_filter.py`
- **描述**：
  - 将现有合规检查逻辑提升为独立的前置过滤器
  - 保留正则匹配 + 增加 LLM 二审兜底（当正则未命中但存在高风险关键词时触发）
  - 返回三种结果：`BLOCK`（直接返回拦截回复）、`WARN`（标记并放行）、`PASS`
- **验收标准**：
  - 命中黑名单规则返回拦截消息
  - 正常查询通过，响应时间 < 50ms
- **预计工期**：2 天

### 任务 1.2：Supervisor Agent 核心实现
- **输入**：`AgentContext` 定义、现有 `retrieval_rule_router.py`
- **输出**：`modules/agent/supervisor.py`
- **描述**：
  - 实现 Supervisor 节点（继承 `BaseAgent`）
  - 调用公共 Memory 检索获取画像、合规规则、交互日志
  - 组装 `AgentContext`
  - 实现 LLM 动态路由：系统提示中注入各 Agent 能力描述，LLM 输出目标 Agent 名称
  - 实现规则兜底：纯问候、简单感谢等直接返回，不清路由
  - 意图澄清：模糊时追问，连续两次无法理解返回兜底回复
- **验收标准**：
  - 能正确将贷款咨询请求路由到 LoanAdvisor Agent
  - 对“你好”、“谢谢”等直接回复不进行路由
- **预计工期**：3 天

### 任务 1.3：LoanAdvisor Agent 实现
- **输入**：现有 `retrieval_knowledge_node`、`call_model_node`、知识库内容
- **输出**：`modules/agent/loan_advisor_agent.py`
- **描述**：
  - 实现 LoanAdvisor Agent（继承 `BaseAgent`）
  - 接收 `AgentContext`，按需调用公共 RAG 层检索知识
  - 调用 LLM 生成回答（复用现有 `RobustLLM`）
  - 支持多轮对话（从 `AgentContext.conversation_summary` 获取历史）
- **验收标准**：
  - 能回答住房贷款、消费贷款、经营贷款等产品咨询
  - 回答内容引用知识库并注明“仅供参考，以银行审批为准”
- **预计工期**：3 天

### 任务 1.4：公共 Tool Registry 框架 + 首批工具实现
- **输入**：`BaseTool` 抽象、`tool_registry.yaml`
- **输出**：`modules/tools/tool_registry.py`、`modules/tools/calculator.py`、`modules/tools/interest_query.py`
- **描述**：
  - 实现 `ToolRegistry`：加载 YAML 配置，提供 `get_tool(name, version_range)` 和权限校验
  - 实现 `calculate_monthly_payment` 工具（等额本息/等额本金）
  - 实现 `query_interest_rate` 工具（从知识库或静态数据返回最新 LPR）
  - 为每个工具定义输入输出 schema（Pydantic 模型）
- **验收标准**：
  - LoanAdvisor Agent 能调用 `calculate_monthly_payment` 计算月供
  - 无权限的 Agent 调用工具返回错误
- **预计工期**：2 天

### 任务 1.5：Agent 图编排与主链路集成
- **输入**：Supervisor、LoanAdvisor Agent、合规过滤器
- **输出**：`modules/agent/multi_agent_graph.py`
- **描述**：
  - 使用 LangGraph 构建多 Agent 图：
    - 用户输入 → 合规过滤 → Memory 检索 → Supervisor → LoanAdvisor Agent → 响应
  - 集成 `SqliteSaver` 持久化状态
  - 在 `app.py` 中接入新图（替换 Phase 0 的占位响应）
- **验收标准**：
  - 用户提问“我想申请住房贷款”可获得完整回复
  - 会话状态持久化，刷新后对话历史不丢失
- **预计工期**：3 天

### 任务 1.6：审计日志与反馈采集埋点
- **输入**：任务 0.3 的监控基础设施
- **输出**：`modules/agent/audit.py`、`modules/agent/feedback_collector.py`
- **描述**：
  - 实现审计日志记录函数：`log_routing`、`log_compliance_action`、`log_tool_call`
  - 在 Supervisor 路由后记录 `routing_detail` 事件
  - 在合规过滤后记录 `compliance_action` 事件
  - 在工具调用后记录工具调用详情
  - 实现 `knowledge_miss`、`negative_feedback` 事件写入消息队列（为 FeedbackLoop 提供数据）
- **验收标准**：
  - 每次用户请求都产生对应的审计日志（JSON 格式，含 `trace_id`）
  - 反馈事件成功写入 Redis Streams
- **预计工期**：1 天

### 任务 1.7：ProfileManager 同步提取路径
- **输入**：现有 `extract_profile_node.py`、`ProfileGate`、`EvidenceTypeInfer`
- **输出**：`modules/memory/profile_manager.py`
- **描述**：
  - 将现有画像提取逻辑拆分为 `ProfileManager.sync_extract()` 和 `async_extract()`
  - 同步路径提取关键实体（月收入、职业、贷款用途、征信、负债、银行关系）
  - 写入 Milvus 后立即调用 `cache.invalidate` 删除画像摘要缓存
  - 在主链路响应返回前完成同步提取
- **验收标准**：
  - 用户提供收入信息后，下一轮对话立即可用
  - 关键实体提取延迟 < 1s
- **预计工期**：2 天

---

## Phase 2：业务扩展（P1）

**目标**：扩展风险评估和贷后管理 Agent，实现多 Agent 编排和人工转接机制。

### 任务 2.1：RiskAssessment Agent 实现
- **输入**：合规规则记忆、监管政策知识库、公共 Tool Registry
- **输出**：`modules/agent/risk_assessment_agent.py`
- **描述**：
  - 实现 RiskAssessment Agent（继承 `BaseAgent`）
  - 调用 `calculate_dti`、`query_regulation` 等工具
  - 结合用户画像生成风险评估报告
  - 对“连三累六”等严重征信问题触发人工断点
- **验收标准**：
  - 能根据用户画像输出风险等级和建议
  - 征信“连三累六”时返回标准话术并记录人工转接事件
- **预计工期**：3 天

### 任务 2.2：AfterLoan Agent 实现
- **输入**：贷后管理知识库、公共 Tool Registry
- **输出**：`modules/agent/after_loan_agent.py`
- **描述**：
  - 实现 AfterLoan Agent（继承 `BaseAgent`）
  - 调用 `calculate_prepayment`、`generate_repayment_schedule`、`generate_settlement_certificate` 等工具
  - 提供展期评估、还款计划查询等服务
- **验收标准**：
  - 能回答“提前还款违约金多少”并试算节省利息
  - 能生成结清证明模板
- **预计工期**：3 天

### 任务 2.3：多 Agent 编排逻辑
- **输入**：Supervisor、LoanAdvisor、RiskAssessment、AfterLoan Agent
- **输出**：更新 `supervisor.py` 的编排逻辑
- **描述**：
  - Supervisor 支持并行调用多个 Agent（如同时咨询产品和评估风险）
  - 实现 Agent 响应整合：将多个 Agent 的输出合并为统一回复
  - 实现 Agent 调度超时处理（单个 Agent 超时不影响其他 Agent）
- **验收标准**：
  - 用户问题“我想贷50万装修，征信有过逾期”同时触发 LoanAdvisor 和 RiskAssessment
  - 超时 Agent 的输出用默认兜底文本替代
- **预计工期**：2 天

### 任务 2.4：Human-in-the-Loop 断点实现
- **输入**：LangGraph `interrupt` 机制
- **输出**：更新 `multi_agent_graph.py`
- **描述**：
  - 在 Supervisor 图中插入 `human_handoff` 节点
  - 实现 7 个断点触发条件检测（合规 BLOCK 申诉、风险阈值、金额超限、征信严重不良、投诉敏感词、用户显式转人工、连续意图理解失败）
  - 每个断点设置超时 5 分钟，超时后执行预设降级策略
  - 断点触发时写入审计日志
- **验收标准**：
  - 输入“投诉”相关关键词能触发人工转接
  - 超时后自动返回降级话术
- **预计工期**：3 天

### 任务 2.5：补充工具实现
- **输入**：`tool_registry.yaml`
- **输出**：`modules/tools/risk_tools.py`、`modules/tools/after_loan_tools.py`
- **描述**：
  - 实现 `calculate_dti`、`calculate_max_loan_amount`、`generate_risk_report`
  - 实现 `calculate_prepayment`、`generate_repayment_schedule`、`generate_settlement_certificate`
  - 为每个工具编写单元测试
- **验收标准**：
  - 所有工具通过权限校验和功能测试
  - 工具调用成功率 > 99%
- **预计工期**：3 天

---

## Phase 3：异步化与闭环（P2）

**目标**：画像异步提取、反馈闭环、监控告警完善。

### 任务 3.1：ProfileManager 异步化
- **输入**：Phase 1 的 `profile_manager.py`、消息队列
- **输出**：`modules/memory/profile_consumer.py`
- **描述**：
  - 实现画像异步消费者：从 Redis Streams 读取消息，提取非关键实体
  - 遵循“先写后删”缓存一致性策略
  - 消费失败消息进入死信队列
- **验收标准**：
  - 非关键实体延迟 < 5s
  - 死信队列可正常重放
- **预计工期**：2 天

### 任务 3.2：交互日志异步化
- **输入**：现有 `log_interaction_node.py`、消息队列
- **输出**：`modules/memory/interaction_consumer.py`
- **描述**：
  - 将交互日志写入逻辑从同步链路移除，改为消费消息队列
  - 保留现有摘要生成、情感分析逻辑
- **验收标准**：
  - 交互日志延迟 < 10s
  - 主链路响应不包含日志写入时间
- **预计工期**：1 天

### 任务 3.3：FeedbackLoop 消费者实现
- **输入**：消息队列中的 `knowledge_miss`、`routing_detail`、`negative_feedback` 事件
- **输出**：`modules/feedback/feedback_loop.py`
- **描述**：
  - 消费反馈事件，统计知识库盲区、路由错误率、用户满意度
  - 输出每日报告（写入日志文件或数据库）
  - 设置阈值自动预警（如知识命中率 < 70%）
- **验收标准**：
  - 每日生成反馈统计报告
  - 知识命中率连续 3 天低于 70% 触发告警
- **预计工期**：2 天

### 任务 3.4：监控告警完善与压测
- **输入**：Prometheus 指标、Grafana 模板
- **输出**：`config/monitoring/alert_rules.yml`、`config/monitoring/dashboard.json`
- **描述**：
  - 配置 Prometheus 告警规则（见设计 v2.1 第 13 章）
  - 导入 Grafana 仪表盘模板
  - 使用 Locust 进行压力测试，记录 QPS 和延迟基线
- **验收标准**：
  - 告警触发测试通过
  - 压测报告输出 P50/P99 延迟和 QPS 上限
- **预计工期**：2 天

---

## 总体时间线

| 阶段 | 任务数 | 预计工期 | 累计工期 |
|------|--------|----------|----------|
| Phase 0 | 4 | 4 天 | 4 天 |
| Phase 1 | 7 | 17 天 | 21 天 |
| Phase 2 | 5 | 14 天 | 35 天 |
| Phase 3 | 4 | 7 天 | 42 天 |

总预计工期：**6 周**（按一人全职投入估算）

---

## 依赖关系图
Phase 0 (基础准备)
├─ 0.1 配置管理
├─ 0.2 抽象基类
├─ 0.3 监控基础设施
└─ 0.4 消息队列
│
Phase 1 (骨架搭建)
├─ 1.1 合规过滤器 ──────────────── 无依赖
├─ 1.2 Supervisor 核心 ─────────── 依赖 0.1, 0.2
├─ 1.3 LoanAdvisor Agent ───────── 依赖 0.2, 1.2
├─ 1.4 Tool Registry 框架 ──────── 依赖 0.2, 0.1
├─ 1.5 图编排与集成 ────────────── 依赖 1.1, 1.2, 1.3, 1.4
├─ 1.6 审计与反馈埋点 ──────────── 依赖 0.3, 0.4
└─ 1.7 ProfileManager 同步路径 ─── 依赖 0.4
│
Phase 2 (业务扩展)
├─ 2.1 RiskAssessment Agent ─────── 依赖 1.4, 1.5
├─ 2.2 AfterLoan Agent ──────────── 依赖 1.4, 1.5
├─ 2.3 多 Agent 编排 ────────────── 依赖 2.1, 2.2
├─ 2.4 Human-in-the-Loop ────────── 依赖 1.5
└─ 2.5 补充工具 ─────────────────── 依赖 1.4
│
Phase 3 (异步化与闭环)
├─ 3.1 ProfileManager 异步化 ────── 依赖 1.7, 0.4
├─ 3.2 交互日志异步化 ───────────── 依赖 0.4
├─ 3.3 FeedbackLoop ─────────────── 依赖 0.4, 1.6
└─ 3.4 监控告警与压测 ──────────── 依赖 0.3, 全部功能

text

---

## 风险与缓解措施

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Supervisor 路由准确率低 | 用户回答不匹配 | Phase 1 中保留规则兜底，积累路由日志用于调优 |
| 工具计算错误 | 金融数据不准确 | 每个工具编写单元测试，交叉验证公式 |
| 异步链路消息丢失 | 画像更新丢失 | 关键实体同步路径兜底；消息持久化 + 死信队列 |
| 人工断点超时频繁 | 用户等待体验差 | 超时阈值可配置，监控超时次数 |
| 多 Agent 调用延迟高 | 用户体验下降 | 并行编排减少串行等待；设置 Agent 调用超时 |