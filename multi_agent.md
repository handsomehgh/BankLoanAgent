# 多 Agent 银行贷款助手系统设计方案 v2.1

## 1. 设计目标

将现有单 Agent 系统升级为多 Agent 协作系统，在保留公共 Memory / RAG 能力的基础上，实现：

- **专业化分工**：每个 Agent 聚焦独立业务域，配备专属工具
- **生产级可靠性**：合规前置、人工兜底、异步解耦、质量闭环、缓存一致性
- **可扩展性**：工具注册制（含版本管理与权限控制）、Agent 可插拔、配置热更新、LLM 动态路由

---

## 2. 系统架构总览
┌─────────────┐
│ 用户输入 │
└──────┬──────┘
│
┌──────────▼──────────┐
│ 公共合规过滤器 │ 同步，<50ms
│ (正则 + LLM二审) │
└──────────┬──────────┘
│
┌──────────▼──────────┐
│ 公共 Memory 检索 │ 画像 / 合规规则 / 交互日志
│ (Supervisor 统一拉取) │
└──────────┬──────────┘
│
┌──────────▼──────────┐
│ Supervisor │ 意图识别 · 路由 · 编排
│ (LLM 动态路由+规则兜底) │
└──────────┬──────────┘
│
┌────────────────────┼────────────────────┐
│ │ │
┌────────▼────────┐ ┌───────▼────────┐ ┌───────▼────────┐
│ LoanAdvisor │ │ RiskAssessment │ │ AfterLoan │
│ (咨询 + 申请) │ │ Agent │ │ Agent │
└────────┬────────┘ └───────┬────────┘ └───────┬────────┘
│ │ │
└────────────────────┼────────────────────┘
│
┌───────────▼───────────┐
│ 公共工具层 │ Tool Registry
│ (权限 + 版本管理) │
└───────────┬───────────┘
│
┌───────────▼───────────┐
│ 公共 RAG 层 │ 知识库召回
│ (请求级缓存) │
└───────────────────────┘

异步链路：
响应返回 → 消息入队 (Redis Streams)
→ ProfileManager 消费者（关键实体同步，非关键异步）
→ FeedbackLoop 消费者（数据采集从 Phase 1 开始）
→ 交互日志消费者（写入 Milvus）

text

**公共基础设施**（不改动现有代码核心逻辑）：
- Memory 检索：用户画像、合规规则、交互日志。**Supervisor 统一拉取后注入子 Agent，子 Agent 不自行查询**。
- RAG 检索：知识库稠密 / 稀疏 / 术语召回 + RRF 融合 + 重排序 + 压缩。**同一轮对话内 query 缓存复用**。
- 合规过滤器：由现有 `compliance_guard_node` 逻辑提升而来，成为所有请求的**前置必经节点**。

---

## 3. Agent 定义

### 3.1 Supervisor Agent

| 属性 | 说明 |
|------|------|
| **职责** | 意图识别、路由分发、多 Agent 编排、兜底回复 |
| **输入** | 用户原始消息 + Memory 检索结果 |
| **输出** | 路由目标 + `AgentContext` 结构体 |
| **路由策略** | **LLM 动态路由为主**：系统提示中注入所有子 Agent 的能力描述，由 LLM 根据用户意图自主选择目标 Agent。**静态规则兜底**：仅覆盖明确合规黑名单、纯问候、简单感谢等快速放行或直接回复场景。新增 Agent 时只需更新提示词，无需修改路由代码。 |
| **核心能力** | - 意图分类（咨询、申请、风险评估、贷后操作、合规问题）<br>- 意图澄清（模糊时追问，连续两次无法理解主动转人工）<br>- 多 Agent 并行/串行编排<br>- 所有 Agent 不匹配时的兜底策略<br>- Memory 统一拉取并组装 `AgentContext` 注入子 Agent |

### 3.2 LoanAdvisor Agent

| 属性 | 说明 |
|------|------|
| **职责** | 产品咨询、申请引导、额度测算、利率查询 |
| **来源** | 现有 `retrieval_knowledge_node` + `call_model_node` + 知识库（FAQ、产品手册、流程指南） |
| **独有工具** | 额度试算、利率查询、还款计划生成、产品对比表、材料清单生成、申请表预填 |

### 3.3 RiskAssessment Agent

| 属性 | 说明 |
|------|------|
| **职责** | 风险评估、征信解读、合规政策解答、风险评分 |
| **来源** | 现有合规规则记忆 + 监管政策知识库 |
| **独有工具** | DTI 计算、征信报告解析（模拟）、风险评分卡、合规条文检索 |

### 3.4 AfterLoan Agent

| 属性 | 说明 |
|------|------|
| **职责** | 贷后操作指引、提前还款计算、展期评估、结清证明生成 |
| **来源** | 现有知识库中贷后管理、还款、合同相关章节 |
| **独有工具** | 提前还款试算、结清证明模板生成、展期评估、还款计划表生成 |

### 3.5 HumanHandoff Agent

| 属性 | 说明 |
|------|------|
| **职责** | 人工转接的处理入口，生成转接上下文摘要 |
| **触发条件** | 见第 5.3 节断点清单 |
| **输入** | 当前对话历史 + 触发原因 |
| **输出** | 转接消息 + 人工坐席上下文摘要 |

---

## 4. AgentContext 统一上下文结构

所有子 Agent 接收标准化的上下文，由 Supervisor 组装，避免格式不一致导致信息丢失。

```python
class AgentContext:
    user_id: str                          # 用户 ID
    session_id: str                       # 会话 ID
    current_query: str                    # 当前用户问题
    user_profile_summary: str             # 脱敏画像摘要（来自 Memory 检索）
    compliance_warnings: List[str]        # 合规警告列表
    conversation_summary: str             # 最近 N 轮对话摘要（来自交互日志）
    retrieved_knowledge: Optional[str]    # RAG 结果（按需填充）
    agent_instruction: str                # Supervisor 给该 Agent 的具体指令
5. 公共工具层（Tool Registry）
所有工具统一注册，Agent 按需调用，工具本身不归属任何 Agent。

5.1 权限模型
每个 Tool 定义时包含 allowed_agents 字段，Agent 调用前进行权限校验，越权调用记录安全事件并返回标准化错误。

工具名	允许调用的 Agent
calculate_monthly_payment	LoanAdvisor, AfterLoan
calculate_max_loan_amount	LoanAdvisor, RiskAssessment
calculate_prepayment	AfterLoan
calculate_dti	RiskAssessment, LoanAdvisor
query_interest_rate	LoanAdvisor
query_product_detail	LoanAdvisor
query_regulation	RiskAssessment
generate_material_checklist	LoanAdvisor
generate_repayment_schedule	LoanAdvisor, AfterLoan
generate_settlement_certificate	AfterLoan
generate_risk_report	RiskAssessment
5.2 版本管理
每个 Tool 定义包含 version 字段（语义化版本，如 1.0.0）。Tool Registry 支持多版本并存。Agent 注册时声明依赖的 Tool 版本范围（如 >=1.0.0, <2.0.0）。工具实现变更时发布新版本，旧版本保留至所有 Agent 升级完成。

5.3 计算类工具
工具名	版本	功能	输入	输出
calculate_monthly_payment	1.0.0	等额本息/等额本金月供	本金、年利率、期限、还款方式	月供、总利息
calculate_max_loan_amount	1.0.0	DTI 反推最高可贷额度	月收入、现有月供	最高额度
calculate_prepayment	1.0.0	提前还款试算	剩余本金、利率、已还期数	节省利息、违约金
calculate_dti	1.0.0	负债率计算	月收入、月债务合计	DTI 百分比
5.4 查询类工具
工具名	版本	功能	输入	输出
query_interest_rate	1.0.0	最新 LPR 及加点政策	产品类型、期限	参考利率区间
query_product_detail	1.0.0	产品详细信息	产品名	额度、期限、条件
query_regulation	1.0.0	监管法规检索	法规名或关键词	条文摘要
5.5 生成类工具
工具名	版本	功能	输入	输出
generate_material_checklist	1.0.0	申请材料清单	贷款类型	材料列表
generate_repayment_schedule	1.0.0	完整还款计划表	本金、利率、期限、方式	期次、月供、本金、利息
generate_settlement_certificate	1.0.0	结清证明模板	用户信息、贷款编号	结清证明文本
generate_risk_report	1.0.0	综合风险评估	用户画像	风险等级、建议
6. 流程设计
6.1 正常流程
合规过滤：用户输入先经过公共合规过滤器，命中 BLOCK 直接返回拦截消息，命中 WARN 标记并继续。

Memory 检索：Supervisor 统一拉取用户画像、合规规则、近期交互日志。

Supervisor 路由：LLM 基于消息和上下文判断目标 Agent，静态规则兜底。组装 AgentContext。

Agent 执行：子 Agent 接收 AgentContext，按需调用 RAG 检索和工具，生成回复。

响应返回：Supervisor 整合结果返回用户。

异步处理：消息入队，ProfileManager 同步提取关键实体 + 异步提取非关键实体，FeedbackLoop 记录埋点数据。

6.2 多 Agent 编排示例
text
用户：“我想贷50万装修，但我征信有过一次逾期，还能申请吗？”
  → Supervisor 识别到：产品咨询 + 风险评估
  → 并行调用 LoanAdvisor + RiskAssessment
  → 整合双方输出
text
用户：“帮我算一下100万30年等额本息月供多少”
  → Supervisor 识别到：LoanAdvisor
  → LoanAdvisor 调用 calculate_monthly_payment(1000000, 4.20%, 30, "等额本息")
  → 返回精确月供数字
6.3 Human-in-the-Loop 断点清单
触发条件	动作	超时策略（默认 5 分钟）
合规过滤器命中 BLOCK，用户提出申诉或要求人工复核	转人工客服，附带命中规则说明	自动返回“正在处理中，稍后通知”
风险评分 > 阈值（可配置）	转人工客户经理	自动放行，标记风险等级
申请金额 > 100 万	标记人工复核	自动放行，标记人工复核
征信含“连三累六”	AI 不给出确定性结论，转人工	返回标准拒绝话术
用户消息中包含投诉敏感词（“投诉”“银监会”“12378”等）	立即转人工，标记高优先级	返回客服热线和工单编号
用户显式要求转人工	直接转接	返回客服热线和工单编号
Supervisor 连续两次无法理解用户意图	主动转人工	自动兜底回复 + 建议拨打客服
实现方式：LangGraph interrupt 机制，Supervisor 图中对应位置暂停，等待人工确认后继续或终止。超时次数纳入 human_handoff_timeout_total 指标。

7. 异步链路设计
text
主链路响应返回后:
  → 消息入队 (Redis Streams)
    → ProfileManager 消费者：画像更新
    → FeedbackLoop 消费者：质量数据采集 + 分析
    → 交互日志消费者：写入 Milvus
7.1 ProfileManager（同步 + 异步混合 + 缓存一致性）
关键实体（同步提取，立即可用）：

实体	原因
monthly_income	直接影响额度计算和 DTI 评估
annual_income	同上
occupation	影响产品推荐和风险评估
loan_purpose	影响合规判断和产品匹配
credit_history	影响风险评分和审批结论
liability_amount	影响 DTI 计算
existing_bank_relation	影响利率优惠
非关键实体（异步提取，允许延迟）：

实体	原因
education	辅助信息
marital_status	辅助信息
vehicle	辅助信息
household_registration	辅助信息
social_security	辅助信息
缓存一致性策略（先写后删）：

向量写入 Milvus 成功后 → 立即删除对应缓存键（profile_summary:{user_id}）

下次读取时自动重建（利用现有 @custom_cached 装饰器）

关键实体同步提取路径保证写入 → 删缓存在同一请求周期内完成

异步提取路径同样遵循先写后删，允许秒级延迟

7.2 FeedbackLoop（数据采集从 Phase 1 开始）
Phase 1 即实现的数据采集埋点：

事件	触发条件	写入字段
knowledge_miss	RAG 检索返回空结果	query, timestamp, session_id
routing_detail	每次 Supervisor 路由	intent, target_agent, confidence, user_id, route_method
negative_feedback	用户连续两轮否定回答或要求转人工	session_id, previous_messages, reason
compliance_action	合规过滤器触发拦截/警告	action_type, rule_id, query
后续阶段：消费者逐步完善分析能力（路由准确率统计、知识库盲区识别、用户满意度趋势），但数据采集从第一天起就开始积累。

8. 缓存一致性设计
多 Agent 并发场景下，采用“先写后删”策略保证缓存与向量库一致：

写入路径：ProfileManager 写入 Milvus → 成功 → 立即调用 cache.invalidate(profile_summary, user_id) 删除缓存

读取路径：get_profile_summary 缓存未命中 → 从 Milvus 查询 → 重建缓存

关键实体同步路径：写入和删缓存在同一请求周期内完成，下一轮对话立即可见

异步路径：消费消息 → 写入 Milvus → 删除缓存，允许秒级延迟

RAG 请求级缓存：同一轮对话内复用，TTL 绑定 session_id + query_hash，会话结束自动失效

9. 审计日志规范
金融场景对审计日志有强制要求，以下审计事件必须记录：

审计事件	记录内容	保留期限
Supervisor 路由决策	trace_id, query, target_agent, confidence, route_method（LLM/规则）	3 年
合规拦截详情	trace_id, query, rule_id, rule_name, action（BLOCK/WARN/APPEND）	5 年
工具调用	trace_id, tool_name, tool_version, caller_agent, params（脱敏）, result_summary	3 年
人工断点触发与处理	trace_id, trigger_reason, handoff_time, resolution, operator_id	5 年
格式要求：

所有审计日志携带 trace_id 用于全链路追踪

统一为 JSON 格式，写入独立的审计日志文件或专用存储

敏感字段（如手机号、身份证）需脱敏后记录

10. 配置管理规范
所有 Agent 配置统一在 config/rules/ 下，按模块分文件：

text
config/rules/
├── supervisor.yaml           # Supervisor 提示词、路由规则
├── loan_advisor.yaml         # LoanAdvisor 提示词、工具列表、工具版本范围
├── risk_assessment.yaml      # RiskAssessment 提示词、工具列表
├── after_loan.yaml           # AfterLoan 提示词、工具列表
└── tool_registry.yaml        # 工具定义（含版本、allowed_agents、输入输出 schema）
变更流程：提交 PR → Code Review → 合并 → ConfigRegistry 自动热加载（复用现有 watchdog 机制）

运行时热更新：灰度 N 值等运行时参数通过 Redis 或配置中心下发，无需重启

版本控制：所有配置文件纳入 Git 管理，变更历史可追溯

11. 现有代码复用映射
现有模块	新系统中的位置	改动程度
retrieve_memory_node	公共 Memory 检索层	无改动，仅调整调用位置到 Supervisor
retrieval_knowledge_node	公共 RAG 检索层	增加请求级缓存
compliance_guard_node	公共合规过滤器（前置）	提升为前置节点，增加 LLM 二审兜底
call_model_node	各 Agent 内部的生成逻辑	拆分为 Agent 各自的生成节点
extract_profile_node	ProfileManager（同步部分）	拆分为关键/非关键两条路径
log_interaction_node	交互日志消费者（异步）	移到消息队列消费者
retrieval_rule_router	Supervisor 路由（规则兜底层）	保留，但仅覆盖强匹配场景
全部 Tool 配置 (YAML)	Tool Registry	增加 version、allowed_agents 字段
build_graph	旧系统链路（保留备用）	完整保留，不修改
12. 开发实施路径
Phase 0：灰度基础设施（前置）
灰度路由层实现（按 user_id 哈希分流）

新旧链路并行运行框架

Phase 1：骨架搭建（P0）
Supervisor 节点实现（LLM 动态路由 + 规则兜底）

AgentContext 结构定义与组装逻辑

公共合规过滤器提升为前置节点

LoanAdvisor Agent（集成现有 RAG + Memory）

公共 Tool Registry 框架 + 2 个核心工具（calculate_monthly_payment, query_interest_rate）

FeedbackLoop 数据采集埋点（knowledge_miss, routing_detail, negative_feedback, compliance_action）

ProfileManager 关键实体同步提取路径 + “先写后删”缓存策略

审计日志基础设施（统一 trace_id 生成与记录）

Phase 2：业务扩展（P1）
RiskAssessment Agent + 工具

AfterLoan Agent + 工具

Human-in-the-Loop 断点实现（LangGraph interrupt + 超时降级策略）

多 Agent 编排逻辑（并行/串行）

Phase 3：异步化与闭环（P2）
ProfileManager 非关键实体异步化

FeedbackLoop 消费者分析逻辑

监控与告警（Prometheus 指标 + 阈值绑定）

13. 监控与告警
指标	说明	告警条件	告警级别
supervisor_routing_total	路由到各 Agent 的次数分布	—	—
supervisor_routing_error_rate	LLM 路由与规则兜底不一致率	> 5%	P1
agent_duration_seconds	每个 Agent 的处理延迟 P50/P99	P99 > 旧系统 × 1.3	P2
tool_call_total	每个工具的调用次数和成功率	—	—
tool_call_failure_rate	工具调用失败率	> 3%	P2
tool_permission_denied_total	工具越权调用次数	> 0 即告警	P1
compliance_block_total	合规拦截次数	突降 50%	P0
human_handoff_total	人工转接次数，按原因分类	—	—
human_handoff_timeout_total	人工断点超时次数	> 10 次/小时	P1
knowledge_miss_rate	知识检索空结果占比	> 30%	P2
feedback_negative_rate	用户负面反馈率	—	—
cache_invalidation_failure_total	缓存删除失败次数	> 0 即告警	P1
14. 与传统单 Agent 的核心区别
维度	单 Agent	多 Agent
功能边界	统一回答所有问题	各域 Agent 能执行实际操作
合规	节点内判断	前置过滤器 + 独立 Agent
工具调用	无	计算、生成、查询，带权限与版本控制
人工介入	无	7 个明确断点触发条件，超时自动降级
上下文传递	隐式 state	标准化 AgentContext 结构
画像提取	同步阻塞	关键实体同步 + 非关键异步，先写后删缓存
路由	N/A	LLM 动态路由 + 规则兜底
质量闭环	无	从 Phase 1 开始采集埋点，持续分析
审计	无	4 类审计事件，全链路 trace_id
配置管理	单文件 YAML	按 Agent 分文件，Git 管理 + Code Review
扩展性	修改核心节点	新增 Agent + 更新 Supervisor 提示词 + 配置新增
15. 风险与应对
风险	影响	应对措施
Supervisor 路由错误	回答不准确，用户困惑	LLM 动态路由 + 规则兜底双保险；路由结果全量记录便于复盘
画像异步化延迟	用户刚提供的信息未立即生效	关键实体同步提取 + 立即删除缓存强制刷新
多 Agent 并发缓存竞态	读取到旧画像	“先写后删”策略，写入成功后删缓存，下次读取重建
多 Agent 重复检索	Milvus 负载增加	Supervisor 统一拉取注入；RAG 请求级缓存
工具越权调用	安全风险	Tool Registry 权限校验 + 越权告警
人工断点阻塞	用户等待过久	5 分钟超时自动降级，超时次数监控告警
消息队列故障	画像和日志丢失	消息持久化 + 死信队列；关键实体有同步路径兜底
工具版本不兼容	Agent 调用异常	Tool Registry 多版本并存，Agent 声明依赖版本范围