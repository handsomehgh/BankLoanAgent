# 多 Agent 银行贷款助手系统设计方案 v2.3（最终全量）

## 1. 设计目标

将现有单 Agent 系统升级为多 Agent 协作系统，在保留公共 Memory / RAG 能力的基础上，实现：

- **专业化分工**：每个 Agent 聚焦独立业务域，配备专属工具
- **生产级可靠性**：合规前置、人工兜底、异步解耦、质量闭环、缓存一致性
- **可扩展性**：工具注册制（代码即定义、版本管理、权限控制）、Agent 可插拔、配置热更新、LLM 动态路由

**设计遵循《多 Agent 系统开发规范与准则》（下文简称《规范》）。**

---

## 2. 系统架构总览

### 2.1 架构模式

采用 **Supervisor + Subgraph** 模式：

- Supervisor 负责意图识别、路由分发和结果整合
- 各子 Agent（LoanAdvisor、RiskAssessment、AfterLoan、HumanHandoff）均实现为独立的 `StateGraph` 子图，嵌入 Supervisor 主图
- 公共能力（Memory 检索、RAG 检索、合规过滤）前置为共享层，由 Supervisor 统一调用并注入子 Agent，避免重复查询
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
│ [独立子图] │ │ [独立子图] │ │ [独立子图] │
└────────┬────────┘ └───────┬────────┘ └───────┬────────┘
│ │ │
└────────────────────┼────────────────────┘
│
┌───────────▼───────────┐
│ 公共工具层 │ Tool Registry + ToolExecutor
│ (代码即定义、权限、版本) │
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

### 2.2 状态管理

- **隔离原则**：每个子 Agent 拥有独立的 `State` 定义（如 `LoanAdvisorState`），与 Supervisor 状态完全隔离。
- **数据传递**：通过统一的 `AgentContext` 结构体在 Supervisor 与子 Agent 之间传递必要数据（用户画像摘要、合规警告、RAG 结果等）。
- **通信约束**：子 Agent 之间**禁止直接通信**，所有协调必须通过 Supervisor 中转。需要跨 Agent 协作时，由 Supervisor 并行调用并整合结果。

### 2.3 子图集成规范

每个子 Agent 的 `StateGraph` 独立定义，编译后作为节点嵌入 Supervisor 主图：

```python
# 子Agent定义（独立文件）
loan_advisor_graph = StateGraph(LoanAdvisorState)
loan_advisor_graph.add_node("generate", generate_response)
loan_advisor_graph.set_entry_point("generate")
loan_advisor_graph.set_finish_point("generate")
loan_advisor_compiled = loan_advisor_graph.compile()

# Supervisor主图集成
supervisor_graph = StateGraph(SupervisorState)
supervisor_graph.add_node("LoanAdvisor", loan_advisor_compiled)  # 子图作为节点

# 调用时通过 AgentContext 映射数据
supervisor_graph.add_edge("supervisor_route", "LoanAdvisor")
数据映射方式：Supervisor 调用子图时，将 SupervisorState 中的字段映射到子图的 AgentContext 参数中。子图返回后，将结果写回 SupervisorState。

2.4 结果整合（Fan-out / Fan-in）
当 Supervisor 并行调用多个子 Agent 时，通过独立的结果整合节点（ResultAggregator）将多路输出合并为统一回复：

收集所有子 Agent 的回复

若所有回复成功，按优先级排序拼接

若部分超时或失败，用兜底文本替代失败分支

最终输出一条完整回复

text
Supervisor 路由 → 并行：
                   ├── LoanAdvisor → 结果A ─┐
                   └── RiskAssessment → 结果B ─┤
                                              ↓
                                    结果整合节点 → 最终回复
3. 提示词管理（符合《规范》第3条）
3.1 存储方式
所有 Agent 的系统提示词（角色定义、工具列表、行为约束）存储在独立的 YAML 配置文件中。

按 Agent 分文件管理：config/rules/supervisor.yaml、loan_advisor.yaml 等。

变更流程：提交 PR → Code Review → 合并 → ConfigRegistry 自动热加载。

3.2 内容规范
每个 Agent 的提示词必须包含：

角色定义：明确 Agent 的职责和专业领域

可用工具列表：声明依赖的工具及版本范围

行为约束：禁止承诺、免责声明、敏感话题处理

输出格式要求：如“涉及利率时必须注明‘仅供参考，以银行最终审批为准’”

3.3 敏感信息管理
禁止在提示词中硬编码任何敏感数据（如 API 密钥、密码、内部 IP）

所有敏感信息通过环境变量或 ConfigRegistry 的安全注入机制提供

3.4 合规审核
涉及金融合规敏感内容的提示词（如合规红线说明），须经风控部门审核后方可上线

3.5 组装方式
任务提示词模板（带变量占位符的 Python 文件）与 YAML 系统提示词协作，运行时注入：

python
# config/prompts/system_prompt.py
SYSTEM_TEMPLATE = """
{agent_role}

## 已知用户画像(长期记忆)
{user_profile}

## 合规红线提醒
{compliance_rule}

## 近期交互历史
{interaction_log}

## 知识库内容
{business_knowledge}

## 行为准则
1. 基于已知画像提供个性化回答；信息不足时主动、礼貌地询问。
2. 严禁做出任何确定性承诺（如“一定批贷”、“保证通过”）。
3. 涉及具体利率、额度等信息，务必注明“仅供参考，以银行最终审批为准”。
4. 回答专业、清晰、有温度，避免使用晦涩的金融术语。
"""
运行时从 YAML 读入 agent_role 注入占位符。

4. Agent 定义
4.1 Supervisor Agent
属性	说明
职责	意图识别、路由分发、多 Agent 编排、兜底回复
输入	用户原始消息 + Memory 检索结果
输出	路由目标 + AgentContext 结构体
路由策略	LLM 动态路由为主：系统提示中注入所有子 Agent 的能力描述，由 LLM 根据用户意图自主选择目标 Agent。静态规则兜底：仅覆盖明确合规黑名单、纯问候、简单感谢等快速放行或直接回复场景。新增 Agent 时只需更新提示词，无需修改路由代码。
状态	SupervisorState（含 messages、user_id、last_extracted_index 等）
4.2 LoanAdvisor Agent
属性	说明
职责	贷款产品咨询、申请引导、额度测算、利率查询、还款计划生成
不处理	征信评估、逾期解读、负债率计算、风险评级；贷后管理、提前还款、结清证明、解押流程；合规法规检索
状态	LoanAdvisorState（独立子图，含 AgentContext、内部消息等）
工具	calculate_monthly_payment, query_interest_rate, calculate_max_loan_amount, generate_material_checklist
4.3 RiskAssessment Agent
属性	说明
职责	风险评估、征信解读、负债率计算、合规政策解答、风险评分
不处理	产品推荐、利率查询、申请流程；提前还款计算、结清证明生成
状态	RiskAssessmentState
工具	calculate_dti, query_regulation, generate_risk_report
4.4 AfterLoan Agent
属性	说明
职责	贷后操作指引、提前还款试算、展期评估、结清证明生成、还款计划查询
不处理	产品咨询、贷款申请、额度测算；征信解读、风险评估
状态	AfterLoanState
工具	calculate_prepayment, generate_repayment_schedule, generate_settlement_certificate
4.5 HumanHandoff Agent
属性	说明
职责	人工转接入口，生成转接上下文摘要。仅在断点触发时被调用
不处理	任何贷款业务咨询，不调用任何业务工具
状态	HumanHandoffState
所有子 Agent 通信必须通过 Supervisor 中转，禁止直接调用。

5. AgentContext 统一上下文结构
所有子 Agent 接收标准化的上下文，由 Supervisor 组装。强制包含全链路追踪和审计依赖：

python
class AgentContext:
    user_id: str                          # 用户ID
    session_id: str                       # 会话ID
    trace_id: str                         # 全链路追踪ID（必须）
    current_query: str                    # 当前用户问题
    user_profile_summary: str             # 脱敏画像摘要
    compliance_warnings: List[str]        # 合规警告列表
    conversation_summary: str             # 最近N轮对话摘要
    retrieved_knowledge: Optional[str]    # RAG结果，按需填充
    agent_instruction: str                # Supervisor给该Agent的指令
    audit_logger: AuditLogger             # 审计日志记录器（必须）
trace_id 和 audit_logger 为强制字段，在所有 Agent 之间统一传递，确保每个操作可追踪、可审计。

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

7. 工具系统（符合《规范》第2条）
所有工具必须严格遵守以下强制性要求。

7.1 工具定义
python
from langchain_core.tools import tool

@tool(version="1.0.0", tags=["LoanAdvisor", "AfterLoan"])
def calculate_monthly_payment(
    principal: float,
    annual_rate: float,
    term_years: int,
    method: str = "等额本息"
) -> dict:
    """计算等额本息或等额本金的月供和总利息。

    Args:
        principal: 贷款本金（元）
        annual_rate: 年利率（%，如 4.2 表示 4.2%）
        term_years: 贷款期限（年）
        method: 还款方式，可选 "等额本息" 或 "等额本金"
    Returns:
        monthly_payment: 月供（元）
        total_interest: 总利息（元）
    """
    # ... 计算逻辑 ...
强制性要求：

要求	说明
必须使用 @tool 装饰器	禁止手动构建工具对象
必须包含 version 参数	语义化版本号，如 "1.0.0"
必须包含 tags 参数	声明允许调用的 Agent 名称列表，如 ["LoanAdvisor"]
必须提供完整类型注解	所有参数和返回值
必须提供 docstring	含功能描述、参数说明（类型+单位）、返回值说明
Schema 自动生成	禁止手工编写 YAML 中的 input_schema / output_schema
7.2 工具调用
必须通过手写 ToolExecutor 调用，禁止在 Agent 节点内直接调用工具函数。

python
# ✅ 正确
result = tool_executor.execute(
    tool_name="calculate_monthly_payment",
    args={"principal": 1000000, "annual_rate": 4.2, "term_years": 30, "method": "等额本息"},
    caller_agent="LoanAdvisor",
    trace_id=trace_id
)

# ❌ 禁止
result = calculate_monthly_payment(**args)
7.3 调用流程（ToolExecutor 强制执行）
ToolExecutor.execute() 必须严格按以下顺序执行：

权限校验：检查 caller_agent 是否在工具的 tags 列表中。越权立即拒绝，记录安全事件，递增 tool_permission_denied_total。

参数验证：使用工具自带的 args_schema 进行 JSON Schema 校验，不通过立即返回错误。

审计日志（调用前）：记录 trace_id、tool_name、tool_version、caller_agent、输入参数（敏感字段脱敏）。

执行工具：调用工具函数，捕获异常。

审计日志（调用后）：记录执行结果摘要、耗时、是否成功。

异常处理：工具执行异常时返回标准化错误结果，记录告警。

7.4 YAML 配置精简
YAML 中仅保留 Agent 对工具的依赖声明：

yaml
# loan_advisor.yaml
tools:
  - name: calculate_monthly_payment
    version_range: ">=1.0.0,<2.0.0"
  - name: query_interest_rate
    version_range: ">=1.0.0"
工具的 Schema、权限、描述均从 @tool 装饰的函数中自动提取，不在 YAML 中重复维护。

7.5 版本管理
同一工具的多版本可同时注册

Agent 通过 version_range 声明兼容版本

不再被任何 Agent 依赖的旧版本标记为 deprecated，观察 2 周后移除

8. 异步链路设计
text
主链路响应返回后:
  → 消息入队 (Redis Streams)
    → ProfileManager 消费者：画像更新
    → FeedbackLoop 消费者：质量数据采集 + 分析
    → 交互日志消费者：写入 Milvus
8.1 ProfileManager（同步 + 异步混合 + 缓存一致性）
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

向量写入 Milvus 成功后 → 立即删除对应缓存键

下次读取时自动重建（利用现有 @custom_cached 装饰器）

关键实体同步路径保证写入 → 删缓存在同一请求周期内完成

异步路径同样遵循先写后删，允许秒级延迟

8.2 FeedbackLoop（数据采集从 Phase 1 开始）
Phase 1 即实现的数据采集埋点：

事件	触发条件	写入字段
knowledge_miss	RAG 检索返回空结果	query, timestamp, session_id
routing_detail	每次 Supervisor 路由	intent, target_agent, confidence, user_id, route_method
negative_feedback	用户连续两轮否定回答或要求转人工	session_id, previous_messages, reason
compliance_action	合规过滤器触发拦截/警告	action_type, rule_id, query
9. 缓存一致性设计
多 Agent 并发场景下，采用“先写后删”策略：

写入路径：ProfileManager 写入 Milvus → 成功 → 立即调用 cache.invalidate(profile_summary, user_id) 删除缓存

读取路径：get_profile_summary 缓存未命中 → 从 Milvus 查询 → 重建缓存

关键实体同步路径：写入和删缓存在同一请求周期内完成

异步路径：消费消息 → 写入 Milvus → 删除缓存，允许秒级延迟

RAG 请求级缓存：同一轮对话内复用，TTL 绑定 session_id + query_hash，会话结束自动失效

10. 审计日志规范
审计事件	记录内容	保留期限
Supervisor 路由决策	trace_id, query, target_agent, confidence, route_method	3 年
合规拦截详情	trace_id, query, rule_id, rule_name, action	5 年
工具调用	trace_id, tool_name, tool_version, caller_agent, 输入参数（脱敏）, 结果摘要, 耗时	3 年
人工断点触发与处理	trace_id, trigger_reason, handoff_time, resolution, operator_id	5 年
所有审计日志携带 trace_id，统一为 JSON 格式，写入独立的审计日志文件或专用存储。敏感字段脱敏后记录。

11. 配置管理（符合《规范》）
text
config/rules/
├── supervisor.yaml           # Supervisor 提示词、路由规则
├── loan_advisor.yaml         # LoanAdvisor 提示词、工具依赖
├── risk_assessment.yaml      # RiskAssessment 提示词、工具依赖
├── after_loan.yaml           # AfterLoan 提示词、工具依赖
└── tool_registry.yaml        # 工具注册清单（仅工具名称列表）
变更流程：提交 PR → Code Review → 合并 → ConfigRegistry 自动热加载

运行时热更新：修改 YAML 无需重启

版本控制：所有配置文件纳入 Git 管理

敏感数据：不得写入配置文件，通过环境变量或安全注入提供

12. 开发实施路径
阶段	核心任务
Phase 0	基础设施：配置管理、抽象基类、消息队列、监控
Phase 1	骨架：合规过滤器、Supervisor、LoanAdvisor 子图、Tool Registry、审计埋点、ProfileManager 同步
Phase 2	扩展：RiskAssessment 子图、AfterLoan 子图、多 Agent 编排、HumanHandoff 断点、补充工具
Phase 3	异步化与闭环：ProfileManager 异步、交互日志异步、FeedbackLoop、告警完善
13. 监控与告警
指标	告警条件	级别
supervisor_routing_error_rate	> 5%	P1
tool_call_failure_rate	> 3%	P2
tool_permission_denied_total	> 0	P1
compliance_block_total	突降 50%	P0
human_handoff_timeout_total	> 10 次/小时	P1
knowledge_miss_rate	> 30%	P2
agent_duration_seconds P99	> 旧系统 × 1.3	P2
14. 风险与应对
风险	应对措施
Supervisor 路由错误	LLM + 规则双保险，路由日志持续复盘
子 Agent 直接通信	代码审查 + 架构约束，禁止绕过 Supervisor
画像异步化延迟	关键实体同步提取 + 立即删缓存强制刷新
多 Agent 并发缓存竞态	“先写后删”策略
多 Agent 重复检索	Supervisor 统一注入，RAG 请求级缓存
工具越权调用	ToolExecutor 权限校验 + 告警
人工断点阻塞	5 分钟超时自动降级
工具版本不兼容	多版本并存，Agent 声明版本范围
提示词泄露敏感信息	YAML 审核机制，禁止硬编码