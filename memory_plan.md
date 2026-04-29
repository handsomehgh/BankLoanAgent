生产级记忆系统演进路线图（完善版）
一、当前状态与目标
你现在拥有一个基于 LangGraph 的单 Agent 系统，已经具备：

完善的记忆存储与检索（LongTermMemoryStore + MilvusVectorStore）

合规前置拦截、多策略检索

同步的画像提取节点，但存在重复调用、无差别提取问题

最终目标：构建一个支持 RAG、多 Agent 协作的生产级记忆系统。

二、核心改造维度总览
维度	当前问题	目标
提取策略	每轮调用 LLM，窗口固定	增量游标 + 轻量过滤 + 已知画像注入
质量控制	依赖存储层去重，无提取级抑制	提取层已知画像对比，输出强校验
异步解耦	同步阻塞主链路	节点门控 → 异步 Worker
知识增强	仅对话历史	引入 RAG，为生成和提取提供外部知识
多 Agent 协同	单 Agent 独占记忆	工具化共享，统一记忆总线
可观测性	较少指标	关键漏斗指标、结构日志、低成本告警
隐私与合规	画像明文存储/传输	敏感字段脱敏、摘要注入控制、接口治理
游标健壮性	无游标	增量游标 + 回退策略 + 全局消息序号
三、分阶段时序改造计划
阶段一：同步门控优化（第 1-2 周，即时落地）
目标：不改动架构，在现有节点内部将 LLM 调用量降低 90%+，同时引入轻量监控与隐私保护。

核心改造项
步骤	改造内容	涉及文件/模块
1.1	AgentState 增加 last_extracted_message_index 字段（全局游标）	config/constants.py
1.2	实现 get_new_user_messages()：基于 message_index 获取增量用户消息，兼容旧 ID（回退）	agent/nodes/extract_profile_node.py
1.3	实现 likely_contains_profile() 规则过滤器（正则 + 可扩展）	同上
1.4	在 LongTermMemoryStore 中添加 get_profile_summary()：获取当前活跃画像摘要，支持脱敏、长度限制	memory/long_term_memory_store.py
1.5	修改 EXTRACT_PROMPT，增加 {known_profile} 占位及指令，引导 LLM 只提取新增/更新信息	prompt/extract_prompt.py
1.6	重构 extract_profile_node 主逻辑：增量消息获取 → 轻量过滤 → 构建目标+上下文 → 注入已知画像 → LLM 调用 → 存储 → 更新游标	agent/nodes/extract_profile_node.py
1.7	优化 infer_evidence_type：仅使用增量用户消息，避免全量扫描	memory/classifiers/
1.8	修复 log_interaction_node 切片错误（取最近 N 条）	agent/nodes/log_interaction_node.py
1.9	消息 message_index 全局递增分配（在消息创建处注入）	agent/nodes/call_model_node.py、用户入口
1.10	引入轻量结构化日志监控：漏斗计数器（进入/跳过/LLM调用/有效提取/回退）	agent/nodes/extract_profile_node.py
1.11	游标回退策略：配置化回退窗口大小（默认 10），回退时告警	config/settings.py、提取节点
1.12	已知画像脱敏与长度控制：敏感字段隐藏，最多注入 10 条，总字符 ≤ 500	memory/long_term_memory_store.py
交付物
增量游标 + 轻量过滤，90% 以上对话不再触发 LLM 提取

提取质量提升（已知画像注入、避免重复提取）

游标回退机制与告警，避免极端场景下大量重复提取

轻量监控日志，可在日志平台直接聚合关键指标

画像脱敏，满足合规基本要求

阶段一实施优先级与排期（共 10 个工作日）
优先级	内容	时间窗口
P0	消息 ID / index 可靠性校验、全局序号分配	第 1 天
P0	画像脱敏与长度控制	第 1 天
P1	轻量监控埋点（结构化日志）	第 2-3 天
P1	回退窗口配置化 + 告警	第 2-3 天
P1	跨会话游标决策与接口预留	第 4 天
P2	集成测试、灰度验证	第 5-10 天
阶段二：RAG 引入（第 3-4 周）
目标：为生成和画像提取注入外部知识，提升回答与提取准确度。

步骤	改造内容	涉及文件/模块
2.1	搭建外部知识库（产品文档、政策），向量化存储（Milvus 新建集合或独立索引）	memory/、新工具脚本
2.2	扩展 BaseRetriever 为 HybridRetriever，支持多源检索（向量记忆 + 外部知识）	retriever/
2.3	修改 retrieve_memory_node，在 formatted_context 中加入 external_knowledge 字段	agent/nodes/retrieve_memory_node.py
2.4	调整 SYSTEM_TEMPLATE，将外部知识注入 prompt（可选项）	prompt/system_prompt.py
2.5	灰度开关：通过配置或关键词判断是否启用 RAG 检索	config/settings.py
2.6	（可选）在画像提取 prompt 中注入术语解释（基于 RAG 结果）	prompt/extract_prompt.py
交付物：

回答更专业、合规规则检索更全面

画像提取在领域术语场景中准确率提升

灰度控制，不影响核心链路稳定性

阶段三：异步 Worker 化（第 5-6 周，视流量需求触发）
目标：将画像提取与交互日志从同步对话流中解耦，适应规模化。

步骤	改造内容	涉及文件/模块
3.1	引入消息队列（Kafka 或 Redis Stream）	基础设施
3.2	改造 extract_profile_node：通过门控后仅发送事件到队列，不再同步调用 LLM	agent/nodes/extract_profile_node.py
3.3	改造 log_interaction_node：同样改为发送事件	agent/nodes/log_interaction_node.py
3.4	开发独立的 ProfileExtractionWorker，消费事件、执行 LLM 提取、合并存储	新模块 workers/
3.5	开发独立的 InteractionLogWorker，消费事件、生成摘要并存储	同上
3.6	Worker 自行管理游标（存储于 Redis 或 memory_store），支持重试与死信	workers/
3.7	增加 Worker 监控指标（队列积压、延迟、失败率）	监控系统
交付物：

对话端到端延迟完全不受记忆处理影响

提取逻辑独立，可横向扩展

游标管理从 State 迁移到独立存储

阶段四：多 Agent 与工具化（第 7 周起，依据业务需求）
目标：支持多 Agent 协作，记忆与 RAG 成为共享基础设施。

步骤	改造内容	涉及文件/模块
4.1	架构选型：Supervisor + 子 Agent 或完全分布式，每个 Agent 为独立 Graph/Node	agent/
4.2	将记忆访问封装为 Tool：
- get_user_profile(user_id)
- search_knowledge(query)
- log_user_event(...)	tools/
4.3	将合规检索保留为强制预处理节点（注入所有 Agent 系统消息）	agent/nodes/compliance_guard_node.py 复用
4.4	画像提取 Worker 升级为多源消费者，接收来自不同 Agent 的对话事件	workers/profile_extraction_worker.py
4.5	画像元数据增加 source_agent 字段，记录提取来源	memory/models/memory_data/memory_schema.py
4.6	AgentState 扩展多 Agent 路由与子任务状态（不影响记忆接口）	agent/state.py
4.7	全链路监控增加 agent_id 维度	监控系统
交付物：

新 Agent 接入只需调用 Tools，无需关心记忆实现

统一的用户画像跨 Agent 实时共享

提取 Worker 支持多 Agent 事件源

四、各阶段依赖关系图
text
阶段一（同步门控 + 游标 + 监控）
   ↓
阶段二（RAG）←──── 依赖阶段一稳定的提取与记忆接口
   ↓
阶段三（异步Worker）←── 依赖阶段一的门控过滤，可独立演进
   ↓
阶段四（多Agent）←──── 依赖阶段二、三提供的共享记忆与知识接口
五、持续优化维度（贯穿所有阶段）
监控指标：提取触发率、有效率、LLM 成本、合并冲突率、画像覆盖率、Worker 队列积压、回退告警次数

隐私合规：敏感信息过滤、用户画像查看/删除接口、审计日志、摘要脱敏

配置开关：各增强功能均可通过配置开关灰度上线，避免全量风险

回滚能力：每阶段改动向后兼容，异常时可快速回退至前一阶段

游标机制：从 State 游标 → 独立存储游标（阶段三）平滑迁移

