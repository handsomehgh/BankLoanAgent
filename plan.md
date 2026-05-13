# 🏦 银行贷款智能助手（BankLoanAgent）—— 完整开发计划与架构总览

## 📋 项目概述
本项目旨在构建一个面向金融领域、具备短期与多类型长期记忆、合规拦截、可扩展检索能力的企业级 AI Agent 系统，最终演进为多 Agent 协作、集成 RAG 知识库与工具调用的智能客服助手。

---

## ✅ 已完成部分（V1.0 — 记忆系统核心）

| 模块                | 功能说明                                                         | 关键技术点                                                                 |
| :------------------ | :--------------------------------------------------------------- | :------------------------------------------------------------------------- |
| **数据模型**        | `MemoryBase` 基类 + `UserProfileMemory`/`InteractionLogMemory`/`ComplianceRuleMemory` 三种专属模型 | Pydantic 强类型校验、枚举管理、`default_factory`、`field_validator` 容错   |
| **映射层**          | `MemoryToStorageMapper` / `StorageToMemoryMapper`                 | 基于 `model_fields` 自动序列化/反序列化；`datetime`/`Enum`/`list`/`dict` 安全转换；`MappingError` 异常体系 |
| **查询构建器**      | `QueryBuilder` 抽象 + `ChromaQueryBuilder` / `MilvusQueryBuilder` 实现 | 插件化设计，将通用 `Query` 对象转换为不同数据库的原生查询语法                |
| **向量存储抽象**    | `BaseVectorStore` 接口                                            | 定义 `add`/`search`/`get`/`update`/`delete` 统一契约                       |
| **Chroma 存储**     | `ChromaVectorStore`                                               | 显式传入 `OpenAIEmbeddingFunction` 统一向量空间；`get`/`search` 格式适配；update 安全合并 |
| **Milvus 存储**     | `MilvusVectorStore`                                               | 混合检索、关键词检索、MMR、动态策略推断；pymilvus 2.6.x API 适配；BM25 Function 集成 |
| **业务逻辑层**      | `LongTermMemoryStore`                                             | 三种记忆的增删改查；冲突解决（置信度+证据权重）；时间衰减遗忘；死信队列；交互日志摘要 |
| **检索器**          | `VectorRetriever`                                                 | 协调三种记忆的检索策略，与底层存储解耦                                     |
| **Agent 节点**      | `retrieve_memory_node` / `extract_profile_node` / `log_interaction_node` / `compliance_guard_node` | LangGraph 图编排；证据类型自动推断；情感检测；合规拦截与免责声明追加        |
| **分类器**          | 证据类型推断 + 情感检测                                           | 规则兜底 + LLM 精判 + 缓存；YAML 配置热加载                                |
| **初始化脚本**      | `init_milvus_collections.py`                                      | 差异化 Schema 设计；BM25 Function 配置；稠密/稀疏/标量/全文索引            |
| **数据导入**        | `seed_profiles.py` / `import_compliance.py`                      | 测试数据导入与验证                                                         |
| **配置管理**        | `config.py` (Pydantic Settings)                                   | 环境变量校验；多 Provider 切换                                             |
| **常量管理**        | `constants.py`                                                    | 全项目枚举统一，消除魔法字符串                                             |
| **错误处理**        | 重试装饰器 + 死信队列 + 降级策略 + 自定义异常                      | `@retry_on_failure`；`MemoryWriteFailedError` 等                            |
| **测试**            | ChromaVectorStore 核心功能测试                                    | 增删改查、语义搜索、冲突解决、按实体键查询                                 |

---

## 🧭 短期规划（V1.1 — 记忆系统完善）

| 任务                     | 预计耗时 | 优先级 |
| :----------------------- | :------- | :----- |
| 完善单元测试（模型、映射器、查询构建器） | 2天      | P0     |
| 完善集成测试（LongTermMemoryStore 全流程） | 2天      | P0     |
| MilvusVectorStore 功能测试 | 1天      | P1     |
| 端到端测试（完整对话流程） | 1天      | P1     |
| 性能基准测试（Chroma vs Milvus） | 1天      | P2     |

---

## 📦 中期规划（V2.0 — 长期记忆升级 + RAG 知识库）

| 任务                     | 预计耗时 | 说明                                                                 |
| :----------------------- | :------- | :------------------------------------------------------------------- |
| **Milvus 生产部署**      | 2天      | 从 Milvus Lite 切换至 Docker/K8s 集群                                |
| **RAG 知识库模块**       | 5天      | 离线文档导入脚本；`KnowledgeRetriever`；Agent 中融合知识检索         |
| **短期记忆持久化升级**   | 1天      | `SQLiteSaver → PostgresSaver` 迁移，支持多实例会话共享               |
| **多模态记忆预留**       | 1天      | Schema 中增加图片 URL 字段；嵌入模型扩展                             |

---

## 🚀 长期规划（V3.0 — 多 Agent + 图数据库 + 工具调用）

| 任务                     | 预计耗时 | 说明                                                                 |
| :----------------------- | :------- | :------------------------------------------------------------------- |
| **多 Agent 架构**        | 5天      | Supervisor + 咨询/评估/风控/人工 Agent；`SharedMemoryBus` 共享记忆    |
| **记忆压缩策略**         | 1天      | `MemoryCompressorNode` 自动摘要早期对话，防止 Token 溢出              |
| **图数据库集成**         | 3天      | Neo4j 存储客户关系、产品关联、合规规则链                              |
| **GraphRAG**             | 3天      | 向量检索 + 图检索混合，回答复杂推理问题                               |
| **工具调用**             | 3天      | 利率计算器、申请表推送、征信查询模拟                                 |

---

## 🛠️ 持续完善（V4.0 — 工程化与可观测性）

| 任务                     | 说明                                                                 |
| :----------------------- | :------------------------------------------------------------------- |
| CI/CD 流水线             | GitHub Actions 自动化测试与部署                                      |
| 可观测性集成             | OpenTelemetry + Jaeger 链路追踪；Prometheus + Grafana 指标监控；ELK 日志分析 |
| 安全性增强               | Milvus RBAC 多租户权限控制                                           |
| 性能调优                 | 索引参数优化、缓存策略、并发处理                                     |

---

## 📊 性能指标规划

| 指标                     | 目标值        | 验证方式                       |
| :----------------------- | :------------ | :----------------------------- |
| 记忆插入 P99 延迟        | < 50ms        | `test_vector_benchmark.py`     |
| 语义检索 P99 延迟        | < 200ms       | 同上                           |
| 语义搜索 Recall@5        | > 0.85        | 人工标注测试集                 |
| LLM 调用成功率           | > 99.9%       | 监控面板                       |
| 死信队列积压量           | < 10 条/小时  | 告警规则                       |

---

## 🏗️ 完整项目架构图

```mermaid
graph TD
    subgraph 用户层
        U[Streamlit 前端界面]
    end

    subgraph 短期记忆层
        STM[ConversationBufferWindowMemory]
        CHECK[SQLiteCheckpointer<br>生产升级PostgresSaver]
        COMPRESS[MemoryCompressorNode<br>自动摘要压缩]
    end

    subgraph Agent编排层
        SUP[Supervisor 主管Agent<br>意图识别 / 任务分发]
        CONSULT[业务咨询Agent]
        ASSESS[资质评估Agent]
        RISK[风控审核Agent]
        HUMAN[人机协同Agent]
    end

    subgraph 抽象接口层
        BMS[BaseMemoryStore]
        BR[BaseRetriever]
        BVS[BaseVectorStore]
        QB[QueryBuilder]
    end

    subgraph 业务逻辑层
        LMS[LongTermMemoryStore<br>冲突解决 / 衰减遗忘 / 死信队列]
        VR[VectorRetriever<br>检索策略调度]
        KR[KnowledgeRetriever<br>RAG知识检索]
    end

    subgraph 共享记忆总线
        SBUS[SharedMemoryBus<br>注入同一个LongTermMemoryStore实例]
    end

    subgraph 存储实现层
        CVS[ChromaVectorStore]
        MVS[MilvusVectorStore]
        N4J[(Neo4j 图数据库)]
    end

    subgraph 数据模型层
        MODELS[UserProfileMemory<br>InteractionLogMemory<br>ComplianceRuleMemory]
        MAPPER[MemoryToStorageMapper<br>StorageToMemoryMapper]
    end

    subgraph 基础设施层
        LLM[LLM Provider<br>DeepSeek / Qwen]
        EMBED[Embedding Model<br>text-embedding-v4]
        DLQ[死信队列]
        OTEL[OpenTelemetry]
        JAEGER[Jaeger 链路追踪]
        PROM[Prometheus + Grafana]
        ELK[ELK 日志分析]
    end

    U <--> SUP
    SUP --> CONSULT & ASSESS & RISK & HUMAN
    CONSULT & ASSESS & RISK & HUMAN <--> LMS & VR & KR
    VR --> BMS
    LMS --> BMS
    BMS --> BVS
    BVS --> CVS & MVS
    KR --> BVS
    LMS --> MODELS & MAPPER
    MODELS <--> MAPPER
    MAPPER --> CVS & MVS
    LMS --> DLQ
    MVS <-.-> N4J
    LLM & EMBED --> CONSULT & ASSESS & RISK & HUMAN
    SUP & CONSULT & ASSESS & RISK & HUMAN --> OTEL
    OTEL --> JAEGER & PROM
    OTEL --> ELK
    SUP --> SBUS
    SBUS --> LMS

    style SUP fill:#4A90D9,color:#fff
    style CONSULT fill:#27AE60,color:#fff
    style ASSESS fill:#27AE60,color:#fff
    style RISK fill:#E67E22,color:#fff
    style HUMAN fill:#E74C3C,color:#fff
    style LMS fill:#8E44AD,color:#fff
    style VR fill:#8E44AD,color:#fff
    style BVS fill:#2C3E50,color:#fff
    style CVS fill:#16A085,color:#fff
    style MVS fill:#16A085,color:#fff
    style N4J fill:#C0392B,color:#fff
    style MODELS fill:#F39C12,color:#fff
    style MAPPER fill:#F39C12,color:#fff
    style STM fill:#3498DB,color:#fff
    style CHECK fill:#3498DB,color:#fff
    style COMPRESS fill:#3498DB,color:#fff
    style KR fill:#8E44AD,color:#fff
    style SBUS fill:#2ECC71,color:#fff
    style OTEL fill:#95A5A6,color:#fff
    style JAEGER fill:#95A5A6,color:#fff
    style PROM fill:#95A5A6,color:#fff
    style ELK fill:#95A5A6,color:#fff