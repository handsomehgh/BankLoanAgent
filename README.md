# 🏦 Loan Advisor Agent — 生产级银行贷款智能助手

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

一个具备**短期记忆**、**长期用户画像记忆**、**记忆冲突解决**、**时间衰减遗忘机制**的企业级 AI Agent 实现，专为金融咨询场景设计。

## ✨ 核心特性

- **短期记忆管理**：基于 LangGraph 的消息窗口与状态持久化（SQLite Checkpointer）。
- **长期画像记忆**：Chroma 向量存储，支持语义检索与用户隔离。
- **记忆冲突解决**：基于 `entity_key` 的高置信度覆盖策略，软删除旧版本，保留审计线索。
- **记忆衰减与遗忘**：艾宾浩斯启发式衰减函数，自动遗忘低权重记忆，支持永久记忆保护。
- **RAG 可扩展架构**：`BaseRetriever` 抽象层，可平滑接入业务知识库、合规规则等多源上下文。
- **自我评估闭环**：生成回答后自动评分，不达标触发重写。
- **生产级工程实践**：依赖倒置、配置分离、Docker 就绪、结构化日志。

## 🧠 记忆系统架构
用户输入 → [检索多源上下文] → [生成回答] → [自我评估] → [提取新画像] → 结束
↑ ↓
长期记忆存储 (Chroma) 冲突检测 + 衰减更新