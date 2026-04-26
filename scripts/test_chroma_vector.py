# author hgh
# version 1.0
"""
ChromaVectorStore 测试脚本
验证增删改查核心功能及冲突解决机制
"""

import os
import sys
import uuid
import logging
from pathlib import Path

# 将项目根目录加入 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.memory_vector_store.chroma_db.chroma_vector_store import ChromaVectorStore
from memory.memory_store.long_term_memory_store import LongTermMemoryStore
from memory.models.memory_constant.constants import MemoryType, MemoryStatus


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 使用临时目录避免污染现有数据
TEST_DIR = "./test_chroma_temp"


def cleanup():
    """清理测试数据"""
    import shutil
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)


def print_memory(label: str, memories: list):
    """格式化打印记忆列表"""
    print(f"\n{label}:")
    if not memories:
        print("  (空)")
        return
    for mem in memories:
        meta = mem.get("metadata", {})
        status = meta.get("status", "N/A")
        print(f"  - [{meta.get('entity_key', 'N/A')}] {mem.get('text', '')} "
              f"(状态: {status}, 置信度: {meta.get('confidence', 'N/A')})")


def test_basic_crud():
    """测试基本的增删改查"""
    print("\n" + "=" * 60)
    print("测试 1: 基本增删改查")
    print("=" * 60)

    vector_store = ChromaVectorStore(TEST_DIR)
    store = LongTermMemoryStore(vector_store=vector_store)

    user_id = "test_user_crud"
    mem_id = str(uuid.uuid4())

    # 1. 插入一条记忆
    print("\n1.1 插入画像记忆...")
    store.add_memory(
        user_id=user_id,
        content="客户年收入约50万元",
        memory_type=MemoryType.USER_PROFILE,
        entity_key="income",
        metadata={
            "source": "chat_extraction",
            "confidence": 0.8,
            "evidence_type": "explicit_statement",
        }
    )

    # 2. 查询验证
    print("\n1.2 查询验证...")
    all_memories = store.get_all_user_profile_memories(user_id)
    print_memory("所有活跃记忆", all_memories)

    assert len(all_memories) == 1, f"预期 1 条记忆，实际 {len(all_memories)}"

    # 3. 更新状态
    print("\n1.3 更新记忆状态为 superseded...")
    mem_id = all_memories[0]["id"]
    store.update_memory_status(mem_id, MemoryType.USER_PROFILE, MemoryStatus.SUPERSEDED.value)

    # 4. 再次查询，应该为空
    print("\n1.4 再次查询活跃记忆...")
    all_memories = store.get_all_user_profile_memories(user_id)
    print_memory("更新后活跃记忆", all_memories)
    assert len(all_memories) == 0, f"预期 0 条活跃记忆，实际 {len(all_memories)}"

    # 5. 删除
    print("\n1.5 删除所有记忆...")
    store.delete_user_memories(user_id, MemoryType.USER_PROFILE)
    all_memories = store.get_all_user_profile_memories(user_id)
    assert len(all_memories) == 0, f"预期 0 条记忆，实际 {len(all_memories)}"

    print("\n✅ 基本增删改查测试通过")


def test_search():
    """测试语义搜索"""
    print("\n" + "=" * 60)
    print("测试 2: 语义搜索")
    print("=" * 60)

    vector_store = ChromaVectorStore(TEST_DIR)
    store = LongTermMemoryStore(vector_store=vector_store)

    user_id = "test_user_search"

    # 插入多条记忆
    print("\n2.1 插入测试数据...")
    test_data = [
        ("income", "客户年收入约50万元", 0.8),
        ("occupation", "客户是互联网公司产品经理", 0.9),
        ("loan_purpose", "客户计划申请200万元房贷", 0.85),
    ]
    for entity_key, content, confidence in test_data:
        store.add_memory(
            user_id=user_id,
            content=content,
            memory_type=MemoryType.USER_PROFILE,
            entity_key=entity_key,
            metadata={"source": "chat_extraction", "confidence": confidence},
        )

    # 2. 语义搜索
    print("\n2.2 搜索'收入'相关记忆...")
    results = store.search_memory(
        user_id=user_id,
        query="收入 工资",
        memory_type=MemoryType.USER_PROFILE,
        limit=2,
    )
    print_memory("搜索结果", results)
    assert len(results) > 0, "搜索结果不应为空"
    assert any("收入" in r.get("text", "") for r in results), "应包含收入相关的记忆"

    # 3. 带置信度过滤的搜索
    print("\n2.3 搜索置信度≥0.9的记忆...")
    results = store.search_memory(
        user_id=user_id,
        query="职业 贷款",
        memory_type=MemoryType.USER_PROFILE,
        limit=3,
        min_confidence=0.9,
    )
    print_memory("高置信度搜索结果", results)

    # 清理
    store.delete_user_memories(user_id, MemoryType.USER_PROFILE)

    print("\n✅ 语义搜索测试通过")


def test_conflict_resolution():
    """测试冲突解决机制"""
    print("\n" + "=" * 60)
    print("测试 3: 冲突解决（高置信度覆盖低置信度）")
    print("=" * 60)

    vector_store = ChromaVectorStore(TEST_DIR)
    store = LongTermMemoryStore(vector_store=vector_store)

    user_id = "test_user_conflict"

    # 1. 插入低置信度记忆
    print("\n3.1 插入低置信度 income 记忆 (0.6)...")
    store.add_memory(
        user_id=user_id,
        content="客户年收入约40万元",
        memory_type=MemoryType.USER_PROFILE,
        entity_key="income",
        metadata={"source": "chat_extraction", "confidence": 0.6},
    )

    # 2. 查看当前状态
    all_memories = store.get_all_user_profile_memories(user_id)
    print_memory("第一条记忆插入后", all_memories)
    assert len(all_memories) == 1, f"预期 1 条记忆，实际 {len(all_memories)}"
    assert all_memories[0]["metadata"]["confidence"] == 0.6

    # 3. 插入高置信度记忆（应触发覆盖）
    print("\n3.2 插入高置信度 income 记忆 (0.95)...")
    store.add_memory(
        user_id=user_id,
        content="客户提供工资流水，年收入确认为55万元",
        memory_type=MemoryType.USER_PROFILE,
        entity_key="income",
        metadata={"source": "bank_statement", "confidence": 0.95},
    )

    # 4. 验证结果
    all_memories = store.get_all_user_profile_memories(user_id)
    print_memory("高置信度记忆插入后", all_memories)

    active_memories = [m for m in all_memories if m["metadata"].get("status") == "active"]
    superseded_memories = [m for m in all_memories if m["metadata"].get("status") == "superseded"]

    print(f"\n活跃记忆数: {len(active_memories)}, 被覆盖记忆数: {len(superseded_memories)}")

    # 断言：只有一条活跃记忆，且是置信度更高的那条
    assert len(active_memories) == 1, f"预期 1 条活跃记忆，实际 {len(active_memories)}"
    assert active_memories[0]["metadata"]["confidence"] == 0.95, "活跃记忆置信度应为 0.95"
    assert "确认为55万元" in active_memories[0]["text"], "活跃记忆应为高置信度版本"

    # 清理
    store.delete_user_memories(user_id, MemoryType.USER_PROFILE)

    print("\n✅ 冲突解决测试通过")


def test_get_by_entity():
    """测试按实体键查询"""
    print("\n" + "=" * 60)
    print("测试 4: 按实体键查询")
    print("=" * 60)

    vector_store = ChromaVectorStore(TEST_DIR)
    store = LongTermMemoryStore(vector_store=vector_store)

    user_id = "test_user_entity"

    # 插入多条不同实体的记忆
    print("\n4.1 插入多条不同实体的记忆...")
    store.add_memory(
        user_id=user_id,
        content="客户年收入约50万元",
        memory_type=MemoryType.USER_PROFILE,
        entity_key="income",
        metadata={"source": "chat_extraction", "confidence": 0.8},
    )
    store.add_memory(
        user_id=user_id,
        content="客户是互联网公司产品经理",
        memory_type=MemoryType.USER_PROFILE,
        entity_key="occupation",
        metadata={"source": "chat_extraction", "confidence": 0.9},
    )

    # 2. 按 entity_key 查询
    print("\n4.2 按 entity_key='income' 查询...")
    results = store.get_memory_by_entity(user_id, "income", MemoryStatus.ACTIVE.value)
    print_memory("income 记忆", results)
    assert len(results) == 1, f"预期 1 条记忆，实际 {len(results)}"
    assert results[0]["text"] == "客户年收入约50万元"

    # 清理
    store.delete_user_memories(user_id, MemoryType.USER_PROFILE)

    print("\n✅ 按实体键查询测试通过")


if __name__ == "__main__":
    try:
        # 清空之前的测试数据
        cleanup()

        # 运行测试
        test_basic_crud()
        test_search()
        test_conflict_resolution()
        test_get_by_entity()

        print("\n" + "=" * 60)
        print("🎉 所有测试通过！")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception("测试执行出错")
        sys.exit(1)
    finally:
        # 清理临时数据
        cleanup()
        if os.path.exists(TEST_DIR):
            print(f"\n提示: 测试数据保留在 {TEST_DIR}，可手动删除")