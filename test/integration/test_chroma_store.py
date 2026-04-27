# author hgh
# version 1.0
import logging
import os.path
import shutil

from config.constants import MemoryType, EvidenceType, MemoryStatus, GeneralFieldNames, MemorySource
from memory.long_term_memory_store import LongTermMemoryStore
from memory.memory_vector_store.chroma_vector_store import ChromaVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_DIR = "./test_chroma_test1"


def cleanup():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)


def print_memory(label: str, memories: list):
    print(f"\n{label}")
    if not memories:
        print(" (空)")
        return
    for mem in memories:
        id = mem.get(GeneralFieldNames.ID, "N/A")
        print(f"ID:{id}")

        similarity = mem.get(GeneralFieldNames.SIMILARITY, 0.0)
        print(f"SIMILARITY:{similarity}")

        meta = mem.get("metadata", {})
        entity_key = meta.get("entity_key", "N/A")
        text = mem.get("text", "N/A")
        print(f"[{entity_key}] {text}")
        for k, v in meta.items():
            if k not in [GeneralFieldNames.ENTITY_KEY, GeneralFieldNames.TEXT]:
                print(f"- {k}: {v}")

def test_base_crud():
    print("\n" + "=" * 60)
    print("测试 1: 基本增删改查")
    print("=" * 60)

    vector_stoer = ChromaVectorStore(TEST_DIR)
    store = LongTermMemoryStore(vector_stoer)

    user_id = "test_user_crud"

    print("\n1.1 插入画像记忆")
    store.add_memory(
        user_id=user_id,
        content="客户年收入约50万元",
        memory_type=MemoryType.USER_PROFILE,
        entity_key="income",
        metadata={
            GeneralFieldNames.SOURCE: MemorySource.CHAT_EXTRACTION.value,
            GeneralFieldNames.CONFIDENCE: 0.8,
            GeneralFieldNames.EVIDENCE_TYPE: EvidenceType.EXPLICIT_STATEMENT,
            GeneralFieldNames.STATUS: MemoryStatus.ACTIVE.value,
            GeneralFieldNames.PERMANENT: False
        }
    )

    print("\n1.2 查询验证....")
    all_memories = store.get_all_user_profile_memories(user_id)
    print_memory("所有活跃记忆", all_memories)
    assert len(all_memories) == 1, f"预期 1 条记忆，实际 {len(all_memories)}"

    print("\n1.3 更新记忆内容....")
    mem_id = all_memories[0]["id"]
    store.update_memory_status(mem_id, MemoryType.USER_PROFILE, MemoryStatus.ACTIVE.value,
                               {GeneralFieldNames.EVIDENCE_TYPE: EvidenceType.CREDIT_REPORT.value})

    print("\n1.4 再次查询活跃记忆...")
    all_memories = store.get_all_user_profile_memories(user_id,status=MemoryStatus.ACTIVE.value)
    print_memory("更新后记忆内容", all_memories)
    assert len(all_memories) == 1, f"预期 1 条活跃记忆，实际 {len(all_memories)}"

    print("\n1.5 更新记忆状态....")
    mem_id = all_memories[0]["id"]
    store.update_memory_status(mem_id, MemoryType.USER_PROFILE, MemoryStatus.SUPERSEDED.value)

    print("\n1.6 再次查询活跃记忆...")
    all_memories = store.get_all_user_profile_memories(user_id,status=MemoryStatus.ACTIVE.value)
    print_memory("更新后活跃记忆", all_memories)
    assert len(all_memories) == 0, f"预期 0 条活跃记忆，实际 {len(all_memories)}"

    print("\n 1.7 删除所有记忆")
    store.delete_user_memories(user_id, MemoryType.USER_PROFILE)
    all_memories = store.get_all_user_profile_memories(user_id)
    assert len(all_memories) == 0, f"预期 0 条记忆，实际 {len(all_memories)}"

    store.delete_user_memories(user_id="test_user_crud",memory_type=MemoryType.USER_PROFILE)

    print("\n✅ 基本增删改查测试通过")

def test_search():
    print("\n" + "=" * 60)
    print("测试 2: 语义搜索")
    print("=" * 60)

    vector_store = ChromaVectorStore(TEST_DIR)
    store = LongTermMemoryStore(vector_store=vector_store)

    user_id = "test_user_search"
    print("\n2.1 插入测试数据...")
    test_data = [
        ("income", "客户年收入约50万元", 0.82),
        ("income", "客户周收入约3万元", 0.81),
        ("income", "客户月收入约10万元", 0.83),
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

    print("\n2.2 搜索'收入'相关记忆")
    results = store.search_memory(user_id,"我的月收入是多少",memory_type=MemoryType.USER_PROFILE,limit=5)
    print_memory("搜索结果", results)
    assert len(results) > 0, "搜索结果不应为空"
    assert any("收入" in r.get("text", "") for r in results), "应包含收入相关的记忆"

    print("\n2.3 搜索置信度>=0.9的记忆")
    results = store.search_memory(
        user_id=user_id,
        query="职业 贷款",
        memory_type=MemoryType.USER_PROFILE,
        limit=3,
        min_confidence=0.9,
    )
    print_memory("高置信度搜索结果", results)

    store.delete_user_memories(user_id="test_user_search", memory_type=MemoryType.USER_PROFILE)

def test_conflict_resolution():
    """测试冲突解决机制"""
    print("\n" + "=" * 60)
    print("测试 3: 冲突解决（高置信度覆盖低置信度）")
    print("=" * 60)

    vector_store = ChromaVectorStore(TEST_DIR)
    store = LongTermMemoryStore(vector_store=vector_store)

    user_id = "test_user_conflict"

    store.delete_user_memories(user_id=user_id, memory_type=MemoryType.USER_PROFILE)

    # 1. 插入低置信度记忆
    print("\n3.1 插入低置信度 income 记忆 (0.6)...")
    store.add_memory(
        user_id=user_id,
        content="客户年收入约40万元",
        memory_type=MemoryType.USER_PROFILE,
        entity_key="income",
        metadata={"source": "chat_extraction", "confidence": 0.6,"evidence_type": EvidenceType.EXPLICIT_STATEMENT},
    )

    # 2. 查看当前状态
    all_memories = store.get_all_user_profile_memories(user_id)
    print("第一条记忆插入后",all_memories)
    assert len(all_memories) == 1, f"预期 1 条记忆，实际 {len(all_memories)}"
    assert all_memories[0]["metadata"]["confidence"] == 0.6

    # 3. 插入高置信度记忆（应触发覆盖）
    print("\n3.2 插入高置信度 income 记忆 (0.95)...")
    store.add_memory(
        user_id=user_id,
        content="客户再次提及，年收入约50万元",
        memory_type=MemoryType.USER_PROFILE,
        entity_key="income",
        metadata={"source": "chat_extraction", "confidence": 0.95,"evidence_type": EvidenceType.EXPLICIT_STATEMENT},
    )

    # 4. 验证结果
    all_memories = store.get_all_user_profile_memories(user_id)
    print_memory("高置信度记忆插入后", all_memories)

    active_memories = [m for m in all_memories if m["metadata"]["status"] == "active"]
    superseded_memories = [m for m in all_memories if m["metadata"]["status"] == "superseded"]
    print(f"\n活跃记忆数: {len(active_memories)}, 被覆盖记忆数: {len(superseded_memories)}")

    assert len(active_memories) == 1, f"预期 1 条活跃记忆，实际 {len(active_memories)}"
    assert active_memories[0]["metadata"]["confidence"] == 0.95, "活跃记忆置信度应为 0.95"
    assert "约50万元" in active_memories[0]["text"], "活跃记忆应为高置信度版本"

    #5. 插入高权重证据类型同置信度记忆
    print("\n3.2 插入高权重证据类型 income 记忆 (BANK_STATEMENT)...")
    store.add_memory(
        user_id=user_id,
        content="客户提供工资流水，年收入确认为65万元",
        memory_type=MemoryType.USER_PROFILE,
        entity_key="income",
        metadata={"source": "chat_extraction", "confidence": 0.95,"evidence_type": EvidenceType.BANK_STATEMENT},
    )

    # 6. 验证结果
    all_memories = store.get_all_user_profile_memories(user_id)
    print_memory("高权重证据类型记忆插入后", all_memories)
    active_memories = [m for m in all_memories if m["metadata"]["status"] == "active"]
    superseded_memories = [m for m in all_memories if m["metadata"]["status"] == "superseded"]
    print(f"\n活跃记忆数: {len(active_memories)}, 被覆盖记忆数: {len(superseded_memories)}")

    assert len(active_memories) == 1, f"预期 1 条活跃记忆，实际 {len(active_memories)}"
    assert len(superseded_memories) == 2, f"预期 2 条挂起记忆，实际 {len(superseded_memories)}"
    assert active_memories[0]["metadata"]["evidence_type"] == EvidenceType.BANK_STATEMENT.value, "活跃记忆证据类型应为 BANK_STATEMENT"
    assert "确认为65万元" in active_memories[0]["text"], "活跃记忆应为高权重证据类型版本"

    store.delete_user_memories(user_id="test_user_conflict", memory_type=MemoryType.USER_PROFILE)

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

def test_apply_forgotten():
    print("\n" + "=" * 60)
    print("测试 4: 记忆遗忘")
    print("=" * 60)

    vector_store = ChromaVectorStore(TEST_DIR)
    store = LongTermMemoryStore(vector_store=vector_store)

    user_id = "test_user_forgotten"

    #1.1 插入多条不同实体的记忆
    print("\n4.1 插入多条不同实体的记忆...")
    store.add_memory(
        user_id=user_id,
        content="客户年收入约50万元",
        memory_type=MemoryType.USER_PROFILE,
        entity_key="income",
        metadata={"source": "chat_extraction", "confidence": 0.8,"last_accessed_at": "2026-03-25T09:00:00"},
    )
    store.add_memory(
        user_id=user_id,
        content="客户是互联网公司产品经理",
        memory_type=MemoryType.USER_PROFILE,
        entity_key="occupation",
        metadata={"source": "chat_extraction", "confidence": 0.9,"last_accessed_at": "2026-04-11T09:00:00"},
    )

    #1.2 查询插入后状态
    results = store.get_all_user_profile_memories(user_id)
    print_memory("记忆现状:",results)

    #1.3 执行记忆遗忘
    count = store.apply_forgetting(MemoryType.USER_PROFILE,user_id)

    #1.4 遗忘后查询
    results = store.get_all_user_profile_memories(user_id)
    print_memory("遗忘后记忆现状:", results)
    forgotten_mem = [m for m in results if m["metadata"]["status"] == MemoryStatus.FORGOTTEN.value]
    assert len(forgotten_mem) == 1, f"预期遗忘一条记忆，实际遗忘{len(forgotten_mem)}条"
    assert forgotten_mem[0]["text"] == "客户年收入约50万元", f"预期遗忘'客户年收入约50万元'实际遗忘{forgotten_mem[0]["text"]}"

    store.delete_user_memories(user_id, MemoryType.USER_PROFILE)


if __name__ == '__main__':
    # test_search()
    # test_conflict_resolution()
    # test_get_by_entity()
    test_apply_forgotten()