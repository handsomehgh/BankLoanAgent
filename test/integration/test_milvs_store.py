# author hgh
# version 1.0
import logging
import time

from config.global_constant.constants import MemoryType, RegistryModules
from config.global_constant.fields import CommonFields
from infra.milvus_client import MilvusClientManager
from modules.memory.memory_business_store.long_term_memory_store import LongTermMemoryStore
from modules.memory.memory_constant.constants import ProfileEntityKey, MemorySource, EvidenceType, MemoryStatus
from modules.memory.memory_constant.fields import MemoryFields
from modules.memory.memory_vector_store.milvus_memory_vector_store import MilvusMemoryVectorStore
from modules.module_services.embeddings import RobustEmbeddings
from utils.config_utils.get_config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_store() -> LongTermMemoryStore:
    registry = get_config()
    llm_config = registry.get_config(RegistryModules.LLM)
    retrieval_config = registry.get_config(RegistryModules.RETRIEVAL)
    memory_config = registry.get_config(RegistryModules.MEMORY_SYSTEM)
    embedder = RobustEmbeddings(
        api_key=llm_config.alibaba_api_key,
        model_name=llm_config.alibaba_emb_name,
        backup_model_name=llm_config.alibaba_emb_backup,
        dimensions=llm_config.dimension
    )
    milvus_client = MilvusClientManager(retrieval_config.milvus_uri)
    vector_store = MilvusMemoryVectorStore(
        milvus_client=milvus_client, embed=embedder, config=memory_config
    )

    store = LongTermMemoryStore(
        vector_store=vector_store, config=memory_config
    )
    return store


def cleanup_user(store: LongTermMemoryStore,user_id: str):
    """delete all memory data of the specified user and clean testing traces"""
    try:
        store.delete_user_memories(user_id)
    except Exception as e:
        logger.error(f"failed to delete user memory: {e}")

def print_memory(label: str, memories: list):
    print(f"\n{label}")
    if not memories:
        print(" (空)")
        return
    for mem in memories:
        id = mem.get(CommonFields.ID, "N/A")
        print(f"ID:{id}")

        similarity = mem.get(CommonFields.SIMILARITY, 0.0)
        print(f"SIMILARITY:{similarity}")

        meta = mem.get("metadata", {})
        entity_key = meta.get("entity_key", "N/A")
        text = mem.get("text", "N/A")
        print(f"[{entity_key}] {text}")
        for k, v in meta.items():
            if k not in [CommonFields.ENTITY_KEY, CommonFields.TEXT]:
                print(f"- {k}: {v}")

def test_add_profile_memory():
    print("\n[Test 1] Add user profile memory")
    TEST_USER = "test_user_add"
    store = get_store()
    try:
        mem_id = store.add_memory(
            user_id=TEST_USER,
            content="客户年收入约50万元",
            memory_type=MemoryType.USER_PROFILE,
            entity_key=ProfileEntityKey.INCOME,
            metadata={
                "source": MemorySource.CHAT_EXTRACTION.value,
                "confidence": 0.8,
                "evidence_type": EvidenceType.EXPLICIT_STATEMENT

            }
        )
        assert mem_id, "Should return valid memory ID"
        results = store.get_all_user_profile_memories(TEST_USER, status=MemoryStatus.ACTIVE)
        assert len(results) == 1, f"There should be one active memory，actual{len(results)}"
        print("  ✅ Added successfully")
        print_memory("The memory content after added",results)

        #clear user profile memory
        cleanup_user(store, TEST_USER)
    except:
        cleanup_user(store, TEST_USER)

def test_base_crud():
    print("\n" + "=" * 60)
    print("测试 1: 基本增删改查")
    print("=" * 60)

    store = get_store()
    user_id = "test_user_crud"

    print("\n1.1 插入画像记忆")
    store.add_memory(
        user_id=user_id,
        content="客户年收入约50万元",
        memory_type=MemoryType.USER_PROFILE,
        entity_key=ProfileEntityKey.INCOME,
        metadata={
            CommonFields.SOURCE: MemorySource.CHAT_EXTRACTION.value,
            CommonFields.CONFIDENCE: 0.8,
            CommonFields.EVIDENCE_TYPE: EvidenceType.EXPLICIT_STATEMENT,
            CommonFields.STATUS: MemoryStatus.ACTIVE.value,
            MemoryFields.PERMANENT: False
        }
    )

    print("\n1.2 查询验证....")
    all_memories = store.get_all_user_profile_memories(user_id)
    print_memory("所有活跃记忆", all_memories)
    assert len(all_memories) == 1, f"预期 1 条记忆，实际 {len(all_memories)}"

    print("\n1.3 更新记忆内容....")
    mem_id = all_memories[0]["id"]
    store.update_memory_status(mem_id, MemoryType.USER_PROFILE, MemoryStatus.ACTIVE,
                               {CommonFields.EVIDENCE_TYPE: EvidenceType.CREDIT_REPORT.value})

    print("\n1.4 再次查询活跃记忆...")
    all_memories = store.get_all_user_profile_memories(user_id,status=MemoryStatus.ACTIVE)
    print_memory("更新后记忆内容", all_memories)
    assert len(all_memories) == 1, f"预期 1 条活跃记忆，实际 {len(all_memories)}"

    print("\n1.5 更新记忆状态....")
    mem_id = all_memories[0]["id"]
    store.update_memory_status(mem_id, MemoryType.USER_PROFILE, MemoryStatus.SUPERSEDED)

    print("\n1.6 再次查询活跃记忆...")
    all_memories = store.get_all_user_profile_memories(user_id,status=MemoryStatus.ACTIVE)
    print_memory("更新后活跃记忆", all_memories)
    assert len(all_memories) == 0, f"预期 0 条活跃记忆，实际 {len(all_memories)}"

    print("\n 1.7 删除所有记忆")
    store.delete_user_memories(user_id=user_id,memory_type=MemoryType.USER_PROFILE)
    time.sleep(2)
    final = store.get_all_user_profile_memories(user_id)
    print_memory("final---------------",all_memories)
    assert len(final) == 0, f"预期用户{user_id}无记忆内容"

    print("\n✅ 基本增删改查测试通过")

def test_search():
    print("\n" + "=" * 60)
    print("测试 2: 语义搜索")
    print("=" * 60)

    store = get_store()
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
            entity_key=ProfileEntityKey(entity_key),
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

    store.delete_user_memories(user_id=user_id, memory_type=MemoryType.USER_PROFILE)

def test_conflict_resolution():
    """测试冲突解决机制"""
    print("\n" + "=" * 60)
    print("测试 3: 冲突解决（高置信度覆盖低置信度）")
    print("=" * 60)

    store = get_store()
    user_id = "test_user_conflict"

    # 1. 插入低置信度记忆
    print("\n3.1 插入低置信度 income 记忆 (0.6)...")
    store.add_memory(
        user_id=user_id,
        content="客户年收入约40万元",
        memory_type=MemoryType.USER_PROFILE,
        entity_key=ProfileEntityKey.INCOME,
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
        entity_key=ProfileEntityKey.INCOME,
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

    store = get_store()
    user_id = "test_user_entity"

    # 插入多条不同实体的记忆
    print("\n4.1 插入多条不同实体的记忆...")
    store.add_memory(
        user_id=user_id,
        content="客户年收入约50万元",
        memory_type=MemoryType.USER_PROFILE,
        entity_key=ProfileEntityKey.INCOME,
        metadata={"source": "chat_extraction", "confidence": 0.8},
    )
    store.add_memory(
        user_id=user_id,
        content="客户是互联网公司产品经理",
        memory_type=MemoryType.USER_PROFILE,
        entity_key=ProfileEntityKey.OCCUPATION,
        metadata={"source": "chat_extraction", "confidence": 0.9},
    )

    # 2. 按 entity_key 查询
    print("\n4.2 按 entity_key='income' 查询...")
    results = store.get_memory_by_entity(user_id, ProfileEntityKey.INCOME, MemoryStatus.ACTIVE)
    print_memory("income 记忆", results)
    assert len(results) == 1, f"预期 1 条记忆，实际 {len(results)}"
    assert results[0]["text"] == "客户年收入约50万元"

    # 清理
    store.delete_user_memories(user_id, MemoryType.USER_PROFILE)

    print("\n✅ 按实体键查询测试通过")

def test_apply_forget():
    print("\n" + "=" * 60)
    print("测试 5: 记忆遗忘")
    print("=" * 60)

    store = get_store()
    user_id = "test_user_forget"

    print("\n5.1 插入多条不同实体的记忆...")
    store.add_memory(
        user_id=user_id,
        content="客户年收入约50万元",
        memory_type=MemoryType.USER_PROFILE,
        entity_key=ProfileEntityKey.INCOME,
        metadata={"source": "chat_extraction", "confidence": 0.8,"last_accessed_at": "2026-03-10T09:00:00"},
    )
    store.add_memory(
        user_id=user_id,
        content="客户是互联网公司产品经理",
        memory_type=MemoryType.USER_PROFILE,
        entity_key=ProfileEntityKey.OCCUPATION,
        metadata={"source": "chat_extraction", "confidence": 0.9,"last_accessed_at": "2026-04-20T09:00:00"},
    )

    count = store.apply_forgetting(MemoryType.USER_PROFILE,user_id)

    assert count == 1,f"预期遗忘1条记忆,实际遗忘{count}条"

    results = store.get_all_user_profile_memories(user_id)
    print_memory("遗忘后记忆-----------------",results)
    active_mem = [m for m in results if results[CommonFields.METADATA].get("status") == "active"]
    assert len(active_mem) == 1,f"预期目前活跃记忆1条,实际{count}条"

    # 清理
    store.delete_user_memories(user_id, MemoryType.USER_PROFILE)

def test_get_active_compliance_rules():
    print("\n" + "=" * 60)
    print("测试 6: 合规记忆")
    print("=" * 60)

    store = get_store()
    rules = store.get_active_compliance_rules(15)
    assert len(rules) == 15,f"预期合规规则15条,实际{len(rules)}"
    print_memory("合规规则",rules[:3])

def test_get_user_profile_memories():
    print("\n" + "=" * 60)
    print("测试 7: 测试获取用户活跃记忆摘要")
    print("=" * 60)

    user_id = "test_user_001"
    store = get_store()

    summary = store.get_profile_summary(user_id)
    assert summary,f"预期存在用户{user_id}的活跃记忆"
    print(summary)


if __name__ == '__main__':
    # test_add_profile_memory()
    # test_base_crud()
    # test_search()
    # test_conflict_resolution()
    # test_get_by_entity()
    # test_get_active_compliance_rules()
    # test_get_user_profile_memories()
    # store = get_store()
    # # test_apply_forget()
    # user_id = "test_user_entity"
    # store.delete_user_memories(user_id=user_id, memory_type=MemoryType.USER_PROFILE)
    # res = store.get_all_user_profile_memories(user_id,MemoryType.USER_PROFILE)
    # print(res)
    # print_memory("11",res)
    store = get_store()
    result = store.get_all_user_profile_memories("test_user_011")
    print(result)