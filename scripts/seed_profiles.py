"""
用户画像测试数据导入脚本
适配 Step 2 新增字段：evidence_type, effective_date, expires_at
"""

import json
import logging
from datetime import datetime
from memory.memory_vector_store.chroma_db.chroma_vector_store import ChromaVectorStore
from memory.models.memory_constant.constants import MemoryType, MemorySource, MemoryStatus, EvidenceType

from memory.memory_store.long_term_memory_store import LongTermMemoryStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def import_profiles(jsonl_path: str):
    """导入用户画像测试数据"""
    vector_store = ChromaVectorStore("../chromadb")
    store = LongTermMemoryStore(vector_store=vector_store)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error(f"第 {line_num} 行 JSON 解析失败: {e}")
                continue

            # 构建符合 UserProfileMetadata 的元数据字典
            metadata = {
                "type": MemoryType.USER_PROFILE.value,
                "source": data.get("source", MemorySource.CHAT_EXTRACTION.value),
                "confidence": data.get("confidence", 0.8),
                "status": MemoryStatus.ACTIVE.value,
                "permanent": data.get("permanent", False),
                # Step 2 新增字段
                "evidence_type": data.get("evidence_type", EvidenceType.EXPLICIT_STATEMENT.value),
                "effective_date": data.get("effective_date", data.get("timestamp", datetime.now().isoformat())),
                "expires_at": data.get("expires_at"),
            }

            try:
                memory_id = store.add_memory(
                    user_id=data["user_id"],
                    content=data["content"],
                    memory_type=MemoryType.USER_PROFILE,
                    entity_key=data["entity_key"],
                    metadata=metadata
                )
                logger.info(f"✅ 导入成功: {data['user_id']} - {data['entity_key']} (ID: {memory_id[:8]}...)")
            except Exception as e:
                logger.error(f"❌ 导入失败: {data.get('user_id')} - {data.get('entity_key')}: {e}")


def verify_import():
    """验证导入结果"""
    vector_store = ChromaVectorStore("../chromadb")
    store = LongTermMemoryStore(vector_store=vector_store)

    print("\n" + "=" * 60)
    print("验证导入结果")
    print("=" * 60)

    for user_id in ["test_user_001", "test_user_002"]:
        print(f"\n👤 用户: {user_id}")
        memories = store.get_all_user_profile_memories(user_id, status="active")
        if memories:
            for mem in memories:
                meta = mem["metadata"]
                effective = meta.get('effective_date')
                if isinstance(effective, datetime):
                    effective_str = effective.strftime('%Y-%m-%d')
                else:
                    effective_str = str(effective)[:10] if effective else 'N/A'
                print(f"  - [{meta.get('entity_key', 'N/A')}] {mem['text']} "
                      f"(置信度: {meta.get('confidence')}, 证据: {meta.get('evidence_type')}, "
                      f"生效: {effective_str})")
        else:
            print("  (无活跃记忆)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="../data/test/user_profiles_test.jsonl", help="JSONL 文件路径")
    parser.add_argument("--verify", action="store_true", help="仅验证，不导入")
    args = parser.parse_args()

    if args.verify:
        verify_import()
    else:
        import_profiles(args.file)
        print("\n✅ 导入完成，开始验证...")
        verify_import()
    #verify_import()
