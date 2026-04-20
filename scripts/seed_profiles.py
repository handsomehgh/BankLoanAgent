# author hgh
# version 1.0
"""
用户画像测试数据导入脚本
用于验证 v1.0 记忆系统的写入、冲突解决和检索功能
"""

import json
import logging
import sys
from pathlib import Path

# 将项目根目录加入 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.chroma_store import ChromaMemoryStore
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def import_profiles(jsonl_path: str):
    """导入用户画像测试数据"""
    store = ChromaMemoryStore(
        persist_dir=config.chroma_persist_dir,
        collection_name="user_profile"
    )

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error(f"第 {line_num} 行 JSON 解析失败: {e}")
                continue

            try:
                memory_id = store.add_memory(
                    user_id=data["user_id"],
                    content=data["content"],
                    entity_key=data["entity_key"],
                    metadata={
                        "type": "user_profile",
                        "confidence": data.get("confidence", 0.7),
                        "source": data.get("source", "imported"),
                        "timestamp": data.get("timestamp"),
                        "permanent": data.get("permanent", False)
                    },
                    permanent=data.get("permanent", False)
                )
                logger.info(f"✅ 导入成功: {data['user_id']} - {data['entity_key']} (ID: {memory_id[:8]}...)")
            except Exception as e:
                logger.error(f"❌ 导入失败: {data.get('user_id')} - {data.get('entity_key')}: {e}")


def verify_import():
    """验证导入结果"""
    store = ChromaMemoryStore(
        persist_dir=config.chroma_persist_dir,
        collection_name="user_profile"
    )

    print("\n" + "=" * 60)
    print("验证导入结果")
    print("=" * 60)

    for user_id in ["test_user_001", "test_user_002"]:
        print(f"\n👤 用户: {user_id}")
        # 获取所有 active 记忆
        memories = store.get_memories_by_entity(user_id, "", status="active")
        # 由于 get_memories_by_entity 需要具体 entity_key，这里用底层方法遍历
        try:
            results = store.collection.get(
                where={"user_id": {"$eq": user_id}, "status": {"$eq": "active"}},
                include=["documents", "metadatas"]
            )
            if results["ids"]:
                for i, doc in enumerate(results["documents"]):
                    meta = results["metadatas"][i]
                    print(f"  - [{meta.get('entity_key', 'N/A')}] {doc} (置信度: {meta.get('confidence')}, 来源: {meta.get('source')})")
            else:
                print("  (无活跃记忆)")
        except Exception as e:
            print(f"  查询失败: {e}")

    print("\n" + "=" * 60)
    print("冲突解决验证 (test_user_001 的 income)")
    print("=" * 60)
    income_memories = store.get_memories_by_entity("test_user_001", "income")
    for mem in income_memories:
        status = mem["metadata"].get("status")
        print(f"  - 状态: {status} | {mem['content']} (置信度: {mem['metadata'].get('confidence')})")
        if status == "superseded":
            superseded_by = mem["metadata"].get("superseded_by", "N/A")
            print(f"    被覆盖，新记忆 ID: {superseded_by[:8] if superseded_by != 'N/A' else 'N/A'}...")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/test/user_profiles_test.jsonl", help="JSONL 文件路径")
    parser.add_argument("--verify", action="store_true", help="仅验证，不导入")
    args = parser.parse_args()

    if args.verify:
        verify_import()
    else:
        import_profiles(args.file)
        print("\n✅ 导入完成，开始验证...")
        verify_import()
