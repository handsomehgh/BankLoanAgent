# author hgh
# version 1.0
import json
import sys
from pathlib import Path
from datetime import datetime

from config.settings import config
from memory.memory_vector_store.chroma_vector_store import ChromaVectorStore
from config.constants import MemoryType, MemoryStatus, SpecialUserID
from memory.memory_vector_store.milvus_vector_store import MilvusVectorStore

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.long_term_memory_store import LongTermMemoryStore


def import_compliance_rules(json_path: str):
    """导入合规规则到长期记忆"""
    # vector_store = ChromaVectorStore("../chromadb")
    # store = LongTermMemoryStore(vector_store=vector_store)
    vec = MilvusVectorStore(config.milvus_uri)
    store = LongTermMemoryStore(vector_store=vec)

    with open(json_path, "r", encoding="utf-8") as f:
        rules = json.load(f)

    for rule in rules:
        # 构建符合 ComplianceRuleMetadata 的元数据字典
        metadata = {
            "confidence": 1.0,
            "status": MemoryStatus.ACTIVE.value,
            "rule_id": rule["rule_id"],
            "rule_name": rule["rule_name"],
            "rule_type": rule["rule_type"],
            "pattern": rule.get("pattern", ""),
            "action": rule["action"],
            "severity": rule["severity"],
            "description": rule["description"],
            "source": rule["source"],
            "priority": rule.get("priority", 100),
            "version": rule.get("version", datetime.now().strftime("%Y-%m-%d")),
            "effective_from": rule.get("effective_from", datetime.now().isoformat()),
            "effective_to": rule.get("effective_to"),
            "template": rule.get("template")
        }

        store.add_memory(
            user_id=SpecialUserID.GLOBAL.value,
            content=rule["description"],
            memory_type=MemoryType.COMPLIANCE_RULE,
            metadata=metadata
        )
        print(f"✅ 导入规则: {rule['rule_id']} - {rule['rule_name']}")

    print(f"\n🎉 共导入 {len(rules)} 条合规规则")


if __name__ == "__main__":
    import_compliance_rules("../data/rules/compliance_rules.json")
    vec = MilvusVectorStore(config.milvus_uri)
    store = LongTermMemoryStore(vector_store=vec)

    rules = store.get_active_compliance_rules(limit=15)
    for rule in rules:
        print(rule)
