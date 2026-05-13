# author hgh
# version 1.0
import json
from pathlib import Path
from datetime import datetime

from config.global_constant.constants import SpecialUserID, MemoryType
from modules.memory.memory_constant.constants import MemoryStatus
from utils.config_utils.memory_test_store import create_test_memory_store


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

def import_compliance_rules(json_path: str):
    """导入合规规则到长期记忆"""
    # vector_store = ChromaVectorStore("../chromadb")
    # store = LongTermMemoryStore(vector_store=vector_store)
    store = create_test_memory_store("http://192.168.24.128:19530")

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
    # import_compliance_rules(PROJECT_ROOT / "data" / "test" / "user_profiles_test.jsonl")
    store = create_test_memory_store("http://192.168.24.128:19530")
    # store.delete_user_memories(user_id="test_user_002",memory_type=MemoryType.USER_PROFILE)
    rules = store.get_all_user_profile_memories(user_id="test_user_001")
    for rule in rules:
        print(rule)
