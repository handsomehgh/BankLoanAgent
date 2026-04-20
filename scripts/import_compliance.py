# author hgh
# version 1.0
import json
import sys
from pathlib import Path

from models.constant.constants import MetadataFields, MemoryType, ComplianceRuleFields, MemoryStatus

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.chroma_store import ChromaMemoryStore
from config import config


def import_compliance_rules(json_path: str):
    """导入合规规则到长期记忆"""
    store = ChromaMemoryStore(
        persist_dir=config.chroma_persist_dir,
        collection_name="compliance_rules"
    )

    with open(json_path, "r", encoding="utf-8") as f:
        rules = json.load(f)

    for rule in rules:
        store.add_memory(
            user_id="GLOBAL",
            content=rule.get("description", ""),
            metadata={
                MetadataFields.TYPE.value: MemoryType.COMPLIANCE_RULE.value,
                ComplianceRuleFields.RULE_ID.value: rule[ComplianceRuleFields.RULE_ID.value],
                ComplianceRuleFields.RULE_NAME.value: rule[ComplianceRuleFields.RULE_NAME.value],
                ComplianceRuleFields.RULE_TYPE.value: rule["type"],
                ComplianceRuleFields.PATTERN.value: rule.get(ComplianceRuleFields.PATTERN.value, ""),
                ComplianceRuleFields.SEVERITY.value: rule[ComplianceRuleFields.SEVERITY.value],
                ComplianceRuleFields.ACTION.value: rule[ComplianceRuleFields.ACTION.value],
                ComplianceRuleFields.TEMPLATE.value: rule.get(ComplianceRuleFields.TEMPLATE.value, ""),
                ComplianceRuleFields.SOURCE.value: rule[ComplianceRuleFields.SOURCE.value],
                MetadataFields.STATUS.value: MemoryStatus.ACTIVE.value
            }
        )
        print(f"✅ 导入规则: {rule[ComplianceRuleFields.RULE_ID.value]} - {rule[ComplianceRuleFields.RULE_NAME.value]}")

    print(f"\n🎉 共导入 {len(rules)} 条合规规则")


if __name__ == "__main__":
    import_compliance_rules("data/rules/compliance_rules.json")
