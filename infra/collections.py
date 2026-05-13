# author hgh
# version 1.0
from config.global_constant.constants import MemoryType


class CollectionNames:
    USER_PROFILE = "user_profile_memories"
    INTERACTION_LOG = "interaction_logs"
    COMPLIANCE_RULE = "compliance_rules"
    BUSINESS_KNOWLEDGE = "business_knowledge"

    @classmethod
    def for_type(cls, memory_type: MemoryType) -> str:
        mapping = {
            MemoryType.USER_PROFILE: cls.USER_PROFILE,
            MemoryType.INTERACTION_LOG: cls.INTERACTION_LOG,
            MemoryType.COMPLIANCE_RULE: cls.COMPLIANCE_RULE,
            MemoryType.BUSINESS_KNOWLEDGE: cls.BUSINESS_KNOWLEDGE,
        }
        return mapping[memory_type]