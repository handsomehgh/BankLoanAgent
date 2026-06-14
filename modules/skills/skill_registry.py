# author hgh
# version 1.0
from typing import Dict, Optional

from config.models.skill_config import SkillConfig


class SkillRegistry:
    def __init__(self):
        self._skills: Dict[str, SkillConfig] = {}

    def register(self, skill_config: SkillConfig):
        self._skills[skill_config.name] = skill_config

    def get(self, name: str) -> Optional[SkillConfig]:
        return self._skills.get(name)

    def get_all(self) -> Dict[str, SkillConfig]:
        return self._skills.copy()