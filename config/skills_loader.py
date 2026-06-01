# author hgh
# version 1.0
from pathlib import Path
from typing import List
import yaml
from config.models.skill_config import SkillConfig

def load_skill_configs(skills_dir: str = "config/skills") -> List[SkillConfig]:
    configs = []
    dir_path = Path(skills_dir)
    if not dir_path.exists():
        return configs
    for yaml_file in dir_path.glob("*.yaml"):
        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            configs.append(SkillConfig(**data))
    return configs