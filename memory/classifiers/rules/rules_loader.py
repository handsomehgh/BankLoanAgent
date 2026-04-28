# author hgh
# version 1.0
"""
YAML configuration Hot Loader

Supports automatic reload based on file modification time,avoiding frequent I/O
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import yaml

logger = logging.getLogger(__name__)


class YAMLConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self._cache: Optional[Dict[str, Any]] = None
        self._last_time: Optional[float] = None

    def load_from_file(self) -> Dict[str, Any]:
        """load YAML configuration from file"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load YAML configuration from {self.config_path}: {e}")
            return {}

    def _should_reload(self) -> bool:
        """check if the has been updated and needs to be reloaded"""
        if not self.config_path.exists():
            logger.warning(f"File {self.config_path} does not exist")
            return False
        current_time = self.config_path.stat().st_mtime
        if self._last_time is None or self._last_time < current_time:
            return True
        return False

    def get_config(self) -> Dict[str, Any]:
        """get configuration from file"""
        if self._should_reload():
            new_config = self.load_from_file()
            if new_config:
                self._cache = new_config
                self._last_time = self.config_path.stat().st_mtime
                logger.info(f"Reloaded config from {self.config_path}")
        if self._cache is None:
            self._cache = self.load_from_file()
            if self.config_path.exists():
                self._last_time = self.config_path.stat().st_mtime
        return self._cache or {}

    def get_strong_keywords(self) -> Dict[str, Any]:
        config = self.get_config()
        return config.get("strong_keywords", {})

    def get_evidence_weights(self) -> Dict[str, Any]:
        config = self.get_config()
        return config.get("evidence_weights", {})

    def get_compliance_severity(self) -> Dict[str, Any]:
        config =self.get_config()
        return config.get("compliance_severity",{})


_evidence_loader: Optional[YAMLConfigLoader] = None
_sentiment_loader: Optional[YAMLConfigLoader] = None
_compliance_loader: Optional[YAMLConfigLoader] = None


def get_evidence_loader() -> YAMLConfigLoader:
    global _evidence_loader
    if _evidence_loader is None:
        config_path = Path(__file__).parent / "evidence_rules.yaml"
        _evidence_loader = YAMLConfigLoader(str(config_path))
    return _evidence_loader


def get_sentiment_loader() -> YAMLConfigLoader:
    global _sentiment_loader
    if _sentiment_loader is None:
        config_path = Path(__file__).parent / "sentiment_rules.yaml"
        _sentiment_loader = YAMLConfigLoader(str(config_path))
    return _sentiment_loader

def get_compliance_loader() -> YAMLConfigLoader:
    global _compliance_loader
    if _compliance_loader is None:
        config_path = Path(__file__).parent / "compliance_rules.yaml"
        _compliance_loader = YAMLConfigLoader(str(config_path))
    return _compliance_loader