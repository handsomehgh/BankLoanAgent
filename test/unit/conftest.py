# 单测共享 fixture。
# 依赖外部服务的 fixture 在服务不可用时自动 skip，保证离线/CI 环境下 test/unit 可完整收集。
import pytest
import requests

from config.global_constant.constants import RegistryModules
from modules.retrieval.rereanker import Reranker
from utils.config_utils.get_config import get_config


@pytest.fixture(scope="session")
def registry():
    return get_config()


@pytest.fixture(scope="session")
def reranker(registry):
    """真实 Reranker（remote 模式）。远端服务不可达时跳过，避免单测红一片"""
    cfg = registry.get_config(RegistryModules.RETRIEVAL).reranker
    if cfg.remote_url:
        try:
            requests.get(cfg.remote_url, timeout=2)
        except Exception:
            pytest.skip(f"reranker 远端服务不可用：{cfg.remote_url}")
    return Reranker(cfg)
