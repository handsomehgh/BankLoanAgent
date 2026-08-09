# author hgh
# version 1.0
import logging
import socket
from typing import Dict, List
from urllib.parse import urlparse

import pytest

from config.global_constant.constants import RegistryModules
from infra.database.milvus_client import MilvusClientManager
from modules.module_services.embeddings import RobustLocalEmbeder
from modules.retrieval.knowledge_vector_store.knowledge_search_engine import KnowledgeSearchEngine
from utils.config_utils.get_config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def search_engine():
    """真实 KnowledgeSearchEngine，Milvus 不可达时跳过（避免 grpc 超时拖慢整个套件）"""
    registry = get_config()
    retrieval_config = registry.get_config(RegistryModules.RETRIEVAL)
    llm_config = registry.get_config(RegistryModules.LLM)
    parsed = urlparse(retrieval_config.milvus_uri)
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=2):
            pass
    except Exception:
        pytest.skip(f"Milvus 不可达：{retrieval_config.milvus_uri}")
    client = MilvusClientManager(retrieval_config.milvus_uri)
    # 与 container._create_embedder 接线一致
    embedder = RobustLocalEmbeder(
        model_name=llm_config.loan_official_embeder_name,
        base_url=llm_config.loan_official_embeder_url,
        dimensions=llm_config.loan_embeder_dimension
    )
    return KnowledgeSearchEngine(client, embedder, retrieval_config)

def get_search_engine() -> KnowledgeSearchEngine:
    registry = get_config()
    retrieval_config = registry.get_config(RegistryModules.RETRIEVAL)
    llm_config = registry.get_config(RegistryModules.LLM)
    client = MilvusClientManager(retrieval_config.milvus_uri)
    # 与 container._create_embedder 接线一致
    embedder = RobustLocalEmbeder(
        model_name=llm_config.loan_official_embeder_name,
        base_url=llm_config.loan_official_embeder_url,
        dimensions=llm_config.loan_embeder_dimension
    )
    return KnowledgeSearchEngine(client, embedder, retrieval_config)

def pretty_custom_print(results: List[Dict]):
    if not results:
        print(" (空)")
        return

    print_fields = ["distance","score","id","parent_doc_id","text","source_file","topics"]
    bro_text = {}
    for i,res in enumerate(results):
        print(f"============result{i + 1}================")
        for k,v in res.items():
            if k in print_fields:
                # if k == "parent_doc_id" and k not in bro_text:
                #     bro_text[k] = v
                print(f"- {k}: {v}")

    for k,v in bro_text.items():
        print(f"- {k}: {v}")


def test_dense_search_no_expr(search_engine):
    results = search_engine.dense_search(query="公积金")
    pretty_custom_print(results)

def test_dense_search_expr(search_engine):
    expr = 'ARRAY_CONTAINS(topics, "还款")'
    results = search_engine.dense_search(query="公积金",filter_expr=expr)
    pretty_custom_print(results)

def test_sparse_search_no_expr(search_engine):
    results = search_engine.sparse_search("公积金")
    pretty_custom_print(results)

def test_term_search_no_expr(search_engine):
    results = search_engine.term_search("利率")
    pretty_custom_print(results)

if __name__ == '__main__':
    # test_sparse_search_no_expr()
    # test_dense_search_no_expr()
    # test_sparse_search_no_expr()
    test_term_search_no_expr(get_search_engine())



