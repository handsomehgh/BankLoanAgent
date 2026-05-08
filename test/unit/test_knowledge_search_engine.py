# author hgh
# version 1.0
import logging
from typing import Dict, List

from config.global_constant.constants import RegistryModules
from infra.milvus_client import MilvusClientManager
from modules.module_services.embeddings import get_embeddings
from modules.retrieval.knowledge_vector_store.knowledge_search_engine import KnowledgeSearchEngine
from utils.config_utils.get_config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_search_engine() -> KnowledgeSearchEngine:
    registry = get_config()
    retrieval_config = registry.get_config(RegistryModules.RETRIEVAL)
    client = MilvusClientManager(retrieval_config.milvus_uri)
    embedder = get_embeddings()
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


def test_dense_search_no_expr():
    searchengine = get_search_engine()
    results = searchengine.dense_search(query="公积金")
    pretty_custom_print(results)

def test_dense_search_expr():
    searchengine = get_search_engine()
    expr = 'ARRAY_CONTAINS(topics, "还款")'
    results = searchengine.dense_search(query="公积金",filter_expr=expr)
    pretty_custom_print(results)

def test_sparse_search_no_expr():
    searchengine = get_search_engine()
    results = searchengine.sparse_search("公积金")
    pretty_custom_print(results)

def test_term_search_no_expr():
    searchengine = get_search_engine()
    results = searchengine.term_search("利率")
    pretty_custom_print(results)

if __name__ == '__main__':
    # test_sparse_search_no_expr()
    # test_dense_search_no_expr()
    # test_sparse_search_no_expr()
    test_term_search_no_expr()



