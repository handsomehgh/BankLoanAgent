# author hgh
# version 1.0
import logging

from pymilvus import FieldSchema, DataType, Function, FunctionType, CollectionSchema, Collection, MilvusException
from pymilvus.orm import utility

from config.global_constant.constants import MemoryType, RegistryModules
from config.global_constant.fields import CommonFields
from infra.collections import CollectionNames
from infra.milvus_client import MilvusClientManager
from utils.config_utils.get_config import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DENSE_INDEX_PARAM = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200}
    }

SPARSE_INDEX_PARAM = {
        "metric_type": "BM25",
        "index_type": "SPARSE_INVERTED_INDEX",
        "params": {"drop_ratio_build": 0.2}
    }

BM25_FUNCTION = Function(
    name="bm25_fn",
    function_type=FunctionType.BM25,
    input_field_names=[CommonFields.TEXT],
    output_field_names=["sparse_vector"]
)

SCALAR_INDEX_FIELDS = ["source_type", "product_type", "status", "source_file","entity_id", "topics", "regulation_names"]

def init_collections():
    COL_FIELDS = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=128, is_primary=True),
        FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"}
        ),
        FieldSchema(name="status", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="confidence", dtype=DataType.FLOAT),
        FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="product_type", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="parent_doc_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="updated_at", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="entity_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="entity_type", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="relation_predicate", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="topics", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=20,max_length=64),
        FieldSchema(name="regulation_names", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=20,max_length=64),
        FieldSchema(name="extra", dtype=DataType.JSON),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="term_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
        # FieldSchema(name="summary_vector", dtype=DataType.FLOAT_VECTOR, dim=1024)
        # FieldSchema(name="faq_similar_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
        # FieldSchema(name="graph_embedding", dtype=DataType.FLOAT_VECTOR, dim=256),
    ]

    col = create_collection_if_not_exist(CollectionNames.for_type(MemoryType.BUSINESS_KNOWLEDGE),COL_FIELDS,"business knowledge")
    create_index(col)

def create_collection_if_not_exist(name: str, fields: list,description: str = ""):
    if utility.has_collection(collection_name=name):
        logger.info(f"collection {name} already exists")
        col = Collection(name=name)
        col.load()
        return col

    logger.info(f"Creating collection {name}")
    schema = CollectionSchema(fields=fields, functions=[BM25_FUNCTION],description=description)
    try:
        col = Collection(name=name, schema=schema)
        logger.info(f"Collection {name} created successfully")
        return col
    except MilvusException as e:
        logger.error(f"Failed to create collection: {name} : {e}")
        raise

def create_index(collection: Collection):
    dense_idx_name = "dense_vector_idx"
    if not collection.has_index(index_name=dense_idx_name):
        collection.create_index(
            index_name=dense_idx_name,
            index_params=DENSE_INDEX_PARAM,
            field_name="dense_vector"
        )
        utility.wait_for_index_building_complete(collection.name, dense_idx_name)
    else:
        logger.info(f"Dense index on dense_vector already exists")

    term_vector_idx__name = "term_vector_idx"
    if not collection.has_index(index_name=term_vector_idx__name):
        collection.create_index(
            field_name="term_vector",
            index_params=DENSE_INDEX_PARAM,
            index_name=term_vector_idx__name
        )
        utility.wait_for_index_building_complete(collection.name, term_vector_idx__name)
    else:
        logger.info(f"Sparse index on term_vector already exists")

    # faq_similar_vector_idx_name = "faq_similar_vector_idx"
    # if not collection.has_index(index_name=faq_similar_vector_idx_name):
    #     collection.create_index(
    #         field_name="faq_similar_vector",
    #         index_params=DENSE_INDEX_PARAM,
    #         index_name=faq_similar_vector_idx_name
    #     )
    #     utility.wait_for_index_building_complete(collection.name, faq_similar_vector_idx_name)
    # else:
    #     logger.info(f"Sparse index on faq_similar_vector already exists")
    #
    # summary_vector_idx_name = "summary_vector_idx"
    # if not collection.has_index(index_name=summary_vector_idx_name):
    #     collection.create_index(
    #         field_name="summary_vector",
    #         index_params=DENSE_INDEX_PARAM,
    #         index_name=summary_vector_idx_name
    #     )
    #     utility.wait_for_index_building_complete(collection.name, summary_vector_idx_name)
    # else:
    #     logger.info(f"Sparse index on summary_vector already exists")
    #
    # graph_embedding_idx_name = "graph_embedding_idx"
    # if not collection.has_index(index_name=graph_embedding_idx_name):
    #     collection.create_index(
    #         field_name="graph_embedding",
    #         index_params=DENSE_INDEX_PARAM,
    #         index_name=graph_embedding_idx_name
    #     )
    #     utility.wait_for_index_building_complete(collection.name, graph_embedding_idx_name)
    # else:
    #     logger.info(f"Sparse index on graph_embedding already exists")

    sparse_idx_name = "sparse_vector_idx"
    if not collection.has_index(index_name=sparse_idx_name):
        collection.create_index(
            field_name="sparse_vector",
            index_params=SPARSE_INDEX_PARAM,
            index_name=sparse_idx_name
        )
        utility.wait_for_index_building_complete(collection.name, sparse_idx_name)
    else:
        logger.info(f"Sparse index on sparse_vector already exists")

    for field in SCALAR_INDEX_FIELDS:
        idx_name = f"scalar_{field}_idx"
        if not collection.has_index(index_name=idx_name):
            try:
                logger.info(f"Creating scalar index {idx_name} on {collection.name}")
                collection.create_index(
                    field_name=field,
                    index_name=idx_name,
                    index_params={"index_type": "INVERTED"}
                )
                utility.wait_for_index_building_complete(collection.name, index_name=idx_name)
            except MilvusException as e:
                logger.error(f"Failed to create scalar index {idx_name} : {e}")

if __name__ == '__main__':
    registry = get_config()
    config = registry.get_config(RegistryModules.RETRIEVAL)
    client = MilvusClientManager(config.milvus_uri)
    # init_collections()
    # utility.drop_collection(CollectionNames.for_type(MemoryType.BUSINESS_KNOWLEDGE))
    # flag = client.has_collection(CollectionNames.for_type(MemoryType.BUSINESS_KNOWLEDGE))
    # col = client.get_collection(CollectionNames.for_type(MemoryType.BUSINESS_KNOWLEDGE))
    # res = col.query_utils("id != '1'",["id","text","status","confidence","source_type", "product_type", "status", "source_file","entity_id", "topics", "regulation_names"])
    # for r in res:
    #     print(r)
    # print(flag)

