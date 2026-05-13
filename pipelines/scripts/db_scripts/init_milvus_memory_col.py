# author hgh
# version 1.0

"""
milvus collection initial scripts

usage:
- create separate collection for three types of long-term memories
- configure dense vector index(HNSW) and sparse vector index(SPARSE_INVERTED_INDEX)
- create a dictionary index for scalar fields
- create a full-text index for text fields(TANTIVY)

mode of operation:
    python scripts/init_milvus_collections.py
"""
import logging
import sys
from pathlib import Path

from pymilvus import FieldSchema, DataType, connections, Collection, MilvusException, CollectionSchema, utility, \
    Function, FunctionType

from config.global_constant.constants import MemoryType

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

COMMON_FIELDS = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535, enable_analyzer=True,
                analyzer_params={"type": "chinese"}),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),  # max_len -> max_length
    FieldSchema(name="status", dtype=DataType.VARCHAR, max_length=20),
    FieldSchema(name="confidence", dtype=DataType.DOUBLE),
    FieldSchema(name="extra", dtype=DataType.JSON),
    FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=30),
    FieldSchema(name="permanent", dtype=DataType.BOOL),
    FieldSchema(name="last_accessed_at", dtype=DataType.VARCHAR, max_length=30),
]

USER_PROFILE_EXTRA_FIELDS = [
    FieldSchema(name="entity_key", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="evidence_type", dtype=DataType.VARCHAR, max_length=32),
    FieldSchema(name="effective_date", dtype=DataType.VARCHAR, max_length=30),
    FieldSchema(name="expires_at", dtype=DataType.VARCHAR, max_length=30),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=32),
    FieldSchema(name="superseded_by", dtype=DataType.VARCHAR, max_length=100),  # 修复：新增字段
]

INTERACTION_LOG_EXTRA_FIELDS = [
    FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=30),
    FieldSchema(name="event_type", dtype=DataType.VARCHAR, max_length=32),
    FieldSchema(name="sentiment", dtype=DataType.VARCHAR, max_length=20),
    FieldSchema(name="key_entities", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=20,
                max_length=20),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=32),
]

COMPLIANCE_RULE_EXTRA_FIELDS = [
    FieldSchema(name="rule_id", dtype=DataType.VARCHAR, max_length=32),
    FieldSchema(name="rule_name", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="rule_type", dtype=DataType.VARCHAR, max_length=32),
    FieldSchema(name="pattern", dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name="severity", dtype=DataType.VARCHAR, max_length=20),
    FieldSchema(name="priority", dtype=DataType.INT64),
    FieldSchema(name="action", dtype=DataType.VARCHAR, max_length=20),
    FieldSchema(name="template", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="effective_to", dtype=DataType.VARCHAR, max_length=30),
    FieldSchema(name="effective_from", dtype=DataType.VARCHAR, max_length=30),
    FieldSchema(name="version", dtype=DataType.VARCHAR, max_length=20),
    FieldSchema(name="superseded_by", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=200),
]

# index configure
DENSE_INDEX_PARAMS = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 200}
}

SPARSE_INDEX_PARAMS = {
    "metric_type": "BM25",
    "index_type": "SPARSE_INVERTED_INDEX",
    "params": {"drop_ratio_build": 0.2}
}

BM25_FUNCTION = Function(
    name="bm25_fn",
    function_type=FunctionType.BM25,
    input_field_names=["text"],
    output_field_names=["sparse_vector"]
)

# the scalar fields that need to create dictionary indexes(choose based on query_utils frequency)
SCALAR_INDEX_FIELDS = {
    MemoryType.USER_PROFILE.value: ["user_id", "entity_key", "status", "confidence"],
    MemoryType.INTERACTION_LOG.value: ["user_id", "session_id", "timestamp", "status", "confidence"],
    MemoryType.COMPLIANCE_RULE.value: ["rule_id", "severity", "status", "confidence"]
}

# collection name
COLLECTION_NAMES = {
    MemoryType.USER_PROFILE.value: "user_profile_memories",
    MemoryType.INTERACTION_LOG.value: "interaction_logs",
    MemoryType.COMPLIANCE_RULE.value: "compliance_rules",
}


def connect_milvus():
    """build milvus connection"""
    try:
        connections.connect(
            alias="default",
            uri="http://192.168.24.128:19530",
            timeout=30
        )
        logger.info(f"Connected to milvus")
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {e}")
        sys.exit(1)


def create_collection_if_not_exist(
        name: str,
        fields: list,
        description: str = ""
) -> Collection:
    """create collection if not exists,and return it"""
    if utility.has_collection(name):
        logger.info(f"Collection {name} already exists")
        col = Collection(name=name)
        col.load()
        return col

    logger.info(f"Creating collection {name}")
    schema = CollectionSchema(fields=fields, functions=[BM25_FUNCTION], description=description)
    try:
        col = Collection(name=name, schema=schema)
        logger.info(f"Collection {name} created successfully")
        return col
    except MilvusException as e:
        logger.error(f"Failed to create collection: {name} : {e}")
        raise


def create_index(col: Collection, dense_field: str = "dense_vector", sparse_field: str = "sparse_vector"):
    """create vector index and scalar index for collection"""
    # dense vector index
    dense_idx_name = "dense_vector_idx"
    if not col.has_index(index_name=dense_idx_name):
        col.create_index(
            field_name=dense_field,
            index_params=DENSE_INDEX_PARAMS,
            index_name=dense_idx_name,
        )
        utility.wait_for_index_building_complete(col.name, index_name=dense_idx_name)
    else:
        logger.info(f"Dense index on {dense_field} already exists")

    # sparse vector index
    sparse_idx_name = "sparse_vector_idx"
    if not col.has_index(index_name=sparse_idx_name):
        col.create_index(
            field_name=sparse_field,
            index_params=SPARSE_INDEX_PARAMS,
            index_name=sparse_idx_name
        )
        utility.wait_for_index_building_complete(col.name, index_name=sparse_idx_name)
    else:
        logger.info(f"Sparse index on {sparse_field} already exists")

    # scalar field index(create dictionary index for high-frequency filter fields)
    col_name = col.name
    if col_name == COLLECTION_NAMES[MemoryType.USER_PROFILE.value]:
        scalar_fields = SCALAR_INDEX_FIELDS[MemoryType.USER_PROFILE.value]
    elif col_name == COLLECTION_NAMES[MemoryType.INTERACTION_LOG.value]:
        scalar_fields = SCALAR_INDEX_FIELDS[MemoryType.INTERACTION_LOG.value]
    elif col_name == COLLECTION_NAMES[MemoryType.COMPLIANCE_RULE.value]:
        scalar_fields = SCALAR_INDEX_FIELDS[MemoryType.COMPLIANCE_RULE.value]
    else:
        scalar_fields = ["user_id", "status"]

    for field in scalar_fields:
        idx_name = f"scalar_{field}_idx"
        if not col.has_index(index_name=idx_name):
            try:
                logger.info(f"Creating scalar index {idx_name} on {col_name}")
                col.create_index(
                    field_name=field,
                    index_name=idx_name
                )
                utility.wait_for_index_building_complete(col.name, index_name=idx_name)
            except MilvusException as e:
                logger.error(f"Failed to create scalar index {idx_name} : {e}")


def load_collection(col: Collection):
    try:
        col.load()
        logger.info(f"Collection '{col.name}' loaded.")
    except MilvusException as e:
        logger.error(f"Failed to load collection '{col.name}': {e}")
        raise


# ========================= main process ==============================
def init_all_collections():
    connect_milvus()

    # user profile collection
    col_profile = create_collection_if_not_exist(
        name=COLLECTION_NAMES[MemoryType.USER_PROFILE.value],
        fields=COMMON_FIELDS + USER_PROFILE_EXTRA_FIELDS,
        description="User profile memories collection"
    )
    create_index(col_profile)
    load_collection(col_profile)

    # interaciton log collection
    col_interaction = create_collection_if_not_exist(
        name=COLLECTION_NAMES[MemoryType.INTERACTION_LOG.value],
        fields=COMMON_FIELDS + INTERACTION_LOG_EXTRA_FIELDS,
        description="Interaction log memories collection",
    )
    create_index(col_interaction)
    load_collection(col_interaction)

    # compliance rule collection
    col_compliance = create_collection_if_not_exist(
        name=COLLECTION_NAMES[MemoryType.COMPLIANCE_RULE.value],
        fields=COMMON_FIELDS + COMPLIANCE_RULE_EXTRA_FIELDS,
        description="Compliance rule memories collection",
    )
    create_index(col_compliance)
    load_collection(col_compliance)
    logger.info("All collections initialized successfully.")


if __name__ == '__main__':
    # try:
    #     init_all_collections()
    # except Exception as e:
    #     logger.exception("Initialization failed")
    #     sys.exit(1)
    connections.connect(
        alias="default",
        uri="http://192.168.24.128:19530",
        timeout=30
    )
    col = Collection(name=COLLECTION_NAMES[MemoryType.USER_PROFILE.value])
    col.load()
    res = col.delete("id in ['1596f346-f309-45d8-9cc7-3612cb963bea','e108494d-c09a-4c66-beec-f119dbe8535']")
    print(res)
    # utility.drop_collection("compliance_rules")
