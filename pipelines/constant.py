# author hgh
# version 1.0
from enum import Enum


class FileLodeType(str,Enum):
    QA = "qa"
    WORD_EXPL = "word_expl"
    MARK_DOWN = "mark_down"

class FileMetadata(str,Enum):
    TERM = "term"
    ENGLISH = "english"
    TOPICS = "topics"
    CONFIDENCE = "confidence"
    CHUNK_ID = "chunk_id"
    PARENT_DOC_ID = "parent_doc_id"
    CHUNK_INDEX = "chunk_index"
    SOURCE_TYPE = "source_type"
    SOURCE_FILE = "source_file"
    PRODUCT_TYPE = "product_type"
    REGULATION_NAMES = "regulation_names"


