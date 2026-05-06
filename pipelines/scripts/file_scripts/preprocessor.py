# author hgh
# version 1.0
import json
import logging
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.document_loaders import UnstructuredMarkdownLoader, TextLoader
from langchain_core.documents import Document

from config.global_constant.constants import KnowledgeFileSourceType, RegistryModules
from config.models.file_process_config import FileProcessConfig, PreprocessingConfig, MetadataExtractionConfig
from config.registry import ConfigRegistry
from pipelines.scripts.file_scripts.chunker import PROJECT_ROOT
from pipelines.constant import FileLodeType, FileMetadata

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_text(text: str, config: PreprocessingConfig) -> str:
    """text clean tool"""
    if not text:
        return ""
    text = text.replace("\r\n","\n").replace("\r","\n")
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    if config.enable_advanced_cleaning:
        import unicodedata
        text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines).strip()

def extract_metadata(text: str, source: KnowledgeFileSourceType, config: MetadataExtractionConfig) -> Dict[str, Any]:
    """metadata extraction tool"""
    metadata = {FileMetadata.SOURCE_TYPE: source.value}

    #product type
    for product, keywords in config.product_keywords.items():
        if any(kw in text for kw in keywords):
            metadata[FileMetadata.PRODUCT_TYPE] = product
            break
    if  FileMetadata.PRODUCT_TYPE not in metadata:
        metadata[FileMetadata.PRODUCT_TYPE] = "通用"

    #topic
    topics = [topic for topic,words in config.topic_keywords.items() if any(w in text for w in words)]
    metadata[FileMetadata.TOPICS] = topics if topics else ["其他"]

    #regulation names
    regs = re.findall(r'《(.*?)》', text)
    if regs:
        metadata[FileMetadata.REGULATION_NAMES] = list(set(regs))
    return metadata

def load_markdown(file_path: Path, source: KnowledgeFileSourceType, config: PreprocessingConfig,meta_conf: MetadataExtractionConfig):
    """general markdown loader"""
    logger.info(f"Loading {source}: {file_path}")
    try:
        loader =UnstructuredMarkdownLoader(str(file_path),mode="elements")
        raw_docs =loader.load()
    except Exception as e1:
        logger.error(f"An error occurred while loading {source},Try downgrading to plain text: {e1}")
        try:
            loader = TextLoader(str(file_path),encoding="utf-8")
            raw_docs = loader.load()
        except Exception as e2:
            logger.error(f"Failed to load text: {e2}")
            return []

    docs = []
    for doc in raw_docs:
        cleaned = clean_text(doc.page_content,config)
        if len(cleaned) < config.min_content_length:
            continue
        meta = extract_metadata(cleaned,source,meta_conf)
        meta.update(doc.metadata)
        docs.append(Document(page_content=cleaned,metadata=meta))
    logger.info(f"from {source} loaded {len(docs)} chunk")
    return docs


def load_glossary(file_path: Path, config: PreprocessingConfig, meta_config: MetadataExtractionConfig) -> List[Document]:
    """
    load the glossary(table format),keeping the three columns：term,english,definition
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to load glossary {file_path}: {e}")
        return []

    lines = content.strip().split("\n")
    strat = 0
    for i, line in enumerate(lines):
        if "|" in line and "---" in line:
            start = i + 1
            break

    docs = []
    for line in lines[start:]:
        if not line.strip() or not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) < 2:
            continue

        term = parts[0]
        if len(parts) == 2:
            definition = parts[1]
            text = f"{term}：{definition}"
            english = ""
        elif len(parts) >= 3:
            english = parts[1]
            definition = parts[2]
            # 有英文时，显示为“术语 (英文)：定义”
            if english:
                text = f"{term} ({english})：{definition}"
            else:
                text = f"{term}：{definition}"
        else:
            text = " ".join(parts)

        cleaned = clean_text(text,config)
        if len(cleaned) < config.min_content_length:
            continue
        meta = extract_metadata(cleaned,KnowledgeFileSourceType.GLOSSARY,meta_config)
        meta[FileMetadata.TERM] = term
        if english:
            meta[FileMetadata.ENGLISH] = english
        docs.append(Document(page_content=cleaned,metadata=meta))

    logger.info(f"Loaded {len(docs)} term from glossary")
    return docs

def load_faq(file_path: Path, config: PreprocessingConfig, meta_config: MetadataExtractionConfig) -> List[Document]:
    """load FAQ(Q:A)"""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to load FAQ {file_path}: {e}")
        return []

    qa_blocks = re.split(r'\n(?=(?:\*\*)?Q\d+[:\.\s])', content)
    docs = []
    for block in qa_blocks:
        if not block.strip():
            continue
        q_match = re.search(r'Q\d+[:\.\s]\s*(.*?)\n', block)
        a_match = re.search(r'A:\s*(.*)', block, re.DOTALL)
        question = q_match.group(1).strip() if q_match else ""
        answer = a_match.group(1).strip() if a_match else ""
        full_text = f"Q: {question}\nA: {answer}" if question and answer else block.strip()
        cleaned = clean_text(full_text, config)
        if len(cleaned) < config.min_content_length:
            continue
        meta = extract_metadata(cleaned, KnowledgeFileSourceType.FAQ, meta_config)
        meta["question"] = question
        meta["answer"] = answer
        docs.append(Document(page_content=cleaned, metadata=meta))
    logger.info(f"从 FAQ 加载了 {len(docs)} 个问答对")
    return docs

class DocumentPreProcessor:
    def __init__(self, file_process_config: FileProcessConfig):
        self.config = file_process_config
        self.loader_registry = {
            FileLodeType.QA: load_faq,
            FileLodeType.WORD_EXPL: load_glossary,
            FileLodeType.MARK_DOWN: load_markdown,
        }

    def process_entry(self) -> List[Document]:
        all_docs = []
        data_dir = self.config.data_dir
        prep_conf = self.config.preprocessing
        meta_conf = self.config.metadata_extraction

        for filename, mapping in self.config.file_source_mapping.items():
            file_path = PROJECT_ROOT / data_dir / filename
            if not file_path.exists():
                logger.warning(f"File doesn't exists，skip: {file_path}")
                continue

            loader_type = mapping.loader_type
            source_type = mapping.source_type

            try:
                if loader_type in (FileLodeType.QA, FileLodeType.WORD_EXPL):
                    loader_func = self.loader_registry[loader_type]
                    docs = loader_func(file_path, prep_conf, meta_conf)
                elif loader_type == FileLodeType.MARK_DOWN:
                    docs = load_markdown(file_path,source_type, prep_conf, meta_conf)
                else:
                    logger.error(f"Unknown file type: {source_type}")
                    continue

                # add to docs
                all_docs.extend(docs)
                logger.info(f"Successfully process  {filename}，obtain {len(docs)} documents")
            except Exception as e:
                logger.error(f"An error occurred while processing {filename} : {e}", exc_info=True)

        # assign id to each chunk
        for doc in all_docs:
            doc.metadata[FileMetadata.CHUNK_ID] = str(uuid.uuid4())
            doc.metadata[FileMetadata.SOURCE_FILE] = filename if 'filename' in doc.metadata else ""

        logger.info(f"Full processing completed，total of {len(all_docs)} documents")
        return all_docs

if __name__ == '__main__':
    registry = ConfigRegistry()
    registry.register_model(
        RegistryModules.FILE_PROCESS, FileProcessConfig,
        Path(PROJECT_ROOT / "config" / "rules" / "file_process_config.yaml")
    )
    registry.load_all()

    file_config = registry.get_config(RegistryModules.FILE_PROCESS)

    preprocessor = DocumentPreProcessor(file_config)
    documents = preprocessor.process_entry()
    output_file = PROJECT_ROOT / file_config.stage1_output_dir / "preprocessed_docs.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(json.dumps({
                "content": doc.page_content,
                "metadata": doc.metadata
            }, ensure_ascii=False) + '\n')
    logger.info(f"预处理结果已保存至 {output_file}")

