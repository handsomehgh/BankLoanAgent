# author hgh
# version 1.0
import json
import logging
import re
import uuid
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.document_loaders import UnstructuredMarkdownLoader, TextLoader
from langchain_core.documents import Document

from config.global_constant.constants import KnowledgeFileSourceType, RegistryModules
from config.models.file_process_config import FileProcessConfig, PreprocessingConfig, MetadataExtractionConfig
from config.registry import ConfigRegistry
from pipelines.scripts.file_scripts.chunker import PROJECT_ROOT
from pipelines.constant import FileLodeType, FileMetadata

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HTMLTableParser(HTMLParser):
    """将 HTML <table> 转为结构化自然语言"""
    def __init__(self):
        super().__init__()
        self.reset()
        self.rows = []
        self.current_row = []
        self.current_cell = ""
        self.in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag in ("td", "th"):
            self.in_cell = True
            self.current_cell = ""
        elif tag == "tr":
            self.current_row = []

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self.in_cell = False
            self.current_row.append(self.current_cell.strip())
        elif tag == "tr":
            if self.current_row:
                self.rows.append(self.current_row)

    def handle_data(self, data):
        if self.in_cell:
            self.current_cell += data

    def parse(self, html: str) -> List[List[str]]:
        self.reset()
        self.rows = []
        self.feed(html)
        return self.rows


def html_table_to_text(html: str) -> str:
    """将 HTML 表格转换为流畅的自然语言描述"""
    parser = HTMLTableParser()
    rows = parser.parse(html)
    if not rows:
        return ""

    # 假定第一行为表头
    headers = rows[0] if rows else []
    descriptions = []
    for row in rows[1:]:
        if not row:
            continue
        # 组合每一行的键值对
        line_parts = []
        for i, cell in enumerate(row):
            if i < len(headers) and headers[i]:
                line_parts.append(f"{headers[i]}：{cell}")
            else:
                line_parts.append(cell)
        descriptions.append("，".join(line_parts))
    return "。\n".join(descriptions)

def clean_text(text: str, config: PreprocessingConfig) -> str:
    """text clean tool"""
    if not text:
        return ""
    text = text.replace("\r\n","\n").replace("\r","\n")
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    if config.enable_advanced_cleaning:
        import unicodedata
        text = unicodedata.normalize('NFKC', text)

        # 移除 Markdown 格式标记（保留文字内容）
        # **粗体** / *斜体* / ***粗斜体***
        text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
        # __粗体__ / _斜体_
        text = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)
        # ~~删除线~~
        text = re.sub(r'~~([^~]+)~~', r'\1', text)
        # `行内代码`
        text = re.sub(r'`([^`]+)`', r'\1', text)
        # [链接文本](url)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        # 图片 ![](url)
        text = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', text)
        # 裸 URL
        text = re.sub(r'https?://\S+', '', text)

        # 合并多个空格为单个
        text = re.sub(r'[^\S\n]+', ' ', text)
        # 合并 3 个以上换行为 2 个
        text = re.sub(r'\n{3,}', '\n\n', text)

    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines).strip()

def extract_metadata(text: str, source: KnowledgeFileSourceType, config: MetadataExtractionConfig) -> Dict[str, Any]:
    """metadata extraction tool"""
    metadata = {FileMetadata.SOURCE_TYPE: source.value}

    #product type
    matched_product = None
    matched_product_score = 0
    for product, keywords in config.product_keywords.items():
        for kw in keywords:
            if kw in text:
                # 长关键词得分更高（"公积金组合贷款" > "公积金"）
                score = len(kw)
                if score > matched_product_score:
                    matched_product = product
                    matched_product_score = score
    metadata[FileMetadata.PRODUCT_TYPE] = matched_product or "通用"

    #topic
    topic_scores = {}
    for topic, words in config.topic_keywords.items():
        for w in words:
            if w in text:
                topic_scores[topic] = topic_scores.get(topic, 0) + len(w)
    sorted_topics = sorted(topic_scores.keys(), key=lambda t: topic_scores[t], reverse=True)
    metadata[FileMetadata.TOPICS] = sorted_topics[:3] if sorted_topics else ["其他"]

    #regulation names
    regs = re.findall(r'《(.*?)》', text)
    if regs:
        metadata[FileMetadata.REGULATION_NAMES] = list(set(regs))
    return metadata


def load_markdown(file_path: Path, source: KnowledgeFileSourceType,
                 config: PreprocessingConfig, meta_conf: MetadataExtractionConfig) -> List[Document]:
    """生产级Markdown加载器：标题继承 + 表格转自然语言"""
    logger.info(f"Loading {source}: {file_path}")
    try:
        loader = UnstructuredMarkdownLoader(str(file_path), mode="elements")
        raw_docs = loader.load()
    except Exception as e1:
        logger.error(f"Failed to load {source}, trying plain text: {e1}")
        try:
            loader = TextLoader(str(file_path), encoding="utf-8")
            raw_docs = loader.load()
        except Exception as e2:
            logger.error(f"Failed to load text: {e2}")
            return []

    docs = []
    # 标题上下文栈
    h1_title = ""
    h2_title = ""

    for doc in raw_docs:
        text = doc.page_content.strip()
        if not text:
            continue

        category = doc.metadata.get("category", "")

        # ---------- 标题处理 ----------
        if category == "Title":
            depth = doc.metadata.get("category_depth", 2)
            if depth == 1:
                h1_title = text
                h2_title = ""
            else:
                h2_title = text
            # 纯标题不单独作为 chunk
            continue

        # ---------- 表格处理 ----------
        if category == "Table":
            html_content = doc.metadata.get("text_as_html", "")
            if html_content:
                try:
                    text = html_table_to_text(html_content)
                except Exception as e:
                    logger.warning(f"表格转自然语言失败，降级为原始文本: {e}")
                    # 降级：使用原始 text，但去重空白
                    text = re.sub(r'\s+', ' ', text)
            else:
                # 无 HTML 备份，简单合并空白
                text = re.sub(r'\s+', ' ', text)

        # ---------- 注入标题前缀 ----------
        prefix_parts = []
        if h1_title:
            prefix_parts.append(h1_title)
        if h2_title:
            prefix_parts.append(h2_title)

        if prefix_parts:
            prefix = " > ".join(prefix_parts)
            text = f"{prefix} | {text}"

        # ---------- 清洗与过滤 ----------
        cleaned = clean_text(text, config)
        if len(cleaned) < config.min_content_length:
            continue

        meta = extract_metadata(cleaned, source, meta_conf)
        meta.update(doc.metadata)
        docs.append(Document(page_content=cleaned, metadata=meta))

    logger.info(f"from {source} loaded {len(docs)} chunks (with title inheritance and table conversion)")
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
        elif len(parts) >= 4:
            english = parts[1]
            definition = parts[2]
            scenario = parts[3]
            if english and english != "-" and english != "—":
                if scenario and scenario != "-" and scenario != "—":
                    text = f"{term} ({english})：{definition}。使用场景：{scenario}"
                else:
                    text = f"{term} ({english})：{definition}"
            else:
                if scenario and scenario != "-" and scenario != "—":
                    text = f"{term}：{definition}。使用场景：{scenario}"
                else:
                    text = f"{term}：{definition}"
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

    qa_blocks = re.split(r'\n(?=(?:\*\*)?Q\d+[:\.\s：])', content)
    docs = []

    # 用于清理块尾部残留的正则（章节标题、分隔线、元数据声明等）
    TAIL_CLEAN_PATTERN = re.compile(
        r'(\n---\n.*$|'  # Markdown 分隔线及之后所有内容
        r'\n##\s+.*$|'  # 二级标题及之后内容
        r'\n#\s+.*$|'  # 一级标题及之后内容
        r'\n>\s*\*\*说明\*\*.*$|'  # 文档元信息声明
        r'\n>\s*\*\*维护部门\*\*.*$|'
        r'\n>\s*\*\*更新日期\*\*.*$)',
        re.DOTALL
    )

    # 用于清理 Markdown 格式标记
    MARKDOWN_CLEAN_PATTERN = re.compile(r'\*{1,3}([^*]+)\*{1,3}')  # **粗体** / *斜体*


    for block in qa_blocks:
        if not block.strip():
            continue

        q_match = re.search(r'\*{0,2}Q\d+[:\.\s：]\*{0,2}\s*(.+?)(?=\n\s*A[:：])', block, re.DOTALL)
        a_match = re.search(r'\n\s*A[:：]\s*(.+)', block, re.DOTALL)

        question = q_match.group(1).strip() if q_match else ""
        answer = a_match.group(1).strip() if a_match else ""

        if not question or not answer:
            # 降级：整个block作为文本
            text = block.strip()
        else:
            # 清洗 answer 尾部的文档残留
            answer = TAIL_CLEAN_PATTERN.sub('', answer).strip()
            # 去除 Markdown 粗体/斜体标记，保留文字
            answer = MARKDOWN_CLEAN_PATTERN.sub(r'\1', answer)
            question = MARKDOWN_CLEAN_PATTERN.sub(r'\1', question)
            text = f"Q: {question}\nA: {answer}"

        cleaned = clean_text(text, config)
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
                    for doc in docs:
                        doc.metadata[FileMetadata.SOURCE_FILE] = filename
                elif loader_type == FileLodeType.MARK_DOWN:
                    docs = load_markdown(file_path,source_type, prep_conf, meta_conf)
                    for doc in docs:
                        doc.metadata[FileMetadata.SOURCE_FILE] = filename
                else:
                    logger.error(f"Unknown file type: {source_type}")
                    continue

                # add to docs
                all_docs.extend(docs)
                logger.info(f"Successfully process  {filename}，obtain {len(docs)} documents")
            except Exception as e:
                logger.error(f"An error occurred while processing {filename} : {e}", exc_info=True)

        for doc in all_docs:
            doc.metadata[FileMetadata.CHUNK_ID] = str(uuid.uuid4())

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

