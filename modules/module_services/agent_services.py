# author hgh
# version 1.0
from dataclasses import dataclass

from config.registry import ConfigRegistry
from modules.memory.base import BaseRetriever
from modules.memory.memory_business_store.base_memory_store import BaseMemoryStore
from modules.memory.memory_utils.profile_gate_util import ProfileGate
from modules.module_services.SummaryGenerator import SummaryGenerator
from modules.module_services.chat_models import RobustLLM
from modules.module_services.evidence_infer import EvidenceTypeInfer
from modules.module_services.profile_extractor import ProfileExtractor
from modules.module_services.sentiment_analyser import SentimentAnalyzer
from modules.retrieval.retrieval_service import RetrievalService
from utils.serialize_utils.seq_generator import SequenceGenerator


@dataclass
class AgentServices:
    """All services and configurations required by the Agent are uniformly passed through dependency injection"""

    # LLM client
    creative_llm: RobustLLM
    precise_llm: RobustLLM

    # memory
    memory_store: BaseMemoryStore
    memory_retriever: BaseRetriever

    # knowledge
    knowledge_retriever: RetrievalService

    # reusable domain service
    summary_generator: SummaryGenerator
    sentiment_analyzer: SentimentAnalyzer
    evidence_infer: EvidenceTypeInfer
    profile_extractor: ProfileExtractor
    profile_gate: ProfileGate

    # configure the registration center
    registry: ConfigRegistry

    #sequece generator
    seq_generator: SequenceGenerator
