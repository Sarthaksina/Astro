"""
LLM Knowledge Integration System for the Cosmic Market Oracle.

This package integrates large language models with astrological knowledge
for market analysis and prediction.
"""

from src.llm_integration.knowledge_base import AstrologicalKnowledgeBase
from src.llm_integration.prompt_engineering import PromptTemplate, PromptLibrary
from src.llm_integration.retrieval_augmentation import RAGSystem
from src.llm_integration.reasoning_chains import ReasoningChain
from src.llm_integration.output_formats import OutputFormatter
from src.llm_integration.explanation_module import ExplanationGenerator
from src.llm_integration.conversation import ConversationalInterface
from src.llm_integration.report_generation import ReportGenerator
from src.llm_integration.knowledge_distillation import KnowledgeDistiller
from src.llm_integration.continuous_learning import ContinuousLearningPipeline

__all__ = [
    'AstrologicalKnowledgeBase',
    'PromptTemplate',
    'PromptLibrary',
    'RAGSystem',
    'ReasoningChain',
    'OutputFormatter',
    'ExplanationGenerator',
    'ConversationalInterface',
    'ReportGenerator',
    'KnowledgeDistiller',
    'ContinuousLearningPipeline',
]
