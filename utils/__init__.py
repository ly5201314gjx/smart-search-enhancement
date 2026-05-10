"""智能搜索增强工具包"""
from .intent_classifier import IntentClassifier, classify
from .result_aggregator import ResultAggregator, SimHash
from .summarizer import TextRankSummarizer, summarize
from .auto_classifier import AutoClassifier
from .quality_scorer import QualityScorer
from .suggestion_engine import SuggestionEngine
from .cross_lang_expander import CrossLangExpander
from .performance_optimizer import PerformanceOptimizer
from .feedback_learner import FeedbackLearner
from .smart_search_engine import SmartSearchEngine, smart_search

__version__ = "1.0.0"
__all__ = [
    "IntentClassifier", "classify",
    "ResultAggregator", "SimHash",
    "TextRankSummarizer", "summarize",
    "AutoClassifier",
    "QualityScorer",
    "SuggestionEngine",
    "CrossLangExpander",
    "PerformanceOptimizer",
    "FeedbackLearner",
    "SmartSearchEngine", "smart_search"
]