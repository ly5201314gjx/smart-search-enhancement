"""主搜索引擎 - 九大算法编排"""
from typing import Dict, List, Optional
from .intent_classifier import IntentClassifier
from .result_aggregator import ResultAggregator
from .summarizer import TextRankSummarizer
from .auto_classifier import AutoClassifier
from .quality_scorer import QualityScorer
from .suggestion_engine import SuggestionEngine
from .cross_lang_expander import CrossLangExpander

class SmartSearchEngine:
    """AI智能搜索引擎"""
    
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.result_aggregator = ResultAggregator()
        self.summarizer = TextRankSummarizer()
        self.auto_classifier = AutoClassifier()
        self.quality_scorer = QualityScorer()
        self.suggestion_engine = SuggestionEngine()
        self.cross_lang_expander = CrossLangExpander()
    
    def search(self, query: str, results: List[Dict], options: Optional[Dict] = None) -> Dict:
        """智能搜索"""
        opts = options or {}
        response = {"query": query, "results": results}
        
        # 1. 意图识别
        intent_result = self.intent_classifier.classify(query)
        response["intent"] = intent_result["intent"]
        response["intent_confidence"] = intent_result["confidence"]
        
        # 2. 跨语言扩展
        if opts.get("cross_lang", True):
            lang_expansions = self.cross_lang_expander.expand(query)
            response["language_expansions"] = lang_expansions
        
        # 3. 结果聚合去重
        response["results"] = self.result_aggregator.aggregate(results)
        
        # 4. 自动分类
        response["results"] = self.auto_classifier.batch_classify(response["results"])
        
        # 5. 质量评分
        for r in response["results"]:
            r["quality"] = self.quality_scorer.score(r, query)
        
        # 6. 摘要生成
        if opts.get("summarize", True) and response["results"]:
            top_result = response["results"][0]
            content = top_result.get("snippet", "") + " " + top_result.get("title", "")
            response["summary"] = self.summarizer.summarize(content)
        
        # 7. 搜索建议
        self.suggestion_engine.add_to_history(query)
        response["suggestions"] = self.suggestion_engine.suggest(query)
        
        return response

def smart_search(query: str, results: List[Dict], **kwargs) -> Dict:
    """便捷函数"""
    engine = SmartSearchEngine()
    return engine.search(query, results, kwargs)

if __name__ == "__main__":
    engine = SmartSearchEngine()
    test_results = [
        {"title": "Python教程", "snippet": "学习Python编程", "url": "https://example.com/python"},
        {"title": "Python进阶", "snippet": "Python高级教程", "url": "https://example.com/python-advanced"}
    ]
    result = engine.search("Python教程", test_results)
    print(f"意图: {result['intent']}, 结果数: {len(result['results'])}")