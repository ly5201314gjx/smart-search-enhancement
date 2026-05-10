"""建议引擎 - 协同过滤+趋势推荐"""
from typing import List, Dict, Tuple
from collections import defaultdict

class SuggestionEngine:
    """搜索建议生成引擎"""
    
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.history = []
        self.trending = defaultdict(float)
        self.cooccurrence = defaultdict(lambda: defaultdict(int))
    
    def add_to_history(self, query: str):
        """添加到历史"""
        if self.history and query != self.history[-1]:
            prev = self.history[-1]
            self.cooccurrence[prev][query] += 1
        self.history.append(query)
    
    def add_trending(self, query: str, velocity: float = 1.0):
        """添加趋势"""
        self.trending[query] += velocity
    
    def get_prefix_suggestions(self, prefix: str) -> List[Tuple[str, float]]:
        """前缀匹配建议"""
        suggestions = []
        for q in self.history:
            if q.startswith(prefix):
                suggestions.append((q, len(prefix) / len(q)))
        return sorted(suggestions, key=lambda x: x[1], reverse=True)[:self.top_k]
    
    def get_collab_suggestions(self, query: str) -> List[Tuple[str, float]]:
        """协同过滤建议"""
        suggestions = []
        for q, count in self.cooccurrence.get(query, {}).items():
            suggestions.append((q, count))
        return sorted(suggestions, key=lambda x: x[1], reverse=True)[:self.top_k]
    
    def get_trending_suggestions(self) -> List[Tuple[str, float]]:
        """趋势推荐"""
        sorted_trending = sorted(self.trending.items(), key=lambda x: x[1], reverse=True)
        return sorted_trending[:self.top_k]
    
    def suggest(self, query: str) -> List[Dict]:
        """综合建议"""
        suggestions = []
        alpha, beta, gamma = 0.35, 0.25, 0.25
        
        # 前缀匹配
        for s, score in self.get_prefix_suggestions(query):
            suggestions.append({"suggestion": s, "score": alpha * score, "type": "prefix"})
        
        # 协同过滤
        for s, score in self.get_collab_suggestions(query):
            suggestions.append({"suggestion": s, "score": beta * score, "type": "collab"})
        
        # 趋势推荐
        for s, score in self.get_trending_suggestions():
            suggestions.append({"suggestion": s, "score": gamma * score, "type": "trending"})
        
        # 去重并排序
        seen = {}
        for s in suggestions:
            key = s["suggestion"]
            if key not in seen or seen[key] < s["score"]:
                seen[key] = s["score"]
        
        final = [{"suggestion": k, "score": v} for k, v in seen.items()]
        return sorted(final, key=lambda x: x["score"], reverse=True)[:self.top_k]

if __name__ == "__main__":
    engine = SuggestionEngine()
    engine.add_to_history("Python教程")
    engine.add_to_history("Python进阶")
    engine.add_to_history("JavaScript入门")
    engine.add_trending("AI人工智能", 10)
    engine.add_trending("机器学习", 8)
    print(engine.suggest("Pyt"))