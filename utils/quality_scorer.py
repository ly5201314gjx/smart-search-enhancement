"""质量评分器 - 6维度质量评估"""
from typing import Dict
import re
from datetime import datetime

class QualityScorer:
    """搜索结果质量评分器"""
    
    WEIGHTS = {
        "completeness": 0.15,
        "freshness": 0.20,
        "trustworthiness": 0.25,
        "readability": 0.10,
        "coverage": 0.20,
        "richness": 0.10
    }
    
    TRUSTED_DOMAINS = {
        ".gov": 0.95, ".edu": 0.95, ".org": 0.85,
        "wikipedia.org": 0.95, "github.com": 0.85,
        "zhihu.com": 0.75, "baidu.com": 0.70
    }
    
    def score_completeness(self, result: Dict) -> float:
        """内容完整性"""
        score = 0
        if result.get("title") and len(result.get("title", "")) >= 5:
            score += 0.3
        if result.get("snippet"):
            score += 0.3
        if result.get("url"):
            score += 0.2
        if result.get("metadata"):
            score += 0.2
        return score
    
    def score_freshness(self, result: Dict) -> float:
        """信息新鲜度"""
        pub_time = result.get("publish_time")
        if not pub_time:
            return 0.5
        try:
            dt = datetime.fromisoformat(pub_time.replace("Z", "+00:00"))
            hours_old = (datetime.now(dt.tzinfo) - dt).total_seconds() / 3600
            return max(0, 1 - hours_old / 168)  # 7天衰减
        except:
            return 0.5
    
    def score_trustworthiness(self, result: Dict) -> float:
        """来源可信度"""
        url = result.get("url", "")
        for domain, score in self.TRUSTED_DOMAINS.items():
            if domain in url:
                return score
        if ".com" in url or ".cn" in url:
            return 0.6
        return 0.4
    
    def score_readability(self, result: Dict) -> float:
        """内容可读性"""
        text = result.get("snippet", "") + result.get("title", "")
        long_words = len(re.findall(r'\w{4,}', text))
        total_words = len(re.findall(r'\w+', text)) or 1
        return max(0.3, 1 - 0.1 * (long_words / total_words))
    
    def score_coverage(self, result: Dict, query: str = "") -> float:
        """关键词覆盖率"""
        if not query:
            return 0.5
        text = (result.get("title", "") + " " + result.get("snippet", "")).lower()
        query_words = set(query.lower().split())
        matched = sum(1 for w in query_words if w in text)
        return matched / len(query_words) if query_words else 0.5
    
    def score_richness(self, result: Dict) -> float:
        """内容丰富度"""
        text = result.get("snippet", "")
        length_score = min(len(text) / 500, 1.0)
        return 0.8 * length_score + 0.2 * (1.0 if result.get("url") else 0.5)
    
    def score(self, result: Dict, query: str = "") -> Dict:
        """综合评分"""
        scores = {
            "completeness": self.score_completeness(result),
            "freshness": self.score_freshness(result),
            "trustworthiness": self.score_trustworthiness(result),
            "readability": self.score_readability(result),
            "coverage": self.score_coverage(result, query),
            "richness": self.score_richness(result)
        }
        total = sum(self.WEIGHTS[k] * scores[k] for k in self.WEIGHTS) * 100
        scores["total"] = round(total, 1)
        scores["level"] = "⭐⭐⭐" if total >= 80 else "⭐⭐" if total >= 60 else "⭐"
        return scores

if __name__ == "__main__":
    scorer = QualityScorer()
    result = {"title": "Python教程", "snippet": "这是一个完整的Python编程教程", "url": "https://github.com/python"}
    print(scorer.score(result, "Python教程"))