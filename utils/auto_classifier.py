"""自动分类器 - 9类内容智能分类"""
import re
from typing import Dict, Tuple

class AutoClassifier:
    """搜索结果自动分类器"""
    
    CATEGORIES = ["News", "Official", "Forum", "Social", "Blog", "Video", "Academic", "Shopping", "Unclassified"]
    
    PATTERNS = {
        "News": {"keywords": ["新闻", "报道", "发布", "今日"], "domains": ["news.", "xinhua", "sina.com.cn/news"]},
        "Official": {"keywords": ["官网", "官方", "公告"], "domains": [".gov", ".edu", ".org"]},
        "Forum": {"keywords": ["论坛", "社区", "讨论"], "domains": ["bbs", "forum", "tieba"]},
        "Social": {"keywords": ["微博", "微信", "twitter"], "domains": ["weibo", "twitter.com", "zhihu"]},
        "Blog": {"keywords": ["博客", "专栏", "文章"], "domains": ["blog", "medium.com", "jianshu"]},
        "Video": {"keywords": ["视频", "movie"], "domains": ["youtube", "bilibili", "youku"]},
        "Academic": {"keywords": ["论文", "研究", "学术"], "domains": [".edu", "doi", "scholar"]},
        "Shopping": {"keywords": ["购买", "价格", "优惠"], "domains": ["taobao", "jd.com", "amazon"]}
    }
    
    def classify(self, result: Dict) -> Tuple[str, float]:
        """分类并返回置信度"""
        url = result.get("url", "")
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        content = f"{title} {snippet}"
        
        scores = {}
        for category, patterns in self.PATTERNS.items():
            score = 0
            for kw in patterns["keywords"]:
                if kw in content:
                    score += 0.4
            for domain in patterns["domains"]:
                if domain in url:
                    score += 0.6
            if score > 0:
                scores[category] = score
        
        if not scores:
            return "Unclassified", 0.3
        
        best = max(scores.items(), key=lambda x: x[1])
        confidence = min(best[1] / 1.5, 1.0)
        return best[0], confidence
    
    def batch_classify(self, results: list) -> list:
        """批量分类"""
        for r in results:
            category, confidence = self.classify(r)
            r["category"] = category
            r["category_confidence"] = confidence
        return results

if __name__ == "__main__":
    clf = AutoClassifier()
    tests = [
        {"url": "https://news.sina.com.cn", "title": "今日新闻报道", "snippet": "最新消息"},
        {"url": "https://github.com/python", "title": "Python教程", "snippet": "学习编程"},
    ]
    for r in clf.batch_classify(tests):
        print(r)