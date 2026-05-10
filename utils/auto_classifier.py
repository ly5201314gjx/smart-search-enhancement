"""
📐 算法四：搜索结果自动分类器 (Auto Classifier)
=================================================

基于特征提取 + 预定义类别原型向量 + Softmax分类的层次聚类分类算法。

类别:
C₁ = News (新闻)
C₂ = Official (官方/权威)
C₃ = Forum (论坛/社区)
C₄ = Social (社交媒体)
C₅ = Blog (博客/专栏)
C₆ = Video (视频)
C₇ = Academic (学术)
C₈ = Shopping (购物)
C₉ = Unclassified (未分类)
"""

import re
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import Counter


# 类别定义（关键词 + URL模式 + 域名特征）
CATEGORY_DEFINITIONS = {
    "News": {
        "keywords": {
            "zh": ["报道", "新闻", "发布", "宣布", "消息", "据悉", "快讯", 
                   "资讯", "时报", "日报", "晚报", "周刊", "记者"],
            "en": ["news", "report", "announce", "breaking", "update", 
                   "coverage", "headline", "press"]
        },
        "url_patterns": [r'/news/', r'/article/', r'/story/', r'/202[0-9]/'],
        "domain_indicators": [".news", "news.", "times", "post", "herald"]
    },
    
    "Official": {
        "keywords": {
            "zh": ["官网", "官方", "公告", "通知", "声明", "政策", "法规",
                   "政府", "部门", "委员会"],
            "en": ["official", "government", "announcement", "policy", 
                   "regulation", "authority", "ministry", "department"]
        },
        "url_patterns": [r'/about', r'/company', r'/official', r'/press/',
                        r'/notice/', r'/policy/'],
        "domain_indicators": [".gov", ".edu", ".org", ".int", ".mil"]
    },
    
    "Forum": {
        "keywords": {
            "zh": ["论坛", "bbs", "社区", "讨论", "问答", "互助", "求助",
                   "帖子", "楼主", "回复"],
            "en": ["forum", "discuss", "thread", "topic", "question", 
                   "answer", "comment", "reply", "post"]
        },
        "url_patterns": [r'/forum', r'/discuss', r'/topic/', r'/thread/',
                        r'/question/', r'/answer/'],
        "domain_indicators": ["bbs", "forum", "stackexchange", "reddit"]
    },
    
    "Social": {
        "keywords": {
            "zh": ["微博", "知乎", "微信", "公众号", "小红书", "抖音",
                   "推特", "分享", "点赞"],
            "en": ["twitter", "reddit", "facebook", "instagram", "tiktok",
                   "share", "follow", "trending", "viral"]
        },
        "url_patterns": [r'/status/', r'/post/', r'/tweet/', r'/share/'],
        "domain_indicators": ["twitter.com", "reddit.com", "facebook.com",
                            "zhihu.com", "weibo.com", "xiaohongshu.com"]
    },
    
    "Blog": {
        "keywords": {
            "zh": ["博客", "专栏", "文章", "原创", "随笔", "日记", "心得",
                   "笔记", "经验", "分享"],
            "en": ["blog", "article", "opinion", "thoughts", "personal",
                   "journal", "diary", "review", "insights"]
        },
        "url_patterns": [r'/blog/', r'/post/', r'/article/', r'/opinion/'],
        "domain_indicators": [".blog", "medium.com", "wordpress", "blogger"]
    },
    
    "Video": {
        "keywords": {
            "zh": ["视频", "录像", "直播", "播放", "观看", "频道", 
                   "youtube", "bilibili", "优酷", "爱奇艺"],
            "en": ["video", "watch", "stream", "live", "channel",
                   "youtube", "bilibili", "tutorial", "episode"]
        },
        "url_patterns": [r'/watch', r'/video/', r'/live/', r'/channel/',
                        r'/v/', r'/play/'],
        "domain_indicators": ["youtube.com", "bilibili.com", "youku.com",
                            "iqiyi.com", "twitch.tv", "vimeo.com"]
    },
    
    "Academic": {
        "keywords": {
            "zh": ["论文", "研究", "学术", "期刊", "引用", "DOI", "文献",
                   "实验室", "博士", "硕士", "科学", "算法", "实验"],
            "en": ["paper", "research", "academic", "journal", "citation",
                   "doi", "study", "experiment", "algorithm", "thesis",
                   "dissertation", "conference", "proceedings"]
        },
        "url_patterns": [r'/doi/', r'/paper/', r'/publication/', r'/article/',
                        r'/abs/', r'/pdf/'],
        "domain_indicators": [".edu", "scholar.", "arxiv.org", "ieee.org",
                            "acm.org", "springer.com", "elsevier.com",
                            "nature.com", "science.org"]
    },
    
    "Shopping": {
        "keywords": {
            "zh": ["价格", "购买", "购物", "优惠", "折扣", "促销", "代购",
                   "淘宝", "京东", "拼多多", "亚马逊", "商品", "店铺"],
            "en": ["buy", "price", "shop", "purchase", "discount", "deal",
                   "amazon", "ebay", "aliexpress", "product", "review",
                   "best price", "cheap", "order"]
        },
        "url_patterns": [r'/product/', r'/item/', r'/dp/', r'/shop/', 
                        r'/buy/', r'/cart/', r'/checkout/'],
        "domain_indicators": ["amazon.", "taobao.com", "jd.com", "ebay.",
                            "aliexpress.", "etsy.com", "walmart.com"]
    }
}


class AutoClassifier:
    """
    搜索结果自动分类器
    
    使用关键词密度 + URL模式匹配 + 域名特征的混合分类方法
    """
    
    def __init__(self, threshold: float = 0.6, temperature: float = 0.5):
        self.threshold = threshold
        self.temperature = temperature
        self.categories = list(CATEGORY_DEFINITIONS.keys()) + ["Unclassified"]
    
    def classify(self, item: Dict) -> str:
        """
        分类单个结果
        
        Returns: 类别标签
        """
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        url = item.get("url", "")
        domain = item.get("domain", "")
        content = item.get("content", f"{title} {snippet}")
        
        scores = {}
        for cat_name, cat_def in CATEGORY_DEFINITIONS.items():
            score = self._compute_score(content, url, domain, cat_def)
            scores[cat_name] = score
        
        # Softmax归一化
        exp_scores = {}
        for cat, score in scores.items():
            exp_scores[cat] = math.exp(score / self.temperature)
        
        total_exp = sum(exp_scores.values())
        probs = {cat: s / total_exp for cat, s in exp_scores.items()}
        
        # 取最高分
        best_cat = max(probs, key=probs.get)
        best_prob = probs[best_cat]
        
        if best_prob >= self.threshold:
            return best_cat
        else:
            return "Unclassified"
    
    def _compute_score(self, content: str, url: str, domain: str,
                       cat_def: Dict) -> float:
        """
        计算类别匹配得分
        
        三部分加权: 关键词(0.4) + URL模式(0.35) + 域名(0.25)
        """
        content_lower = content.lower()
        url_lower = url.lower()
        domain_lower = domain.lower()
        
        # 关键词密度得分 (0.4)
        kw_score = self._keyword_score(content_lower, cat_def["keywords"])
        
        # URL模式得分 (0.35)
        url_score = self._url_pattern_score(url_lower, cat_def["url_patterns"])
        
        # 域名指示得分 (0.25)
        domain_score = self._domain_score(domain_lower or url_lower, 
                                         cat_def["domain_indicators"])
        
        return 0.4 * kw_score + 0.35 * url_score + 0.25 * domain_score
    
    def _keyword_score(self, text: str, 
                       keywords: Dict[str, List[str]]) -> float:
        """关键词密度评分"""
        if not text:
            return 0.0
        
        total_matches = 0
        for lang_keywords in keywords.values():
            for kw in lang_keywords:
                count = text.count(kw.lower())
                total_matches += count
        
        if total_matches == 0:
            return 0.0
        
        # 归一化: 每个词最多贡献0.2分
        score = min(total_matches * 0.15, 1.0)
        return score
    
    def _url_pattern_score(self, url: str, 
                           patterns: List[str]) -> float:
        """URL模式匹配评分"""
        if not url:
            return 0.0
        
        for pattern in patterns:
            if re.search(pattern, url, re.I):
                return 0.9  # 匹配一个模式即高分
        
        return 0.0
    
    def _domain_score(self, domain_or_url: str, 
                      indicators: List[str]) -> float:
        """域名指示评分"""
        if not domain_or_url:
            return 0.0
        
        for indicator in indicators:
            if indicator.lower() in domain_or_url.lower():
                return 0.95
        
        return 0.0
    
    def classify_batch(self, items: List[Dict]) -> List[Tuple[Dict, str]]:
        """批量分类"""
        return [(item, self.classify(item)) for item in items]


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    classifier = AutoClassifier()
    
    test_items = [
        {"title": "Bitcoin Price Breaks $100K - Reuters", 
         "snippet": "Bitcoin hits new all-time high", 
         "url": "https://www.reuters.com/technology/bitcoin-100k-2026",
         "domain": "reuters.com"},
        {"title": "比特币突破10万美元 - 知乎",
         "snippet": "如何看待比特币突破10万美元",
         "url": "https://www.zhihu.com/question/123456",
         "domain": "zhihu.com"},
        {"title": "比特币白皮书 - 研究论文",
         "snippet": "Bitcoin: A Peer-to-Peer Electronic Cash System",
         "url": "https://arxiv.org/abs/1234.5678",
         "domain": "arxiv.org"},
    ]
    
    print(f"{'标题':<35} {'分类':<12}")
    print("=" * 50)
    for item in test_items:
        cat = classifier.classify(item)
        print(f"{item['title'][:33]:<35} {cat:<12}")