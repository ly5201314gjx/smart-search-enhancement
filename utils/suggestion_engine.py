"""
📐 算法六：搜索建议生成引擎 (Suggestion Engine)
================================================

基于四路信号混合的搜索建议生成系统:

1. 前缀匹配 (Prefix): 搜索历史补全
2. 协同过滤 (Collaborative): "搜索了X的人也搜索了Y"
3. 热门趋势 (Trending): 当前搜索热点
4. 语义扩展 (Semantic): 词向量近义词
"""

import re
import math
import json
import time
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter


@dataclass
class SearchSuggestion:
    """搜索建议"""
    text: str                       # 建议文本
    score: float                    # 综合得分 [0,1]
    source: str                     # 来源: prefix/collab/trend/semantic
    rationale: str = ""             # 推荐理由
    category: str = "general"       # 建议类别


class SuggestionEngine:
    """
    搜索建议生成引擎
    
    架构:
    输入: 当前查询Q₀ + 搜索历史H + 热门池T
    处理: 四路信号并行计算 → 加权融合 → Top-K选取
    输出: 搜索建议列表
    """
    
    def __init__(self, data_dir: str = "../data"):
        self.data_dir = data_dir
        
        # 搜索历史（内存缓存）
        self.search_history: List[str] = []
        self.history_timestamps: List[float] = []
        
        # 共现矩阵 M[i][j] = co_occurrence_count
        self.cooccurrence: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        
        # 热门趋势池
        self.trending_pool: Dict[str, float] = {}
        self.trending_history: Dict[str, List[float]] = defaultdict(list)
        
        # 语义扩展缓存
        self.semantic_cache: Dict[str, List[Tuple[str, float]]] = {}
        
        # 预设的常见搜索建议模板
        self.common_suggestions = self._init_common_suggestions()
        
        # 加载历史数据
        self._load_history()
    
    def _init_common_suggestions(self) -> Dict[str, List[str]]:
        """
        预设常见搜索建议（冷启动用）
        
        按类别分组
        """
        return {
            "technology": [
                "AI最新进展", "Python教程", "机器学习入门",
                "GPT-5发布", "深度学习框架对比",
                "AI news", "Python tutorial", "machine learning"
            ],
            "crypto": [
                "比特币价格", "以太坊行情", "Solana分析",
                "加密货币新闻", "DeFi最新动态",
                "Bitcoin price", "Ethereum news", "crypto market"
            ],
            "news": [
                "今日热点新闻", "科技新闻", "全球经济",
                "breaking news", "world news", "tech news"
            ],
            "academic": [
                "Transformer论文", "最新研究论文", "学术搜索",
                "research papers", "machine learning papers"
            ]
        }
    
    def generate(self, 
                 query: Dict,
                 results: List[Dict],
                 weights: Optional[Dict[str, float]] = None,
                 top_k: int = 5) -> List[SearchSuggestion]:
        """
        生成搜索建议
        
        Args:
            query: 当前查询信息
            results: 搜索结果列表
            weights: 各路信号的权重配置
            top_k: 返回建议数量
        
        Returns:
            搜索建议列表
        """
        query_text = query.get("processed_text", query.get("original_text", ""))
        
        if not query_text:
            return []
        
        w = weights or {
            "prefix_match": 0.35,
            "collaborative": 0.25,
            "trending": 0.25,
            "semantic_expansion": 0.15
        }
        
        # 记录当前查询到历史
        self._add_to_history(query_text)
        
        # 并行收集各路候选
        candidates = {}
        
        # 1. 前缀匹配
        prefix_candidates = self._prefix_match(query_text, top_k * 2)
        for text, score in prefix_candidates:
            candidates[text] = candidates.get(text, 0) + w["prefix_match"] * score
        
        # 2. 协同过滤
        collab_candidates = self._collaborative_filter(query_text, top_k * 2)
        for text, score in collab_candidates:
            candidates[text] = candidates.get(text, 0) + w["collaborative"] * score
        
        # 3. 热门趋势
        trend_candidates = self._trending_queries(query_text, top_k * 2)
        for text, score in trend_candidates:
            candidates[text] = candidates.get(text, 0) + w["trending"] * score
        
        # 4. 语义扩展
        semantic_candidates = self._semantic_expansion(query_text, top_k * 2)
        for text, score in semantic_candidates:
            candidates[text] = candidates.get(text, 0) + w["semantic_expansion"] * score
        
        # 5. 从结果提取（关键词+实体）
        result_candidates = self._extract_from_results(query_text, results, top_k)
        for text, score in result_candidates:
            candidates[text] = candidates.get(text, 0) + score * 0.2  # 低权重
        
        # 排序选取Top-K
        sorted_candidates = sorted(
            candidates.items(), key=lambda x: x[1], reverse=True
        )[:top_k]
        
        # 构建建议对象
        suggestions = []
        for text, score in sorted_candidates:
            if text.lower() == query_text.lower():
                continue  # 排除自身
            
            source = self._identify_source(text, query_text)
            rationale = self._generate_rationale(source, text)
            
            suggestions.append(SearchSuggestion(
                text=text,
                score=round(min(score, 1.0), 3),
                source=source,
                rationale=rationale,
                category=self._categorize_suggestion(text)
            ))
        
        return suggestions
    
    def _prefix_match(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """
        信号一：前缀匹配搜索补全
        
        在搜索历史中找以query为前缀的查询
        """
        query_lower = query.lower()
        candidates = []
        
        # 从历史中匹配
        for i, hist_q in enumerate(self.search_history):
            if hist_q.lower().startswith(query_lower) and len(hist_q) > len(query):
                # 时间衰减
                age = time.time() - self.history_timestamps[i]
                recency = math.exp(-0.1 * age / 3600)  # 10小时半衰期
                
                # 补全完整度得分
                completeness = len(query) / len(hist_q)
                
                score = 0.6 * completeness + 0.4 * recency
                candidates.append((hist_q, score))
        
        # 从常见建议中匹配
        for category, suggestions in self.common_suggestions.items():
            for sug in suggestions:
                if sug.lower().startswith(query_lower) and len(sug) > len(query):
                    candidates.append((sug, 0.3))  # 预设建议权重较低
        
        # 去重排序
        seen = set()
        unique_candidates = []
        for text, score in candidates:
            if text not in seen:
                seen.add(text)
                unique_candidates.append((text, score))
        
        unique_candidates.sort(key=lambda x: x[1], reverse=True)
        return unique_candidates[:top_k]
    
    def _collaborative_filter(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """
        信号二：协同过滤
        
        "搜索了X的人也搜索了Y"
        """
        query_lower = query.lower()
        
        # 查找与当前查询共现的查询
        related_queries = defaultdict(int)
        
        if query_lower in self.cooccurrence:
            for other_q, count in self.cooccurrence[query_lower].items():
                related_queries[other_q] += count
        
        if not related_queries:
            return []
        
        # 按共现次数排序
        sorted_related = sorted(
            related_queries.items(), key=lambda x: x[1], reverse=True
        )
        
        # 归一化得分
        max_count = sorted_related[0][1] if sorted_related else 1
        results = [
            (q, min(c / max_count, 1.0)) 
            for q, c in sorted_related[:top_k]
        ]
        
        return results
    
    def _trending_queries(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """
        信号三：热门趋势
        
        基于搜索频率的增长率计算趋势得分
        """
        query_lower = query.lower()
        candidates = []
        
        now = time.time()
        one_hour_ago = now - 3600
        two_hours_ago = now - 7200
        
        for q, timestamps in self.trending_history.items():
            if q.lower() == query_lower:
                continue
            
            # 最近1小时频次
            recent_count = sum(1 for t in timestamps if t > one_hour_ago)
            # 前1小时频次
            prev_count = sum(1 for t in timestamps 
                           if two_hours_ago < t <= one_hour_ago)
            
            # 趋势速度（增长率）
            velocity = (recent_count - prev_count) / (prev_count + 0.001)
            
            # 基础热度
            base_popularity = min(len(timestamps) / 100, 1.0)
            
            score = 0.6 * base_popularity + 0.4 * min(velocity, 2.0) / 2.0
            candidates.append((q, score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 如果没有足够的热门查询，补充预设
        if len(candidates) < top_k:
            for category, suggestions in self.common_suggestions.items():
                for sug in suggestions:
                    if sug.lower() != query_lower:
                        candidates.append((sug, 0.2))
        
        seen = set()
        unique = []
        for text, score in candidates:
            if text not in seen:
                seen.add(text)
                unique.append((text, score))
        
        return unique[:top_k]
    
    def _semantic_expansion(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """
        信号四：语义扩展
        
        基于关键词的同义词/近义词/上下位词扩展
        """
        query_lower = query.lower()
        
        # 如果缓存中有，直接返回
        if query_lower in self.semantic_cache:
            return self.semantic_cache[query_lower][:top_k]
        
        # 简化版语义扩展（基于规则）
        expansions = []
        
        # 中英文同义扩展
        synonym_map = {
            "比特币": "Bitcoin BTC 加密货币 数字黄金 virtual currency",
            "以太坊": "Ethereum ETH 智能合约 smart contract",
            "价格": "price 行情 走势 报价 market value",
            "新闻": "news 资讯 报道 头条 headlines",
            "教程": "tutorial guide 指南 入门 教学",
            "股票": "stock 股市 大盘 指数 equity",
            "AI": "人工智能 机器学习 深度学习 machine learning",
            "天气": "weather 气候 温度 预报 forecast",
            "python": "Python 编程 代码 code programming",
            "论文": "paper 文献 研究 论文 research article"
        }
        
        for key, expansions_str in synonym_map.items():
            if key.lower() in query_lower:
                for exp in expansions_str.split():
                    expansions.append((query.replace(key, exp).strip(), 0.7))
        
        # 前缀扩展（常见后缀）
        suffixes = ["最新", "2026", "教程", "价格", "新闻", "推荐", "排行",
                   "对比", "分析", "review", "price", "news", "2026", "tutorial"]
        for suffix in suffixes:
            if suffix not in query_lower:
                expansions.append((f"{query} {suffix}", 0.3))
        
        # 缓存结果
        self.semantic_cache[query_lower] = expansions
        
        return expansions[:top_k]
    
    def _extract_from_results(self, query: str, results: List[Dict], 
                               top_k: int) -> List[Tuple[str, float]]:
        """从搜索结果中提取关键词作为建议"""
        candidates = Counter()
        
        for result in results[:5]:  # 只看前5个结果
            title = result.get("title", "")
            # 提取标题中不在查询里的词
            query_words = set(query.lower().split())
            title_words = set()
            
            for word in re.findall(r'[\u4e00-\u9fff\w]+', title.lower()):
                title_words.add(word)
            
            new_words = title_words - query_words
            for word in new_words:
                if len(word) >= 2:  # 过滤单字
                    candidates[word] += 1
        
        # 取最高频的
        return [(word, min(count / 3, 1.0)) 
                for word, count in candidates.most_common(top_k)]
    
    def _identify_source(self, suggestion: str, query: str) -> str:
        """识别建议来源"""
        query_lower = query.lower()
        sug_lower = suggestion.lower()
        
        if sug_lower.startswith(query_lower):
            return "prefix"
        elif any(kw in sug_lower for kw in query_lower.split()):
            return "semantic"
        else:
            return "trending"
    
    def _generate_rationale(self, source: str, text: str) -> str:
        """生成推荐理由"""
        rationales = {
            "prefix": f"补全: {text}",
            "collab": f"相关搜索: {text}",
            "trending": f"热门: {text}",
            "semantic": f"相关: {text}"
        }
        return rationales.get(source, f"推荐: {text}")
    
    def _categorize_suggestion(self, text: str) -> str:
        """建议分类"""
        for category, suggestions in self.common_suggestions.items():
            for sug in suggestions:
                if sug.lower() in text.lower() or text.lower() in sug.lower():
                    return category
        return "general"
    
    def _add_to_history(self, query: str):
        """添加到搜索历史"""
        self.search_history.append(query)
        self.history_timestamps.append(time.time())
        
        # 更新共现矩阵（与最近3个查询建立关联）
        if len(self.search_history) >= 2:
            current = query.lower()
            for past_q in self.search_history[-4:-1]:  # 前3个
                past_lower = past_q.lower()
                if current != past_lower:
                    self.cooccurrence[current][past_lower] += 1
                    self.cooccurrence[past_lower][current] += 1
        
        # 更新趋势
        self.trending_history[query.lower()].append(time.time())
        
        # 限制历史大小
        if len(self.search_history) > 1000:
            self.search_history = self.search_history[-500:]
            self.history_timestamps = self.history_timestamps[-500:]
        
        # 持久化
        self._save_history()
    
    def _load_history(self):
        """加载搜索历史（持久化）"""
        # 简化版：在生产中会从JSON文件加载
        pass
    
    def _save_history(self):
        """保存搜索历史"""
        # 简化版：在生产中会写入JSON文件
        pass


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    engine = SuggestionEngine()
    
    # 模拟搜索历史
    test_history = [
        "比特币价格", "以太坊行情", "比特币今日", "加密货币新闻",
        "AI最新进展", "Python教程", "深度学习"
    ]
    for q in test_history:
        engine._add_to_history(q)
    
    # 生成建议
    test_queries = ["比", "比特币", "AI"]
    
    for q in test_queries:
        suggestions = engine.generate(
            {"processed_text": q, "original_text": q},
            [],
            top_k=5
        )
        print(f"\n查询: '{q}'")
        for sug in suggestions:
            print(f"  [{sug.source:8}] {sug.text:<20} (score={sug.score:.3f})")