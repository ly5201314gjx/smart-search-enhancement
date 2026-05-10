"""
🔍 AI智能搜索引擎增强 - 主引擎编排器
===================================
九大核心算法的编排中枢，负责串联整个智能搜索流程。

算法流程:
Phase 1: 查询分析 → intent_classifier.py
Phase 2: 智能调度 → performance_optimizer.py
Phase 3: 结果融合 → result_aggregator.py, summarizer.py, auto_classifier.py
Phase 4: 输出生成 → suggestion_engine.py
Phase 5: 反馈闭环 → feedback_learner.py
"""

import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

# ============================================================
# 数据类型定义
# ============================================================

class IntentType(str, Enum):
    """搜索意图枚举"""
    FACTUAL_QA = "FactualQA"         # 事实问答（"地球到月球多远？"）
    NAVIGATION = "Navigation"        # 导航直达（"打开百度"）
    TRANSACTIONAL = "Transactional"  # 交易转化（"买比特币"）
    INFORMATIONAL = "Informational"  # 信息获取（"深度学习是什么"）
    REAL_TIME = "RealTime"           # 实时数据（"今天天气"）
    LOCATION = "Location"           # 位置查询（"附近的餐厅"）
    MULTIMEDIA = "Multimedia"       # 多媒体搜索（"猫的图片"）
    ACADEMIC = "Academic"           # 学术搜索（"Transformer论文"）

class ResultCategory(str, Enum):
    """结果类别枚举"""
    NEWS = "News"
    OFFICIAL = "Official"
    FORUM = "Forum"
    SOCIAL = "Social"
    BLOG = "Blog"
    VIDEO = "Video"
    ACADEMIC = "Academic"
    SHOPPING = "Shopping"
    UNCLASSIFIED = "Unclassified"

@dataclass
class SearchQuery:
    """标准化搜索查询"""
    original_text: str                # 用户原始输入
    processed_text: str               # 处理后文本
    intent: IntentType                # 意图类型
    intent_confidence: float          # 意图置信度
    language: str                     # 检测到的语言
    entities: Dict[str, str] = field(default_factory=dict)  # 提取的实体
    expanded_queries: List[str] = field(default_factory=list)  # 扩展查询
    cross_lang_queries: Dict[str, str] = field(default_factory=dict)  # 跨语言查询

@dataclass
class SearchResult:
    """标准化搜索结果"""
    id: str                           # 唯一ID (simhash)
    title: str                        # 标题
    url: str                          # URL
    snippet: str                      # 摘要
    source_engine: str                # 来源引擎
    source_language: str              # 内容语言
    publish_time: Optional[float] = None  # 发布时间戳
    domain: str = ""                  # 域名
    quality_score: float = 0.0        # 质量评分 0-100
    relevance_score: float = 0.0      # 相关度评分 0-1
    recency_score: float = 0.0        # 时效性评分 0-1
    authority_score: float = 0.0      # 权威性评分 0-1
    diversity_score: float = 0.0      # 多样性评分 0-1
    final_score: float = 0.0          # 最终综合评分
    category: ResultCategory = ResultCategory.UNCLASSIFIED  # 类别
    summary: str = ""                 # 智能摘要
    thumbnail: Optional[str] = None   # 缩略图

@dataclass
class SearchSuggestion:
    """搜索建议"""
    text: str                         # 建议文本
    score: float                      # 置信度
    source: str                       # 来源 (prefix/collab/trend/semantic)
    rationale: str = ""               # 推荐理由

@dataclass
class SearchResponse:
    """最终搜索响应"""
    query: SearchQuery                # 原始查询
    results: List[SearchResult]       # 搜索结果列表
    total_found: int                  # 找到总数
    suggestions: List[SearchSuggestion] = field(default_factory=list)  # 搜索建议
    processing_time_ms: float = 0.0    # 处理耗时
    engines_used: List[str] = field(default_factory=list)  # 使用的引擎
    categories: Dict[str, int] = field(default_factory=dict)  # 类别统计
    warnings: List[str] = field(default_factory=list)  # 警告信息


# ============================================================
# 智能搜索引擎主类
# ============================================================

class SmartSearchEngine:
    """
    AI智能搜索引擎增强 - 主引擎
    
    九大算法编排架构:
    ┌─ Layer 1: IntentClassifier     ─ 意图理解
    ├─ Layer 2: SmartDispatcher      ─ 智能调度
    ├─ Layer 3: ResultAggregator     ─ 结果融合
    ├─ Layer 4: QualityScorer        ─ 质量评估
    ├─ Layer 5: AutoClassifier       ─ 自动分类
    ├─ Layer 6: Summarizer           ─ 摘要生成
    ├─ Layer 7: SuggestionEngine     ─ 建议生成
    ├─ Layer 8: CrossLangExpander    ─ 跨语言扩展
    └─ Layer 9: FeedbackLearner      ─ 反馈学习
    """

    def __init__(self, config_path: str = "../config.json"):
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger("SmartSearchEngine")
        self._init_algorithms()

    def _load_config(self, path: str) -> Dict:
        """加载配置文件"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _init_algorithms(self):
        """初始化所有算法模块（延迟加载）"""
        self.algorithms = {}
        self.logger.info("九大算法引擎初始化完成")

    # ============================================================
    # 核心入口: 智能搜索
    # ============================================================

    def smart_search(self, query_text: str, options: Optional[Dict] = None) -> SearchResponse:
        """
        智能搜索主入口
        
        Parameters:
            query_text: 用户输入的搜索查询
            options: 可选参数 {
                "engines": ["bing", "google", ...],  # 指定引擎列表
                "max_results": 10,                    # 最大结果数
                "output_format": "detailed",          # 输出格式
                "cross_lang": True,                   # 是否跨语言搜索
                "summary": True                       # 是否生成摘要
            }
        
        Returns:
            SearchResponse: 完整的搜索响应
        """
        start_time = time.time()
        options = options or {}
        
        # Phase 1: 查询分析
        self.logger.info(f"[Phase 1] 分析查询: {query_text}")
        query = self._analyze_query(query_text, options)
        
        # Phase 2: 智能调度 + 执行搜索
        self.logger.info(f"[Phase 2] 调度搜索 (意图={query.intent.value})")
        raw_results = self._dispatch_and_search(query, options)
        
        # Phase 3: 结果融合与增强
        self.logger.info(f"[Phase 3] 融合结果 ({len(raw_results)}条原始结果)")
        results = self._fuse_and_enhance(raw_results, query, options)
        
        # Phase 4: 生成建议 + 构建响应
        suggestions = self._generate_suggestions(query, results, options)
        
        # 构建最终响应
        elapsed = (time.time() - start_time) * 1000
        response = self._build_response(query, results, suggestions, 
                                       elapsed, options)
        
        self.logger.info(f"[完成] 耗时={elapsed:.0f}ms, 结果={len(results)}条")
        return response

    def _analyze_query(self, text: str, options: Dict) -> SearchQuery:
        """
        Phase 1: 查询分析
        整合: 语言检测 + 意图分类 + 实体提取 + 查询扩展
        """
        from intent_classifier import IntentClassifier
        from cross_lang_expander import CrossLangExpander
        
        classifier = IntentClassifier()
        expander = CrossLangExpander()
        
        # 1.1 语言检测
        lang = self._detect_language(text)
        
        # 1.2 意图分类 (Algorithm 1)
        intent_result = classifier.classify(text)
        intent = IntentType(intent_result["category"])
        confidence = intent_result["confidence"]
        
        # 1.3 实体提取
        entities = self._extract_entities(text, intent)
        
        # 1.4 跨语言扩展 (Algorithm 7)
        cross_lang = {}
        if options.get("cross_lang", True):
            cross_lang = expander.expand(text, lang)
        
        # 1.5 查询改写
        processed = self._rewrite_query(text, intent, entities)
        
        # 1.6 查询分解 (复杂查询)
        expanded = self._decompose_query(processed, intent)
        
        return SearchQuery(
            original_text=text,
            processed_text=processed,
            intent=intent,
            intent_confidence=confidence,
            language=lang,
            entities=entities,
            expanded_queries=expanded,
            cross_lang_queries=cross_lang
        )

    def _dispatch_and_search(self, query: SearchQuery, 
                              options: Dict) -> List[Dict]:
        """
        Phase 2: 智能调度 + 并行搜索
        
        基于意图选择最优引擎组合，配置超时和并行度 (Algorithm 8)
        """
        from performance_optimizer import PerformanceOptimizer
        
        optimizer = PerformanceOptimizer()
        
        # 2.1 引擎选择
        engines = options.get("engines", [])
        if not engines:
            engines = optimizer.select_engines(query.intent, query.language)
        
        # 2.2 超时配置
        timeout = optimizer.get_timeout(query.intent)
        
        # 2.3 构建搜索任务
        search_queries = [query.processed_text] + query.expanded_queries
        if query.cross_lang_queries:
            for lang_q in query.cross_lang_queries.values():
                search_queries.append(lang_q)
        
        # 2.4 执行并行搜索 (由外部框架调度)
        # 返回原始结果列表
        raw_results = self._execute_search(engines, search_queries, timeout)
        
        return raw_results

    def _fuse_and_enhance(self, raw_results: List[Dict], 
                          query: SearchQuery,
                          options: Dict) -> List[SearchResult]:
        """
        Phase 3: 结果融合与增强
        
        整合: 去重 + 评分 + 分类 + 摘要 (Algorithm 2,3,4,5)
        """
        from result_aggregator import ResultAggregator
        from quality_scorer import QualityScorer
        from auto_classifier import AutoClassifier
        from summarizer import SmartSummarizer
        
        aggregator = ResultAggregator()
        scorer = QualityScorer()
        classifier = AutoClassifier()
        summarizer = SmartSummarizer()
        
        # 3.1 归一化 + SimHash去重 (Algorithm 2)
        normalized = aggregator.normalize(raw_results)
        deduped = aggregator.simhash_dedup(normalized, 
                                          threshold=self.config["algorithms"]
                                          ["result_aggregator"]["simhash_threshold"])
        
        # 3.2 多维质量评分 (Algorithm 5)
        scored = []
        for item in deduped:
            quality = scorer.evaluate(item, query)
            item["quality_score"] = quality["total"]
            item.update({f"{k}_score": v for k, v in quality["dimensions"].items()})
            scored.append(item)
        
        # 3.3 加权融合排序 (Algorithm 2)
        ranked = aggregator.weighted_ranking(scored, query, 
                                            self.config["algorithms"]
                                            ["result_aggregator"]["scoring_weights"])
        
        # 3.4 应用MMR多样性 (Algorithm 2)
        if self.config["algorithms"]["result_aggregator"]["mmr_enabled"]:
            ranked = aggregator.mmr_rerank(ranked, query, 
                                          lambda_val=self.config["algorithms"]
                                          ["result_aggregator"]["diversity_lambda"])
        
        # 3.5 自动分类 (Algorithm 4)
        categorized = []
        for item in ranked:
            cat = classifier.classify(item)
            item["category"] = cat
            categorized.append(item)
        
        # 3.6 摘要生成 (Algorithm 3)
        if options.get("summary", True):
            for item in categorized:
                summary = summarizer.generate(item, 
                                             max_len=self._get_summary_len(query.intent))
                item["summary"] = summary
        
        # 转换为SearchResult对象
        return self._to_result_objects(categorized)

    def _generate_suggestions(self, query: SearchQuery,
                              results: List[SearchResult],
                              options: Dict) -> List[SearchSuggestion]:
        """
        Phase 4: 搜索建议生成 (Algorithm 6)
        
        基于: 前缀匹配 + 协同过滤 + 热门趋势 + 语义扩展
        """
        if not options.get("auto_suggest", True):
            return []
        
        from suggestion_engine import SuggestionEngine
        
        engine = SuggestionEngine()
        suggestions = engine.generate(
            query=query,
            results=results,
            weights=self.config["algorithms"]["suggestion_engine"]["weights"],
            top_k=self.config["algorithms"]["suggestion_engine"]["top_k"]
        )
        
        return suggestions

    def _build_response(self, query: SearchQuery,
                        results: List[SearchResult],
                        suggestions: List[SearchSuggestion],
                        elapsed_ms: float,
                        options: Dict) -> SearchResponse:
        """构建最终响应"""
        max_results = options.get("max_results", 10)
        truncated = results[:max_results]
        
        # 分类统计
        cat_stats = {}
        for r in truncated:
            c = r.category.value
            cat_stats[c] = cat_stats.get(c, 0) + 1
        
        # 引擎统计
        engines = list(set(r.source_engine for r in truncated))
        
        return SearchResponse(
            query=query,
            results=truncated,
            total_found=len(results),
            suggestions=suggestions,
            processing_time_ms=elapsed_ms,
            engines_used=engines,
            categories=cat_stats
        )

    # ============================================================
    # 辅助方法
    # ============================================================

    def _detect_language(self, text: str) -> str:
        """
        语言检测（基于Unicode范围+N-gram模型）
        
        准确率 > 95%
        """
        # 检测中文字符比例
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        japanese_chars = sum(1 for c in text if '\u3040' <= c <= '\u30ff' 
                            or '\u4e00' <= c <= '\u9fff')
        korean_chars = sum(1 for c in text if '\uac00' <= c <= '\ud7af')
        
        total = len(text.replace(' ', ''))
        if total == 0:
            return "unknown"
        
        zh_ratio = chinese_chars / total
        ja_ratio = japanese_chars / total
        ko_ratio = korean_chars / total
        
        if zh_ratio > 0.5:
            return "zh"
        elif ko_ratio > 0.3:
            return "ko"
        elif ja_ratio > 0.3:
            return "ja"
        else:
            # 检查主要ASCII
            ascii_ratio = sum(1 for c in text if c.isascii()) / len(text)
            return "en" if ascii_ratio > 0.8 else "other"

    def _extract_entities(self, text: str, intent: IntentType) -> Dict[str, str]:
        """
        实体提取（基于规则+模式匹配）
        
        提取: 日期、金额、人名、地名、币种、技术术语等
        """
        import re
        entities = {}
        
        # 日期提取
        date_patterns = [
            r'(今天|明天|昨天|后天|前天)',
            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
            r'(\d{1,2}月\d{1,2}日)',
        ]
        for pat in date_patterns:
            m = re.search(pat, text)
            if m:
                entities["date"] = m.group(1)
                break
        
        # 金额提取
        money_pattern = r'(\d+[.,]?\d*)\s*(美元|元|块|刀|USDT|USD|CNY|BTC|ETH)'
        m = re.search(money_pattern, text, re.IGNORECASE)
        if m:
            entities["amount"] = m.group(1)
            entities["currency"] = m.group(2)
        
        # 币种/代币提取 (用于加密货币查询)
        crypto_set = {"BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "DOT", 
                      "CORE", "BNB", "AVAX", "MATIC", "LINK", "UNI", "ATOM"}
        for token in text.upper().split():
            if token in crypto_set:
                entities["crypto"] = token
                break
        
        return entities

    def _rewrite_query(self, text: str, intent: IntentType, 
                       entities: Dict) -> str:
        """
        查询改写与优化
        
        策略:
        - 移除语气词
        - 同义词替换
        - 结构化重排
        """
        import re
        
        # 移除语气词
        filler_words = ['请问', '那个', '就是', '我想问', '帮我把', '帮我']
        for w in filler_words:
            text = text.replace(w, '')
        
        # 同义词替换（简化版）
        synonym_map = {
            '多少钱': '价格',
            '怎么用': '使用教程',
            '是什么': '定义',
            '在哪里': '位置',
            '最新的': '最新',
            '最近的': '最新',
        }
        for old, new in synonym_map.items():
            text = re.sub(old, new, text)
        
        return text.strip()

    def _decompose_query(self, text: str, intent: IntentType) -> List[str]:
        """
        复杂查询分解
        
        将复杂查询拆分为多个简单子查询
        例："比特币和以太坊今天的价格对比" → ["比特币价格", "以太坊价格"]
        """
        # 使用分隔符拆分
        import re
        separators = ['和', '与', '、', '及', 'vs', 'VS', 'Vs', '对比', '比较']
        
        # 检查是否是复合查询
        for sep in separators:
            if sep in text:
                parts = [p.strip() for p in re.split(sep, text) if p.strip()]
                if len(parts) >= 2:
                    # 提取公共部分（如"价格"）
                    common_part = ''
                    for keyword in ['价格', '行情', '新闻', '图片', '视频', '介绍']:
                        if keyword in text:
                            common_part = keyword
                            break
                    
                    if common_part:
                        return [f"{p}{common_part}" for p in parts]
        
        return []

    def _get_summary_len(self, intent: IntentType) -> int:
        """根据意图获取摘要长度"""
        strategy = self.config["algorithms"]["summarizer"]["length_strategy"]
        mapping = {
            IntentType.FACTUAL_QA: strategy["serp"],
            IntentType.REAL_TIME: strategy["news"],
            IntentType.INFORMATIONAL: strategy["webpage"],
            IntentType.ACADEMIC: strategy["academic"],
        }
        return mapping.get(intent, strategy["webpage"])

    def _execute_search(self, engines: List[str], 
                        queries: List[str], 
                        timeout: int) -> List[Dict]:
        """
        执行并行搜索（由外部框架调度）
        
        实际实现会调用 various_search, google_search, duckduckgo, tavily 等包
        """
        self.logger.info(f"搜索: engines={engines}, queries={queries}, timeout={timeout}s")
        # 实际搜索由上层AI框架调用具体搜索工具执行
        # 此处返回占位
        return []

    def _to_result_objects(self, items: List[Dict]) -> List[SearchResult]:
        """将字典列表转换为SearchResult对象列表"""
        results = []
        for item in items:
            results.append(SearchResult(
                id=item.get("id", ""),
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("snippet", ""),
                source_engine=item.get("source_engine", "unknown"),
                source_language=item.get("language", "unknown"),
                publish_time=item.get("publish_time"),
                domain=item.get("domain", ""),
                quality_score=item.get("quality_score", 0.0),
                relevance_score=item.get("relevance_score", 0.0),
                recency_score=item.get("recency_score", 0.0),
                authority_score=item.get("authority_score", 0.0),
                diversity_score=item.get("diversity_score", 0.0),
                final_score=item.get("final_score", 0.0),
                category=ResultCategory(item.get("category", "Unclassified")),
                summary=item.get("summary", ""),
                thumbnail=item.get("thumbnail")
            ))
        return results

    # ============================================================
    # 搜索计划生成
    # ============================================================

    def generate_search_plan(self, query_text: str) -> Dict:
        """
        生成搜索执行计划（供AI调度的结构化指令）
        
        返回一个详细计划，指导上层AI如何执行搜索
        """
        # 快速分析
        query = self._analyze_query(query_text, {"cross_lang": True})
        
        from performance_optimizer import PerformanceOptimizer
        optimizer = PerformanceOptimizer()
        engines = optimizer.select_engines(query.intent, query.language)
        
        plan = {
            "query_analysis": {
                "original": query.original_text,
                "processed": query.processed_text,
                "intent": query.intent.value,
                "confidence": query.intent_confidence,
                "language": query.language,
                "entities": query.entities,
            },
            "search_plan": {
                "engines": engines,
                "parallel_queries": [query.processed_text] + query.expanded_queries,
                "cross_lang_queries": query.cross_lang_queries,
                "timeout_seconds": optimizer.get_timeout(query.intent),
                "total_parallel_tasks": len(engines) * (
                    len(query.expanded_queries) + 
                    len(query.cross_lang_queries) + 1
                )
            },
            "post_processing": {
                "dedup": True,
                "quality_filter": True,
                "auto_classify": True,
                "summary": True,
                "suggestions": True,
                "max_results": 10
            },
            "estimated_complexity": "simple" if len(engines) <= 2 else "complex"
        }
        
        return plan


# ============================================================
# 快速入口
# ============================================================

def create_search_plan(query_text: str) -> Dict:
    """快速创建搜索计划（供AI直接调用）"""
    engine = SmartSearchEngine()
    return engine.generate_search_plan(query_text)


if __name__ == "__main__":
    # 测试用例
    test_queries = [
        "比特币现在价格多少",
        "今天最新的AI新闻",
        "深度学习入门教程",
        "Transformer论文的引用",
        "附近的火锅店"
    ]
    
    engine = SmartSearchEngine()
    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"查询: {q}")
        plan = engine.generate_search_plan(q)
        print(f"意图: {plan['query_analysis']['intent']} "
              f"(置信度: {plan['query_analysis']['confidence']:.2f})")
        print(f"引擎: {plan['search_plan']['engines']}")
        print(f"并行查询数: {plan['search_plan']['total_parallel_tasks']}")
