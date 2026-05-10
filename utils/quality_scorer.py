"""
📐 算法五：搜索质量评估器 (Search Quality Scorer)
==================================================

多维质量评分模型:
QA_Score = Σ(wᵢ · Qᵢ(E))

Q₁: 内容完整性 (Completeness)
Q₂: 信息新鲜度 (Freshness)  
Q₃: 来源可信度 (Trustworthiness)
Q₄: 内容可读性 (Readability)
Q₅: 关键词覆盖率 (Coverage)
Q₆: 内容丰富度 (Richness)
"""

import re
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class QualityResult:
    """质量评分结果"""
    total: float                    # 总分 [0, 100]
    grade: str                      # 等级: ⭐⭐⭐/⭐⭐/⭐/❌
    dimensions: Dict[str, float]    # 各维度得分
    pass_threshold: bool            # 是否通过质量过滤
    warnings: List[str]             # 警告信息


class QualityScorer:
    """
    搜索质量评估器
    
    六维度评分模型，输出 [0, 100] 的综合质量分数
    """
    
    # 各维度权重
    DEFAULT_WEIGHTS = {
        "completeness": 0.15,
        "freshness": 0.20,
        "trustworthiness": 0.25,
        "readability": 0.10,
        "coverage": 0.20,
        "richness": 0.10
    }
    
    # 质量阈值
    THRESHOLDS = {
        "excellent": 80,    # ⭐⭐⭐
        "standard": 60,     # ⭐⭐
        "low_quality": 40,  # ⭐
        "reject": 40        # ❌ (低于此分数过滤)
    }
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        # 校验权重和=1
        total_w = sum(self.weights.values())
        if abs(total_w - 1.0) > 0.001:
            # 归一化
            for k in self.weights:
                self.weights[k] /= total_w
    
    def evaluate(self, item: Dict, query: Dict) -> QualityResult:
        """
        六维度质量评估
        
        Args:
            item: 搜索结果条目（已归一化）
            query: 搜索查询信息
        
        Returns:
            QualityResult: 质量评分结果
        """
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        url = item.get("url", "")
        content = item.get("content", item.get("snippet", ""))
        query_text = query.get("processed_text", query.get("original_text", ""))
        
        warnings = []
        
        # Q₁: 内容完整性
        completeness = self._score_completeness(title, snippet, url)
        if completeness < 0.3:
            warnings.append("内容完整度低")
        
        # Q₂: 信息新鲜度
        freshness = self._score_freshness(item)
        
        # Q₃: 来源可信度
        trustworthiness = self._score_trustworthiness(url, item.get("source_engine", ""))
        if trustworthiness < 0.4:
            warnings.append("来源可信度较低")
        
        # Q₄: 内容可读性
        readability = self._score_readability(content)
        if readability < 0.3:
            warnings.append("可读性差")
        
        # Q₅: 关键词覆盖率
        coverage = self._score_coverage(content, query_text)
        if coverage < 0.2:
            warnings.append("关键词覆盖率低")
        
        # Q₆: 内容丰富度
        richness = self._score_richness(content, item)
        
        # 综合评分
        dimensions = {
            "completeness": round(completeness, 3),
            "freshness": round(freshness, 3),
            "trustworthiness": round(trustworthiness, 3),
            "readability": round(readability, 3),
            "coverage": round(coverage, 3),
            "richness": round(richness, 3)
        }
        
        total = sum(
            self.weights.get(dim, 0) * score 
            for dim, score in dimensions.items()
        )
        total_score = round(total * 100, 1)
        
        # 等级判定
        grade = self._get_grade(total_score)
        
        # 过滤判定
        pass_threshold = total_score >= self.THRESHOLDS["reject"]
        
        return QualityResult(
            total=total_score,
            grade=grade,
            dimensions=dimensions,
            pass_threshold=pass_threshold,
            warnings=warnings
        )
    
    def _score_completeness(self, title: str, snippet: str, 
                             url: str) -> float:
        """
        Q₁: 内容完整性评分
        
        维度:
        - title有内容且长度≥5字: 0.3
        - 有snippet: 0.3
        - 有thumbnail/image: 0.2
        - 有结构化数据: 0.2
        """
        score = 0.0
        
        # 标题完整度
        if title and len(title) >= 5:
            score += 0.3
        elif title and len(title) >= 2:
            score += 0.15
        
        # 摘要完整度
        if snippet and len(snippet) >= 20:
            score += 0.3
        elif snippet and len(snippet) >= 5:
            score += 0.15
        
        # URL完整度
        if url and len(url) >= 10:
            score += 0.2
        
        # 域名可识别
        if url and re.search(r'https?://[^/]+\.[a-z]{2,}', url):
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_freshness(self, item: Dict) -> float:
        """
        Q₂: 信息新鲜度
        
        基于发布时间或间接信号
        """
        publish_time = item.get("publish_time")
        if publish_time is not None:
            # 使用指数衰减模型
            import time
            delta_hours = (time.time() - publish_time) / 3600
            if delta_hours < 0:
                return 0.0
            
            # 24小时内新鲜度最高
            if delta_hours <= 24:
                return 1.0 - (delta_hours / 24) * 0.2
            elif delta_hours <= 168:  # 1周内
                return 0.8 - (delta_hours - 24) / (168 - 24) * 0.3
            elif delta_hours <= 720:  # 1月内
                return 0.5 - (delta_hours - 168) / (720 - 168) * 0.2
            else:
                return max(0.1, 0.3 - math.log10(delta_hours / 720) * 0.1)
        
        # 无时间戳，使用间接信号
        # 标题/摘要包含"最新"等词
        text = (item.get("title", "") + " " + item.get("snippet", "")).lower()
        fresh_indicators = ["最新", "刚刚", "今日", "昨天", "今天",
                          "new", "latest", "breaking", "just now", "today"]
        
        indicator_score = sum(1 for ind in fresh_indicators if ind in text)
        return min(0.5 + indicator_score * 0.1, 0.8)
    
    def _score_trustworthiness(self, url: str, engine: str) -> float:
        """
        Q₃: 来源可信度
        
        基于域名权威性 + 引擎可信度
        """
        from result_aggregator import AuthorityScorer
        
        auth_scorer = AuthorityScorer()
        auth_result = auth_scorer.score(url)
        
        # 域名权威性占70%，引擎可信度占30%
        engine_scores = {
            "google": 0.95, "bing": 0.90, "tavily": 0.88,
            "duckduckgo": 0.85, "baidu": 0.75, "sogou": 0.70
        }
        engine_score = engine_scores.get(engine.lower(), 0.60)
        
        return 0.7 * auth_result["total"] + 0.3 * engine_score
    
    def _score_readability(self, text: str) -> float:
        """
        Q₄: 内容可读性
        
        使用简化的可读性公式:
        - 中文: 长词(≥4字)比例越低越易读
        - 英文: 短句比例越高越易读
        """
        if not text:
            return 0.0
        
        # 检测语言
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        
        if chinese_chars > len(text) * 0.3:
            # 中文可读性
            # 长词比例（≥4字成语/术语）
            words = re.findall(r'[\u4e00-\u9fff]+', text)
            if not words:
                return 0.5
            
            long_words = sum(1 for w in words if len(w) >= 4)
            long_ratio = long_words / len(words)
            
            return max(0, min(1, 1.0 - long_ratio * 0.5))
        else:
            # 英文可读性（近似Flesch公式）
            sentences = re.split(r'[.!?]+', text)
            sentences = [s for s in sentences if s.strip()]
            
            if not sentences:
                return 0.5
            
            words = text.split()
            if not words:
                return 0.5
            
            avg_words_per_sentence = len(words) / len(sentences)
            avg_syllables = sum(
                self._count_syllables(w) for w in words
            ) / len(words)
            
            # Flesch Reading Ease ≈ 206.835 - 1.015*avg_words - 84.6*avg_syllables
            flesch = 206.835 - 1.015 * avg_words_per_sentence - 84.6 * avg_syllables
            
            # 归一化到 [0, 1]
            return max(0, min(1, (flesch + 20) / 140))
    
    def _score_coverage(self, content: str, query: str) -> float:
        """
        Q₅: 关键词覆盖率
        
        coverage = count(unique(Q_tokens ∩ E_tokens)) / count(unique(Q_tokens))
        """
        if not query or not content:
            return 0.0
        
        query_tokens = set(self._tokenize(query.lower()))
        content_tokens = set(self._tokenize(content.lower()))
        
        if not query_tokens:
            return 1.0
        
        intersection = query_tokens & content_tokens
        if not intersection:
            return 0.1  # 至少匹配一个保底
        
        return len(intersection) / len(query_tokens)
    
    def _score_richness(self, content: str, item: Dict) -> float:
        """
        Q₆: 内容丰富度
        
        - 内容长度: min(chars/500, 1.0) × 0.6
        - 多媒体: has_image/video × 0.2
        - 外部链接: count(external_links) × 0.1 (上限0.2)
        """
        # 内容长度得分
        content_len = len(content)
        length_score = min(content_len / 500, 1.0) * 0.6
        
        # 多媒体得分
        multimedia_score = 0.0
        if item.get("thumbnail"):
            multimedia_score += 0.1
        if re.search(r'\.(jpg|png|gif|mp4|mp3|svg)', 
                     item.get("url", ""), re.I):
            multimedia_score += 0.1
        
        # 链接丰富度
        link_count = len(re.findall(r'https?://', content)) if content else 0
        link_score = min(link_count / 5, 1.0) * 0.2
        
        return min(length_score + multimedia_score + link_score, 1.0)
    
    def _get_grade(self, score: float) -> str:
        """根据分数返回等级"""
        if score >= self.THRESHOLDS["excellent"]:
            return "⭐⭐⭐"
        elif score >= self.THRESHOLDS["standard"]:
            return "⭐⭐"
        elif score >= self.THRESHOLDS["low_quality"]:
            return "⭐"
        else:
            return "❌"
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        tokens = []
        i = 0
        while i < len(text):
            char = text[i]
            if '\u4e00' <= char <= '\u9fff':
                tokens.append(char)
            elif char.isalnum():
                j = i
                while j < len(text) and text[j].isalnum():
                    j += 1
                tokens.append(text[i:j].lower())
                i = j
                continue
            i += 1
        return tokens
    
    def _count_syllables(self, word: str) -> int:
        """
        简单音节计数（英文）
        
        近似统计: 元音组数量
        """
        word = word.lower().strip()
        if not word:
            return 1
        
        count = 0
        vowels = 'aeiouy'
        prev_is_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
        
        # 末尾e不发音
        if word.endswith('e') and count > 1:
            count -= 1
        
        return max(1, count)


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    scorer = QualityScorer()
    
    test_items = [
        {
            "title": "比特币突破10万美元大关 - 彭博社",
            "snippet": "比特币今日突破10万美元，创历史新高。分析师认为机构投资者入场是主因。",
            "url": "https://www.reuters.com/technology/bitcoin-100k-2026",
            "source_engine": "google",
            "publish_time": 1746800000,
            "thumbnail": "https://example.com/thumb.jpg"
        },
        {
            "title": "测试",
            "snippet": "",
            "url": "https://example.com/test",
            "source_engine": "unknown",
        }
    ]
    
    query = {"processed_text": "比特币价格", "original_text": "比特币价格"}
    
    for item in test_items:
        result = scorer.evaluate(item, query)
        print(f"\n标题: {item['title']}")
        print(f"综合得分: {result.total:.1f} ({result.grade})")
        print(f"各维度: {result.dimensions}")
        if result.warnings:
            print(f"警告: {result.warnings}")