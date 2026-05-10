"""
文本处理工具函数库

提供分词、向量化、相似度计算等基础功能
"""

import re
import math
import hashlib
from typing import Dict, List, Set, Optional, Tuple
from collections import Counter, defaultdict


class TextUtils:
    """文本处理工具集"""
    
    @staticmethod
    def tokenize(text: str, min_len: int = 1) -> List[str]:
        """
        通用分词（支持中英文混合）
        
        - 中文: 按字符分词（可升级为jieba）
        - 英文: 按空格分词
        - 数字: 独立保留
        """
        tokens = []
        i = 0
        while i < len(text):
            char = text[i]
            
            # 中文/日文/韩文字符
            if '\u4e00' <= char <= '\u9fff' or \
               '\u3040' <= char <= '\u30ff' or \
               '\uac00' <= char <= '\ud7af':
                tokens.append(char)
            
            # 英文/数字
            elif char.isalnum():
                j = i
                while j < len(text) and (text[j].isalnum() or text[j] in "._-"):
                    j += 1
                word = text[i:j].lower()
                if len(word) >= min_len:
                    tokens.append(word)
                i = j
                continue
            
            i += 1
        
        return tokens
    
    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """分句"""
        # 中文标点
        text = re.sub(r'([。！？；\n])', r'\1||', text)
        # 英文标点
        text = re.sub(r'([.!?;])\s+', r'\1||', text)
        
        sentences = [s.strip() for s in text.split('||') if s.strip()]
        return [s for s in sentences if len(s) >= 5]
    
    @staticmethod
    def extract_keywords(text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        提取关键词（基于TF统计）
        
        简化版: 统计词频并返回到高频词
        """
        tokens = TextUtils.tokenize(text)
        if not tokens:
            return []
        
        # 过滤停用词
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
                     '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
                     '你', '会', '着', '没有', '看', '好', '自己', '这',
                     'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
                     'been', 'being', 'have', 'has', 'had', 'do', 'does',
                     'did', 'will', 'would', 'could', 'should', 'may',
                     'might', 'shall', 'can', 'need', 'to', 'of', 'in',
                     'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                     'through', 'during', 'before', 'after', 'above', 'below'}
        
        filtered = [t for t in tokens if t not in stop_words and len(t) >= 2]
        
        if not filtered:
            return [(t, 1.0) for t in tokens[:top_k]]
        
        counter = Counter(filtered)
        total = sum(counter.values())
        
        return [(word, count/total) for word, count in 
                counter.most_common(top_k)]
    
    @staticmethod
    def cosine_similarity(vec1: Dict[str, float], 
                          vec2: Dict[str, float]) -> float:
        """计算两个TF向量的余弦相似度"""
        all_keys = set(vec1.keys()) | set(vec2.keys())
        if not all_keys:
            return 0.0
        
        v1 = [vec1.get(k, 0) for k in all_keys]
        v2 = [vec2.get(k, 0) for k in all_keys]
        
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        
        if norm1 * norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    @staticmethod
    def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
        """Jaccard相似度"""
        intersection = set1 & set2
        union = set1 | set2
        return len(intersection) / len(union) if union else 0.0
    
    @staticmethod
    def text_hash(text: str, algorithm: str = "md5") -> str:
        """文本哈希"""
        h = hashlib.new(algorithm)
        h.update(text.encode('utf-8'))
        return h.hexdigest()
    
    @staticmethod
    def detect_language(text: str) -> str:
        """语言检测"""
        if not text:
            return "unknown"
        
        chinese = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        japanese = sum(1 for c in text if '\u3040' <= c <= '\u30ff')
        korean = sum(1 for c in text if '\uac00' <= c <= '\ud7af')
        total = len(text.replace(' ', ''))
        
        if total == 0:
            return "unknown"
        
        ratios = {
            "zh": chinese / total,
            "ja": japanese / total,
            "ko": korean / total,
        }
        
        if chinese > 0 and japanese > 0 and korean == 0:
            return "ja" if japanese/total > 0.2 else "zh"
        
        best = max(ratios, key=ratios.get)
        if ratios[best] > 0.3:
            return best
        
        ascii_ratio = sum(1 for c in text if c.isascii() and c.isalpha()) / len(text)
        return "en" if ascii_ratio > 0.6 else "other"


class EmbeddingCache:
    """
    嵌入向量缓存（简化版）
    
    在生产环境中可以用 sentence-transformers 等模型
    """
    
    def __init__(self, max_cache: int = 1000):
        self.cache: Dict[str, List[float]] = {}
        self.max_cache = max_cache
    
    def get_or_compute(self, text: str, 
                       compute_fn=None) -> List[float]:
        """获取或计算嵌入"""
        key = self._key(text)
        if key in self.cache:
            return self.cache[key]
        
        if compute_fn:
            embedding = compute_fn(text)
        else:
            embedding = self._simple_embed(text)
        
        if len(self.cache) >= self.max_cache:
            # 简单LRU: 删除最早的
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        
        self.cache[key] = embedding
        return embedding
    
    def _key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    def _simple_embed(self, text: str) -> List[float]:
        """
        简化嵌入: 基于字符n-gram的哈希向量
        
        这不是真正的语义嵌入，而是用于快速原型
        """
        tokens = TextUtils.tokenize(text)
        if not tokens:
            return [0.0] * 64
        
        # 基于哈希的简单嵌入
        vec = [0.0] * 64
        for token in tokens:
            h = hashlib.md5(token.encode()).digest()
            for i in range(min(64, len(h))):
                vec[i] += (h[i] / 255.0)
        
        # 归一化
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm > 0 else vec
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()