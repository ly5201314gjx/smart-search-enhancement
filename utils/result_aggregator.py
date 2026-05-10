"""结果聚合器 - SimHash去重+多维度评分"""
import hashlib
from typing import List, Dict, Tuple
from collections import defaultdict

class SimHash:
    """SimHash文档指纹算法"""
    
    @staticmethod
    def hash_token(token: str) -> int:
        """MD5哈希转64位整数"""
        return int(hashlib.md5(token.encode()).hexdigest(), 16) % (2**64)
    
    @staticmethod
    def compute(text: str, hash_bits: int = 64) -> int:
        """计算SimHash指纹"""
        v = [0] * hash_bits
        tokens = text.split()
        for token in tokens:
            h = SimHash.hash_token(token)
            for i in range(hash_bits):
                bit = (h >> i) & 1
                v[i] += 1 if bit else -1
        fingerprint = 0
        for i in range(hash_bits):
            if v[i] > 0:
                fingerprint |= (1 << i)
        return fingerprint
    
    @staticmethod
    def hamming_distance(h1: int, h2: int) -> int:
        """计算汉明距离"""
        xor = h1 ^ h2
        return bin(xor).count('1')
    
    @staticmethod
    def similarity(h1: int, h2: int, bits: int = 64) -> float:
        """计算相似度"""
        return 1 - SimHash.hamming_distance(h1, h2) / bits

class ResultAggregator:
    """搜索结果聚合器"""
    
    ENGINE_WEIGHTS = {"Google": 0.95, "Bing": 0.90, "DuckDuckGo": 0.85, "Baidu": 0.75}
    
    def __init__(self, dedup_threshold: float = 0.85):
        self.dedup_threshold = dedup_threshold
        self.simhash = SimHash()
    
    def compute_score(self, result: Dict) -> float:
        """计算综合得分"""
        engine_weight = self.ENGINE_WEIGHTS.get(result.get("engine", "Other"), 0.5)
        recency = result.get("recency", 0.5)
        relevance = result.get("relevance", 0.5)
        return 0.15 * engine_weight + 0.35 * relevance + 0.25 * recency + 0.25 * result.get("authority", 0.5)
    
    def deduplicate(self, results: List[Dict]) -> List[Dict]:
        """SimHash去重"""
        seen = {}
        unique = []
        for r in results:
            key = r.get("title", "")[:50]
            fingerprint = self.simhash.compute(key)
            is_duplicate = False
            for _, (_, fp) in seen.items():
                if self.simhash.similarity(fingerprint, fp) > self.dedup_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                seen[key] = (r, fingerprint)
                unique.append(r)
        return unique
    
    def aggregate(self, results: List[Dict]) -> List[Dict]:
        """聚合流程"""
        # 去重
        unique = self.deduplicate(results)
        # 评分
        for r in unique:
            r["score"] = self.compute_score(r)
        # 排序
        return sorted(unique, key=lambda x: x.get("score", 0), reverse=True)

if __name__ == "__main__":
    agg = ResultAggregator()
    test_results = [
        {"title": "Python教程第一部分", "engine": "Google", "relevance": 0.9, "recency": 0.8},
        {"title": "Python教程第一部分", "engine": "Baidu", "relevance": 0.8, "recency": 0.7},
        {"title": "JavaScript入门", "engine": "Bing", "relevance": 0.6, "recency": 0.9},
    ]
    print(agg.aggregate(test_results))