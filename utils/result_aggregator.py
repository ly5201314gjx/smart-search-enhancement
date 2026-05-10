"""
📐 算法二：多引擎结果聚合器 (Multi-Engine Result Aggregator)
=============================================================

核心功能:
1. SimHash 64位文档指纹去重
2. BM25相关度评分
3. 时效性指数衰减模型
4. 权威性多维度评分
5. MMR多样性重排序

数学模型详见 SKILL.md 算法二
"""

import re
import math
import hashlib
import time
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict


# ============================================================
# SimHash 实现
# ============================================================

class SimHash:
    """
    SimHash 64位文档指纹算法
    
    用于大规模文档去重，支持汉明距离快速计算
    """
    
    def __init__(self, hash_bits: int = 64):
        self.hash_bits = hash_bits
        self.fingerprint_cache = {}  # LRU缓存
        
    def compute(self, text: str) -> int:
        """
        计算文本的64位SimHash指纹
        
        Algorithm:
        1. 分词并计算TF-IDF权重
        2. 每个词计算MD5 → 64bit hash
        3. 加权累加得到64维向量V
        4. 对V中每个分量: >0 → 1, ≤0 → 0
        """
        if text in self.fingerprint_cache:
            return self.fingerprint_cache[text]
        
        # Step 1: 分词
        tokens = self._tokenize(text)
        if not tokens:
            return 0
        
        # Step 2: 计算词频权重（简化TF）
        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        max_tf = max(tf.values()) if tf else 1
        
        # Step 3: 初始化64维向量V
        V = [0] * self.hash_bits
        
        # Step 4: 对每个词加权累加
        for token, freq in tf.items():
            weight = freq / max_tf  # 归一化词频权重
            token_hash = self._md5_to_bits(token)
            
            for i in range(self.hash_bits):
                if token_hash[i]:
                    V[i] += weight
                else:
                    V[i] -= weight
        
        # Step 5: 生成最终指纹
        fingerprint = 0
        for i in range(self.hash_bits):
            if V[i] > 0:
                fingerprint |= (1 << i)
        
        self.fingerprint_cache[text] = fingerprint
        return fingerprint
    
    def hamming_distance(self, f1: int, f2: int) -> int:
        """计算两个指纹的汉明距离"""
        xor = f1 ^ f2
        # 计算二进制中1的个数（popcount）
        # 使用 Brian Kernighan 算法
        distance = 0
        while xor:
            xor &= (xor - 1)
            distance += 1
        return distance
    
    def similarity(self, f1: int, f2: int) -> float:
        """
        计算相似度
        
        sim = 1 - hamming_distance / bits
        
        Returns: [0, 1], 1表示完全相同
        """
        distance = self.hamming_distance(f1, f2)
        return 1.0 - (distance / self.hash_bits)
    
    def is_duplicate(self, f1: int, f2: int, threshold: float = 0.85) -> bool:
        """判断是否重复（相似度 > threshold）"""
        return self.similarity(f1, f2) > threshold
    
    def _tokenize(self, text: str) -> List[str]:
        """分词，保留中英文和数字"""
        # 中文按字划分
        tokens = []
        i = 0
        while i < len(text):
            char = text[i]
            if '\u4e00' <= char <= '\u9fff':
                tokens.append(char)
            elif char.isalnum():
                # 提取连续的字母数字
                j = i
                while j < len(text) and text[j].isalnum():
                    j += 1
                tokens.append(text[i:j].lower())
                i = j
                continue
            i += 1
        
        return [t for t in tokens if len(t) >= 1]
    
    def _md5_to_bits(self, text: str) -> List[int]:
        """MD5哈希 → 64位二进制列表"""
        md5_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        # 取前16个hex字符 → 64 bits
        bits = []
        for char in md5_hash[:16]:
            val = int(char, 16)
            for j in range(4):
                bits.append((val >> (3 - j)) & 1)
        return bits
    
    def clear_cache(self):
        """清空缓存"""
        self.fingerprint_cache.clear()


# ============================================================
# BM25 相关度评分
# ============================================================

class BM25Scorer:
    """
    BM25 文本相关度评分
    
    BM25(Q, E) = ∑(IDF(qᵢ) · f(qᵢ, E) · (k₁+1) / (f(qᵢ, E) + k₁·(1-b + b·|E|/avgdl)))
    """
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.avgdl = 100.0  # 平均文档长度（预估）
        self.N = 1000000    # 文档总数（预估值）
        self.df = {}        # 文档频率
        self.corpus_size = 0
    
    def fit(self, corpus: List[str]):
        """从语料库学习IDF"""
        self.corpus_size = len(corpus)
        tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        
        # 计算文档频率
        for doc in tokenized_corpus:
            unique_terms = set(doc)
            for term in unique_terms:
                self.df[term] = self.df.get(term, 0) + 1
        
        # 计算平均文档长度
        doc_lengths = [len(doc) for doc in tokenized_corpus]
        self.avgdl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 100.0
    
    def score(self, query: str, document: str) -> float:
        """
        计算 BM25 得分
        """
        query_tokens = self._tokenize(query)
        doc_tokens = self._tokenize(document)
        doc_len = len(doc_tokens)
        
        if doc_len == 0 or not query_tokens:
            return 0.0
        
        # 计算词频
        tf = defaultdict(int)
        for token in doc_tokens:
            tf[token] += 1
        
        score = 0.0
        for q_term in set(query_tokens):
            if q_term not in tf:
                continue
            
            # IDF
            df_q = self.df.get(q_term, 1)
            idf = math.log((self.N - df_q + 0.5) / (df_q + 0.5) + 1)
            
            # TF
            f = tf[q_term]
            
            # BM25公式
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            
            score += idf * numerator / denominator
        
        return score
    
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


# ============================================================
# 时效性评分 (时间衰减模型)
# ============================================================

class RecencyScorer:
    """
    时效性评分 - 指数衰减模型
    
    S_recency(t) = exp(-λ · Δt)
    
    半衰期自适应调整:
    - RealTime: T_½ = 2h
    - News: T_½ = 12h
    - General: T_½ = 72h
    - Academic: T_½ = 8760h (1年)
    """
    
    HALF_LIVES = {
        "RealTime": 2,
        "RealTime_data": 0.5,
        "News": 12,
        "General": 72,
        "Knowledge": 720,
        "Academic": 8760,
        "Navigation": 87600,
    }
    
    def score(self, publish_time: Optional[float], 
              query_type: str = "General") -> Tuple[float, float]:
        """
        计算时效性得分
        
        Returns:
            (score, lambda_value)
        """
        if publish_time is None:
            return 0.5, 0.0  # 无时间戳时默认中等分数
        
        # 计算时间差（小时）
        now = time.time()
        delta_hours = (now - publish_time) / 3600
        
        if delta_hours < 0:
            return 0.0, 0.0  # 未来时间 → 0分
        
        # 选择半衰期
        half_life = self.HALF_LIVES.get(query_type, 72)
        lambda_val = math.log(2) / half_life
        
        # 指数衰减
        score = math.exp(-lambda_val * delta_hours)
        
        return max(0.0, min(1.0, score)), lambda_val


# ============================================================
# 权威性评分
# ============================================================

class AuthorityScorer:
    """
    权威性评分 - 三维度模型
    
    S_authority = 0.4·D_domain + 0.4·D_trusted + 0.2·D_quality
    """
    
    # 域名类型得分
    DOMAIN_SCORES = {
        ".gov": 0.95,
        ".edu": 0.90,
        ".org": 0.80,
        ".ac.cn": 0.85,
        ".edu.cn": 0.85,
        ".mil": 0.90,
        ".int": 0.85,
        ".com": 0.60,
        ".net": 0.60,
        ".co": 0.55,
        ".io": 0.50,
        ".info": 0.40,
        ".biz": 0.40,
        ".xyz": 0.30,
        ".top": 0.25,
    }
    
    # 可信域名白名单
    TRUSTED_DOMAINS = {
        "wikipedia.org": 0.95,
        "reuters.com": 0.95,
        "ap.org": 0.95,
        "bbc.com": 0.90,
        "bbc.co.uk": 0.90,
        "nature.com": 0.95,
        "sciencedirect.com": 0.90,
        "ieee.org": 0.90,
        "acm.org": 0.90,
        "github.com": 0.85,
        "stackoverflow.com": 0.85,
        "medium.com": 0.70,
        "zhihu.com": 0.75,
        "baike.baidu.com": 0.80,
        "csdn.net": 0.60,
        "cnblogs.com": 0.60,
        "jianshu.com": 0.55,
        "36kr.com": 0.60,
        "huxiu.com": 0.65,
        "ssrn.com": 0.80,
        "arxiv.org": 0.90,
        "scholar.google.com": 0.90,
    }
    
    def score(self, url: str) -> Dict[str, float]:
        """
        计算权威性评分
        
        Returns: {
            "total": 总分,
            "domain_type": 域名类型分,
            "trusted": 可信域名分,
            "quality": 质量信号分
        }
        """
        domain = self._extract_domain(url)
        
        # D_domain: 域名类型得分
        d_domain = 0.30  # 默认
        for suffix, score in self.DOMAIN_SCORES.items():
            if domain.endswith(suffix):
                d_domain = score
                break
        
        # D_trusted: 可信域名得分
        d_trusted = 0.40  # 默认
        for trusted_domain, score in self.TRUSTED_DOMAINS.items():
            if trusted_domain in domain:
                d_trusted = score
                break
        
        # D_quality: 质量信号（基于URL结构）
        d_quality = 0.50  # 默认
        # HTTPS加分
        if url.startswith("https://"):
            d_quality += 0.15
        # 路径深度适中加分
        path_depth = len(url.split("/")) - 3
        if 2 <= path_depth <= 5:
            d_quality += 0.15
        # 不含奇怪参数加分
        if "?" not in url.split("/")[-1]:
            d_quality += 0.10
        # 过长路径扣分
        if path_depth > 8:
            d_quality -= 0.10
        
        d_quality = max(0.0, min(1.0, d_quality))
        
        # 总分
        total = 0.4 * d_domain + 0.4 * d_trusted + 0.2 * d_quality
        
        return {
            "total": total,
            "domain_type": d_domain,
            "trusted": d_trusted,
            "quality": d_quality
        }
    
    def _extract_domain(self, url: str) -> str:
        """从URL提取域名"""
        match = re.search(r'https?://([^/]+)', url)
        if match:
            return match.group(1).lower()
        return url.lower()


# ============================================================
# 主聚合器
# ============================================================

class ResultAggregator:
    """
    结果聚合器 - 整合去重、评分、排序、MMR
    
    完整处理流程:
    1. normalize: 统一各引擎返回的原始结果格式
    2. simhash_dedup: SimHash指纹去重
    3. weighted_ranking: 加权融合排序
    4. mmr_rerank: MMR多样性重排序
    """
    
    def __init__(self):
        self.simhash = SimHash()
        self.bm25 = BM25Scorer()
        self.recency = RecencyScorer()
        self.authority = AuthorityScorer()
    
    def normalize(self, raw_results: List[Dict]) -> List[Dict]:
        """
        归一化: 将各引擎返回的不同格式统一化
        """
        normalized = []
        for item in raw_results:
            normalized.append({
                "id": item.get("id", ""),
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
                "source_engine": item.get("source_engine", "unknown"),
                "language": item.get("language", "unknown"),
                "publish_time": item.get("publish_time"),
                "domain": self.authority._extract_domain(
                    item.get("url", "")
                ),
                "thumbnail": item.get("thumbnail", ""),
                "content": item.get("content", item.get("snippet", "")),
                "original": item
            })
        return normalized
    
    def simhash_dedup(self, items: List[Dict], 
                      threshold: float = 0.85) -> List[Dict]:
        """
        SimHash指纹去重
        
        Args:
            items: 归一化后的结果列表
            threshold: 相似度阈值，>threshold判定为重复
        
        Returns:
            去重后的结果列表（保留第一个出现的）
        """
        if not items:
            return []
        
        unique_items = []
        seen_fingerprints = []  # 存放指纹和对应结果
        
        for item in items:
            # 计算标题+摘要的指纹
            content = item.get("title", "") + " " + item.get("snippet", "")
            fp = self.simhash.compute(content)
            
            # 检查是否与已有结果重复
            is_duplicate = False
            for existing_fp, _ in seen_fingerprints:
                if self.simhash.is_duplicate(fp, existing_fp, threshold):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_items.append(item)
                seen_fingerprints.append((fp, item))
        
        return unique_items
    
    def weighted_ranking(self, items: List[Dict], 
                          query: Dict,
                          weights: Dict) -> List[Dict]:
        """
        加权融合排序
        
        Score = α·S_engine + β·S_relevance + γ·S_recency + δ·S_authority + ε·S_diversity
        
        Args:
            items: 去重后的结果列表
            weights: 权重配置
        """
        query_text = query.get("processed_text", "")
        query_type = query.get("intent", "General")
        
        # BM25相关度
        corpus = [item.get("title", "") + " " + item.get("snippet", "") 
                 for item in items]
        if corpus:
            self.bm25.fit(corpus)
        
        for item in items:
            content = item.get("title", "") + " " + item.get("snippet", "")
            
            # 引擎可信度
            engine = item.get("source_engine", "unknown")
            s_engine = self._engine_score(engine)
            
            # 相关度 (BM25)
            s_relevance = self.bm25.score(query_text, content)
            s_relevance = max(0.0, min(1.0, s_relevance * 2))
            
            # 时效性
            s_recency, _ = self.recency.score(
                item.get("publish_time"), 
                query_type
            )
            
            # 权威性
            auth_result = self.authority.score(item.get("url", ""))
            s_authority = auth_result["total"]
            
            # 多样性（初始设为1.0，后续MMR调整）
            s_diversity = 1.0
            
            # 加权综合得分
            weights_map = {
                "s_engine": weights.get("engine_reliability", 0.15),
                "s_relevance": 0.35,
                "s_recency": 0.20,
                "s_authority": 0.20,
                "s_diversity": 0.10,
            }
            
            final_score = (
                weights_map["s_engine"] * s_engine +
                weights_map["s_relevance"] * s_relevance +
                weights_map["s_recency"] * s_recency +
                weights_map["s_authority"] * s_authority +
                weights_map["s_diversity"] * s_diversity
            )
            
            item.update({
                "engine_score": s_engine,
                "relevance_score": s_relevance,
                "recency_score": s_recency,
                "authority_score": s_authority,
                "diversity_score": s_diversity,
                "final_score": final_score,
                "quality_score": final_score * 100
            })
        
        # 按最终得分排序
        items.sort(key=lambda x: x["final_score"], reverse=True)
        return items
    
    def mmr_rerank(self, items: List[Dict], 
                    query: Dict,
                    lambda_val: float = 0.5,
                    top_k: int = 20) -> List[Dict]:
        """
        MMR (Maximum Marginal Relevance) 多样性重排序
        
        MMR = λ·sim(Eᵢ, Q) - (1-λ)·maxⱼ_∈_selected sim(Eᵢ, Eⱼ)
        
        Args:
            items: 已排序的结果列表
            lambda_val: 精度-多样性平衡参数 [0.3, 0.7]
            top_k: 只对Top-K应用MMR
        """
        if not items or top_k <= 1:
            return items
        
        # 只对Top-K应用MMR
        candidates = items[:top_k]
        remaining = items[top_k:]
        
        if len(candidates) <= 1:
            return items
        
        # 计算所有指纹
        fingerprints = {}
        for item in candidates:
            content = item.get("title", "") + " " + item.get("snippet", "")
            fp = self.simhash.compute(content)
            fingerprints[id(item)] = fp
        
        # MMR选择
        selected = [candidates[0]]  # 先取最高分
        candidates = candidates[1:]
        
        # 预计算所有指纹
        query_content = query.get("processed_text", "")
        query_fp = self.simhash.compute(query_content)
        
        while candidates and len(selected) < top_k:
            best_item = None
            best_mmr = float('-inf')
            
            for item in candidates:
                item_fp = fingerprints[id(item)]
                
                # sim(Eᵢ, Q)
                sim_with_query = self.simhash.similarity(item_fp, query_fp)
                
                # max sim(Eᵢ, Eⱼ) for Eⱼ in selected
                max_sim_selected = 0.0
                for sel in selected:
                    sel_fp = fingerprints[id(sel)]
                    sim = self.simhash.similarity(item_fp, sel_fp)
                    max_sim_selected = max(max_sim_selected, sim)
                
                # MMR评分
                mmr_score = lambda_val * sim_with_query - \
                           (1 - lambda_val) * max_sim_selected
                
                # 使用原始分数作为偏置（优先高分结果）
                boosted_mmr = item.get("final_score", 0) + mmr_score
                
                if boosted_mmr > best_mmr:
                    best_mmr = boosted_mmr
                    best_item = item
            
            if best_item:
                selected.append(best_item)
                candidates.remove(best_item)
        
        # 更新多样性分数
        for item in selected:
            item_fp = fingerprints[id(item)]
            max_sim = 0.0
            for other in selected:
                if other is item:
                    continue
                other_fp = fingerprints[id(other)]
                sim = self.simhash.similarity(item_fp, other_fp)
                max_sim = max(max_sim, sim)
            
            item["diversity_score"] = 1.0 - max_sim
            # 重新计算最终得分（含多样性）
            current_score = item.get("final_score", 0)
            item["final_score"] = 0.9 * current_score + 0.1 * (1.0 - max_sim)
        
        selected.sort(key=lambda x: x["final_score"], reverse=True)
        
        return selected + remaining
    
    def _engine_score(self, engine: str) -> float:
        """引擎可信度评分"""
        scores = {
            "google": 0.95,
            "bing": 0.90,
            "tavily": 0.88,
            "duckduckgo": 0.85,
            "baidu": 0.75,
            "sogou": 0.70,
            "quark": 0.65,
        }
        return scores.get(engine.lower(), 0.50)


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    # SimHash测试
    simhash = SimHash()
    text1 = "比特币今日价格突破10万美元"
    text2 = "比特币今日价格突破10万美金"
    text3 = "以太坊价格今日下跌"
    
    fp1 = simhash.compute(text1)
    fp2 = simhash.compute(text2)
    fp3 = simhash.compute(text3)
    
    print(f"SimHash测试:")
    print(f"  '{text1}' vs '{text2}': sim={simhash.similarity(fp1, fp2):.3f}")
    print(f"  '{text1}' vs '{text3}': sim={simhash.similarity(fp1, fp3):.3f}")
    
    # 权威性测试
    auth = AuthorityScorer()
    urls = [
        "https://www.reuters.com/technology/bitcoin-price",
        "https://en.wikipedia.org/wiki/Bitcoin",
        "https://www.bbc.com/news/technology",
        "https://some-random-blog.xyz/crypto",
    ]
    
    print(f"\n权威性评分:")
    for url in urls:
        result = auth.score(url)
        print(f"  {url}: {result['total']:.2f}")