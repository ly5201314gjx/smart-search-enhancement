"""
📐 算法三：TextRank 摘要生成器 (Smart Summarizer)
====================================================

基于TextRank的抽取式摘要算法:

1. 句子图构建 (节点=句子, 边=语义相似度)
2. 迭代计算PR值
3. 位置加权
4. 冗余去除
5. 长度压缩
"""

import re
import math
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict


class TextRankSummarizer:
    """
    TextRank 抽取式摘要
    
    核心技术:
    - 句子相似度: TF-IDF + 词向量混合
    - PageRank迭代: damping_factor=0.85
    - 位置偏置: 前2句权重1.5x
    - 冗余去除: 相似度阈值0.75
    """
    
    def __init__(self, 
                 damping_factor: float = 0.85,
                 max_iter: int = 100,
                 convergence: float = 0.0001,
                 redundancy_threshold: float = 0.75):
        self.d = damping_factor
        self.max_iter = max_iter
        self.convergence = convergence
        self.redundancy_threshold = redundancy_threshold
    
    def generate(self, text: str, max_len: int = 300) -> str:
        """
        生成摘要
        
        Args:
            text: 原文
            max_len: 最大摘要长度（字符数）
        
        Returns:
            摘要文本
        """
        if not text or len(text) < max_len:
            return text
        
        # Step 1: 分句
        sentences = self._split_sentences(text)
        if len(sentences) <= 3:
            return text[:max_len]
        
        # Step 2: 句子图构建 + TextRank迭代
        ranked_sentences = self._textrank(sentences)
        
        # Step 3: 选择摘要句子
        selected = self._select_sentences(ranked_sentences, sentences, max_len)
        
        # Step 4: 排序恢复原文顺序
        selected.sort(key=lambda x: x[1])  # 按原位置排序
        
        summary = " ".join(s[0] for s in selected)
        return summary[:max_len]
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        分句（支持中英文）
        
        - 中文: 。！？；分隔
        - 英文: .!?; 分隔
        """
        # 处理中文标点
        text = re.sub(r'([。！？；])', r'\1\n', text)
        # 处理英文标点（考虑缩写）
        text = re.sub(r'([.!?;])\s+', r'\1\n', text)
        
        sentences = [s.strip() for s in text.split('\n') if s.strip()]
        
        # 过滤过短句子
        sentences = [s for s in sentences if len(s) >= 5]
        
        return sentences if sentences else [text]
    
    def _textrank(self, sentences: List[str]) -> List[Tuple[str, float, int]]:
        """
        TextRank 迭代计算句子权重
        
        PR(vᵢ) = (1-d) + d · Σ(wⱼᵢ / Σwⱼₖ · PR(vⱼ))
        """
        n = len(sentences)
        if n <= 3:
            return [(s, 1.0, i) for i, s in enumerate(sentences)]
        
        # Step 1: 计算相似度矩阵
        sim_matrix = self._build_similarity_matrix(sentences)
        
        # Step 2: 初始化PR值
        pr = [1.0 / n] * n
        
        # Step 3: 迭代计算
        for iteration in range(self.max_iter):
            new_pr = []
            for i in range(n):
                # 计算从其他节点传入的权重
                incoming = 0.0
                for j in range(n):
                    if i != j and sim_matrix[j][i] > 0:
                        outgoing_sum = sum(
                            sim_matrix[j][k] 
                            for k in range(n) if k != j
                        )
                        if outgoing_sum > 0:
                            incoming += (sim_matrix[j][i] / outgoing_sum) * pr[j]
                
                new_pr_i = (1 - self.d) + self.d * incoming
                new_pr.append(new_pr_i)
            
            # 检查收敛
            diff = sum(abs(new_pr[i] - pr[i]) for i in range(n))
            pr = new_pr
            
            if diff < self.convergence:
                break
        
        # Step 4: 位置加权
        for i in range(n):
            if i <= 1:  # 前2句（标题/导语）
                pr[i] *= 1.5
            elif i <= 4:  # 第3-5句
                pr[i] *= 1.2
            elif i > 10:  # 10句以后打折
                pr[i] *= 0.8
        
        # 返回 (句子, PR值, 位置)
        result = [(sentences[i], pr[i], i) for i in range(n)]
        return result
    
    def _build_similarity_matrix(self, sentences: List[str]) -> List[List[float]]:
        """
        构建句子相似度矩阵
        
        基于TF向量和词重叠的混合相似度
        """
        n = len(sentences)
        matrix = [[0.0] * n for _ in range(n)]
        
        # 计算每个句子的TF向量
        tf_vectors = []
        for sent in sentences:
            tf_vectors.append(self._tf_vector(sent))
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._sentence_similarity(
                    sentences[i], sentences[j],
                    tf_vectors[i], tf_vectors[j]
                )
                matrix[i][j] = sim
                matrix[j][i] = sim
        
        return matrix
    
    def _tf_vector(self, text: str) -> Dict[str, float]:
        """计算TF向量"""
        tokens = self._tokenize(text)
        if not tokens:
            return {}
        
        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        
        max_tf = max(tf.values())
        return {k: v / max_tf for k, v in tf.items()}
    
    def _sentence_similarity(self, s1: str, s2: str,
                              tf1: Dict[str, float],
                              tf2: Dict[str, float]) -> float:
        """
        句子相似度计算
        
        混合: 词重叠比例 + TF向量余弦
        """
        tokens1 = set(self._tokenize(s1))
        tokens2 = set(self._tokenize(s2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Jaccard相似度
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        jaccard = len(intersection) / len(union) if union else 0
        
        # TF向量余弦相似度
        all_tokens = tokens1 | tokens2
        vec1 = [tf1.get(t, 0) for t in all_tokens]
        vec2 = [tf2.get(t, 0) for t in all_tokens]
        
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        cosine = dot / (norm1 * norm2) if norm1 * norm2 > 0 else 0
        
        # 混合: 0.4 Jaccard + 0.6 余弦
        sim = 0.4 * jaccard + 0.6 * cosine
        return max(0, min(sim, 1.0))
    
    def _select_sentences(self, 
                          ranked: List[Tuple[str, float, int]],
                          original_sentences: List[str],
                          max_len: int) -> List[Tuple[str, int]]:
        """
        选择摘要句子 + 冗余去除
        
        策略:
        1. 按PR值降序排列
        2. 逐个选取，跳过与已选句子相似度>阈值的
        3. 直到达到长度限制
        """
        # 按PR值降序排列
        ranked_sorted = sorted(ranked, key=lambda x: x[1], reverse=True)
        
        selected_sentences = []  # [(text, original_index)]
        selected_vectors = []    # TF向量列表
        
        for sent, pr, idx in ranked_sorted:
            # 长度检查
            current_len = sum(len(s[0]) for s in selected_sentences)
            if current_len + len(sent) > max_len and selected_sentences:
                break
            
            # 冗余检查
            sent_vec = self._tf_vector(sent)
            is_redundant = False
            for existing_vec in selected_vectors:
                sim = self._vector_similarity(sent_vec, existing_vec)
                if sim > self.redundancy_threshold:
                    is_redundant = True
                    break
            
            if not is_redundant:
                selected_sentences.append((sent, idx))
                selected_vectors.append(sent_vec)
        
        # 如果没选到任何句子，选最高分的一个
        if not selected_sentences and ranked_sorted:
            selected_sentences = [(ranked_sorted[0][0], ranked_sorted[0][2])]
        
        return selected_sentences
    
    def _vector_similarity(self, v1: Dict[str, float], 
                           v2: Dict[str, float]) -> float:
        """两个TF向量的余弦相似度"""
        all_keys = set(v1.keys()) | set(v2.keys())
        if not all_keys:
            return 0.0
        
        vec1 = [v1.get(k, 0) for k in all_keys]
        vec2 = [v2.get(k, 0) for k in all_keys]
        
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        return dot / (norm1 * norm2) if norm1 * norm2 > 0 else 0
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
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
                if j - i >= 2:  # 只保留长度≥2的英文词
                    tokens.append(text[i:j].lower())
                i = j
                continue
            i += 1
        return tokens


class SmartSummarizer(TextRankSummarizer):
    """
    智能摘要生成器（增强版）
    
    支持:
    - 自适应长度
    - 意图感知
    - 多语言
    """
    
    def generate(self, 
                 item: Dict, 
                 max_len: int = 300,
                 intent: str = "General") -> str:
        """
        生成搜索结果摘要
        
        Args:
            item: 搜索结果条目
            max_len: 最大长度
            intent: 搜索意图
        
        Returns:
            摘要文本
        """
        # 优先使用snippet
        snippet = item.get("snippet", "")
        if len(snippet) >= max_len * 0.5:
            return snippet[:max_len]
        
        # 组合标题+摘要
        title = item.get("title", "")
        content = item.get("content", snippet)
        
        text = f"{title}. {content}"
        
        # 调用父类TextRank摘要
        return super().generate(text, max_len)


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    # 长文本测试
    test_text = """
    今天比特币价格突破10万美元大关，创下历史新高。这是比特币自2009年诞生以来的最高价格。
    分析师指出，此次上涨主要受到机构投资者大规模入场的影响。多家大型金融机构近期宣布将比特币纳入资产配置。
    此外，全球宏观经济不确定性也推动了投资者对比特币等数字资产的需求。市场数据显示，比特币在过去24小时内涨幅超过8%。
    技术面上，比特币突破了关键阻力位，下一个目标价位在12万美元。然而，也有分析师警告市场可能存在过热风险。
    投资者应该注意风险管理，合理配置资产。加密货币市场波动性较大，不适合所有投资者。
    """
    
    summarizer = TextRankSummarizer()
    summary = summarizer.generate(test_text, max_len=100)
    
    print("原文:")
    print(test_text[:200] + "...")
    print("\n摘要:")
    print(summary)