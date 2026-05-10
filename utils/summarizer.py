"""摘要生成器 - TextRank抽取式摘要"""
import re
from typing import List, Tuple
from collections import defaultdict

class TextRankSummarizer:
    """基于TextRank的抽取式摘要生成"""
    
    def __init__(self, damping: float = 0.85, max_iter: int = 100):
        self.damping = damping
        self.max_iter = max_iter
    
    def split_sentences(self, text: str) -> List[str]:
        """分句"""
        return re.split(r'[。！？\n]+', text)
    
    def compute_similarity(self, s1: str, s2: str) -> float:
        """句子相似度（基于词重叠）"""
        words1 = set(s1)
        words2 = set(s2)
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)
    
    def build_graph(self, sentences: List[str]) -> List[List[float]]:
        """构建相似度图"""
        n = len(sentences)
        graph = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.compute_similarity(sentences[i], sentences[j])
                graph[i][j] = graph[j][i] = sim
        return graph
    
    def textrank(self, graph: List[List[float]], max_sentences: int = 3) -> List[int]:
        """TextRank算法"""
        n = len(graph)
        if n == 0:
            return []
        scores = [1.0 / n] * n
        for _ in range(self.max_iter):
            new_scores = [(1 - self.damping) / n + self.damping * sum(graph[i][j] * scores[j] for j in range(n)) / sum(graph[i]) for i in range(n)] if sum(graph[i]) > 0 else [1.0 / n] * n
            if all(abs(new_scores[i] - scores[i]) < 0.0001 for i in range(n)):
                break
            scores = new_scores
        indexed_scores = list(enumerate(scores))
        # 位置加权
        for i, score in indexed_scores[:2]:
            scores[i] *= 1.5
        return [i for i, _ in sorted(indexed_scores, key=lambda x: x[1], reverse=True)[:max_sentences]]
    
    def summarize(self, text: str, max_length: int = 300, max_sentences: int = 3) -> str:
        """生成摘要"""
        sentences = [s.strip() for s in self.split_sentences(text) if s.strip()]
        if len(sentences) <= max_sentences:
            return text[:max_length]
        graph = self.build_graph(sentences)
        top_indices = self.textrank(graph, max_sentences)
        summary = '。'.join(sentences[i] for i in sorted(top_indices))
        return summary[:max_length] + '...' if len(summary) > max_length else summary

def summarize(text: str, max_length: int = 300) -> str:
    """便捷函数"""
    summarizer = TextRankSummarizer()
    return summarizer.summarize(text, max_length)

if __name__ == "__main__":
    test = "这是一个测试文本。今天的天气很好。阳光明媚。温度适宜。适合外出活动。这是一个测试文本。今天的天气很好。阳光明媚。温度适宜。适合外出活动。"
    print(summarize(test, 100))