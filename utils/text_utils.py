"""文本处理工具函数"""
import re
from typing import List, Tuple

def normalize_text(text: str) -> str:
    """文本规范化"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:。，！？；：]', '', text)
    return text.strip()

def split_words(text: str, lang: str = "auto") -> List[str]:
    """分词"""
    text = normalize_text(text)
    if lang == "zh":
        return list(text)
    return text.split()

def extract_keywords(text: str, top_k: int = 10) -> List[Tuple[str, float]]:
    """提取关键词（简易版）"""
    words = split_words(text)
    freq = {}
    for w in words:
        if len(w) >= 2:
            freq[w] = freq.get(w, 0) + 1
    total = sum(freq.values()) or 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [(w, round(c / total, 4)) for w, c in sorted_words[:top_k]]

def remove_html(text: str) -> str:
    """移除HTML标签"""
    return re.sub(r'<[^>]+>', '', text)

def truncate(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """截断文本"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def highlight_keywords(text: str, keywords: List[str], mark: str = "**") -> str:
    """高亮关键词"""
    for kw in keywords:
        text = re.sub(f'({re.escape(kw)})', f'{mark}\\1{mark}', text, flags=re.IGNORECASE)
    return text

if __name__ == "__main__":
    text = "Python是一门<strong>编程语言</strong>，广泛用于数据分析、机器学习等领域。"
    print(normalize_text(text))
    print(remove_html(text))
    print(truncate("这是一段很长的文本内容...", 10))