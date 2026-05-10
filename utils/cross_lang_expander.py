"""
📐 算法七：跨语言搜索扩展器 (Cross-Language Search Expander)
=============================================================

基于语言检测 + 桥接表 + 翻译映射的多语言搜索扩展。

架构:
1. 语言检测 (Unicode范围 + N-gram)
2. 桥接策略选择 (根据源语言)
3. 翻译生成 (内置词对映射)
4. 并行搜索策略
5. 结果加权融合
"""

import re
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ============================================================
# 内置中英常见词对映射
# ============================================================

ZH_EN_MAP = {
    # 技术
    "人工智能": "artificial intelligence",
    "机器学习": "machine learning",
    "深度学习": "deep learning",
    "自然语言": "natural language",
    "计算机": "computer",
    "算法": "algorithm",
    "数据": "data",
    "网络": "network",
    "安全": "security",
    "软件": "software",
    "硬件": "hardware",
    "编程": "programming",
    "代码": "code",
    "开发": "development",
    "框架": "framework",
    "模型": "model",
    "训练": "training",
    
    # 加密货币
    "比特币": "bitcoin",
    "以太坊": "ethereum",
    "加密货币": "cryptocurrency",
    "区块链": "blockchain",
    "挖矿": "mining",
    "代币": "token",
    "去中心化": "decentralized",
    "智能合约": "smart contract",
    
    # 通用
    "价格": "price",
    "新闻": "news",
    "教程": "tutorial",
    "指南": "guide",
    "最新": "latest",
    "今天": "today",
    "市场": "market",
    "分析": "analysis",
    "趋势": "trend",
    "行情": "market data",
    "报告": "report",
    "研究": "research",
    "论文": "paper",
    "下载": "download",
    "免费": "free",
}

EN_ZH_MAP = {v: k for k, v in ZH_EN_MAP.items()}


class CrossLangExpander:
    """
    跨语言搜索扩展器
    
    支持: 中文→英文, 英文→中文, 日文→英文+中文, 韩文→英文+中文
    """
    
    def __init__(self):
        self.zh_en_map = ZH_EN_MAP
        self.en_zh_map = EN_ZH_MAP
        
        # 桥接策略表
        self.bridge_table = {
            "zh": ["zh", "en"],           # 中文搜索：中英文并行
            "en": ["en", "zh"],           # 英文搜索：英文为主，中文为辅
            "ja": ["ja", "en", "zh"],     # 日文搜索：日英中
            "ko": ["ko", "en", "zh"],     # 韩文搜索：韩英中
            "other": ["en", "zh"]         # 其他语言：英中
        }
        
        # 语言权重（中文查询时）
        self.lang_weight = {
            "zh": 1.0,
            "en": 0.85,
            "ja": 0.70,
            "ko": 0.70,
            "other": 0.50
        }
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        检测文本语言
        
        Returns: (language_code, confidence)
        """
        if not text:
            return "unknown", 0.0
        
        # 计算各语种字符比例
        total_chars = len(text.strip())
        if total_chars == 0:
            return "unknown", 0.0
        
        zh_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        ja_chars = sum(1 for c in text if '\u3040' <= c <= '\u30ff')
        ja_chars += sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        ko_chars = sum(1 for c in text if '\uac00' <= c <= '\ud7af')
        en_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        
        zh_ratio = zh_chars / total_chars
        ja_ratio = ja_chars / total_chars
        ko_ratio = ko_chars / total_chars
        en_ratio = en_chars / total_chars
        
        # 决策逻辑
        if ko_ratio > 0.3:
            return "ko", min(ko_ratio + 0.2, 0.98)
        
        if ja_ratio > 0.3 and zh_ratio > 0.1:
            # 有汉字+假名 → 日文
            return "ja", min(ja_ratio + 0.1, 0.95)
        
        if zh_ratio > 0.4:
            return "zh", min(zh_ratio + 0.2, 0.98)
        
        if en_ratio > 0.6:
            return "en", min(en_ratio + 0.1, 0.95)
        
        # 混合语言
        if zh_ratio > 0.2 and en_ratio > 0.2:
            return "zh", 0.6  # 中英混合，偏向中文
        
        return "other", 0.4
    
    def expand(self, text: str, source_lang: Optional[str] = None) -> Dict[str, str]:
        """
        扩展为多语言查询
        
        Args:
            text: 原始查询文本
            source_lang: 源语言（可选，不传则自动检测）
        
        Returns:
            {language_code: translated_query, ...}
        """
        if source_lang is None:
            lang_result = self.detect_language(text)
            source_lang = lang_result[0]
        
        if source_lang == "unknown":
            return {}
        
        bridge_langs = self.bridge_table.get(source_lang, ["en", "zh"])
        result = {}
        
        for target_lang in bridge_langs:
            if target_lang == source_lang:
                result[target_lang] = text
            else:
                translated = self._translate(text, source_lang, target_lang)
                if translated and translated != text:
                    result[target_lang] = translated
        
        return result
    
    def _translate(self, text: str, from_lang: str, 
                   to_lang: str) -> Optional[str]:
        """
        翻译查询（基于词对映射）
        
        支持: zh↔en 双向翻译
        其他语言对返回None（由外部API处理）
        """
        # 只支持中英互译
        if (from_lang, to_lang) == ("zh", "en"):
            return self._zh_to_en(text)
        elif (from_lang, to_lang) == ("en", "zh"):
            return self._en_to_zh(text)
        else:
            # 转英文桥接
            if from_lang in ("ja", "ko"):
                # 先转英文（使用内置映射）
                en_text = self._zh_to_en(text)  # 简化
                if en_text and to_lang == "en":
                    return en_text
            return None
    
    def _zh_to_en(self, text: str) -> str:
        """中译英（基于关键词替换）"""
        result = text
        
        # 按长度降序排列（优先匹配长词）
        sorted_map = sorted(
            self.zh_en_map.items(), 
            key=lambda x: len(x[0]), 
            reverse=True
        )
        
        for zh, en in sorted_map:
            if zh in result:
                result = result.replace(zh, en)
        
        # 如果没有任何替换，返回None
        return result if result != text else None
    
    def _en_to_zh(self, text: str) -> str:
        """英译中（基于关键词替换）"""
        result = text.lower()
        
        sorted_map = sorted(
            self.en_zh_map.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        
        for en, zh in sorted_map:
            if en.lower() in result:
                result = result.replace(en.lower(), zh)
        
        return result if result != text.lower() else None
    
    def get_language_weight(self, lang: str, query_lang: str) -> float:
        """获取语言权重"""
        return self.lang_weight.get(lang, 0.5)
    
    def explain(self, text: str) -> Dict:
        """解释跨语言扩展过程"""
        lang, confidence = self.detect_language(text)
        expansions = self.expand(text, lang)
        
        return {
            "original": text,
            "detected_language": lang,
            "language_confidence": confidence,
            "expansions": expansions,
            "bridge_strategy": self.bridge_table.get(lang, []),
            "explanation": (
                f"检测为{lang}语(置信度{confidence:.0%})，"
                f"扩展为{len(expansions)}种语言查询"
            )
        }


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    expander = CrossLangExpander()
    
    test_queries = [
        "比特币价格",
        "machine learning tutorial",
        "人工智能最新进展",
        "Bitcoin price today",
    ]
    
    for q in test_queries:
        result = expander.explain(q)
        print(f"\n原始: {result['original']}")
        print(f"语言: {result['detected_language']} (置信度: {result['language_confidence']:.0%})")
        print(f"扩展: {result['expansions']}")