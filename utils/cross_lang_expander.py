"""跨语言扩展器 - 中英日韩自动翻译"""
from typing import Dict, List, Tuple

class CrossLangExpander:
    """跨语言搜索扩展器"""
    
    BRIDGE_TABLE = {
        "zh": ["zh", "en"],
        "en": ["en", "zh"],
        "ja": ["ja", "en", "zh"],
        "ko": ["ko", "en", "zh"],
        "other": ["en", "zh"]
    }
    
    TRANSLATIONS = {
        "hello": {"zh": "你好", "ja": "こんにちは", "ko": "안녕하세요"},
        "python": {"zh": "Python编程", "ja": "Pythonプログラミング", "ko": "Python프로그래밍"},
        "ai": {"zh": "人工智能", "ja": "人工知能", "ko": "인공지능"},
        "search": {"zh": "搜索", "ja": "検索", "ko": "검색"},
        "tutorial": {"zh": "教程", "ja": "チュートリアル", "ko": "튜토리얼"}
    }
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """语言检测"""
        if any('\u4e00' <= c <= '\u9fff' for c in text):
            return "zh", 0.98
        if any('\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' for c in text):
            return "ja", 0.95
        if any('\uac00' <= c <= '\ud7af' for c in text):
            return "ko", 0.95
        return "en", 0.9
    
    def translate(self, text: str, from_lang: str, to_lang: str) -> str:
        """简易翻译（实际应调用翻译API）"""
        words = text.lower().split()
        translated = []
        for w in words:
            if w in self.TRANSLATIONS:
                translated.append(self.TRANSLATIONS[w].get(to_lang, w))
            else:
                translated.append(w)
        return " ".join(translated)
    
    def expand(self, query: str) -> List[Dict]:
        """扩展查询"""
        lang, confidence = self.detect_language(query)
        languages = self.BRIDGE_TABLE.get(lang, self.BRIDGE_TABLE["other"])
        
        results = [{"query": query, "language": lang, "confidence": confidence, "is_original": True}]
        
        for target_lang in languages:
            if target_lang != lang:
                translated = self.translate(query, lang, target_lang)
                results.append({
                    "query": translated,
                    "language": target_lang,
                    "confidence": 0.8,
                    "is_original": False
                })
        
        return results

if __name__ == "__main__":
    expander = CrossLangExpander()
    print(expander.expand("Python教程"))
    print(expander.expand("Python tutorial"))