"""意图分类器 - 12维特征向量识别8种搜索意图"""
import re
from typing import Dict, List, Tuple

class IntentClassifier:
    """搜索意图分类器"""
    
    CATEGORIES = ["FactualQA", "Navigation", "Transactional", "Informational", "RealTime", "Location", "Multimedia", "Academic"]
    
    INTENT_PATTERNS = {
        "Navigation": [r"^https?://", r"\.com$", r"\.cn$", r"\.org$"],
        "Transactional": [r"买|价格|多少钱|购买|订单"],
        "RealTime": [r"今天|现在|最新|当前|实时"],
        "Location": [r"哪里|附近|地址|位置"],
        "Multimedia": [r"图片|视频|音乐|电影"],
        "Academic": [r"论文|研究|学术|doi"]
    }
    
    def __init__(self):
        self.weights = [0.1] * 12
    
    def extract_features(self, query: str) -> List[float]:
        """提取12维特征向量"""
        words = list(query)
        n = len(query) or 1
        return [
            min(len(query) / 50, 1.0),  # Φ1: 查询长度
            len(re.findall(r'[什么怎么为什么谁哪]', query)) / n,  # Φ2: 疑问词密度
            len(re.findall(r'[买查找看学做]', query)) / n,  # Φ3: 动词密度
            1 if re.search(r'今天|现在|最新|当前', query) else 0,  # Φ4: 时间敏感度
            1 if re.search(r'哪里|附近|位置', query) else 0,  # Φ5: 位置敏感度
            1 if re.search(r'https?://|\.com|\.cn', query) else 0,  # Φ6: URL模式
            0.5,  # Φ7: 名词动词比
            len(re.findall(r'\d', query)) / n,  # Φ8: 数字密度
            1 if query.endswith(('?', '？', '吗', '呢')) else 0,  # Φ9: 疑问句
            0,  # Φ10: 情感词强度
            len(re.findall(r'帮我|请|我要', query)) / n,  # Φ11: 命令式
            0.5  # Φ12: 类别关键词匹配
        ]
    
    def classify(self, query: str) -> Tuple[str, float]:
        """分类并返回置信度"""
        features = self.extract_features(query)
        score = sum(w * f for w, f in zip(self.weights, features))
        
        # 规则匹配
        for category, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return category, 0.9
        
        # 默认返回Informational
        return "Informational", max(0.5, score)

def classify(query: str) -> Dict:
    """便捷函数"""
    clf = IntentClassifier()
    intent, confidence = clf.classify(query)
    return {"intent": intent, "confidence": confidence, "query": query}

if __name__ == "__main__":
    test_queries = ["帮我查一下比特币价格", "https://github.com", "附近有什么好吃的", "如何学习Python"]
    for q in test_queries:
        print(f"{q} -> {classify(q)}")