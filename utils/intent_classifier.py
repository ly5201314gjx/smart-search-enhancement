"""
📐 算法一：搜索意图分类器 (Search Intent Classifier)
====================================================

基于12维特征向量 + 加权逻辑回归 + 规则增强的混合意图识别系统

数学模型：
P(Cⱼ|Q) = 1 / (1 + exp(-Wⱼᵀ·Φ(Q) - bⱼ))

特征维度 Φ(Q) = [Φ₁, Φ₂, ..., Φ₁₂]:

Φ₁: 查询长度 (归一化)
Φ₂: 疑问词密度
Φ₃: 动词密度
Φ₄: 时间敏感度
Φ₅: 位置敏感度
Φ₆: URL模式匹配
Φ₇: 名词-动词比
Φ₈: 数字密度
Φ₉: 疑问句比例
Φ₁₀: 情感词强度
Φ₁₁: 命令式语气强度
Φ₁₂: 类别关键词匹配得分
"""

import re
import math
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

# ============================================================
# 意图类别定义
# ============================================================

INTENT_CATEGORIES = [
    "FactualQA",      # 事实问答
    "Navigation",     # 导航直达
    "Transactional",  # 交易/购买
    "Informational",  # 信息获取
    "RealTime",       # 实时数据
    "Location",       # 位置查询
    "Multimedia",     # 多媒体
    "Academic"        # 学术搜索
]

# ============================================================
# 特征提取规则表
# ============================================================

# 疑问词表（多语言）
QUESTION_WORDS = {
    'zh': ['什么', '怎么', '为什么', '如何', '多少', '哪个', '谁', '哪里', 
           '何时', '是否', '几', '哪', '吗', '呢', '么'],
    'en': ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'is', 'are',
           'do', 'does', 'did', 'can', 'could', 'would', 'should', 'has', 'have']
}

# 动词/行为词表
ACTION_VERBS = {
    'zh': ['买', '下载', '看', '查', '找', '玩', '学', '做', '开', '打',
           '发', '放', '装', '换', '改', '修', '建', '提', '设'],
    'en': ['buy', 'download', 'watch', 'find', 'play', 'learn', 'make', 'open',
           'get', 'create', 'install', 'change', 'build', 'search', 'read']
}

# 时间敏感关键词
TIME_KEYWORDS = {
    'zh': ['今天', '明天', '昨天', '现在', '实时', '最新', '最近', '当前',
           '此刻', '今', '2025', '2026', '刚刚', '正在'],
    'en': ['today', 'tomorrow', 'yesterday', 'now', 'latest', 'current', 
           'recent', 'live', 'breaking', 'upcoming', '2025', '2026']
}

# 位置关键词
LOCATION_KEYWORDS = {
    'zh': ['附近', '哪里', '什么地方', '位置', '地址', '导航', '周围',
           '周边', '距', '在', '附近有'],
    'en': ['near', 'nearby', 'where', 'location', 'address', 'around',
           'close', 'place near', 'restaurants near']
}

# 多媒体关键词
MEDIA_KEYWORDS = {
    'zh': ['图片', '视频', '音乐', '照片', '音频', '图像', '画', '拍',
           '截图', '壁纸', '头像', '表情包'],
    'en': ['image', 'picture', 'photo', 'video', 'music', 'audio', 'gallery',
           'wallpaper', 'meme', 'gif', 'drawing']
}

# 学术关键词
ACADEMIC_KEYWORDS = {
    'zh': ['论文', '研究', '学术', '引用', '期刊', '会议', 'doi', '文献',
           '课题', '博士', '硕士', '科学', '实验', '数据', '分析', '方法'],
    'en': ['paper', 'research', 'academic', 'citation', 'journal', 'conference',
           'doi', 'literature', 'thesis', 'dissertation', 'study', 'method']
}

# 交易/购买关键词
TRANSACTION_KEYWORDS = {
    'zh': ['买', '价格', '多少钱', '购买', '优惠', '便宜', '代购', '出售',
           '交易', '卖', '付费', '订阅', '会员', '折扣', '促销', '团购'],
    'en': ['buy', 'price', 'cost', 'purchase', 'discount', 'sale', 'deal',
           'cheap', 'order', 'subscribe', 'premium', 'membership', 'offer']
}

# 导航关键词
NAV_KEYWORDS = {
    'zh': ['打开', '进入', '访问', '官网', '网站', '首页', '平台', '链接',
           '直达', '去', '上'],
    'en': ['open', 'go to', 'visit', 'login', 'sign in', 'website', 'homepage',
           'official site', 'portal']
}


@dataclass
class IntentResult:
    """意图分类结果"""
    category: str                    # 意图类别
    confidence: float                # 置信度 [0, 1]
    feature_vector: List[float]      # 原始特征向量
    probability_distribution: Dict[str, float]  # 全类别概率分布
    rule_triggered: bool = False     # 是否触发规则引擎
    rule_name: str = ""              # 触发的规则名称


class IntentClassifier:
    """
    搜索意图分类器
    
    整体流程:
    1. 提取12维特征向量 Φ(Q)
    2. 规则引擎快速匹配（高优先级规则优先）
    3. 加权逻辑回归计算概率分布
    4. 置信度判断 + 决策融合
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # 模型权重（预训练值，可在线更新）
        self.weights = self._init_weights()
        self.biases = self._init_biases()
        
        # 规则引擎规则（按优先级排序）
        self.rules = self._init_rules()

    def _init_weights(self) -> Dict[str, List[float]]:
        """
        初始化类别权重矩阵 Wⱼ ∈ ℝ¹²
        
        每个类别对应一个12维权重向量
        """
        return {
            "FactualQA":     [0.5, 1.2, -0.3, 0.1, 0.2, -0.5, 1.0, 0.3, 0.8, 0.1, 0.2, 0.6],
            "Navigation":    [-0.2, -0.5, 0.3, -0.2, -0.1, 1.5, -0.3, -0.2, -0.5, -0.1, 0.8, 0.3],
            "Transactional": [0.1, -0.3, 1.2, -0.1, -0.3, 0.5, 0.2, 0.8, -0.2, 0.4, 0.6, 0.5],
            "Informational": [0.6, 0.5, 0.2, -0.2, -0.1, -0.3, 0.8, 0.1, 0.3, 0.1, 0.1, 0.4],
            "RealTime":      [0.3, 0.2, -0.1, 1.5, -0.2, -0.3, 0.1, 0.5, 0.2, 0.1, 0.1, 0.3],
            "Location":      [0.1, 0.3, 0.1, -0.1, 1.5, -0.2, -0.1, -0.1, 0.2, -0.1, -0.1, 0.2],
            "Multimedia":    [-0.1, -0.2, -0.1, -0.1, -0.1, -0.2, 0.1, -0.1, -0.1, 0.5, 0.1, 1.2],
            "Academic":      [0.4, 0.1, -0.2, -0.1, -0.2, -0.3, 0.6, 0.4, 0.1, 0.1, 0.1, 1.0]
        }

    def _init_biases(self) -> Dict[str, float]:
        """初始化类别偏置项 bⱼ"""
        return {
            "FactualQA": -1.5,
            "Navigation": -2.0,
            "Transactional": -2.5,
            "Informational": -1.0,
            "RealTime": -2.0,
            "Location": -3.0,
            "Multimedia": -2.5,
            "Academic": -2.0
        }

    def _init_rules(self) -> List[Dict]:
        """
        规则引擎初始化（优先级从高到低）
        
        每个规则: {name, match_func, intent}
        """
        return [
            {
                "name": "纯URL导航",
                "priority": 1,
                "match": lambda q: bool(re.match(
                    r'^(https?://|www\.|http://)', q.strip()
                )),
                "intent": "Navigation"
            },
            {
                "name": "学术标识码",
                "priority": 2,
                "match": lambda q: bool(re.search(
                    r'\b(10\.\d{4,}/|arXiv:|doi:|PMID:)\b', q, re.IGNORECASE
                )),
                "intent": "Academic"
            },
            {
                "name": "时间敏感+事件",
                "priority": 3,
                "match": lambda q: (
                    self._count_keywords(q, TIME_KEYWORDS) >= 1 and
                    self._count_keywords(q, ACADEMIC_KEYWORDS) == 0
                ),
                "intent": "RealTime"
            },
            {
                "name": "位置+附近",
                "priority": 4,
                "match": lambda q: (
                    self._count_keywords(q, LOCATION_KEYWORDS) >= 1
                ),
                "intent": "Location"
            },
            {
                "name": "多媒体文件",
                "priority": 5,
                "match": lambda q: (
                    self._count_keywords(q, MEDIA_KEYWORDS) >= 1 or
                    bool(re.search(r'\.(jpg|png|gif|mp4|mp3|pdf)$', q.lower()))
                ),
                "intent": "Multimedia"
            },
            {
                "name": "交易+数字",
                "priority": 6,
                "match": lambda q: (
                    self._count_keywords(q, TRANSACTION_KEYWORDS) >= 1 and
                    bool(re.search(r'\d+', q))
                ),
                "intent": "Transactional"
            },
            {
                "name": "购买/交易意图",
                "priority": 7,
                "match": lambda q: (
                    self._count_keywords(q, TRANSACTION_KEYWORDS) >= 2
                ),
                "intent": "Transactional"
            },
            {
                "name": "导航/直达意图",
                "priority": 8,
                "match": lambda q: (
                    self._count_keywords(q, NAV_KEYWORDS) >= 1 and
                    self._count_keywords(q, QUESTION_WORDS) == 0
                ),
                "intent": "Navigation"
            },
            {
                "name": "学术研究意图",
                "priority": 9,
                "match": lambda q: (
                    self._count_keywords(q, ACADEMIC_KEYWORDS) >= 1
                ),
                "intent": "Academic"
            }
        ]

    def classify(self, query: str) -> IntentResult:
        """
        主分类方法
        
        Parameters:
            query: 用户输入查询文本
            
        Returns:
            IntentResult: 包含分类结果和详细信息
        """
        # Step 1: 提取12维特征向量
        phi = self._extract_features(query)
        
        # Step 2: 规则引擎快速匹配
        rule_result = self._apply_rules(query)
        
        # Step 3: 逻辑回归计算
        logreg_probs = self._logistic_regression(phi)
        
        # Step 4: 规则-模型融合决策
        if rule_result["triggered"]:
            # 规则引擎给出了明确判断
            rule_category = rule_result["intent"]
            rule_confidence = rule_result["confidence"]
            model_confidence = logreg_probs.get(rule_category, 0)
            
            # 融合: max(规则置信度, 模型置信度) 加权
            if rule_confidence > 0.8:
                # 高置信度规则 → 直接采纳
                final_category = rule_category
                final_confidence = max(rule_confidence, model_confidence * 0.7)
            elif rule_confidence > 0.5:
                # 中等置信度 → 规则+模型混合
                final_category = rule_category
                final_confidence = 0.6 * rule_confidence + 0.4 * model_confidence
            else:
                # 低置信度规则 → 交给模型
                final_category = max(logreg_probs, key=logreg_probs.get)
                final_confidence = logreg_probs[final_category]
            
            return IntentResult(
                category=final_category,
                confidence=min(final_confidence, 1.0),
                feature_vector=phi,
                probability_distribution=logreg_probs,
                rule_triggered=True,
                rule_name=rule_result["rule_name"]
            )
        else:
            # 纯模型决策
            best_category = max(logreg_probs, key=logreg_probs.get)
            best_conf = logreg_probs[best_category]
            
            # 低置信度兜底
            if best_conf < 0.4:
                best_category = "Informational"
                best_conf = 0.4
            
            return IntentResult(
                category=best_category,
                confidence=best_conf,
                feature_vector=phi,
                probability_distribution=logreg_probs,
                rule_triggered=False
            )

    def _extract_features(self, query: str) -> List[float]:
        """
        提取12维特征向量 Φ(Q)
        
        详细算法见SKILL.md中算法一
        """
        q_lower = query.lower()
        tokens = self._tokenize(query)
        n = len(tokens) if tokens else 1
        
        # Φ₁: 查询长度 (归一化)
        phi1 = min(len(query) / 50.0, 1.0)
        
        # Φ₂: 疑问词密度
        q_word_count = self._count_keywords_multi(query, QUESTION_WORDS)
        phi2 = q_word_count / n
        
        # Φ₃: 动词密度
        verb_count = self._count_keywords_multi(query, ACTION_VERBS)
        phi3 = verb_count / n
        
        # Φ₄: 时间敏感度
        phi4 = 1.0 if self._count_keywords_multi(query, TIME_KEYWORDS) >= 1 else 0.0
        
        # Φ₅: 位置敏感度
        phi5 = 1.0 if self._count_keywords_multi(query, LOCATION_KEYWORDS) >= 1 else 0.0
        
        # Φ₆: URL模式匹配
        phi6 = 1.0 if re.search(r'(https?://|www\.|\.com|\.cn|\.org|\.net)', 
                               q_lower) else 0.0
        
        # Φ₇: 名词-动词比 (近似)
        noun_count = max(len(tokens) - verb_count - q_word_count, 1)
        phi7 = noun_count / (verb_count + 0.001)
        phi7 = min(phi7 / 10.0, 1.0)  # 归一化
        
        # Φ₈: 数字密度
        digit_count = sum(1 for c in query if c.isdigit())
        phi8 = digit_count / len(query) if len(query) > 0 else 0
        
        # Φ₉: 疑问句比例
        phi9 = 1.0 if re.search(r'[？?]$|吗$|呢$|么$', query) else 0.0
        
        # Φ₁₀: 情感词强度（简化的情感分析）
        positive_words = ['好', '棒', '厉害', '优秀', '推荐', '热门', 'top',
                         'best', 'great', 'amazing', 'popular', 'trending']
        negative_words = ['差', '烂', '垃圾', '坑', '骗', '问题', 'bad', 
                         'worst', 'scam', 'problem', 'error']
        
        pos_count = sum(1 for w in positive_words if w in q_lower)
        neg_count = sum(1 for w in negative_words if w in q_lower)
        phi10 = min((pos_count - neg_count) / 3 + 0.5, 1.0)
        phi10 = max(phi10, 0.0)
        
        # Φ₁₁: 命令式语气强度
        command_count = self._count_keywords_multi(query, {
            'zh': ['帮', '请', '给我', '我要', '我需要', '快点'],
            'en': ['please', 'i want', 'i need', 'give me', 'help']
        })
        phi11 = min(command_count / 2.0, 1.0)
        
        # Φ₁₂: 类别关键词匹配得分
        max_cat_score = 0
        for cat_name, cat_keywords in self._get_category_keywords().items():
            score = sum(1 for kw in cat_keywords if kw in q_lower)
            max_cat_score = max(max_cat_score, score)
        phi12 = min(max_cat_score / 5.0, 1.0)
        
        return [phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8, phi9, phi10, phi11, phi12]

    def _logistic_regression(self, phi: List[float]) -> Dict[str, float]:
        """
        加权逻辑回归
        
        P(Cⱼ|Q) = 1 / (1 + exp(-Wⱼᵀ·Φ(Q) - bⱼ))
        """
        scores = {}
        
        for category in INTENT_CATEGORIES:
            w = self.weights[category]
            b = self.biases[category]
            
            # 计算线性组合: Wⱼᵀ·Φ(Q) + bⱼ
            z = sum(w_i * phi_i for w_i, phi_i in zip(w, phi)) + b
            
            # Sigmoid函数
            prob = 1.0 / (1.0 + math.exp(-z))
            scores[category] = prob
        
        # Softmax归一化
        total = sum(math.exp(s) for s in scores.values())
        return {cat: math.exp(s) / total for cat, s in scores.items()}

    def _apply_rules(self, query: str) -> Dict:
        """
        规则引擎匹配
        
        按优先级遍历规则表，返回第一个匹配的规则
        """
        for rule in self.rules:
            try:
                if rule["match"](query):
                    return {
                        "triggered": True,
                        "intent": rule["intent"],
                        "rule_name": rule["name"],
                        "confidence": max(0.6, 1.0 - rule["priority"] * 0.05)
                    }
            except Exception:
                continue
        
        return {"triggered": False}

    def _tokenize(self, text: str) -> List[str]:
        """分词（支持中英文）"""
        # 先按空格分英文
        tokens = []
        # 中文按字符分
        for char in text:
            if '\u4e00' <= char <= '\u9fff' or '\u3040' <= char <= '\u30ff':
                tokens.append(char)
            elif char.isalnum():
                tokens.append(char)
        return [t for t in tokens if t.strip()]

    def _count_keywords(self, text: str, keyword_dict: Dict) -> int:
        """统计关键词出现次数（多语言）"""
        text_lower = text.lower()
        count = 0
        for lang_keywords in keyword_dict.values():
            for kw in lang_keywords:
                count += text_lower.count(kw.lower())
        return count

    def _count_keywords_multi(self, text: str, keyword_dict: Dict) -> int:
        """多语言关键词计数"""
        return self._count_keywords(text, keyword_dict)

    def _get_category_keywords(self) -> Dict[str, List[str]]:
        """获取每个类别的关键词列表"""
        return {
            "FactualQA": list(set(
                QUESTION_WORDS['zh'] + QUESTION_WORDS['en']
            )),
            "Navigation": list(set(
                NAV_KEYWORDS['zh'] + NAV_KEYWORDS['en']
            )),
            "Transactional": list(set(
                TRANSACTION_KEYWORDS['zh'] + TRANSACTION_KEYWORDS['en']
            )),
            "RealTime": list(set(
                TIME_KEYWORDS['zh'] + TIME_KEYWORDS['en']
            )),
            "Location": list(set(
                LOCATION_KEYWORDS['zh'] + LOCATION_KEYWORDS['en']
            )),
            "Multimedia": list(set(
                MEDIA_KEYWORDS['zh'] + MEDIA_KEYWORDS['en']
            )),
            "Academic": list(set(
                ACADEMIC_KEYWORDS['zh'] + ACADEMIC_KEYWORDS['en']
            )),
            "Informational": ["what is", "how to", "tutorial", "guide", 
                             "介绍", "教程", "指南", "是什么", "意思是"]
        }

    def explain(self, query: str) -> Dict:
        """
        解释分类决策过程（可解释AI）
        
        返回每个维度的贡献度
        """
        result = self.classify(query)
        phi = result.feature_vector
        
        feature_names = [
            "查询长度", "疑问词密度", "动词密度", "时间敏感度",
            "位置敏感度", "URL模式", "名词-动词比", "数字密度",
            "疑问句比例", "情感强度", "命令语气", "关键词匹配"
        ]
        
        # 计算每个特征的贡献度
        w = self.weights[result.category]
        contributions = {}
        for name, weight, value in zip(feature_names, w, phi):
            contributions[name] = {
                "weight": weight,
                "value": value,
                "contribution": weight * value,
                "impact": "正向" if weight * value > 0 else "负向"
            }
        
        return {
            "query": query,
            "predicted_intent": result.category,
            "confidence": result.confidence,
            "rule_triggered": result.rule_triggered,
            "rule_name": result.rule_name if result.rule_triggered else "N/A",
            "top_3_alternatives": sorted(
                result.probability_distribution.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3],
            "feature_contributions": dict(
                sorted(contributions.items(), 
                      key=lambda x: abs(x[1]["contribution"]), 
                      reverse=True)
            )
        }


# ============================================================
# 快速测试
# ============================================================

if __name__ == "__main__":
    classifier = IntentClassifier()
    
    test_cases = [
        "今天比特币价格多少",
        "打开百度",
        "附近有什么好吃的",
        "深度学习入门教程",
        "帮我找一下猫的图片",
        "the impact of transformer on NLP",
        "iPhone 16多少钱",
        "最新的AI新闻",
        "论文doi:10.1000/xyz123",
    ]
    
    print(f"{'查询':<30} {'意图':<15} {'置信度':<10} {'规则':<10}")
    print("=" * 70)
    
    for q in test_cases:
        result = classifier.classify(q)
        rule_info = result.rule_name if result.rule_triggered else "-"
        print(f"{q:<30} {result.category:<15} "
              f"{result.confidence:<10.2f} {rule_info:<10}")
    
    print("\n\n=== 可解释AI示例 ===")
    explanation = classifier.explain("今天比特币价格多少")
    print(f"查询: {explanation['query']}")
    print(f"预测: {explanation['predicted_intent']} (置信度: {explanation['confidence']:.2f})")
    print(f"Top-3候选: {explanation['top_3_alternatives']}")
    print("\n特征贡献度(Top-5):")
    for name, info in list(explanation['feature_contributions'].items())[:5]:
        print(f"  {name}: 权重={info['weight']:.2f} 值={info['value']:.2f} 贡献={info['contribution']:.3f} ({info['impact']})")