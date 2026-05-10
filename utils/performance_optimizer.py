"""
📐 算法八：搜索性能优化器 (Search Performance Optimizer)
=========================================================

自适应引擎选择 + 超时控制 + 提前终止策略。

核心策略:
1. 意图感知引擎选择
2. 地域感知引擎优先级
3. 动态超时控制
4. 提前终止（已获取足够高质量结果时取消待处理请求）
"""

from typing import Dict, List, Optional, Set, Tuple


class PerformanceOptimizer:
    """
    搜索性能优化器
    
    根据搜索意图和用户地域，智能选择最优引擎组合和超时配置
    """
    
    # 引擎延迟评级（秒）
    ENGINE_LATENCY = {
        "google": 0.8,
        "bing": 0.6,
        "duckduckgo": 1.0,
        "baidu": 0.5,
        "sogou": 0.7,
        "quark": 0.9,
        "tavily": 1.5,
    }
    
    # 引擎质量评级
    ENGINE_QUALITY = {
        "google": 0.95,
        "bing": 0.90,
        "tavily": 0.88,
        "duckduckgo": 0.85,
        "baidu": 0.75,
        "sogou": 0.70,
        "quark": 0.65,
    }
    
    # 意图→引擎选择策略
    INTENT_ENGINE_MAP = {
        "RealTime": {
            "priority": ["google", "bing", "baidu"],
            "max_engines": 3,
            "description": "实时查询，速度快优先"
        },
        "FactualQA": {
            "priority": ["google", "bing", "tavily"],
            "max_engines": 3,
            "description": "事实查询，准确度优先"
        },
        "Academic": {
            "priority": ["google", "tavily", "duckduckgo", "bing"],
            "max_engines": 4,
            "description": "学术搜索，覆盖面优先"
        },
        "Navigation": {
            "priority": ["baidu", "google", "bing"],
            "max_engines": 2,
            "description": "导航直达，速度优先"
        },
        "Location": {
            "priority": ["baidu", "google", "bing"],
            "max_engines": 2,
            "description": "位置查询，本地优先"
        },
        "Multimedia": {
            "priority": ["google", "bing", "baidu"],
            "max_engines": 3,
            "description": "多媒体搜索，覆盖面优先"
        },
        "Transactional": {
            "priority": ["google", "bing", "baidu"],
            "max_engines": 3,
            "description": "交易查询，准确度优先"
        },
        "Informational": {
            "priority": ["google", "bing", "duckduckgo", "baidu", "tavily"],
            "max_engines": 3,
            "description": "信息查询，综合最优"
        }
    }
    
    # 意图→超时配置
    INTENT_TIMEOUT = {
        "RealTime": 5,        # 实时查询，5秒超时
        "FactualQA": 8,       # 事实查询，8秒
        "Academic": 30,       # 学术搜索，30秒（长查询）
        "Navigation": 5,      # 导航，5秒
        "Location": 8,        # 位置，8秒
        "Multimedia": 10,     # 多媒体，10秒
        "Transactional": 8,   # 交易，8秒
        "Informational": 10,  # 信息查询，10秒
        "default": 10
    }
    
    # 地域→引擎偏好
    REGION_ENGINE_PREFERENCE = {
        "CN": {
            "primary": ["baidu", "bing", "sogou", "quark"],
            "secondary": ["google", "duckduckgo"],
            "description": "中国大陆用户，百度优先"
        },
        "US": {
            "primary": ["google", "bing", "duckduckgo"],
            "secondary": ["tavily"],
            "description": "美国用户，Google优先"
        },
        "EU": {
            "primary": ["google", "duckduckgo", "bing"],
            "secondary": ["tavily"],
            "description": "欧洲用户，DuckDuckGo可选"
        },
        "OTHER": {
            "primary": ["google", "bing", "duckduckgo"],
            "secondary": [],
            "description": "其他地区，通用选择"
        }
    }
    
    def __init__(self, region: str = "CN"):
        self.region = region
    
    def select_engines(self, intent: str, language: str = "zh",
                       max_engines: Optional[int] = None) -> List[str]:
        """
        根据意图和语言选择最优引擎
        
        Args:
            intent: 搜索意图
            language: 用户语言
            max_engines: 最大引擎数（默认按策略）
        
        Returns:
            按优先级排序的引擎列表
        """
        # 获取意图引擎策略
        strategy = self.INTENT_ENGINE_MAP.get(
            intent, 
            self.INTENT_ENGINE_MAP["Informational"]
        )
        
        # 获取地域引擎偏好
        region_pref = self.REGION_ENGINE_PREFERENCE.get(
            self.region,
            self.REGION_ENGINE_PREFERENCE["OTHER"]
        )
        
        # 组合选择: 意图优先 × 地域偏好
        priority_engines = strategy["priority"]
        max_n = max_engines or strategy["max_engines"]
        
        # 按地域调整优先级
        adjusted = []
        
        # 优先选地域主引擎中也在策略列表里的
        for engine in region_pref["primary"]:
            if engine in priority_engines:
                adjusted.append(engine)
        
        # 补充其他策略推荐的引擎
        for engine in priority_engines:
            if engine not in adjusted:
                adjusted.append(engine)
        
        # 如果不够，补充地域备选
        if len(adjusted) < max_n:
            for engine in region_pref["secondary"]:
                if engine not in adjusted:
                    adjusted.append(engine)
        
        return adjusted[:max_n]
    
    def get_timeout(self, intent: str) -> int:
        """获取超时配置（秒）"""
        return self.INTENT_TIMEOUT.get(intent, self.INTENT_TIMEOUT["default"])
    
    def should_early_stop(self, results_so_far: List[Dict], 
                          intent: str) -> bool:
        """
        判断是否应提前终止待处理请求
        
        条件:
        - 已获得≥3个高质量结果(quality > 75)
        - 或已获得≥5个标准结果(quality > 60)
        """
        high_quality = sum(
            1 for r in results_so_far 
            if r.get("quality_score", 0) > 75
        )
        standard_quality = sum(
            1 for r in results_so_far
            if r.get("quality_score", 0) > 60
        )
        
        if high_quality >= 3:
            return True
        if standard_quality >= 5:
            return True
        
        return False
    
    def get_search_plan(self, query_text: str, intent: str, 
                        language: str = "zh") -> Dict:
        """生成完整搜索计划"""
        engines = self.select_engines(intent, language)
        timeout = self.get_timeout(intent)
        
        return {
            "engines": engines,
            "timeout_seconds": timeout,
            "parallel_count": len(engines),
            "early_stop_enabled": True,
            "estimated_latency": max(
                [self.ENGINE_LATENCY.get(e, 1.0) for e in engines]
            ),
            "strategy": f"意图={intent}, 地域={self.region}, 引擎={engines}",
            "priority_engines": engines[:2],
            "fallback_engines": engines[2:]
        }


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    optimizer = PerformanceOptimizer(region="CN")
    
    test_cases = [
        ("RealTime", "zh"),
        ("Academic", "en"),
        ("Navigation", "zh"),
        ("Informational", "zh"),
        ("Location", "zh"),
    ]
    
    print(f"{'意图':<15} {'语言':<5} {'引擎选择':<30} {'超时':<5}")
    print("=" * 60)
    
    for intent, lang in test_cases:
        plan = optimizer.get_search_plan("test", intent, lang)
        engines_str = ", ".join(plan["engines"])
        print(f"{intent:<15} {lang:<5} {engines_str:<30} {plan['timeout_seconds']:<5}s")