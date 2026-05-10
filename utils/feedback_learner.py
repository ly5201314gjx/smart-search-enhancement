"""
📐 算法九：反馈闭环学习器 (Feedback Loop Learner)
====================================================

基于用户行为信号的在线学习系统。

信号类型:
- 正向: 点击(+1.0), 停留>30s(+0.5), 滚动(+0.3), 复制(+0.8)
- 负向: 快速返回(-1.0), 点击后即退(-0.8), 跳过(-0.5), 无交互(-0.2)

学习更新: W_new = W_old + η · (feedback - predicted) · Φ
"""

import json
import time
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter


@dataclass
class UserAction:
    """用户行为记录"""
    action_type: str                 # 行为类型
    result_id: str                   # 目标结果ID
    weight: float                    # 行为权重
    timestamp: float                 # 行为时间
    session_id: str = ""             # 会话ID


@dataclass
class FeedbackSignal:
    """反馈信号"""
    result_id: str
    total_feedback: float            # 累计反馈值
    action_count: int                # 行为次数
    positive_count: int              # 正向行为数
    negative_count: int              # 负向行为数
    avg_dwell_time: float            # 平均停留时间
    last_interaction: float          # 最后交互时间


class FeedbackLearner:
    """
    反馈闭环学习器
    
    功能:
    1. 用户行为追踪
    2. 隐式/显式反馈收集
    3. 排序参数在线更新
    4. 个性化偏好学习
    """
    
    # 行为权重定义
    POSITIVE_SIGNALS = {
        "click": 1.0,              # 点击结果
        "dwell_30s": 0.5,          # 停留超过30秒
        "scroll": 0.3,             # 滚动/翻页
        "copy": 0.8,               # 复制内容
        "bookmark": 1.2,           # 收藏
        "share": 1.0,              # 分享
        "deep_click": 0.7,         # 点击进入深度链接
    }
    
    NEGATIVE_SIGNALS = {
        "quick_return": -1.0,      # 快速返回搜索(<5s)
        "click_and_back": -0.8,    # 点击后立即返回
        "skip_to_other": -0.5,     # 跳过选其他
        "no_interaction": -0.2,    # 无任何交互
        "pogo_stick": -0.7,        # 反复点击同一结果
        "close_page": -0.3,        # 关闭页面
    }
    
    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate
        
        # 行为日志
        self.actions: List[UserAction] = []
        
        # 结果反馈缓存
        self.feedback_cache: Dict[str, FeedbackSignal] = {}
        
        # 排序参数（可在线更新）
        self.sorting_weights = {
            "engine_reliability": 0.15,
            "relevance": 0.35,
            "recency": 0.20,
            "authority": 0.20,
            "diversity": 0.10
        }
        
        # 用户偏好
        self.user_preferences = {
            "preferred_language": "zh",
            "preferred_sources": [],
            "preferred_categories": [],
            "avoided_domains": [],
            "query_history_embedding": []  # 简化版
        }
        
        # 学习统计
        self.stats = {
            "total_actions": 0,
            "total_updates": 0,
            "avg_feedback_score": 0.0,
            "learning_progress": 0.0
        }
    
    def record_action(self, action_type: str, result_id: str, 
                     session_id: str = "") -> UserAction:
        """
        记录用户行为
        
        Args:
            action_type: 行为类型
            result_id: 结果标识
            session_id: 会话ID
        
        Returns:
            创建的UserAction对象
        """
        action = UserAction(
            action_type=action_type,
            result_id=result_id,
            weight=self._get_signal_weight(action_type),
            timestamp=time.time(),
            session_id=session_id
        )
        
        self.actions.append(action)
        self.stats["total_actions"] += 1
        
        # 更新反馈缓存
        self._update_feedback_cache(result_id, action)
        
        # 每10个行为触发一次更新
        if len(self.actions) % 10 == 0:
            self._update_sorting_weights()
        
        return action
    
    def get_feedback(self, result_id: str) -> Optional[FeedbackSignal]:
        """获取某个结果的反馈"""
        return self.feedback_cache.get(result_id)
    
    def get_all_feedback(self) -> Dict[str, FeedbackSignal]:
        """获取所有反馈"""
        return dict(self.feedback_cache)
    
    def evaluate_result_quality(self, result_id: str) -> float:
        """
        基于反馈评估结果质量
        
        Returns: [-1, 1] 范围的分值
        """
        signal = self.feedback_cache.get(result_id)
        if not signal:
            return 0.0
        
        # 归一化反馈
        norm_feedback = signal.total_feedback / max(signal.action_count, 1)
        return max(-1.0, min(1.0, norm_feedback))
    
    def update_preferences(self, query_text: str, 
                           clicked_results: List[str]):
        """更新用户偏好"""
        # 简化版：记录偏好来源
        for result_id in clicked_results:
            if result_id not in self.user_preferences["preferred_sources"]:
                self.user_preferences["preferred_sources"].append(result_id)
        
        # 限制长度
        if len(self.user_preferences["preferred_sources"]) > 100:
            self.user_preferences["preferred_sources"] = \
                self.user_preferences["preferred_sources"][-50:]
    
    def adjust_weights_for_result(self, result: Dict, 
                                   query: Dict) -> Dict:
        """
        根据学习到的偏好调整结果评分
        
        返回调整后的评分
        """
        adjusted = dict(result)
        
        # 来源偏好增强
        source = result.get("source_engine", "")
        if source in self.user_preferences["preferred_sources"]:
            adjusted["final_score"] = adjusted.get("final_score", 0) * 1.1
        
        # 类别偏好增强
        category = adjusted.get("category", "")
        if category in self.user_preferences["preferred_categories"]:
            adjusted["final_score"] = adjusted.get("final_score", 0) * 1.05
        
        # 避免的域名扣分
        domain = result.get("domain", "")
        if domain in self.user_preferences["avoided_domains"]:
            adjusted["final_score"] = adjusted.get("final_score", 0) * 0.7
        
        return adjusted
    
    def generate_performance_report(self) -> Dict:
        """生成学习性能报告"""
        positive_count = sum(
            1 for a in self.actions 
            if a.weight > 0
        )
        negative_count = sum(
            1 for a in self.actions 
            if a.weight < 0
        )
        
        total_positive = sum(
            a.weight for a in self.actions if a.weight > 0
        )
        total_negative = sum(
            a.weight for a in self.actions if a.weight < 0
        )
        
        return {
            "total_actions": len(self.actions),
            "positive_actions": positive_count,
            "negative_actions": negative_count,
            "positive_to_negative_ratio": (
                positive_count / max(negative_count, 1)
            ),
            "net_feedback_score": total_positive + total_negative,
            "current_sorting_weights": self.sorting_weights,
            "learning_rate": self.lr,
            "cached_feedback_count": len(self.feedback_cache),
            "user_preferences_learned": {
                "preferred_sources": len(
                    self.user_preferences["preferred_sources"]
                ),
                "avoided_domains": len(
                    self.user_preferences["avoided_domains"]
                )
            }
        }
    
    def _get_signal_weight(self, action_type: str) -> float:
        """获取行为对应的权重"""
        if action_type in self.POSITIVE_SIGNALS:
            return self.POSITIVE_SIGNALS[action_type]
        elif action_type in self.NEGATIVE_SIGNALS:
            return self.NEGATIVE_SIGNALS[action_type]
        return 0.0
    
    def _update_feedback_cache(self, result_id: str, action: UserAction):
        """更新反馈缓存"""
        if result_id not in self.feedback_cache:
            self.feedback_cache[result_id] = FeedbackSignal(
                result_id=result_id,
                total_feedback=0.0,
                action_count=0,
                positive_count=0,
                negative_count=0,
                avg_dwell_time=0.0,
                last_interaction=0.0
            )
        
        signal = self.feedback_cache[result_id]
        signal.total_feedback += action.weight
        signal.action_count += 1
        
        if action.weight > 0:
            signal.positive_count += 1
        elif action.weight < 0:
            signal.negative_count += 1
        
        signal.last_interaction = action.timestamp
    
    def _update_sorting_weights(self):
        """
        在线更新排序权重
        
        W_new = W_old + η · (feedback_avg - predicted) · Φ
        """
        if not self.feedback_cache:
            return
        
        # 计算平均反馈
        feedback_values = [
            s.total_feedback / max(s.action_count, 1)
            for s in self.feedback_cache.values()
        ]
        
        if not feedback_values:
            return
        
        avg_feedback = sum(feedback_values) / len(feedback_values)
        
        # 当前预测值（简化版：权重和）
        predicted = sum(self.sorting_weights.values())
        
        # 更新误差
        error = avg_feedback - predicted
        
        # 按比例调整各维度权重（使用固定梯度）
        if abs(error) > 0.1:  # 显著误差才调整
            adjustment = self.lr * error
            
            # 调整权重（保持总和=1）
            for key in self.sorting_weights:
                self.sorting_weights[key] += adjustment * 0.2
            
            # 归一化
            total = sum(self.sorting_weights.values())
            for key in self.sorting_weights:
                self.sorting_weights[key] /= total
            
            self.stats["total_updates"] += 1
            self.stats["avg_feedback_score"] = avg_feedback
            
            progress = min(self.stats["total_updates"] / 100, 1.0)
            self.stats["learning_progress"] = progress


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    learner = FeedbackLearner()
    
    # 模拟用户行为序列
    test_actions = [
        ("click", "result_1"),
        ("dwell_30s", "result_1"),
        ("scroll", "result_1"),
        ("quick_return", "result_2"),
        ("skip_to_other", "result_3"),
        ("click", "result_1"),
        ("copy", "result_1"),
    ]
    
    for action_type, result_id in test_actions:
        learner.record_action(action_type, result_id)
    
    print("反馈学习报告:")
    report = learner.generate_performance_report()
    for key, value in report.items():
        print(f"  {key}: {value}")
    
    print("\n各结果反馈:")
    for rid, signal in learner.get_all_feedback().items():
        print(f"  {rid}: total={signal.total_feedback:.1f}, "
              f"actions={signal.action_count}, "
              f"quality={learner.evaluate_result_quality(rid):.2f}")