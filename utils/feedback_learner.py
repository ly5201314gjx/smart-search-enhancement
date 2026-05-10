"""反馈学习器 - 点击模型在线学习"""
from typing import Dict, List
from collections import defaultdict

class FeedbackLearner:
    """用户反馈学习器"""
    
    POSITIVE_SIGNALS = {"click": 1.0, "dwell_30s": 0.5, "scroll": 0.3, "copy": 0.8, "share": 1.2}
    NEGATIVE_SIGNALS = {"quick_return": -1.0, "click_and_back": -0.8, "skip": -0.5, "no_interaction": -0.2}
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.feedback_history = defaultdict(list)
        self.result_scores = defaultdict(float)
    
    def record_feedback(self, result_id: str, signal_type: str, value: float = 1.0):
        """记录反馈"""
        weight = self.POSITIVE_SIGNALS.get(signal_type, self.NEGATIVE_SIGNALS.get(signal_type, 0))
        self.feedback_history[result_id].append({"signal": signal_type, "weight": weight, "value": value})
    
    def compute_feedback_score(self, result_id: str) -> float:
        """计算累计反馈得分"""
        feedbacks = self.feedback_history.get(result_id, [])
        if not feedbacks:
            return 0.0
        total = sum(f["weight"] * f["value"] for f in feedbacks)
        count = len(feedbacks)
        return total / count if count > 0 else 0.0
    
    def update_result_score(self, result_id: str, base_score: float) -> float:
        """更新结果得分"""
        feedback = self.compute_feedback_score(result_id)
        self.result_scores[result_id] = base_score + self.learning_rate * feedback
        return self.result_scores[result_id]
    
    def learn(self, result_id: str, signal_type: str, base_score: float = 0.5):
        """学习流程"""
        self.record_feedback(result_id, signal_type)
        return self.update_result_score(result_id, base_score)
    
    def get_top_results(self, result_ids: List[str], base_scores: Dict[str, float]) -> List[str]:
        """获取优化后的排序"""
        scored = {}
        for rid in result_ids:
            base = base_scores.get(rid, 0.5)
            scored[rid] = self.update_result_score(rid, base)
        return sorted(scored.keys(), key=lambda x: scored[x], reverse=True)

if __name__ == "__main__":
    learner = FeedbackLearner()
    learner.learn("result_1", "click")
    learner.learn("result_1", "dwell_30s")
    learner.learn("result_2", "quick_return")
    print(f"result_1 score: {learner.result_scores['result_1']:.3f}")
    print(f"result_2 score: {learner.result_scores['result_2']:.3f}")