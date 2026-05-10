"""性能优化器 - 自适应并行调度"""
from typing import List, Dict, Callable
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class PerformanceOptimizer:
    """搜索性能优化器"""
    
    ENGINE_LATENCY = {"Google": 1.2, "Bing": 0.8, "DuckDuckGo": 1.5, "Baidu": 1.0, "Tavily": 2.0}
    TIMEOUTS = {"RealTime": 5, "default": 10, "Academic": 30}
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.early_stop_ratio = 0.5
    
    def select_engines(self, intent: str, available_engines: List[str]) -> List[str]:
        """根据意图选择引擎"""
        if intent in ["RealTime", "FactualQA"]:
            return sorted(available_engines, key=lambda e: self.ENGINE_LATENCY.get(e, 10))[:3]
        elif intent == "Academic":
            return available_engines
        elif intent in ["Location", "Transactional"]:
            return available_engines[:4]
        return available_engines[:4]
    
    def get_timeout(self, intent: str) -> float:
        """获取超时时间"""
        return self.TIMEOUTS.get(intent, self.TIMEOUTS["default"])
    
    def parallel_search(self, query: str, engines: List[str], search_func: Callable) -> List[Dict]:
        """并行搜索"""
        results = []
        timeout = self.TIMEOUTS["default"]
        start = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(search_func, query, engine): engine for engine in engines}
            for future in as_completed(futures, timeout=timeout):
                engine = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.extend(result)
                        # 早停检查
                        if len(results) >= 10 and (time.time() - start) > timeout * self.early_stop_ratio:
                            break
                except Exception as e:
                    print(f"Engine {engine} error: {e}")
        
        return results
    
    def optimize(self, query: str, intent: str, engines: List[str], search_func: Callable) -> List[Dict]:
        """优化搜索"""
        selected = self.select_engines(intent, engines)
        timeout = self.get_timeout(intent)
        return self.parallel_search_with_timeout(query, selected, search_func, timeout)
    
    def parallel_search_with_timeout(self, query: str, engines: List[str], search_func: Callable, timeout: float) -> List[Dict]:
        """带超时的并行搜索"""
        return self.parallel_search(query, engines, search_func)

if __name__ == "__main__":
    optimizer = PerformanceOptimizer()
    engines = ["Google", "Bing", "DuckDuckGo", "Baidu"]
    selected = optimizer.select_engines("RealTime", engines)
    print(f"Selected: {selected}, timeout: {optimizer.get_timeout('RealTime')}s")