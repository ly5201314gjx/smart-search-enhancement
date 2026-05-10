# 增强搜索使用示例

## 基础搜索示例

```python
from utils import SmartSearchEngine

engine = SmartSearchEngine()

results = [
    {"title": "Python教程", "snippet": "学习Python编程", "url": "https://example.com/python"},
    {"title": "JavaScript入门", "snippet": "Web开发教程", "url": "https://example.com/js"}
]

response = engine.search("Python教程", results)
print(response["intent"])  # Informational
print(len(response["results"]))
```

## 意图识别示例

```python
from utils import classify

# 不同查询的意图识别
queries = [
    "比特币现在价格多少",  # RealTime
    "打开 github.com",     # Navigation
    "附近有什么好吃的",     # Location
    "如何学习Python",      # Informational
]

for q in queries:
    result = classify(q)
    print(f"{q} -> {result['intent']}")
```

## 摘要生成示例

```python
from utils import summarize

text = "Python是一门高级编程语言。它具有简洁易懂的语法，广泛应用于Web开发、数据科学、人工智能等领域。Python的设计哲学强调代码的可读性和简洁性。"

summary = summarize(text, max_length=100)
print(summary)
```

## 质量评分示例

```python
from utils import QualityScorer

scorer = QualityScorer()
result = {
    "title": "Python官方文档",
    "snippet": "Python编程语言官方网站",
    "url": "https://docs.python.org"
}

quality = scorer.score(result, "Python")
print(f"质量分数: {quality['total']}")
print(f"等级: {quality['level']}")
```

## 跨语言搜索示例

```python
from utils import CrossLangExpander

expander = CrossLangExpander()
expansions = expander.expand("Python教程")
for e in expansions:
    print(f"{e['language']}: {e['query']}")
```

## 搜索建议示例

```python
from utils import SuggestionEngine

engine = SuggestionEngine()
engine.add_to_history("Python教程")
engine.add_to_history("Python进阶")
engine.add_trending("AI人工智能", 10)

suggestions = engine.suggest("Pyt")
for s in suggestions:
    print(f"- {s['suggestion']}")
```