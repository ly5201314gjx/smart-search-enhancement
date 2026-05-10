---
template_type: detailed
intended_use: 详细搜索结果展示
max_results: 10
---

# 🔍 搜索结果: "{{ query }}"

## 📋 查询分析
| 项目 | 值 |
|------|-----|
| 意图识别 | {{ intent }} (置信度: {{ intent_confidence }}) |
| 检测语言 | {{ language }} |
| 使用引擎 | {{ engines | join(", ") }} |
| 处理耗时 | {{ processing_time_ms }}ms |
| 总结果数 | {{ total_found }} |

{% if expanded_queries %}
### 🔄 查询扩展
{% for q in expanded_queries %}
- {{ q }}
{% endfor %}
{% endif %}

---

## 📊 结果分类统计
{% for cat, count in categories.items() %}
- **{{ cat }}**: {{ count }}条
{% endfor %}

---

## 📄 搜索结果
{% for result in results %}
<details>
<summary>
<strong>{{ loop.index }}. [{{ result.quality_grade }}] {{ result.title }}</strong>
<em>({{ result.source_engine }} · 相关:{{ (result.relevance_score * 100)|int }}% · 新鲜:{{ (result.recency_score * 100)|int }}% · 权威:{{ (result.authority_score * 100)|int }}%)</em>
</summary>

> 📎 **链接**: [{{ result.url }}]({{ result.url }})
> 📂 **类别**: {{ result.category }}
> 🏷️ **域名**: {{ result.domain }}

**摘要:**
{{ result.summary or result.snippet }}

{% if result.thumbnail %}
![缩略图]({{ result.thumbnail }})
{% endif %}
</details>

{% else %}
> 😕 未找到相关结果，请尝试其他关键词。
{% endfor %}

---

## 🔎 搜索建议
{% for sug in suggestions %}
- **{{ sug.text }}** ({{ sug.rationale }})
{% endfor %}

---

*智能搜索引擎增强 v1.0 · 由9大核心算法驱动*