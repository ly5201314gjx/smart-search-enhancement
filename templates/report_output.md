---
template_type: report
intended_use: 深度搜索结果分析报告
max_results: 20
---

# 📊 搜索分析报告: "{{ query }}"

## 1️⃣ 查询概述

- **原始查询**: {{ query }}
- **处理查询**: {{ processed_query }}
- **意图分类**: {{ intent }} (置信度: {{ "%.1f%%"|format(intent_confidence * 100) }})
- **语言**: {{ language }}
- **搜索引擎**: {{ engines | join(" + ") }}
- **响应时间**: {{ "%.0f"|format(processing_time_ms) }}ms
- **结果总量**: {{ total_found }}

{% if query_expansion %}
### 🔄 查询扩展执行
{% for type, q in query_expansion.items() %}
- [{{ type }}] {{ q }}
{% endfor %}
{% endif %}

---

## 2️⃣ 质量分析

| 等级 | 数量 | 占比 |
|------|------|------|
| ⭐⭐⭐ 优质 | {{ quality_counts.excellent }} | {{ "%.1f%%"|format(quality_counts.excellent / total_found * 100) if total_found > 0 else 0 }} |
| ⭐⭐ 标准 | {{ quality_counts.standard }} | {{ "%.1f%%"|format(quality_counts.standard / total_found * 100) if total_found > 0 else 0 }} |
| ⭐ 低质 | {{ quality_counts.low }} | {{ "%.1f%%"|format(quality_counts.low / total_found * 100) if total_found > 0 else 0 }} |
| ❌ 过滤 | {{ quality_counts.rejected }} | {{ "%.1f%%"|format(quality_counts.rejected / total_found * 100) if total_found > 0 else 0 }} |

---

## 3️⃣ 分类分布

{% for cat, count in categories.items() %}
- **{{ cat }}**: {{ count }}条 ({{ "%.1f%%"|format(count / total_found * 100) if total_found > 0 else 0 }})
{% endfor %}

---

## 4️⃣ Top 结果详情

{% for result in results %}
### {{ loop.index }}. {{ result.title }}

| 属性 | 值 |
|------|-----|
| 链接 | [访问]({{ result.url }}) |
| 来源 | {{ result.source_engine }} |
| 域名 | {{ result.domain }} |
| 类别 | {{ result.category }} |
| 质量 | {{ result.quality_score }}/100 |
| 相关度 | {{ "%.0f%%"|format(result.relevance_score * 100) }} |
| 时效性 | {{ "%.0f%%"|format(result.recency_score * 100) }} |
| 权威性 | {{ "%.0f%%"|format(result.authority_score * 100) }} |
| 多样性 | {{ "%.0f%%"|format(result.diversity_score * 100) }} |

**智能摘要:**
> {{ result.summary or result.snippet }}

{% if result.thumbnail %}
![图片]({{ result.thumbnail }})
{% endif %}

---

{% endfor %}

## 5️⃣ 搜索建议

{% for sug in suggestions %}
- **{{ sug.text }}** (来源: {{ sug.source }}, 相关度: {{ sug.score }})
{% endfor %}

---

## 6️⃣ 优化建议

{% if warnings %}
{% for w in warnings %}
- ⚠️ {{ w }}
{% endfor %}
{% else %}
- ✅ 搜索过程正常完成
{% endif %}

---

*报告由AI智能搜索引擎增强 v1.0 自动生成 · 9大核心算法驱动*