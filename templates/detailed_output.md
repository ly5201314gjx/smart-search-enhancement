## 🔍 搜索详情报告

### 查询信息
- **查询词**: {{query}}
- **识别意图**: {{intent}} ({{intent_confidence}}%)
- **语言**: {{language}}

### 结果概览
- 总结果数: {{result_count}}
- 去重后: {{unique_count}}
- 搜索耗时: {{search_time}}ms

### 结果分类统计
{% for cat, count in categories.items() %}
- {{cat}}: {{count}}个
{% endfor %}

### 质量分布
- ⭐⭐⭐ 优质: {{quality_excellent}}个
- ⭐⭐ 标准: {{quality_standard}}个
- ⭐ 低质: {{quality_low}}个

### 详细结果

{% for result in results %}
#### {{loop.index}}. {{result.title}}
| 属性 | 值 |
|-----|-----|
| 类别 | {{result.category}} |
| 质量分 | {{result.quality.total}} |
| 来源 | {{result.domain}} |
| 时效 | {{result.freshness}} |

{{result.snippet}}

[查看原文]({{result.url}})

---
{% endfor %}

### 相关建议
{% for s in suggestions %}
- {{s.suggestion}}
{% endfor %}