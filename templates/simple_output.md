## 搜索结果

**{{query}}**

共找到 {{result_count}} 个结果

{% for result in results %}
### {{loop.index}}. {{result.title}}
- 类别: {{result.category}}
- 质量: {{result.quality.level}}
- 链接: {{result.url}}

{{result.snippet}}

{% endfor %}

---
*搜索时间: {{search_time}}ms*