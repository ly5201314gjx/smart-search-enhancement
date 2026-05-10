---
template_type: simple
intended_use: 快速搜索反馈
max_results: 5
---

## 🔍 "{{ query }}" 的搜索结果

{% if intent %}
*[意图: {{ intent }} | 语言: {{ language }} | 引擎: {{ engines | join(", ") }}]*
{% endif %}

{% for result in results %}
**{{ loop.index }}. [{{ result.title }}]({{ result.url }})**
   {{ result.snippet[:150] }}{% if result.snippet|length > 150 %}...{% endif %}
   *{{ result.source_engine }} · {{ result.quality_score }}分*
{% else %}
*没有找到相关结果*
{% endfor %}

{% if suggestions %}
**🔎 您可能还想搜:**
{% for sug in suggestions[:3] %}
- {{ sug.text }}
{% endfor %}
{% endif %}

*处理时间: {{ processing_time_ms }}ms · 结果数: {{ total_found }}*
