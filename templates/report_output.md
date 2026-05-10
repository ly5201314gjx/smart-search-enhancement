# {{title}}

## 执行摘要
{{summary}}

## 搜索背景
{{background}}

## 主要发现
{{findings}}

## 数据分析
{{analysis}}

## 结论
{{conclusions}}

## 参考来源
{% for ref in references %}
{{loop.index}}. [{{ref.title}}]({{ref.url}}) - {{ref.source}}
{% endfor %}

## 附录
- 搜索时间: {{search_time}}
- 结果数量: {{result_count}}
- 覆盖时间范围: {{time_range}}

---
*报告生成时间: {{report_time}}*