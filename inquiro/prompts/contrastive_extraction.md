# Contrastive Trajectory Extraction Prompt

You are an expert research strategist analyzing completed task trajectories.
Your goal is to identify **what distinguishes successful research runs from failed ones**,
and distill these differences into reusable, actionable insights.

---

## SUCCESSFUL TRAJECTORIES ({{ success_count }} total)

These runs achieved high checklist coverage (≥ 0.70) AND high confidence (≥ 0.60):

{% for s in success_summaries %}
### Success [{{ loop.index }}]
- **Topic**: {{ s.topic }}
- **Sub-item**: {{ s.sub_item_id or "N/A" }}
- **Context tags**: {{ s.context_tags | join(", ") or "none" }}
- **Metrics**: coverage={{ "%.2f"|format(s.metrics.checklist_coverage) }}, confidence={{ "%.2f"|format(s.metrics.confidence) }}, evidence={{ s.metrics.evidence_count }}, rounds={{ s.metrics.search_rounds }}, cost_usd={{ "%.4f"|format(s.metrics.cost_usd) }}
- **Decision**: {{ s.metrics.decision or "N/A" }}
- **Tool usage**: {{ s.tool_call_counts }}
- **Failed tool calls**: {{ s.failed_tool_calls }}
- **Wall time (s)**: {{ "%.1f"|format(s.wall_time_seconds) }}
{% endfor %}

---

## FAILED TRAJECTORIES ({{ failure_count }} total)

These runs had low coverage (< 0.50) OR low confidence (< 0.40):

{% for f in failure_summaries %}
### Failure [{{ loop.index }}]
- **Topic**: {{ f.topic }}
- **Sub-item**: {{ f.sub_item_id or "N/A" }}
- **Context tags**: {{ f.context_tags | join(", ") or "none" }}
- **Metrics**: coverage={{ "%.2f"|format(f.metrics.checklist_coverage) }}, confidence={{ "%.2f"|format(f.metrics.confidence) }}, evidence={{ f.metrics.evidence_count }}, rounds={{ f.metrics.search_rounds }}, cost_usd={{ "%.4f"|format(f.metrics.cost_usd) }}
- **Decision**: {{ f.metrics.decision or "N/A" }}
- **Tool usage**: {{ f.tool_call_counts }}
- **Failed tool calls**: {{ f.failed_tool_calls }}
- **Wall time (s)**: {{ "%.1f"|format(f.wall_time_seconds) }}
{% endfor %}

---

## ANALYSIS TASK

Analyze the patterns above and extract up to **{{ max_experiences }}** insights that:

1. Explain **specific strategies or tool choices** that separate success from failure
2. Are **actionable** — an agent reading this can immediately change its behavior
3. Are **generalizable** — not specific to a single topic, but applicable across similar tasks
4. Reference **observable evidence** from the trajectory data (tool usage, round counts, evidence counts)

---

## OUTPUT FORMAT

Respond with a **JSON array only** — no prose, no markdown fences, no explanation.

Valid categories: {{ valid_categories | join(" | ") }}

```
[
  {
    "category": "<one of the valid categories above>",
    "insight": "<specific, actionable insight in 1-3 sentences>",
    "context_tags": ["<tag1>", "<tag2>"],
    "applicable_sub_items": ["*"]
  }
]
```

Rules:
- `category` MUST be one of: {{ valid_categories | join(", ") }}
- `insight` MUST be specific and actionable (not vague like "search better")
- `context_tags` should reflect the observable context where this insight applies
- `applicable_sub_items` use `["*"]` for general insights, or specific IDs if narrow
- Do NOT include any text outside the JSON array
