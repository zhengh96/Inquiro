# PRINCIPLE DISTILLATION TASK

You are a meta-learning analyst reviewing accumulated research insights from multiple
independent evaluations. Your sole task is to distill **exactly 3** reusable operating
principles from this evidence base.

---

## ACCUMULATED INSIGHTS

The following insights were collected across {insight_count} evaluations. They are
grouped by category and ranked by fitness score (higher = more validated).

{insights_by_category}

---

## DISTILLATION INSTRUCTIONS

Review all insights above. Identify **cross-cutting patterns** — strategies, behaviors,
or heuristics that recur across different topics and consistently correlate with
improved evidence coverage and confidence.

A valid principle MUST satisfy ALL of the following criteria:

1. **Imperative form** — begins with an action verb (e.g., "Prioritize", "Always",
   "Verify", "Combine", "Avoid")
2. **Actionable** — concrete enough that a researcher agent can apply it directly
3. **Cross-cutting** — applies across diverse topics, not tied to a specific domain
4. **Non-redundant** — distinct from the other two principles in scope and approach
5. **Evidence-grounded** — traceable to at least 2 source insights in the list above

Do NOT distill principles that:
- Are trivially obvious ("Collect more evidence")
- Contradict each other
- Duplicate existing patterns without adding specificity

---

## OUTPUT FORMAT

Respond with a **JSON array only**. No preamble, no explanation, no markdown fences.
The array must contain exactly 3 objects:

[
  {
    "text": "<imperative principle sentence>",
    "source_insight_ids": ["<id1>", "<id2>", ...]
  },
  {
    "text": "<imperative principle sentence>",
    "source_insight_ids": ["<id1>", "<id2>", ...]
  },
  {
    "text": "<imperative principle sentence>",
    "source_insight_ids": ["<id1>", "<id2>", ...]
  }
]
