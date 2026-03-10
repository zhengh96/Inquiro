# Round Reflection Prompt Template

Used by `RoundReflectionMechanism` to generate structured inter-round
reflections via a lightweight LLM call (~$0.02 Haiku pricing per round).

**Format placeholders** (Python `.format()` syntax):

| Placeholder | Source |
|---|---|
| `{round_num}` | Current round number (int) |
| `{evidence_count}` | Evidence items collected (int) |
| `{coverage:.0%}` | Checklist coverage as percentage (float 0–1) |
| `{tool_summary}` | Tool names with call counts (str) |
| `{previous_reflections_section}` | Prior round strategies (str, may be empty) |

Literal JSON braces in the template use `{{` / `}}` (Python format escaping).

---

You are analyzing the results of search round {round_num} for a research task.

## Round {round_num} Summary
- Evidence collected: {evidence_count}
- Coverage achieved: {coverage:.0%}
- Tools used: {tool_summary}
{previous_reflections_section}
## Task

Analyze this round and provide a structured reflection in JSON format.
Your analysis must be concise (<= 200 tokens output) and actionable.

```json
{{
  "what_worked": "Brief description of effective strategies observed in this round",
  "what_failed": "Brief description of ineffective approaches or missed opportunities",
  "strategy": "Specific strategic adjustment for the next search round",
  "tool_recommendations": {{
    "tool_name": "increase|decrease|maintain"
  }},
  "priority_gaps": ["gap_item_1", "gap_item_2", "gap_item_3"]
}}
```

## Guidelines

- **what_worked**: Focus on query formulations, source types, or tools that yielded
  high-quality evidence this round.
- **what_failed**: Note queries with low yield, irrelevant results, or duplicated
  effort that should be avoided.
- **strategy**: Propose ONE specific change — a new research angle, unexplored
  information source, or refined query strategy.
- **tool_recommendations**: Only list tools where usage should meaningfully change.
  Omit tools with no recommendation.
- **priority_gaps**: List up to 5 specific topics, questions, or checklist items
  that remain uncovered after this round.

Be concise and actionable. Focus on what to change, not what to repeat.
