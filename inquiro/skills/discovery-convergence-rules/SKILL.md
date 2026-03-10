---
name: discovery-convergence-rules
description: "Complete reference for how DiscoveryLoop decides to stop iterating: the 5 stop conditions, their controlling parameters, INTENSITY_PRESETS, and focus prompt generation between rounds. Use when diagnosing why a loop terminated early or ran too many rounds, when selecting an IntensityLevel for a new task, when reasoning about gap-filling search focus in rounds > 1, when tuning convergence parameters for cost vs coverage tradeoffs, or when a loop stopped with low coverage and you need to understand which stop condition fired. Also use when constructing focus prompts for subsequent rounds or when deciding whether to continue searching or accept current coverage. Do NOT use for evidence scoring or condensation (use evidence-condenser). Do NOT use for confidence calibration (use confidence-calibration-rules). Do NOT use for search query construction (use query-templates)."
license: Apache-2.0
---

# Discovery Convergence Rules

`DiscoveryLoop` iterates through rounds of search -> clean -> analyze -> gap until
one of five deterministic stop conditions is satisfied. This skill documents all five
conditions in priority order, their controlling `DiscoveryConfig` parameters, the
`INTENSITY_PRESETS` reference table, and the focus prompt mechanism.

---

## Overview

The DiscoveryLoop is the core multi-round research orchestrator. After each round,
it checks five stop conditions in fixed priority order. The first matching condition
terminates the loop. Configuration is controlled by `DiscoveryConfig` parameters,
with `INTENSITY_PRESETS` providing sensible defaults for each intensity level.

---

## Details

### Five Stop Conditions (Priority Order)

Checked by `GapAnalysis._check_convergence()` at the end of each round, in fixed
order. The first matching condition wins.

**1. CONVERGED -- `coverage_threshold_reached`**

```
coverage_ratio >= config.coverage_threshold
```

The **desired** termination state. Coverage meets or exceeds the threshold.
Parameter: `coverage_threshold` (default 0.80).

**2. BUDGET_EXHAUSTED -- `max_cost_per_subitem_exhausted`**

```
cost_spent >= config.max_cost_per_subitem
```

Budget cap reached. Loop stops regardless of coverage. Results returned with
whatever coverage was achieved.
Parameter: `max_cost_per_subitem` (default 8.0 USD).

**3. MAX_ROUNDS_REACHED -- `max_rounds_reached`**

```
round_number >= config.max_rounds
```

Maximum rounds completed without converging. Prevents unbounded iteration.
Parameter: `max_rounds` (default 3).

**4. DIMINISHING_RETURNS -- `diminishing_returns`**

```
(coverage_ratio - previous_coverage) < config.convergence_delta
    for config.convergence_patience consecutive rounds
```

Coverage improvement per round has fallen below minimum delta for consecutive
rounds. Skipped on round 1 (no previous data).
Parameters: `convergence_delta` (default 0.08), `convergence_patience` (default 1).

**5. SEARCH_EXHAUSTED -- `search_exhausted`**

```
round_number > 1 AND evidence_count < config.min_evidence_per_round
```

Too few new evidence items found. Search space likely saturated. Only checked
from round 2 onward.
Parameter: `min_evidence_per_round` (default 3).

### INTENSITY_PRESETS Reference Table

Defined in `inquiro/core/types.py`. Each preset is a `dict[str, Any]` applied as
base configuration for `DiscoveryConfig`.

| Parameter | STANDARD | DISCOVERY |
|-----------|:--------:|:---------:|
| `max_rounds` | **2** | **5** |
| `coverage_threshold` | **0.75** | **0.85** |
| `convergence_delta` | **0.08** | **0.05** |
| `convergence_patience` | **1** | **2** |
| `min_evidence_per_round` | **3** | **5** |
| `timeout_per_round` (s) | **600** | **300** |
| `timeout_total` (s) | **1800** | **1500** |
| `max_cost_per_subitem` ($) | **20.0** | **200.0** |
| `enable_parallel_search` | True | True |
| `max_parallel_agents` | **3** | **3** |
| `gap_focus_max_items` | **3** | **3** |
| `enable_synthesis` | True | True |
| `analysis_model_count` | **3** | **3** |

**Pipeline behaviour summary**:
- **STANDARD**: 1-2 rounds, parallel search, 3-model analysis with voting, synthesis, $20 budget, 600s/round, 1800s total
- **DISCOVERY**: 1-5 rounds, parallel search, 3-model analysis, synthesis with voting, $200 budget, 300s/round, 1500s total

### Focus Prompt Generation Mechanism

After each non-final round, `DiscoveryLoop` generates a structured focus prompt for
the next round's search. Contains up to 5 sections:

1. **Primary directive** -- Search for uncovered items using DIFFERENT terms from
   previous rounds
2. **Enumerated uncovered items** -- Up to `gap_focus_max_items` items with synonym
   suggestions
3. **Source priority** -- Government databases, peer-reviewed journals, systematic
   reviews, meta-analyses
4. **Exclusion list** -- Already-covered items the agent should avoid re-searching
   (only included when covered items exist; re-searching covered items wastes rounds
   and budget without improving coverage ratio)
5. **Follow-up rules** (optional) -- Appended from query template `follow_up_rules`

**Broadening Fallback**: When ALL checklist items are covered (at threshold), the
focus prompt switches to: "Strengthen evidence depth for [covered items]. Seek
additional corroborating sources."

**Example**:
```
Input: Round 1 completes with coverage_ratio=0.60, coverage_threshold=0.80.
       Uncovered items: R3 (long-term safety), R7 (regulatory status).

Stop condition check:
  1. CONVERGED? 0.60 < 0.80 -> No
  2. BUDGET_EXHAUSTED? $1.20 < $8.00 -> No
  3. MAX_ROUNDS_REACHED? round 1 < max_rounds 3 -> No
  4. DIMINISHING_RETURNS? skipped (no previous round data)
  5. SEARCH_EXHAUSTED? round 1, skipped

Result: loop continues to round 2.

Focus prompt generated:
  "Search for uncovered items using different terms than round 1.
   Priority gaps: R3 (long-term safety) -- try 'chronic effects', 'safety follow-up';
                  R7 (regulatory status) -- try official filings, authority approval records.
   Do not re-search: R1, R2, R4, R5 (already covered)."
```

---

## Checklist-Aware Convergence

Coverage ratio is computed against the **checklist**, not raw evidence counts. Understanding
how checklist items map to convergence decisions is critical for both the DiscoveryLoop
orchestrator and the SearchAgent operating within each round.

### Per-Round Checklist Tracking

After each round, GapAnalysis computes coverage by counting checklist items that have at
least one supporting evidence item (any tier). The coverage ratio determines the next action:

```
coverage_ratio = covered_required_items / total_required_items
```

Only **required** checklist items contribute to the coverage ratio. Optional items are tracked
but do not affect convergence decisions.

| Round Result | Coverage Ratio | Next Action |
|-------------|---------------|-------------|
| Strong first round | >= threshold | CONVERGE (stop) |
| Good progress | Previous + delta >= threshold | One more round may converge |
| Plateau | Coverage gain < convergence_delta | DIMINISHING_RETURNS if patience exhausted |
| Cold start | < 0.30 after round 1 | Likely needs query strategy pivot |

### Convergence vs Continuing: Decision Guide

Use this decision tree when reasoning about whether to continue searching:

1. **Is coverage >= threshold?** -> Stop (CONVERGED). Additional rounds add cost without
   improving the coverage metric.

2. **Is coverage improving?** (gain >= convergence_delta)
   - Yes -> Continue. Each round is making meaningful progress toward threshold.
   - No -> Check patience counter. If exhausted -> Stop (DIMINISHING_RETURNS).

3. **Are uncovered items searchable?** Review uncovered checklist items:
   - If uncovered items have clear alternative query strategies -> Continue.
   - If uncovered items have been searched with multiple strategies already
     -> Evidence may genuinely not exist. Consider stopping.

4. **Is budget sufficient for another round?** Check remaining budget against
   average cost per round. If < 1 round worth of budget -> Stop soon.

### Prioritizing Uncovered Items

When generating focus prompts for the next round, prioritize uncovered items by:

1. **Required before optional** -- Required items directly affect coverage ratio;
   optional items are informational only.
2. **High-impact before low-impact** -- Items that multiple other assessments depend
   on should be prioritized (e.g., safety data that gates go/no-go decisions).
3. **Likely findable before unlikely** -- Items with clear search strategies and known
   data sources should be targeted first; obscure items may need specialized databases.
4. **Under-searched before well-searched** -- Items that have not been targeted by any
   query yet are more likely to yield results than items that have been searched
   multiple times without success.

### Example: Multi-Round Checklist Progression

```
Round 1: coverage 0.50 (4/8 required items)
  Covered: R1, R2, R4, R6
  Uncovered: R3, R5, R7, R8
  -> Continue (0.50 < 0.80 threshold, budget available)

Round 2: coverage 0.75 (6/8 required items), gain = +0.25
  Newly covered: R3, R7
  Still uncovered: R5, R8
  -> Continue (0.75 < 0.80, gain 0.25 >> delta 0.08)

Round 3: coverage 0.875 (7/8 required items), gain = +0.125
  Newly covered: R5
  Still uncovered: R8
  -> CONVERGED (0.875 >= 0.80 threshold)
  Note: R8 remains uncovered but coverage threshold is met.
```

---

## Constraints

- Treat INTENSITY_PRESETS defaults as a calibrated baseline -- changing them without
  understanding the downstream effects on cost and coverage can cause systematic
  over-spending (too-lenient thresholds) or premature termination (too-strict ones).
- Keep `convergence_patience` at 1 or higher -- a value of 0 would trigger
  DIMINISHING_RETURNS on every round regardless of actual progress, making the
  parameter meaningless.
- Use `max_rounds=1` for intentional single-round mode rather than setting an
  artificially high `coverage_threshold` -- the latter still runs multiple rounds,
  paying full search cost before the threshold forces termination.

---

## Config Parameters Quick Reference

| Field | Type | Default | Governs |
|-------|------|---------|---------|
| `max_rounds` | int | 3 | Stop condition 3 |
| `coverage_threshold` | float | 0.80 | Stop condition 1 |
| `convergence_delta` | float | 0.08 | Stop condition 4 |
| `convergence_patience` | int | 1 | Stop condition 4 |
| `min_evidence_per_round` | int | 3 | Stop condition 5 |
| `max_cost_per_subitem` | float | 8.0 | Stop condition 2 |
| `gap_focus_max_items` | int | 3 | Max uncovered items in focus prompt |
| `enable_synthesis` | bool | True | Run SynthesisExp after all rounds |
| `analysis_model_count` | int | 3 | Models for parallel analysis |
