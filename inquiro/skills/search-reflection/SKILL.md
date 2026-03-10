---
name: search-reflection
description: "Mid-search self-reflection protocol for SearchAgent. Teaches the agent to periodically assess checklist coverage, identify evidence gaps, and adjust search strategy using ThinkTool checkpoints. Use DURING search execution after every 3-4 tool calls, BEFORE calling finish to ensure adequate coverage, when search results seem one-sided or biased toward a single source, or when you suspect tunnel vision on early results. Also use when evidence volume is high but checklist coverage feels low -- reflection catches this mismatch. Do NOT use for post-search evidence grading (use evidence-grader). Do NOT use for convergence or stop-condition reasoning (use discovery-convergence-rules). Do NOT use for query construction (use query-templates)."
license: Apache-2.0
---

# Search Reflection

A structured self-reflection protocol that guides SearchAgent to periodically
assess progress, identify gaps, and adapt search strategy mid-execution.

---

## Overview

Evidence search without reflection leads to tunnel vision: the agent fixates on
early results and misses entire checklist dimensions. This skill provides a
systematic protocol for in-context self-assessment using `think` tool
checkpoints at defined intervals during a search round.

---

## When to Use

- **After every 3-4 tool calls**: Pause and assess what you have found so far.
- **When a search returns zero results**: Reflect on whether the query terms
  are correct or if alternative phrasings are needed.
- **Before calling `finish`**: Mandatory final reflection to confirm you have
  adequate evidence or to document remaining gaps explicitly.
- **When evidence seems one-sided**: Check if you have searched for
  contradicting or alternative viewpoints.

---

## Details

### Reflection Protocol

Use the `think` tool at each checkpoint. Structure your reflection using the
five-step framework below.

#### Step 1: Checklist Coverage Count

Enumerate checklist items and mark each as covered or uncovered.

```
think:
  Checklist coverage assessment:
  - [x] R1 (item description) -- covered by E3, E7
  - [x] R2 (item description) -- covered by E1
  - [ ] R3 (item description) -- NOT YET COVERED
  - [x] R4 (item description) -- covered by E5
  - [ ] O1 (item description) -- NOT YET COVERED (optional)

  Coverage: 3/4 required items (75%), 0/1 optional items (0%)
```

Count **required** items separately from **optional** items. Required coverage
is what determines search adequacy.

#### Step 2: Gap Identification

For each uncovered item, assess WHY it is uncovered:

| Gap Type | Description | Action |
|----------|-------------|--------|
| **Not yet searched** | No queries targeted this item | Search with targeted queries |
| **Searched, no results** | Queries returned nothing relevant | Try alternative terms, different tools |
| **Searched, weak results** | Only Tier 3-4 evidence found | Try authoritative sources, different databases |
| **Partially covered** | Evidence exists but incomplete | Deepen with follow-up queries |

```
think:
  Gap analysis:
  - R3 (long-term safety): Not yet searched. Need to target regulatory
    databases and longitudinal study registries.
  - O1 (market comparison): Searched via Brave, only found news articles
    (Tier 3). Try PubMed for systematic reviews.
```

#### Step 3: Source Diversity Check

Verify you are not over-relying on a single MCP tool:

```
think:
  Source diversity:
  - brave_search: 5 queries -> 8 evidence items
  - pubmed: 2 queries -> 4 evidence items
  - perplexity: 0 queries -> 0 evidence items  <-- UNUSED
  - chembl: 1 query -> 1 evidence item

  Concern: Over-reliance on brave_search. Should use perplexity for
  synthesis-oriented queries and pubmed for clinical data.
```

Aim for at least 2-3 different tools contributing evidence. Single-tool
dependency creates systematic blind spots.

#### Step 4: Strategy Adjustment

Based on gaps and source diversity, decide your next actions:

| Situation | Strategy |
|-----------|----------|
| Many uncovered required items | Broaden search, target uncovered items directly |
| All required covered, low confidence | Seek corroborating sources for weak evidence |
| One stubborn gap | Pivot to alternative databases, try broader terms |
| Good coverage, some optional gaps | Wrap up if required coverage >= 80%, note optional gaps |
| Evidence conflicts detected | Search for resolution evidence (newer studies, larger samples) |

```
think:
  Strategy adjustment:
  - Priority 1: Search R3 (long-term safety) via regulatory databases
  - Priority 2: Strengthen R2 evidence with a second independent source
  - Priority 3: Skip O1 for now (optional, low priority)
  - Tool plan: Use bohrium for regulatory filings, pubmed for clinical data
```

#### Step 5: Confidence Self-Assessment

Before finishing, calibrate your confidence against actual evidence:

| Coverage Level | Confidence Range | Finish Decision |
|----------------|-----------------|-----------------|
| All required items covered, 2+ sources each | 0.80-0.95 | Finish with high confidence |
| All required items covered, some single-source | 0.65-0.80 | Finish, note single-source items |
| Most required items covered (>= 75%) | 0.50-0.70 | Finish if out of tool budget, note gaps |
| Significant required gaps (< 75%) | 0.30-0.50 | Continue searching if budget allows |
| Major gaps, few evidence items | 0.10-0.30 | Document as search failure, finish with gaps |

### Pre-Finish Reflection Template

Before calling `finish`, perform this mandatory checkpoint:

```
think:
  === PRE-FINISH REFLECTION ===

  1. COVERAGE: X/Y required items covered (Z%)
     Uncovered: [list items]

  2. EVIDENCE QUALITY:
     - Tier 1 evidence: N items
     - Tier 2 evidence: N items
     - Tier 3-4 evidence: N items
     - Total unique sources: N

  3. SOURCE DIVERSITY: Used N/M available tools
     Under-utilized: [list tools]

  4. CONFLICTS: [any contradictions noted]

  5. CONFIDENCE: X.XX
     Justification: [brief rationale tied to coverage and quality]

  6. DECISION: FINISH / CONTINUE SEARCHING
     If continue: [next 1-2 queries planned]
```

### Common Anti-Patterns

Avoid these reflection failures:

- **Skipping reflection**: Going straight from search to finish without
  assessing coverage leads to systematic gaps.
- **Counting evidence, not coverage**: Having 20 evidence items means nothing
  if they all cover the same 2 checklist items.
- **Anchoring on first results**: Early positive results create confirmation
  bias. Actively search for contradicting evidence.
- **Tool fixation**: Using only one search tool because it returned results
  first. Different tools access different evidence bases.
- **Premature confidence**: Assigning high confidence based on volume rather
  than checklist coverage and source diversity.

---

## Constraints

- Always count coverage against the **checklist**, not by total evidence volume
  -- 50 evidence items covering 2 of 8 required items is worse than 8 items
  covering all 8, because the checklist defines what constitutes adequate
  research and uncovered items represent genuine knowledge gaps.
- Perform at least one reflection before calling finish -- the pre-finish
  reflection is mandatory because it catches systematic blind spots that
  accumulate during search; agents that skip it consistently miss 15-30% of
  required checklist items.
- Use the `think` tool for reflections, not regular output -- think tool
  reflections are internal reasoning that does not consume output tokens or
  pollute the evidence record, keeping the final output clean and focused.
- Do not inflate confidence based on evidence quantity alone -- confidence must
  correlate with checklist coverage ratio and source tier distribution; high
  volume with low coverage signals broad but shallow search, not thorough
  research.
