---
name: evidence-condenser
description: "Prevent analysis failure from evidence overflow by condensing large collections into quality-ranked subsets using deterministic 6-signal scoring. Without condensation, >150 evidence items cause AnalysisExp token budget exhaustion and silent truncation. Use when evidence count exceeds 150, when tuning scoring signal weights, when debugging why specific evidence was selected or dropped during condensation, when understanding source quality tier mappings for new MCP tools, or when analysis results seem incomplete and you suspect important evidence was lost to truncation. Also use when adding a new search tool and you need to assign it an appropriate evidence tag to avoid systematic deprioritization. Do NOT use for evidence quality grading or confidence calibration (use evidence-grader). Do NOT use for pre-Finish output validation (use quality-gate or evidence-validator). Do NOT use for URL-based source tagging (use evidence-source-classification)."
license: Apache-2.0
---

# Evidence Condenser

The `EvidenceCondenser` reduces large evidence pools to a quality-ranked subset
that fits within LLM context limits. It is deterministic (same input -> same
output), domain-agnostic, and requires zero LLM calls for Tier 0 and Tier 1.

**Source module**: `inquiro/core/evidence_condenser.py`

---

## Overview

When `DiscoveryLoop` accumulates evidence across multiple search rounds, the
total count can far exceed the ~128K token budget of the analysis LLM. The
condenser is called inside `DiscoveryLoop._run_analysis()` to reduce the
evidence list before it reaches `AnalysisExp`.

### Three-Tier Strategy

| Tier | Trigger Condition | Target Count | Technique |
|------|-------------------|--------------|-----------|
| **Tier 0** | <= 150 items | (unchanged) | Passthrough -- no filtering |
| **Tier 1** | 151-400 items | <= 160 | Multi-signal score + greedy saturation |
| **Tier 2** | > 400 items | <= 150 | Tier 1 selection + tag-grouped text summaries |

---

## The 6 Scoring Signals

All signals are normalised to `[0, 1]` before weighting. The composite score:

```
score = SUM(signal_i * weight_i)
```

### Default Weights

| Signal | Config Field | Default Weight | What It Measures |
|--------|-------------|:-------------:|------------------|
| `keyword_relevance` | `weight_keyword_relevance` | **0.35** | Fraction of checklist keywords present in evidence summary |
| `source_quality` | `weight_source_quality` | **0.15** | Quality tier of the `evidence_tag` field (see tag quality map) |
| `quality_label` | `weight_quality_label` | **0.15** | Quality tier from the `quality_label` field |
| `journal_quality` | `weight_journal_quality` | **0.10** | DOI prefix lookup score from caller-supplied journal map |
| `structural_completeness` | `weight_structural_completeness` | **0.10** | Fraction of provenance metadata fields populated |
| `round_recency` | `weight_round_recency` | **0.05** | Normalised search-round number (later rounds = gap-filling) |

**Total default weight sum**: 0.90. Weights do not need to sum to 1.0 --
asymmetric weighting is intentional (e.g., setting `weight_keyword_relevance=1.0`
for keyword-only mode is valid).

---

## Signal Lookup Tables

### tag_quality_map -- Source Quality by Evidence Tag

Maps `Evidence.evidence_tag` to a normalised quality score:

| Tag | Score | Rationale |
|-----|:-----:|-----------|
| `regulatory` | **1.0** | Official government filings; highest epistemic authority |
| `clinical_trial` | **0.8** | Registered trials; strong controlled evidence |
| `academic` | **0.6** | Peer-reviewed publications; moderate quality |
| `patent` | **0.4** | IP filings; valuable for prior art but limited empirical validity |
| `other` | **0.2** | Unclassified web results; lowest quality baseline |
| *(unknown tag)* | **0.3** | Tags absent from map receive below-`other` default |

When adding a new MCP search tool, assign an appropriate `evidence_tag` value.
Without this, new tools default to 0.3, causing their evidence to be
systematically deprioritized. See `journal-quality` skill for operational
guidance on tag assignment.

### quality_label_map -- Quality Label Score

Maps `Evidence.quality_label` to a normalised score:

| Label | Score | Notes |
|-------|:-----:|-------|
| `tier_1` | **1.0** | EvidenceTier key |
| `tier_2` | **0.75** | |
| `tier_3` | **0.50** | |
| `tier_4` | **0.25** | |
| `high` | **1.0** | Legacy label |
| `medium` | **0.50** | Legacy label |
| `low` | **0.25** | Legacy label |
| *(None / absent)* | **0.50** | `default_quality_score` |

### structural_completeness -- Field-Presence Weights

Rewards evidence carrying provenance metadata:

| Field | Weight |
|-------|:------:|
| `doi` | 0.30 |
| `url` | 0.20 |
| `clinical_trial_id` | 0.20 |
| `evidence_tag` | 0.15 |
| `quality_label` | 0.15 |

Maximum possible score: **1.0** (all fields populated).

---

## Selection Algorithm (Tier 1 & 2)

```
1. Build checklist keyword union from all checklist_items texts.
2. Compute max_round across the evidence pool.
3. Score every item with 6 additive normalised signals.
4. Sort descending, take top N items.
5. Tag safety net: force-insert top item for any tag type not yet represented.
```

Selection is purely score-based — stronger MCP sources naturally contribute
more items by scoring higher. Diversity is ensured by the tag safety net
rather than artificial per-source caps.

### Tag Safety Net

After the selection pass, checks whether every observed `evidence_tag` type has
at least one representative in the selected set:

- For each tag **absent** from the selected set, the highest-scoring item with
  that tag is **force-inserted** (appended beyond the nominal target count).
- Controlled by `CondenserConfig.enable_tag_safety_net` (default `True`).

**Purpose**: Prevents rare but high-value tag categories (e.g., `regulatory`
documents that score lower on keyword overlap) from being completely excluded.

### Tier 2 Group Summaries (Phase 1b — LLM Enrichment)

For evidence that did NOT make the primary selection, Tier 2 generates one
`GroupSummary` per `evidence_tag` type. Two summarisation modes:

**Template fallback** (when `condenser_summarizer_model` is empty):
- Lists original count vs included count
- Sample search queries that produced the excluded items
- Static transparency footer

**LLM enrichment** (when `condenser_summarizer_model` is set, e.g. `"haiku"`):
- DiscoveryLoop calls `GroupSummarizer.summarize()` for each tag group
- LLM produces a 100-300 word synthesis of key findings and contradictions
- Falls back to template text on LLM failure (graceful degradation)
- Enabled by default in DISCOVERY intensity preset (`haiku`)
- Raw excluded evidence is accessible via `CondensedEvidence.excluded_groups`

**Per-sub-item source quality override**: The `tag_quality_map` can be
overridden per sub-item via the checklist `source_priorities` field in
the orchestration layer. This adjusts scoring weights so that
context-appropriate evidence (e.g. patents for prior-art analysis,
regulatory filings for compliance assessments) ranks higher during
condensation

---

## CondenserConfig Reference

| Field | Default | Description |
|-------|---------|-------------|
| `tier1_threshold` | 150 | Max items for Tier 0 passthrough |
| `tier2_threshold` | 400 | Max items for Tier 1; above -> Tier 2 |
| `tier1_target` | 160 | Primary evidence budget for Tier 1 |
| `tier2_target` | 150 | Primary evidence budget for Tier 2 |
| `source_saturation_cap` | 20 | DEPRECATED — no longer used |
| `enable_tag_safety_net` | True | Force-insert under-represented tags |
| `default_quality_score` | 0.5 | Score when `quality_label` is None |
| `doi_prefix_default_score` | 0.0 | Score for DOIs with no journal match |
| `weight_keyword_relevance` | 0.35 | Weight for keyword relevance signal |
| `weight_source_quality` | 0.15 | Weight for source quality signal |
| `weight_quality_label` | 0.15 | Weight for quality label signal |
| `weight_journal_quality` | 0.10 | Weight for journal quality signal |
| `weight_structural_completeness` | 0.10 | Weight for structural completeness |
| `weight_round_recency` | 0.05 | Weight for round recency signal |

---

## Integration Point

```python
# EvidenceCondenser is called inside DiscoveryLoop._run_analysis():
condenser = EvidenceCondenser(CondenserConfig())
condensed = condenser.condense(all_evidence, task.checklist)
# -> condensed.evidence  (filtered list[Evidence] for AnalysisExp)
# -> condensed.meta      (CondensationMeta with tier, counts, transparency_footer)
```

The `transparency_footer` is injected into the analysis context so the
AnalysisAgent can interpret any evidence gaps caused by condensation.

---

## Available Scripts

### condense.py -- Condense evidence to a target count

```
use_skill(skill_name="evidence-condenser", action="run_script",
  script_name="condense.py",
  script_args="--input /tmp/evidence.json --checklist /tmp/checklist.json")
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--input FILE` | Path to evidence JSON array (required) |
| `--checklist FILE` | Path to checklist JSON (for keyword_relevance signal) |
| `--force-tier N` | Force Tier 1 or Tier 2 regardless of count |
| `--tier1-target N` | Override tier1_target (default: 160) |
| `--config FILE` | Custom CondenserConfig JSON for weight tuning |

**Output** (JSON):
```json
{
  "tier": 1,
  "original_count": 230,
  "condensed_count": 160,
  "transparency_footer": "230 evidence items total; 160 selected ...",
  "evidence": [...]
}
```

### preview.py -- Preview condensation without modifying state

```
use_skill(skill_name="evidence-condenser", action="run_script",
  script_name="preview.py",
  script_args="--input /tmp/evidence.json --checklist /tmp/checklist.json")
```

**Output**: Summary table of kept vs. dropped items with score breakdown.

---

## Reference Documents

Read `signal-weights.md` when tuning weights for a specific domain or when
investigating why a particular evidence item scored unexpectedly:

```
use_skill(skill_name="evidence-condenser", action="get_reference",
  reference_name="signal-weights.md")
```
