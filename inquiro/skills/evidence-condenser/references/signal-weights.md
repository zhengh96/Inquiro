# Evidence Condenser — Signal Weights Reference 📊

Detailed explanation of each scoring signal used by `EvidenceCondenser`, including
the formula, normalisation method, and tuning guidance.

**Source**: `inquiro/core/evidence_condenser.py` → `_compute_score()`

---

## Composite Score Formula

```
score = (keyword_relevance × 0.40)
      + (source_quality    × 0.20)
      + (quality_label     × 0.15)
      + (completeness      × 0.10)
      + (round_recency     × 0.05)
```

All signals are normalised to `[0, 1]` before weighting. Weights do **not** need
to sum to 1.0 — asymmetric weighting is intentional (e.g. keyword-only mode with
`weight_keyword_relevance=1.0`).

---

## Signal 1: Keyword Relevance (weight 0.40)

**Purpose**: Rewards evidence whose summary text matches the current checklist queries.

**Formula**:
```
keyword_relevance = min(|summary_kw ∩ checklist_kw| / |checklist_kw|, 1.0)
```

Where:
- `summary_kw` = lowercase words (≥3 chars) extracted from `evidence.summary`
- `checklist_kw` = union of words from all checklist item texts

**Normalisation**: fraction in `[0.0, 1.0]`; capped at 1.0 for over-matching.

**Fallback**: if `checklist_kw` is empty → score = 0.0

**Tuning guidance**:
- Increase weight when checklist coverage is critical (standard evaluation mode).
- Decrease weight when checklist is sparse or very generic.

---

## Signal 2: Source Quality (weight 0.20)

**Purpose**: Rewards evidence from higher-quality source types.

**Formula**:
```
source_quality = tag_quality_map.get(evidence.evidence_tag, 0.3)
```

**Default `tag_quality_map`**:

| `evidence_tag` | Score |
|----------------|:-----:|
| `regulatory`   | 1.00  |
| `clinical_trial` | 0.80 |
| `academic`     | 0.60  |
| `patent`       | 0.40  |
| `other`        | 0.20  |
| *(unknown)*    | 0.30  |

See `journal-quality` skill for full documentation of the tag system.

**Normalisation**: direct lookup; values are pre-normalised in the map.

**Tuning guidance**:
- Override `tag_quality_map` in `CondenserConfig` for domain-specific assessments
  (e.g., boost `patent` to 0.7 for Freedom-to-Operate analysis).

---

## Signal 3: Quality Label (weight 0.15)

**Purpose**: Rewards evidence items that have been labelled as high-quality by
upstream pipeline stages (`EvidencePipeline` quality gate).

**Formula**:
```
quality_label_score = quality_label_map.get(evidence.quality_label, default_quality_score)
```

**Default `quality_label_map`**:

| `quality_label` | Score |
|-----------------|:-----:|
| `tier_1`        | 1.00  |
| `high`          | 1.00  |
| `tier_2`        | 0.75  |
| `tier_3`        | 0.50  |
| `medium`        | 0.50  |
| `tier_4`        | 0.25  |
| `low`           | 0.25  |
| *(None)*        | 0.50  |

**Normalisation**: direct lookup; missing label uses `default_quality_score` (0.5).

**Tuning guidance**:
- Tighten when evidence pipeline quality labelling is reliable.
- Set `default_quality_score=0.0` to aggressively penalise unlabelled evidence.

---

## Signal 4: Structural Completeness (weight 0.10)

**Purpose**: Rewards evidence items that carry full provenance metadata, enabling
proper citation and verifiability in the final report.

**Formula**:
```
completeness = 0.20 × (url is not None)
             + 0.30 × (doi is not None)
             + 0.20 × (clinical_trial_id is not None)
             + 0.15 × (evidence_tag is not None)
             + 0.15 × (quality_label is not None)
```

**Normalisation**: weighted sum in `[0.0, 1.0]`.

**Field weights rationale**:
- `doi` (0.30): most reliable scholarly identifier
- `url` (0.20): citability in rendered reports
- `clinical_trial_id` (0.20): mandatory for clinical evidence
- `evidence_tag` + `quality_label` (0.15 each): pipeline metadata

**Tuning guidance**: rarely needs adjustment; reflects citation completeness.

---

## Signal 5: Round Recency (weight 0.05)

**Purpose**: Slight boost for evidence from later search rounds, which represent
gap-filling queries targeting missing checklist coverage.

**Formula**:
```
if max_round > 1:
    recency = min((round_number - 1) / (max_round - 1), 1.0)
else:
    recency = 0.0
```

**Normalisation**: linear interpolation over `[1, max_round]`.

**Rationale**: Round 1 evidence is broad; later rounds are targeted to gaps.
Gap-filling evidence is often the crucial missing piece for high-coverage synthesis.

**Tuning guidance**:
- Set `weight_round_recency=0.0` to disable recency bias (pure quality scoring).
- Increase to 0.10–0.15 when gap-filling rounds are particularly targeted.

---

## Tuning Example: FTO Assessment

For Freedom-to-Operate (FTO) assessments, patents are high-value:

```python
from inquiro.core.evidence_condenser import CondenserConfig

config = CondenserConfig(
    weight_keyword_relevance=0.35,
    weight_source_quality=0.30,  # boosted
    weight_quality_label=0.15,
    weight_structural_completeness=0.15,  # boosted (DOI/patent ID)
    weight_round_recency=0.05,
    tag_quality_map={
        "regulatory":     1.0,
        "clinical_trial": 0.6,
        "academic":       0.5,
        "patent":         0.9,   # boosted for FTO
        "other":          0.2,
    },
)
```

---

## Source Saturation Cap

After scoring, the greedy selection pass caps each MCP source at
`source_saturation_cap` items (default: 20). This prevents a single
search server (e.g., PubMed) from dominating the selection even when it
returns the highest-scoring items.

After the greedy pass, remaining budget slots are filled from saturated
sources in score order.

**Tag safety net**: if any `evidence_tag` type is completely absent from the
selected set, the highest-scoring item with that tag is force-inserted
(may exceed `tier1_target` by the number of absent tags).
