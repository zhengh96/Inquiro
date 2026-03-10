---
name: evidence-grader
description: "Unified framework for evidence quality grading, cross-source conflict resolution, and multi-report synthesis. Covers Tier 1-4 evidence strength classification, confidence calibration ranges, corroboration and contradiction detection, 5-step conflict resolution, and synthesis decision logic. Use BEFORE assigning any confidence value, DURING synthesis when reconciling findings from multiple reports, WHEN resolving conflicting sources, WHEN justifying why certain evidence gaps are acceptable, or WHEN deciding which source to trust when two studies disagree. Also use when you need to explain a confidence score, determine whether sources are truly independent, or assess whether a finding has sufficient corroboration. Do NOT use for pipeline-specific per-Agent confidence calibration (use confidence-calibration-rules). Do NOT use for evidence selection or condensation scoring (use evidence-condenser). Do NOT use for pre-Finish output validation (use quality-gate)."
license: Apache-2.0
---

# Evidence Grader

A unified framework for evaluating evidence reliability, resolving conflicts
between sources, and synthesizing findings into coherent conclusions. This is
the single source of truth for evidence quality assessment in the pipeline.

---

## Evidence Strength Tiers

Tiers are **hierarchical**: higher tiers supersede lower ones when conflicts arise.

### Tier 1 -- Primary / Authoritative Sources

**Definition**: Original research or official regulatory determinations.

- Peer-reviewed publications in indexed journals
- Official regulatory filings, approvals, and government records
- Registered study data from authoritative registries
- Government statistical databases and national registries
- Systematic reviews and meta-analyses from high-impact sources

**Confidence if single source**: 0.75-0.85
**Confidence if multiple sources agree**: 0.85-0.95

### Tier 2 -- Established Secondary Sources

**Definition**: Curated, professionally-authored analysis by recognized institutions.

- Industry market reports from recognized research firms
- Peer-reviewed conference presentations at major domain conferences
- Patent filings with examiner analysis (USPTO, EPO, WIPO)
- Official guidance documents from regulatory authorities

**Confidence if single source**: 0.65-0.75
**Confidence if supported by Tier 1**: 0.70-0.85

### Tier 3 -- Supplementary Sources

**Definition**: Preliminary, expert-driven, or partially-validated information.

- Pre-prints without peer review
- News articles from credible domain-specific outlets
- Expert opinion pieces and editorials (recognized field leaders, disclosed affiliations)
- Company press releases and investor presentations
- In silico predictions and computational analyses (needs experimental validation)

**Confidence if single source**: 0.35-0.50
**Confidence if supported by Tier 1/2**: 0.50-0.70

### Tier 4 -- Weak / Unverifiable Sources

**Definition**: Highly speculative, anecdotal, or biased sources -- use only for context.

- Blog posts and social media
- Unattributed claims or anonymous sources
- Outdated publications (> 10 years old without recent corroboration)
- Single anecdotal case reports (n = 1)

**Confidence**: 0.10-0.30

---

## Confidence Calibration Ranges

| Evidence Pattern | Confidence Range |
|---|---|
| Multiple Tier 1 sources agree | 0.85-0.95 |
| Single Tier 1 + Tier 2 corroboration | 0.70-0.85 |
| Only Tier 2 sources available | 0.55-0.70 |
| Mixed signals (Tier 1 disagree) | 0.40-0.60 |
| Only Tier 3 sources, no Tier 1/2 | 0.30-0.50 |
| Single weak source, no corroboration | 0.15-0.35 |
| No evidence found | 0.05-0.15 |

For pipeline-specific per-Agent calibration protocols (SearchAgent adaptive
mode, AnalysisAgent protocol, SynthesisAgent 4-factor assessment), load
`confidence-calibration-rules`.

**Example**:
```
Input: Two independent Tier 1 studies (different cohorts) both report
       10-15% event rate. One Tier 2 market report corroborates the finding.

Grading:
  - Two independent Tier 1 sources agree -> corroboration confirmed
  - Tier 2 adds further support; no conflicting signals
  -> Confidence: 0.87 (high end of "Single Tier 1 + Tier 2 corroboration"
     upgraded by multi-source Tier 1 agreement)
  -> Tier assignment: Tier 1 (primary sources are authoritative)
```

---

## Cross-Reference Rules

When reconciling findings from multiple reports, apply these rules systematically.

### Core Rules

1. **Corroboration check**: When the same finding appears in multiple independent
   reports (different cohorts, methodologies, time periods), note it as
   cross-validated and increase confidence.

2. **Contradiction detection**: When reports disagree, document both sides
   explicitly -- which reports disagree, the strength of evidence on each side,
   and your assessment of which is more reliable and why.

3. **Gap identification**: Note topics covered by some reports but not others.
   Classify gaps as critical (must address before finishing) or acceptable
   (document and continue).

4. **Evidence chain tracking**: Attribute every claim to a specific report ID
   (R1, R2, etc.). Check citation chains -- if R3 cites R1 with the same data,
   they are NOT independent.

---

## Conflict Resolution (5 Steps)

When evidence sources **disagree**, resolve conflicts systematically:

1. **Weight by tier** -- Tier 1 sources override all lower tiers; if Tier 1
   sources disagree, investigate why
2. **Weight by recency** -- Within the same tier, prefer more recent data;
   exception for foundational studies
3. **Weight by specificity** -- Direct evidence on exact question > indirect
   analogies or surrogate populations
4. **Weight by rigor / sample size** -- Larger multi-center studies > smaller
   single-center; empirical > computational
5. **Document the conflict** -- Always note both sides and your resolution
   rationale before drawing conclusions

**Example**:
```
CONFLICT: Study A (8% event rate, Method X+confirmatory, Population A)
          Study B (28% event rate, Method X only, Population B)

Root cause: Detection method AND study population differ.
Resolution: 8% more reliable (confirmatory method reduces false positives).
            28% is upper bound under less stringent detection.
Confidence: 0.85 for 8% as the more rigorous estimate.
```

---

## Synthesis Decision Framework

When synthesizing across multiple reports:

| Pattern | Confidence | Action |
|---------|-----------|--------|
| Strong consensus (3+ independent reports agree) | 0.85-0.95 | Proceed with high confidence |
| Moderate consensus (2 reports agree, none contradict) | 0.70-0.80 | Note consensus strength |
| Mixed signals (reports explicitly disagree) | 0.40-0.60 | Flag for deep-dive |
| Single source (only 1 report covers the topic) | 0.30-0.50 | Note as single-source; increase uncertainty |

---

## Quality Grading Checklist

Before assigning confidence, verify:

- [ ] **Source identified clearly** (author, journal, date, link if available)
- [ ] **Tier assigned with rationale** (why this tier, not a different one?)
- [ ] **Recency appropriate** (is this data still current?)
- [ ] **Sample size or scope noted** (N, population, geographic region)
- [ ] **Conflict resolution documented** (if multiple sources exist)
- [ ] **Confidence justified** (range + reasoning)

---

## Constraints

- Do not assign confidence >= 0.80 when any required checklist item lacks
  primary-source coverage -- uncovered required items signal a genuine gap that
  downstream scoring will penalize, and overstating confidence here misleads
  the synthesis stage.
- Do not treat non-independent sources (shared citation chain) as corroboration --
  two papers citing the same underlying dataset do not constitute independent
  replication and inflate confidence without adding real epistemic weight.
- Document every conflict, even minor ones -- undocumented conflicts surface later
  as unexplained confidence divergence between models and require costly re-analysis
  to trace.
- Tie confidence to specific coverage counts and evidence patterns rather than
  using round numbers like 0.75 -- round numbers suggest estimation rather than
  calibrated assessment and cannot be audited against the evidence record.

---

## Reference Documents

For detailed grading criteria, publication quality indicators, and sample size
benchmarks:

```
use_skill(skill_name="evidence-grader", action="get_reference",
  reference_name="grading-criteria.md")
```

For worked examples of corroboration, contradiction resolution, gap analysis,
and synthesis:

```
use_skill(skill_name="evidence-grader", action="get_reference",
  reference_name="synthesis-examples.md")
```
