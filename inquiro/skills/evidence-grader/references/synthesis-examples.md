# Cross-Reference Synthesis Examples — Reference 🔀

Worked examples for corroborating findings, resolving contradictions, and synthesizing evidence across multiple research reports.

---

## Contents

1. [Corroboration — When Reports Agree](#1-corroboration--when-reports-agree)
2. [Contradiction Resolution](#2-contradiction-resolution)
3. [Gap Identification](#3-gap-identification)
4. [Evidence Chain & Attribution](#4-evidence-chain--attribution)
5. [Synthesis Decision Framework](#5-synthesis-decision-framework)
6. [Confidence Calibration Table](#6-confidence-calibration-table)

---

## 1. Corroboration — When Reports Agree ✓

### Strong Consensus (3+ reports)

**Scenario**: EGFR mutation prevalence in NSCLC

| Source | Finding | Tier |
|--------|---------|------|
| PubMed meta-analysis (N=8,500) | 40–50% in Asian patients | Tier 1 |
| ClinicalTrials.gov (N=320) | 45% in Chinese cohort | Tier 1 |
| FDA drug label | ~45% in unselected population | Tier 1 |

**Synthesis**:
```
Consensus: ~43–45% prevalence
Confidence: 0.90+
Reason: 3 independent Tier 1 sources, different methodologies, same result
Attribution: R1 (academic), R2 (clinical), R3 (regulatory)
```

Use 43–45% as primary estimate. If a single number is needed, use the FDA label figure (45%).

---

### Moderate Consensus (2 reports, mixed tiers)

**Scenario**: BRAF inhibitor response rate in V600E melanoma

| Source | Finding | Tier |
|--------|---------|------|
| NEJM Phase 3 RCT (N=675) | 50% ORR (CI 47–53%) | Tier 1 |
| Company investor presentation | "~50% response" | Tier 3 |
| Analyst report | "likely 60–65%" | Tier 4 |

**Synthesis**:
```
Primary finding: 50% (R1, Tier 1)
Supporting: R2 consistent but lower quality
Caveat: R3 speculation, not clinical data
Confidence: 0.75–0.80
```

Use 50% (CI 47–53%) from R1. Note R3's higher claim is market speculation, not evidence.

---

## 2. Contradiction Resolution 🚨

### Pattern: Same metric, different numbers → investigate methodology

**Scenario**: Immunogenicity risk for IL-17A inhibitor

| Source | ADA Rate | Assay | Route | Tier |
|--------|---------|-------|-------|------|
| Secukinumab Phase 3 (Lancet, N=500) | 8% (CI 5–11%) | ELISA + confirmatory | IV | Tier 1 |
| Ixekizumab Phase 3 (N=480) | 28% (CI 24–32%) | ELISA only | SC | Tier 1 |

**Root cause analysis**:

| Factor | Impact |
|--------|--------|
| Assay method: ELISA+confirmatory vs ELISA only | HIGH — confirmatory reduces false positives |
| Route: IV vs SC | MEDIUM — SC injections generally more immunogenic |
| Timing: Week 8 vs Week 12 | MEDIUM — later = more ADA accumulation |

**Resolution**:
```
True immunogenic ADA: ~8–10% (R1 — more rigorous assay, use as primary)
Total ADA including low-affinity: ~25–30% (R2 — upper bound, note assay caveat)

Confidence in primary (8–10%): 0.85
Confidence in upper bound (25–30%): 0.65 (ELISA-only prone to false positives)
```

State both: "~8% clinically relevant ADA; up to 28% by less stringent assay."

---

### Pattern: Different scopes, not a real contradiction

**Scenario**: EGFR mutation prevalence

| Source | Finding | Scope |
|--------|---------|-------|
| Global meta-analysis (N=50+ trials) | 40–50% | All ethnicities |
| China registry (N=5,000) | 65% | Chinese patients only |

**Resolution**:
```
Not a contradiction — different populations:
  Global average: 40–50% (R1 correct for world market)
  China-specific: 65% (R2 correct for Chinese patients)
  Asia overall: ~45–65%

Both reports are correct. Clarify scope before citing.
```

---

## 3. Gap Identification 📋

**Scenario**: ALK+ NSCLC treatment

| Topic | Crizotinib trial | Alectinib trial | FDA letters | Coverage |
|-------|-----------------|-----------------|-------------|---------|
| Clinical efficacy | ✅ | ✅ | ⚠️ | ✅ COVERED |
| Adverse events | ⚠️ | ⚠️ | ✅ | ✅ COVERED |
| Long-term safety | ❌ | ❌ | ❌ | 🚨 MISSING |
| Resistance mechanisms | ❌ | ❌ | ❌ | 🚨 MISSING |
| CNS efficacy | ❌ | ✅ | ❌ | ⚠️ SPARSE |

**Actions**:
- Long-term safety → search patient registries and long-term follow-up publications
- Resistance → search "ALK inhibitor resistance mechanisms"
- CNS → search for head-to-head CNS efficacy data

Proceed with efficacy/safety analysis; explicitly flag the two missing topics.

---

## 4. Evidence Chain & Attribution 🔗

Good attribution explains the reasoning chain, not just the conclusion.

**❌ Wrong**:
> "EGFR inhibitors are ineffective in ROS1+ disease."

**✅ Better**:
> "EGFR inhibitors show limited efficacy in ROS1+ NSCLC (R3: 8% response in PROFILE 1001), likely due to kinase domain differences between EGFR and ROS1 (R1, R2). ROS1-specific inhibitors achieve 60–70% response (R4), confirming target selectivity."

**What makes this good**:
- Every claim has a report ID
- Mechanism (R1, R2) explains the clinical pattern (R3 vs R4)
- Comparative evidence shows why target matters

---

## 5. Synthesis Decision Framework 🎯

```
Synthesizing a finding:
  │
  ├─ ≥3 reports agree?
  │   YES → Strong consensus → Confidence 0.85–0.95
  │
  ├─ 2 reports agree, none contradict?
  │   YES → Moderate consensus → Confidence 0.70–0.80
  │
  ├─ Reports disagree?
  │   YES → Analyze root cause
  │     ├─ Explainable (different assay/population/timing)?
  │     │   YES → Conditional → Confidence 0.50–0.70; state conditions
  │     └─ Unexplainable true conflict?
  │         YES → Low → Confidence 0.30–0.50; flag for expert review
  │
  ├─ Only 1 report?
  │   YES → Single-source → Confidence 0.40–0.65 (depends on tier)
  │
  └─ No reports cover topic?
      → Evidence gap → flag as "not publicly available" or "requires additional research"
```

---

## 6. Confidence Calibration Table 🎲

| Evidence Pattern | Confidence Range |
|-----------------|-----------------|
| 3+ Tier 1 sources agree | 0.85–0.95 |
| 2 Tier 1 + 1 Tier 2 agree | 0.80–0.90 |
| 2 Tier 1 sources agree | 0.75–0.85 |
| 1 Tier 1 + supportive Tier 2 | 0.70–0.80 |
| Only Tier 2 sources | 0.55–0.70 |
| Contradiction explained by methodology | 0.50–0.70 |
| Unexplained contradiction | 0.30–0.50 |
| Single source, no corroboration | 0.40–0.65 |

These ranges align with the evidence-grader skill confidence rules.
