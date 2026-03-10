# Evidence Grading Criteria — Detailed Reference 📖

Complete reference guide for assessing evidence reliability, completeness, and cross-source validation. Use in conjunction with the main Evidence Grader SKILL.md.

---

## 1. Source Reliability Assessment 🏢

### Publication Quality Indicators

**Impact Factor & Journal Ranking**:
- ✅ **IF > 10** — High-reliability journals (Nature, Science, Lancet, JAMA, etc.)
- ⚠️ **IF 5–10** — Strong journals (most major medical/pharma publications)
- ⚠️ **IF 1–5** — Reputable but lower-impact journals
- ❌ **IF < 1 or unknown** — Lower reliability unless peer-reviewed by recognized body

**Peer Review Status** (Hierarchy):
1. ✅ Double-blind peer review (reviewers and authors unknown to each other) — HIGHEST
2. ✅ Single-blind peer review (reviewers anonymous, authors known) — HIGH
3. ⚠️ Editorial review (editor selection, no peer review) — MODERATE
4. ❌ No formal review (blog, social media, unvetted sources) — LOWEST

**Retraction & Correction History**:
- ✅ **No retractions or corrections** — Confidence boost
- ⚠️ **Minor correction (typographical error, minor data fix)** — Acceptable, note if it affects key findings
- ⚠️ **Expression of concern (editor questioning validity)** — Use with caution
- ❌ **Retraction (paper withdrawn due to error or misconduct)** — DO NOT cite; find alternative source

**Conflict of Interest Assessment**:
- ✅ **No apparent COI; funding from government/non-profit** — Low risk
- ⚠️ **Funding from industry but results are negative/neutral** — Acceptable
- ⚠️ **Industry funding AND positive results, but methodology is rigorous** — Acceptable with caution
- ❌ **Industry funding AND positive results AND methodology questionable** — High risk; require independent corroboration

---

### Data Quality Indicators 📊

**Sample Size Benchmarks** (by study type):

| Study Type | Tier 1 (Strong) | Tier 2 (Acceptable) | Tier 3+ (Weak) |
|---|---|---|---|
| **Epidemiology** | N ≥ 10,000 | N 1,000–10,000 | N < 1,000 |
| **Clinical trial** | N ≥ 500 | N 100–500 | N < 100 |
| **Cohort study** | N ≥ 1,000 | N 200–1,000 | N < 200 |
| **Case-control** | N ≥ 500 | N 100–500 | N < 100 |
| **Case report/series** | N/A | N/A | N < 30 |
| **In vitro study** | N ≥ 3 replicates | N 2 replicates | N = 1 replicate |

**Study Design Hierarchy** (Strongest → Weakest):

1. ✅ **Randomized Controlled Trial (RCT)** — Gold standard, prospective assignment
2. ✅ **Prospective cohort study** — Follows subjects forward in time
3. ⚠️ **Retrospective cohort study** — Uses existing data, prone to bias
4. ⚠️ **Case-control study** — Smaller sample, specific to outcome
5. ⚠️ **Cross-sectional study** — Snapshot in time, no temporal direction
6. ❌ **Case series/report** — Anecdotal, smallest evidence
7. ❌ **In silico/computational** — Predictions only, no experimental validation

**Statistical Rigor Checklist**:
- [ ] **Pre-specified primary endpoint** — Defined BEFORE data collection
- [ ] **Multiplicity correction** — Bonferroni, FDR, or similar for multiple comparisons
- [ ] **95% confidence intervals** — Not just p-values
- [ ] **Power calculation** — Sample size justified by statistics
- [ ] **Blinding** — Double-blind preferred; open-label acceptable if clear justification
- [ ] **Intention-to-treat analysis** — Maintains randomization benefits
- [ ] **Conflict of interest disclosure** — All authors and funding disclosed

**Reproducibility & Generalizability**:
- ✅ **Findings replicated across ≥2 independent studies** — Strong corroboration
- ✅ **Findings hold across diverse populations** — Geographic, ethnic, demographic diversity
- ✅ **Findings stable across time periods** — Not just one time point
- ⚠️ **Single study; others ongoing or conflicting** — Provisional
- ❌ **No replication attempts; contradicted by other work** — Weak

---

## 2. Evidence Completeness Checklist ✅

For a topic to be considered **"well-evidenced"** (confidence ≥ 0.80):

- [ ] **≥2 independent Tier 1 sources** — Single source insufficient for high confidence
- [ ] **Covers the specific context** — Not just analogous target/indication
  - Example: ❌ "Biologic immunogenicity" is NOT equivalent to ✅ "XAb-123 immunogenicity"
- [ ] **Quantitative data available** — Not just qualitative assessments
  - Example: ✅ "60% ± 5% response rate (95% CI)" vs ❌ "good response rates"
- [ ] **Temporal coverage: recent data** — Typically data from ≤5 years
  - Exception: Foundational studies (even if older) acceptable if no newer data contradicts
- [ ] **Geographic/regulatory scope matches** — If analyzing US market, US-specific data preferred
  - Example: FDA approval data for US indications; EMA for Europe

**Common Completeness Gaps**:

| Gap Type | Example | Severity | Fix |
|---|---|---|---|
| **No mechanism data** | Efficacy known but MOA unclear | Medium | Search for "mechanism of action" + target |
| **Only clinical data; no preclinical** | Phase 3 efficacy but no target engagement | Low–Medium | Search for "target engagement" + biomarker studies |
| **Geographic mismatch** | EU data only; need China analysis | High | Search for China-specific trial or regulatory data |
| **Outdated evidence** | 2015 paper; field evolved significantly | High | Search for "recent", "updated", "2023/2024" data |
| **Single-source dependency** | Only one trial reports this endpoint | Medium | Search for "independent validation", similar studies |

---

## 3. Gap Classification 🎯

### Critical Gaps (MUST address before finalizing) 🔴

- **No evidence from any tier** on a required checklist item
  - Example: Checklist requires "biomarker data" but zero publications or trials report biomarkers
  - Action: Run targeted searches; if still nothing, flag as unmet research need

- **Only contradictory evidence with no resolution**
  - Example: Study A reports 60% efficacy; Study B reports 30%; impossible to reconcile
  - Action: Investigate WHY they differ (population? endpoint? methodology?); document conflict with analysis

- **Key quantitative metric completely missing**
  - Example: No reported response rate, safety rate, market size, or other critical metric
  - Action: Search specifically for this metric; if not published, note as "not publicly disclosed"

### Moderate Gaps (SHOULD address if possible) 🟡

- **Only Tier 3 evidence on an important topic**
  - Example: Only blog posts and company press releases on a safety concern
  - Action: Try to find Tier 1/2 (regulatory, clinical) data; if not available, note limitation

- **Outdated evidence (> 5 years) without recent corroboration**
  - Example: Last study on drug efficacy was 2018; nothing since
  - Action: Search for "recent updates", conference presentations, ongoing trials

- **Evidence from different geographic context**
  - Example: Only Japanese trial data; analysis is for US market
  - Action: Search for US/Western data; if unavailable, note applicability limitations

### Minor Gaps (MAY note but acceptable) 🟢

- **Missing optional checklist items** — Non-mandatory elements
- **Lack of Tier 1 when Tier 2 is consistent** — If multiple Tier 2 sources agree strongly
- **Forecast data unavailable but historical trends clear** — Can extrapolate with caveats
- **Expert opinion missing** — Nice-to-have but not required

---

## 4. Cross-Source Validation (Summary) 🔀

When multiple sources cover the same finding, assess true independence before counting as corroboration.

**Sources are NOT independent if**:
- Study B cites Study A with the same data (circular)
- Same authors publishing same data in different journals
- Post-hoc analysis of the same trial

**Sources ARE independent if** they use different patient cohorts, different time periods, and different methodologies — even if they reach the same conclusion.

For detailed contradiction resolution examples and the full synthesis framework, load:
```
use_skill(skill_name="cross-reference-rules", action="get_reference",
  reference_name="synthesis-examples.md")
```

---

## 5. Quick Reference: Evidence Strength by Domain 🎓

### Pharmaceutical Efficacy Data

| Evidence Type | Tier | Confidence |
|---|---|---|
| Peer-reviewed Phase 3 RCT (N ≥ 300) | Tier 1 | 0.85–0.95 |
| FDA approval letter with clinical data | Tier 1 | 0.85–0.95 |
| Meta-analysis of multiple RCTs | Tier 1 | 0.85–0.95 |
| Phase 2 RCT (N 100–300) | Tier 1/2 border | 0.75–0.85 |
| Company press release on trial results | Tier 3 | 0.40–0.60 |
| Conference abstract (peer-reviewed) | Tier 2/3 border | 0.50–0.70 |

### Safety & Immunogenicity Data

| Evidence Type | Tier | Confidence |
|---|---|---|
| Regulatory safety review (FDA/EMA) | Tier 1 | 0.85–0.95 |
| Peer-reviewed safety follow-up study | Tier 1 | 0.75–0.85 |
| Pharmacovigilance database (e.g., VAERS) | Tier 2 | 0.60–0.75 |
| Spontaneous adverse event reports | Tier 3 | 0.30–0.50 |
| Company safety statement | Tier 3 | 0.40–0.60 |

### Market & Competitive Data

| Evidence Type | Tier | Confidence |
|---|---|---|
| IQVIA/GlobalData pharma market report | Tier 2 | 0.65–0.75 |
| Company investor presentation | Tier 3 | 0.40–0.60 |
| News article from STAT News | Tier 2/3 | 0.50–0.70 |
| Industry analyst forecast | Tier 2/3 | 0.50–0.70 |

---

## 6. Using This Reference 📚

**When grading evidence, follow this sequence**:

1. **Identify source type** → Section 5 "Evidence Strength by Domain" table
2. **Check publication quality** → Section 1 (Impact Factor, peer review, COI)
3. **Verify data quality** → Section 1 (sample size, study design, statistical rigor)
4. **Classify gaps** → Section 3 (critical / moderate / minor)
5. **Cross-validate across sources** → Section 4 pointer → `synthesis-examples.md` in cross-reference-rules
6. **Assign confidence** → Section 5 table ranges + SKILL.md calibration rules

**Red flags that warrant deeper investigation**:
- ⚠️ Single-source evidence on critical topic
- ⚠️ All evidence > 5 years old with no recent updates
- ⚠️ Evidence from only one tool/database (lack of diversity)
- ⚠️ Contradictory findings with no clear winner
- ⚠️ Industry-funded studies with uniformly positive results
- ⚠️ No quantitative data; only qualitative assessments

---

**Back to main SKILL.md**: Use Evidence Grader SKILL.md for the hands-on framework; use this reference for detailed criteria and decision-making.
