# Query Template — Structure & Guidelines 📋

**Template for**: [TOPIC NAME]
**Use this template for**: Creating new query templates following best practices

---

## How to Use This Template

This file shows the **standard structure** for all query templates. When creating a new template for your topic:

1. Copy this structure
2. Replace `[BRACKETS]` with your topic-specific content
3. Follow the guidelines in each section
4. Verify your queries work with the recommended tools

---

## Template Content

### Part 1: Alias Expansion 🏷️

```markdown
## Alias Expansion

Expand these entity types BEFORE searching:

**Target Aliases** ([GENE/PROTEIN NAME]):
- Official gene symbol: [e.g., EGFR, IL17A]
- Alternative gene symbols: [e.g., ERBB1, HER1]
- Full protein name: [e.g., epidermal growth factor receptor]
- Common abbreviations: [e.g., EGFRvIII, HER1]
- Isoforms/variants (if relevant): [e.g., EGFR L858R mutation]

**Indication Aliases** ([DISEASE/CONDITION]):
- Full disease name: [e.g., non-small cell lung cancer]
- Common abbreviations: [e.g., NSCLC]
- ICD-10 codes: [e.g., C34]
- Subtypes: [e.g., lung adenocarcinoma]
- Related conditions: [e.g., EGFR+ NSCLC as biomarker-defined subset]

**Modality Aliases** ([THERAPY TYPE]):
- Drug class: [e.g., monoclonal antibody]
- Specific formulations: [e.g., human antibody, chimeric antibody]
- Administration routes: [e.g., intravenous, subcutaneous]
- Related therapeutics: [e.g., recombinant antibody, antibody-drug conjugate]

**Searchable alias string**:
[RESULT]: ([alias1] OR [alias2] OR [alias3] OR "[full name]")
```

**Why this matters**: Missing aliases = missed evidence. For example, searching "EGFR" alone misses 40% of papers that use "epidermal growth factor receptor" or "ErbB-1".

---

### Part 2: Initial Queries (Priority-Ordered) 🎯

```markdown
## Initial Queries (Priority Order)

Run queries in this order. Each is ranked by expected evidence yield.

### Q1 [Priority 1] → [RECOMMENDED TOOL]

**Research Objective**: [What this query answers]
**Query Template**:
[SPECIFIC, DETAILED QUERY TEXT]

Include variable placeholders: {target}, {indication}, {modality}
Key elements to request:
- Quantitative data (rates, percentages, sample sizes)
- Specific study types (clinical trials, regulatory data)
- Comparisons (vs approved drugs, historical control)

**Key data to extract**: [What to look for in results]

---

### Q2 [Priority 1] → [RECOMMENDED TOOL]

[Same structure as Q1]

---

### Q3 [Priority 2] → [RECOMMENDED TOOL]

[Same structure, lower priority]
```

**Guidelines**:
- **Q1-Q2** should be your highest-yield searches (Priority 1)
- **Q3+** fill gaps and add depth (Priority 2-3)
- **Tool selection** should be strategic — avoid using same tool twice if possible
- **4-6 queries** is usually sufficient; more dilutes effort

---

### Part 3: Tool Allocation Strategy ⚙️

```markdown
## Tool Allocation Strategy

| Tool | % of Searches | Best For | Notes |
|------|---|---|---|
| [Tool 1] | [XX]% | [Best use case] | [Rationale] |
| [Tool 2] | [XX]% | [Best use case] | [Rationale] |
| [Tool 3] | [XX]% | [Best use case] | [Rationale] |
| **Total** | **100%** | | |
```

**Guidelines**:
- Percentages must sum to 100%
- Aim for ≥ 2 tools (single-tool bias is a quality risk)
- Primary tool should be the one that best covers your topic type

---

### Part 4: Follow-up Guidance 🔄

```markdown
## Follow-up Guidance (Gap-Driven Search Logic)

### IF [gap condition 1] (e.g., "< 5 papers on biomarkers")
→ Search for: "[follow-up query]" on [TOOL]

### IF [gap condition 2]
→ Search for: "[follow-up query]" on [TOOL]

### Coverage Checklist
- [ ] Have you covered [requirement 1]?
- [ ] Have you covered [requirement 2]?
- [ ] Have you found data from [geography/time period]?
```

**Guidelines**:
- List 3-5 common gaps relevant to your topic
- For each gap, specify the exact follow-up query
- Include a coverage checklist with 5-7 must-have items

---

### Part 5: Evidence Strength Tiers 📊

```markdown
## Evidence Strength Tiers

**Tier 1 — Highest Reliability** ⭐⭐⭐⭐⭐
- [Specific criteria for this topic, e.g., Phase 3 RCT with N ≥ 300]
- Confidence if single source: 0.X–0.Y
- Confidence if multiple sources agree: 0.X–0.Y

**Tier 2 — Strong Secondary Sources** ⭐⭐⭐⭐
- [Specific criteria, e.g., Phase 2 data, industry market report]
- Confidence if single source: 0.X–0.Y

**Tier 3 — Supplementary Sources** ⭐⭐⭐
- [Specific criteria, e.g., conference abstract, company press release]

**Tier 4 — Weak/Unverifiable** ⚠️
- [Specific criteria, e.g., blog posts, unverified claims]
```

**Guidelines**:
- Define each tier for YOUR SPECIFIC TOPIC — market data tiers differ from clinical data tiers
- Give concrete examples so users know exactly what qualifies
- Align confidence ranges with the Evidence Grader framework

---

## Best Practices for Query Templates ✨

### ✅ DO

1. **Be specific**: "IL-17A monoclonal antibody immunogenicity" not "immunogenicity"
2. **Include quantifiable data requests**: "report sample size", "report rates", "include 95% CI"
3. **Specify tool rationale**: Explain why PubMed for this, Patents for that
4. **Prioritize ruthlessly**: Identify your TOP 2-3 queries; rest are supplementary
5. **Test your queries**: Run them before finalizing to verify they work
6. **Design for different results**: Plan follow-up for both "too many" and "too few" results

### ❌ DON'T

1. **Use generic terms**: "efficacy" alone won't work; use "[drug name] efficacy"
2. **Assume one tool covers everything**: Combine PubMed (literature) + Patents (IP) + Trials (data)
3. **Create too many queries**: 4-6 is usually enough; more dilutes effort
4. **Make queries tool-independent**: Different tools have different syntax — optimize per tool

---

## Template Completion Checklist ✅

Before finalizing your template:

- [ ] Alias expansion covers all major synonyms and variants
- [ ] Q1-Q2 are high-priority, well-researched queries
- [ ] Tool allocation sums to 100% and is justified
- [ ] Follow-up guidance covers 3-5 likely gaps
- [ ] Tier 1 defined with concrete, topic-specific criteria
- [ ] All variable placeholders are documented
- [ ] Queries have been tested/verified to work
- [ ] Template matches structure of other topic templates (see immunogenicity.md as example)
