---
name: evidence-source-classification
description: "URL-based evidence source classification rules and noise filtering patterns for EvidencePipeline. Maps evidence URLs to 5 categories (academic, patent, clinical_trial, regulatory, other) using regex patterns from a declarative YAML registry. Use when adding new source domains to the classification system, tuning noise patterns from MCP responses, understanding how evidence URLs are mapped to EvidenceTag categories, debugging why an evidence item was tagged incorrectly, or filtering garbage output from new MCP tools. Also use when evidence items are being silently dropped and you suspect noise pattern false positives. All authoritative pattern data lives in references/patterns.yaml -- edit that file to extend coverage without touching Python code. Do NOT use for evidence quality scoring or condensation (use evidence-condenser). Do NOT use for evidence quality grading (use evidence-grader). Do NOT use for tag-to-quality-score mapping (use journal-quality)."
license: Apache-2.0
---

# Evidence Source Classification

A declarative pattern registry that drives URL-based evidence source tagging and
noise filtering inside `EvidencePipeline`.

---

## Overview

`EvidencePipeline` tags each evidence item with an `EvidenceTag` derived solely
from its URL. The classification algorithm iterates through ordered regex rules
and returns the first match. All pattern data lives in `references/patterns.yaml`
so coverage can be extended without modifying Python source.

The same YAML file also holds the `noise_patterns` list -- plain-text substrings
that identify garbage output from MCP servers.

Downstream consumers: `journal-quality` and `evidence-condenser` use the assigned
tags for quality scoring.

---

## Details

### Five Evidence Categories

| EvidenceTag | Meaning | Primary Sources |
|---|---|---|
| `academic` | Peer-reviewed journals, pre-prints, scholarly indexers | PubMed, PMC, DOI, Nature, Science, Cell, Lancet, NEJM, JAMA, bioRxiv, medRxiv, Google Scholar, NCBI Books |
| `patent` | Patent offices and aggregators | Google Patents, USPTO, EPO, WIPO PatentScope, Lens.org |
| `clinical_trial` | Publicly registered study databases | ClinicalTrials.gov, CenterWatch, WHO ICTRP, ISRCTN, ANZCTR, EU Clinical Trials |
| `regulatory` | Government authority documents and filings | Major regulatory agencies (e.g., national health, safety, environmental authorities) |
| `other` | Everything not matched above | Falls back when no regex matches |

### How URL Pattern Matching Works

1. `EvidencePipeline` loads patterns at module import time from
   `references/patterns.yaml` (falls back to hardcoded defaults if missing).
2. `_compile_tag_rules()` compiles each pattern string into `re.Pattern` with
   `re.IGNORECASE`.
3. Rules are tested in order: **academic -> patent -> clinical_trial -> regulatory**.
   First match wins; unmatched URLs receive `EvidenceTag.OTHER`.

Order matters: a URL matching multiple categories is classified by whichever
category appears first. In practice the categories are disjoint enough that
ordering is rarely decisive.

### How Noise Filtering Works

Noise patterns are plain substrings (not regex):

```python
if pattern.lower() in summary.lower():
    # discard evidence item
```

Categories of known noise:
- **MCP session markers** -- "AI Search Session Created", "Session ID:"
- **Search count announcements** -- "Found **", announcement patterns
- **Empty-result messages** -- "No papers found", "No results found for"
- **Error strings** -- "Error:", "API error:", "MCP error"
- **Network failures** -- "Connection timed out", "Rate limit exceeded"

Evidence items are also filtered if summary < `MIN_EVIDENCE_LENGTH` (default
50 characters).

### How to Extend Patterns

**Adding a new academic domain**: append to
`source_classification.academic.url_patterns` in `references/patterns.yaml`:

```yaml
- "newjournal\\.org"
```

Dots in domain names must be escaped as `\\.`. All patterns use `re.IGNORECASE`.

**Adding a new noise pattern**: append a plain substring to `noise_patterns`:

```yaml
- "Search quota exceeded"
```

**Adding a new top-level category**:

1. Add a new key under `source_classification` in `patterns.yaml`.
2. Add the corresponding `EvidenceTag` value to `inquiro/core/types.py`.
3. Add the loop in `_compile_tag_rules()` in `evidence_pipeline.py`.

---

## Constraints

- Keep pattern data domain-agnostic at this Inquiro layer -- domain-specific URL
  patterns belong in the TargetMaster configuration layer. Mixing domain terms
  into Inquiro patterns breaks the architecture boundary and requires Inquiro
  changes whenever a new domain is added.
- Use plain substrings in `noise_patterns`, not regex. Noise patterns are
  substring-matched for performance; regex in the noise list would require
  recompiling patterns on every evidence item and would change behavior silently
  if a partial regex match occurs.
- Understand the first-match-wins semantics before reordering categories. The
  current order (academic -> patent -> clinical_trial -> regulatory) reflects
  frequency of occurrence, not priority. Changing the order without auditing
  edge cases can silently misclassify URLs that partially match multiple patterns.

---

## Reference Documents

All authoritative pattern data:

```
references/patterns.yaml
```
