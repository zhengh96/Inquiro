---
name: query-templates
description: "Expert search query templates optimized by research topic. Provides pre-researched query sequences, tool allocation strategies, and gap-driven follow-up guidance for each evaluation dimension. Domain-specific template references are provided by the orchestration layer. Load these templates IMMEDIATELY AFTER alias-expansion and BEFORE your first search. Also use when you are unsure which search tool to use for a specific query type, when you need gap-driven follow-up strategies, or when your initial searches returned weak results and you need expert-curated alternative phrasings. Do NOT use without completing alias-expansion first -- templates contain entity placeholders that require populated alias blocks. Do NOT modify template query order without understanding the coverage impact rationale. Do NOT use for evidence quality assessment (use evidence-grader after search)."
license: Apache-2.0
---

# Query Templates

Expert-curated search query sequences for structured evidence research.

---

## Overview

Each evaluation dimension has a pre-researched query template that specifies
which search terms to use, which tools to target, and how to follow up on gaps.
Templates are designed by domain experts and ranked by expected coverage impact.

---

## Details

### How to Use

1. Identify your checklist topic from the available topic references
2. Load the matching template:
   ```
   use_skill(skill_name="query-templates", action="get_reference",
     reference_name="<topic>.md")
   ```
3. Substitute your specific entity aliases into the `{variable}` placeholders
4. Execute queries in order; follow gap-driven guidance for additional searches

### Available Topics

Topic references are provided by the orchestration layer and correspond to
the evaluation checklist dimensions. Use `list_references` to discover all
available topic templates for your current evaluation context.

> **Runtime injection**: Templates are domain-specific and stored in the domain
> layer (e.g., `targetmaster/configs/query_templates/`). They are injected at
> runtime by `TaskMapper`. If `get_reference()` returns "not found", verify the
> template file exists in the domain configuration directory.

### Reference Templates

Domain-specific templates are provided by the orchestration layer and cover all
assessment dimensions defined in the evaluation catalog. Use `list_references`
to discover which topic templates are available for your current evaluation
context -- the set varies by domain configuration.

### Template Structure

Each topic template contains:

1. **Alias Expansion** -- Pre-filled entity names and synonyms
2. **Initial Queries** -- Priority-ordered search sequences
3. **Tool Allocation Strategy** -- Which tools for which query types
4. **Follow-up Guidance** -- Gap-driven search logic
5. **Evidence Strength Tiers** -- How to assess results for this topic

### Usage Example

```
Step 1: Ensure alias-expansion is done
  Entity 1: PRIMARY-X, ALT-X1, ALT-X2
  Context: APPLICATION-Y, ABBREV-Y

Step 2: Load template
  use_skill(skill_name="query-templates", action="get_reference",
    reference_name="<topic>.md")

Step 3: Execute queries (substitute aliases into placeholders)
  Query 1 (Source A): "PRIMARY-X <topic keyword>" OR "ALT-X1 <topic keyword>"
  Query 2 (Source B): Regulatory/official records for PRIMARY-X

Step 4: Gap-driven follow-up
  Gap: Long-term data (> 2 years)?
  -> Search: "PRIMARY-X" + "long-term follow-up"
```

---

## Constraints

- Complete alias-expansion before loading templates. Templates contain
  `{entity_aliases}` and similar placeholders that produce generic, unfocused
  queries when left unfilled -- a query like `"entity treatment"` instead of
  `"PRIMARY-X OR ALT-X1 treatment"` will return mostly irrelevant results.
- Execute initial queries in the order specified. Query order reflects expected
  coverage impact; earlier queries retrieve higher-yield evidence, and later
  queries are designed to fill the gaps that earlier ones typically leave.
  Reordering without understanding this rationale can leave high-priority
  checklist items uncovered after the first few queries.
- Follow the tool allocation guidance to match search tools to query types.
  Each tool has different strengths (e.g., regulatory databases for approval
  records, academic indexers for clinical data); using the wrong tool for a
  query type yields weaker evidence.
- When a search returns no results, follow the follow-up guidance before
  concluding that evidence is unavailable. Absence of evidence at one phrasing
  does not equal evidence of absence -- the follow-up guidance captures
  alternative phrasings that domain experts know are also productive.
- Compare at least 2 sources in the same tier before drawing conclusions.
  Single-source findings are treated as low-confidence; cross-source agreement
  is what pushes confidence into the high tier.

---

## Reference Documents

Templates are loaded via `get_reference()` as described in the How to Use section.
