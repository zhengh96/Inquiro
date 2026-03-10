---
name: alias-expansion
description: "Expand entity names, identifiers, and related terms into comprehensive synonym lists BEFORE starting any search. Poor alias coverage is the #1 cause of missed evidence -- a single missing synonym can leave an entire evidence category undiscovered. Use this skill first, before query-templates, whenever you are about to research any entity, concept, or topic. Also use when search results seem thin or when you suspect missing synonyms, regional naming variants, or alternative identifiers. Document your alias block in working memory. Do NOT use after searches have already started (aliases must be prepared upfront). Do NOT use for post-search synonym matching (use synonym-resolver) or for abbreviation normalization of stored data (use data-normalizer). Do NOT invent aliases -- only use terms from literature or official databases."
license: Apache-2.0
---

# Alias Expansion

Systematic entity enumeration is the foundation of comprehensive evidence research.
This skill generates complete synonym lists for your research entities **before
searching**.

---

## Overview

Before any search begins, enumerate all known names, identifiers, synonyms,
and variant terms for each research entity. Incomplete alias coverage leads to missed
evidence -- search tools only find what you ask for, and a missing synonym can leave
an entire category of evidence undiscovered.

---

## How to Use

- **Before your first search**: complete this step first, as it is the foundation
  of the entire search workflow. Skipping it means every subsequent query is built
  on an incomplete term set.
- **When building the alias block**: load the expansion guide for each entity type
  and follow it systematically.
- **When search results seem thin**: revisit your alias block and check for missing
  synonyms, subtypes, or regional naming variants.

---

## Details

### Steps

1. Load the expansion guide for your entity type:
   ```
   use_skill(skill_name="alias-expansion", action="get_reference",
     reference_name="<entity_type>.md")
   ```
2. Follow the guide to enumerate aliases for your specific entities
3. Document your alias block in working memory
4. When loading query-templates, substitute aliases into the entity placeholder
   variables (e.g. `{entity_aliases}`, `{context}`, `{approach}`) defined by
   your domain configuration

### Available References

| Reference | Entity Type | When to Load |
|-----------|-------------|--------------|
| `target_expansion.md` | Primary research entity | Always |
| `indication_expansion.md` | Research context / scope | Always |
| `modality_expansion.md` | Approach / method / technique | Always |

> **Runtime injection**: These reference files are domain-specific and live in
> the orchestration layer. The `SkillService` injects them at startup via
> `INQUIRO_EXTERNAL_SKILL_REFS`, so `get_reference()` resolves them transparently
> at runtime. If `get_reference()` returns "not found", verify the env var is set
> and the source files exist.

### Three Entity Types

**Type 1: Primary Entity** -- The main subject of your research.

```
PRIMARY ENTITY: ENTITY-X

Official identifiers: ENTITY-X, ALT-ID-1, ALT-ID-2
Full names: Full Entity Name, Alternative Full Name
Variants: ENTITY-X variant A, ENTITY-X variant B
Related terms: ENTITY-X inhibitor, ENTITY-X mutation

Searchable term: (ENTITY-X OR ALT-ID-1 OR ALT-ID-2)
```

**Type 2: Context / Scope** -- The broader application domain or scope.

```
CONTEXT: Application Domain Y

Full names: full application domain name, alternative name
Subtypes: subtype A, subtype B, subtype C
Codes: standard classification codes (if applicable)

Searchable term: ("application domain Y" OR "alternative name" OR "subtype A")
```

**Type 3: Approach / Method** -- The method, technique, or modality.

```
APPROACH: Method Z

Category: method category, technique type
Variations: variation A, variation B, variant approach
Specifications: specific parameters, conditions

Searchable term: ("method Z" OR "variation A" OR "technique type")
```

### Complete Alias Block Format

```
========================================
ALIAS BLOCK: [Research Topic]
========================================

ENTITY 1: Primary Entity
  Identifiers: [official IDs]
  Full Names: [names]
  Searchable term: ([id] OR [name])

----------------------------------------

ENTITY 2: Context / Scope
  Full Name: [context name]
  Abbreviation: [abbrev]
  Subtypes: [subtypes]
  Searchable term: ([name] OR [abbrev])

----------------------------------------

ENTITY 3: Approach / Method
  Category: [category]
  Variations: [variations]
  Searchable term: ([category] OR [variation])

========================================
```

---

## Constraints

- Only use terms confirmed in literature or official databases. Invented aliases
  introduce false search terms that waste budget and pollute evidence pools with
  irrelevant results.
- Complete alias expansion before searching. Search tools only find what you ask
  for -- a single missing synonym can leave an entire evidence category undiscovered.
- Aim for 5-15 core aliases per entity. Over-expansion with marginal terms dilutes
  the quality signal and inflates search costs without proportional coverage gains.
- Include subtypes and variants in your alias block. They often have distinct
  evidence bases, and omitting them creates systematic blind spots in coverage.
- Include regional and regulatory naming variants across jurisdictions, as evidence
  from different markets may be filed under jurisdiction-specific names.

---

## Reference Documents

Entity-specific expansion guides are loaded via `get_reference()` as described
in the Steps section above.
