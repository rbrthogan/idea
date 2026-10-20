# Template authoring

Use **Browse templates → Create manually** or **Create with AI** in the app.
For a bundled template, copy one of the YAML files in this directory and edit
it. Built-ins cover Drabble, Game Design, Arcade Game Design and AI research.

A run saves its own template snapshot. Later edits do not replace those authored
instructions, although shared runtime rules apply when the run is processed again.

## The three main fields

| Field | What belongs here |
| --- | --- |
| `requirements` | Individual content constraints: feasibility, form or domain rules |
| `output` | Title/body presentation, required sections and optional body word count |
| `comparison_criteria` | Ordered quality preferences for pairwise judging |

The app owns the title/content response structure. Do not put a JSON envelope or
`Title:`/`Content:` transport labels in requirements. Word counts are model
instructions, not a guarantee that every result complies. The formatter preserves
the actual result rather than rewriting it to appear compliant.

## Minimal example

```yaml
name: Small research proposals
description: Experiments one researcher can run
version: '1.0.0'
author: Your name
created_date: '2026-10-20'
metadata:
  item_type: research proposals
requirements:
  - Use one consumer GPU and publicly available data.
  - Distinguish predictions from evidence; do not invent findings.
output:
  title: A concise research question.
  content: Include a hypothesis, method, evaluation and risks.
  word_count: {min: 400, max: 600}
prompts:
  context: 'Generate varied research concepts. Return CONCEPTS: followed by the list.'
  specific_prompt: 'Develop a testable direction from {context_pool}.'
  format: 'Preserve the original title and body from {input_text}.'
  critique: 'Assess the experimental design of {idea}.'
  refine: 'Improve {idea} using this critique: {critique}.'
  breed: 'Develop one coherent new proposal from these ingredients: {ideas}.'
  genotype_encode: |
    Identify the task, domain, method, model family and evaluation approach.
    Proposal: {idea_content}
comparison_criteria:
  - Scientific value
  - Experimental design
  - Originality
```

`requirements` defaults to `[]`. `output` may be omitted or null. Word-count
bounds are positive integers; use equal bounds for an exact requested length,
just `min` or `max` for one bound, or omit `word_count` for no requested length.
The title is excluded. Names, description, semantic version, author, date,
metadata and the seven prompt fields above are required.

## Prompt roles

| Prompt | Required variables |
| --- | --- |
| `context` | None |
| `specific_prompt` | `{context_pool}` |
| `format` | `{input_text}` |
| `critique` | `{idea}` |
| `refine` | `{idea}`, `{critique}` |
| `breed` | `{ideas}` |
| `genotype_encode` | `{idea_content}` |

Optional `oracle_constraints` and `example_idea_prompts` guide Oracle proposals.
An optional `prompts.comparison_prompt` customizes judging; the runtime still
supplies both candidates and the verdict protocol. Unknown variables are
rejected. Use doubled braces, such as `{{name}}`, for literal braces in variable
prompts. Requirements, output and context prose remain literal text.

The runtime supplies requirements/output guidance to generation, refinement and
breeding, and supplies the candidate contract to critique and comparison.
Breeding receives sampled ingredients and the child direction, not full parent
drafts. Genotype extraction receives its separate domain guidance.

## Keep genes reusable

[`gene_prompts.py`](../../gene_prompts.py) owns the shared ingredient rules used
by extraction and mutation. In `genotype_encode`, describe relevant domain traits
and include `{idea_content}`. Optionally give one small decomposition example;
do not repeat the shared rules or prescribe a fixed vocabulary.

Prefer traits that stand on their own, combining broad and distinctive concepts.
A whole plot or design compressed into one phrase is difficult to recombine.
The generation-boundary manager can merge synonyms and split bundles into shared
concepts plus meaningful remainders; see [gene records](../../../docs/GeneCatalogue.md).

## Compatibility

Save validates YAML types, word-count bounds, version/date and prompt variables.
Legacy `special_requirements`, `{requirements}` and `{requirements_digest}` remain
readable; new templates use structured fields. Existing snapshots are preserved.
The exact field schema is in [`validation.py`](../validation.py).
