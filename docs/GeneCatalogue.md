# Gene records and history

Genes are short, reusable ingredients: a genre, setting, method, mechanic,
character type or other supported trait. They should combine into new ideas,
rather than encode an entire candidate in one dense phrase.

## Extraction, consolidation, breeding

1. **Extract once per changed candidate.** Save raw genes with a content and
   prompt fingerprint. Extraction and mutation share the ingredient rules in
   [`gene_prompts.py`](../idea/gene_prompts.py); templates add domain guidance.
2. **Consolidate the committed pool.** At a changed generation boundary, send
   current gene names, normalized current genotypes and new observations.
   Historical raw genotypes and old explanations are excluded from the prompt.
3. **Apply grouped changes.** The manager returns only edits. A group supplies
   shared factors plus each source's meaningful remainder. A standalone split
   can use an empty shared list. Unmentioned genes remain unchanged.
4. **Breed from saved factors.** Sampling and mutation operate on the normalized
   genotype. Each child saves the exact parent genotypes, pool revision and
   selected ingredients used at birth.

Gene extraction and consolidation use the selected evolution model for both
Google and OpenAI runs, with the same thinking level or token budget as the core
model. These requests count toward the same run budget. There is one pass per changed
boundary, with no catalogue cap, occupancy pressure or repeated cleanup until
convergence.

Validation requires complete, nonempty mappings and valid source indexes.
Changes apply atomically; invalid output retains the prior valid pool. Structural
validation cannot prove semantic accuracy: review for unsupported traits,
overbroad merges and lost distinctions.

## Read the UI

**History** shows occupancy over generations. Hover to preview a gene, click to
pin it, and clear or press Escape to unpin. The left panel lists unique ideas
carrying that gene across history, with their first carrying generation.
Select an idea to highlight its genes; open it to inspect its text and evidence.

The right-hand charts show occupancy and mean carrier fitness. **Now** is the
current population count; **Ever** counts unique historical carriers. A selected
historical generation has its own count. Missing measurements are unavailable,
not zero, and fitness comparisons need their coverage counts.

Adjacent occupied generations connect only when a carrier survives or a child
names a previous carrier as a parent. Gaps separate unrelated occurrences.
This is evidence of candidate continuity, not proof that the gene was sampled
or caused the offspring's behavior.

History uses the **latest normalized vocabulary**. Consolidation can change
which historical ideas share a label; raw observations and actual birth inputs
remain preserved. Catalogue counts and high carrier fitness do not establish a
gene's causal benefit or prove evolution improved diversity.

## Implementation

[`genes.py`](../idea/genes.py) stores observations and catalogue entries;
[`gene_pool.py`](../idea/gene_pool.py) builds the working vocabulary, validates
edits and projects history; [`evolution_orchestrator.py`](../idea/evolution_orchestrator.py)
invokes the pipeline.

`scripts/observe_genes.py` is a separate legacy extraction/matching utility.
Its exports are useful for inspection but do not run the main engine's global
consolidation or change breeding.
