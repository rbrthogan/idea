# How evolution works

A run starts with generated ideas and evolves them over several generations.
Templates supply the domain, requirements and judging criteria; the engine
supplies the shared generation, evaluation and breeding machinery.

## The loop

1. **Generate and refine** the initial population from varied context seeds.
2. **Evaluate** candidates with pairwise LLM comparisons and embedding
   diversity. Comparisons update Elo; diversity measures distance from nearby
   peers using summaries of the judging criteria.
3. **Select** survivors and parents using combined quality/diversity fitness.
   Age decay affects survival. Islands evolve separately with migration.
4. **Create replacements** through independent births, single-parent variants
   or crossover. Breeding samples saved ingredients; mutation replaces some
   ingredients with new ones. An optional Oracle proposes alternatives.
5. **Commit the full generation**, extract new genes and consolidate the pool
   for the next breeding round. Save evidence, lineage, usage and progress.

Defaults are two islands, 50% replacement, 30% mutation and a birth mix of
50% independent / 25% single-parent / 25% crossover. Very small populations
reduce the island count so breeding remains possible. Saved run configuration
is the authority for an individual experiment.

Default fitness combines normalized Elo and diversity:

```text
fitness = alpha × quality + (1 − alpha) × diversity
```

The default `alpha` is 0.7. Selection is probabilistic; a below-average idea can
survive or reproduce. A gene's carrier fitness is an association, not an
estimate of its causal value. Gene occupancy itself does not affect selection.
See [evaluator contracts](PluggableEvaluators.md) for custom metrics and failures.

## Genes

Each candidate keeps its raw extraction and a normalized genotype. The manager
receives the current committed vocabulary, current normalized candidates and
new observations, then returns grouped changes only. It can merge synonyms or
split bundles into shared concepts and meaningful remainders.

There is one pass per changed population boundary, shared across islands.
Old raw observations and merge explanations stay local to the saved record;
they are not replayed into every manager prompt. Breeding uses saved factors
and records its exact parent inputs. Invalid consolidation leaves the last
valid representation available.

The UI projects history through the latest vocabulary. Historical occupancy
can therefore change after consolidation, while the actual input used for an
old birth remains unchanged. See [gene records and history](GeneCatalogue.md).

## Persistence and execution

Local runs use a supervised task. Production dispatches evolutions to a
Cloud Run Job; web and worker share Firestore, the encryption secret and private
artifact storage. An ownership token and renewable lease fence progress,
checkpoint and completion writes, preventing replaced workers from publishing
over a successor.

Completed children are saved in a generation draft. The visible committed
history remains at the last full generation until all slots are ready. Resume
reuses finished slots, selection and parent plans. Abrupt termination can lose
an unfinished provider response; a failed checkpoint is not successful saving.
Auto-rating still runs in the web process and can lose unfinished work on
termination.

Usage records preserve reported tokens and price snapshots. The app's budget
is based on recorded estimates, not provider billing enforcement. Missing usage,
in-flight requests, infrastructure and unrelated provider activity limit that
estimate. Costs and model capabilities are defined in
[`idea/config.py`](../idea/config.py); runtime behavior uses
[`idea/provider_execution.py`](../idea/provider_execution.py).

## Code map

| Area | Entry point |
| --- | --- |
| Web app and API | [`idea/viewer.py`](../idea/viewer.py) |
| Validated run configuration | [`idea/evolution_types.py`](../idea/evolution_types.py), [`idea/run_requests.py`](../idea/run_requests.py) |
| Generation loop | [`idea/evolution_orchestrator.py`](../idea/evolution_orchestrator.py) |
| Selection and population state | [`idea/evolution.py`](../idea/evolution.py) |
| Agents and provider calls | [`idea/llm.py`](../idea/llm.py) |
| Gene extraction and consolidation | [`idea/genes.py`](../idea/genes.py), [`idea/gene_pool.py`](../idea/gene_pool.py) |
| Evaluation and diversity | [`idea/evaluation/`](../idea/evaluation/), [`idea/diversity.py`](../idea/diversity.py) |
| Ownership and workers | [`idea/run_service.py`](../idea/run_service.py), [`idea/worker.py`](../idea/worker.py) |
| Saved records | [`idea/database.py`](../idea/database.py), [`idea/evolution_persistence.py`](../idea/evolution_persistence.py) |
| Browser UI | [`idea/static/`](../idea/static/) |
