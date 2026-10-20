# Evaluator reference

The default evaluation suite combines pairwise LLM Elo and embedding diversity.
Backend callers can supply a versioned suite with additional computed metrics,
display-only evidence or no evaluation. The app does not currently expose
generated-code execution; the former prototype is
[archived](../archive/prime-visualization/README.md).

## Contracts

[`idea/evaluation/models.py`](../idea/evaluation/models.py) is the schema authority.

| Type | Purpose |
| --- | --- |
| `EvaluationSuiteConfig` | Ordered components and fitness configuration |
| `EvaluationRecord` | Component/version, generation/scope, status, metrics, evidence and errors |
| `MetricConfig` | Weight, normalization and penalty score |
| `ArtifactRef` | Opaque storage key, media type, digest, size and producer provenance |

Components produce evidence; fitness aggregation combines configured metrics;
survival applies population policy afterward. Evidence can remain visible
without affecting selection by omitting its metric or giving it zero weight.
Artifact binaries live outside Firestore history.

Default `fitness_alpha` maps to Elo weight `alpha` and diversity weight
`1 - alpha`, both using running normalization. Explicit suites replace that
mapping. Metric outputs follow a higher-is-better convention: convert a
minimize-only measurement at the runner boundary rather than relying on the
aggregator to infer its meaning.

## Missing and failed evidence

Records retain `ok`, `failed`, `skipped` or `cached` status. Failed comparisons
do not become ties. A component selects:

- `fail_run`: stop when its required evidence is unavailable.
- `penalize`: retain the failure and use the configured penalty for aggregation.
- `omit`: omit its metric and renormalize the remaining weights.

A numeric penalty is a selection policy, not a measured result. UI charts
must preserve measurement coverage and distinguish unavailable evidence from
zero.

## Computed runners

A runner returns a structured result with `schema_version`, `status`,
`metrics`, optional artifact descriptions and a summary. It must honor its
declared timeout and execution boundary.

`builtin.character_count` is the allowlisted in-process example. It counts
`content`, `title` or `title_and_content`, optionally excluding whitespace:

```yaml
id: length
kind: computed
runner_ref: builtin.character_count
metric_names: [character_count]
parameters:
  field: content
  include_whitespace: true
```

This component performs no dynamic source execution or external I/O. It is a
contract example, not a recommendation to reward verbosity. Untrusted generated
code must never execute inside the web process; a subprocess timeout alone is
not an isolation boundary.

See [`legacy.py`](../idea/evaluation/legacy.py) for the default suite and
[`aggregation.py`](../idea/evaluation/aggregation.py) for normalization and
failure handling.
