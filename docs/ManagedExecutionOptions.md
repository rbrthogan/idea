# Managed execution options for generated artifacts

Research date: 2026-07-22. Pricing and preview terms are time-sensitive and
must be checked again before enabling billing or production traffic.

## Recommendation

Use a deliberately narrow two-step rollout:

1. **Prototype with Gemini API built-in Code Execution.** It is the smallest
   remote trust boundary: one Gemini request can generate Python, execute it in
   Google's environment, and return an inline Matplotlib image. There is no
   separate runtime charge; generated code, execution results, reasoning, and
   the final response are billed as model tokens. It is Python-only, has a
   30-second execution limit, and may retry failed code up to five times.
   The app now contains this as an opt-in, injected reference runner; no web
   route enables it by default.
2. **Use Agent Platform Sandbox/Code Execution when this app must execute an
   exact source bundle independently of a model turn.** The direct sandbox API
   is a better fit for a reproducible artifact producer or generic computed
   scorer than adopting an autonomous agent. It is designed for isolated,
   networkless execution of untrusted code and can be called from a local app.

Do not make either Managed Agents variant the core execution dependency yet.
Google currently documents two related surfaces whose defaults and terms differ:

- **Managed agents on the Gemini API** are in Public Preview. Their reusable
  Linux environment is free of compute charges during preview, but outbound
  network access is unrestricted unless the caller explicitly disables it, and
  Google says one interaction typically consumes 100,000 to 3 million tokens.
  Google's measured examples put common tasks around $0.25–$3.25, with complex
  3–5-million-token workflows reaching about $5. The agent also lacks structured
  output, though it exposes a total-token cap. That autonomy and cost variance
  are a poor fit for one bounded render and strict result manifest.
- **Managed Agents API on Gemini Enterprise Agent Platform** starts with no
  external network or ambient credentials, which is the safer default. It is
  currently Pre-GA, evaluation-only, unsuitable for confidential data, and may
  not be used commercially or in production under the published preview terms.

Both are useful experiments for larger multi-step coding workflows. Neither is
as small or predictable as the Gemini API's built-in Code Execution tool for
this first artifact producer.

## How this maps to Idea Evolution

Artifact production and judging remain separate:

```text
idea text
  -> artifact producer (Gemini built-in Code Execution initially)
  -> enriched idea {text, image artifact, source, run metadata}
  -> the same pairwise LLM judge, now given both ideas and their artifacts
  -> pairwise quality/Elo contribution

idea text and/or artifacts
  -> zero or more computed scorers
  -> finite numeric scores where larger is always fitter

idea population
  -> diversity contribution

pairwise quality + computed scores + diversity
  -> configured fitness aggregation
```

The image is evidence attached to the idea. It is not evaluated by a separate
"visual judge" component. The ordinary pairwise judge receives the matching
artifact for each candidate and applies template-specific comparison criteria.
The implemented adapter sends both candidate texts and their matching PNG bytes
in one comparison call, and includes artifact digests in both cache identities.

A computed scorer is provider-neutral. Counting characters, running a local
trusted function, invoking a simulation in Agent Platform Sandbox, or calling
another model API all implement the same contract:

```json
{
  "schema_version": 1,
  "status": "ok",
  "metrics": {"fitness_signal": 0.82},
  "artifacts": [],
  "summary": "Optional bounded evidence"
}
```

Every published score already has fitness direction applied: a larger number
is better. A scorer for latency or error may therefore publish `-latency` or a
bounded inverse transform. The fitness aggregator does not guess direction.

## Option comparison

| Option | Best use here | Execution boundary | Cost shape | Important limits |
| --- | --- | --- | --- | --- |
| Gemini API Code Execution | First prime-visualization prototype | Google-managed Python tool inside the model request | Model tokens only; no added runtime fee | Python only, 30 seconds, outputs inline, model controls execution loop |
| Agent Platform Sandbox / Code Execution | Exact generated source and generic simulations | Direct isolated sandbox API, independent of agent framework | Current Agent Platform compute/memory rates and free tier | Preview; Cloud setup and lifecycle management required |
| Gemini API Managed Agents / Antigravity | Exploratory multi-step coding agent | Managed Linux sandbox with Bash/files; explicitly set `network: "disabled"` because egress is open by default | Standard model tokens; preview sandbox compute currently unbilled; typically 100K–3M tokens per interaction | Public Preview; more autonomous and cost-variable than needed |
| Enterprise Agent Platform Managed Agents | Governed evaluation of managed autonomous agents | Isolated Linux sandbox with network and ambient credentials disabled by default | Standard model tokens; consult preview pricing | Pre-GA evaluation only; not for commercial, production, confidential, or sensitive workloads |
| Local Docker/Podman | Offline/private fallback | Local container with explicit mounts and limits | Local machine resources | Requires a correctly installed and maintained container runtime |
| Trusted in-process Python scorer | Tiny checked-in deterministic metrics | Application process | Negligible | Never accepts generated source; only registered reviewed callables |

## Current Gemini 3.6 Flash pricing

Gemini 3.6 Flash is GA as `gemini-3.6-flash`. Google lists a 1,048,576-token
input limit, 65,536-token output limit, multimodal input, structured output,
thinking, and Code Execution support.

Gemini Developer API pricing per one million tokens:

| Consumption mode | Input | Output (including thinking) | Cached input |
| --- | ---: | ---: | ---: |
| Standard | $1.50 | $7.50 | $0.15 |
| Batch | $0.75 | $3.75 | $0.075 |
| Flex | $0.75 | $3.75 | $0.075 |
| Priority | $2.70 | $13.50 | $0.27 |

The app now includes the Standard rates in its model cost estimator. Code
Execution can add substantial intermediate tokens: generated code and execution
results count in billing as they move through the model loop. Cost controls
should therefore cap prompt size, output tokens, retries, artifact dimensions,
and per-generation artifact count even though runtime itself has no added fee.

For the broader Agent Platform, Google's current unified pricing page lists a
monthly account free tier of 50 Agent Compute vCPU-hours and 100 Agent Memory
GiB-hours, then $0.085/vCPU-hour and $0.009/GiB-hour. Runtime and sandbox
environments, including Code Execution, use those resources. Managed agents on
the Gemini API are a separate preview case whose environment compute is
currently unbilled; their model inference and intermediate reasoning still use
standard token rates.

## Safe minimal integration profile

The first remote producer should be narrower than a general shell agent:

- fixed `gemini-3.6-flash` model and Code Execution tool;
- explicit `allow_generated_code_execution`, an enabled injected runner, and an
  injected `ArtifactStore`; the web path creates these only when
  `IDEA_ENABLE_GEMINI_CODE_ARTIFACTS=true` and the saved suite also opts in;
- candidate text delimited as untrusted data; no credentials are placed in the
  prompt, and callers must not put secrets in candidate text;
- deterministic Matplotlib/Pillow-only output requirements;
- exactly one bounded primary PNG plus Python source and bounded JSON metadata;
- a 180-second end-to-end interaction timeout by default (configurable up to
  300 seconds), while each provider-managed code execution remains limited to
  30 seconds; 8,192 output tokens by default; and one requested render;
- validate decoded PNG bytes and dimensions through `ArtifactStore` before
  attaching the opaque artifact reference to the candidate;
- persist generated source, model ID, prompt digest, usage, and result digest;
- include artifact-model usage in the evolution token/cost summary and
  `max_budget`, including usage restored from a checkpoint;
- request non-stored interactions and cap model output at 8,192 tokens by default;
- never execute the returned Python in the app or on the user's machine;
- never trust the model's summary as the execution result contract;
- pairwise judging receives each image only alongside its owning candidate;
- reject missing, corrupt, stale, or ambiguous required refs before judging;
- execution or artifact failure remains explicit and follows the configured
  failure policy rather than becoming a tie or zero.

The local web integration partitions the artifact root by hashes of the
authenticated user and evolution IDs. Browser image/source requests go through
an authenticated endpoint that first proves the reference belongs to that
user's evolution, then verifies the stored byte count and digest. Deleting an
evolution through the web path also deletes its referenced content-addressed
objects one by one. When the server opt-in is active, Advanced Config exposes
an off-by-default visualization-artifact switch that submits the safe bundled
render/pairwise/diversity suite. Multi-instance deployment still needs an object-store
backend and a durable retention worker; a process-local filesystem is not a
production artifact service.

If reproducibility or arbitrary simulations outgrow the built-in tool, keep the
same producer/scorer interfaces and replace only the runner with the direct
Agent Platform Sandbox adapter or the pinned local container adapter.

## Primary sources

- [Gemini 3.6 Flash model](https://ai.google.dev/gemini-api/docs/models/gemini-3.6-flash)
- [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Gemini API Code Execution](https://ai.google.dev/gemini-api/docs/code-execution)
- [Gemini API managed agents overview](https://ai.google.dev/gemini-api/docs/agents)
- [Antigravity agent capabilities and estimated costs](https://ai.google.dev/gemini-api/docs/antigravity-agent)
- [Gemini API managed agent environments](https://ai.google.dev/gemini-api/docs/agent-environment)
- [Managed Agents sandbox environment](https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/managed-agents/sandbox-environment)
- [Gemini Enterprise Agent Platform pricing](https://cloud.google.com/products/gemini-enterprise-agent-platform/pricing)
- [Agent Platform Code Execution](https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/sandbox/code-execution-overview)
