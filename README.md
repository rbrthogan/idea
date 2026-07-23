# Idea Evolution

Idea Evolution is a small web app that uses multiple LLM agents (Ideator, Critic, Breeder, Oracle) to generate, evolve, and rate ideas over several generations. You can run a full end‑to‑end evolution, visualize diversity, and rank ideas manually or automatically.

![Screenshot](docs/IDEA_screen1.png)
![Screenshot](docs/IDEA_screen2.png)
![Evolution Overview](docs/evolution_overview.png)

## What you can do

- Generate and evolve ideas with selectable Gemini models, including Gemini 3.6 Flash
- Track population diversity over generations (embeddings‑based)
- Keep a configurable fraction of each generation as survivors (instead of full replacement)
- Use hybrid fitness for selection: normalized Elo + normalized diversity
- Apply age‑decayed survival pressure so very old ideas phase out gradually
- Rate ideas with an ELO‑style tournament (manual and automated)
- Create and edit YAML prompt templates in the UI
- Save/load evolutions and re‑rate them later
- Configure a versioned evaluator suite in the backend (the existing LLM
  tournament + diversity behavior remains the default)

## Quickstart

1) Install dependencies

```bash
uv sync
```

2) Configure your API key

```bash
export GEMINI_API_KEY=your_api_key_here
export ENCRYPTION_KEY="$(openssl rand -base64 32)"
```

`ENCRYPTION_KEY` is required before user API keys can be stored or read. Keep
the same secret across restarts and provide it through your deployment's secret
manager; the application intentionally has no production fallback key and
rejects secrets shorter than 32 bytes.

3) Run the app

```bash
uv run uvicorn idea.viewer:app
```

For local Codex/browser testing where Google popups get in the way, you can
enable the localhost-only development auth shim:

```bash
IDEA_DEV_AUTH=1 uv run uvicorn idea.viewer:app --host 127.0.0.1 --port 8000
```

The development shim uses a per-process random bearer token and is non-admin by
default; set `IDEA_DEV_AUTH_ADMIN=1` only when admin testing is required.
`run_local.sh` also binds to `127.0.0.1` unless `IDEA_BIND_HOST` is explicitly
set. Browser access is same-origin by default; a separate frontend must be
allow-listed with comma-separated `IDEA_CORS_ORIGINS`.

Local JSON exports are stored in an opaque per-user directory under
`data/local-users/`. Root-level `data/*.json` files are not exposed through the
rating or history APIs by default. For a temporary legacy migration, opt in
explicitly with `IDEA_ALLOW_UNSCOPED_LEGACY_LOCAL_FILES=1`; rating a legacy file
creates a private user-scoped copy and leaves the root file unchanged.

4) Open the UI

- Evolution viewer/creator: http://localhost:8000/
- Idea rater: http://localhost:8000/rate

## A 1‑minute tour

- Pick a template (or generate a new one), choose model and population settings, then click Start Evolution.
- Watch ideas arrive plus a diversity chart for each generation.
- Each generation keeps some survivors and breeds only the configured replacement fraction.
- Parent selection for children is weighted by hybrid fitness (`alpha * elo_norm + (1 - alpha) * diversity_norm`).
- Survivor selection uses the same fitness with an age decay factor so older ideas are less likely to persist forever.
- The Oracle can still replace the least interesting idea (closest to the centroid) to avoid convergence.
- Use Save Results to snapshot an evolution; visit Rate Ideas to compare and rank saved runs (manual or automated; costs are estimated and shown).

See the explainer: `docs/EvolutionOverview.md`.

## Running tests

```bash
uv run pytest -q
```

## Structure (high‑level)

- `idea/viewer.py` – FastAPI app serving the viewer, rater, and template APIs
- `idea/evolution.py` – Evolution engine, hybrid fitness, survivor selection, parent allocation, diversity tracking
- `idea/evaluation/` – Typed evaluator configs/results, legacy LLM/diversity adapters, computed-runner boundaries, and fitness aggregation
- `idea/artifact_store.py` – Contained, content-addressed local artifact storage
- `idea/llm.py` – LLM wrapper and agents (Ideator, Formatter, Critic, Breeder, Oracle)
- `idea/prompts/` – YAML templates system, validation, loader, and Oracle prompt bundle
- `idea/static/` – Frontend HTML/CSS/JS (viewer and rater UIs)

## Templates

YAML templates live in `idea/prompts/templates/`.

- `airesearch.yaml` – GPU‑poor AI research proposals
- `game_design.yaml` – Browser game designs
- `drabble.yaml` – 100‑word stories

Create new templates from the UI (on the main page) or by adding YAML files. See `idea/prompts/templates/README.md` for the schema.

## Configuration tips

- Models and pricing are defined in `idea/config.py` and used to estimate token costs for evolution and auto‑rating.
- Gemini 3 models use thinking levels; Gemini 2.5 models use thinking budgets. The UI exposes the supported control for each model.
- Advanced evolution controls include:
  - `mutationRate` (breeding mutation on new children)
  - `replacementRate` (fraction of population replaced per generation)
  - `fitnessAlpha` (weight of Elo vs diversity in hybrid fitness)
  - `ageDecayRate` (how quickly older ideas lose survival probability)

## Evaluators and artifacts

Evaluation is being migrated from a hard-wired LLM tournament to a composable
suite. Runs with no explicit `evaluation` configuration retain the legacy
behavior: pairwise LLM Elo weighted by `fitnessAlpha`, embedding diversity
weighted by `1 - fitnessAlpha`, then age decay for survivor selection.

The typed backend now supports maximize-only numeric scores, normalization,
weights, failure policies, and a no-evaluation mode. A generic computed
evaluator consumes the strict result contract documented in
[`docs/PluggableEvaluators.md`](docs/PluggableEvaluators.md) through an injected
runner; a checked-in Python function or isolated script is only one possible
implementation. Runners convert costs, latency, and other minimize-only
measurements into scores where larger is fitter. Generated code execution is
disabled by default. For a local opt-in Gemini Code Execution workflow, run:

```bash
./run_local.sh --visual-artifacts
```

The flag enables `IDEA_ENABLE_GEMINI_CODE_ARTIFACTS` and defaults
`IDEA_ARTIFACT_STORE_ROOT` to `$PWD/data/artifacts`. Set that environment
variable first if you want a different location.

An evaluation suite must still explicitly set
`allow_generated_code_execution: true` and declare an `llm_code_artifact`
component. The web app then creates a private per-user/per-evolution store,
uses only Gemini 3.6 Flash's remote Code Execution tool, caps the request at 30
seconds and 8,192 output tokens, and never executes returned Python locally.
After the server opt-in, the browser exposes a still-off-by-default **Generate
visualization artifacts** switch under Advanced Config. It submits the bundled
render → ordinary pairwise judge → diversity suite; callers can instead send a
custom typed `evaluation` suite to `/api/start-evolution`.
Artifact-runner token usage is included in saved usage totals and the run's
`maxBudget`; resumed runs retain the previously spent amount.
Generated images are served only through an authenticated ownership-checked
route; the idea modal exposes Overview, Artifacts, Code & run, and Evaluations
tabs only when corresponding data exists. The investigated remote and local
isolation options, current Gemini 3.6 Flash pricing, and recommended staged
rollout are documented in
[`docs/ManagedExecutionOptions.md`](docs/ManagedExecutionOptions.md).

Artifact-producing components enrich candidates before the existing pairwise
LLM judge. There is no separate artifact-only LLM evaluator: when a pairwise
component explicitly requires a current image artifact, that same judge receives
both candidates' text and matching PNG bytes in one multimodal comparison.

For a zero-trust-boundary computed example, `builtin.character_count` is an
allow-listed in-process runner that emits one configurable maximize-only length
metric. It never evaluates candidate source or performs I/O. External
simulations and API-backed scorers still require an explicitly injected runner.

Local artifact output can be imported into the content-addressed artifact store
after traversal, symlink, media-type, size, and digest checks. Firestore history
stores references rather than artifact bytes.

The bundled `prime_visualization` template treats the Ulam spiral as only one
optional anti-copy precedent among modular maps, prime races, binomial
divisibility, gap/constellation views, and factor geometry. It asks for explicit
mappings, falsifiable nulls, deterministic implementation plans, and
PNG/metadata/result contracts. The template never fabricates results; an
explicit artifact producer may subsequently implement a proposal.

## Troubleshooting

- “GEMINI_API_KEY environment variable is not set” banner: export the key and restart the app.
- Empty diversity chart for “current evolution”: save the evolution and re‑open it from the list (see also known issue in `TODO.md`).

## Roadmap

See `TODO.md` for planned features, research ideas, and open bugs.
