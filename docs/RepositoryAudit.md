# Repository cleanup and extensibility audit

Date: 2026-07-22

Scope: architecture, security, evaluator extensibility, persistence, and the artifact-facing interface. The baseline audited at the start of this work passed 210 tests after the bundled Node runtime was added to `PATH`.

## Executive summary

The application has strong experimental functionality, but its extension cost is rising because the data and control-flow contracts remain implicit. The highest-leverage work is to establish typed candidate/evaluation/artifact boundaries before adding generated code or rich previews.

Findings in this pass:

- 4 critical security or data-integrity risks;
- 8 high-severity architectural or usability risks;
- 5 medium-severity maintainability risks.

Two critical exposure paths—the unauthenticated legacy local-file workflow and
unsafe development-auth defaults—are hardened in this branch. Generated-code
execution must remain disabled until the stored-XSS surface and an isolated
execution backend are addressed; the Firestore publication race also remains.

## Critical findings

### 1. Untrusted generated text can reach `innerHTML`

Locations include `idea/static/js/viewer.js` card rendering around lines 3001-3081, Oracle analysis around 6103-6118, and `idea/static/js/rater.js` around 1630-1645 and 1993-2008. The custom Markdown renderer escapes angle brackets but interpolates arbitrary link destinations into `href` around `viewer.js:1158`.

Impact: stored XSS is possible today. Adding generated SVG, HTML, source, logs, or artifact captions would multiply the attack surface.

Recommendation: use `textContent` for plain fields, adopt a maintained allow-list Markdown sanitizer, filter URL protocols, and treat generated SVG/HTML as active content. Rasterize it or serve it from an isolated origin through a restrictive sandboxed frame and CSP.

### 2. Legacy save endpoint permitted unauthenticated path traversal

The legacy `POST /api/save-evolution` path accepted a client filename and wrote `DATA_DIR / filename` without authentication or containment checks (`idea/viewer.py`, formerly around lines 2340-2368). Absolute paths and `../` segments could overwrite any process-writable file. Permissive CORS made browser-driven exploitation possible.

Status: fixed in this branch. Local saves and rating mutations now require
authentication, use opaque per-user directories, perform no-follow reads and
atomic replacement writes, and hide unscoped root legacy files unless the
operator explicitly enables the migration flag. Long-term, remove the
filename-based endpoint and save through the authenticated repository using
server-controlled evolution IDs.

### 3. Local development authentication can be exposed beyond localhost

The development workflow combined a known/default development identity, broad
CORS, and a run script that bound to `0.0.0.0`. A local convenience mode could
become an account/admin bypass if the port was reachable from another machine
or an untrusted local webpage.

Status: hardened in this branch. The run script binds to loopback by default,
development auth requires a per-process random bearer token, admin access is
opt-in, and CORS defaults to same-origin with an explicit allow-list for split
frontends. A prominent unsafe-mode banner remains worthwhile when an operator
explicitly broadens these settings.

### 4. Firestore history chunk publication is non-transactional

Chunked evolution saves publish a parent document and clean old chunk versions in separate operations (`idea/database.py`, chunk save/delete paths around lines 275-382). Concurrent saves can interleave so one writer deletes chunks referenced by the parent published by another writer.

Impact: a successful save can later load as incomplete or corrupt.

Recommendation: publish immutable versioned chunks, atomically compare-and-set the parent manifest, and garbage-collect only unreferenced versions after a grace period. Add a concurrency test with deliberately interleaved writers.

## High-severity architecture findings

### 1. Evaluation is an implementation detail of `Critic`

`EvolutionOrchestrator` calls `engine.critic.get_tournament_ranks` directly, and `EvolutionEngine` hard-codes normalized Elo plus diversity. `Critic` combines critique/refinement, pairwise judging, cache identity, Swiss ranking, and evaluation metadata.

Impact: a numeric script evaluator or artifact judge requires invasive branches in orchestration, selection, persistence, cost reporting, and UI.

Recommendation: separate artifact producers, evaluators, fitness aggregation, and survival policy as described in `docs/PluggableEvaluators.md`.

Status: the first boundary is implemented in this branch. Legacy pairwise and
diversity behavior now runs through typed components and aggregation; isolated
artifact/code execution remains deliberately unavailable.

### 2. Candidate shape is an ad-hoc dictionary

`idea/models.py` defines only `title` and `content`, while lineage, ratings, genotypes, island state, selection metrics, and provenance are added dynamically. Shape normalization is repeated in LLM formatting, diversity, persistence, rating, and JavaScript.

Impact: fields can silently disappear. In particular, `Formatter` preserves a hand-picked metadata subset, which is unsafe for artifact/evaluation records.

Recommendation: introduce a versioned `CandidateRecord` and one recursive codec. Preserve the current dictionary wire format during migration, but centralize conversions and unknown-field behavior.

### 3. Configuration is repeated in five places

The start endpoint, `EvolutionEngine.__init__`, `EvolutionConfig`, UI restore logic, and `EvolutionSerializer` each enumerate configuration fields manually.

Impact: additions routinely fail to resume, persist, or restore in the browser even when a new run works.

Recommendation: make one validated run-config model authoritative; derive API parsing, persistence, defaults, and UI schema from it. Treat the current typed config mirror as an intermediate migration step.

### 4. Persistence is shallow and history duplicates rich data

The serializer special-cases UUIDs, lists, and one embedded model, while Firestore history repeats full candidates across generations. Binary artifacts or base64 previews would amplify storage and chunking costs.

Recommendation: use recursive serialization for typed records, store binary artifacts in an artifact store, and persist only opaque references/digests in candidate snapshots.

### 5. Evaluation failures become neutral evidence

Parallel pair evaluation can turn errors into a missing/unknown result, while the Elo update path treats unknown outcomes as draws.

Impact: broken evaluators silently influence selection rather than remaining visibly failed.

Recommendation: record `ok`, `failed`, `skipped`, and `cached` explicitly. Required, optional, and display-only failure policies must be configured; none should default to a tie or zero.

### 6. Post-hoc rating duplicates and drifts from evolution evaluation

Auto-rating builds its own Critic and tournament flow in `idea/viewer.py`. Saved template metadata and the canonical `config.idea_type` path are not consistently reused.

Recommendation: route evolution-time and post-hoc evaluation through the same evaluation service, with an explicit population/scope and persisted evaluator config.

### 7. The frontend assumes text plus exactly three metrics

`getIdeaData`, tree inspector, ranked view, and charts are all specialized to title/content and Fitness/Elo/Diversity. Cards also skip rerendering unless a small fixed set of metadata changes.

Impact: artifact readiness, script status, or dynamic evaluator results can remain invisible or stale.

Recommendation: render from artifact and evaluation collections. Use one shared idea inspector across Cards, Tree, and Ranked, with capability-driven Overview, Artifacts, Code & Run, Evaluations, and Lineage tabs.

### 8. Core keyboard, contrast, and responsive behavior misses WCAG AA

Clickable `div`/`span` controls, incomplete tab semantics, mouse-only sort headers, low-contrast muted/accent colors, sub-44px controls, and fixed/sidebar widths affect central workflows. Dynamic status and charts lack accessible equivalents.

Recommendation: prioritize native controls and keyboard semantics, contrast tokens, live-region summaries, accessible chart tables, and container-aware layouts before expanding UI density.

## Medium-severity maintainability findings

### 1. Frontend files are monolithic

`viewer.js` is over 8,000 lines and `viewer.html` over 2,000 lines with inline application logic. Hidden cards are built even when Tree is the active view, and resize work repeatedly reads layout geometry.

Recommendation: extract a state/model layer and small presenters for history, candidates, tree, ranked rows, charts, and the shared inspector. Render only the active overview and lazy-load detail payloads.

### 2. Visual system is generic and internally inconsistent

The current Inter + indigo/slate/emerald palette, gradients, pills, rounded cards, and shadows read as generic AI SaaS. Direct Bootstrap/white color use bypasses otherwise useful CSS tokens; dark mode and reduced motion are absent.

Recommendation: retain the purposeful tree/ranked/inspector structure, then normalize the visual system around the research-instrument brief in `.impeccable.md`.

### 3. Dependency declarations can drift

Both `pyproject.toml`/`uv.lock` and `requirements.txt` exist, and at least one bound differs. The development shell may also lack `uv` or Node even though tests depend on them.

Recommendation: make `pyproject.toml` plus `uv.lock` authoritative, generate compatibility requirements when needed, and document/use the bundled Node path for JS contract tests.

### 4. Sensitive request logging is too broad

The start endpoint logged the complete request body. Future evaluator definitions or generated source would leak into application logs.

Status: this branch validates the body shape and logs only the field count.
Establish structured redaction before adding evaluator prompts, source bundles,
or execution traces.

### 5. Artifact lifecycle is not yet defined

There is no ownership, authorization, retention, deletion, or storage abstraction for rich output.

Recommendation: add an `ArtifactStore` with opaque keys, authenticated access, digest/size/media metadata, containment checks, and deletion tied to evolution ownership before any producer is enabled.

## Positive foundations

- The test suite is broad and fast, with useful Python and JavaScript contract coverage.
- The evolution tree and ranked view expose meaningful provenance and selection state.
- Central CSS tokens, progress payloads, checkpoint/resume, custom templates, and comparison caching provide strong seams to refine.
- The recent orchestrator and persistence extractions show that incremental migration is practical without rewriting the application.
- The UI already has complementary Cards, Tree, and Ranked modes; the missing piece is a shared inspector/data contract rather than another overview.

## Recommended priority

1. Close path traversal, dev-auth exposure, and stored-XSS paths.
2. Make chunk publication race-safe.
3. Land typed evaluation/artifact records and the legacy evaluator adapter.
4. Centralize candidate serialization and run configuration.
5. Add a safe artifact store and deterministic checked-in script evaluator.
6. Add isolated container execution, then LLM code production and multimodal judging.
7. Build the shared inspector and dynamic evaluation/artifact UI.
