# Repository cleanup and extensibility audit

Date: 2026-07-22

Scope: architecture, security, evaluator extensibility, persistence, generated
artifacts, and the artifact-facing interface. The initial baseline passed 210
tests. This branch adds regression and adversarial coverage for the work below.

## Executive summary

The four critical issues found at the start of this pass are remediated in this
branch: stored-XSS paths, unauthenticated local-file traversal, unsafe
development-auth defaults, and non-transactional Firestore chunk publication.
The evaluator architecture is no longer tied to only Elo plus diversity: typed
pairwise, diversity, computed-score, and artifact-producing components now feed
one configurable fitness aggregator.

The main remaining risks are operational rather than blockers for the first
prime-visualization experiment:

1. synchronous Firestore calls still run inside async request/task functions;
2. production artifact storage is still a single-host filesystem rather than a
   multi-instance object store with a retention worker;
3. API/UI configuration parsing and the large frontend modules still duplicate
   concepts that now have canonical backend models;
4. the bounded accessibility pass needs real browser, contrast, reflow, and
   chart-equivalent QA;
5. encryption has no key identifier or rotation workflow yet.

## Critical findings and disposition

### 1. Generated/server text reached unsafe DOM sinks — fixed for known routes

The viewer, rater, template library, admin screen, and authentication header
previously interpolated generated or server-controlled strings into HTML and
attributes. Link destinations also needed protocol validation.

The affected first-party routes now use DOM text assignment or escaping, exact
same-origin checks for authenticated fetch signing, and an allow-list for web
and local-relative Markdown links. Cross-origin token leakage, entity-obfuscated
schemes, generated titles/content, Oracle analysis, user/template fields, and
valid-link behavior have regression tests. Generated HTML/SVG remains inert and
is not mounted as active DOM.

Remaining constraint: the two small Markdown renderers should be replaced by a
single maintained sanitizer before their feature set grows. Active SVG/HTML
would still require rasterization or a separately isolated origin.

### 2. Legacy local-save traversal and authorization — fixed

Local saves, reads, deletes, and rating mutations require authentication and
use opaque per-user directories. File names are path-free, reads reject
symlinks/non-regular files, and writes use contained atomic replacement.
Unscoped root legacy files are hidden unless an operator explicitly enables the
migration flag; rating one creates a private user copy.

### 3. Development authentication/network exposure — fixed

The development server binds to loopback by default. Development auth uses a
per-process random bearer token, grants no admin rights unless separately opted
in, and CORS defaults to same-origin. Split frontends require an explicit
allow-list.

### 4. Firestore manifest/chunk races and silent corruption — fixed

Evolution saves now stage immutable, versioned chunks and publish their parent
manifest with an optimistic transaction bound to the revision originally
loaded by the engine. The successful revision is threaded back into the engine;
a stale engine raises `EvolutionSaveConflict` instead of reporting success or
overwriting newer state.

Published chunks carry field/index/count/schema metadata and a whole-payload
SHA-256 digest. Reconstruction rejects missing, reordered, malformed, inactive,
or digest-mismatched chunks. If a manifest changes during a read, the reader
retries exactly once against the new revision; an unchanged corrupt revision
raises `EvolutionStorageIntegrityError` rather than returning partial/raw data.

Old versions become `retired` at publication time, and their garbage-collection
grace starts at retirement rather than original creation. Abandoned stages are
also collected after grace. Cleanup after a successful publish is best-effort,
so a GC outage does not make a durable save look failed.

Deletion atomically publishes a separate tombstone before removing the parent.
That tombstone prevents an in-flight or later stale save from resurrecting the
same evolution ID; child cleanup is idempotent. Tests cover sequential stale
writers, reader/publish interleaving, retirement grace, missing/malformed/digest
corruption, staged-orphan cleanup, delete/save races, and post-publish GC errors.

Operational follow-up: run the same cases against the Firestore emulator in CI,
not only the deterministic fake. Rename remains a lightweight metadata update;
it should eventually participate in the same revision-aware repository API.

## High-severity architecture findings and disposition

### 1. Evaluation was embedded in `Critic` — core boundary implemented

Evaluation suites now declare ordered components, evidence dependencies,
failure policies, and weighted maximize-only fitness metrics. The legacy suite
is synthesized when no explicit config exists, preserving pairwise Elo,
embedding diversity, and age-decayed survival behavior.

Supported component classes are:

- the existing pairwise LLM judge;
- embedding diversity;
- generic computed scorers returning finite numeric metrics where larger is
  fitter;
- LLM code artifact producers that enrich candidates before the same pairwise
  judge runs.

Failures remain `failed`, `skipped`, or `cached`; provider errors, timeouts,
invalid judge responses, and incomplete comparisons no longer become ties or
zeroes. Aggregation applies explicit fail, penalize, or omit policy.

### 2. Candidate data was an ad-hoc shallow dictionary — boundary implemented

`CandidateRecord` and its recursive wire codec preserve UUIDs, nested Pydantic
values, artifact references, evaluator records, lineage, and unknown extension
fields. Unsupported values fail rather than being silently stringified.
Formatter/refiner merges retain candidate metadata instead of enumerating and
dropping new fields.

The dictionary wire shape remains for compatibility. A future migration can
make `CandidateRecord` the in-memory type after the frontend and older saved
runs no longer depend on loose dictionaries.

### 3. Run configuration was repeated — canonical backend model implemented

`EvolutionRunConfig` now owns every engine run parameter and is used for
checkpoint serialization, restoration, live config payloads, and engine
snapshots. Runtime-injected runners and artifact stores are explicitly excluded
from persistence and re-injected on resume.

Remaining duplication: the start endpoint and browser controls still parse and
enumerate fields manually. Generate their request/UI schema from the canonical
model rather than adding another mirror.

### 4. Rich persistence and artifact ownership — first complete local slice

Binary artifacts live in a bounded, content-addressed `ArtifactStore`; candidate
snapshots persist opaque IDs, keys, digests, sizes, media types, producer IDs,
and candidate-content digests. Import rejects traversal, symlinks, special
files, spoofed media, oversized files/images, and mutations during copy.

The opt-in web integration partitions stores by hashed user/evolution IDs.
Artifact reads require authentication, prove the reference belongs to that
user's evolution, recheck size/digest, and return a non-executable media type
with `nosniff` and restrictive CSP. Web deletion removes referenced objects
individually after the evolution is durably deleted.

Remaining production work: implement the same interface over object storage,
signed/streamed delivery, durable retention/GC, quotas, and multi-instance
ownership. Full histories still duplicate candidate snapshots across
generations and should eventually use immutable candidate/event records.

### 5. Evolution-time and post-hoc judging drifted — unified

Auto-rating resolves the saved canonical suite, idea type, component identity,
requirements, and failure behavior, then runs the same `LLMPairwiseEvaluator`.
It now accepts the evolution's authenticated artifact loader when the saved
judge requires images. Request-specific Swiss rounds remain an explicit
post-hoc control.

### 6. The interface assumed text plus three metrics — partial but useful fix

Cards now detect artifact-version changes, show an artifact count and primary
PNG preview, and preserve artifact/evaluation data through the live JSON
adapter. The shared idea modal exposes capability-driven Overview, Artifacts,
Code & run, and Evaluations tabs; source and evaluator fields are inserted with
text DOM APIs. Authenticated `fetch` hydrates image blobs, so bearer tokens are
not placed in image URLs.

Tree/ranked views still have specialized metric presentation, and the frontend
remains a large file. The next UI refactor should make the inspector a reusable
module and render arbitrary fitness contributions consistently in every view.

### 7. Keyboard and announcement gaps — high-impact subset fixed

Cards/Tree/Ranked uses roving tabs and named panels. Template/history/ranking
selection and sort controls use native buttons; sort direction has `aria-sort`;
progress uses appropriate live regions; principal dialogs have programmatic
names and focus return; affected controls have visible focus styles. The new
idea-detail tabs also use roving arrow-key behavior.

Remaining WCAG 2.2 AA work: measure all color combinations, increase compact
targets toward 44×44 CSS pixels, add data-table equivalents and summaries for
charts, verify zoom/reflow to 400%, and perform keyboard/screen-reader/browser
QA on responsive layouts.

## Medium-severity maintainability findings

### Frontend and route modules remain monolithic

`viewer.js`, `viewer.py`, and the main template remain too large. Extract a
state/data layer and modules for history, candidates, tree/ranked views,
artifacts, charts, and the shared inspector. Avoid building hidden views and
re-reading layout geometry unnecessarily.

### Dependency ownership is clearer but dual declarations remain

`requirements.txt` now matches the `google-genai` bound in `pyproject.toml`, and
the lockfile is authoritative for reproducible development. The project still
ships both the deprecated `google-generativeai` SDK and the newer SDK during
migration. Remove the old client after the remaining wrapper code moves, and
generate compatibility requirements from the project metadata rather than
maintaining them manually.

### Encryption is fail-closed but lacks rotation

There is no known production fallback key. Missing or shorter-than-32-byte
secrets raise a configuration error, invalid ciphertext raises a distinct
decryption error, and the settings status endpoint returns only a masked hint,
never the decrypted key. The local emulator alone receives a clearly development-
only key. Production should next store a key/version identifier with each
ciphertext and support staged decrypt-old/encrypt-new rotation via Secret
Manager or KMS.

### Synchronous storage calls can block the async server

Most functions in `database.py` are declared async but call the synchronous
Firestore client directly. Under latency or concurrent users this blocks the
event loop. Move to Firestore's async client or wrap cohesive repository
operations in a bounded worker pool; do not scatter individual transaction
steps across threads.

### Generated execution remains a deliberately narrow backend

The Gemini reference runner is fixed to Gemini 3.6 Flash Code Execution,
disabled by default, capped at a 180-second interaction timeout (configurable
up to 300 seconds) and 8,192 output tokens, requests
non-stored interactions, accepts exactly one bounded PNG, and never executes
returned Python locally. This is appropriate for the first visualization
experiment, not an arbitrary simulator.

Generic untrusted simulations still need the direct Agent Platform Sandbox or
a pinned, networkless container implementation. Managed autonomous agents are
too costly and variable for the single-render hot path; see
`ManagedExecutionOptions.md`.

## Recommended next priority

1. Add Firestore-emulator race/integrity tests and move blocking repository work
   off the async event loop.
2. Implement an object-backed `ArtifactStore` with quotas, durable retention,
   and multi-instance delivery.
3. Derive start-request and browser configuration schemas from
   `EvolutionRunConfig`.
4. Split the viewer route and frontend monoliths around the shared candidate
   and inspector contracts.
5. Add the direct isolated sandbox/container runner for arbitrary computed
   simulations while retaining the same result manifest.
6. Add encryption key IDs/rotation, remove the legacy Gemini SDK, and complete
   browser-based WCAG QA.
