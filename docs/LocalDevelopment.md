# Local development

The recommended setup uses the Firestore emulator and loopback development
sign-in. No cloud database or Firebase project setup is needed.

## Start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), Node with
[Firebase CLI](https://firebase.google.com/docs/cli), and Java JDK 21+.
Check that `firebase --version` and `java -version` work.

From the repository root:

```bash
uv sync --locked
./run_local.sh
```

Open [IDEA](http://127.0.0.1:8000/). The
[rater](http://127.0.0.1:8000/rate) compares saved candidates; the
[emulator UI](http://127.0.0.1:4000/) shows local database records.
Use `127.0.0.1` consistently because browser storage is separate for `localhost`.

## Keys and persistence

Save each provider's key in **Settings**. For local development only, the app
can fall back to `GEMINI_API_KEY` or `OPENAI_API_KEY` from the environment or an
ignored `.env` file. Production users and workers require saved user keys.
The launcher parses `.env` as data rather than executing it as shell code.

**Ctrl+C** stops the app and an emulator started by the launcher. Wait for its
export to finish: `data/firestore-emulator/` is imported on the next start.
An emulator started separately manages its own persistence. Abrupt termination
can lose changes since the last export.

If you override `ENCRYPTION_KEY` or `ENCRYPTION_SALT`, keep them consistent across
restarts or saved provider keys will become unreadable. Never use the
development encryption key against a production database.

| Problem | Check |
| --- | --- |
| Missing provider key | Save the selected provider's key in Settings, or configure its environment variable and restart. |
| Provider rejects requests | Check key restrictions, account access, billing and quota with that provider. |
| Port 8000 or 8080 is busy | Stop the earlier app/emulator or deliberately reuse the existing emulator. |
| No Java runtime | Install a JDK and confirm `java -version` succeeds. |
| Settings disappear | Allow a clean emulator export; reuse the same data directory and encryption settings. |
| Launcher reports remote Firestore | Firebase CLI or `firebase.json` is missing. Fix local prerequisites; remote mode requires explicit credentials and encryption configuration. |

## Tests and source

```bash
uv run pytest -q --strict-markers
```

Default tests block unexpected network requests and remove ambient provider
credentials. Optional live checks require explicit opt-in. The read-only
`tests/browser_fixture_app.py` and `tests/browser_workbench_checks.js` support
browser checks without starting model work.

See [the engine overview](EvolutionOverview.md) for code entry points,
[template authoring](../idea/prompts/templates/README.md) for YAML customization,
and [deployment](../DEPLOY.md) for cloud operation.
