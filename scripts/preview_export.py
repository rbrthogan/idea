#!/usr/bin/env python3
"""Inspect a saved evolution with the existing viewer, without application services.

From the repository root::

    .venv/bin/python scripts/preview_export.py output/annotated-history.json --port 8136
    # Follow a running job without reloading the page manually:
    .venv/bin/python scripts/preview_export.py output/live-run/drabble.json --port 8142 --watch

Open http://127.0.0.1:8136. This server binds only to localhost, never writes the
export, and rejects all write requests. No provider credentials or database are
loaded. The viewer's existing CDN fonts and UI libraries may use the network.
Stop the foreground process with Ctrl-C when finished.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
import uvicorn

from idea.config import DEFAULT_MODEL, LLM_MODELS, THINKING_MODEL_CONFIG


DISABLED_MESSAGE = "Read-only export preview: generation, editing and saving are disabled."


def flatten_candidate(candidate: dict) -> dict:
    """Expose the document fields while retaining every saved metadata field."""
    result = copy.deepcopy(candidate)
    document = result.pop("idea", None)
    if isinstance(document, dict):
        result["title"] = document.get("title") or result.get("title") or "Untitled"
        result["content"] = document.get("content", "")
    elif isinstance(document, str):
        result.setdefault("title", "Untitled")
        result["content"] = document
    else:
        result.setdefault("title", "Untitled")
        result.setdefault("content", "")
    return result


def prepare_export(export: dict, source_name: str) -> dict:
    """Normalize wire shape only; never manufacture observations or evaluation data."""
    if not isinstance(export, dict):
        raise ValueError("Input must be an evolution export object")
    if isinstance(export.get("data"), dict) and "history" not in export:
        export = export["data"]
    run = copy.deepcopy(export)
    history = run.get("history")
    if not history and isinstance(run.get("population"), list):
        history = [run["population"]]
    if not isinstance(history, list) or not history:
        raise ValueError("Export has no saved history or population")
    if any(not isinstance(generation, list) or any(not isinstance(candidate, dict)
           for candidate in generation) for generation in history):
        raise ValueError("Each history generation must be a list of candidate objects")
    run["history"] = [[flatten_candidate(candidate) for candidate in generation] for generation in history]
    if isinstance(run.get("population"), list):
        run["population"] = [flatten_candidate(candidate) for candidate in run["population"]]
    else:
        run["population"] = copy.deepcopy(run["history"][-1])
    run.setdefault("evolution_id", "export-" + hashlib.sha256(source_name.encode()).hexdigest()[:12])
    run.setdefault("name", source_name)
    run["evolution_name"] = run.get("evolution_name") or run["name"]
    run.setdefault("config", {})
    run.setdefault("current_generation", len(history) - 1)
    run.setdefault("total_generations", run["config"].get("generations", len(history) - 1))
    run.setdefault("history_version", len(history))
    run.setdefault("status", "complete")
    run.update(is_running=False, history_available=True, read_only_preview=True)
    return run


def create_app(export: dict, source_name: str = "Saved evolution", *, source_path: Path | None = None) -> FastAPI:
    run = prepare_export(export, source_name)
    identity = str(run["evolution_id"])
    # Give each exact dataset its own browser-cache namespace. A previous preview
    # must not replace newly annotated data through the viewer's cache recovery.
    digest = hashlib.sha256(json.dumps(run, sort_keys=True).encode()).hexdigest()[:16]
    config = run["config"]
    template_id = str(config.get("idea_type") or "saved_export")
    template = copy.deepcopy(run.get("template_data") or {})
    template.setdefault("name", template_id.replace("_", " ").capitalize())
    template.setdefault("description", "Template snapshot from the saved export.")
    catalog = {template_id: template}
    item = {
        "id": identity, "display_name": run["evolution_name"], "type": "evolution",
        "timestamp": run.get("created_at") or run.get("timestamp")
        or run.get("gene_observation_run", {}).get("created_at"),
        "status": run["status"], "generations": run["current_generation"],
        "total_generations": run["total_generations"], "idea_type": template_id,
        "config": config, "execution_archived": True,
        **{f"can_{action}": False for action in ("continue", "resume", "rate", "use_config", "rename", "delete")},
    }
    source_stamp = None
    refresh_error = None
    refresh_lock = threading.Lock()

    def refresh_source():
        nonlocal source_stamp, refresh_error, digest
        if source_path is None:
            return
        with refresh_lock:
            try:
                stat = source_path.stat()
                stamp = (stat.st_mtime_ns, stat.st_size)
                if stamp == source_stamp:
                    refresh_error = None
                    return
                updated = prepare_export(json.loads(source_path.read_text()), source_name)
                if str(updated["evolution_id"]) != identity:
                    raise ValueError("Checkpoint identity changed")
                new_digest = hashlib.sha256(json.dumps(updated, sort_keys=True).encode()).hexdigest()[:16]
                run.clear()
                run.update(updated)
                config.clear()
                config.update(run["config"])
                template.update(run.get("template_data") or {})
                item.update(display_name=run["evolution_name"], status=run["status"],
                            generations=run["current_generation"], total_generations=run["total_generations"])
                digest, source_stamp, refresh_error = new_digest, stamp, None
            except (OSError, ValueError, TypeError):
                # A missing/partial checkpoint must not erase the last valid preview.
                refresh_error = "Latest checkpoint unavailable; showing the last valid checkpoint."

    app = FastAPI(title="IDEA read-only export preview")
    app.mount("/static", StaticFiles(directory=ROOT / "idea/static"), name="static")
    templates = Environment(loader=FileSystemLoader(ROOT / "idea/static/html"), autoescape=select_autoescape())

    @app.middleware("http")
    async def reject_writes(request: Request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            return JSONResponse({"status": "error", "message": DISABLED_MESSAGE}, status_code=405)
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    def page(request: Request):
        refresh_source()
        html = templates.get_template("viewer.html").render(
            request=request, api_key_missing=True,
            dev_auth_config={"enabled": True, "token": "local-export-preview", "uid": f"export-{digest}",
                             "email": "", "displayName": "Local export preview", "isAdmin": False},
            default_model=DEFAULT_MODEL, llm_models=LLM_MODELS, thinking_config=THINKING_MODEL_CONFIG,
        )
        # The normal viewer already handles all rendering. This adapter only
        # selects the export after normal initialization and hides write controls.
        startup_item = json.dumps({**item, "type": "legacy_checkpoint"}, ensure_ascii=True).replace("<", "\\u003c")
        html = html.replace("</head>", """<style>
          #exportPreviewNotice { margin: 0; padding: .65rem 1rem; background: #f3eefb; color: #35284a; }
          #actionButtonsFooter, #apiKeyAlert, #newEvolutionBtn, #downloadButton,
          .nav-buttons a[href='/rate'], .sidebar-utility-section,
          #view-main .controls-section > :not(:first-child):not(#evolutionSelect):not(#recentHistoryList),
          #view-main .controls-section > h6 { display: none !important; }
        </style></head>""")
        html = html.replace('<body class="research-workbench">', '<body class="research-workbench">'
            '<p id="exportPreviewNotice" role="note"><strong>Read-only export preview.</strong> '
            'Saved candidates, history and gene observations. Generation and edits are disabled; '
            'displayed scores and costs come from this export.</p>')
        script = """<script>
          (() => {
            const normalInitialize = initializeAuthenticatedWorkspace;
            let previewInitialization;
            initializeAuthenticatedWorkspace = function () {
              if (!window.currentUser) return Promise.resolve();
              if (!previewInitialization) previewInitialization = (async () => {
                await normalInitialize();
                await loadHistoryItemData(PREVIEW_ITEM);
                const name = document.getElementById('evolution-name');
                if (name) name.textContent = PREVIEW_ITEM.display_name;
              })();
              return previewInitialization;
            };
          })();
        </script>""".replace("PREVIEW_ITEM", startup_item)
        if source_path is not None:
            script += """<script>
              (() => {
                let version = INITIAL_VERSION;
                let busy = false;
                async function refreshPreview() {
                  if (busy || document.hidden || !currentEvolutionData) return;
                  // Leave an open story or in-progress search undisturbed.
                  if (currentModalIdea && !document.getElementById('ideaModal')?.hidden) return;
                  if (document.activeElement?.matches('input, textarea, select')) return;
                  busy = true;
                  const notice = document.getElementById('exportPreviewNotice');
                  try {
                    const response = await fetch('/api/preview-status', {cache: 'no-store'});
                    if (!response.ok) throw new Error('Preview unavailable');
                    const status = await response.json();
                    if (version !== status.version) {
                      const view = evolutionViewMode, scope = geneScope, selected = selectedGeneId;
                      const explorer = {...getGeneExplorerState()}, scroll = window.scrollY;
                      const focusedId = document.activeElement?.id;
                      historyItems = historyItems.map(entry => entry.id === status.item.id ? {...entry, ...status.item} : entry);
                      await loadHistoryItemData({...status.item, type: 'legacy_checkpoint'});
                      if (!currentEvolutionData || (currentEvolutionData.checkpoint_id ?? null) !== status.checkpoint_id)
                        throw new Error('Checkpoint changed during refresh');
                      getGeneExplorerState.state = explorer;
                      geneScope = scope;
                      selectedGeneId = selected;
                      setEvolutionViewMode(view);
                      if (focusedId) document.getElementById(focusedId)?.focus({preventScroll:true});
                      window.scrollTo(0, scroll);
                      version = status.version;
                    }
                    notice.textContent = status.warning || `Read-only live preview · ${status.snapshots} saved generations · checks for new checkpoints every 15 seconds.`;
                  } catch (error) {
                    notice.textContent = 'Live preview reconnecting · showing the last loaded checkpoint.';
                  } finally { busy = false; }
                }
                setInterval(refreshPreview, 15000);
                document.addEventListener('visibilitychange', refreshPreview);
                window.addEventListener('load', () => {
                  document.getElementById('exportPreviewNotice').textContent = 'Read-only live preview · checks for new checkpoints every 15 seconds.';
                });
              })();
            </script>""".replace("INITIAL_VERSION", json.dumps(digest))
        return HTMLResponse(html.replace("</body>", script + "</body>"), headers={"Cache-Control": "no-store"})

    @app.get("/api/{path:path}")
    def api(path: str):
        refresh_source()
        path = path.rstrip("/")
        if path == "preview-status":
            return JSONResponse({"version": digest, "item": item, "warning": refresh_error,
                                 "checkpoint_id": run.get("checkpoint_id"), "snapshots": len(run["history"])},
                                headers={"Cache-Control": "no-store"})
        if path == "progress":
            return run
        if path == "history":
            return {"status": "success", "items": [item]}
        if path == "evolutions":
            return [{"id": identity, "filename": run["evolution_name"], "timestamp": item["timestamp"]}]
        if path == f"evolution/{identity}":
            return {"id": identity, "name": run["evolution_name"], "data": run}
        if path == "generations":
            return run["history"]
        if path == "checkpoints":
            return {"status": "success", "checkpoints": []}
        if path == "template-types":
            return {"status": "success", "templates": [{"id": template_id, **template}]}
        if path == "templates":
            return {"status": "success", "templates": catalog}
        if path == f"templates/{template_id}":
            return {"status": "success", "template": template}
        if path == "models":
            return {"models": LLM_MODELS, "default": DEFAULT_MODEL}
        if path == "settings/status":
            return {"has_api_key": False, "api_key_set": False, "api_keys": {}, "key_sources": {},
                    "provider_keys": {}, "execution_preference": "local", "generated_code_execution_enabled": False,
                    "execution_options": {"local": {"available": False}, "remote": {"available": False}}}
        if path == "user/status":
            return {"authenticated": True, "has_api_key": False, "user": {"uid": f"export-{digest}", "is_admin": False}}
        return JSONResponse({"status": "error", "message": "This route is unavailable in the read-only export preview."}, status_code=404)

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Saved evolution or annotated history JSON")
    parser.add_argument("--port", type=int, default=8136, help="Local port (default: 8136)")
    parser.add_argument("--watch", action="store_true", help="Refresh the preview when saved checkpoints change")
    args = parser.parse_args(argv)
    try:
        app = create_app(json.loads(args.input.read_text()), args.input.stem,
                         source_path=args.input if args.watch else None)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Read-only export preview: http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
