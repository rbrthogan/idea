# viewer.py
import logging
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, Response
import os
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import hashlib
import time
import math
import stat
import traceback
from pathlib import Path
import json
from pydantic import BaseModel
from datetime import datetime
import uuid
from typing import Dict, List, Tuple, Optional, Any, Set

from idea.evolution import EvolutionEngine, get_default_template_id
from idea.evolution_types import EvolutionRunConfig
from idea.evaluation import (
    ArtifactEvidenceError,
    ArtifactRef,
    ArtifactStoreLoader,
    ComponentKind,
    EvaluationStatus,
    GeminiCodeExecutionRunner,
    LLMPairwiseEvaluator,
    is_builtin_computed_runner,
    resolve_evaluation_suite,
)
from idea.artifact_store import ArtifactStore, ArtifactStoreError
from idea.llm import Critic
from idea.swiss_tournament import elo_update, generate_swiss_round_pairs
from idea.config import (
    LLM_MODELS,
    DEFAULT_MODEL,
    DEFAULT_CREATIVE_TEMP,
    DEFAULT_TOP_P,
    THINKING_MODEL_CONFIG,
    estimate_generation_token_cost,
    SMTP_SERVER,
    SMTP_PORT,
    SMTP_USERNAME,
    SMTP_PASSWORD,
    ADMIN_EMAIL,
    RUN_LEASE_SECONDS,
    RUN_HEARTBEAT_SECONDS,
    RUN_STATE_WRITE_INTERVAL_SECONDS,
    GLOBAL_MAX_ACTIVE_RUNS,
)
from idea.template_manager import router as template_router
from idea.prompts.loader import list_available_templates
from idea.admin import router as admin_router
from idea.auth import require_auth, UserInfo, get_current_user, get_dev_auth_client_config
from fastapi import Depends
from idea import database as db
from idea.user_state import user_states, UserEvolutionState
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

# --- Initialize and configure FastAPI ---
app = FastAPI()

# Same-origin is the safe default. Deployments with a separate frontend can
# opt in to a comma-separated allow-list; wildcard browser access is never
# enabled implicitly, especially when localhost development auth is active.
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("IDEA_CORS_ORIGINS", "").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# Include routers
app.include_router(template_router)
app.include_router(admin_router)

# Mount static folder with custom config
app.mount("/static", StaticFiles(directory="idea/static"), name="static")

# Templates
templates = Jinja2Templates(directory="idea/static/html")

# Add this near other constants
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

LOCAL_EVOLUTION_USERS_DIR = "local-users"
ALLOW_UNSCOPED_LEGACY_LOCAL_FILES_ENV = "IDEA_ALLOW_UNSCOPED_LEGACY_LOCAL_FILES"
ENABLE_GEMINI_CODE_ARTIFACTS_ENV = "IDEA_ENABLE_GEMINI_CODE_ARTIFACTS"
ARTIFACT_STORE_ROOT_ENV = "IDEA_ARTIFACT_STORE_ROOT"

# Unified evolutions directory
EVOLUTIONS_DIR = Path("data/evolutions")
EVOLUTIONS_DIR.mkdir(exist_ok=True)

# Instance identity (used for run leasing/coordination)
INSTANCE_ID = os.getenv("HOSTNAME") or f"local-{uuid.uuid4()}"
ACTIVE_RUN_STATUSES = {"starting", "in_progress", "resuming", "continuing", "stopping"}
TRANSIENT_PROGRESS_FLAGS = {"oracle_update", "elite_selection_update", "checkpoint_saved"}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _update_autorating_state(state: Optional[UserEvolutionState], updates: Dict[str, Any]) -> None:
    """Merge updates into per-user autorating progress with monotonic versioning."""
    if state is None:
        return

    current = state.autorating_status if isinstance(state.autorating_status, dict) else {}
    next_payload = current.copy()
    next_payload.update(updates)
    next_payload["version"] = int(current.get("version", 0)) + 1
    next_payload["updated_at"] = _now_ms()
    state.autorating_status = next_payload


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _normalize_tournament_count(value: float) -> float:
    # UI uses 0.25 increments; normalize backend input to the same grid.
    return max(0.25, round(value * 4) / 4)


def _resolve_tournament_settings(
    pop_size: int,
    tournament_count_input: Any,
    legacy_rounds_input: Any,
) -> Tuple[float, int, int]:
    full_tournament_rounds = max(1, pop_size - 1)

    if tournament_count_input is not None:
        tournament_count = _normalize_tournament_count(float(tournament_count_input))
        target_tournament_rounds = max(
            1,
            _round_half_up(tournament_count * full_tournament_rounds),
        )
        return tournament_count, full_tournament_rounds, target_tournament_rounds

    legacy_rounds = int(legacy_rounds_input if legacy_rounds_input is not None else full_tournament_rounds)
    target_tournament_rounds = max(1, legacy_rounds)
    tournament_count = target_tournament_rounds / full_tournament_rounds
    return tournament_count, full_tournament_rounds, target_tournament_rounds


def _derive_run_status(update_data: Dict[str, Any]) -> str:
    if update_data.get("error"):
        return "error"
    if update_data.get("is_running") is False:
        if update_data.get("is_stopped") or update_data.get("is_resumable"):
            return "paused"
        return "complete"
    if update_data.get("is_resuming"):
        return "resuming"
    if update_data.get("is_continuing"):
        return "continuing"
    return "in_progress"


def _is_run_stale(run_data: Dict[str, Any], now_ms: int) -> bool:
    if not run_data:
        return False
    if run_data.get("status") not in ACTIVE_RUN_STATUSES:
        return False
    return run_data.get("lease_expires_at_ms", 0) <= now_ms


def _should_honor_remote_stop(run_state: Optional[Dict[str, Any]]) -> bool:
    """
    Determine whether a run-state stop flag should stop the in-memory engine.

    We only honor explicit active stop requests (`status == \"stopping\"`) to avoid
    stale stop flags from previous runs immediately stopping newly started runs.
    """
    if not run_state:
        return False
    if not run_state.get("stop_requested"):
        return False
    return run_state.get("status") == "stopping"


async def _claim_run_slot(user_id: str, evolution_id: str, evolution_name: str,
                          total_generations: int, start_time: str,
                          tournament_count: Optional[float] = None,
                          full_tournament_rounds: Optional[int] = None,
                          target_tournament_rounds: Optional[int] = None) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Claim the per-user run slot (and global lock if enabled)."""
    run_payload = {
        "evolution_id": evolution_id,
        "evolution_name": evolution_name,
        "current_generation": 0,
        "total_generations": total_generations,
        "progress": 0,
        "start_time": start_time,
        "status": "starting",
        "status_message": "Starting evolution...",
        "tournament_count": tournament_count,
        "full_tournament_rounds": full_tournament_rounds,
        "target_tournament_rounds": target_tournament_rounds,
        "stop_requested": False,
    }

    if GLOBAL_MAX_ACTIVE_RUNS == 1:
        lock_result = await db.claim_global_run_lock(
            owner_id=INSTANCE_ID,
            user_id=user_id,
            evolution_id=evolution_id,
            lease_seconds=RUN_LEASE_SECONDS,
        )
        if not lock_result.get("ok"):
            return False, lock_result.get("data"), "global"

    claim_result = await db.claim_active_run(
        user_id=user_id,
        run_data=run_payload,
        lease_seconds=RUN_LEASE_SECONDS,
        owner_id=INSTANCE_ID,
    )

    if not claim_result.get("ok"):
        if GLOBAL_MAX_ACTIVE_RUNS == 1:
            await db.release_global_run_lock(INSTANCE_ID)
        return False, claim_result.get("data"), "user"

    return True, claim_result.get("data"), None


async def _refresh_run_state(user_id: str, state: UserEvolutionState, update_data: Dict[str, Any],
                             force: bool = False) -> None:
    """Persist lightweight run state for reconnect/resume."""
    if not state.run_owner_id:
        return

    try:
        now_monotonic = time.monotonic()
        if not force and (now_monotonic - state.run_last_write) < RUN_STATE_WRITE_INTERVAL_SECONDS:
            return

        run_status = _derive_run_status(update_data)
        status_message = update_data.get("status_message") or state.status.get("status_message", "")
        progress = update_data.get("progress", state.status.get("progress", 0))
        current_generation = update_data.get("current_generation", state.status.get("current_generation", 0))
        total_generations = update_data.get("total_generations", state.status.get("total_generations", 0))
        checkpoint_id = update_data.get("checkpoint_id") or state.status.get("checkpoint_id")

        run_update = {
            "status": run_status,
            "status_message": status_message,
            "progress": progress,
            "current_generation": current_generation,
            "total_generations": total_generations,
            "history_version": update_data.get("history_version", state.history_version),
            "is_running": update_data.get("is_running", run_status in ACTIVE_RUN_STATUSES),
            "is_stopped": update_data.get("is_stopped"),
            "is_resumable": update_data.get("is_resumable"),
            "checkpoint_id": checkpoint_id,
            "evolution_id": getattr(state.engine, "evolution_id", None),
            "evolution_name": getattr(state.engine, "evolution_name", None),
            "start_time": state.status.get("start_time"),
            "tournament_count": update_data.get("tournament_count", state.status.get("tournament_count")),
            "full_tournament_rounds": update_data.get("full_tournament_rounds", state.status.get("full_tournament_rounds")),
            "target_tournament_rounds": update_data.get("target_tournament_rounds", state.status.get("target_tournament_rounds")),
        }

        for field_name in (
            "token_counts",
            "diversity_history",
            "island_diversity_history",
            "migration_history",
        ):
            value = update_data.get(field_name, state.status.get(field_name))
            if value is not None:
                if field_name == "token_counts":
                    value = _compact_token_counts_for_run_state(value)
                run_update[field_name] = convert_uuids_to_strings(value)

        if update_data.get("error"):
            run_update["last_error"] = update_data.get("error")

        await db.update_active_run(
            user_id=user_id,
            updates=run_update,
            lease_seconds=RUN_LEASE_SECONDS,
            owner_id=state.run_owner_id,
        )

        if GLOBAL_MAX_ACTIVE_RUNS == 1:
            await db.refresh_global_run_lock(state.run_owner_id, RUN_LEASE_SECONDS)

        state.run_last_write = now_monotonic
    except Exception:
        logger.warning("Failed to persist run state", exc_info=True)


def _merge_status_update(state: UserEvolutionState, update_data: Dict[str, Any]) -> None:
    """
    Merge a progress update into state.status while keeping event-like flags transient.

    Flags such as oracle/elite/checkpoint updates should only be true on the update
    that emits them; they must not remain sticky across subsequent polls.
    """
    for flag in TRANSIENT_PROGRESS_FLAGS:
        if flag not in update_data:
            state.status.pop(flag, None)
    state.status.update(update_data)


def _compact_token_counts_for_run_state(token_counts: Any) -> Any:
    """Keep active-run cost fresh without writing large diagnostic event logs every second."""
    if not isinstance(token_counts, dict):
        return token_counts
    compact = {
        key: value
        for key, value in token_counts.items()
        if key != "diagnostics"
    }
    diagnostics = token_counts.get("diagnostics")
    if isinstance(diagnostics, dict):
        compact["diagnostics"] = {
            "agents": diagnostics.get("agents", {}),
            "totals": diagnostics.get("totals", {}),
            "events_truncated": bool(diagnostics.get("events")),
        }
    return compact


def _attach_live_engine_progress_fields(state: UserEvolutionState) -> None:
    """Keep live progress payloads aligned with the current engine state."""
    engine = state.engine
    if engine is None:
        return

    if hasattr(engine, "tournament_history"):
        state.status["tournament_history"] = engine.tournament_history
    for field_name in ("diversity_history", "island_diversity_history", "migration_history"):
        value = getattr(engine, field_name, None)
        if value is not None:
            state.status[field_name] = value.copy() if isinstance(value, list) else value

    get_total_token_count = getattr(engine, "get_total_token_count", None)
    if callable(get_total_token_count):
        state.status["token_counts"] = get_total_token_count()


def _serialize_engine_history(engine: Any) -> List[List[Dict[str, Any]]]:
    history = getattr(engine, "history", None)
    if not isinstance(history, list):
        return []
    return [[idea_to_dict(idea) for idea in generation] for generation in history]


def _get_live_evolution_payload(
    state: UserEvolutionState,
    evolution_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return the in-memory evolution snapshot when it matches an evolution id."""
    engine = state.engine
    live_id = state.status.get("evolution_id") or getattr(engine, "evolution_id", None)
    if not live_id:
        return None
    if evolution_id and live_id != evolution_id:
        return None

    latest_history = state.latest_data if isinstance(state.latest_data, list) else []
    engine_history = _serialize_engine_history(engine) if engine is not None else []
    history = engine_history if len(engine_history) > len(latest_history) else latest_history
    if not history:
        return None

    if history is not state.latest_data:
        state.latest_data = history

    current_generation = state.status.get(
        "current_generation",
        getattr(engine, "current_generation", max(0, len(history) - 1)),
    )
    total_generations = state.status.get(
        "total_generations",
        getattr(engine, "generations", current_generation),
    )

    payload: Dict[str, Any] = {
        "evolution_id": live_id,
        "name": state.status.get("evolution_name")
        or getattr(engine, "evolution_name", live_id),
        "status": _derive_run_status(state.status),
        "config": _engine_config_payload(engine) if engine is not None else {},
        "current_generation": current_generation,
        "total_generations": total_generations,
        "history": history,
        "history_version": state.history_version,
        "contexts": state.status.get("contexts", getattr(engine, "contexts", [])),
        "specific_prompts": state.status.get(
            "specific_prompts",
            getattr(engine, "specific_prompts", []),
        ),
        "breeding_prompts": state.status.get(
            "breeding_prompts",
            getattr(engine, "breeding_prompts", []),
        ),
        "tournament_history": state.status.get(
            "tournament_history",
            getattr(engine, "tournament_history", []),
        ),
        "diversity_history": state.status.get(
            "diversity_history",
            getattr(engine, "diversity_history", []),
        ),
        "island_diversity_history": state.status.get(
            "island_diversity_history",
            getattr(engine, "island_diversity_history", []),
        ),
        "migration_history": state.status.get(
            "migration_history",
            getattr(engine, "migration_history", []),
        ),
        "token_counts": state.status.get("token_counts"),
    }

    get_total_token_count = getattr(engine, "get_total_token_count", None)
    if payload["token_counts"] is None and callable(get_total_token_count):
        payload["token_counts"] = get_total_token_count()

    return payload


def _engine_config_payload(engine: EvolutionEngine) -> Dict[str, Any]:
    return EvolutionRunConfig.from_engine(engine).to_persistence_dict()


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _artifact_store_path(user_id: str, evolution_id: str) -> Path:
    root = Path(os.getenv(ARTIFACT_STORE_ROOT_ENV, "data/artifacts"))
    user_partition = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:32]
    evolution_partition = hashlib.sha256(evolution_id.encode("utf-8")).hexdigest()[:32]
    return root / user_partition / evolution_partition


def _artifact_store_for_evolution(
    user_id: str,
    evolution_id: str,
    *,
    create: bool,
) -> Optional[ArtifactStore]:
    store_path = _artifact_store_path(user_id, evolution_id)
    if not create and not store_path.is_dir():
        return None
    return ArtifactStore(store_path)


def _web_artifact_runtime(
    *,
    user_id: str,
    evolution_id: str,
    api_key: str,
) -> tuple[Dict[str, Any], Optional[ArtifactStore]]:
    if not _env_enabled(ENABLE_GEMINI_CODE_ARTIFACTS_ENV):
        return {}, None
    store = _artifact_store_for_evolution(user_id, evolution_id, create=True)
    return {
        "llm_code_artifact": GeminiCodeExecutionRunner(
            enabled=True,
            api_key=api_key,
        )
    }, store


def _live_or_persisted_artifact_store(
    state: Optional[UserEvolutionState],
    *,
    user_id: str,
    evolution_id: str,
) -> Optional[Any]:
    engine = getattr(state, "engine", None) if state is not None else None
    if (
        engine is not None
        and getattr(engine, "evolution_id", None) == evolution_id
        and getattr(engine, "artifact_store", None) is not None
    ):
        return engine.artifact_store
    return _artifact_store_for_evolution(user_id, evolution_id, create=False)


def _web_evaluation_component_available(component: Any) -> bool:
    """Return whether the web process has a deliberately safe runner boundary."""

    if not getattr(component, "enabled", False):
        return True
    kind = getattr(getattr(component, "kind", None), "value", None)
    if kind in {"llm_pairwise", "embedding_diversity"}:
        return True
    if kind == "llm_code_artifact":
        return _env_enabled(ENABLE_GEMINI_CODE_ARTIFACTS_ENV)
    return kind == "computed" and is_builtin_computed_runner(
        getattr(component, "runner_ref", "")
    )


def _autorating_pairwise_evaluator(
    evolution_data: Dict[str, Any],
    *,
    artifact_loader: Optional[ArtifactStoreLoader] = None,
) -> LLMPairwiseEvaluator:
    """Resolve the saved run's canonical pairwise evaluation policy."""

    config = evolution_data.get("config")
    if not isinstance(config, dict):
        config = {}
    try:
        fitness_alpha = float(config.get("fitness_alpha", 0.7))
    except (TypeError, ValueError):
        fitness_alpha = 0.7
    suite = resolve_evaluation_suite(
        config.get("evaluation"),
        fitness_alpha=fitness_alpha,
    )
    if not suite.enabled:
        raise ValueError("this evolution has evaluation disabled")
    component = next(
        (
            candidate
            for candidate in suite.components
            if candidate.enabled and candidate.kind == ComponentKind.LLM_PAIRWISE
        ),
        None,
    )
    if component is None:
        raise ValueError("this evolution does not configure pairwise LLM evaluation")
    artifact_requirements = [
        requirement
        for requirement in component.requires
        if requirement.startswith("artifact.")
    ]
    if artifact_requirements:
        if artifact_loader is None:
            raise ValueError(
                "artifact-dependent auto-rating needs the evolution's authenticated "
                "artifact store, but it is unavailable"
            )

    # The auto-rate request owns its number of Swiss rounds.  Everything else
    # (component identity, evidence requirements, metric and failure policy)
    # comes from the saved run configuration.
    return LLMPairwiseEvaluator(
        component.model_copy(update={"rounds": None}),
        artifact_loader=artifact_loader,
    )


def _get_live_engine_for_identifier(
    state: UserEvolutionState,
    identifier: Optional[str],
) -> Optional[EvolutionEngine]:
    engine = state.engine
    if engine is None or not identifier:
        return None
    identifiers = {
        getattr(engine, "evolution_id", None),
        getattr(engine, "checkpoint_id", None),
    }
    if identifier in identifiers:
        return engine
    return None


def _history_total_ideas(history: Any) -> int:
    if not isinstance(history, list):
        return 0
    return sum(len(generation) for generation in history if isinstance(generation, list))


def _apply_history_capabilities(item: Dict[str, Any]) -> None:
    status = item.get("status")
    item["can_resume"] = status in ["paused", "in_progress", "force_stopped"]
    item["can_continue"] = status == "complete"
    item["can_rate"] = item.get("generations", 0) > 0


async def _heartbeat_loop(user_id: str, state: UserEvolutionState) -> None:
    """Background heartbeat to keep leases alive during long steps."""
    try:
        while True:
            await asyncio.sleep(RUN_HEARTBEAT_SECONDS)
            if state.engine is None or not state.status.get("is_running"):
                return
            if not state.run_owner_id:
                continue
            # Check for stop request from another instance
            run_state = await db.get_active_run(user_id)
            if _should_honor_remote_stop(run_state) and not state.engine.stop_requested:
                state.engine.stop_evolution()
                await db.update_active_run(
                    user_id=user_id,
                    updates={"status": "stopping", "status_message": "Stop requested - waiting for safe point"},
                    lease_seconds=RUN_LEASE_SECONDS,
                    owner_id=state.run_owner_id,
                )
            await db.update_active_run(
                user_id=user_id,
                updates={},
                lease_seconds=RUN_LEASE_SECONDS,
                owner_id=state.run_owner_id,
            )
            if GLOBAL_MAX_ACTIVE_RUNS == 1:
                await db.refresh_global_run_lock(state.run_owner_id, RUN_LEASE_SECONDS)
    except asyncio.CancelledError:
        return


async def _finalize_run_state(user_id: str, state: UserEvolutionState, update_data: Dict[str, Any]) -> None:
    """Mark run as finished and release global lock."""
    try:
        await _refresh_run_state(user_id, state, update_data, force=True)
        await db.update_active_run(
            user_id=user_id,
            updates={"is_running": False, "active": False, "stop_requested": False},
            lease_seconds=0,
            owner_id=state.run_owner_id,
        )
        if GLOBAL_MAX_ACTIVE_RUNS == 1:
            await db.release_global_run_lock(state.run_owner_id)
    except Exception:
        logger.warning("Failed to finalize run state", exc_info=True)
    finally:
        state.stop_heartbeat()


class ApiKeyRequest(BaseModel):
    api_key: str


class ContactRequest(BaseModel):
    name: str
    email: str
    message: str

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/")
def serve_viewer(request: Request):
    """Serves the viewer page"""
    return templates.TemplateResponse(
        request,
        "viewer.html",
        {
            "dev_auth_config": get_dev_auth_client_config(request),
            "llm_models": LLM_MODELS,
            "default_model": DEFAULT_MODEL,
            "thinking_config": THINKING_MODEL_CONFIG,
            "gemini_code_artifacts_enabled": _env_enabled(
                ENABLE_GEMINI_CODE_ARTIFACTS_ENV
            ),
        },
    )

@app.get("/rate")
def serve_rater(request: Request):
    """Serves the rater page"""
    return templates.TemplateResponse(
        request,
        "rater.html",
        {"dev_auth_config": get_dev_auth_client_config(request)},
    )

@app.get("/api/user/status")
async def get_user_status(user: UserInfo = Depends(require_auth)):
    """Get the status of the current user (e.g. has_api_key)"""
    api_key = await db.get_user_api_key(user.uid)
    return {
        "has_api_key": api_key is not None
    }

@app.post("/api/settings/api-key")
async def set_api_key(request: ApiKeyRequest, user: UserInfo = Depends(require_auth)):
    """Save user's encoded API key"""
    api_key = request.api_key.strip()
    if not api_key:
        return JSONResponse({"status": "error", "message": "API key cannot be empty"}, status_code=400)

    try:
        await db.save_user_api_key(user.uid, api_key)
        return JSONResponse({"status": "success", "message": "API Key saved successfully"})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/api/settings/status")
async def get_settings_status(user: UserInfo = Depends(require_auth)):
    """Check if API key is set"""
    try:
        api_key = await db.get_user_api_key(user.uid)
        is_missing = api_key is None
        masked_key = None
        if not is_missing:
            if len(api_key) > 8:
                masked_key = f"{api_key[:4]}...{api_key[-4:]}"
            else:
                masked_key = "***"

        return JSONResponse({
            "api_key_missing": is_missing,
            "masked_key": masked_key,
            "is_admin": user.is_admin,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "status": "error",
            "message": f"Internal Server Error: {str(e)}"
        }, status_code=500)




@app.post("/api/contact")
async def send_contact_email(request: ContactRequest):
    """
    Send a contact email to the admin.
    """
    try:
        if not SMTP_USERNAME or not SMTP_PASSWORD or not ADMIN_EMAIL:
            logger.info(
                "SMTP not configured. Message received from %s <%s>: %s",
                request.name,
                request.email,
                request.message,
            )
            # Return success even if email not sent, as we logged it (simulated behavior for dev)
            return JSONResponse({
                "status": "success",
                "message": "Message received! (SMTP not configured, checked logs)"
            })

        msg = MIMEMultipart()
        msg['From'] = SMTP_USERNAME
        msg['To'] = ADMIN_EMAIL
        msg['Subject'] = f"Idea App Contact: {request.name}"
        msg['Reply-To'] = request.email

        body = f"Name: {request.name}\nEmail: {request.email}\n\nMessage:\n{request.message}"
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        text = msg.as_string()
        server.sendmail(SMTP_USERNAME, ADMIN_EMAIL, text)
        server.quit()

        return JSONResponse({"status": "success", "message": "Email sent successfully"})
    except Exception as e:
        logger.exception("Error sending contact email")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/api/template-types")
async def get_template_types(user: UserInfo = Depends(require_auth)):
    """Get available template types for the evolution UI dropdown"""
    try:
        templates = list_available_templates()

        # System templates first
        combined_templates: Dict[str, Dict[str, Any]] = {}
        for template_id, template_info in templates.items():
            if 'error' in template_info:
                continue
            combined_templates[template_id] = {
                "id": template_id,
                "name": template_info.get('name', template_id.replace('_', ' ').title()),
                "description": template_info.get('description', ''),
                "type": template_info.get('type', 'unknown'),
                "author": template_info.get('author', 'Unknown'),
                "is_system": True,
            }

        # Add user templates from Firestore
        user_templates = await db.list_user_templates(user.uid)
        for template in user_templates:
            template_id = template.get("id")
            if not template_id or template_id in combined_templates:
                continue
            combined_templates[template_id] = {
                "id": template_id,
                "name": template.get("name", template_id.replace("_", " ").title()),
                "description": template.get("description", ""),
                "type": "custom",
                "author": template.get("author", "User"),
                "is_system": False,
            }

        # Sort by name
        template_types = list(combined_templates.values())
        template_types.sort(key=lambda x: x['name'].lower())

        return JSONResponse({
            "status": "success",
            "templates": template_types
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/api/start-evolution")
async def start_evolution(request: Request, user: UserInfo = Depends(require_auth)):
    """
    Runs the complete evolution and returns the final results
    """
    try:
        # Get user-specific state
        state = await user_states.get(user.uid)

        # Check if this user already has an evolution running
        if state.status.get("is_running") and state.engine is not None:
            return JSONResponse(
                {"status": "error", "message": "You already have an evolution running. Please wait for it to complete or stop it first."},
                status_code=409,
            )

        # Check for API key in DB
        api_key = await db.get_user_api_key(user.uid)
        if not api_key:
            return JSONResponse(
                {"status": "error", "message": "API Key not configured. Please set it in Settings."},
                status_code=400,
            )

        data = await request.json()
        if not isinstance(data, dict):
            return JSONResponse(
                {"status": "error", "message": "Request body must be a JSON object"},
                status_code=400,
            )
        logger.info(
            "Received start-evolution request",
            extra={"request_field_count": len(data)},
        )

        # Clear the latest evolution data when starting a new evolution
        state.latest_data = []

        pop_size = int(data.get('popSize', 3))
        generations = int(data.get('generations', 2))
        idea_type = data.get('ideaType', get_default_template_id())
        model_type = data.get('modelType', 'gemini-2.0-flash-lite')

        # Get creative and tournament parameters with defaults
        try:
            creative_temp = float(data.get('creativeTemp', DEFAULT_CREATIVE_TEMP))
            top_p = float(data.get('topP', DEFAULT_TOP_P))
            print(f"Parsed creative values: temp={creative_temp}, top_p={top_p}")
        except ValueError as e:
            print(f"Error parsing creative values: {e}")
            # Use defaults if parsing fails
            creative_temp = DEFAULT_CREATIVE_TEMP
            top_p = DEFAULT_TOP_P

        # Tournament semantics:
        # - full tournament rounds = pop_size - 1
        # - tournamentCount controls how many full tournaments (with fractional support)
        try:
            tournament_count, full_tournament_rounds, target_tournament_rounds = _resolve_tournament_settings(
                pop_size=pop_size,
                tournament_count_input=data.get('tournamentCount'),
                legacy_rounds_input=data.get('tournamentRounds'),
            )
            print(
                "Resolved tournament settings:",
                f"count={tournament_count}, full_rounds={full_tournament_rounds}, target_rounds={target_tournament_rounds}",
            )
        except Exception as e:
            print(f"Error resolving tournament settings: {e}")
            tournament_count = 1.0
            full_tournament_rounds = max(1, pop_size - 1)
            target_tournament_rounds = full_tournament_rounds

        # Get mutation rate with default
        try:
            mutation_rate = float(data.get('mutationRate', 0.2))
            print(f"Parsed mutation rate: {mutation_rate}")
        except ValueError as e:
            print(f"Error parsing mutation rate: {e}")
            mutation_rate = 0.2

        seed_context_pool_size = data.get('seedContextPoolSize')
        if seed_context_pool_size is not None:
            try:
                seed_context_pool_size = max(1, int(seed_context_pool_size))
                print(f"Parsed seed context pool size: {seed_context_pool_size}")
            except (TypeError, ValueError):
                print(f"Error parsing seed context pool size: {seed_context_pool_size}")
                seed_context_pool_size = None
        else:
            print("No seed context pool size specified (using default auto behavior)")

        try:
            replacement_rate = float(data.get('replacementRate', 0.5))
            print(f"Parsed replacement rate: {replacement_rate}")
        except ValueError as e:
            print(f"Error parsing replacement rate: {e}")
            replacement_rate = 0.5

        try:
            fitness_alpha = float(data.get('fitnessAlpha', 0.7))
            print(f"Parsed fitness alpha: {fitness_alpha}")
        except ValueError as e:
            print(f"Error parsing fitness alpha: {e}")
            fitness_alpha = 0.7

        try:
            age_decay_rate = float(data.get('ageDecayRate', 0.25))
            print(f"Parsed age decay rate: {age_decay_rate}")
        except ValueError as e:
            print(f"Error parsing age decay rate: {e}")
            age_decay_rate = 0.25

        try:
            age_decay_floor = float(data.get('ageDecayFloor', 0.35))
            age_decay_floor = max(0.0, min(1.0, age_decay_floor))
            print(f"Parsed age decay floor: {age_decay_floor}")
        except (TypeError, ValueError) as e:
            print(f"Error parsing age decay floor: {e}")
            age_decay_floor = 0.35

        raw_refine_all_ideas = data.get('refineAllIdeas', False)
        if isinstance(raw_refine_all_ideas, bool):
            refine_all_ideas = raw_refine_all_ideas
        else:
            refine_all_ideas = str(raw_refine_all_ideas).strip().lower() not in {
                "false",
                "0",
                "no",
                "off",
            }

        try:
            survivor_refinement_rate = float(data.get('survivorRefinementRate', 0.1))
            survivor_refinement_rate = max(0.0, min(1.0, survivor_refinement_rate))
            print(f"Parsed survivor refinement rate: {survivor_refinement_rate}")
        except (TypeError, ValueError) as e:
            print(f"Error parsing survivor refinement rate: {e}")
            survivor_refinement_rate = 0.1

        try:
            island_count = int(data.get('islandCount', 2))
            island_count = max(1, min(5, island_count))
            island_count = min(island_count, max(1, pop_size))
            print(f"Parsed island count: {island_count}")
        except (TypeError, ValueError) as e:
            print(f"Error parsing island count: {e}")
            island_count = min(2, max(1, pop_size))

        try:
            migration_rate = float(data.get('migrationRate', 0.05))
            migration_rate = max(0.0, min(1.0, migration_rate))
            print(f"Parsed migration rate: {migration_rate}")
        except (TypeError, ValueError) as e:
            print(f"Error parsing migration rate: {e}")
            migration_rate = 0.05

        try:
            migration_novelty_protection_quantile = float(
                data.get('migrationNoveltyProtectionQuantile', 0.8)
            )
            migration_novelty_protection_quantile = max(
                0.0,
                min(1.0, migration_novelty_protection_quantile),
            )
            print(
                "Parsed migration novelty protection quantile: "
                f"{migration_novelty_protection_quantile}"
            )
        except (TypeError, ValueError) as e:
            print(f"Error parsing migration novelty protection quantile: {e}")
            migration_novelty_protection_quantile = 0.8

        # Thinking controls (model-aware): supports explicit level or custom budget.
        thinking_level = _normalize_thinking_level(data.get("thinkingLevel"))
        if data.get("thinkingLevel") is not None and thinking_level is None:
            print(f"Ignoring invalid thinkingLevel value: {data.get('thinkingLevel')!r}")

        thinking_budget = None
        raw_thinking_budget = data.get("thinkingBudget")
        if raw_thinking_budget is not None and raw_thinking_budget != "":
            try:
                thinking_budget = int(raw_thinking_budget)
            except (TypeError, ValueError):
                print(f"Ignoring invalid thinkingBudget value: {raw_thinking_budget!r}")

        model_thinking_cfg = THINKING_MODEL_CONFIG.get(model_type, {})
        supports_thinking = bool(model_thinking_cfg.get("supports_thinking", False))
        if not supports_thinking:
            thinking_level = None
            thinking_budget = None
        else:
            allowed_levels = set(model_thinking_cfg.get("allowed_levels") or [])
            if thinking_level is not None and allowed_levels and thinking_level not in allowed_levels:
                print(
                    "Ignoring thinking level unsupported by model:",
                    f"model={model_type}, thinking_level={thinking_level}",
                )
                thinking_level = None

            if thinking_budget is not None:
                # Explicit custom budget takes precedence over level presets.
                thinking_level = None
            elif thinking_level == "off" and not bool(model_thinking_cfg.get("allow_off", False)):
                thinking_level = model_thinking_cfg.get("off_replacement_level", "low")

            if thinking_level is None and thinking_budget is None:
                thinking_level, thinking_budget = _get_default_thinking_settings(model_type)
        print(
            "Resolved thinking settings:",
            f"model={model_type}, thinking_level={thinking_level}, thinking_budget={thinking_budget}",
        )

        # Get max budget parameter
        max_budget = data.get('maxBudget')
        if max_budget is not None:
            try:
                max_budget = float(max_budget)
                print(f"Parsed max budget: ${max_budget}")
            except ValueError:
                print(f"Error parsing max budget: {max_budget}")
                max_budget = None
        else:
            print("No max budget specified")

        print(f"Starting evolution with pop_size={pop_size}, generations={generations}, "
              f"idea_type={idea_type}, model_type={model_type}, "
              f"creative_temp={creative_temp}, top_p={top_p}, "
              f"tournament: count={tournament_count}, full_rounds={full_tournament_rounds}, target_rounds={target_tournament_rounds}, "
              f"mutation_rate={mutation_rate}, seed_context_pool_size={seed_context_pool_size}, replacement_rate={replacement_rate}, "
              f"fitness_alpha={fitness_alpha}, age_decay_rate={age_decay_rate}, "
              f"refine_all_ideas={refine_all_ideas}, survivor_refinement_rate={survivor_refinement_rate}, "
              f"island_count={island_count}, migration_rate={migration_rate}, "
              f"migration_novelty_protection_quantile={migration_novelty_protection_quantile}, "
              f"thinking_level={thinking_level}, thinking_budget={thinking_budget}, max_budget={max_budget}")

        # Resolve template source. System templates are bundled; custom templates live in Firestore.
        template_data = None
        system_templates = list_available_templates()
        is_valid_system_template = (
            idea_type in system_templates and 'error' not in system_templates.get(idea_type, {})
        )

        if not is_valid_system_template:
            user_template = await db.get_user_template(user.uid, idea_type)
            if user_template:
                print(f"Using custom template '{idea_type}' from Firestore")
                template_data = user_template
            else:
                return JSONResponse(
                    {"status": "error", "message": f"Template '{idea_type}' not found"},
                    status_code=400,
                )

        try:
            evaluation_config = resolve_evaluation_suite(
                data.get("evaluation"),
                fitness_alpha=fitness_alpha,
            )
        except (TypeError, ValueError) as exc:
            return JSONResponse(
                {
                    "status": "error",
                    "message": f"Invalid evaluation configuration: {exc}",
                },
                status_code=400,
            )

        unavailable_components = [
            component.id
            for component in evaluation_config.components
            if not _web_evaluation_component_available(component)
        ]
        if unavailable_components:
            return JSONResponse(
                {
                    "status": "error",
                    "message": (
                        "Evaluation execution backend is not configured for: "
                        + ", ".join(unavailable_components)
                    ),
                },
                status_code=400,
            )

        planned_evolution_id = str(uuid.uuid4())
        has_artifact_producer = any(
            component.enabled
            and component.kind == ComponentKind.LLM_CODE_ARTIFACT
            for component in evaluation_config.components
        )
        evaluation_runners: Dict[str, Any] = {}
        artifact_store: Optional[ArtifactStore] = None
        if has_artifact_producer:
            try:
                evaluation_runners, artifact_store = _web_artifact_runtime(
                    user_id=user.uid,
                    evolution_id=planned_evolution_id,
                    api_key=api_key,
                )
            except ArtifactStoreError as exc:
                logger.exception("Artifact store setup failed")
                return JSONResponse(
                    {
                        "status": "error",
                        "message": f"Artifact store is unavailable: {exc}",
                    },
                    status_code=500,
                )

        # Create the evolution engine (assigned after run claim)
        engine = EvolutionEngine(
            pop_size=pop_size,
            generations=generations,
            idea_type=idea_type,
            model_type=model_type,
            creative_temp=creative_temp,
            top_p=top_p,
            tournament_rounds=target_tournament_rounds,
            tournament_count=tournament_count,
            full_tournament_rounds=full_tournament_rounds,
            mutation_rate=mutation_rate,
            seed_context_pool_size=seed_context_pool_size,
            refine_all_ideas=refine_all_ideas,
            survivor_refinement_rate=survivor_refinement_rate,
            island_count=island_count,
            migration_rate=migration_rate,
            migration_novelty_protection_quantile=migration_novelty_protection_quantile,
            replacement_rate=replacement_rate,
            fitness_alpha=fitness_alpha,
            age_decay_rate=age_decay_rate,
            age_decay_floor=age_decay_floor,
            thinking_level=thinking_level,
            thinking_budget=thinking_budget,
            max_budget=max_budget,
            api_key=api_key,
            user_id=user.uid,  # Store user_id for Firestore scoping
            template_data=template_data,
            evaluation_config=evaluation_config,
            evaluation_runners=evaluation_runners,
            artifact_store=artifact_store,
        )

        # Initialize evolution with name (auto-generates if not provided)
        evolution_name = (data.get('evolutionName') or '').strip() or None
        engine.initialize_evolution(
            name=evolution_name,
            evolution_id=planned_evolution_id,
        )
        print(f"Evolution initialized: '{engine.evolution_name}' (ID: {engine.evolution_id})")

        # Claim run slot (and global lock if enabled)
        start_time = datetime.now().isoformat()
        ok, existing, scope = await _claim_run_slot(
            user_id=user.uid,
            evolution_id=engine.evolution_id,
            evolution_name=engine.evolution_name,
            total_generations=generations,
            start_time=start_time,
            tournament_count=engine.tournament_count,
            full_tournament_rounds=engine.full_tournament_rounds,
            target_tournament_rounds=engine.tournament_rounds,
        )
        if not ok:
            status = 429 if scope == "global" else 409
            message = "System busy. Another evolution is running. Try again shortly." if scope == "global" else \
                "You already have an evolution running. Please wait for it to complete or stop it first."
            return JSONResponse(
                {"status": "error", "message": message, "scope": scope, "active_run": existing},
                status_code=status,
            )

        # Bind engine to user state after claim
        state.reset_run_tracking()
        state.run_owner_id = INSTANCE_ID
        state.engine = engine

        # Clear the queue
        state.reset_queue()
        state.history_version = 0
        state.last_sent_history_version = -1

        # Set up evolution status
        state.status = {
            "current_generation": 0,
            "total_generations": generations,
            "is_running": True,
            "history": [],
            "history_version": state.history_version,
            "history_changed": False,
            "contexts": [],
            "progress": 0,
            "start_time": start_time,  # Track when evolution started
            "evolution_id": state.engine.evolution_id,
            "evolution_name": state.engine.evolution_name,
            "config": _engine_config_payload(state.engine),
            "tournament_count": state.engine.tournament_count,
            "full_tournament_rounds": state.engine.full_tournament_rounds,
            "target_tournament_rounds": state.engine.tournament_rounds,
            "refine_all_ideas": state.engine.refine_all_ideas,
            "survivor_refinement_rate": state.engine.survivor_refinement_rate,
            "island_count": state.engine.island_count,
            "migration_rate": state.engine.migration_rate,
            "migration_novelty_protection_quantile": state.engine.migration_novelty_protection_quantile,
        }

        # Put initial status in queue
        await state.queue.put(state.status.copy())

        # Start heartbeat to keep leases alive during long steps
        state.start_heartbeat(asyncio.create_task(_heartbeat_loop(user.uid, state)))

        # Start evolution in background task
        asyncio.create_task(run_evolution_task(state))

        return JSONResponse({
            "status": "success",
            "message": "Evolution started",
            "evolution_id": state.engine.evolution_id,
            "evolution_name": state.engine.evolution_name,
            "config": _engine_config_payload(state.engine),
            "contexts": [],
            "history_version": state.history_version,
            "history_changed": False,
            "tournament_count": state.engine.tournament_count,
            "full_tournament_rounds": state.engine.full_tournament_rounds,
            "target_tournament_rounds": state.engine.tournament_rounds,
            "refine_all_ideas": state.engine.refine_all_ideas,
            "survivor_refinement_rate": state.engine.survivor_refinement_rate,
            "island_count": state.engine.island_count,
            "migration_rate": state.engine.migration_rate,
            "migration_novelty_protection_quantile": state.engine.migration_novelty_protection_quantile,
            "specific_prompts": []  # Will be populated as evolution progresses
        })
    except Exception:
        logger.exception("Failed to start evolution")
        return JSONResponse({
            "status": "error",
            "message": "Failed to start evolution",
        }, status_code=500)

@app.post("/api/stop-evolution")
async def stop_evolution(user: UserInfo = Depends(require_auth)):
    """
    Request the evolution to stop gracefully
    """
    state = await user_states.get(user.uid)

    if state.engine is None:
        # Allow stop requests from non-owner instances via Firestore
        active_run = await db.get_active_run(user.uid)
        run_appears_active = bool(
            active_run and (
                active_run.get("status") in ACTIVE_RUN_STATUSES
                or active_run.get("is_running")
                or active_run.get("active")
            )
        )
        if not run_appears_active:
            return JSONResponse(
                {"status": "error", "message": "No evolution is currently running"},
                status_code=400,
            )
        # Do not refresh lease here; that can keep stale runs alive after worker crashes.
        await db.update_active_run(
            user_id=user.uid,
            updates={"status": "stopping", "status_message": "Stop requested", "stop_requested": True},
            owner_id=active_run.get("owner_id"),
        )
        return JSONResponse({
            "status": "success",
            "message": "Stop request queued - evolution will halt at the next safe point"
        })

    # Request stop
    state.engine.stop_evolution()

    if state.run_owner_id:
        await db.update_active_run(
            user_id=user.uid,
            updates={"status": "stopping", "status_message": "Stop requested - waiting for safe point", "stop_requested": True},
            lease_seconds=RUN_LEASE_SECONDS,
            owner_id=state.run_owner_id,
        )

    return JSONResponse({
        "status": "success",
        "message": "Stop request sent - evolution will halt at the next safe point"
    })

@app.post("/api/force-stop-evolution")
async def force_stop_evolution(user: UserInfo = Depends(require_auth)):
    """
    Force stop the evolution immediately, saving a checkpoint for later resumption.
    Use this when the graceful stop is taking too long.
    """
    state = await user_states.get(user.uid)

    if state.engine is None:
        # Handle stale/cross-instance runs where this process doesn't own an in-memory engine.
        active_run = await db.get_active_run(user.uid)
        run_appears_active = bool(
            active_run and (
                active_run.get("status") in ACTIVE_RUN_STATUSES
                or active_run.get("is_running")
                or active_run.get("active")
            )
        )
        if not run_appears_active:
            return JSONResponse(
                {"status": "error", "message": "No evolution is currently running"},
                status_code=400,
            )

        checkpoint_id = active_run.get("checkpoint_id")
        await db.update_active_run(
            user_id=user.uid,
            updates={
                "status": "force_stopped",
                "status_message": "Evolution force stopped",
                "is_running": False,
                "active": False,
                "is_stopped": True,
                "is_resumable": True,
                "stop_requested": False,
                "checkpoint_id": checkpoint_id,
            },
            lease_seconds=0,
            owner_id=active_run.get("owner_id"),
        )

        # Best-effort release in single-run mode.
        if GLOBAL_MAX_ACTIVE_RUNS == 1 and active_run.get("owner_id"):
            await db.release_global_run_lock(active_run.get("owner_id"))

        state.reset_run_tracking()
        state.status.update(
            {
                "is_running": False,
                "is_stopped": True,
                "is_resumable": True,
                "checkpoint_id": checkpoint_id,
                "status_message": "Evolution force stopped",
            }
        )

        return JSONResponse(
            {
                "status": "success",
                "message": "Stale run state force stopped.",
                "checkpoint_id": checkpoint_id,
                "is_resumable": True,
            }
        )

    # Save checkpoint before forcing stop
    checkpoint_id = None
    try:
        await state.engine.save_checkpoint(status='force_stopped')
        checkpoint_id = state.engine.checkpoint_id
    except Exception as e:
        print(f"Warning: Failed to save checkpoint during force stop: {e}")

    # Mark as stopped
    state.engine.stop_requested = True
    state.engine.is_stopped = True

    # Update status
    state.status["is_running"] = False
    state.status["is_stopped"] = True
    state.status["is_resumable"] = True
    state.status["checkpoint_id"] = checkpoint_id

    if state.run_owner_id:
        await _finalize_run_state(
            user_id=user.uid,
            state=state,
            update_data={
                "is_running": False,
                "is_stopped": True,
                "is_resumable": True,
                "checkpoint_id": checkpoint_id,
                "status_message": "Evolution force stopped",
            },
        )
        state.reset_run_tracking()

    return JSONResponse({
        "status": "success",
        "message": "Evolution force stopped. A checkpoint has been saved.",
        "checkpoint_id": checkpoint_id,
        "is_resumable": True
    })

@app.get("/api/checkpoints")
async def list_checkpoints_endpoint(user: UserInfo = Depends(require_auth)):
    """
    List all available evolution checkpoints for the current user.
    """
    try:
        checkpoints = await EvolutionEngine.list_checkpoints_for_user(user.uid)
        return JSONResponse({
            "status": "success",
            "checkpoints": checkpoints
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.get("/api/checkpoints/{checkpoint_id}")
async def get_checkpoint(
    checkpoint_id: str,
    user: UserInfo = Depends(require_auth),
):
    """
    Get details of a specific checkpoint.
    """
    try:
        data = await db.get_checkpoint(user.uid, checkpoint_id)
        if data is None:
            return JSONResponse({
                "status": "error",
                "message": "Checkpoint not found"
            }, status_code=404)

        return JSONResponse({
            "status": "success",
            "checkpoint": data
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.delete("/api/checkpoints/{checkpoint_id}")
async def delete_checkpoint(
    checkpoint_id: str,
    user: UserInfo = Depends(require_auth),
):
    """
    Delete a specific checkpoint.
    """
    try:
        success = await db.delete_checkpoint(user.uid, checkpoint_id)
        if success:
            return JSONResponse({
                "status": "success",
                "message": "Checkpoint deleted"
            })
        else:
            return JSONResponse({
                "status": "error",
                "message": "Checkpoint not found"
            }, status_code=404)
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/api/resume-evolution")
async def resume_evolution(request: Request, user: UserInfo = Depends(require_auth)):
    """
    Resume an evolution from a checkpoint.
    """
    # Get user-specific state
    state = await user_states.get(user.uid)

    # Check if already running
    if state.status.get("is_running"):
        return JSONResponse(
            {"status": "error", "message": "You already have an evolution running."},
            status_code=409,
        )

    # Check for API key in DB
    api_key = await db.get_user_api_key(user.uid)
    if not api_key:
        return JSONResponse(
            {"status": "error", "message": "API Key not configured."},
            status_code=400,
        )

    data = await request.json()
    checkpoint_id = data.get('checkpointId')

    if not checkpoint_id:
        return JSONResponse(
            {"status": "error", "message": "checkpointId is required"},
            status_code=400,
        )

    print(f"Resuming evolution from checkpoint: {checkpoint_id}")

    # Load the engine from Firestore
    evaluation_runners, artifact_store = _web_artifact_runtime(
        user_id=user.uid,
        evolution_id=checkpoint_id,
        api_key=api_key,
    )
    state.engine = await EvolutionEngine.load_evolution_for_user(
        user.uid,
        checkpoint_id,
        api_key=api_key,
        evaluation_runners=evaluation_runners,
        artifact_store=artifact_store,
    )
    if state.engine is None:
        return JSONResponse(
            {"status": "error", "message": "Failed to load evolution or checkpoint"},
            status_code=500,
        )
    if state.engine.unsupported_evaluation_components:
        unavailable = ", ".join(state.engine.unsupported_evaluation_components)
        state.engine = None
        return JSONResponse(
            {
                "status": "error",
                "message": f"Evaluation execution backend is not configured for: {unavailable}",
            },
            status_code=400,
        )

    # Claim run slot (and global lock if enabled)
    start_time = datetime.now().isoformat()
    ok, existing, scope = await _claim_run_slot(
        user_id=user.uid,
        evolution_id=state.engine.evolution_id,
        evolution_name=state.engine.evolution_name,
        total_generations=state.engine.generations,
        start_time=start_time,
        tournament_count=state.engine.tournament_count,
        full_tournament_rounds=state.engine.full_tournament_rounds,
        target_tournament_rounds=state.engine.tournament_rounds,
    )
    if not ok:
        status = 429 if scope == "global" else 409
        message = "System busy. Another evolution is running. Try again shortly." if scope == "global" else \
            "You already have an evolution running."
        state.engine = None
        state.reset_run_tracking()
        return JSONResponse(
            {"status": "error", "message": message, "scope": scope, "active_run": existing},
            status_code=status,
        )

    state.reset_run_tracking()
    state.run_owner_id = INSTANCE_ID
    await db.update_active_run(
        user_id=user.uid,
        updates={"status": "resuming", "status_message": f"Resuming from generation {state.engine.current_generation}..."},
        lease_seconds=RUN_LEASE_SECONDS,
        owner_id=state.run_owner_id,
    )

    # Clear the queue
    state.reset_queue()
    initial_history = [[idea_to_dict(idea) for idea in gen] for gen in state.engine.history]
    state.latest_data = initial_history
    state.history_version = 1 if initial_history else 0
    state.last_sent_history_version = -1

    # Set up evolution status
    state.status = {
        "current_generation": state.engine.current_generation,
        "total_generations": state.engine.generations,
        "is_running": True,
        "is_resuming": True,
        "checkpoint_id": checkpoint_id,
        "history": initial_history,
        "history_version": state.history_version,
        "history_changed": True,
        "contexts": state.engine.contexts,
        "specific_prompts": state.engine.specific_prompts,
        "breeding_prompts": state.engine.breeding_prompts,
        "progress": (state.engine.current_generation / state.engine.generations) * 100 if state.engine.generations > 0 else 0,
        "start_time": start_time,  # Track when evolution resumed
        "evolution_id": state.engine.evolution_id,
        "evolution_name": state.engine.evolution_name,
        "tournament_count": state.engine.tournament_count,
        "full_tournament_rounds": state.engine.full_tournament_rounds,
        "target_tournament_rounds": state.engine.tournament_rounds,
    }

    # Put initial status in queue
    await state.queue.put(state.status.copy())

    # Start heartbeat to keep leases alive during long steps
    state.start_heartbeat(asyncio.create_task(_heartbeat_loop(user.uid, state)))

    # Start evolution in background task
    asyncio.create_task(run_resume_evolution_task(state))

    return JSONResponse({
        "status": "success",
        "message": f"Resuming from generation {state.engine.current_generation}/{state.engine.generations}",
        "checkpoint_id": checkpoint_id,
        "evolution_id": state.engine.evolution_id,
        "evolution_name": state.engine.evolution_name,
        "current_generation": state.engine.current_generation,
        "total_generations": state.engine.generations,
        "contexts": state.engine.contexts,
        "specific_prompts": state.engine.specific_prompts,
        "history_version": state.history_version,
        "history_changed": True,
        "tournament_count": state.engine.tournament_count,
        "full_tournament_rounds": state.engine.full_tournament_rounds,
        "target_tournament_rounds": state.engine.tournament_rounds,
    })

@app.post("/api/continue-evolution")
async def continue_evolution(request: Request, user: UserInfo = Depends(require_auth)):
    """
    Continue a completed evolution for additional generations.
    This can work with either a checkpoint or a saved evolution file.
    """
    # Get user-specific state
    state = await user_states.get(user.uid)

    # Check if already running
    if state.status.get("is_running"):
        return JSONResponse(
            {"status": "error", "message": "You already have an evolution running."},
            status_code=409,
        )

    # Check for API key in DB
    api_key = await db.get_user_api_key(user.uid)
    if not api_key:
        return JSONResponse(
            {"status": "error", "message": "API Key not configured."},
            status_code=400,
        )

    data = await request.json()
    # Support both camelCase and snake_case parameter names
    checkpoint_id = data.get('checkpoint_id') or data.get('checkpointId')
    evolution_id = data.get('evolution_id') or data.get('evolutionId')
    additional_generations = int(data.get('additional_generations') or data.get('additionalGenerations', 3))

    if not checkpoint_id and not evolution_id:
        return JSONResponse(
            {"status": "error", "message": "Either checkpoint_id or evolution_id is required"},
            status_code=400,
        )

    print(f"Continuing evolution for {additional_generations} more generations")

    # Load the engine from Firestore (works for both checkpoints and evolutions)
    evolution_or_checkpoint_id = checkpoint_id or evolution_id
    live_engine = _get_live_engine_for_identifier(state, evolution_or_checkpoint_id)
    if live_engine is not None:
        state.engine = live_engine
        try:
            await state.engine.save_checkpoint(status=_derive_run_status(state.status))
        except Exception:
            logger.warning("Failed to repair live evolution checkpoint before continue", exc_info=True)
    else:
        evaluation_runners, artifact_store = _web_artifact_runtime(
            user_id=user.uid,
            evolution_id=evolution_or_checkpoint_id,
            api_key=api_key,
        )
        state.engine = await EvolutionEngine.load_evolution_for_user(
            user.uid,
            evolution_or_checkpoint_id,
            api_key=api_key,
            evaluation_runners=evaluation_runners,
            artifact_store=artifact_store,
        )

    if state.engine is None:
        return JSONResponse(
            {"status": "error", "message": "Failed to load evolution state"},
            status_code=500,
        )
    if state.engine.unsupported_evaluation_components:
        unavailable = ", ".join(state.engine.unsupported_evaluation_components)
        state.engine = None
        return JSONResponse(
            {
                "status": "error",
                "message": f"Evaluation execution backend is not configured for: {unavailable}",
            },
            status_code=400,
        )

    # Claim run slot (and global lock if enabled)
    start_time = datetime.now().isoformat()
    ok, existing, scope = await _claim_run_slot(
        user_id=user.uid,
        evolution_id=state.engine.evolution_id,
        evolution_name=state.engine.evolution_name,
        total_generations=state.engine.generations + additional_generations,
        start_time=start_time,
        tournament_count=state.engine.tournament_count,
        full_tournament_rounds=state.engine.full_tournament_rounds,
        target_tournament_rounds=state.engine.tournament_rounds,
    )
    if not ok:
        status = 429 if scope == "global" else 409
        message = "System busy. Another evolution is running. Try again shortly." if scope == "global" else \
            "You already have an evolution running."
        state.engine = None
        state.reset_run_tracking()
        return JSONResponse(
            {"status": "error", "message": message, "scope": scope, "active_run": existing},
            status_code=status,
        )

    state.reset_run_tracking()
    state.run_owner_id = INSTANCE_ID
    await db.update_active_run(
        user_id=user.uid,
        updates={"status": "continuing", "status_message": f"Continuing from generation {state.engine.current_generation}..."},
        lease_seconds=RUN_LEASE_SECONDS,
        owner_id=state.run_owner_id,
    )

    # Clear the queue
    state.reset_queue()
    initial_history = [[idea_to_dict(idea) for idea in gen] for gen in state.engine.history]
    state.latest_data = initial_history
    state.history_version = 1 if initial_history else 0
    state.last_sent_history_version = -1

    # Set up evolution status
    new_total = state.engine.generations + additional_generations
    state.status = {
        "current_generation": state.engine.current_generation,
        "total_generations": new_total,
        "is_running": True,
        "is_continuing": True,
        "checkpoint_id": state.engine.checkpoint_id,
        "history": initial_history,
        "history_version": state.history_version,
        "history_changed": True,
        "contexts": state.engine.contexts,
        "specific_prompts": state.engine.specific_prompts,
        "breeding_prompts": state.engine.breeding_prompts,
        "progress": (state.engine.current_generation / new_total) * 100 if new_total > 0 else 0,
        "start_time": start_time,  # Track when evolution continued
        "evolution_id": state.engine.evolution_id,
        "evolution_name": state.engine.evolution_name,
        "tournament_count": state.engine.tournament_count,
        "full_tournament_rounds": state.engine.full_tournament_rounds,
        "target_tournament_rounds": state.engine.tournament_rounds,
    }

    # Put initial status in queue
    await state.queue.put(state.status.copy())

    # Start heartbeat to keep leases alive during long steps
    state.start_heartbeat(asyncio.create_task(_heartbeat_loop(user.uid, state)))

    # Start evolution in background task with additional generations
    asyncio.create_task(run_continue_evolution_task(state, additional_generations))

    return JSONResponse({
        "status": "success",
        "message": f"Continuing for {additional_generations} more generations",
        "checkpoint_id": state.engine.checkpoint_id,
        "evolution_id": state.engine.evolution_id,
        "evolution_name": state.engine.evolution_name,
        "current_generation": state.engine.current_generation,
        "total_generations": new_total,
        "contexts": state.engine.contexts,
        "specific_prompts": state.engine.specific_prompts,
        "history_version": state.history_version,
        "history_changed": True,
        "tournament_count": state.engine.tournament_count,
        "full_tournament_rounds": state.engine.full_tournament_rounds,
        "target_tournament_rounds": state.engine.tournament_rounds,
    })

@app.post("/api/continue-saved-evolution")
async def continue_saved_evolution(request: Request, user: UserInfo = Depends(require_auth)):
    """
    Continue a saved evolution (from data/*.json) for additional generations.
    This creates a checkpoint from the saved file and then continues.
    """
    # Get user-specific state
    state = await user_states.get(user.uid)

    # Check if already running
    if state.status.get("is_running"):
        return JSONResponse(
            {"status": "error", "message": "You already have an evolution running."},
            status_code=409,
        )

    # Check for API key in DB
    api_key = await db.get_user_api_key(user.uid)
    if not api_key:
        return JSONResponse(
            {"status": "error", "message": "API Key not configured."},
            status_code=400,
        )

    data = await request.json()
    evolution_id = data.get('evolution_id')
    additional_generations = int(data.get('additional_generations', 3))

    if not evolution_id:
        return JSONResponse(
            {"status": "error", "message": "evolution_id is required"},
            status_code=400,
        )

    print(f"Continuing saved evolution '{evolution_id}' for {additional_generations} more generations")

    # Load evolution from Firestore
    live_engine = _get_live_engine_for_identifier(state, evolution_id)
    if live_engine is not None:
        state.engine = live_engine
        try:
            await state.engine.save_checkpoint(status=_derive_run_status(state.status))
        except Exception:
            logger.warning("Failed to repair live evolution checkpoint before continue", exc_info=True)
    else:
        evaluation_runners, artifact_store = _web_artifact_runtime(
            user_id=user.uid,
            evolution_id=evolution_id,
            api_key=api_key,
        )
        state.engine = await EvolutionEngine.load_evolution_for_user(
            user.uid,
            evolution_id,
            api_key=api_key,
            evaluation_runners=evaluation_runners,
            artifact_store=artifact_store,
        )

    if state.engine is None:
        return JSONResponse(
            {"status": "error", "message": f"Failed to load evolution '{evolution_id}'"},
            status_code=500,
        )
    if state.engine.unsupported_evaluation_components:
        unavailable = ", ".join(state.engine.unsupported_evaluation_components)
        state.engine = None
        return JSONResponse(
            {
                "status": "error",
                "message": f"Evaluation execution backend is not configured for: {unavailable}",
            },
            status_code=400,
        )

    # Claim run slot (and global lock if enabled)
    start_time = datetime.now().isoformat()
    ok, existing, scope = await _claim_run_slot(
        user_id=user.uid,
        evolution_id=state.engine.evolution_id,
        evolution_name=state.engine.evolution_name,
        total_generations=state.engine.generations + additional_generations,
        start_time=start_time,
        tournament_count=state.engine.tournament_count,
        full_tournament_rounds=state.engine.full_tournament_rounds,
        target_tournament_rounds=state.engine.tournament_rounds,
    )
    if not ok:
        status = 429 if scope == "global" else 409
        message = "System busy. Another evolution is running. Try again shortly." if scope == "global" else \
            "You already have an evolution running."
        state.engine = None
        state.reset_run_tracking()
        return JSONResponse(
            {"status": "error", "message": message, "scope": scope, "active_run": existing},
            status_code=status,
        )

    state.reset_run_tracking()
    state.run_owner_id = INSTANCE_ID
    await db.update_active_run(
        user_id=user.uid,
        updates={"status": "continuing", "status_message": f"Continuing from generation {state.engine.current_generation}..."},
        lease_seconds=RUN_LEASE_SECONDS,
        owner_id=state.run_owner_id,
    )

    # Clear the queue
    state.reset_queue()
    initial_history = [[idea_to_dict(idea) for idea in gen] for gen in state.engine.history]
    state.latest_data = initial_history
    state.history_version = 1 if initial_history else 0
    state.last_sent_history_version = -1

    # Set up evolution status
    new_total = state.engine.generations + additional_generations
    state.status = {
        "current_generation": state.engine.current_generation,
        "total_generations": new_total,
        "is_running": True,
        "is_continuing": True,
        "checkpoint_id": state.engine.checkpoint_id,
        "history": initial_history,
        "history_version": state.history_version,
        "history_changed": True,
        "contexts": state.engine.contexts,
        "specific_prompts": state.engine.specific_prompts,
        "breeding_prompts": state.engine.breeding_prompts,
        "progress": (state.engine.current_generation / new_total) * 100 if new_total > 0 else 0,
        "start_time": start_time,  # Track when evolution continued
        "evolution_id": state.engine.evolution_id,
        "evolution_name": state.engine.evolution_name,
        "tournament_count": state.engine.tournament_count,
        "full_tournament_rounds": state.engine.full_tournament_rounds,
        "target_tournament_rounds": state.engine.tournament_rounds,
    }

    # Put initial status in queue
    await state.queue.put(state.status.copy())

    # Start heartbeat to keep leases alive during long steps
    state.start_heartbeat(asyncio.create_task(_heartbeat_loop(user.uid, state)))

    # Start evolution in background task with additional generations
    asyncio.create_task(run_continue_evolution_task(state, additional_generations))

    return JSONResponse({
        "status": "success",
        "message": f"Continuing saved evolution for {additional_generations} more generations",
        "checkpoint_id": state.engine.checkpoint_id,
        "evolution_id": state.engine.evolution_id,
        "evolution_name": state.engine.evolution_name,
        "current_generation": state.engine.current_generation,
        "total_generations": new_total,
        "history": initial_history,
        "contexts": state.engine.contexts,
        "specific_prompts": state.engine.specific_prompts,
        "history_version": state.history_version,
        "history_changed": True,
        "tournament_count": state.engine.tournament_count,
        "full_tournament_rounds": state.engine.full_tournament_rounds,
        "target_tournament_rounds": state.engine.tournament_rounds,
    })

async def run_resume_evolution_task(state: UserEvolutionState):
    """Run resumed evolution in background with progress updates"""
    engine = state.engine

    async def progress_callback(update_data):
        # Convert Idea objects to dictionaries for JSON serialization
        if 'history' in update_data and isinstance(update_data['history'], list):
            serialized_history = [
                [idea_to_dict(idea) for idea in generation]
                for generation in update_data['history']
            ]
            if serialized_history and serialized_history != state.latest_data:
                state.latest_data = serialized_history
                state.history_version += 1
                update_data["history_changed"] = True
                update_data["history"] = serialized_history
            else:
                update_data["history_changed"] = False
                update_data.pop("history", None)
        else:
            update_data["history_changed"] = False

        update_data["history_version"] = state.history_version
        update_data["tournament_count"] = engine.tournament_count
        update_data["full_tournament_rounds"] = engine.full_tournament_rounds
        update_data["target_tournament_rounds"] = engine.tournament_rounds

        # Add token counts if evolution is complete
        if update_data.get('is_running') is False and 'error' not in update_data:
            if hasattr(engine, 'get_total_token_count'):
                update_data['token_counts'] = engine.get_total_token_count()

        _merge_status_update(state, update_data)
        _attach_live_engine_progress_fields(state)

        # Clear queue and add update
        state.reset_queue()
        await state.queue.put(state.status.copy())

        # Persist run state for reconnects
        if update_data.get("is_running") is False or update_data.get("error"):
            await _finalize_run_state(engine.user_id, state, update_data)
            state.reset_run_tracking()
        else:
            await _refresh_run_state(engine.user_id, state, update_data)

    # Run the resumed evolution
    await engine.resume_evolution_with_updates(progress_callback)

async def run_continue_evolution_task(state: UserEvolutionState, additional_generations: int):
    """Run continued evolution in background with progress updates"""
    engine = state.engine

    async def progress_callback(update_data):
        # Convert Idea objects to dictionaries for JSON serialization
        if 'history' in update_data and isinstance(update_data['history'], list):
            serialized_history = [
                [idea_to_dict(idea) for idea in generation]
                for generation in update_data['history']
            ]
            if serialized_history and serialized_history != state.latest_data:
                state.latest_data = serialized_history
                state.history_version += 1
                update_data["history_changed"] = True
                update_data["history"] = serialized_history
            else:
                update_data["history_changed"] = False
                update_data.pop("history", None)
        else:
            update_data["history_changed"] = False

        update_data["history_version"] = state.history_version
        update_data["tournament_count"] = engine.tournament_count
        update_data["full_tournament_rounds"] = engine.full_tournament_rounds
        update_data["target_tournament_rounds"] = engine.tournament_rounds

        # Add token counts if evolution is complete
        if update_data.get('is_running') is False and 'error' not in update_data:
            if hasattr(engine, 'get_total_token_count'):
                update_data['token_counts'] = engine.get_total_token_count()

        _merge_status_update(state, update_data)
        _attach_live_engine_progress_fields(state)

        # Clear queue and add update
        state.reset_queue()
        await state.queue.put(state.status.copy())

        # Persist run state for reconnects
        if update_data.get("is_running") is False or update_data.get("error"):
            await _finalize_run_state(engine.user_id, state, update_data)
            state.reset_run_tracking()
        else:
            await _refresh_run_state(engine.user_id, state, update_data)

    # Run the evolution with additional generations
    await engine.resume_evolution_with_updates(progress_callback, additional_generations=additional_generations)

async def run_evolution_task(state: UserEvolutionState):
    """Run evolution in background with progress updates"""
    engine = state.engine

    # Define the progress callback function
    async def progress_callback(update_data):
        # Add evolution identity to every update
        update_data['evolution_id'] = engine.evolution_id
        update_data['evolution_name'] = engine.evolution_name
        update_data["tournament_count"] = engine.tournament_count
        update_data["full_tournament_rounds"] = engine.full_tournament_rounds
        update_data["target_tournament_rounds"] = engine.tournament_rounds

        # Convert Idea objects to dictionaries for JSON serialization
        if 'history' in update_data and isinstance(update_data['history'], list):
            serialized_history = [
                [idea_to_dict(idea) for idea in generation]
                for generation in update_data['history']
            ]

            # Store the latest evolution data for rating
            if serialized_history and serialized_history != state.latest_data:
                state.latest_data = serialized_history
                state.history_version += 1
                update_data["history_changed"] = True
                update_data["history"] = serialized_history
            else:
                update_data["history_changed"] = False
                update_data.pop("history", None)
        else:
            update_data["history_changed"] = False
        update_data["history_version"] = state.history_version

        # If evolution is complete, add token counts
        if update_data.get('is_running') is False and 'error' not in update_data:
            # Get token counts from the engine
            if hasattr(engine, 'get_total_token_count'):
                update_data['token_counts'] = engine.get_total_token_count()
                print(f"Evolution complete. Total tokens: {update_data['token_counts']['total']}")

        # Update the evolution status by merging the new data
        _merge_status_update(state, update_data)
        _attach_live_engine_progress_fields(state)

        # Clear the queue before adding new update to avoid backlog
        state.reset_queue()

        # Add the update to the queue
        # We put the full status (copy) into the queue to ensure
        # the frontend gets the complete state, not just the partial update.
        await state.queue.put(state.status.copy())

        # Log progress
        gen = update_data.get('current_generation', 0)
        progress = update_data.get('progress', 0)
        status = update_data.get('status_message', '')
        gen_label = "0 (Initial)" if gen == 0 else gen

        log_msg = f"Progress update: Generation {gen_label}, Progress: {progress:.2f}%"
        if status:
            log_msg += f" - {status}"
        print(log_msg)

        # Persist run state for reconnects
        if update_data.get("is_running") is False or update_data.get("error"):
            await _finalize_run_state(engine.user_id, state, update_data)
            state.reset_run_tracking()
        else:
            await _refresh_run_state(engine.user_id, state, update_data)

    # Run the evolution with progress updates
    await engine.run_evolution_with_updates(progress_callback)

def convert_uuids_to_strings(obj):
    """Recursively convert all UUID objects to strings in a data structure"""
    if isinstance(obj, uuid.UUID):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: convert_uuids_to_strings(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_uuids_to_strings(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_uuids_to_strings(item) for item in obj)
    else:
        return obj

def idea_to_dict(idea) -> dict:
    """Convert an Idea object or idea dictionary to a dictionary for JSON serialization"""
    def add_shared_metadata(result: dict, source: dict) -> dict:
        if "breeding_genotypes" in source:
            result["breeding_genotypes"] = convert_uuids_to_strings(source.get("breeding_genotypes", []))
        for key in (
            "island_id",
            "origin_island_id",
            "migrated_from_id",
            "migration_generation",
            "migration_event",
            "survivor_refined",
            "survivor_refined_from_id",
            "survivor_refinement_rate",
            "selection_metrics",
            "survival_metrics",
            "generation_metrics",
            "rating_evaluations",
            "artifact_ids",
            "artifact_refs",
            "evaluations",
        ):
            if key in source:
                value = source.get(key)
                result[key] = str(value) if isinstance(value, uuid.UUID) else value
        return result

    # If idea is already a dictionary with 'id' and 'idea' keys
    if isinstance(idea, dict) and 'id' in idea and 'idea' in idea:
        idea_obj = idea['idea']
        idea_id = idea['id']

        # Get parent IDs if they exist and convert UUIDs to strings
        parent_ids = [str(pid) if isinstance(pid, uuid.UUID) else pid for pid in idea.get('parent_ids', [])]

        # Get Oracle metadata if it exists
        oracle_generated = idea.get('oracle_generated', False)
        oracle_analysis = idea.get('oracle_analysis', '')

        # Get Elite/Creative metadata if it exists and convert UUIDs to strings
        elite_selected = idea.get('elite_selected', False)
        elite_source_id = str(idea.get('elite_source_id')) if idea.get('elite_source_id') else None
        elite_source_generation = idea.get('elite_source_generation')
        elite_selected_source = idea.get('elite_selected_source', False)
        elite_target_generation = idea.get('elite_target_generation')

        # If the idea object has title and content attributes
        if hasattr(idea_obj, 'title') and hasattr(idea_obj, 'content'):
            result = {
                "id": str(idea_id),
                "title": idea_obj.title,
                "content": idea_obj.content,
                "parent_ids": parent_ids,
                "match_count": idea.get('match_count', 0),
                "auto_match_count": idea.get('auto_match_count', 0),
                "manual_match_count": idea.get('manual_match_count', 0)
            }
            # Add Oracle metadata if present
            if oracle_generated:
                result["oracle_generated"] = oracle_generated
                result["oracle_analysis"] = oracle_analysis
            # Add Elite/Creative metadata if present
            if elite_selected:
                result["elite_selected"] = elite_selected
                result["elite_source_id"] = elite_source_id
                result["elite_source_generation"] = elite_source_generation
            if elite_selected_source:
                result["elite_selected_source"] = elite_selected_source
                result["elite_target_generation"] = elite_target_generation
            return add_shared_metadata(result, idea)
        # If the idea object is a string
        elif isinstance(idea_obj, str):
            result = {
                "id": str(idea_id),
                "title": "Untitled",
                "content": idea_obj,
                "parent_ids": parent_ids,
                "match_count": idea.get('match_count', 0),
                "auto_match_count": idea.get('auto_match_count', 0),
                "manual_match_count": idea.get('manual_match_count', 0)
            }
            # Add Oracle metadata if present
            if oracle_generated:
                result["oracle_generated"] = oracle_generated
                result["oracle_analysis"] = oracle_analysis
            # Add Elite/Creative metadata if present
            if elite_selected:
                result["elite_selected"] = elite_selected
                result["elite_source_id"] = elite_source_id
                result["elite_source_generation"] = elite_source_generation
            if elite_selected_source:
                result["elite_selected_source"] = elite_selected_source
                result["elite_target_generation"] = elite_target_generation
            return add_shared_metadata(result, idea)
        # If the idea object is already a dict
        elif isinstance(idea_obj, dict):
            result = idea_obj.copy()
            result["id"] = str(idea_id)
            result["parent_ids"] = parent_ids
            result["match_count"] = idea.get('match_count', 0)
            result["auto_match_count"] = idea.get('auto_match_count', 0)
            result["manual_match_count"] = idea.get('manual_match_count', 0)
            # Add Oracle metadata if present
            if oracle_generated:
                result["oracle_generated"] = oracle_generated
                result["oracle_analysis"] = oracle_analysis
            # Add Elite/Creative metadata if present
            if elite_selected:
                result["elite_selected"] = elite_selected
                result["elite_source_id"] = elite_source_id
                result["elite_source_generation"] = elite_source_generation
            if elite_selected_source:
                result["elite_selected_source"] = elite_selected_source
                result["elite_target_generation"] = elite_target_generation
            return add_shared_metadata(result, idea)

    # Legacy case: idea is a direct Idea object
    elif hasattr(idea, 'title') and hasattr(idea, 'content'):
        result = {
            "title": idea.title,
            "content": idea.content,
            "parent_ids": [],
            "match_count": getattr(idea, 'match_count', 0),
            "auto_match_count": getattr(idea, 'auto_match_count', 0),
            "manual_match_count": getattr(idea, 'manual_match_count', 0)
        }
        # Check for Oracle metadata on the idea object itself
        if hasattr(idea, 'oracle_generated') and idea.oracle_generated:
            result["oracle_generated"] = idea.oracle_generated
            result["oracle_analysis"] = getattr(idea, 'oracle_analysis', '')
        return result

    # Fallback case: idea is a string
    elif isinstance(idea, str):
        return {
            "title": "Untitled",
            "content": idea,
            "parent_ids": [],
            "match_count": 0
        }

    # Last resort: return the idea as is if it's a dict, or an empty dict
    return idea if isinstance(idea, dict) else {}


def _candidate_artifact_refs(evolution_data: Dict[str, Any]):
    """Yield raw artifact refs from candidate collections without trusting ids."""

    collections: List[Any] = []
    history = evolution_data.get("history")
    if isinstance(history, list):
        collections.extend(history)
    population = evolution_data.get("population")
    if isinstance(population, list):
        collections.append(population)

    for candidate_collection in collections:
        if not isinstance(candidate_collection, list):
            continue
        for candidate in candidate_collection:
            if not isinstance(candidate, dict):
                continue
            refs = candidate.get("artifact_refs")
            if not isinstance(refs, list):
                continue
            for raw_ref in refs:
                if isinstance(raw_ref, dict):
                    yield raw_ref


def _owned_artifact_ref(
    evolution_data: Dict[str, Any],
    artifact_id: str,
) -> Optional[ArtifactRef]:
    matches: List[ArtifactRef] = []
    for raw_ref in _candidate_artifact_refs(evolution_data):
        if raw_ref.get("id") != artifact_id:
            continue
        try:
            matches.append(ArtifactRef.model_validate(raw_ref))
        except Exception as exc:
            raise ArtifactEvidenceError(
                "the evolution contains an invalid matching artifact reference"
            ) from exc
    if not matches:
        return None
    first = matches[0]
    if any(match != first for match in matches[1:]):
        raise ArtifactEvidenceError(
            "the evolution contains ambiguous matching artifact references"
        )
    return first


def _delete_owned_artifact_objects(
    evolution_data: Dict[str, Any],
    store: Any,
) -> None:
    storage_keys: Set[str] = set()
    for raw_ref in _candidate_artifact_refs(evolution_data):
        try:
            storage_keys.add(ArtifactRef.model_validate(raw_ref).storage_key)
        except Exception:
            logger.warning("Skipping invalid artifact reference during cleanup")
    for storage_key in storage_keys:
        try:
            store.delete(storage_key)
        except Exception:
            logger.exception("Failed to delete evolution artifact %s", storage_key)


@app.get("/api/evolution/{evolution_id}/artifacts/{artifact_id}")
async def get_evolution_artifact(
    evolution_id: str,
    artifact_id: str,
    user: UserInfo = Depends(require_auth),
):
    """Return one owned, digest-verified artifact using a non-executable type."""

    state = await user_states.get(user.uid)
    engine = getattr(state, "engine", None)
    if engine is not None and getattr(engine, "evolution_id", None) == evolution_id:
        evolution_data = {
            "history": getattr(engine, "history", []),
            "population": getattr(engine, "population", []),
        }
    else:
        evolution_data = await db.get_evolution(user.uid, evolution_id)
    if not evolution_data:
        raise HTTPException(status_code=404, detail="Evolution not found")

    try:
        artifact_ref = _owned_artifact_ref(evolution_data, artifact_id)
    except ArtifactEvidenceError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if artifact_ref is None:
        raise HTTPException(status_code=404, detail="Artifact not found")

    store = _live_or_persisted_artifact_store(
        state,
        user_id=user.uid,
        evolution_id=evolution_id,
    )
    if store is None:
        raise HTTPException(status_code=404, detail="Artifact store is unavailable")
    try:
        contents = ArtifactStoreLoader(store).load(artifact_ref)
    except ArtifactEvidenceError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    response_types = {
        "image/png": "image/png",
        "text/x-python": "text/plain; charset=utf-8",
        "application/json": "application/json; charset=utf-8",
        "text/plain": "text/plain; charset=utf-8",
    }
    media_type = response_types.get(artifact_ref.media_type)
    if media_type is None:
        raise HTTPException(status_code=415, detail="Artifact media type is not displayable")
    return Response(
        content=contents,
        media_type=media_type,
        headers={
            "Cache-Control": "private, max-age=31536000, immutable",
            "Content-Security-Policy": "default-src 'none'; sandbox",
            "X-Content-Type-Options": "nosniff",
        },
    )

@app.get("/api/generations")
async def api_get_generations(user: UserInfo = Depends(require_auth)):
    """
    Returns a JSON array of arrays: each generation is an array of ideas.
    Each idea is {title, content}.
    """
    state = await user_states.get(user.uid)

    # If engine is None, use the latest evolution data
    if state.engine is None:
        if state.latest_data:
            print(f"Returning latest evolution data with {len(state.latest_data)} generations")
            return JSONResponse(state.latest_data)
        else:
            print("No evolution data available")
            return JSONResponse([])  # Return empty array if no data is available

    # If engine is available, use its history
    result = []
    for generation in state.engine.history:
        gen_list = []
        for prop in generation:
            # Use idea_to_dict to properly serialize the idea with all metadata
            gen_list.append(idea_to_dict(prop))
        result.append(gen_list)

    # Store the result as the latest evolution data
    state.latest_data = result

    return JSONResponse(result)

@app.get("/api/generations/{gen_id}")
async def api_get_generation(gen_id: int, user: UserInfo = Depends(require_auth)):
    """
    Returns ideas for a specific generation.
    """
    state = await user_states.get(user.uid)
    if state.engine is None:
        return JSONResponse({"error": "No evolution running."}, status_code=404)
    ideas = state.engine.get_ideas_by_generation(gen_id)
    if not ideas:
        return JSONResponse({"error": "Invalid generation index."}, status_code=404)
    # Use idea_to_dict to properly serialize ideas with all metadata
    output = [idea_to_dict(i) for i in ideas]
    return JSONResponse(output)

@app.get("/api/progress")
async def get_progress(request: Request, user: UserInfo = Depends(require_auth)):
    """Returns the current progress of the evolution"""
    state = await user_states.get(user.uid)
    include_history = request.query_params.get("includeHistory", "").lower() in {"1", "true", "yes"}

    if state.engine is not None:
        # If there's a new update in the queue, get it
        try:
            if not state.queue.empty():
                state.status = await state.queue.get()
        except Exception as e:
            print(f"Error getting queue updates: {e}")

        _attach_live_engine_progress_fields(state)

        response = state.status.copy()
        response["history_version"] = state.history_version
        history_changed = state.history_version != state.last_sent_history_version
        response["history_changed"] = history_changed
        response["history_available"] = bool(state.latest_data)

        include_history_now = include_history or history_changed
        if include_history_now and state.latest_data:
            response["history"] = state.latest_data
            state.last_sent_history_version = state.history_version
        else:
            response.pop("history", None)

        # One-shot event flags should be emitted once, then cleared.
        for flag in TRANSIENT_PROGRESS_FLAGS:
            if response.get(flag):
                state.status.pop(flag, None)

        return JSONResponse(convert_uuids_to_strings(response))

    # Fallback to Firestore-backed run state for reconnects or cross-instance requests
    run_state = await db.get_active_run(user.uid)
    if run_state:
        now_ms = _now_ms()
        stale = _is_run_stale(run_state, now_ms)

        response = {
            "current_generation": run_state.get("current_generation", 0),
            "total_generations": run_state.get("total_generations", 0),
            "progress": run_state.get("progress", 0),
            "status_message": run_state.get("status_message", ""),
            "start_time": run_state.get("start_time"),
            "evolution_id": run_state.get("evolution_id"),
            "evolution_name": run_state.get("evolution_name"),
            "checkpoint_id": run_state.get("checkpoint_id"),
            "is_running": run_state.get("status") in ACTIVE_RUN_STATUSES and not stale,
            "tournament_count": run_state.get("tournament_count"),
            "full_tournament_rounds": run_state.get("full_tournament_rounds"),
            "target_tournament_rounds": run_state.get("target_tournament_rounds"),
            "history_version": run_state.get("history_version", 0),
            "history_changed": False,
        }

        for field_name in (
            "token_counts",
            "diversity_history",
            "island_diversity_history",
            "migration_history",
        ):
            if run_state.get(field_name) is not None:
                response[field_name] = run_state.get(field_name)

        if run_state.get("is_stopped"):
            response["is_stopped"] = True
        if run_state.get("is_resumable"):
            response["is_resumable"] = True

        if run_state.get("status") in {"paused", "force_stopped", "stopped"} or stale:
            response["is_stopped"] = True
            response["is_resumable"] = True
            response["stop_message"] = "Evolution paused (worker disconnected). You can resume."

        if run_state.get("status") == "complete":
            response["is_running"] = False

        if run_state.get("last_error"):
            response["error"] = run_state.get("last_error")

        if stale:
            await db.update_active_run(
                user_id=user.uid,
                updates={
                    "status": "paused",
                    "is_running": False,
                    "is_stopped": True,
                    "is_resumable": True,
                    "status_message": "Evolution paused (worker disconnected).",
                },
                lease_seconds=0,
                owner_id=run_state.get("owner_id"),
            )

        # Attach history/context from persisted evolution if available
        evolution_id = run_state.get("evolution_id")
        if evolution_id:
            response["history_available"] = True
            if include_history:
                evolution_data = await db.get_evolution(user.uid, evolution_id)
                if evolution_data:
                    response.update({
                        "history": evolution_data.get("history", []),
                        "contexts": evolution_data.get("contexts", []),
                        "specific_prompts": evolution_data.get("specific_prompts", []),
                        "breeding_prompts": evolution_data.get("breeding_prompts", []),
                        "diversity_history": response.get(
                            "diversity_history",
                            evolution_data.get("diversity_history", []),
                        ),
                        "island_diversity_history": response.get(
                            "island_diversity_history",
                            evolution_data.get("island_diversity_history", []),
                        ),
                        "migration_history": response.get(
                            "migration_history",
                            evolution_data.get("migration_history", []),
                        ),
                        "token_counts": response.get(
                            "token_counts",
                            evolution_data.get("token_counts"),
                        ),
                    })
        else:
            response["history_available"] = False

        state.status = response
        return JSONResponse(convert_uuids_to_strings(response))

    # No active run in Firestore and no in-memory engine. Ensure stale local running flags
    # do not keep the UI in a phantom "running" state.
    if state.status.get("is_running"):
        state.status["is_running"] = False
        state.status["is_stopped"] = True
        state.status["is_resumable"] = bool(state.status.get("checkpoint_id"))
        if not state.status.get("status_message"):
            state.status["status_message"] = "No active evolution run found."

    return JSONResponse(convert_uuids_to_strings(state.status))


def _allow_unscoped_legacy_local_files() -> bool:
    """Return whether root-level ``data/*.json`` compatibility is enabled."""

    return os.getenv(ALLOW_UNSCOPED_LEGACY_LOCAL_FILES_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _validate_local_evolution_filename(filename: Any) -> str:
    """Validate a single JSON basename without resolving or following it."""

    if not isinstance(filename, str):
        raise HTTPException(status_code=400, detail="Filename must be a string")

    filename = filename.strip()
    candidate = Path(filename)
    if (
        not filename
        or len(filename) > 255
        or len(os.fsencode(filename)) > 255
        or "\x00" in filename
        or candidate.is_absolute()
        or candidate.name != filename
        or "/" in filename
        or "\\" in filename
        or candidate.suffix.lower() != ".json"
    ):
        raise HTTPException(
            status_code=400,
            detail="Filename must be a plain .json filename without directory components",
        )
    return filename


def _resolved_data_root() -> Path:
    try:
        root = DATA_DIR.resolve(strict=True)
        root_stat = root.stat()
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Local data directory is unavailable") from exc
    if not stat.S_ISDIR(root_stat.st_mode):
        raise HTTPException(status_code=500, detail="Local data path is not a directory")
    return root


def _contained_directory(parent: Path, name: str, *, create: bool) -> Optional[Path]:
    """Open-or-create one server-controlled directory component without symlinks."""

    candidate = parent / name
    if create:
        try:
            candidate.mkdir(mode=0o700, exist_ok=True)
        except OSError as exc:
            raise HTTPException(status_code=500, detail="Local user directory is unavailable") from exc

    try:
        candidate_stat = candidate.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Local user directory is unavailable") from exc

    if stat.S_ISLNK(candidate_stat.st_mode) or not stat.S_ISDIR(candidate_stat.st_mode):
        raise HTTPException(status_code=500, detail="Local user directory is invalid")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(parent)
        os.chmod(resolved, 0o700)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Local user directory is invalid") from exc
    return resolved


def _local_user_evolution_directory(user_id: str, *, create: bool) -> Optional[Path]:
    """Return a stable, opaque, traversal-safe directory for one authenticated user."""

    if not isinstance(user_id, str) or not user_id:
        raise HTTPException(status_code=401, detail="Authenticated user ID is required")
    root = _resolved_data_root()
    users_root = _contained_directory(root, LOCAL_EVOLUTION_USERS_DIR, create=create)
    if users_root is None:
        return None
    user_key = "user-" + hashlib.sha256(user_id.encode("utf-8")).hexdigest()
    return _contained_directory(users_root, user_key, create=create)


def _open_local_directory_fd(directory: Path) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_fd = -1
    try:
        directory_fd = os.open(directory, flags)
        directory_stat = os.fstat(directory_fd)
    except OSError as exc:
        if directory_fd >= 0:
            os.close(directory_fd)
        raise HTTPException(status_code=500, detail="Local evolution directory is unavailable") from exc
    if not stat.S_ISDIR(directory_stat.st_mode):
        os.close(directory_fd)
        raise HTTPException(status_code=500, detail="Local evolution directory is invalid")
    return directory_fd


def _read_local_json_file(
    directory: Path,
    filename: Any,
) -> Optional[Tuple[Any, os.stat_result]]:
    """Read one regular JSON file through a no-follow directory-FD boundary."""

    safe_filename = _validate_local_evolution_filename(filename)
    directory_fd = _open_local_directory_fd(directory)
    file_fd = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            file_fd = os.open(safe_filename, flags, dir_fd=directory_fd)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise HTTPException(
                status_code=400,
                detail="Local evolution must be a regular, non-symbolic-link file",
            ) from exc

        file_stat = os.fstat(file_fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise HTTPException(
                status_code=400,
                detail="Local evolution must be a regular, non-symbolic-link file",
            )
        with os.fdopen(file_fd, "r", encoding="utf-8") as handle:
            file_fd = -1
            return json.load(handle), file_stat
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(directory_fd)


def _atomic_write_local_json_file(directory: Path, filename: Any, payload: Any) -> Path:
    """Atomically replace one user-scoped JSON file without following symlinks."""

    safe_filename = _validate_local_evolution_filename(filename)
    directory_fd = _open_local_directory_fd(directory)
    temp_name = f".evolution-{uuid.uuid4().hex}.tmp"
    temp_fd = -1
    committed = False
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        temp_fd = os.open(temp_name, flags, 0o600, dir_fd=directory_fd)
        os.fchmod(temp_fd, 0o600)
        with os.fdopen(temp_fd, "w", encoding="utf-8") as handle:
            temp_fd = -1
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            temp_name,
            safe_filename,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        committed = True
        try:
            os.fsync(directory_fd)
        except OSError:
            pass
        return directory / safe_filename
    finally:
        if temp_fd >= 0:
            os.close(temp_fd)
        if not committed:
            try:
                os.unlink(temp_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            except OSError:
                pass
        os.close(directory_fd)


def _write_user_local_evolution(user: UserInfo, filename: Any, payload: Any) -> Path:
    safe_filename = _validate_local_evolution_filename(filename)
    directory = _local_user_evolution_directory(user.uid, create=True)
    assert directory is not None
    return _atomic_write_local_json_file(directory, safe_filename, payload)


def _list_local_json_filenames(directory: Path) -> List[str]:
    """List only direct regular JSON files; symlinks and nested paths are ignored."""

    filenames: List[str] = []
    try:
        for entry in os.scandir(directory):
            if not entry.is_file(follow_symlinks=False):
                continue
            try:
                filenames.append(_validate_local_evolution_filename(entry.name))
            except HTTPException:
                continue
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Local evolution directory is unavailable") from exc
    return sorted(filenames)


@app.post("/api/save-evolution")
async def save_evolution(request: Request, user: UserInfo = Depends(require_auth)):
    """Save an authenticated user's evolution export to the local data directory."""
    try:
        # Parse the request body manually
        data = await request.json()
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")

        if 'data' not in data or 'filename' not in data:
            print("Missing required fields in request")
            return JSONResponse({
                "status": "error",
                "message": "Missing required fields: data or filename"
            }, status_code=400)

        evolution_data = data['data']
        filename = data['filename']
        file_path = _write_user_local_evolution(user, filename, evolution_data)

        print(f"Saving evolution data to file: {filename}")
        if isinstance(evolution_data, dict):
            print(f"Data keys: {evolution_data.keys()}")
        else:
            print(f"Data is not a dictionary: {type(evolution_data)}")

        print(f"Successfully saved evolution data to {file_path}")
        return JSONResponse({"status": "success", "file_path": str(file_path)})
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error saving evolution: {e}")
        print(traceback.format_exc())
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)


@app.get('/api/evolutions')
async def get_evolutions(user: UserInfo = Depends(require_auth)):
    """Get list of evolutions for the current user from Firestore"""
    try:
        evolutions = await db.list_evolutions(user.uid)
        evolutions_list = []
        for ev in evolutions:
            evolutions_list.append({
                'id': ev.get('id'),
                'timestamp': ev.get('updated_at') or ev.get('created_at', ''),
                'filename': ev.get('name', 'Unnamed Evolution')
            })
        return JSONResponse(evolutions_list)
    except Exception as e:
        print(f"Error listing evolutions: {e}")
        return JSONResponse([])

@app.get('/api/history')
async def get_unified_history(user: UserInfo = Depends(require_auth)):
    """
    Get unified history of all evolutions for the current user.
    Loads from Firestore.
    """
    history_items = []
    state = await user_states.get(user.uid)

    try:
        # 1. Load evolutions from Firestore (user-scoped)
        evolutions = await EvolutionEngine.list_evolutions_for_user(user.uid)
        for ev in evolutions:
            item = {
                'id': ev.get('id'),
                'type': 'evolution',
                'status': ev.get('status', 'unknown'),
                'timestamp': ev.get('updated_at') or ev.get('created_at', ''),
                'created_at': ev.get('created_at'),
                'display_name': ev.get('name', 'Unnamed Evolution'),
                'generations': ev.get('generation', 0),
                'total_generations': ev.get('total_generations', 0),
                'pop_size': ev.get('pop_size', 0),
                'idea_type': ev.get('idea_type', 'Unknown'),
                'model_type': ev.get('model_type', 'Unknown'),
                'total_ideas': ev.get('total_ideas', 0),
                'config': ev.get('config') or {},
                'can_use_config': bool(ev.get('config')),
                'can_resume': False,
                'can_continue': False,
                'can_rate': False,
                'can_rename': True,
                'can_delete': True,
            }
            live_data = _get_live_evolution_payload(state, item['id'])
            if live_data:
                live_history = live_data.get("history", [])
                item.update({
                    'status': live_data.get('status', item['status']),
                    'generations': live_data.get('current_generation', item['generations']),
                    'total_generations': live_data.get('total_generations', item['total_generations']),
                    'total_ideas': _history_total_ideas(live_history) or item.get('total_ideas', 0),
                })
            _apply_history_capabilities(item)
            history_items.append(item)

        live_data = _get_live_evolution_payload(state)
        existing_ids = {item['id'] for item in history_items}
        if live_data and live_data.get("evolution_id") not in existing_ids:
            live_history = live_data.get("history", [])
            item = {
                'id': live_data.get('evolution_id'),
                'type': 'evolution',
                'status': live_data.get('status', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'created_at': getattr(state.engine, "created_at", None),
                'display_name': live_data.get('name') or live_data.get('evolution_id'),
                'generations': live_data.get('current_generation', 0),
                'total_generations': live_data.get('total_generations', 0),
                'pop_size': len(live_history[-1]) if live_history else 0,
                'idea_type': getattr(state.engine, "idea_type", "Unknown"),
                    'model_type': getattr(state.engine, "model_type", "Unknown"),
                    'total_ideas': _history_total_ideas(live_history),
                    'config': live_data.get('config') or {},
                    'can_use_config': bool(live_data.get('config')),
                    'can_resume': False,
                'can_continue': False,
                'can_rate': False,
                'can_rename': True,
                'can_delete': True,
            }
            _apply_history_capabilities(item)
            history_items.append(item)

        # 2. Load legacy checkpoints from Firestore (for backwards compatibility)
        checkpoints = await EvolutionEngine.list_checkpoints_for_user(user.uid)
        existing_ids = {item['id'] for item in history_items}
        for cp in checkpoints:
            checkpoint_id = cp.get('id', '')
            # Skip if already in unified storage
            if checkpoint_id in existing_ids:
                continue

            history_items.append({
                'id': checkpoint_id,
                'type': 'legacy_checkpoint',
                'status': cp.get('status', 'unknown'),
                'timestamp': cp.get('time', ''),
                'display_name': f"Checkpoint {checkpoint_id[:16]}..." if len(checkpoint_id) > 16 else checkpoint_id,
                'generations': cp.get('generation', 0),
                'total_generations': cp.get('total_generations', 0),
                'pop_size': cp.get('pop_size', 0),
                'idea_type': cp.get('idea_type', 'Unknown'),
                'model_type': cp.get('model_type', 'Unknown'),
                'config': cp.get('config') or {},
                'can_use_config': bool(cp.get('config')),
                'can_resume': cp.get('status') in ['paused', 'in_progress', 'force_stopped'],
                'can_continue': cp.get('status') == 'complete',
                'can_rate': cp.get('generation', 0) > 0,
                'can_rename': False,  # Legacy format can't be renamed
                'can_delete': True,
            })

        # 3. Load authenticated user-scoped local saves. Root-level legacy files
        # are deliberately excluded unless the operator enables the compatibility
        # flag; a scoped save with the same id always wins.
        local_sources: List[Tuple[Path, bool]] = []
        user_local_directory = _local_user_evolution_directory(user.uid, create=False)
        if user_local_directory is not None:
            local_sources.append((user_local_directory, False))
        if _allow_unscoped_legacy_local_files():
            local_sources.append((_resolved_data_root(), True))

        for local_directory, is_unscoped_legacy in local_sources:
            for filename in _list_local_json_filenames(local_directory):
                file_path = local_directory / filename
                try:
                    evolution_id = Path(filename).stem
                    # Skip if already in unified storage, checkpoints, or the
                    # higher-priority user-scoped local directory.
                    if evolution_id in existing_ids or any(
                        item['id'] == evolution_id for item in history_items
                    ):
                        continue

                    loaded = _read_local_json_file(local_directory, filename)
                    if loaded is None:
                        continue
                    data, file_stat = loaded
                    if not isinstance(data, dict):
                        raise ValueError("local evolution root must be a JSON object")

                    history = data.get('history', [])
                    config = data.get('config', {})
                    num_generations = len(history)
                    pop_size = len(history[0]) if history else 0

                    history_items.append({
                        'id': evolution_id,
                        'type': 'legacy_saved',
                        'status': 'complete',
                        'timestamp': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                        'display_name': evolution_id.replace('_', ' ').replace('-', ' '),
                        'generations': num_generations,
                        'total_generations': num_generations,
                        'pop_size': pop_size,
                        'idea_type': config.get('idea_type') or data.get('idea_type', 'Unknown'),
                        'model_type': config.get('model_type', 'Unknown'),
                        'total_ideas': sum(len(gen) for gen in history),
                        'config': config,
                        'can_use_config': bool(config),
                        'can_continue': True,
                        'can_rate': True,
                        'can_rename': False,
                        'can_delete': not is_unscoped_legacy,
                    })
                except Exception as e:
                    print(f"Error reading local evolution file {file_path}: {e}")
                    continue

        # Sort by timestamp descending (most recent first)
        history_items.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        return JSONResponse({
            'status': 'success',
            'items': history_items,
            'total_count': len(history_items)
        })

    except Exception as e:
        import traceback
        print(f"Error getting unified history: {e}")
        traceback.print_exc()
        return JSONResponse({
            'status': 'error',
            'message': str(e),
            'items': []
        }, status_code=500)

@app.post('/api/evolution/{evolution_id}/rename')
async def rename_evolution_endpoint(request: Request, evolution_id: str, user: UserInfo = Depends(require_auth)):
    """Rename an evolution"""
    try:
        data = await request.json()
        new_name = data.get('name', '').strip()

        if not new_name:
            return JSONResponse({
                'status': 'error',
                'message': 'Name cannot be empty'
            }, status_code=400)

        # Rename in Firestore
        if await EvolutionEngine.rename_evolution_for_user(user.uid, evolution_id, new_name):
            return JSONResponse({
                'status': 'success',
                'message': f'Evolution renamed to "{new_name}"',
                'name': new_name
            })

        return JSONResponse({
            'status': 'error',
            'message': 'Evolution not found or cannot be renamed'
        }, status_code=404)

    except Exception as e:
        return JSONResponse({
            'status': 'error',
            'message': str(e)
        }, status_code=500)

@app.delete('/api/evolution/{evolution_id}')
async def delete_evolution_endpoint(evolution_id: str, user: UserInfo = Depends(require_auth)):
    """Delete an evolution"""
    try:
        state = await user_states.get(user.uid)
        engine = getattr(state, "engine", None)
        if engine is not None and getattr(engine, "evolution_id", None) == evolution_id:
            evolution_data = {
                "history": getattr(engine, "history", []),
                "population": getattr(engine, "population", []),
            }
        else:
            evolution_data = await db.get_evolution(user.uid, evolution_id) or {}
        artifact_store = _live_or_persisted_artifact_store(
            state,
            user_id=user.uid,
            evolution_id=evolution_id,
        )
        # Delete from Firestore
        if await EvolutionEngine.delete_evolution_for_user(user.uid, evolution_id):
            if artifact_store is not None and evolution_data:
                _delete_owned_artifact_objects(evolution_data, artifact_store)
            return JSONResponse({'status': 'success', 'message': 'Evolution deleted'})

        return JSONResponse({
            'status': 'error',
            'message': 'Evolution not found'
        }, status_code=404)

    except Exception as e:
        return JSONResponse({
            'status': 'error',
            'message': str(e)
        }, status_code=500)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@app.post('/api/evolution/{evolution_id}/repair')
async def repair_evolution_endpoint(request: Request, evolution_id: str, user: UserInfo = Depends(require_auth)):
    """Repair a stale persisted evolution using the browser's cached complete state."""
    try:
        body = await request.json()
        client_data = body.get("data", body) if isinstance(body, dict) else {}
        if not isinstance(client_data, dict):
            return JSONResponse({"status": "error", "message": "Invalid repair payload"}, status_code=400)

        client_evolution_id = client_data.get("evolution_id") or client_data.get("evolutionId")
        if client_evolution_id and client_evolution_id != evolution_id:
            return JSONResponse({"status": "error", "message": "Evolution id mismatch"}, status_code=400)

        client_history = client_data.get("history")
        if not isinstance(client_history, list) or not client_history:
            return JSONResponse({"status": "error", "message": "Repair payload must include history"}, status_code=400)

        existing = await db.get_evolution(user.uid, evolution_id) or {}
        existing_history = existing.get("history", [])
        if not isinstance(existing_history, list):
            existing_history = []

        client_generation = _safe_int(
            client_data.get("current_generation"),
            max(0, len(client_history) - 1),
        )
        existing_generation = _safe_int(
            existing.get("current_generation"),
            max(0, len(existing_history) - 1),
        )
        if len(client_history) < len(existing_history) or (
            len(client_history) == len(existing_history)
            and client_generation <= existing_generation
        ):
            return JSONResponse({
                "status": "skipped",
                "message": "Persisted evolution is already at least as recent.",
            })

        config = dict(existing.get("config") or client_data.get("config") or {})
        total_generations = _safe_int(
            client_data.get("total_generations"),
            _safe_int(config.get("generations"), client_generation),
        )
        config["generations"] = max(_safe_int(config.get("generations"), 0), total_generations)
        if "pop_size" not in config and client_history and isinstance(client_history[-1], list):
            config["pop_size"] = len(client_history[-1])

        repaired = dict(existing)
        repaired.update({
            "evolution_id": evolution_id,
            "name": (
                existing.get("name")
                or client_data.get("evolution_name")
                or client_data.get("name")
                or evolution_id
            ),
            "status": client_data.get("status") or (
                "paused"
                if client_data.get("is_stopped") or client_data.get("is_resumable")
                else "complete"
                if client_data.get("is_running") is False
                else existing.get("status", "in_progress")
            ),
            "current_generation": client_generation,
            "config": config,
            "history": client_history,
            "population": client_data.get("population") or client_history[-1],
        })

        for field_name in (
            "contexts",
            "specific_prompts",
            "breeding_prompts",
            "tournament_history",
            "diversity_history",
            "island_diversity_history",
            "migration_history",
            "token_counts",
            "comparison_cache",
            "fitness_elo_stats",
            "fitness_diversity_stats",
            "fitness_metric_stats",
        ):
            if client_data.get(field_name) is not None:
                repaired[field_name] = client_data.get(field_name)

        if not repaired.get("created_at"):
            repaired["created_at"] = datetime.now().isoformat()

        await db.save_evolution(user.uid, evolution_id, repaired)
        return JSONResponse({
            "status": "repaired",
            "current_generation": repaired["current_generation"],
            "history_length": len(client_history),
        })
    except Exception as e:
        logger.exception("Failed to repair evolution from client cache")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get('/api/evolution/{evolution_id}')
async def get_evolution(evolution_id: str, user: UserInfo = Depends(require_auth)):
    """Get evolution data from Firestore"""
    try:
        state = await user_states.get(user.uid)
        live_data = _get_live_evolution_payload(state, evolution_id)
        # Load from Firestore
        data = await db.get_evolution(user.uid, evolution_id)
        if data:
            # The durable evolution remains useful even when the optional
            # active-run overlay is temporarily unavailable (for example,
            # during local-file tests or a transient Firestore failure).
            try:
                active_run = await db.get_active_run(user.uid)
            except Exception:
                logger.warning(
                    "Unable to load active-run overlay for evolution %s",
                    evolution_id,
                    exc_info=True,
                )
                active_run = None
            if active_run and active_run.get("evolution_id") == evolution_id:
                for field_name in (
                    "token_counts",
                    "diversity_history",
                    "island_diversity_history",
                    "migration_history",
                ):
                    if active_run.get(field_name) is not None:
                        data[field_name] = active_run.get(field_name)
            if live_data:
                saved_history = data.get("history", [])
                if not isinstance(saved_history, list):
                    saved_history = []
                live_history = live_data.get("history", [])
                if not isinstance(live_history, list):
                    live_history = []
                if len(live_history) >= len(saved_history):
                    data.update({
                        "status": live_data.get("status", data.get("status")),
                        "config": live_data.get("config", data.get("config", {})),
                        "current_generation": live_data.get(
                            "current_generation",
                            data.get("current_generation"),
                        ),
                        "history": live_history,
                        "history_version": live_data.get("history_version"),
                        "contexts": live_data.get("contexts", data.get("contexts", [])),
                        "specific_prompts": live_data.get(
                            "specific_prompts",
                            data.get("specific_prompts", []),
                        ),
                        "breeding_prompts": live_data.get(
                            "breeding_prompts",
                            data.get("breeding_prompts", []),
                        ),
                        "tournament_history": live_data.get(
                            "tournament_history",
                            data.get("tournament_history", []),
                        ),
                        "diversity_history": live_data.get(
                            "diversity_history",
                            data.get("diversity_history", []),
                        ),
                        "island_diversity_history": live_data.get(
                            "island_diversity_history",
                            data.get("island_diversity_history", []),
                        ),
                        "migration_history": live_data.get(
                            "migration_history",
                            data.get("migration_history", []),
                        ),
                        "token_counts": live_data.get(
                            "token_counts",
                            data.get("token_counts"),
                        ),
                    })
            return JSONResponse({
                'id': evolution_id,
                'name': data.get('name', evolution_id),
                'timestamp': data.get('updated_at') or data.get('created_at'),
                'data': data
            })

        if live_data:
            data = dict(live_data)
            data["evolution_id"] = evolution_id
            return JSONResponse({
                'id': evolution_id,
                'name': data.get('name', evolution_id),
                'timestamp': datetime.now().isoformat(),
                'data': data
            })

        raise HTTPException(status_code=404, detail="Evolution not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _load_rating_evolution(
    evolution_id: str,
    user: UserInfo,
) -> Tuple[Optional[Path], Dict[str, Any], str]:
    """Load Firestore first, then the user's local namespace and opt-in legacy root."""

    filename = _validate_local_evolution_filename(f"{evolution_id}.json")
    firestore_data = await db.get_evolution(user.uid, evolution_id)
    if firestore_data:
        return None, firestore_data, "firestore"

    user_directory = _local_user_evolution_directory(user.uid, create=False)
    if user_directory is not None:
        loaded = _read_local_json_file(user_directory, filename)
        if loaded is not None:
            evolution_data, _ = loaded
            if not isinstance(evolution_data, dict):
                raise HTTPException(status_code=400, detail="Local evolution must be a JSON object")
            return user_directory / filename, evolution_data, "file"

    if _allow_unscoped_legacy_local_files():
        legacy_directory = _resolved_data_root()
        loaded = _read_local_json_file(legacy_directory, filename)
        if loaded is not None:
            evolution_data, _ = loaded
            if not isinstance(evolution_data, dict):
                raise HTTPException(status_code=400, detail="Local evolution must be a JSON object")
            return legacy_directory / filename, evolution_data, "legacy_file"

    raise HTTPException(status_code=404, detail="Evolution not found")


async def _save_rating_evolution(
    evolution_id: str,
    evolution_data: Dict[str, Any],
    source: str,
    user: UserInfo,
) -> None:
    """Persist Firestore in place, or atomically save a user-scoped local copy."""

    if source == "firestore":
        await db.save_evolution(user.uid, evolution_id, evolution_data)
        return

    if source not in {"file", "legacy_file"}:
        raise HTTPException(status_code=400, detail="Unknown local evolution source")
    # Legacy root files are never modified. The first rating write migrates a
    # private copy into the authenticated user's namespace.
    _write_user_local_evolution(user, f"{evolution_id}.json", evolution_data)


def _resolve_rating_title_content(idea: Dict[str, Any]) -> Tuple[str, str]:
    title = idea.get("title")
    content = idea.get("content")

    nested = idea.get("idea")
    if isinstance(nested, dict):
        title = nested.get("title", title)
        content = nested.get("content", content)
    elif hasattr(nested, "title") or hasattr(nested, "content"):
        title = getattr(nested, "title", title)
        content = getattr(nested, "content", content)

    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("{") and "\"title\"" in stripped:
            try:
                parsed = json.loads(stripped)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                title = parsed.get("title", title)
                content = parsed.get("content", content)

    if title is None or str(title).strip() == "":
        title = "Untitled"
    else:
        title = str(title)

    if content is None:
        content = ""
    elif not isinstance(content, str):
        content = str(content)

    return title, content


def _normalize_rating_fields(idea: Dict[str, Any], fallback_id: str) -> None:
    if not idea.get("id"):
        idea["id"] = fallback_id
    idea["id"] = str(idea["id"])

    title, content = _resolve_rating_title_content(idea)
    idea["title"] = title
    idea["content"] = content

    ratings = idea.get("ratings")
    if isinstance(ratings, (int, float)):
        ratings = {"auto": ratings, "manual": ratings}
    elif not isinstance(ratings, dict):
        ratings = {}

    auto_rating = ratings.get("auto", idea.get("elo", 1500))
    manual_rating = ratings.get("manual", 1500)

    try:
        auto_rating = float(auto_rating)
    except Exception:
        auto_rating = 1500.0
    try:
        manual_rating = float(manual_rating)
    except Exception:
        manual_rating = 1500.0

    ratings["auto"] = int(round(auto_rating))
    ratings["manual"] = int(round(manual_rating))
    idea["ratings"] = ratings
    idea["elo"] = ratings["auto"]

    for field in ("match_count", "auto_match_count", "manual_match_count"):
        try:
            idea[field] = int(idea.get(field, 0) or 0)
        except Exception:
            idea[field] = 0


def _collect_latest_ideas_for_rating(evolution_data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Tuple[int, int]]]:
    latest_by_id: Dict[str, Dict[str, Any]] = {}
    latest_index_by_id: Dict[str, Tuple[int, int]] = {}
    seq = 0

    for gen_index, generation in enumerate(evolution_data.get("history", [])):
        for idea_index, idea in enumerate(generation):
            _normalize_rating_fields(idea, f"idea_{seq}")
            seq += 1
            idea["generation"] = gen_index + 1
            idea_id = idea["id"]
            latest_by_id[idea_id] = idea
            latest_index_by_id[idea_id] = (gen_index, idea_index)

    return list(latest_by_id.values()), latest_index_by_id


def _parse_pair_entry(entry: Any) -> Optional[Tuple[str, str]]:
    if isinstance(entry, dict):
        a_id = entry.get("a_id")
        b_id = entry.get("b_id")
    elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
        a_id = entry[0]
        b_id = entry[1]
    else:
        return None

    if a_id is None or b_id is None:
        return None

    a_id = str(a_id)
    b_id = str(b_id)
    if a_id == b_id:
        return None

    return tuple(sorted((a_id, b_id)))


def _get_rating_swiss_state(evolution_data: Dict[str, Any], rating_type: str) -> Dict[str, Any]:
    root = evolution_data.setdefault("rating_swiss_state", {})
    state = root.get(rating_type)
    if not isinstance(state, dict):
        state = {}
        root[rating_type] = state

    if not isinstance(state.get("match_history"), list):
        state["match_history"] = []
    if not isinstance(state.get("bye_counts"), dict):
        state["bye_counts"] = {}
    if not isinstance(state.get("pending_pairs"), list):
        state["pending_pairs"] = []
    try:
        state["round_number"] = int(state.get("round_number", 0) or 0)
    except Exception:
        state["round_number"] = 0

    return state


def _normalize_pair_history_entries(entries: List[Any]) -> Tuple[List[Dict[str, str]], Set[Tuple[str, str]]]:
    normalized: List[Dict[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for entry in entries:
        parsed = _parse_pair_entry(entry)
        if not parsed or parsed in seen:
            continue
        seen.add(parsed)
        normalized.append({"a_id": parsed[0], "b_id": parsed[1]})
    return normalized, seen


def _pop_pending_pair(state: Dict[str, Any], ideas_by_id: Dict[str, Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    normalized_pending: List[Dict[str, str]] = []
    for pair in state.get("pending_pairs", []):
        parsed = _parse_pair_entry(pair)
        if not parsed:
            continue
        a_id, b_id = parsed
        if a_id in ideas_by_id and b_id in ideas_by_id:
            normalized_pending.append({"a_id": a_id, "b_id": b_id})

    state["pending_pairs"] = normalized_pending
    if not normalized_pending:
        return None, None

    pair = normalized_pending.pop(0)
    state["pending_pairs"] = normalized_pending
    return ideas_by_id[pair["a_id"]], ideas_by_id[pair["b_id"]]


def _next_manual_swiss_pair(
    evolution_data: Dict[str, Any],
    all_ideas: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
    if len(all_ideas) < 2:
        return None, None, {}

    state = _get_rating_swiss_state(evolution_data, "manual")
    ideas_by_id = {idea["id"]: idea for idea in all_ideas}

    idea_a, idea_b = _pop_pending_pair(state, ideas_by_id)
    if idea_a and idea_b:
        return idea_a, idea_b, {
            "round_number": state.get("round_number", 0),
            "remaining_pairs_in_round": len(state.get("pending_pairs", [])),
            "bye_idea_id": state.get("last_bye_id"),
        }

    id_to_index = {idea["id"]: idx for idx, idea in enumerate(all_ideas)}
    index_to_id = {idx: idea_id for idea_id, idx in id_to_index.items()}
    ranks = {idx: float(all_ideas[idx]["ratings"]["manual"]) for idx in range(len(all_ideas))}

    history_entries, _ = _normalize_pair_history_entries(state.get("match_history", []))
    committed_history: Set[Tuple[int, int]] = set()
    for entry in history_entries:
        a_id = entry["a_id"]
        b_id = entry["b_id"]
        if a_id in id_to_index and b_id in id_to_index:
            idx_a = id_to_index[a_id]
            idx_b = id_to_index[b_id]
            committed_history.add((min(idx_a, idx_b), max(idx_a, idx_b)))
    state["match_history"] = history_entries

    bye_counts_raw = state.get("bye_counts", {})
    bye_counts: Dict[int, int] = {}
    if isinstance(bye_counts_raw, dict):
        for idea_id, count in bye_counts_raw.items():
            idea_id = str(idea_id)
            if idea_id not in id_to_index:
                continue
            try:
                bye_counts[id_to_index[idea_id]] = int(count)
            except Exception:
                continue

    temp_history = set(committed_history)
    pairs, bye_idx = generate_swiss_round_pairs(ranks, temp_history, bye_counts)

    state["round_number"] = int(state.get("round_number", 0)) + 1
    state["bye_counts"] = {
        index_to_id[idx]: int(count)
        for idx, count in bye_counts.items()
        if count > 0 and idx in index_to_id
    }
    state["pending_pairs"] = [
        {"a_id": index_to_id[idx_a], "b_id": index_to_id[idx_b]}
        for idx_a, idx_b in pairs
        if idx_a in index_to_id and idx_b in index_to_id
    ]
    if bye_idx is not None and bye_idx in index_to_id:
        state["last_bye_id"] = index_to_id[bye_idx]
    else:
        state["last_bye_id"] = None

    idea_a, idea_b = _pop_pending_pair(state, ideas_by_id)
    return idea_a, idea_b, {
        "round_number": state.get("round_number", 0),
        "remaining_pairs_in_round": len(state.get("pending_pairs", [])),
        "bye_idea_id": state.get("last_bye_id"),
    }


def _normalize_thinking_level(level: Optional[Any]) -> Optional[str]:
    if level is None:
        return None
    normalized = str(level).strip().lower()
    if not normalized:
        return None
    if normalized in {"off", "minimal", "low", "medium", "high"}:
        return normalized
    return None


def _get_default_thinking_settings(model_name: str) -> Tuple[Optional[str], Optional[int]]:
    """Get default thinking settings for a model."""
    config = THINKING_MODEL_CONFIG.get(model_name, {})
    if not config.get("supports_thinking", False):
        return None, None

    default_level = _normalize_thinking_level(config.get("default_level"))
    default_budget = config.get("default_budget")

    try:
        parsed_budget = int(default_budget) if default_budget is not None else None
    except (TypeError, ValueError):
        parsed_budget = None

    # Prefer explicit level defaults when configured; fallback to budget otherwise.
    if default_level is not None:
        return default_level, None
    return None, parsed_budget


def _get_autorating_costs(critic_agent: Critic) -> Dict[str, Any]:
    """Build autorating token/cost payload for the frontend."""
    summary_func = getattr(critic_agent, "get_token_usage_summary", None)
    if callable(summary_func):
        critic_summary = summary_func()
        critic_input = int(critic_summary.get("input", 0) or 0)
        critic_output = int(critic_summary.get("output", 0) or 0)
        critic_thoughts = int(critic_summary.get("thoughts", 0) or 0)
        critic_cached_input = int(critic_summary.get("cached_input", 0) or 0)
        critic_uncached_input = int(critic_summary.get("uncached_input", 0) or 0)
        critic_tool_input = int(critic_summary.get("tool_input", 0) or 0)
        critic_total = int(
            critic_summary.get("total", critic_input + critic_output + critic_thoughts)
            or 0
        )
        critic_model = critic_summary.get("model") or getattr(critic_agent, "model_name", "gemini-2.0-flash")
        critic_input_cost = float(critic_summary.get("input_cost", 0.0) or 0.0)
        critic_uncached_input_cost = float(critic_summary.get("uncached_input_cost", 0.0) or 0.0)
        critic_cached_input_cost = float(critic_summary.get("cached_input_cost", 0.0) or 0.0)
        critic_output_cost = float(critic_summary.get("output_cost", 0.0) or 0.0)
        critic_thought_cost = float(critic_summary.get("thought_cost", 0.0) or 0.0)
        total_cost = float(critic_summary.get("cost", 0.0) or 0.0)
    else:
        critic_input = int(getattr(critic_agent, "input_token_count", 0) or 0)
        critic_output = int(getattr(critic_agent, "output_token_count", 0) or 0)
        critic_thoughts = int(getattr(critic_agent, "thought_token_count", 0) or 0)
        critic_cached_input = int(getattr(critic_agent, "cached_input_token_count", 0) or 0)
        critic_tool_input = int(getattr(critic_agent, "tool_input_token_count", 0) or 0)
        critic_model = getattr(critic_agent, "model_name", "gemini-2.0-flash")
        cost_estimate = estimate_generation_token_cost(
            critic_model,
            input_tokens=critic_input,
            output_tokens=critic_output,
            thought_tokens=critic_thoughts,
            cached_input_tokens=critic_cached_input,
            tool_input_tokens=critic_tool_input,
        )
        critic_uncached_input = int(cost_estimate["uncached_input_tokens"])
        critic_total = int(
            getattr(
                critic_agent,
                "total_token_count",
                critic_input + critic_output + critic_thoughts + critic_tool_input,
            )
            or 0
        )
        critic_input_cost = float(cost_estimate["input_cost"])
        critic_uncached_input_cost = float(cost_estimate["uncached_input_cost"])
        critic_cached_input_cost = float(cost_estimate["cached_input_cost"])
        critic_output_cost = float(cost_estimate["output_cost"])
        critic_thought_cost = float(cost_estimate["thought_cost"])
        total_cost = float(cost_estimate["total_cost"])

    estimates = {}
    for model in LLM_MODELS:
        model_id = model["id"]
        model_name = model.get("name", model_id)
        cost_estimate = estimate_generation_token_cost(
            model_id,
            input_tokens=critic_input,
            output_tokens=critic_output,
            thought_tokens=critic_thoughts,
            cached_input_tokens=critic_cached_input,
            tool_input_tokens=critic_tool_input,
        )
        estimates[model_id] = {"name": model_name, "cost": cost_estimate["total_cost"]}

    return {
        "critic": {
            "total": critic_total,
            "input": critic_input,
            "output": critic_output,
            "thoughts": critic_thoughts,
            "cached_input": critic_cached_input,
            "uncached_input": critic_uncached_input,
            "tool_input": critic_tool_input,
            "model": critic_model,
            "cost": total_cost,
        },
        "total": critic_total,
        "total_input": critic_input,
        "total_output": critic_output,
        "total_thoughts": critic_thoughts,
        "total_cached_input": critic_cached_input,
        "total_uncached_input": critic_uncached_input,
        "total_tool_input": critic_tool_input,
        "total_billable_output": critic_output + critic_thoughts,
        "cost": {
            "input_cost": critic_input_cost,
            "uncached_input_cost": critic_uncached_input_cost,
            "cached_input_cost": critic_cached_input_cost,
            "output_cost": critic_output_cost,
            "thought_cost": critic_thought_cost,
            "billable_output_cost": critic_output_cost + critic_thought_cost,
            "total_cost": total_cost,
            "currency": "USD",
        },
        "models": {"critic": critic_model},
        "estimates": estimates,
    }


@app.post("/api/submit-rating")
async def submit_rating(
    request: Request,
    user: UserInfo = Depends(require_auth),
):
    """Submit a manual Swiss comparison result."""
    try:
        data = await request.json()
        idea_a_id = str(data.get("idea_a_id") or "")
        idea_b_id = str(data.get("idea_b_id") or "")
        outcome = data.get("outcome")
        evolution_id = data.get("evolution_id")

        if outcome not in {"A", "B", "tie"}:
            raise HTTPException(status_code=400, detail="Outcome must be one of: A, B, tie")

        print(f"Submitting manual Swiss rating for {idea_a_id} vs {idea_b_id}, outcome: {outcome}")

        _file_path, evolution_data, source = await _load_rating_evolution(evolution_id, user=user)
        all_ideas, _ = _collect_latest_ideas_for_rating(evolution_data)
        ideas_by_id = {idea["id"]: idea for idea in all_ideas}

        idea_a = ideas_by_id.get(idea_a_id)
        idea_b = ideas_by_id.get(idea_b_id)
        if not idea_a or not idea_b:
            raise HTTPException(status_code=404, detail="Ideas not found")

        idea_a["match_count"] += 1
        idea_b["match_count"] += 1
        idea_a["manual_match_count"] += 1
        idea_b["manual_match_count"] += 1

        new_elo_a, new_elo_b = elo_update(
            float(idea_a["ratings"]["manual"]),
            float(idea_b["ratings"]["manual"]),
            outcome,
        )
        idea_a["ratings"]["manual"] = int(round(new_elo_a))
        idea_b["ratings"]["manual"] = int(round(new_elo_b))

        manual_state = _get_rating_swiss_state(evolution_data, "manual")
        normalized_history, seen_history = _normalize_pair_history_entries(manual_state.get("match_history", []))
        pair_key = tuple(sorted((idea_a_id, idea_b_id)))
        if pair_key not in seen_history:
            normalized_history.append({"a_id": pair_key[0], "b_id": pair_key[1]})
        manual_state["match_history"] = normalized_history

        pending = []
        for pair in manual_state.get("pending_pairs", []):
            parsed = _parse_pair_entry(pair)
            if not parsed or parsed == pair_key:
                continue
            pending.append({"a_id": parsed[0], "b_id": parsed[1]})
        manual_state["pending_pairs"] = pending

        await _save_rating_evolution(
            evolution_id=evolution_id,
            evolution_data=evolution_data,
            source=source,
            user=user,
        )

        return JSONResponse({
            "status": "success",
            "updated_elos": {
                idea_a_id: idea_a["ratings"]["manual"],
                idea_b_id: idea_b["ratings"]["manual"],
            },
            "updated_match_counts": {
                idea_a_id: {
                    "total": idea_a["match_count"],
                    "manual": idea_a["manual_match_count"],
                    "auto": idea_a["auto_match_count"],
                },
                idea_b_id: {
                    "total": idea_b["match_count"],
                    "manual": idea_b["manual_match_count"],
                    "auto": idea_b["auto_match_count"],
                },
            },
        })

    except HTTPException as e:
        return JSONResponse({
            "status": "error",
            "message": str(e.detail),
        }, status_code=e.status_code)
    except Exception as e:
        print(f"Error submitting rating: {e}")
        return JSONResponse({
            "status": "error",
            "message": str(e),
        }, status_code=500)


@app.post("/api/auto-rate")
async def auto_rate(
    request: Request,
    user: UserInfo = Depends(require_auth),
):
    """Automatically rate ideas using Swiss rounds from the Critic tournament flow."""
    state: Optional[UserEvolutionState] = None
    try:
        data = await request.json()
        if not isinstance(data, dict):
            raise HTTPException(
                status_code=400,
                detail="Auto-rating request body must be a JSON object",
            )
        evolution_id = data.get("evolutionId")
        if not isinstance(evolution_id, str) or not evolution_id.strip():
            raise HTTPException(
                status_code=400,
                detail="Evolution ID is required",
            )
        evolution_id = evolution_id.strip()
        model_id = data.get("modelId", DEFAULT_MODEL)
        skip_save = bool(data.get("skipSave", False))
        idea_type = data.get("ideaType")

        if user:
            state = await user_states.get(user.uid)

        _file_path, evolution_data, source = await _load_rating_evolution(evolution_id, user=user)
        if not idea_type:
            stored_config = evolution_data.get("config")
            if not isinstance(stored_config, dict):
                stored_config = {}
            idea_type = (
                stored_config.get("idea_type")
                or evolution_data.get("idea_type")
                or get_default_template_id()
            )

        artifact_store = _live_or_persisted_artifact_store(
            state,
            user_id=user.uid,
            evolution_id=evolution_id,
        )
        artifact_loader = (
            ArtifactStoreLoader(artifact_store)
            if artifact_store is not None
            else None
        )
        try:
            pairwise_evaluator = _autorating_pairwise_evaluator(
                evolution_data,
                artifact_loader=artifact_loader,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Auto-rating is unavailable: {exc}",
            ) from exc

        all_ideas, _ = _collect_latest_ideas_for_rating(evolution_data)

        if len(all_ideas) < 2:
            return JSONResponse({
                "status": "error",
                "message": "Not enough ideas to compare (minimum 2 required)",
            }, status_code=400)

        # Align auto-rating with evolution-loop semantics:
        # - full_tournament_rounds = pop_size - 1
        # - tournamentCount controls fractional/multiple Swiss tournaments
        legacy_rounds_input = data.get("numRounds", data.get("numComparisons"))
        try:
            tournament_count, full_tournament_rounds, num_rounds = _resolve_tournament_settings(
                pop_size=len(all_ideas),
                tournament_count_input=data.get("tournamentCount"),
                legacy_rounds_input=legacy_rounds_input,
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tournament settings: {e}",
            ) from e

        print(
            "Starting auto Swiss rating for evolution "
            f"{evolution_id}: count={tournament_count}, rounds={num_rounds}, "
            f"full_rounds={full_tournament_rounds}, ideas={len(all_ideas)}"
        )

        existing_total_comparisons = sum(idea.get("match_count", 0) for idea in all_ideas) // 2
        expected_pairs_per_round = max(1, len(all_ideas) // 2)
        expected_total_pairs = expected_pairs_per_round * num_rounds

        # Use the same API-key resolution strategy as the evolution flow.
        # 1) Prefer the authenticated user's stored key.
        # 2) Fallback to GEMINI_API_KEY env var.
        user_api_key = None
        if user:
            try:
                user_api_key = await db.get_user_api_key(user.uid)
            except Exception as e:
                print(f"Warning: failed to fetch user API key for auto-rate: {e}")

        resolved_api_key = user_api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_api_key:
            return JSONResponse({
                "status": "error",
                "message": "No Gemini API key configured. Set it in Settings before auto-rating.",
            }, status_code=400)

        default_thinking_level, default_thinking_budget = _get_default_thinking_settings(model_id)
        critic = Critic(
            provider="google_generative_ai",
            model_name=model_id,
            temperature=DEFAULT_CREATIVE_TEMP,
            top_p=DEFAULT_TOP_P,
            thinking_level=default_thinking_level,
            thinking_budget=default_thinking_budget,
            api_key=resolved_api_key,
        )

        rounds_details: List[Dict[str, Any]] = []

        replay_ranks = {idx: float(idea["ratings"]["auto"]) for idx, idea in enumerate(all_ideas)}
        results: List[Dict[str, Any]] = []
        win_counts = {"A": 0, "B": 0, "tie": 0}
        next_round_to_apply = 0
        next_pair_to_apply = 0

        def _sorted_ideas() -> List[Dict[str, Any]]:
            return sorted(all_ideas, key=lambda x: x["ratings"]["auto"], reverse=True)

        def _count_completed_rounds() -> int:
            completed = 0
            for round_record in rounds_details:
                pairs = round_record.get("pairs", [])
                if all(pair.get("winner") in {"A", "B", "tie"} for pair in pairs):
                    completed += 1
                else:
                    break
            return completed

        def _apply_resolved_pairs() -> None:
            nonlocal next_round_to_apply, next_pair_to_apply
            while next_round_to_apply < len(rounds_details):
                round_record = rounds_details[next_round_to_apply]
                pairs = round_record.get("pairs", [])

                while next_pair_to_apply < len(pairs):
                    pair = pairs[next_pair_to_apply]
                    winner = pair.get("winner")
                    if winner not in {"A", "B", "tie"}:
                        return

                    idx_a = pair.get("a_idx")
                    idx_b = pair.get("b_idx")
                    if idx_a is None or idx_b is None:
                        next_pair_to_apply += 1
                        continue
                    if idx_a < 0 or idx_b < 0 or idx_a >= len(all_ideas) or idx_b >= len(all_ideas):
                        next_pair_to_apply += 1
                        continue

                    resolved_winner = winner
                    idea_a = all_ideas[idx_a]
                    idea_b = all_ideas[idx_b]

                    idea_a["match_count"] += 1
                    idea_b["match_count"] += 1

                    next_a, next_b = elo_update(
                        replay_ranks[idx_a],
                        replay_ranks[idx_b],
                        resolved_winner,
                    )
                    replay_ranks[idx_a] = float(int(round(next_a)))
                    replay_ranks[idx_b] = float(int(round(next_b)))

                    idea_a["ratings"]["auto"] = int(round(replay_ranks[idx_a]))
                    idea_b["ratings"]["auto"] = int(round(replay_ranks[idx_b]))
                    idea_a["elo"] = idea_a["ratings"]["auto"]
                    idea_b["elo"] = idea_b["ratings"]["auto"]

                    results.append({
                        "idea_a": idea_a.get("id", "unknown"),
                        "idea_b": idea_b.get("id", "unknown"),
                        "outcome": resolved_winner,
                        "new_elo_a": idea_a["ratings"]["auto"],
                        "new_elo_b": idea_b["ratings"]["auto"],
                        "round": round_record.get("round"),
                    })
                    win_counts[resolved_winner] += 1
                    next_pair_to_apply += 1

                next_round_to_apply += 1
                next_pair_to_apply = 0

        _update_autorating_state(
            state,
            {
                "is_running": True,
                "status": "in_progress",
                "status_message": f"Running Swiss round 1/{num_rounds}...",
                "progress": 0,
                "requested_rounds": int(num_rounds),
                "completed_rounds": 0,
                "tournament_count": float(tournament_count),
                "full_tournament_rounds": int(full_tournament_rounds),
                "target_tournament_rounds": int(num_rounds),
                "total_matches": int(expected_total_pairs),
                "completed_matches": 0,
                "completed_comparisons": int(existing_total_comparisons),
                "new_comparisons": 0,
                "win_counts": win_counts.copy(),
                "ideas": _sorted_ideas(),
                "token_counts": None,
                "error": None,
            },
        )

        def _progress_callback(completed_pairs: int, total_pairs: int) -> None:
            _apply_resolved_pairs()
            total_pairs_safe = max(1, int(total_pairs or expected_total_pairs))
            completed_pairs_safe = max(0, min(int(completed_pairs), total_pairs_safe))
            round_num = min(num_rounds, (completed_pairs_safe // expected_pairs_per_round) + 1)

            _update_autorating_state(
                state,
                {
                    "is_running": True,
                    "status": "in_progress",
                    "status_message": f"Running Swiss round {round_num}/{num_rounds}...",
                    "progress": (completed_pairs_safe / total_pairs_safe) * 100,
                    "requested_rounds": int(num_rounds),
                    "completed_rounds": _count_completed_rounds(),
                    "tournament_count": float(tournament_count),
                    "full_tournament_rounds": int(full_tournament_rounds),
                    "target_tournament_rounds": int(num_rounds),
                    "total_matches": int(total_pairs_safe),
                    "completed_matches": int(completed_pairs_safe),
                    "completed_comparisons": int(existing_total_comparisons + len(results)),
                    "new_comparisons": int(len(results)),
                    "win_counts": win_counts.copy(),
                    "ideas": _sorted_ideas(),
                },
            )

        def _run_tournament() -> None:
            comparison_cache = evolution_data.setdefault("comparison_cache", {})
            if not isinstance(comparison_cache, dict):
                comparison_cache = {}
                evolution_data["comparison_cache"] = comparison_cache
            evaluation_result = pairwise_evaluator.evaluate_population(
                critic,
                all_ideas,
                idea_type,
                rounds=num_rounds,
                progress_callback=_progress_callback,
                details=rounds_details,
                full_tournament_rounds=full_tournament_rounds,
                should_stop=lambda: False,
                comparison_cache=comparison_cache,
                evaluation_generation=evolution_data.get("current_generation"),
            )
            failed_records = [
                record
                for record in evaluation_result.batch.records
                if record.status == EvaluationStatus.FAILED
            ]
            if failed_records:
                errors = sorted(
                    {
                        record.error or "pairwise evaluation failed"
                        for record in failed_records
                    }
                )
                raise RuntimeError("; ".join(errors))

        await asyncio.to_thread(_run_tournament)
        _apply_resolved_pairs()

        if not skip_save:
            await _save_rating_evolution(
                evolution_id=evolution_id,
                evolution_data=evolution_data,
                source=source,
                user=user,
            )

        final_token_counts = _get_autorating_costs(critic)
        final_ideas = _sorted_ideas()
        total_comparisons = existing_total_comparisons + len(results)
        _update_autorating_state(
            state,
            {
                "is_running": False,
                "status": "complete",
                "status_message": (
                    f"Completed {num_rounds} Swiss round{'s' if num_rounds != 1 else ''}."
                ),
                "progress": 100,
                "requested_rounds": int(num_rounds),
                "completed_rounds": int(num_rounds),
                "tournament_count": float(tournament_count),
                "full_tournament_rounds": int(full_tournament_rounds),
                "target_tournament_rounds": int(num_rounds),
                "total_matches": int(expected_total_pairs),
                "completed_matches": int(len(results)),
                "completed_comparisons": int(total_comparisons),
                "new_comparisons": int(len(results)),
                "win_counts": win_counts.copy(),
                "ideas": final_ideas,
                "token_counts": final_token_counts,
                "error": None,
            },
        )

        return JSONResponse({
            "status": "success",
            "results": results,
            "ideas": final_ideas,
            "completed_comparisons": total_comparisons,
            "new_comparisons": len(results),
            "completed_rounds": int(num_rounds),
            "new_rounds": int(num_rounds),
            "tournament_count": float(tournament_count),
            "full_tournament_rounds": int(full_tournament_rounds),
            "target_tournament_rounds": int(num_rounds),
            "token_counts": final_token_counts,
        })

    except HTTPException as e:
        _update_autorating_state(
            state,
            {
                "is_running": False,
                "status": "error",
                "status_message": "Auto-rating failed.",
                "error": str(e.detail),
            },
        )
        return JSONResponse({
            "status": "error",
            "message": str(e.detail),
        }, status_code=e.status_code)
    except Exception:
        logger.exception("Auto-rating failed")
        _update_autorating_state(
            state,
            {
                "is_running": False,
                "status": "error",
                "status_message": "Auto-rating failed.",
                "error": "Auto-rating failed",
            },
        )
        return JSONResponse({
            "status": "error",
            "message": "Auto-rating failed",
        }, status_code=500)


@app.get("/api/auto-rate/progress")
async def get_auto_rate_progress(
    user: Optional[UserInfo] = Depends(get_current_user),
):
    """Returns live per-match progress for the active auto-rating request."""
    if not user:
        return JSONResponse({
            "is_running": False,
            "status": "idle",
            "status_message": "",
            "progress": 0,
        })

    state = await user_states.get(user.uid)
    return JSONResponse(convert_uuids_to_strings(state.autorating_status))


@app.get("/api/models")
async def get_models():
    """Return the list of available LLM models."""
    return JSONResponse({
        "models": LLM_MODELS,
        "default": DEFAULT_MODEL,
        "thinking": THINKING_MODEL_CONFIG,
    })


@app.post("/api/reset-ratings")
async def reset_ratings(
    request: Request,
    user: UserInfo = Depends(require_auth),
):
    """Reset ratings for an evolution."""
    try:
        data = await request.json()
        evolution_id = data.get("evolutionId")
        rating_type = data.get("ratingType", "all")

        _file_path, evolution_data, source = await _load_rating_evolution(evolution_id, user=user)

        for generation in evolution_data.get("history", []):
            for idea in generation:
                _normalize_rating_fields(idea, f"idea_{uuid.uuid4()}")

                if rating_type in {"all", "auto"}:
                    idea["ratings"]["auto"] = 1500
                    idea["elo"] = 1500
                    idea["auto_match_count"] = 0

                if rating_type in {"all", "manual"}:
                    idea["ratings"]["manual"] = 1500
                    idea["manual_match_count"] = 0

                if rating_type == "all":
                    idea["match_count"] = 0
                elif rating_type == "auto":
                    idea["match_count"] = idea.get("manual_match_count", 0)
                elif rating_type == "manual":
                    idea["match_count"] = idea.get("auto_match_count", 0)

        swiss_root = evolution_data.get("rating_swiss_state")
        if isinstance(swiss_root, dict):
            if rating_type == "all":
                evolution_data["rating_swiss_state"] = {}
            elif rating_type in {"auto", "manual"}:
                swiss_root.pop(rating_type, None)

        await _save_rating_evolution(
            evolution_id=evolution_id,
            evolution_data=evolution_data,
            source=source,
            user=user,
        )

        return JSONResponse({
            "status": "success",
            "message": f"{rating_type.capitalize()} ratings and match counts reset successfully",
        })

    except HTTPException as e:
        return JSONResponse({
            "status": "error",
            "message": str(e.detail),
        }, status_code=e.status_code)
    except Exception as e:
        print(f"Error resetting ratings: {e}")
        return JSONResponse({
            "status": "error",
            "message": str(e),
        }, status_code=500)


@app.post("/api/get-efficient-pair")
async def get_efficient_pair(
    request: Request,
    user: UserInfo = Depends(require_auth),
):
    """Get the next Swiss pair for manual rating."""
    try:
        data = await request.json()
        evolution_id = data.get("evolution_id")
        if not evolution_id:
            raise HTTPException(status_code=400, detail="Evolution ID is required")

        _file_path, evolution_data, source = await _load_rating_evolution(evolution_id, user=user)
        all_ideas, _ = _collect_latest_ideas_for_rating(evolution_data)
        if len(all_ideas) < 2:
            raise HTTPException(status_code=400, detail="Not enough ideas for comparison")

        idea_a, idea_b, swiss_status = _next_manual_swiss_pair(evolution_data, all_ideas)
        if idea_a is None or idea_b is None:
            raise HTTPException(status_code=500, detail="Failed to select Swiss pair")

        await _save_rating_evolution(
            evolution_id=evolution_id,
            evolution_data=evolution_data,
            source=source,
            user=user,
        )

        return {
            "idea_a": idea_a,
            "idea_b": idea_b,
            "status": "success",
            "swiss_status": swiss_status,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_efficient_pair: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the server
@app.delete("/api/settings/api-key")
async def delete_api_key(user: UserInfo = Depends(require_auth)):
    """Delete the API key for the user"""
    try:
        await db.delete_user_api_key(user.uid)
        return JSONResponse({"status": "success", "message": "API Key deleted successfully"})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(
        "idea.viewer:app", host="127.0.0.1", port=8000, reload=True
    )
