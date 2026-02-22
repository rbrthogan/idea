# viewer.py
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import os
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from asyncio import Queue
import time
import math
import traceback
from pathlib import Path
import json
from pydantic import BaseModel
from datetime import datetime
import uuid
from typing import Dict, List, Tuple, Optional, Any, Set

from idea.evolution import EvolutionEngine
from idea.llm import Critic
from idea.swiss_tournament import elo_update, generate_swiss_round_pairs
from idea.config import (
    LLM_MODELS,
    DEFAULT_MODEL,
    DEFAULT_CREATIVE_TEMP,
    DEFAULT_TOP_P,
    THINKING_MODEL_CONFIG,
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
from idea.auth import require_auth, UserInfo, get_current_user
from fastapi import Depends
from idea import database as db
from idea.user_state import user_states, UserEvolutionState
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- Initialize and configure FastAPI ---
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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
    except Exception as e:
        print(f"Warning: failed to persist run state: {e}")


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
    except Exception as e:
        print(f"Warning: failed to finalize run state: {e}")
    finally:
        state.stop_heartbeat()

# Add this class for the request body
class SaveEvolutionRequest(BaseModel):
    data: dict
    filename: str

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
    return templates.TemplateResponse("viewer.html", {"request": request})

@app.get("/rate")
def serve_rater(request: Request):
    """Serves the rater page"""
    return templates.TemplateResponse("rater.html", {"request": request})

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
            "api_key": api_key,
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
            print("SMTP not configured. Message received but not sent.")
            print(f"From: {request.name} <{request.email}>")
            print(f"Message: {request.message}")
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
        print(f"Error sending email: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/api/debug/auth")
async def debug_auth(user: UserInfo = Depends(require_auth)):
    """Debug endpoint to verify auth and user info."""
    return {"status": "ok", "user": user.to_dict()}

@app.get("/api/debug/db")

async def debug_db(user: UserInfo = Depends(require_auth)):
    """Debug endpoint to verify database connection."""
    try:
        # Test basic DB read
        key = await db.get_user_api_key(user.uid)
        return {"status": "ok", "key_present": key is not None}
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            if 'state' in locals() and state.run_owner_id:
                await _finalize_run_state(user.uid, state, {"error": str(e), "is_running": False})
                state.reset_run_tracking()
        except Exception:
            pass
        return JSONResponse({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }, status_code=500)

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

def get_default_template_id():
    """Get the first available template ID as default"""
    try:
        templates = list_available_templates()
        # Find the first template without errors
        for template_id, template_info in templates.items():
            if 'error' not in template_info:
                return template_id
        # Fallback to airesearch if nothing else works
        return 'airesearch'
    except:
        return 'airesearch'

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
        print(f"Received request data: {data}")

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
            if thinking_budget is not None:
                # Explicit custom budget takes precedence over level presets.
                thinking_level = None
            elif thinking_level == "off" and not bool(model_thinking_cfg.get("allow_off", False)):
                thinking_level = "low"

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
            replacement_rate=replacement_rate,
            fitness_alpha=fitness_alpha,
            age_decay_rate=age_decay_rate,
            thinking_level=thinking_level,
            thinking_budget=thinking_budget,
            max_budget=max_budget,
            api_key=api_key,
            user_id=user.uid,  # Store user_id for Firestore scoping
            template_data=template_data,
        )

        # Initialize evolution with name (auto-generates if not provided)
        evolution_name = (data.get('evolutionName') or '').strip() or None
        engine.initialize_evolution(name=evolution_name)
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
            "tournament_count": state.engine.tournament_count,
            "full_tournament_rounds": state.engine.full_tournament_rounds,
            "target_tournament_rounds": state.engine.tournament_rounds,
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
            "contexts": [],
            "history_version": state.history_version,
            "history_changed": False,
            "tournament_count": state.engine.tournament_count,
            "full_tournament_rounds": state.engine.full_tournament_rounds,
            "target_tournament_rounds": state.engine.tournament_rounds,
            "specific_prompts": []  # Will be populated as evolution progresses
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
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
async def get_checkpoint(checkpoint_id: str):
    """
    Get details of a specific checkpoint.
    """
    try:
        from pathlib import Path
        checkpoint_path = Path("data/checkpoints") / f"checkpoint_{checkpoint_id}.json"
        if not checkpoint_path.exists():
            return JSONResponse({
                "status": "error",
                "message": "Checkpoint not found"
            }, status_code=404)

        import json
        with open(checkpoint_path) as f:
            data = json.load(f)

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
async def delete_checkpoint(checkpoint_id: str):
    """
    Delete a specific checkpoint.
    """
    try:
        success = EvolutionEngine.delete_checkpoint(checkpoint_id)
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
    state.engine = await EvolutionEngine.load_evolution_for_user(user.uid, checkpoint_id, api_key=api_key)
    if state.engine is None:
        return JSONResponse(
            {"status": "error", "message": "Failed to load evolution or checkpoint"},
            status_code=500,
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
    state.engine = await EvolutionEngine.load_evolution_for_user(user.uid, evolution_or_checkpoint_id, api_key=api_key)

    if state.engine is None:
        return JSONResponse(
            {"status": "error", "message": "Failed to load evolution state"},
            status_code=500,
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
    state.engine = await EvolutionEngine.load_evolution_for_user(user.uid, evolution_id, api_key=api_key)

    if state.engine is None:
        return JSONResponse(
            {"status": "error", "message": f"Failed to load evolution '{evolution_id}'"},
            status_code=500,
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

async def load_engine_from_evolution(evolution_id: str, api_key: Optional[str] = None) -> Optional[EvolutionEngine]:
    """
    Create an EvolutionEngine from a saved evolution file.
    This allows continuing evolutions that weren't saved as checkpoints.
    """
    try:
        # Try unified evolutions first
        # EvolutionEngine can handle loading from file path now
        engine = EvolutionEngine.load_evolution(evolution_id, api_key=api_key)
        if engine:
            return engine

        # Fallback for manual loading if needed (legacy files)
        file_path = DATA_DIR / f"{evolution_id}.json"
        if not file_path.exists():
            print(f"Evolution file not found: {file_path}")
            return None

        import json
        with open(file_path) as f:
            data = json.load(f)

        # Extract configuration from the saved data
        config = data.get('config', {})
        history = data.get('history', [])

        if not history:
            print("No history found in evolution file")
            return None

        # Create engine with default or saved config
        engine = EvolutionEngine(
            idea_type=config.get('idea_type', data.get('idea_type', get_default_template_id())),
            pop_size=config.get('pop_size', len(history[-1]) if history else 5),
            generations=len(history),  # Current number of generations
            model_type=config.get('model_type', DEFAULT_MODEL),
            creative_temp=config.get('creative_temp', DEFAULT_CREATIVE_TEMP),
            top_p=config.get('top_p', DEFAULT_TOP_P),
            tournament_rounds=config.get('tournament_rounds', 1),
            tournament_count=config.get('tournament_count'),
            full_tournament_rounds=config.get('full_tournament_rounds'),
            thinking_level=config.get('thinking_level'),
            thinking_budget=config.get('thinking_budget'),
            max_budget=config.get('max_budget'),
            mutation_rate=config.get('mutation_rate', 0.2),
            seed_context_pool_size=config.get('seed_context_pool_size'),
            replacement_rate=config.get('replacement_rate', 0.5),
            fitness_alpha=config.get('fitness_alpha', 0.7),
            age_decay_rate=config.get('age_decay_rate', 0.25),
            age_decay_floor=config.get('age_decay_floor', 0.35),
            api_key=api_key,
        )

        # Restore state from the saved evolution
        engine.contexts = data.get('contexts', [])
        engine.specific_prompts = data.get('specific_prompts', [])
        engine.breeding_prompts = data.get('breeding_prompts', [])
        engine.tournament_history = data.get('tournament_history', [])
        engine.diversity_history = data.get('diversity_history', [])
        engine.current_generation = len(history)
        engine.fitness_elo_stats = data.get("fitness_elo_stats", {"count": 0, "mean": 0.0, "m2": 0.0})
        engine.fitness_diversity_stats = data.get(
            "fitness_diversity_stats", {"count": 0, "mean": 0.0, "m2": 0.0}
        )

        # Restore history and population
        def deserialize_idea(idea_data):
            if not isinstance(idea_data, dict):
                return idea_data
            result = dict(idea_data)
            if 'id' in result and isinstance(result['id'], str):
                try:
                    result['id'] = uuid.UUID(result['id'])
                except ValueError:
                    result['id'] = uuid.uuid4()
            if 'parent_ids' in result:
                result['parent_ids'] = [
                    uuid.UUID(pid) if isinstance(pid, str) else pid
                    for pid in result['parent_ids']
                ]
            # Restore Idea object if needed
            if 'title' in result and 'content' in result and 'idea' not in result:
                from idea.models import Idea
                result['idea'] = Idea(title=result.get('title'), content=result.get('content', ''))
            return result

        engine.history = [[deserialize_idea(idea) for idea in gen] for gen in history]
        engine.population = engine.history[-1] if engine.history else []

        # Generate a new checkpoint ID for this continuation
        from datetime import datetime
        engine.checkpoint_id = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

        print(f"Loaded evolution from file: {evolution_id}, generations: {len(history)}")
        return engine

    except Exception as e:
        print(f"Error loading evolution from file: {e}")
        import traceback
        traceback.print_exc()
        return None

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
            return result
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
            return result
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
            return result

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

@app.get("/test-static")
def test_static():
    """Debug route to list static files"""
    static_dir = "idea/static"
    files = []
    for root, dirs, filenames in os.walk(static_dir):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return {"files": files, "js_exists": os.path.exists("idea/static/js/viewer.js")}

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

        # Always include latest tournament history if available
        if hasattr(state.engine, "tournament_history"):
            state.status["tournament_history"] = state.engine.tournament_history

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
                        "diversity_history": evolution_data.get("diversity_history", []),
                        "token_counts": evolution_data.get("token_counts"),
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

@app.post("/api/save-evolution")
async def save_evolution(request: Request):
    """Save evolution data to file"""
    try:
        # Parse the request body manually
        data = await request.json()
        print(f"Received save request: {data}")

        if 'data' not in data or 'filename' not in data:
            print("Missing required fields in request")
            return JSONResponse({
                "status": "error",
                "message": "Missing required fields: data or filename"
            }, status_code=400)

        evolution_data = data['data']
        filename = data['filename']

        print(f"Saving evolution data to file: {filename}")
        if isinstance(evolution_data, dict):
            print(f"Data keys: {evolution_data.keys()}")
        else:
            print(f"Data is not a dictionary: {type(evolution_data)}")

        file_path = DATA_DIR / filename
        print(f"Full file path: {file_path}")

        with open(file_path, "w") as f:
            json.dump(evolution_data, f, indent=2)

        print(f"Successfully saved evolution data to {file_path}")
        return JSONResponse({"status": "success", "file_path": str(file_path)})
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

    try:
        # 1. Load evolutions from Firestore (user-scoped)
        evolutions = await EvolutionEngine.list_evolutions_for_user(user.uid)
        for ev in evolutions:
            history_items.append({
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
                'can_resume': ev.get('status') in ['paused', 'in_progress', 'force_stopped'],
                'can_continue': ev.get('status') == 'complete',
                'can_rate': ev.get('generation', 0) > 0,
                'can_rename': True,
                'can_delete': True,
            })

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
                'can_resume': cp.get('status') in ['paused', 'in_progress', 'force_stopped'],
                'can_continue': cp.get('status') == 'complete',
                'can_rate': cp.get('generation', 0) > 0,
                'can_rename': False,  # Legacy format can't be renamed
                'can_delete': True,
            })

        # 3. Load legacy saved evolutions from data/*.json (for backwards compatibility)
        for file_path in DATA_DIR.glob('*.json'):
            try:
                evolution_id = file_path.stem
                # Skip if already in unified storage or checkpoints
                if evolution_id in existing_ids or any(item['id'] == evolution_id for item in history_items):
                    continue

                file_stat = file_path.stat()
                with open(file_path) as f:
                    data = json.load(f)

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
                    'can_continue': True,
                    'can_rate': True,
                    'can_rename': False,  # Legacy format
                    'can_delete': True,
                })
            except Exception as e:
                print(f"Error reading legacy file {file_path}: {e}")
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
        # Delete from Firestore
        if await EvolutionEngine.delete_evolution_for_user(user.uid, evolution_id):
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

@app.get('/api/evolution/{evolution_id}')
async def get_evolution(evolution_id: str, user: UserInfo = Depends(require_auth)):
    """Get evolution data from Firestore"""
    try:
        # Load from Firestore
        data = await db.get_evolution(user.uid, evolution_id)
        if data:
            return JSONResponse({
                'id': evolution_id,
                'name': data.get('name', evolution_id),
                'timestamp': data.get('updated_at') or data.get('created_at'),
                'data': data
            })

        raise HTTPException(status_code=404, detail="Evolution not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def _load_rating_evolution(
    evolution_id: str,
    user: Optional[UserInfo] = None,
) -> Tuple[Optional[Path], Dict[str, Any], str]:
    """Load rating evolution from Firestore (preferred) or local file fallback."""
    if user:
        firestore_data = await db.get_evolution(user.uid, evolution_id)
        if firestore_data:
            return None, firestore_data, "firestore"

    file_path = DATA_DIR / f"{evolution_id}.json"
    if file_path.exists():
        with open(file_path) as f:
            evolution_data = json.load(f)
        return file_path, evolution_data, "file"

    raise HTTPException(status_code=404, detail="Evolution not found")


async def _save_rating_evolution(
    evolution_id: str,
    evolution_data: Dict[str, Any],
    source: str,
    user: Optional[UserInfo] = None,
    file_path: Optional[Path] = None,
) -> None:
    """Persist rating evolution back to its source."""
    if source == "firestore":
        if not user:
            raise HTTPException(status_code=401, detail="Authentication required to save this evolution")
        await db.save_evolution(user.uid, evolution_id, evolution_data)
        return

    target_path = file_path or (DATA_DIR / f"{evolution_id}.json")
    with open(target_path, "w") as f:
        json.dump(evolution_data, f, indent=2)


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
    if normalized in {"off", "low", "medium", "high"}:
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
    critic_input = getattr(critic_agent, "input_token_count", 0)
    critic_output = getattr(critic_agent, "output_token_count", 0)
    critic_total = getattr(critic_agent, "total_token_count", 0)

    from idea.config import model_prices_per_million_tokens

    critic_model = getattr(critic_agent, "model_name", "gemini-2.0-flash")
    default_price = {"input": 0.1, "output": 0.4}
    critic_pricing = model_prices_per_million_tokens.get(critic_model, default_price)

    critic_input_cost = (critic_pricing["input"] * critic_input) / 1_000_000
    critic_output_cost = (critic_pricing["output"] * critic_output) / 1_000_000
    total_cost = critic_input_cost + critic_output_cost

    estimates = {}
    for model in LLM_MODELS:
        model_id = model["id"]
        model_name = model.get("name", model_id)
        pricing = model_prices_per_million_tokens.get(model_id, default_price)
        est_cost = (
            pricing["input"] * critic_input / 1_000_000
            + pricing["output"] * critic_output / 1_000_000
        )
        estimates[model_id] = {"name": model_name, "cost": est_cost}

    return {
        "critic": {
            "total": critic_total,
            "input": critic_input,
            "output": critic_output,
            "model": critic_model,
            "cost": total_cost,
        },
        "total": critic_total,
        "total_input": critic_input,
        "total_output": critic_output,
        "cost": {
            "input_cost": critic_input_cost,
            "output_cost": critic_output_cost,
            "total_cost": total_cost,
            "currency": "USD",
        },
        "models": {"critic": critic_model},
        "estimates": estimates,
    }


@app.post("/api/submit-rating")
async def submit_rating(
    request: Request,
    user: Optional[UserInfo] = Depends(get_current_user),
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

        file_path, evolution_data, source = await _load_rating_evolution(evolution_id, user=user)
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
            file_path=file_path,
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
    user: Optional[UserInfo] = Depends(get_current_user),
):
    """Automatically rate ideas using Swiss rounds from the Critic tournament flow."""
    state: Optional[UserEvolutionState] = None
    try:
        data = await request.json()
        evolution_id = data.get("evolutionId")
        model_id = data.get("modelId", DEFAULT_MODEL)
        skip_save = bool(data.get("skipSave", False))
        idea_type = data.get("ideaType")

        if user:
            state = await user_states.get(user.uid)

        file_path, evolution_data, source = await _load_rating_evolution(evolution_id, user=user)
        if not idea_type:
            idea_type = evolution_data.get("idea_type", get_default_template_id())

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
                    idea_a["auto_match_count"] += 1
                    idea_b["auto_match_count"] += 1

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
            critic.get_tournament_ranks(
                all_ideas,
                idea_type,
                rounds=num_rounds,
                progress_callback=_progress_callback,
                details=rounds_details,
                full_tournament_rounds=full_tournament_rounds,
            )

        await asyncio.to_thread(_run_tournament)
        _apply_resolved_pairs()

        if not skip_save:
            await _save_rating_evolution(
                evolution_id=evolution_id,
                evolution_data=evolution_data,
                source=source,
                user=user,
                file_path=file_path,
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
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in auto-rate: {e}")
        print(error_details)
        _update_autorating_state(
            state,
            {
                "is_running": False,
                "status": "error",
                "status_message": "Auto-rating failed.",
                "error": str(e),
            },
        )
        return JSONResponse({
            "status": "error",
            "message": str(e),
            "details": error_details,
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
    user: Optional[UserInfo] = Depends(get_current_user),
):
    """Reset ratings for an evolution."""
    try:
        data = await request.json()
        evolution_id = data.get("evolutionId")
        rating_type = data.get("ratingType", "all")

        file_path, evolution_data, source = await _load_rating_evolution(evolution_id, user=user)

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
            file_path=file_path,
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
    user: Optional[UserInfo] = Depends(get_current_user),
):
    """Get the next Swiss pair for manual rating."""
    try:
        data = await request.json()
        evolution_id = data.get("evolution_id")
        if not evolution_id:
            raise HTTPException(status_code=400, detail="Evolution ID is required")

        file_path, evolution_data, source = await _load_rating_evolution(evolution_id, user=user)
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
            file_path=file_path,
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

@app.get("/api/debug/idea/{evolution_id}/{idea_id}")
async def debug_idea_state(evolution_id: str, idea_id: str):
    """Debug endpoint to check the current state of an idea"""
    try:
        # Load the evolution data
        file_path = DATA_DIR / f"{evolution_id}.json"
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Evolution not found")

        with open(file_path) as f:
            evolution_data = json.load(f)

        # Find the idea
        found_idea = None
        for generation in evolution_data.get('history', []):
            for idea in generation:
                if idea.get('id') == idea_id:
                    found_idea = idea
                    break
            if found_idea:
                break

        if not found_idea:
            raise HTTPException(status_code=404, detail="Idea not found")

        return {
            'id': found_idea.get('id'),
            'title': found_idea.get('title', 'Untitled'),
            'ratings': found_idea.get('ratings', {}),
            'match_count': found_idea.get('match_count', 0),
            'manual_match_count': found_idea.get('manual_match_count', 0),
            'auto_match_count': found_idea.get('auto_match_count', 0),
            'elo': found_idea.get('elo', 1500)
        }

    except Exception as e:
        print(f"Error in debug_idea_state: {e}")
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
