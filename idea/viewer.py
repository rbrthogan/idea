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
import random
from pathlib import Path
import json
from pydantic import BaseModel
from datetime import datetime
import uuid
from typing import Dict, List, Tuple, Optional

from idea.evolution import EvolutionEngine
from idea.llm import Critic
from idea.config import LLM_MODELS, DEFAULT_MODEL, DEFAULT_CREATIVE_TEMP, DEFAULT_TOP_P
from idea.template_manager import router as template_router
from idea.prompts.loader import list_available_templates
from idea.admin import router as admin_router
from idea.auth import require_auth, UserInfo
from fastapi import Depends
from idea import database as db
from idea.user_state import user_states, UserEvolutionState

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


# Add this class for the request body
class SaveEvolutionRequest(BaseModel):
    data: dict
    filename: str

class ApiKeyRequest(BaseModel):
    api_key: str

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
        return JSONResponse({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }, status_code=500)

@app.get("/api/template-types")
async def get_template_types():
    """Get available template types for the evolution UI dropdown"""
    try:
        templates = list_available_templates()

        # Format for dropdown - include both YAML and Python templates
        template_types = []
        for template_id, template_info in templates.items():
            # Include templates that don't have errors
            if 'error' not in template_info:
                template_types.append({
                    "id": template_id,
                    "name": template_info.get('name', template_id.replace('_', ' ').title()),
                    "description": template_info.get('description', ''),
                    "type": template_info.get('type', 'unknown'),
                    "author": template_info.get('author', 'Unknown')
                })

        # Sort by name
        template_types.sort(key=lambda x: x['name'])

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
        if state.status.get("is_running"):
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

        # Get tournament parameters with defaults
        try:
            tournament_size = int(data.get('tournamentSize', 5))
            tournament_comparisons = int(data.get('tournamentComparisons', 35))
            print(f"Parsed tournament values: size={tournament_size}, comparisons={tournament_comparisons}")
        except ValueError as e:
            print(f"Error parsing tournament values: {e}")
            # Use defaults if parsing fails
            tournament_size = 5
            tournament_comparisons = 35

        # Get mutation rate with default
        try:
            mutation_rate = float(data.get('mutationRate', 0.2))
            print(f"Parsed mutation rate: {mutation_rate}")
        except ValueError as e:
            print(f"Error parsing mutation rate: {e}")
            mutation_rate = 0.2

        # Get Oracle parameters with defaults

        # Get thinking budget parameter (only for Gemini 2.5 models)
        thinking_budget = data.get('thinkingBudget')
        if thinking_budget is not None:
            thinking_budget = int(thinking_budget)
            print(f"Parsed thinking budget: {thinking_budget}")
        else:
            print("No thinking budget specified (non-2.5 model or not set)")

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
              f"tournament: size={tournament_size}, comparisons={tournament_comparisons}, "
              f"mutation_rate={mutation_rate}, thinking_budget={thinking_budget}, max_budget={max_budget}")

        # Create and run evolution with specified parameters
        state.engine = EvolutionEngine(
            pop_size=pop_size,
            generations=generations,
            idea_type=idea_type,
            model_type=model_type,
            creative_temp=creative_temp,
            top_p=top_p,
            tournament_size=tournament_size,
            tournament_comparisons=tournament_comparisons,
            mutation_rate=mutation_rate,
            thinking_budget=thinking_budget,
            max_budget=max_budget,
            api_key=api_key,
        )

        # Initialize evolution with name (auto-generates if not provided)
        evolution_name = (data.get('evolutionName') or '').strip() or None
        state.engine.initialize_evolution(name=evolution_name)
        print(f"Evolution initialized: '{state.engine.evolution_name}' (ID: {state.engine.evolution_id})")

        # Generate contexts for each idea
        contexts = state.engine.generate_contexts()

        # Clear the queue
        state.reset_queue()

        # Set up evolution status
        state.status = {
            "current_generation": 0,
            "total_generations": generations,
            "is_running": True,
            "history": [],
            "contexts": contexts,
            "progress": 0
        }

        # Put initial status in queue
        await state.queue.put(state.status.copy())

        # Start evolution in background task
        asyncio.create_task(run_evolution_task(state))

        return JSONResponse({
            "status": "success",
            "message": "Evolution started",
            "evolution_id": state.engine.evolution_id,
            "evolution_name": state.engine.evolution_name,
            "contexts": contexts,
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
        return JSONResponse(
            {"status": "error", "message": "No evolution is currently running"},
            status_code=400,
        )

    # Request stop
    state.engine.stop_evolution()

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
        return JSONResponse(
            {"status": "error", "message": "No evolution is currently running"},
            status_code=400,
        )

    # Save checkpoint before forcing stop
    checkpoint_id = None
    try:
        state.engine.save_checkpoint(status='force_stopped')
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

    return JSONResponse({
        "status": "success",
        "message": "Evolution force stopped. A checkpoint has been saved.",
        "checkpoint_id": checkpoint_id,
        "is_resumable": True
    })

@app.get("/api/checkpoints")
async def list_checkpoints():
    """
    List all available evolution checkpoints.
    """
    try:
        checkpoints = EvolutionEngine.list_checkpoints()
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

    # Load the engine from checkpoint
    state.engine = EvolutionEngine.load_checkpoint(checkpoint_id, api_key=api_key)
    if state.engine is None:
        return JSONResponse(
            {"status": "error", "message": "Failed to load checkpoint"},
            status_code=500,
        )

    # Clear the queue
    state.reset_queue()

    # Set up evolution status
    state.status = {
        "current_generation": state.engine.current_generation,
        "total_generations": state.engine.generations,
        "is_running": True,
        "is_resuming": True,
        "checkpoint_id": checkpoint_id,
        "history": [[idea_to_dict(idea) for idea in gen] for gen in state.engine.history],
        "contexts": state.engine.contexts,
        "specific_prompts": state.engine.specific_prompts,
        "breeding_prompts": state.engine.breeding_prompts,
        "progress": (state.engine.current_generation / state.engine.generations) * 100 if state.engine.generations > 0 else 0
    }

    # Put initial status in queue
    await state.queue.put(state.status.copy())

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
        "specific_prompts": state.engine.specific_prompts
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

    # Load the engine
    if checkpoint_id:
        state.engine = EvolutionEngine.load_checkpoint(checkpoint_id, api_key=api_key)
    else:
        # Load from evolution file and create a checkpoint
        state.engine = await load_engine_from_evolution(evolution_id, api_key=api_key)

    if state.engine is None:
        return JSONResponse(
            {"status": "error", "message": "Failed to load evolution state"},
            status_code=500,
        )

    # Clear the queue
    state.reset_queue()

    # Set up evolution status
    new_total = state.engine.generations + additional_generations
    state.status = {
        "current_generation": state.engine.current_generation,
        "total_generations": new_total,
        "is_running": True,
        "is_continuing": True,
        "checkpoint_id": state.engine.checkpoint_id,
        "history": [[idea_to_dict(idea) for idea in gen] for gen in state.engine.history],
        "contexts": state.engine.contexts,
        "specific_prompts": state.engine.specific_prompts,
        "breeding_prompts": state.engine.breeding_prompts,
        "progress": (state.engine.current_generation / new_total) * 100 if new_total > 0 else 0
    }

    # Put initial status in queue
    await state.queue.put(state.status.copy())

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
        "specific_prompts": state.engine.specific_prompts
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

    # Load from evolution file and create engine
    state.engine = await load_engine_from_evolution(evolution_id, api_key=api_key)

    if state.engine is None:
        return JSONResponse(
            {"status": "error", "message": f"Failed to load evolution '{evolution_id}'"},
            status_code=500,
        )

    # Clear the queue
    state.reset_queue()

    # Set up evolution status
    new_total = state.engine.generations + additional_generations
    state.status = {
        "current_generation": state.engine.current_generation,
        "total_generations": new_total,
        "is_running": True,
        "is_continuing": True,
        "checkpoint_id": state.engine.checkpoint_id,
        "history": [[idea_to_dict(idea) for idea in gen] for gen in state.engine.history],
        "contexts": state.engine.contexts,
        "specific_prompts": state.engine.specific_prompts,
        "breeding_prompts": state.engine.breeding_prompts,
        "progress": (state.engine.current_generation / new_total) * 100 if new_total > 0 else 0
    }

    # Put initial status in queue
    await state.queue.put(state.status.copy())

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
        "history": [[idea_to_dict(idea) for idea in gen] for gen in state.engine.history],
        "contexts": state.engine.contexts,
        "specific_prompts": state.engine.specific_prompts
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
            tournament_size=config.get('tournament_size', 5),
            tournament_comparisons=config.get('tournament_comparisons', 35),
            thinking_budget=config.get('thinking_budget'),
            max_budget=config.get('max_budget'),
            mutation_rate=config.get('mutation_rate', 0.2),
            api_key=api_key,
        )

        # Restore state from the saved evolution
        engine.contexts = data.get('contexts', [])
        engine.specific_prompts = data.get('specific_prompts', [])
        engine.breeding_prompts = data.get('breeding_prompts', [])
        engine.diversity_history = data.get('diversity_history', [])
        engine.current_generation = len(history)

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
            update_data['history'] = [
                [idea_to_dict(idea) for idea in generation]
                for generation in update_data['history']
            ]
            if update_data['history']:
                state.latest_data = update_data['history']

        # Add token counts if evolution is complete
        if update_data.get('is_running') is False and 'error' not in update_data:
            if hasattr(engine, 'get_total_token_count'):
                update_data['token_counts'] = engine.get_total_token_count()

        state.status.update(update_data)

        # Clear queue and add update
        state.reset_queue()
        await state.queue.put(state.status.copy())

    # Run the resumed evolution
    await engine.resume_evolution_with_updates(progress_callback)

async def run_continue_evolution_task(state: UserEvolutionState, additional_generations: int):
    """Run continued evolution in background with progress updates"""
    engine = state.engine

    async def progress_callback(update_data):
        # Convert Idea objects to dictionaries for JSON serialization
        if 'history' in update_data and isinstance(update_data['history'], list):
            update_data['history'] = [
                [idea_to_dict(idea) for idea in generation]
                for generation in update_data['history']
            ]
            if update_data['history']:
                state.latest_data = update_data['history']

        # Add token counts if evolution is complete
        if update_data.get('is_running') is False and 'error' not in update_data:
            if hasattr(engine, 'get_total_token_count'):
                update_data['token_counts'] = engine.get_total_token_count()

        state.status.update(update_data)

        # Clear queue and add update
        state.reset_queue()
        await state.queue.put(state.status.copy())

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

        # Convert Idea objects to dictionaries for JSON serialization
        if 'history' in update_data and isinstance(update_data['history'], list):
            update_data['history'] = [
                [idea_to_dict(idea) for idea in generation]
                for generation in update_data['history']
            ]

            # Store the latest evolution data for rating
            if update_data['history']:
                state.latest_data = update_data['history']

        # If evolution is complete, add token counts
        if update_data.get('is_running') is False and 'error' not in update_data:
            # Get token counts from the engine
            if hasattr(engine, 'get_total_token_count'):
                update_data['token_counts'] = engine.get_total_token_count()
                print(f"Evolution complete. Total tokens: {update_data['token_counts']['total']}")

        # Update the evolution status by merging the new data
        state.status.update(update_data)

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
async def get_progress(user: UserInfo = Depends(require_auth)):
    """Returns the current progress of the evolution"""
    state = await user_states.get(user.uid)

    # If there's a new update in the queue, get it
    try:
        # Get the latest update from the queue without waiting
        if not state.queue.empty():
            state.status = await state.queue.get()
    except Exception as e:
        print(f"Error getting queue updates: {e}")

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
async def get_evolutions(request: Request):
    """Get list of evolutions from file system"""
    evolutions_list = []

    try:
        for file_path in DATA_DIR.glob('*.json'):
            file_stat = file_path.stat()
            evolutions_list.append({
                'id': file_path.stem,  # Use filename without extension as ID
                'timestamp': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                'filename': file_path.name
            })

        # Sort by timestamp descending
        evolutions_list.sort(key=lambda x: x['timestamp'], reverse=True)
        return JSONResponse(evolutions_list)
    except Exception as e:
        print(f"Error reading evolution files: {e}")
        return JSONResponse([])

@app.get('/api/history')
async def get_unified_history():
    """
    Get unified history of all evolutions.
    Combines new unified storage + legacy checkpoints + legacy saved files.
    """
    history_items = []

    try:
        # 1. Load from unified evolutions directory (new format)
        evolutions = EvolutionEngine.list_evolutions()
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

        # 2. Load legacy checkpoints (for backwards compatibility)
        checkpoints = EvolutionEngine.list_checkpoints()
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
async def rename_evolution(request: Request, evolution_id: str):
    """Rename an evolution"""
    try:
        data = await request.json()
        new_name = data.get('name', '').strip()

        if not new_name:
            return JSONResponse({
                'status': 'error',
                'message': 'Name cannot be empty'
            }, status_code=400)

        # Try unified evolutions first
        if EvolutionEngine.rename_evolution(evolution_id, new_name):
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
async def delete_evolution_endpoint(evolution_id: str):
    """Delete an evolution"""
    try:
        # Try unified evolutions first
        if EvolutionEngine.delete_evolution(evolution_id):
            return JSONResponse({'status': 'success', 'message': 'Evolution deleted'})

        # Try legacy checkpoint
        if EvolutionEngine.delete_checkpoint(evolution_id):
            return JSONResponse({'status': 'success', 'message': 'Checkpoint deleted'})

        # Try legacy saved file
        legacy_path = DATA_DIR / f"{evolution_id}.json"
        if legacy_path.exists():
            legacy_path.unlink()
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
async def get_evolution(request: Request, evolution_id: str):
    """Get evolution data from file"""
    try:
        # Try unified evolutions directory first
        evolution_path = EVOLUTIONS_DIR / f"{evolution_id}.json"
        if evolution_path.exists():
            with open(evolution_path) as f:
                data = json.load(f)
                return JSONResponse({
                    'id': evolution_id,
                    'name': data.get('name', evolution_id),
                    'timestamp': data.get('updated_at') or data.get('created_at'),
                    'data': data
                })

        # Fall back to legacy data directory
        file_path = DATA_DIR / f"{evolution_id}.json"
        if file_path.exists():
            with open(file_path) as f:
                data = json.load(f)
                return JSONResponse({
                    'id': evolution_id,
                    'timestamp': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    'data': data
                })

        raise HTTPException(status_code=404, detail="Evolution not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/submit-rating")
async def submit_rating(request: Request):
    """Submit a manual rating for a pair of ideas"""
    try:
        data = await request.json()
        idea_a_id = data.get('idea_a_id')
        idea_b_id = data.get('idea_b_id')
        outcome = data.get('outcome')
        evolution_id = data.get('evolution_id')

        print(f"Submitting rating for {idea_a_id} vs {idea_b_id}, outcome: {outcome}")

        # Load the evolution data
        file_path = DATA_DIR / f"{evolution_id}.json"
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Evolution not found")

        with open(file_path) as f:
            evolution_data = json.load(f)

        # Find the ideas
        idea_a = None
        idea_b = None

        for generation in evolution_data.get('history', []):
            for idea in generation:
                if idea.get('id') == idea_a_id:
                    idea_a = idea
                elif idea.get('id') == idea_b_id:
                    idea_b = idea

        if not idea_a or not idea_b:
            raise HTTPException(status_code=404, detail="Ideas not found")

        # Log current state before update
        print(f"Before update - Idea A ({idea_a_id}): manual_match_count={idea_a.get('manual_match_count', 0)}, manual_elo={idea_a.get('ratings', {}).get('manual', 1500)}")
        print(f"Before update - Idea B ({idea_b_id}): manual_match_count={idea_b.get('manual_match_count', 0)}, manual_elo={idea_b.get('ratings', {}).get('manual', 1500)}")

        # Initialize ratings if not present
        if 'ratings' not in idea_a:
            idea_a['ratings'] = {'auto': 1500, 'manual': 1500}
        elif isinstance(idea_a['ratings'], (int, float)):
            old_elo = idea_a['ratings']
            idea_a['ratings'] = {'auto': old_elo, 'manual': old_elo}

        if 'ratings' not in idea_b:
            idea_b['ratings'] = {'auto': 1500, 'manual': 1500}
        elif isinstance(idea_b['ratings'], (int, float)):
            old_elo = idea_b['ratings']
            idea_b['ratings'] = {'auto': old_elo, 'manual': old_elo}

        # Initialize match counts if not present
        if 'match_count' not in idea_a:
            idea_a['match_count'] = 0
        if 'match_count' not in idea_b:
            idea_b['match_count'] = 0

        # Initialize manual match counts if not present
        if 'manual_match_count' not in idea_a:
            idea_a['manual_match_count'] = 0
        if 'manual_match_count' not in idea_b:
            idea_b['manual_match_count'] = 0

        # Initialize auto match counts if not present
        if 'auto_match_count' not in idea_a:
            idea_a['auto_match_count'] = 0
        if 'auto_match_count' not in idea_b:
            idea_b['auto_match_count'] = 0

        # Increment match counts
        idea_a['match_count'] += 1
        idea_b['match_count'] += 1

        # Increment manual match counts specifically
        idea_a['manual_match_count'] += 1
        idea_b['manual_match_count'] += 1

        # Convert outcome to numeric value
        if outcome == "A":
            outcome_value = 1
        elif outcome == "B":
            outcome_value = 0
        else:  # Tie
            outcome_value = 0.5

        # Calculate new Elos for manual ratings
        k_factor = 32
        expected_a = 1 / (1 + 10 ** ((idea_b['ratings']['manual'] - idea_a['ratings']['manual']) / 400))
        expected_b = 1 / (1 + 10 ** ((idea_a['ratings']['manual'] - idea_b['ratings']['manual']) / 400))

        idea_a['ratings']['manual'] = round(idea_a['ratings']['manual'] + k_factor * (outcome_value - expected_a))
        idea_b['ratings']['manual'] = round(idea_b['ratings']['manual'] + k_factor * (1 - outcome_value - expected_b))

        # Log state after update
        print(f"After update - Idea A ({idea_a_id}): manual_match_count={idea_a['manual_match_count']}, manual_elo={idea_a['ratings']['manual']}")
        print(f"After update - Idea B ({idea_b_id}): manual_match_count={idea_b['manual_match_count']}, manual_elo={idea_b['ratings']['manual']}")

        # Save the updated data
        with open(file_path, 'w') as f:
            json.dump(evolution_data, f, indent=2)

        print(f"Successfully saved rating data to {file_path}")

        # Return the updated ELO ratings
        return JSONResponse({
            'status': 'success',
            'updated_elos': {
                idea_a_id: idea_a['ratings']['manual'],
                idea_b_id: idea_b['ratings']['manual']
            },
            'updated_match_counts': {
                idea_a_id: {
                    'total': idea_a['match_count'],
                    'manual': idea_a['manual_match_count'],
                    'auto': idea_a['auto_match_count']
                },
                idea_b_id: {
                    'total': idea_b['match_count'],
                    'manual': idea_b['manual_match_count'],
                    'auto': idea_b['auto_match_count']
                }
            }
        })

    except Exception as e:
        print(f"Error submitting rating: {e}")
        return JSONResponse({
            'status': 'error',
            'message': str(e)
        }, status_code=500)

@app.post("/api/auto-rate")
async def auto_rate(request: Request):
    """Automatically rate ideas using the Critic agent from llm.py"""
    try:
        data = await request.json()
        num_comparisons = int(data.get('numComparisons', 10))
        evolution_id = data.get('evolutionId')
        model_id = data.get('modelId', DEFAULT_MODEL)
        skip_save = data.get('skipSave', False)
        elo_range = int(data.get('eloRange', 100))  # Get ELO range from request, default to 100

        # Get idea_type from the request or use a default
        idea_type = data.get('ideaType', get_default_template_id())

        print(f"Starting auto-rating for evolution {evolution_id} with {num_comparisons} comparisons using model {model_id}")
        print(f"Using ELO range of ±{elo_range} for matching ideas")

        # Load the evolution data
        file_path = DATA_DIR / f"{evolution_id}.json"
        if not file_path.exists():
            print(f"Evolution file not found: {file_path}")
            raise HTTPException(status_code=404, detail="Evolution not found")

        with open(file_path) as f:
            evolution_data = json.load(f)

        # Try to extract idea_type from the evolution data if not provided in request
        if 'idea_type' in evolution_data and not idea_type:
            idea_type = evolution_data.get('idea_type', get_default_template_id())

        # Create a mapping from idea ID to its location in the evolution_data structure
        idea_map = {}

        # Extract all ideas from all generations with generation info
        all_ideas = []
        for gen_index, generation in enumerate(evolution_data.get('history', [])):
            for idea_index, idea in enumerate(generation):
                # Add an ID if not present
                if 'id' not in idea:
                    idea['id'] = f"idea_{len(all_ideas)}"

                # Store the location of this idea in the evolution_data structure
                idea_map[idea['id']] = (gen_index, idea_index)

                # Initialize ratings if not present
                if 'ratings' not in idea:
                    idea['ratings'] = {
                        'auto': 1500,
                        'manual': 1500
                    }
                elif isinstance(idea['ratings'], (int, float)):
                    # Convert old format to new format
                    old_elo = idea['ratings']
                    idea['ratings'] = {
                        'auto': old_elo,
                        'manual': old_elo
                    }
                elif 'auto' not in idea['ratings']:
                    idea['ratings']['auto'] = 1500
                elif 'manual' not in idea['ratings']:
                    idea['ratings']['manual'] = 1500

                # Initialize match count if not present
                if 'match_count' not in idea:
                    idea['match_count'] = 0

                # Initialize auto_match_count if not present
                if 'auto_match_count' not in idea:
                    idea['auto_match_count'] = 0

                # Initialize manual_match_count if not present
                if 'manual_match_count' not in idea:
                    idea['manual_match_count'] = 0

                # For backward compatibility
                idea['elo'] = idea['ratings']['auto']

                # Add generation info
                idea['generation'] = gen_index + 1
                all_ideas.append(idea)

        print(f"Found {len(all_ideas)} ideas to rate")

        if len(all_ideas) < 2:
            return JSONResponse({
                'status': 'error',
                'message': 'Not enough ideas to compare (minimum 2 required)'
            }, status_code=400)

        # Determine the appropriate thinking budget for the model
        def get_default_thinking_budget(model_name):
            """Get the default thinking budget for a model, same as main app logic"""
            from idea.config import THINKING_BUDGET_CONFIG

            # Only 2.5 and 3.0 models support thinking budget
            if "2.5" not in model_name and "3-pro" not in model_name:
                return None

            # Get the config for this model and use its default value
            # This matches the main app logic:
            # - gemini-2.5-pro: default = 128 (minimum, can't disable)
            # - gemini-2.5-flash: default = 0 (disabled)
            # - gemini-2.5-flash-lite: default = 0 (disabled)
            config = THINKING_BUDGET_CONFIG.get(model_name, {})
            return config.get('default', 0)  # Default to 0 (disabled) if not found

        thinking_budget = get_default_thinking_budget(model_id)

        # Create a critic agent from llm.py with the specified model and app defaults
        # Use the same defaults as the evolution engine to ensure consistency
        critic = Critic(
            provider="google_generative_ai",
            model_name=model_id,
            temperature=DEFAULT_CREATIVE_TEMP,
            top_p=DEFAULT_TOP_P,
            thinking_budget=thinking_budget
        )

        # Perform the requested number of comparisons
        results = []
        total_comparisons_completed = 0

        # First, count existing match counts to track total comparisons
        for idea in all_ideas:
            total_comparisons_completed += idea.get('match_count', 0)

        # Divide by 2 since each comparison involves 2 ideas
        total_comparisons_completed = total_comparisons_completed // 2

        print(f"Starting with {total_comparisons_completed} existing comparisons")

        # Maximum ELO difference for matching ideas - use the value from the request
        max_elo_diff = elo_range

        enable_parallel_autorate = os.environ.get("ENABLE_PARALLEL_AUTORATE", "1") != "0"

        if enable_parallel_autorate and num_comparisons > 1 and len(all_ideas) >= 4:
            # Build pairs using the existing selection strategy on a snapshot
            pairs = []
            for _ in range(num_comparisons):
                idea_a, idea_b = select_efficient_pair(all_ideas, rating_type='auto', max_elo_diff=max_elo_diff)
                if idea_a is None or idea_b is None:
                    continue
                idx_a = all_ideas.index(idea_a)
                idx_b = all_ideas.index(idea_b)
                pairs.append((idx_a, idx_b))

            from idea.ratings import parallel_evaluate_pairs
            concurrency = int(os.environ.get("AUTORATE_CONCURRENCY", "8"))
            results_parallel = parallel_evaluate_pairs(
                pairs=pairs,
                items=all_ideas,
                compare_fn=lambda a, b, _: critic.compare_ideas(a, b, idea_type),
                idea_type=idea_type,
                concurrency=concurrency,
                randomize_presentation=True,
            )

            for idx_a, idx_b, winner in results_parallel:
                idea_a = all_ideas[idx_a]
                idea_b = all_ideas[idx_b]
                if winner is None:
                    continue
                idea_a['match_count'] += 1
                idea_b['match_count'] += 1
                idea_a['auto_match_count'] += 1
                idea_b['auto_match_count'] += 1

                # ELO update identical to sequential path below
                k_factor = 32
                expected_a = 1 / (1 + 10 ** ((idea_b['ratings']['auto'] - idea_a['ratings']['auto']) / 400))
                expected_b = 1 / (1 + 10 ** ((idea_a['ratings']['auto'] - idea_b['ratings']['auto']) / 400))
                if winner == "A":
                    idea_a['ratings']['auto'] = round(idea_a['ratings']['auto'] + k_factor * (1 - expected_a))
                    idea_b['ratings']['auto'] = round(idea_b['ratings']['auto'] + k_factor * (0 - expected_b))
                elif winner == "B":
                    idea_a['ratings']['auto'] = round(idea_a['ratings']['auto'] + k_factor * (0 - expected_a))
                    idea_b['ratings']['auto'] = round(idea_b['ratings']['auto'] + k_factor * (1 - expected_b))
                else:
                    idea_a['ratings']['auto'] = round(idea_a['ratings']['auto'] + k_factor * (0.5 - expected_a))
                    idea_b['ratings']['auto'] = round(idea_b['ratings']['auto'] + k_factor * (0.5 - expected_b))

                # Back-compat field
                idea_a['elo'] = idea_a['ratings']['auto']
                idea_b['elo'] = idea_b['ratings']['auto']

                # Update original evolution data structure
                if idea_a['id'] in idea_map:
                    gen_idx, idea_idx = idea_map[idea_a['id']]
                    evolution_data['history'][gen_idx][idea_idx]['ratings'] = idea_a['ratings']
                    evolution_data['history'][gen_idx][idea_idx]['elo'] = idea_a['elo']
                    evolution_data['history'][gen_idx][idea_idx]['match_count'] = idea_a['match_count']
                    evolution_data['history'][gen_idx][idea_idx]['auto_match_count'] = idea_a['auto_match_count']
                if idea_b['id'] in idea_map:
                    gen_idx, idea_idx = idea_map[idea_b['id']]
                    evolution_data['history'][gen_idx][idea_idx]['ratings'] = idea_b['ratings']
                    evolution_data['history'][gen_idx][idea_idx]['elo'] = idea_b['elo']
                    evolution_data['history'][gen_idx][idea_idx]['match_count'] = idea_b['match_count']
                    evolution_data['history'][gen_idx][idea_idx]['auto_match_count'] = idea_b['auto_match_count']

                # Record the result for API response accounting
                results.append({
                    'idea_a': idea_a.get('id', 'unknown'),
                    'idea_b': idea_b.get('id', 'unknown'),
                    'outcome': winner,
                    'new_elo_a': idea_a['ratings']['auto'],
                    'new_elo_b': idea_b['ratings']['auto']
                })
        else:
            for i in range(num_comparisons):
                print(f"Comparison {i+1}/{num_comparisons}")

                # Use the shared efficient pair selection function
                idea_a, idea_b = select_efficient_pair(all_ideas, rating_type='auto', max_elo_diff=max_elo_diff)

                if idea_a is None or idea_b is None:
                    print("Failed to select suitable pair, skipping this comparison")
                    continue

                # Get ELO ratings (kept for potential logging/future use)
                # elo_a = idea_a['ratings']['auto']
                # elo_b = idea_b['ratings']['auto']

                # Randomize presentation order to eliminate positional bias
                if random.random() < 0.5:
                    # Present ideas in original order (A first, B second)
                    winner = critic.compare_ideas(idea_a, idea_b, idea_type)
                    order_swapped = False
                else:
                    # Present ideas in swapped order (B first, A second)
                    winner = critic.compare_ideas(idea_b, idea_a, idea_type)
                    order_swapped = True
                    # Adjust winner interpretation for swapped order
                    if winner == "A":
                        winner = "B"  # Model chose first position, but that was actually idea_b
                    elif winner == "B":
                        winner = "A"  # Model chose second position, but that was actually idea_a
                    # "tie" remains "tie"

                print(f"Winner: {winner} (order_swapped: {order_swapped})")

                # Skip this comparison if there was an error (winner is None)
                if winner is None:
                    print("Skipping this comparison due to an error")
                    continue

                # Increment match counts
                idea_a['match_count'] += 1
                idea_b['match_count'] += 1

                # Increment auto match counts specifically
            idea_a['auto_match_count'] += 1
            idea_b['auto_match_count'] += 1

            # Convert to outcome format (1 = A wins, 0 = B wins, 0.5 = tie)
            if winner == "A":
                outcome = 1
            elif winner == "B":
                outcome = 0
            else:  # Tie
                outcome = 0.5

            # Calculate new Elos for auto ratings
            k_factor = 32
            expected_a = 1 / (1 + 10 ** ((idea_b['ratings']['auto'] - idea_a['ratings']['auto']) / 400))
            expected_b = 1 / (1 + 10 ** ((idea_a['ratings']['auto'] - idea_b['ratings']['auto']) / 400))

            idea_a['ratings']['auto'] = round(idea_a['ratings']['auto'] + k_factor * (outcome - expected_a))
            idea_b['ratings']['auto'] = round(idea_b['ratings']['auto'] + k_factor * (1 - outcome - expected_b))

            # Update the elo field for backward compatibility
            idea_a['elo'] = idea_a['ratings']['auto']
            idea_b['elo'] = idea_b['ratings']['auto']

            # Update the original ideas in the evolution_data structure
            if idea_a['id'] in idea_map:
                gen_idx, idea_idx = idea_map[idea_a['id']]
                evolution_data['history'][gen_idx][idea_idx]['ratings'] = idea_a['ratings']
                evolution_data['history'][gen_idx][idea_idx]['elo'] = idea_a['elo']
                evolution_data['history'][gen_idx][idea_idx]['match_count'] = idea_a['match_count']
                evolution_data['history'][gen_idx][idea_idx]['auto_match_count'] = idea_a['auto_match_count']

            if idea_b['id'] in idea_map:
                gen_idx, idea_idx = idea_map[idea_b['id']]
                evolution_data['history'][gen_idx][idea_idx]['ratings'] = idea_b['ratings']
                evolution_data['history'][gen_idx][idea_idx]['elo'] = idea_b['elo']
                evolution_data['history'][gen_idx][idea_idx]['match_count'] = idea_b['match_count']
                evolution_data['history'][gen_idx][idea_idx]['auto_match_count'] = idea_b['auto_match_count']

            # Record the result
            results.append({
                'idea_a': idea_a.get('id', 'unknown'),
                'idea_b': idea_b.get('id', 'unknown'),
                'outcome': winner,
                'new_elo_a': idea_a['ratings']['auto'],
                'new_elo_b': idea_b['ratings']['auto']
            })

            # Save after every comparison to ensure match counts are properly tracked
            if not skip_save:
                with open(file_path, 'w') as f:
                    json.dump(evolution_data, f, indent=2)

                # Only log every 5 comparisons to reduce console output
                if (i + 1) % 5 == 0:
                    print(f"Saved progress after {i + 1} comparisons")

        print(f"Completed {len(results)} comparisons")

        # Save the final updated Elo scores back to the file (unless skipSave is True)
        if not skip_save:
            with open(file_path, 'w') as f:
                json.dump(evolution_data, f, indent=2)

        # Calculate costs similar to evolution module
        def get_autorating_costs(critic_agent):
            """Get the cost information for autorating"""
            # Get token counts from the critic agent
            critic_input = getattr(critic_agent, 'input_token_count', 0)
            critic_output = getattr(critic_agent, 'output_token_count', 0)
            critic_total = getattr(critic_agent, 'total_token_count', 0)

            # Get pricing information from config
            from idea.config import model_prices_per_million_tokens

            # Get model name for the critic
            critic_model = getattr(critic_agent, 'model_name', 'gemini-2.0-flash')

            # Default pricing if model not found in config
            default_price = {"input": 0.1, "output": 0.4}

            # Get pricing for the model
            critic_pricing = model_prices_per_million_tokens.get(critic_model, default_price)

            # Calculate cost for critic
            critic_input_cost = (critic_pricing["input"] * critic_input) / 1_000_000
            critic_output_cost = (critic_pricing["output"] * critic_output) / 1_000_000
            total_cost = critic_input_cost + critic_output_cost

            # Also estimate total cost for each available model using the critic
            # token counts so users can compare pricing.
            from idea.config import LLM_MODELS

            estimates = {}
            for model in LLM_MODELS:
                model_id = model['id']
                model_name = model.get('name', model_id)
                pricing = model_prices_per_million_tokens.get(model_id, default_price)
                est_cost = (
                    pricing['input'] * critic_input / 1_000_000
                    + pricing['output'] * critic_output / 1_000_000
                )
                estimates[model_id] = {'name': model_name, 'cost': est_cost}

            return {
                'critic': {
                    'total': critic_total,
                    'input': critic_input,
                    'output': critic_output,
                    'model': critic_model,
                    'cost': total_cost
                },
                'total': critic_total,
                'total_input': critic_input,
                'total_output': critic_output,
                'cost': {
                    'input_cost': critic_input_cost,
                    'output_cost': critic_output_cost,
                    'total_cost': total_cost,
                    'currency': 'USD'
                },
                'models': {
                    'critic': critic_model
                },
                'estimates': estimates
            }

        # Get cost information
        cost_info = get_autorating_costs(critic)

        # Return the results with sorted ideas including generation info
        # Calculate total comparisons (existing + new)
        total_comparisons = total_comparisons_completed + len(results)

        return JSONResponse({
            'status': 'success',
            'results': results,
            'ideas': sorted(all_ideas, key=lambda x: x['ratings']['auto'], reverse=True),
            'completed_comparisons': total_comparisons,
            'new_comparisons': len(results),
            'token_counts': cost_info
        })

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in auto-rate: {e}")
        print(error_details)
        return JSONResponse({
            'status': 'error',
            'message': str(e),
            'details': error_details
        }, status_code=500)

@app.get("/api/models")
async def get_models():
    """Return the list of available LLM models"""
    return JSONResponse({
        "models": LLM_MODELS,
        "default": DEFAULT_MODEL
    })

@app.post("/api/reset-ratings")
async def reset_ratings(request: Request):
    """Reset ratings for an evolution"""
    try:
        data = await request.json()
        evolution_id = data.get('evolutionId')
        rating_type = data.get('ratingType', 'all')  # 'all', 'auto', or 'manual'

        # Load the evolution data
        file_path = DATA_DIR / f"{evolution_id}.json"
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Evolution not found")

        with open(file_path) as f:
            evolution_data = json.load(f)

        # Reset ratings based on type
        for generation in evolution_data.get('history', []):
            for idea in generation:
                # Initialize ratings object if needed
                if 'ratings' not in idea:
                    idea['ratings'] = {'auto': 1500, 'manual': 1500}
                elif isinstance(idea['ratings'], (int, float)):
                    old_elo = idea['ratings']
                    idea['ratings'] = {'auto': old_elo, 'manual': old_elo}

                # Reset the specified rating type(s)
                if rating_type == 'all' or rating_type == 'auto':
                    idea['ratings']['auto'] = 1500
                    idea['elo'] = 1500  # For backward compatibility
                    # Reset auto match count
                    idea['auto_match_count'] = 0

                if rating_type == 'all' or rating_type == 'manual':
                    idea['ratings']['manual'] = 1500
                    # Reset manual match count
                    idea['manual_match_count'] = 0

                # Reset total match count if resetting all
                if rating_type == 'all':
                    idea['match_count'] = 0
                # Update total match count if resetting one type
                elif rating_type == 'auto' and 'manual_match_count' in idea:
                    idea['match_count'] = idea.get('manual_match_count', 0)
                elif rating_type == 'manual' and 'auto_match_count' in idea:
                    idea['match_count'] = idea.get('auto_match_count', 0)

        # Save the updated data
        with open(file_path, 'w') as f:
            json.dump(evolution_data, f, indent=2)

        return JSONResponse({
            'status': 'success',
            'message': f'{rating_type.capitalize()} ratings and match counts reset successfully'
        })

    except Exception as e:
        print(f"Error resetting ratings: {e}")
        return JSONResponse({
            'status': 'error',
            'message': str(e)
        }, status_code=500)

def select_efficient_pair(all_ideas: List[Dict], rating_type: str = 'auto', max_elo_diff: int = 100) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Select a pair of ideas efficiently, prioritizing those with fewer matches and similar ELO ratings.

    Args:
        all_ideas: List of all ideas to choose from
        rating_type: 'auto' for auto ratings, 'manual' for manual ratings
        max_elo_diff: Maximum ELO difference for matching ideas

    Returns:
        Tuple of (idea_a, idea_b) or (None, None) if no suitable pair found
    """
    if len(all_ideas) < 2:
        return None, None

    # Track failed attempts to find suitable pairs
    failed_attempts = 0
    max_failed_attempts = 50
    current_max_elo_diff = max_elo_diff

    # Choose the appropriate match count field based on rating type
    match_count_field = 'auto_match_count' if rating_type == 'auto' else 'manual_match_count'
    rating_field = 'auto' if rating_type == 'auto' else 'manual'

    # Sort ideas by match count (ascending) to prioritize less-rated ideas
    sorted_ideas = sorted(all_ideas, key=lambda x: x.get(match_count_field, 0))

    # Take the bottom 50% of ideas (those with fewer matches)
    candidate_pool_size = max(2, len(sorted_ideas) // 2)
    candidate_pool = sorted_ideas[:candidate_pool_size]

    # Find the minimum match count to identify truly underrated ideas
    min_match_count = min(idea.get(match_count_field, 0) for idea in candidate_pool)

    # Create a priority pool of ideas with the minimum match count
    priority_pool = [idea for idea in candidate_pool if idea.get(match_count_field, 0) == min_match_count]

    print(f"Selected candidate pool of {len(candidate_pool)} ideas with fewest {rating_type} matches")
    print(f"Priority pool has {len(priority_pool)} ideas with {min_match_count} matches")

    while True:
        # If we've tried too many times, gradually relax the ELO difference constraint
        if failed_attempts >= max_failed_attempts:
            current_max_elo_diff = current_max_elo_diff * 1.5  # Increase by 50%
            print(f"Relaxing ELO difference constraint to ±{current_max_elo_diff} after {failed_attempts} failed attempts")

            # Also expand the candidate pool if we're still struggling
            candidate_pool_size = min(len(sorted_ideas), candidate_pool_size + len(sorted_ideas) // 4)
            candidate_pool = sorted_ideas[:candidate_pool_size]
            # Update priority pool as well
            min_match_count = min(idea.get(match_count_field, 0) for idea in candidate_pool)
            priority_pool = [idea for idea in candidate_pool if idea.get(match_count_field, 0) == min_match_count]
            print(f"Expanded candidate pool to {len(candidate_pool)} ideas")

            failed_attempts = 0  # Reset counter after relaxing

        # Try to select from the priority pool first (ideas with fewest matches)
        if len(priority_pool) >= 2:
            try:
                # Randomly select from the priority pool instead of always picking the first
                idea_a = random.choice(priority_pool)

                # Find ideas within ELO range
                elo_a = idea_a['ratings'][rating_field]
                compatible_ideas = [
                    idea for idea in all_ideas
                    if idea['id'] != idea_a['id'] and
                    abs(idea['ratings'][rating_field] - elo_a) <= current_max_elo_diff
                ]

                if compatible_ideas:
                    # Select a random compatible idea
                    idea_b = random.choice(compatible_ideas)
                    elo_b = idea_b['ratings'][rating_field]

                    # Log match counts for transparency
                    match_count_a = idea_a.get(match_count_field, 0)
                    match_count_b = idea_b.get(match_count_field, 0)
                    print(f"Selected efficient pair: {idea_a.get('id')} (ELO: {elo_a}, Matches: {match_count_a}) vs {idea_b.get('id')} (ELO: {elo_b}, Matches: {match_count_b})")

                    return idea_a, idea_b
                else:
                    # If no compatible ideas, remove idea_a from priority pool and try again
                    priority_pool.remove(idea_a)
                    failed_attempts += 1
            except Exception as e:
                print(f"Error selecting from priority pool: {e}")
                # Fallback to random selection
                if len(all_ideas) >= 2:
                    idea_a, idea_b = random.sample(all_ideas, 2)
                    return idea_a, idea_b
                failed_attempts += 1
        elif len(candidate_pool) >= 2:
            try:
                # Fallback to random selection from candidate pool
                idea_a = random.choice(candidate_pool)

                # Find ideas within ELO range
                elo_a = idea_a['ratings'][rating_field]
                compatible_ideas = [
                    idea for idea in all_ideas
                    if idea['id'] != idea_a['id'] and
                    abs(idea['ratings'][rating_field] - elo_a) <= current_max_elo_diff
                ]

                if compatible_ideas:
                    # Select a random compatible idea
                    idea_b = random.choice(compatible_ideas)
                    elo_b = idea_b['ratings'][rating_field]

                    # Log match counts for transparency
                    match_count_a = idea_a.get(match_count_field, 0)
                    match_count_b = idea_b.get(match_count_field, 0)
                    print(f"Selected efficient pair from candidate pool: {idea_a.get('id')} (ELO: {elo_a}, Matches: {match_count_a}) vs {idea_b.get('id')} (ELO: {elo_b}, Matches: {match_count_b})")

                    return idea_a, idea_b
                else:
                    # If no compatible ideas, remove idea_a from candidate pool and try again
                    candidate_pool.remove(idea_a)
                    failed_attempts += 1
            except Exception as e:
                print(f"Error selecting from candidate pool: {e}")
                # Fallback to random selection
                if len(all_ideas) >= 2:
                    idea_a, idea_b = random.sample(all_ideas, 2)
                    return idea_a, idea_b
                failed_attempts += 1
        else:
            try:
                # Fallback to random selection if candidate pool is depleted
                idea_a, idea_b = random.sample(all_ideas, 2)

                # Get their ELO ratings
                elo_a = idea_a['ratings'][rating_field]
                elo_b = idea_b['ratings'][rating_field]

                # Check if they're within the allowed ELO difference
                if abs(elo_a - elo_b) <= current_max_elo_diff:
                    print(f"Found suitable pair with ELO difference: {abs(elo_a - elo_b)}")
                    return idea_a, idea_b
                else:
                    failed_attempts += 1
            except Exception as e:
                print(f"Error in random selection: {e}")
                # Just pick any two ideas as a last resort
                if len(all_ideas) >= 2:
                    idea_a, idea_b = random.sample(all_ideas, 2)
                    return idea_a, idea_b
                failed_attempts += 1

        # If we've tried too many times, just use any random pair
        if failed_attempts >= max_failed_attempts * 2:
            try:
                idea_a, idea_b = random.sample(all_ideas, 2)
                elo_a = idea_a['ratings'][rating_field]
                elo_b = idea_b['ratings'][rating_field]
                print(f"Using random pair with ELO difference {abs(elo_a - elo_b)} after {failed_attempts} failed attempts")
                return idea_a, idea_b
            except Exception as e:
                print(f"Error in last resort selection: {e}")
                # Absolute last resort - just pick the first two ideas
                if len(all_ideas) >= 2:
                    return all_ideas[0], all_ideas[1] if len(all_ideas) > 1 else all_ideas[0]
                return None, None

@app.post("/api/get-efficient-pair")
async def get_efficient_pair(request: Request):
    """Get an efficiently selected pair of ideas for manual rating"""
    try:
        data = await request.json()
        evolution_id = data.get('evolution_id')
        elo_range = data.get('elo_range', 100)

        print(f"Getting efficient pair for evolution {evolution_id} with ELO range ±{elo_range}")

        if not evolution_id:
            raise HTTPException(status_code=400, detail="Evolution ID is required")

        # Load the evolution data
        file_path = DATA_DIR / f"{evolution_id}.json"
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Evolution not found")

        with open(file_path) as f:
            evolution_data = json.load(f)

        # Collect all ideas from all generations, but only keep the latest version of each idea
        all_ideas = []
        idea_versions = {}  # Track the latest version of each idea

        for generation in evolution_data.get('history', []):
            for idea in generation:
                # Initialize ratings and match counts if not present
                if 'ratings' not in idea:
                    idea['ratings'] = {'auto': 1500, 'manual': 1500}
                elif isinstance(idea['ratings'], (int, float)):
                    old_elo = idea['ratings']
                    idea['ratings'] = {'auto': old_elo, 'manual': old_elo}
                elif 'manual' not in idea['ratings']:
                    idea['ratings']['manual'] = 1500

                # Initialize match counts if not present
                if 'match_count' not in idea:
                    idea['match_count'] = 0
                if 'manual_match_count' not in idea:
                    idea['manual_match_count'] = 0
                if 'auto_match_count' not in idea:
                    idea['auto_match_count'] = 0

                # Add ID if not present
                if 'id' not in idea:
                    idea['id'] = f"idea_{len(all_ideas)}"

                # Always keep the latest version of each idea (overwrites previous versions)
                idea_versions[idea['id']] = idea

        # Convert to list of latest versions only
        all_ideas = list(idea_versions.values())

        # Log some sample match counts for debugging
        print(f"Loaded {len(all_ideas)} ideas from file")
        sample_ideas = all_ideas[:5]  # Show first 5 ideas
        for i, idea in enumerate(sample_ideas):
            print(f"  Sample idea {i+1} ({idea.get('id', 'no-id')}): manual_match_count={idea.get('manual_match_count', 0)}, manual_elo={idea.get('ratings', {}).get('manual', 1500)}")

        if len(all_ideas) < 2:
            raise HTTPException(status_code=400, detail="Not enough ideas for comparison")

        # Select an efficient pair for manual rating
        idea_a, idea_b = select_efficient_pair(all_ideas, rating_type='manual', max_elo_diff=elo_range)

        if idea_a is None or idea_b is None:
            raise HTTPException(status_code=500, detail="Failed to select suitable pair")

        print(f"Selected pair: {idea_a.get('id')} (manual_match_count={idea_a.get('manual_match_count', 0)}) vs {idea_b.get('id')} (manual_match_count={idea_b.get('manual_match_count', 0)})")

        return {
            "idea_a": idea_a,
            "idea_b": idea_b,
            "status": "success"
        }

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
