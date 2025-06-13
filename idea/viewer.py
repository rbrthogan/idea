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
import numpy as np

from idea.evolution import EvolutionEngine
from idea.models import Idea
from idea.llm import Critic
from idea.config import LLM_MODELS, DEFAULT_MODEL
from idea.template_manager import router as template_router
from idea.prompts.loader import list_available_templates

# --- Initialize and configure FastAPI ---
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include template management routes
app.include_router(template_router)

# Mount static folder with custom config
app.mount("/static", StaticFiles(directory="idea/static"), name="static")

# Templates
templates = Jinja2Templates(directory="idea/static/html")

# Global engine instance
engine = None

# Flag indicating whether the API key is available
API_KEY_MISSING = os.getenv("GEMINI_API_KEY") in (None, "")

# Global queue for evolution updates
evolution_queue = Queue()
evolution_status = {
    "current_generation": 0,
    "total_generations": 0,
    "is_running": False,
    "history": []
}

# Store the latest evolution data for rating
latest_evolution_data = []

# Add this near other constants
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Add this class for the request body
class SaveEvolutionRequest(BaseModel):
    data: dict
    filename: str

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/")
def serve_viewer(request: Request):
    """Serves the viewer page"""
    return templates.TemplateResponse("viewer.html", {"request": request, "api_key_missing": API_KEY_MISSING})

@app.get("/rate")
def serve_rater(request: Request):
    """Serves the rater page"""
    return templates.TemplateResponse("rater.html", {"request": request, "api_key_missing": API_KEY_MISSING})

@app.get("/templates")
def serve_template_manager(request: Request):
    """Serves the template management page"""
    return templates.TemplateResponse("templates.html", {"request": request, "api_key_missing": API_KEY_MISSING})

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
async def start_evolution(request: Request):
    """
    Runs the complete evolution and returns the final results
    """
    global engine, evolution_status, evolution_queue, latest_evolution_data

    if API_KEY_MISSING:
        return JSONResponse(
            {"status": "error", "message": "GEMINI_API_KEY not configured"},
            status_code=400,
        )

    data = await request.json()
    print(f"Received request data: {data}")

    # Clear the latest evolution data when starting a new evolution
    latest_evolution_data = []

    pop_size = int(data.get('popSize', 3))
    generations = int(data.get('generations', 2))
    idea_type = data.get('ideaType', get_default_template_id())
    model_type = data.get('modelType', 'gemini-2.0-flash-lite')

    # Get temperature parameters with defaults
    try:
        ideator_temp = float(data.get('ideatorTemp', 2.0))
        critic_temp = float(data.get('criticTemp', 1.5))
        breeder_temp = float(data.get('breederTemp', 2.0))
        print(f"Parsed temperature values: ideator={ideator_temp}, critic={critic_temp}, breeder={breeder_temp}")
    except ValueError as e:
        print(f"Error parsing temperature values: {e}")
        # Use defaults if parsing fails
        ideator_temp = 1.0
        critic_temp = 0.7
        breeder_temp = 1.0

    # Get tournament parameters with defaults
    try:
        tournament_size = int(data.get('tournamentSize', 5))
        tournament_comparisons = int(data.get('tournamentComparisons', 20))
        print(f"Parsed tournament values: size={tournament_size}, comparisons={tournament_comparisons}")
    except ValueError as e:
        print(f"Error parsing tournament values: {e}")
        # Use defaults if parsing fails
        tournament_size = 5
        tournament_comparisons = 20

    # Get genotype breeding parameters with defaults
    use_genotype_breeding = data.get('useGenotypeBreeding', False)
    try:
        genotype_encoder_temp = float(data.get('genotypeEncoderTemp', 1.2))
        print(f"Parsed genotype values: use_genotype_breeding={use_genotype_breeding}, encoder_temp={genotype_encoder_temp}")
    except ValueError as e:
        print(f"Error parsing genotype values: {e}")
        # Use defaults if parsing fails
        genotype_encoder_temp = 1.2

    print(f"Starting evolution with pop_size={pop_size}, generations={generations}, "
          f"idea_type={idea_type}, model_type={model_type}, "
          f"temperatures: ideator={ideator_temp}, critic={critic_temp}, breeder={breeder_temp}, "
          f"tournament: size={tournament_size}, comparisons={tournament_comparisons}, "
          f"genotype_breeding={use_genotype_breeding}, genotype_encoder_temp={genotype_encoder_temp}")

    # Create and run evolution with specified parameters
    engine = EvolutionEngine(
        pop_size=pop_size,
        generations=generations,
        idea_type=idea_type,
        model_type=model_type,
        ideator_temp=ideator_temp,
        critic_temp=critic_temp,
        breeder_temp=breeder_temp,
        tournament_size=tournament_size,
        tournament_comparisons=tournament_comparisons,
        use_genotype_breeding=use_genotype_breeding,
        genotype_encoder_temp=genotype_encoder_temp
    )

    # Generate contexts for each idea
    contexts = engine.generate_contexts()

    # Clear the queue
    while not evolution_queue.empty():
        try:
            evolution_queue.get_nowait()
        except:
            break

    # Set up evolution status
    evolution_status = {
        "current_generation": 0,
        "total_generations": generations,
        "is_running": True,
        "history": [],
        "contexts": contexts,
        "progress": 0
    }

    # Put initial status in queue
    await evolution_queue.put(evolution_status.copy())

    # Start evolution in background task
    asyncio.create_task(run_evolution_task(engine))

    return JSONResponse({
        "status": "success",
        "message": "Evolution started",
        "contexts": contexts
    })

@app.post("/api/stop-evolution")
async def stop_evolution():
    """
    Request the evolution to stop gracefully
    """
    global engine

    if engine is None:
        return JSONResponse(
            {"status": "error", "message": "No evolution is currently running"},
            status_code=400,
        )

    # Request stop
    engine.stop_evolution()

    return JSONResponse({
        "status": "success",
        "message": "Stop request sent - evolution will halt at the next safe point"
    })

async def run_evolution_task(engine):
    """Run evolution in background with progress updates"""
    global evolution_status, evolution_queue, latest_evolution_data

    # Define the progress callback function
    async def progress_callback(update_data):
        global evolution_status, evolution_queue, latest_evolution_data

        # Convert Idea objects to dictionaries for JSON serialization
        if 'history' in update_data and isinstance(update_data['history'], list):
            update_data['history'] = [
                [idea_to_dict(idea) for idea in generation]
                for generation in update_data['history']
            ]

            # Store the latest evolution data for rating
            if update_data['history']:
                latest_evolution_data = update_data['history']

        # If evolution is complete, add token counts
        if update_data.get('is_running') is False and 'error' not in update_data:
            # Get token counts from the engine
            if hasattr(engine, 'get_total_token_count'):
                update_data['token_counts'] = engine.get_total_token_count()
                print(f"Evolution complete. Total tokens: {update_data['token_counts']['total']}")

        # Update the evolution status
        evolution_status = update_data

        # Clear the queue before adding new update to avoid backlog
        while not evolution_queue.empty():
            try:
                evolution_queue.get_nowait()
            except:
                break

        # Add the update to the queue
        await evolution_queue.put(update_data)

        # Log progress
        gen = update_data.get('current_generation', 0)
        progress = update_data.get('progress', 0)
        gen_label = "0 (Initial)" if gen == 0 else gen
        print(f"Progress update: Generation {gen_label}, Progress: {progress:.2f}%")

    # Run the evolution with progress updates
    await engine.run_evolution_with_updates(progress_callback)

def idea_to_dict(idea) -> dict:
    """Convert an Idea object or idea dictionary to a dictionary for JSON serialization"""
    # If idea is already a dictionary with 'id' and 'idea' keys
    if isinstance(idea, dict) and 'id' in idea and 'idea' in idea:
        idea_obj = idea['idea']
        idea_id = idea['id']

        # Get parent IDs if they exist
        parent_ids = idea.get('parent_ids', [])

        # If the idea object has title and content attributes
        if hasattr(idea_obj, 'title') and hasattr(idea_obj, 'content'):
            return {
                "id": str(idea_id),
                "title": idea_obj.title,
                "content": idea_obj.content,
                "parent_ids": parent_ids,
                "match_count": idea.get('match_count', 0),
                "auto_match_count": idea.get('auto_match_count', 0),
                "manual_match_count": idea.get('manual_match_count', 0)
            }
        # If the idea object is a string
        elif isinstance(idea_obj, str):
            return {
                "id": str(idea_id),
                "title": "Untitled",
                "content": idea_obj,
                "parent_ids": parent_ids,
                "match_count": idea.get('match_count', 0),
                "auto_match_count": idea.get('auto_match_count', 0),
                "manual_match_count": idea.get('manual_match_count', 0)
            }
        # If the idea object is already a dict
        elif isinstance(idea_obj, dict):
            result = idea_obj.copy()
            result["id"] = str(idea_id)
            result["parent_ids"] = parent_ids
            result["match_count"] = idea.get('match_count', 0)
            result["auto_match_count"] = idea.get('auto_match_count', 0)
            result["manual_match_count"] = idea.get('manual_match_count', 0)
            return result

    # Legacy case: idea is a direct Idea object
    elif hasattr(idea, 'title') and hasattr(idea, 'content'):
        return {
            "title": idea.title,
            "content": idea.content,
            "parent_ids": [],
            "match_count": getattr(idea, 'match_count', 0),
            "auto_match_count": getattr(idea, 'auto_match_count', 0),
            "manual_match_count": getattr(idea, 'manual_match_count', 0)
        }

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
def api_get_generations():
    """
    Returns a JSON array of arrays: each generation is an array of ideas.
    Each idea is {title, content}.
    """
    global latest_evolution_data

    # If engine is None, use the latest evolution data
    if engine is None:
        if latest_evolution_data:
            print(f"Returning latest evolution data with {len(latest_evolution_data)} generations")
            return JSONResponse(latest_evolution_data)
        else:
            print("No evolution data available")
            return JSONResponse([])  # Return empty array if no data is available

    # If engine is available, use its history
    result = []
    for generation in engine.history:
        gen_list = []
        for prop in generation:
            gen_list.append({
                "title": prop.title,
                "content": prop.content
            })
        result.append(gen_list)

    # Store the result as the latest evolution data
    latest_evolution_data = result

    return JSONResponse(result)

@app.get("/api/generations/{gen_id}")
def api_get_generation(gen_id: int):
    """
    Returns ideas for a specific generation.
    """
    ideas = engine.get_ideas_by_generation(gen_id)
    if not ideas:
        return JSONResponse({"error": "Invalid generation index."}, status_code=404)
    output = [{"title": i.title, "content": i.content} for i in ideas]
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
async def get_progress():
    """Returns the current progress of the evolution"""
    global evolution_status

    # If there's a new update in the queue, get it
    try:
        # Get the latest update from the queue without waiting
        if not evolution_queue.empty():
            evolution_status = await evolution_queue.get()
    except Exception as e:
        print(f"Error getting queue updates: {e}")

    return JSONResponse(evolution_status)

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

@app.get('/api/evolution/{evolution_id}')
async def get_evolution(request: Request, evolution_id: str):
    """Get evolution data from file"""
    try:
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

        # Save the updated data
        with open(file_path, 'w') as f:
            json.dump(evolution_data, f, indent=2)

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

        # Create a critic agent from llm.py with the specified model
        critic = Critic(model_name=model_id)

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

        # Track failed attempts to find suitable pairs
        failed_attempts = 0
        max_failed_attempts = 50  # Maximum number of attempts before relaxing constraints

        for i in range(num_comparisons):
            print(f"Comparison {i+1}/{num_comparisons}")

            # Try to find a pair of ideas with similar ELO ratings
            found_suitable_pair = False
            current_max_elo_diff = max_elo_diff

            # Sort ideas by match count (ascending) to prioritize less-rated ideas
            sorted_ideas = sorted(all_ideas, key=lambda x: x.get('auto_match_count', 0))

            # Take the bottom 50% of ideas (those with fewer matches)
            candidate_pool_size = max(2, len(sorted_ideas) // 2)
            candidate_pool = sorted_ideas[:candidate_pool_size]

            print(f"Selected candidate pool of {len(candidate_pool)} ideas with fewest matches")

            while not found_suitable_pair:
                # If we've tried too many times, gradually relax the ELO difference constraint
                if failed_attempts >= max_failed_attempts:
                    current_max_elo_diff = current_max_elo_diff * 1.5  # Increase by 50%
                    print(f"Relaxing ELO difference constraint to ±{current_max_elo_diff} after {failed_attempts} failed attempts")

                    # Also expand the candidate pool if we're still struggling
                    candidate_pool_size = min(len(sorted_ideas), candidate_pool_size + len(sorted_ideas) // 4)
                    candidate_pool = sorted_ideas[:candidate_pool_size]
                    print(f"Expanded candidate pool to {len(candidate_pool)} ideas")

                    failed_attempts = 0  # Reset counter after relaxing

                # Try to select from the candidate pool first
                if len(candidate_pool) >= 2:
                    try:
                        # Select first idea from candidate pool (with fewest matches)
                        idea_a = candidate_pool[0]

                        # Find ideas within ELO range
                        elo_a = idea_a['ratings']['auto']
                        compatible_ideas = [
                            idea for idea in all_ideas
                            if idea['id'] != idea_a['id'] and
                            abs(idea['ratings']['auto'] - elo_a) <= current_max_elo_diff
                        ]

                        if compatible_ideas:
                            # Select a random compatible idea
                            idea_b = random.choice(compatible_ideas)
                            # Define elo_b here to ensure it's always set
                            elo_b = idea_b['ratings']['auto']
                            found_suitable_pair = True
                        else:
                            # If no compatible ideas, remove idea_a from candidate pool and try again
                            candidate_pool.remove(idea_a)
                            failed_attempts += 1
                    except Exception as e:
                        print(f"Error selecting from candidate pool: {e}")
                        # Fallback to random selection
                        idea_a, idea_b = random.sample(all_ideas, 2)
                        elo_a = idea_a['ratings']['auto']
                        elo_b = idea_b['ratings']['auto']
                        failed_attempts += 1
                else:
                    try:
                        # Fallback to random selection if candidate pool is depleted
                        idea_a, idea_b = random.sample(all_ideas, 2)

                        # Get their auto ELO ratings
                        elo_a = idea_a['ratings']['auto']
                        elo_b = idea_b['ratings']['auto']

                        # Check if they're within the allowed ELO difference
                        if abs(elo_a - elo_b) <= current_max_elo_diff:
                            found_suitable_pair = True
                            print(f"Found suitable pair with ELO difference: {abs(elo_a - elo_b)}")
                        else:
                            failed_attempts += 1
                    except Exception as e:
                        print(f"Error in random selection: {e}")
                        # Just pick any two ideas as a last resort
                        idea_a, idea_b = random.sample(all_ideas, 2)
                        elo_a = idea_a['ratings']['auto']
                        elo_b = idea_b['ratings']['auto']
                        failed_attempts += 1

                # If we've tried too many times, just use any random pair
                if failed_attempts >= max_failed_attempts * 2:
                    try:
                        idea_a, idea_b = random.sample(all_ideas, 2)
                        elo_a = idea_a['ratings']['auto']
                        elo_b = idea_b['ratings']['auto']
                        print(f"Using random pair with ELO difference {abs(elo_a - elo_b)} after {failed_attempts} failed attempts")
                        found_suitable_pair = True
                    except Exception as e:
                        print(f"Error in last resort selection: {e}")
                        # Absolute last resort - just pick the first two ideas
                        idea_a = all_ideas[0]
                        idea_b = all_ideas[1] if len(all_ideas) > 1 else all_ideas[0]
                        elo_a = idea_a['ratings']['auto']
                        elo_b = idea_b['ratings']['auto']
                        found_suitable_pair = True

            # Ensure elo_a and elo_b are defined
            if 'elo_a' not in locals():
                # Ensure idea_a has a proper ratings structure
                if 'ratings' not in idea_a:
                    idea_a['ratings'] = {'auto': 1500, 'manual': 1500}
                elif 'auto' not in idea_a['ratings']:
                    idea_a['ratings']['auto'] = 1500
                elo_a = idea_a['ratings']['auto']

            if 'elo_b' not in locals():
                # Ensure idea_b has a proper ratings structure
                if 'ratings' not in idea_b:
                    idea_b['ratings'] = {'auto': 1500, 'manual': 1500}
                elif 'auto' not in idea_b['ratings']:
                    idea_b['ratings']['auto'] = 1500
                elo_b = idea_b['ratings']['auto']

            # Log match counts for transparency
            match_count_a = idea_a.get('auto_match_count', 0)
            match_count_b = idea_b.get('auto_match_count', 0)
            print(f"Comparing idea {idea_a.get('id')} (ELO: {elo_a}, Matches: {match_count_a}) vs {idea_b.get('id')} (ELO: {elo_b}, Matches: {match_count_b})")

            # Use the critic to determine the winner - pass the idea_type parameter
            winner = critic.compare_ideas(idea_a, idea_b, idea_type)
            print(f"Winner: {winner}")

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

# Run the server
if __name__ == "__main__":
    uvicorn.run(
        "idea.viewer:app", host="127.0.0.1", port=8000, reload=True
    )
