# main.py
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
from flask import jsonify
from datetime import datetime
import numpy as np

from idea.evolution import EvolutionEngine
from idea.models import Idea
from idea.llm import Critic
from idea.config import LLM_MODELS, DEFAULT_MODEL

# --- Initialize and configure FastAPI ---
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder with custom config
app.mount("/static", StaticFiles(directory="idea/static"), name="static")

# Templates
templates = Jinja2Templates(directory="idea/static/html")

# Global engine instance
engine = None

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
    return templates.TemplateResponse("viewer.html", {"request": request})

@app.get("/rate")
def serve_rater(request: Request):
    """Serves the rater page"""
    return templates.TemplateResponse("rater.html", {"request": request})

@app.post("/api/start-evolution")
async def start_evolution(request: Request):
    """
    Runs the complete evolution and returns the final results
    """
    global engine, evolution_status, evolution_queue, latest_evolution_data
    data = await request.json()
    print(f"Received request data: {data}")

    # Clear the latest evolution data when starting a new evolution
    latest_evolution_data = []

    pop_size = int(data.get('popSize', 3))
    generations = int(data.get('generations', 2))
    idea_type = data.get('ideaType', 'airesearch')
    model_type = data.get('modelType', 'gemini-1.5-flash')
    # Use a fixed context type
    context_type = "context_prompt"

    # Get temperature parameters with defaults
    try:
        ideator_temp = float(data.get('ideatorTemp', 1.0))
        critic_temp = float(data.get('criticTemp', 0.7))
        breeder_temp = float(data.get('breederTemp', 1.0))
        print(f"Parsed temperature values: ideator={ideator_temp}, critic={critic_temp}, breeder={breeder_temp}")
    except ValueError as e:
        print(f"Error parsing temperature values: {e}")
        # Use defaults if parsing fails
        ideator_temp = 1.0
        critic_temp = 0.7
        breeder_temp = 1.0

    print(f"Starting evolution with pop_size={pop_size}, generations={generations}, "
          f"idea_type={idea_type}, model_type={model_type} "
          f"temperatures: ideator={ideator_temp}, critic={critic_temp}, breeder={breeder_temp}")

    # Create and run evolution with specified parameters
    engine = EvolutionEngine(
        pop_size=pop_size,
        generations=generations,
        idea_type=idea_type,
        model_type=model_type,
        ideator_temp=ideator_temp,
        critic_temp=critic_temp,
        breeder_temp=breeder_temp
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

def idea_to_dict(idea: Idea) -> dict:
    """Convert an Idea object to a dictionary"""
    return {
        "title": idea.title,
        "proposal": idea.proposal
    }


@app.get("/api/generations")
def api_get_generations():
    """
    Returns a JSON array of arrays: each generation is an array of proposals.
    Each proposal is {title, proposal}.
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
                "proposal": prop.proposal
            })
        result.append(gen_list)

    # Store the result as the latest evolution data
    latest_evolution_data = result

    return JSONResponse(result)

@app.get("/api/generations/{gen_id}")
def api_get_generation(gen_id: int):
    """
    Returns proposals for a specific generation.
    """
    proposals = engine.get_proposals_by_generation(gen_id)
    if not proposals:
        return JSONResponse({"error": "Invalid generation index."}, status_code=404)
    output = [{"title": p.title, "proposal": p.proposal} for p in proposals]
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

        # Get idea_type from the request or use a default
        idea_type = data.get('ideaType', 'airesearch')

        print(f"Starting auto-rating for evolution {evolution_id} with {num_comparisons} comparisons using model {model_id}")

        # Load the evolution data
        file_path = DATA_DIR / f"{evolution_id}.json"
        if not file_path.exists():
            print(f"Evolution file not found: {file_path}")
            raise HTTPException(status_code=404, detail="Evolution not found")

        with open(file_path) as f:
            evolution_data = json.load(f)

        # Try to extract idea_type from the evolution data if not provided in request
        if 'idea_type' in evolution_data and not idea_type:
            idea_type = evolution_data.get('idea_type', 'airesearch')

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
        for i in range(num_comparisons):
            print(f"Comparison {i+1}/{num_comparisons}")

            # Randomly select two different ideas
            idea_a, idea_b = random.sample(all_ideas, 2)

            print(f"Comparing idea {idea_a.get('id')} vs {idea_b.get('id')}")

            # Use the critic to determine the winner - pass the idea_type parameter
            winner = critic.compare_ideas(idea_a, idea_b, idea_type)
            print(f"Winner: {winner}")

            # Skip this comparison if there was an error (winner is None)
            if winner is None:
                print("Skipping this comparison due to an error")
                continue

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

            if idea_b['id'] in idea_map:
                gen_idx, idea_idx = idea_map[idea_b['id']]
                evolution_data['history'][gen_idx][idea_idx]['ratings'] = idea_b['ratings']
                evolution_data['history'][gen_idx][idea_idx]['elo'] = idea_b['elo']

            # Record the result
            results.append({
                'idea_a': idea_a.get('id', 'unknown'),
                'idea_b': idea_b.get('id', 'unknown'),
                'outcome': winner,
                'new_elo_a': idea_a['ratings']['auto'],
                'new_elo_b': idea_b['ratings']['auto']
            })

            # Save after every 5 comparisons to ensure progress is not lost
            if (i + 1) % 5 == 0 and not skip_save:
                with open(file_path, 'w') as f:
                    json.dump(evolution_data, f, indent=2)
                print(f"Saved progress after {i + 1} comparisons")

        print(f"Completed {len(results)} comparisons")

        # Save the final updated Elo scores back to the file (unless skipSave is True)
        if not skip_save:
            with open(file_path, 'w') as f:
                json.dump(evolution_data, f, indent=2)

        # Return the results with sorted ideas including generation info
        return JSONResponse({
            'status': 'success',
            'results': results,
            'ideas': sorted(all_ideas, key=lambda x: x['ratings']['auto'], reverse=True)
        })

    except Exception as e:
        import traceback
        print(f"Error in auto-rate: {e}")
        print(traceback.format_exc())
        return JSONResponse({
            'status': 'error',
            'message': str(e)
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

                if rating_type == 'all' or rating_type == 'manual':
                    idea['ratings']['manual'] = 1500

        # Save the updated data
        with open(file_path, 'w') as f:
            json.dump(evolution_data, f, indent=2)

        return JSONResponse({
            'status': 'success',
            'message': f'{rating_type.capitalize()} ratings reset successfully'
        })

    except Exception as e:
        print(f"Error resetting ratings: {e}")
        return JSONResponse({
            'status': 'error',
            'message': str(e)
        }, status_code=500)

# Run the server
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)