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
    global engine
    data = await request.json()
    pop_size = int(data.get('popSize', 3))
    generations = int(data.get('generations', 2))
    idea_type = data.get('ideaType', 'airesearch')
    model_type = data.get('modelType', 'gemini-1.5-flash')
    context_type = data.get('contextType', 'random_words')

    print(f"Starting evolution with pop_size={pop_size}, generations={generations}, "
          f"idea_type={idea_type}, model_type={model_type}, context_type={context_type}")

    # Create and run evolution with specified parameters
    engine = EvolutionEngine(
        pop_size=pop_size,
        generations=generations,
        idea_type=idea_type,
        model_type=model_type,
        context_type=context_type
    )

    # Generate contexts for each idea
    contexts = []
    for _ in range(pop_size):
        context = engine.ideator.generate_context(context_type,
            engine.ideator.idea_field_map.get(idea_type))
        contexts.append(context)

    engine.run_evolution()

    # Convert results to JSON-serializable format
    history = [[idea_to_dict(idea) for idea in generation]
              for generation in engine.history]

    return JSONResponse({
        "status": "success",
        "history": history,
        "contexts": contexts
    })

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
    if engine is None:
        return JSONResponse([])  # Return empty array if evolution hasn't started

    result = []
    for generation in engine.history:
        gen_list = []
        for prop in generation:
            gen_list.append({
                "title": prop.title,
                "proposal": prop.proposal
            })
        result.append(gen_list)
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
        while not evolution_queue.empty():
            evolution_status = await evolution_queue.get()
    except Exception as e:
        print(f"Error getting queue updates: {e}")

    return JSONResponse(evolution_status)

@app.post("/api/save-evolution")
async def save_evolution(request: SaveEvolutionRequest):
    """Save evolution data to file"""
    try:
        file_path = DATA_DIR / request.filename
        with open(file_path, "w") as f:
            json.dump(request.data, f, indent=2)
        return JSONResponse({"status": "success"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

        print(f"Starting auto-rating for evolution {evolution_id} with {num_comparisons} comparisons using model {model_id}")

        # Validate model ID
        valid_model_ids = [model["id"] for model in LLM_MODELS]
        if model_id not in valid_model_ids:
            return JSONResponse({
                'status': 'error',
                'message': f'Invalid model ID: {model_id}'
            }, status_code=400)

        # Load the evolution data
        file_path = DATA_DIR / f"{evolution_id}.json"
        if not file_path.exists():
            print(f"Evolution file not found: {file_path}")
            raise HTTPException(status_code=404, detail="Evolution not found")

        with open(file_path) as f:
            evolution_data = json.load(f)

        # Extract all ideas from all generations with generation info
        all_ideas = []
        for gen_index, generation in enumerate(evolution_data.get('history', [])):
            for idea in generation:
                # Add an ID if not present
                if 'id' not in idea:
                    idea['id'] = f"idea_{len(all_ideas)}"

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

            # Use the critic to determine the winner
            winner = critic.compare_ideas(idea_a, idea_b)
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

            # Record the result
            results.append({
                'idea_a': idea_a.get('id', 'unknown'),
                'idea_b': idea_b.get('id', 'unknown'),
                'outcome': winner,
                'new_elo_a': idea_a['ratings']['auto'],
                'new_elo_b': idea_b['ratings']['auto']
            })

        print(f"Completed {len(results)} comparisons")

        # Save the updated Elo scores back to the file (unless skipSave is True)
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