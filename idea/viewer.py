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
    data = await request.json()
    try:
        # Process the rating and calculate new ELOs
        k_factor = 32
        idea_a_id = data['idea_a_id']
        idea_b_id = data['idea_b_id']
        outcome = data['outcome']

        # Calculate ELO updates
        # ... (implement ELO calculation logic here)

        # Return the updated ELOs
        return JSONResponse({
            "status": "success",
            "updated_elos": {
                idea_a_id: new_elo_a,
                idea_b_id: new_elo_b
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.route('/api/save-evolution', methods=['POST'])
def save_evolution():
    try:
        data = request.json

        # Get filename from request or generate default
        filename = data.get('filename', '')
        if not filename:
            timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            filename = f"evolution-results-{timestamp}.json"

        # Ensure filename has .json extension
        if not filename.endswith('.json'):
            filename += '.json'

        # Ensure the data directory exists
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        os.makedirs(data_dir, exist_ok=True)

        # Full path to save file
        file_path = os.path.join(data_dir, filename)

        # Save the data
        with open(file_path, 'w') as f:
            json.dump(data['data'], f, indent=2)

        return jsonify({
            'success': True,
            'message': f'File saved as {filename}',
            'path': file_path
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Run the server
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)