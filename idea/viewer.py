# main.py
import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import os
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from asyncio import Queue
import random

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

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/")
def serve_index(request: Request):
    """
    Serves the main HTML page (via Jinja2). This page will load
    Bootstrap and our custom script, which allows configuration.
    """
    return templates.TemplateResponse("viewer.html", {"request": request})

@app.post("/api/start-evolution")
async def start_evolution(request: Request):
    """
    Runs the complete evolution and returns the final results
    """
    global engine
    data = await request.json()
    pop_size = int(data.get('popSize', 3))
    generations = int(data.get('generations', 2))
    idea_type = data.get('ideaType', 'airesearch')  # Default to airesearch if not specified

    print(f"Starting evolution with pop_size={pop_size}, generations={generations}, idea_type={idea_type}")

    # Create and run evolution with specified idea type
    engine = EvolutionEngine(
        pop_size=pop_size,
        generations=generations,
        idea_type=idea_type
    )
    engine.run_evolution()

    # Convert results to JSON-serializable format
    history = [[idea_to_dict(idea) for idea in generation]
              for generation in engine.history]

    return JSONResponse({
        "status": "success",
        "history": history
    })

def idea_to_dict(idea: Idea) -> dict:
    """Convert an Idea object to a dictionary"""
    return {
        "title": idea.title,
        "proposal": idea.proposal
    }

# async def run_evolution_background(engine):
#     """
#     Runs evolution in background and updates status via queue
#     """
#     global evolution_status
#     try:
#         print("Starting evolution background task...")

#         # Seed initial population
#         print("Seeding initial population...")
#         engine.population = engine.ideator.seed_ideas(engine.pop_size, engine.context_type, engine.idea_type)
#         engine.population = [engine.formatter.format_idea(idea) for idea in engine.population]
#         engine.history.append(engine.population)

#         # Convert Idea objects to dictionaries for JSON serialization
#         serializable_history = [[idea_to_dict(idea) for idea in generation]
#                               for generation in engine.history]

#         evolution_status["current_generation"] = 1
#         evolution_status["history"] = serializable_history
#         await evolution_queue.put(evolution_status.copy())
#         print(f"Generation 1 complete. Population size: {len(engine.population)}")

#         # Run evolution generations
#         for gen in range(engine.generations):
#             print(f"Starting generation {gen + 2}...")
#             chunk_size = 5
#             new_population = []
#             random.shuffle(engine.population)

#             for i in range(0, len(engine.population), chunk_size):
#                 group = engine.population[i : i + chunk_size]
#                 group = engine.critic.remove_worst_idea(group)
#                 new_idea = engine.ideator.generate_new_idea(group)
#                 group.append(new_idea)
#                 new_population.extend(group)

#             engine.population = new_population
#             engine.population = [engine.critic.refine(idea) for idea in engine.population]
#             engine.population = [engine.formatter.format_idea(idea) for idea in engine.population]
#             engine.history.append(engine.population)

#             # Convert Idea objects to dictionaries for JSON serialization
#             serializable_history = [[idea_to_dict(idea) for idea in generation]
#                                   for generation in engine.history]

#             # Update status and notify through queue
#             evolution_status["current_generation"] = gen + 2
#             evolution_status["history"] = serializable_history
#             await evolution_queue.put(evolution_status.copy())
#             print(f"Generation {gen + 2} complete. Population size: {len(engine.population)}")

#             # Small delay to prevent blocking
#             await asyncio.sleep(0.1)

#         print("Evolution complete!")
#         evolution_status["is_running"] = False
#         await evolution_queue.put(evolution_status.copy())

#     except Exception as e:
#         print(f"Error in evolution background task: {e}")
#         evolution_status["is_running"] = False
#         evolution_status["error"] = str(e)
#         await evolution_queue.put(evolution_status.copy())
#         raise

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

# Run the server
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)