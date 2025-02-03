from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse

import random
import uvicorn
import json
import uuid
import os
from pathlib import Path

from idea.models import FlattenedIdea, RatingResult

app = FastAPI()

# Mount static files correctly
app.mount("/static", StaticFiles(directory="idea/static"), name="static")

templates = Jinja2Templates(directory="idea/static/html")
# list of templates

# ---------------------------------------------------------
# 1. Load data from a JSON file containing your list-of-list
#    structure: each element is a "generation," which is a
#    list of Idea objects. For example:
#
#    [
#      [
#         {"title": "...", "proposal": "..."},
#         {"title": "...", "proposal": "..."}
#      ],
#      [
#         {"title": "...", "proposal": "..."},
#         ...
#      ]
#    ]
#
#    Adjust FILE_PATH to your actual file.
# ---------------------------------------------------------
FILE_PATH = "data/gemini_flash_exp1.json"

# Global store for flattened ideas keyed by idea_id
IDEAS_DB = {}

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def load_ideas_from_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Flatten the structure: each generation is data[generation_index]
    # which is a list of Idea objects
    # We'll generate a unique ID for each idea
    flattened = {}
    for g_idx, generation in enumerate(data):
        for i_idx, idea in enumerate(generation):
            # Create a unique ID for the idea
            idea_id = str(uuid.uuid4())
            # Start with default ELO of 1200
            flattened_idea = FlattenedIdea(
                id=idea_id,
                generation_index=g_idx,
                idea_index=i_idx,
                title=idea["title"],
                proposal=idea["proposal"],
                elo=1200.0
            )
            flattened[idea_id] = flattened_idea
    return flattened

# Initialize the global dictionary
try:
    IDEAS_DB = load_ideas_from_file(FILE_PATH)
except FileNotFoundError:
    print(f"Could not find {FILE_PATH}. Make sure the file exists.")
    IDEAS_DB = {}

# ---------------------------------------------------------
# 2. Utility functions
# ---------------------------------------------------------

def compute_elo(
    rating_a: float, rating_b: float,
    outcome: float,  # 1.0 if A wins, 0.0 if B wins, 0.5 if tie
    k_factor: float = 32.0
):
    """
    Compute the updated ELO rating for 'player A' after a match.
    'player B' rating can be derived similarly.

    - rating_a: ELO of A before match
    - rating_b: ELO of B before match
    - outcome:  1.0 if A wins, 0.5 if tie, 0.0 if A loses
    - k_factor: How quickly the ELO adjusts
    Returns new_rating_a, new_rating_b
    """
    # Expected score for A
    expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))
    # For B, outcome is reversed from A's perspective
    expected_b = 1.0 - expected_a

    # A's new rating
    new_rating_a = rating_a + k_factor * (outcome - expected_a)
    # B's new rating = symmetrical, if A's outcome is 1, B's is 0
    if outcome == 1.0:
        # B loses
        outcome_b = 0.0
    elif outcome == 0.0:
        # B wins
        outcome_b = 1.0
    else:
        # tie
        outcome_b = 0.5

    new_rating_b = rating_b + k_factor * (outcome_b - expected_b)
    return new_rating_a, new_rating_b

# ---------------------------------------------------------
# 3. API endpoints
# ---------------------------------------------------------
@app.get("/")
def serve_index(request: Request):
    """
    Serves the main HTML page with Bootstrap styling
    """
    return templates.TemplateResponse("rater.html", {"request": request})


@app.get("/random-pair")
def get_random_pair():
    """
    Returns two random distinct ideas from the database
    so a user can compare them side by side.
    """
    if len(IDEAS_DB) < 2:
        raise HTTPException(status_code=400, detail="Not enough ideas to form a pair.")
    idea_ids = list(IDEAS_DB.keys())
    random_pair = random.sample(idea_ids, 2)
    idea_a = IDEAS_DB[random_pair[0]]
    idea_b = IDEAS_DB[random_pair[1]]
    return {"ideaA": idea_a, "ideaB": idea_b}

@app.post("/rate")
def post_rating(result: RatingResult):
    """
    Endpoint that takes the IDs of two ideas plus an outcome
    indicating which idea "won" or if it was a tie.
    Updates the ELO rating for both ideas accordingly.
    """
    if result.idea_a_id not in IDEAS_DB or result.idea_b_id not in IDEAS_DB:
        raise HTTPException(status_code=404, detail="Idea ID not found in database.")

    idea_a = IDEAS_DB[result.idea_a_id]
    idea_b = IDEAS_DB[result.idea_b_id]

    if result.outcome not in ["A", "B", "tie"]:
        raise HTTPException(status_code=400, detail="Invalid outcome value. Must be 'A', 'B', or 'tie'.")

    # outcome from A's perspective
    if result.outcome == "A":
        # A wins, outcome = 1.0
        new_a, new_b = compute_elo(idea_a.elo, idea_b.elo, 1.0)
    elif result.outcome == "B":
        # A loses, outcome = 0.0
        new_a, new_b = compute_elo(idea_a.elo, idea_b.elo, 0.0)
    else:
        # tie
        new_a, new_b = compute_elo(idea_a.elo, idea_b.elo, 0.5)

    # Update the stored ratings
    idea_a.elo = new_a
    idea_b.elo = new_b

    # Persist if needed (in-memory for this example)
    IDEAS_DB[result.idea_a_id] = idea_a
    IDEAS_DB[result.idea_b_id] = idea_b

    return {"msg": "ELO ratings updated successfully.",
            "idea_a_new_elo": idea_a.elo,
            "idea_b_new_elo": idea_b.elo}

@app.get("/ranking")
def get_ranking():
    """
    Returns a list of all ideas sorted by ELO descending.
    """
    sorted_ideas = sorted(IDEAS_DB.values(), key=lambda x: x.elo, reverse=True)
    return sorted_ideas

@app.get("/mean-elo-per-generation")
def get_mean_elo():
    """
    For each generation index, compute the average ELO of all ideas in that generation.
    Returns a dict {generation_index: mean_elo}.
    """
    # We'll collect ELO scores by generation
    elo_by_gen = {}
    count_by_gen = {}
    for idea in IDEAS_DB.values():
        g_idx = idea.generation_index
        elo_by_gen[g_idx] = elo_by_gen.get(g_idx, 0.0) + idea.elo
        count_by_gen[g_idx] = count_by_gen.get(g_idx, 0) + 1

    mean_elo_by_gen = {}
    for g_idx, total_elo in elo_by_gen.items():
        mean_elo_by_gen[g_idx] = total_elo / count_by_gen[g_idx]

    return mean_elo_by_gen

@app.get("/api/list-evolution-files")
def list_evolution_files():
    """List all JSON files in the data directory"""
    files = [f.name for f in DATA_DIR.glob("*.json")]
    return JSONResponse(files)

@app.get("/api/load-evolution/{filename}")
def load_evolution_file(filename: str):
    """Load a specific evolution file"""
    try:
        file_path = DATA_DIR / filename
        if not file_path.exists():
            raise HTTPException(404, "File not found")
        with open(file_path) as f:
            return JSONResponse(json.load(f))
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/save-evolution")
def save_evolution(data: dict):
    """Save evolution data to file"""
    try:
        filename = data.get("filename")
        file_path = DATA_DIR / filename
        with open(file_path, "w") as f:
            json.dump(data["data"], f, indent=2)
        return JSONResponse({"status": "success"})
    except Exception as e:
        raise HTTPException(500, str(e))

# Run the server
if __name__ == "__main__":
    uvicorn.run("idea.rater:app", host="127.0.0.1", port=8001, reload=True)