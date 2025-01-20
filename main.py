# main.py
import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse

from idea.evolution import EvolutionEngine
from idea.llm import LLMWrapper

# --- Initialize and configure FastAPI ---
app = FastAPI()

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Instantiate engine
llm = LLMWrapper()
engine = EvolutionEngine(llm=llm, pop_size=20, generations=10)
engine.run_evolution()


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/")
def serve_index(request: Request):
    """
    Serves the main HTML page (via Jinja2). This page will load
    Bootstrap and our custom script, which fetches generations.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/generations")
def api_get_generations():
    """
    Returns a JSON array of arrays: each generation is an array of proposals.
    Each proposal is {title, proposal}.
    """
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


# Run the server
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)