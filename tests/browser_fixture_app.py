"""Offline UI preview: python tests/browser_fixture_app.py --port 8124.

The server shuts down after eight minutes and never imports application services
or makes provider/database calls. Its data is deliberately small and deterministic.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
import uvicorn
from idea.config import LLM_MODELS, DEFAULT_MODEL, THINKING_MODEL_CONFIG, estimate_generation_token_cost
from idea.billing import build_billing_response, compact_usage, record_price_usage

app = FastAPI()
app.mount("/static", StaticFiles(directory=ROOT / "idea/static"), name="static")
templates = Environment(loader=FileSystemLoader(ROOT / "idea/static/html"), autoescape=select_autoescape())
suite = {
    "components": [{"id": "clarity", "kind": "computed"}],
    "fitness": {"metrics": [{"component_id": "clarity", "metric_name": "testability", "weight": 1, "direction": "maximize", "normalization": "min_max"}]},
}


def candidate(identifier, title, content, score, parent_ids=None, generation=0):
    return {
        "id": identifier, "title": title, "content": content,
        "generation": generation, "birth_generation": generation,
        "parent_ids": parent_ids or [], "island_id": generation % 2,
        "elo": 1500 + int(score * 100), "ratings": {"auto": 1500, "manual": 1500},
        "selection_metrics": {"fitness": score, "elo": 1500 + int(score * 100), "diversity": 0.4 + score / 3},
        "evaluations": {"clarity": {"status": "ok", "metrics": {"testability": score}, "normalized_metrics": {"testability": score}, "duration_ms": 3, "schema_version": 1}},
    }


history = [[
    candidate("seed-a", "A garden of prime numbers", "## The question\nCan modular planting patterns make prime gaps tangible?\n\nBuild a deterministic grid and compare it against a shuffled control.\n\n### Evidence to collect\n- Gap distributions\n- Replicable seeds\n- A visible null model", 0.62),
    candidate("seed-b", "Listen to the gaps", "Map the distance between consecutive primes to musical intervals. Compare the resulting rhythm with a random sequence.", 0.49),
    candidate("seed-c", "An atlas of remainders", "Plot prime positions across modular maps. Record the mapping, sample size, and control distribution.", 0.74),
], [
    candidate("child-a", "The prime garden, with a control", "## A falsifiable visual experiment\nArrange prime positions in a modular garden, then compare the spacing with a seeded random permutation.\n\n### Method\n1. Fix the domain and random seed.\n2. Render both maps using the same scale.\n3. Report the difference in nearest-neighbor distances.\n\nA striking image is the beginning of the investigation; a repeated measurement is the evidence.", 0.87, ["seed-a", "seed-c"], 1),
    candidate("seed-c", "An atlas of remainders", "Plot prime positions across modular maps. Record the mapping, sample size, and control distribution.", 0.74),
    candidate("child-b", "Hear the shuffled universe", "Listen to prime gaps beside a shuffled baseline. Ask participants to identify the original sequence before revealing which is which.", 0.67, ["seed-b"], 1),
]]
run = {
    "evolution_id": "fixture-run", "name": "Prime patterns · first exploration", "evolution_name": "Prime patterns · first exploration",
    "status": "complete", "is_running": False, "current_generation": 1, "total_generations": 1,
    "history": history, "history_version": 1,
    "config": {"idea_type": "prime_visualization", "model_type": "gemini-3.6-flash", "pop_size": 3, "generations": 1, "fitness_alpha": .7, "evaluation": suite},
    "diversity_history": [], "tournament_history": [], "created_at": "2026-10-13T12:00:00Z",
}
catalog = {
    "game_design": {"name": "Game design", "description": "Create distinctive games with satisfying play.", "author": "IDEA", "is_system": True},
    "airesearch": {"name": "Research questions", "description": "Generate tractable research proposals with clear hypotheses and modest compute requirements.", "author": "IDEA", "is_system": True},
    "drabble": {"name": "Small stories", "description": "Explore voice, character, and unexpected turns in exactly one hundred words.", "author": "IDEA", "is_system": True},
}


def fixture_costs(model):
    cost = estimate_generation_token_cost(model, input_tokens=10000,
        cached_input_tokens=5000, output_tokens=1000, thought_tokens=500,
        priced_at="2026-10-13")
    usage = {"model": model, "input": 10000, "cached_input": 5000,
        "output": 1000, "thoughts": 500, "total": 11500,
        "requests": 1, "cost": cost["total_cost"]}
    bins = {}
    record_price_usage(bins, cost, usage)
    usage["usage_by_price"] = list(bins.values())
    usage["cache_savings"] = sum(row["cache_savings"] for row in bins.values())
    return {"cost": {"total_cost": cost["total_cost"]}, "ideator": usage,
            "total_input": 10000, "total_cached_input": 5000,
            "total_output": 1000, "total_thoughts": 500}


billing_runs = [
    {**run, "_billing_summary": compact_usage(fixture_costs("gemini-3.8-flash")), "updated_at": run["created_at"]},
    {"evolution_id": "fixture-stories", "name": "Story seeds", "status": "complete", "updated_at": run["created_at"],
     "_billing_summary": compact_usage(fixture_costs("gpt-6-luna"))},
]


@app.get("/")
@app.get("/rate")
def page(request: Request):
    name = "rater.html" if request.url.path == "/rate" else "viewer.html"
    return HTMLResponse(templates.get_template(name).render(
        request=request, api_key_missing=False,
        dev_auth_config={"enabled": True, "token": "offline-fixture", "uid": "fixture-user", "email": "researcher@example.test", "display_name": "Local researcher", "is_admin": False},
        default_model=DEFAULT_MODEL,
        llm_models=LLM_MODELS,
        thinking_config=THINKING_MODEL_CONFIG,
    ))


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def api(path: str, request: Request):
    path = path.rstrip("/")
    if path == "template-types":
        return {"status": "success", "templates": [{"id": key, **value} for key, value in catalog.items()]}
    if path == "templates":
        return {"status": "success", "templates": catalog}
    if path.startswith("templates/"):
        return {"status": "success", "template": catalog.get(path.split("/")[-1], {})}
    if path == "models":
        return {"models": LLM_MODELS, "default": DEFAULT_MODEL}
    if path == "billing":
        return build_billing_response(billing_runs, [])
    if path == "settings/status":
        return {"has_api_key": True, "api_key_set": True, "api_keys": {"google_generative_ai": {"configured": True, "masked_key": "***", "source": "environment"}, "openai": {"configured": True, "masked_key": "***", "source": "stored"}}, "key_sources": {"google_generative_ai": "environment", "openai": "stored"}, "provider_keys": {"google_generative_ai": {"has_api_key": True, "api_key_hint": "fixture"}, "openai": {"has_api_key": True}}, "execution_preference": "local", "generated_code_execution_enabled": False, "execution_options": {"local": {"available": False}, "remote": {"available": False}}}
    if path == "user/status":
        return {"authenticated": True, "has_api_key": True, "user": {"uid": "fixture-user", "is_admin": False}}
    if path == "history":
        return {"status": "success", "items": [{"id": "fixture-run", "display_name": run["name"], "type": "evolution", "timestamp": run["created_at"], "status": "complete", "generations": 1, "total_generations": 1, "idea_type": "prime_visualization", "can_continue": False, "can_rate": True, "can_use_config": False, "execution_archived": True, "config": run["config"]}]}
    if path == "evolutions":
        return [{"id": "fixture-run", "filename": run["name"], "timestamp": run["created_at"]}]
    if path == "evolution/fixture-run":
        return {"id": "fixture-run", "name": run["name"], "data": run}
    if path == "generations":
        return history
    if path == "checkpoints":
        return {"status": "success", "checkpoints": []}
    if path == "progress":
        return {"is_running": False, "history": [], "history_version": 0, "current_generation": 0, "total_generations": 0}
    if path == "start-evolution":
        from fastapi.responses import JSONResponse
        return JSONResponse({"message": "Offline preview: generation is disabled."}, status_code=400)
    if path == "get-efficient-pair":
        return {"idea_a": history[-1][0], "idea_b": history[-1][2], "pair_token": "offline-fixture-pair-token", "swiss_status": {"round": 1, "completed_matches": 0, "total_matches": 3}}
    return {"status": "success"}


async def main(port, seconds):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    async def stop():
        await asyncio.sleep(seconds)
        server.should_exit = True
    timer = asyncio.create_task(stop())
    try:
        await server.serve()
    finally:
        timer.cancel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8124)
    parser.add_argument("--seconds", type=int, default=480)
    args = parser.parse_args()
    try:
        asyncio.run(main(args.port, min(480, args.seconds)))
    except KeyboardInterrupt:
        pass
