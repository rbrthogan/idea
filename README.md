# Idea Evolution

Idea Evolution is a small web application that iteratively evolves ideas using
LLM agents.  The system can generate new proposals, critique them and breed new
generations while also allowing humans to rank the results.

## Features

- Generate and evolve ideas using Gemini models
- Rate ideas using an ELO‑style tournament
- View the full evolution history and ratings
- YAML prompt templates that can be edited in the app
- Save and load previous evolutions

## Outline of Evolution

1. Generate some initial context to help with idea generation
2. Using this context, generate a set of initial ideas
3. Critique + Refine each idea
4. Group ideas into batches and run a tournament with pairwise comparisons to find the best ideas
5. Breed the best ideas to create a new generation (with best ideas preferentially breeding)
6. Go to 3

## Installation

1. Create a Python virtual environment.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Set the `GEMINI_API_KEY` environment variable with your Google Generative AI key.

## Running the App

Start the server:
```bash
uvicorn idea.viewer:app
```

Then visit:
- Evolution viewer/creator: <http://localhost:8000/>
- Idea rater: <http://localhost:8000/rate>
- Template manager: <http://localhost:8000/templates>

## Running Tests

Run the unit tests with:
```bash
pytest
```

## Architecture

- `viewer.py` – FastAPI application serving the viewer, rater and template manager
- `evolution.py` – Core evolution engine
- `llm.py` – Gemini LLM wrapper and agent definitions
- `prompts/` – Prompt templates (YAML and Python)
- `static/` – Frontend assets
  - `html/` – HTML templates
  - `js/` – JavaScript files
  - `css/` – Stylesheets

## Available Templates

Example YAML templates live in `idea/prompts/templates`:

- `airesearch.yaml` – AI research proposals
- `game_design.yaml` – Browser game designs
- `drabble.yaml` – 100‑word stories

New templates can be created from the Template Manager UI.

# TODO

See [TODO.md](TODO.md) for a list of planned features, research ideas, and bug fixes.
