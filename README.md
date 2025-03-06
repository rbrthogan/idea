# Idea Evolution

An application for evolving research ideas using LLMs and human feedback.

## Features

- Generate and evolve research ideas using LLM
- Rate ideas using ELO system
- View evolution history and ratings
- Save and load previous evolutions

## Outline of Evolution

1. Generate some initial context to help with idea generation
2. Using this context, generate a set of initial ideas
3. Critique + Refine each idea
4. Group ideas into batches and have a tournament with pairwise comparisons to find the best ideas
5. Breed the best ideas to create a new generation (with best ideas preferentially breeding)
6. Go to 3

## Running the App

Start the server:
```bash
uvicorn idea.viewer:app
```

Then visit:
- Evolution viewer/creator: http://localhost:8000/
- Idea rater: http://localhost:8000/rate

## Architecture

- `viewer.py`: Main FastAPI application serving both viewer and rater functionality
- `evolution.py`: Core evolution engine
- `models.py`: Data models
- `llm.py`: LLM integration with Agent definitions
- `prompts/`: Prompt templates
- `static/`: Frontend assets
  - `html/`: HTML templates
  - `js/`: JavaScript files
  - `css/`: Stylesheets


# TODO
- Tune temperatures for each (context generation for example should be high) - optionally leave idea generation temp as configurable in UI
- For rankings, ensure each idea is one side of the comparison
- Improve intial context generation for idea seeding - too much in common for each idea
- Update to evolution to use the genotype/phenotype model
- Have an evolution variant to take human ratings as input during evolution
- TODO: add a modifier to each generations prompt based on critique feedback on overall quality of generation