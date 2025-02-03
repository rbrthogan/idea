# Idea Evolution

An application for evolving research ideas using LLMs and human feedback.

## Features

- Generate and evolve research ideas using LLM
- Rate ideas using ELO system
- View evolution history and ratings
- Save and load previous evolutions

## Running the App

Start the server:
```bash
uvicorn idea.viewer:app --port 8002
```

Then visit:
- Evolution viewer/creator: http://localhost:8002/
- Idea rater: http://localhost:8002/rate

## Architecture

- `viewer.py`: Main FastAPI application serving both viewer and rater functionality
- `evolution.py`: Core evolution engine
- `models.py`: Data models
- `llm.py`: LLM integration
- `static/`: Frontend assets
  - `html/`: HTML templates
  - `js/`: JavaScript files
  - `css/`: Stylesheets

- combine viewer and rater into one app
- Do automated rating with LLM
- Update to evolution to use the genotype/phenotype model
- Have an evolution variant to take human ratings as input during evolution
