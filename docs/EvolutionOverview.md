# Evolution Engine Overview

This document describes the main algorithm implemented in `idea/evolution.py` and how the supporting components interact.


## Key Components

- **Ideator** (`llm.py`)
  - Generates initial context and specific prompts for idea creation.
- **Formatter** (`llm.py`)
  - Cleans and structures raw ideas into `Idea` objects.
- **Critic** (`llm.py`)
  - Provides critique and refinement for ideas and runs pairwise comparisons.
- **Breeder** (`llm.py`)
  - Encodes ideas to a genotype, performs crossover and produces new ideas.
- **Oracle** (`llm.py`)
  - Analyzes the whole population and proposes a new idea to increase diversity.
- **DiversityCalculator** (`diversity.py`)
  - Uses Gemini embeddings to measure population diversity and centroid distance.

These agents are wrapped around Gemini models and share token accounting.

Below is a simplified animation of the evolution loop described in this document.
Each step is highlighted sequentially so you can follow how ideas move through
the engine:

![Evolution process animation](idea/static/img/evolution_process.gif)

## Algorithm Flow

1. **Seeding**
   - The engine generates `pop_size` context pools using the Ideator.
   - Each context pool is turned into a specific prompt and an initial idea.
   - Critic refines each idea and the Formatter structures it. The results form generation 0.
   - Diversity metrics are computed for the initial population.

2. **Evolution Loop** (`generations` iterations)
   - **Tournament Ranking**: ideas are grouped and compared via the Critic to assign ELO ratings.
   - **Parent Allocation**: `_allocate_parent_slots` distributes breeding opportunities based on rankings while capping dominance to maintain diversity.
   - **Breeding**: selected parents are encoded to genotypes, combined and converted back to new ideas using Ideator and Breeder helper logic. Each child is refined and formatted.
   - **Oracle Diversification**: after breeding, the Oracle inspects all generations and generates a new idea. It replaces the current idea closest to the population centroid.
   - **Elite Selection**: the most diverse idea (farthest from the centroid) is marked and carried over unchanged into the next generation where it is simply refined and formatted.
   - **Diversity Calculation**: after updates the DiversityCalculator records metrics and embeddings for the entire history.
   - Progress updates are sent to the UI throughout the loop.

3. **Completion**
   - When all generations are finished (or a stop is requested) a final update is sent including diversity statistics and token counts for each agent.

## Diversity and Embeddings

The engine stores Gemini embeddings for every idea. These are used for:

- Computing overall and per-generation diversity scores.
- Finding the population centroid to choose which idea the Oracle should replace.
- Selecting the most diverse idea for elite selection.

Embeddings are cached so that centroid calculations across all generations are efficient.

## File Locations

- `idea/evolution.py` – main engine implementing the above steps.
- `idea/llm.py` – contains agent classes used by the engine.
- `idea/diversity.py` – helper for calculating diversity and embeddings.
- `idea/prompts/` – YAML templates that define prompts for each idea type.
- `idea/viewer.py` – FastAPI app providing the UI and progress updates.

Reading these files together will give a full picture of how ideas evolve over time.

The animation in this document was generated with
`scripts/generate_evolution_animation.py` which can be run to regenerate the GIF
if needed.