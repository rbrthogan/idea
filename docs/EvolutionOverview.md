# Evolution Engine Overview

This document describes the main algorithm implemented in `idea/evolution.py` and how the supporting components interact.

![Evolution Engine Overview](evolution_overview.png)

## Key Components

- **Ideator** (`llm.py`)
  - Generates initial context and specific prompts; seeds Gen 0 ideas.
- **Formatter** (`llm.py`)
  - Converts raw LLM output into structured `Idea` objects.
- **Critic** (`llm.py`)
  - Critiques, refines, and runs pairwise comparisons; updates ELO.
- **Breeder** (`llm.py`)
  - Encodes to genotypes and breeds new ideas via a prompt‑driven pipeline.
- **Oracle** (`llm.py`)
  - Analyzes all generations and proposes a replacement prompt to diversify.
- **DiversityCalculator** (`diversity.py`)
  - Uses Gemini embeddings for diversity metrics and centroid calculations.

These agents are wrapped around Gemini models and share token accounting.

Below is a simplified outline of how ideas move through the engine each generation.

## Algorithm Flow

1. **Seeding**
   - The engine generates `pop_size` context pools using the Ideator.
   - Each context pool is turned into a specific prompt and an initial idea.
   - Critic refines each idea and the Formatter structures it. The results form generation 0.
   - Diversity metrics are computed for the initial population.

2. **Evolution Loop** (`generations` iterations)
   - **Tournament Ranking**: ideas are compared via the Critic to update ELO.
   - **Parent Allocation**: `_allocate_parent_slots` caps dominance and spreads breeding chances.
   - **Breeding**: parents → genotypes → context → specific prompt → new idea; then refine + format.
   - **Oracle Diversification**: Oracle generates a replacement prompt; the least interesting idea (closest to the all‑history centroid) is replaced and then refined + formatted.
   - **Creative Selection**: the most diverse idea (farthest from centroid) is preserved into the next generation and shown with a ⭐ in the UI.
   - **Diversity Calculation**: metrics recorded every generation; embeddings cached.
   - Progress updates go to the UI throughout.

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

## Practical Tips

- Use the Template section on the main page to generate or pick a template.
- Increase creative temperature and top_p for more exploration; use Oracle + creative selection to avoid convergence.
- Save results and use the Rater to compare runs; auto‑rating shows estimated LLM token costs.

Note: the UI visualizes progress and diversity live.