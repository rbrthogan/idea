# Evolution Engine Overview

This document describes the main algorithm implemented across `idea/evolution.py` and `idea/evolution_orchestrator.py`, and how the supporting components interact.

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
- **Hybrid Fitness + Survival** (`evolution.py`)
  - Computes per-idea fitness from normalized Elo and normalized diversity.
  - Selects survivors using age-decayed fitness (older ideas get gradually lower survival odds).
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
   - **Tournament Ranking**: ideas are compared via the Critic to update Elo.
   - **Fitness Scoring**:
     - Diversity is computed per idea as distance from the current population centroid.
     - Elo and diversity are normalized with running cross-generation statistics.
     - Hybrid fitness is computed as:
       - `fitness = alpha * elo_norm + (1 - alpha) * diversity_norm`
   - **Survivor Selection**:
     - Only a fraction of ideas are replaced each generation (`replacement_rate`).
     - Survivors are sampled by age-decayed fitness:
       - `survival_score = fitness * (floor + (1 - floor) * exp(-age_decay_rate * age))`
     - This avoids hard-killing old ideas while still encouraging turnover.
   - **Parent Allocation + Breeding**:
     - Child slots are filled using fitness-weighted parent selection with slot caps to avoid dominance.
     - Children are bred, then refined + formatted.
     - Mutation applies to newly bred children via the Breeder mutation setting.
   - **Oracle Diversification**: Oracle generates a replacement prompt; the least interesting idea (closest to the all‑history centroid) is replaced and then refined + formatted.
   - **Diversity Calculation**: metrics recorded every generation; embeddings cached.
   - Progress updates go to the UI throughout.

3. **Completion**
   - When all generations are finished (or a stop is requested) a final update is sent including diversity statistics and token counts for each agent.

## Diversity and Embeddings

The engine stores Gemini embeddings for every idea. These are used for:

- Computing overall and per-generation diversity scores.
- Finding the population centroid to choose which idea the Oracle should replace.
- Computing per-idea diversity distances used in hybrid fitness.

Embeddings are cached so that centroid calculations across all generations are efficient.

## File Locations

- `idea/evolution.py` – main engine implementing the above steps.
- `idea/evolution_orchestrator.py` – canonical generation loop (new runs + resume/continue).
- `idea/llm.py` – contains agent classes used by the engine.
- `idea/diversity.py` – helper for calculating diversity and embeddings.
- `idea/prompts/` – YAML templates that define prompts for each idea type.
- `idea/viewer.py` – FastAPI app providing the UI and progress updates.

Reading these files together will give a full picture of how ideas evolve over time.

## Practical Tips

- Use the Template section on the main page to generate or pick a template.
- Increase creative temperature and top_p for more exploration.
- Tune `replacementRate`, `fitnessAlpha`, and `ageDecayRate` together:
  - Lower `replacementRate` preserves more lineage continuity.
  - Lower `fitnessAlpha` increases diversity pressure.
  - Higher `ageDecayRate` increases turnover of older ideas.
- Save results and use the Rater to compare runs; auto‑rating shows estimated LLM token costs.

Note: the UI visualizes progress and diversity live.
