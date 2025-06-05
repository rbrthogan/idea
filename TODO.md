# TODO

## Features
- TODO: visualise generation history as a graph of ideas and connections between them (parents/relationships), highlight the ranks etc
- TODO: styling needs work
- TODO: update "start auto rating" button state when in progress
- TODO: comparison prompt is in the evolution.py file, should be in a prompts folder
- TODO: add costs for auto rating

## Research Ideas
- Update to evolution to use the genotype/phenotype model
- Have an evolution variant to take human ratings as input during evolution
- TODO: use embeddings to monitor similarity/diversity of ideas, track in UI, and maybe cull duplicates ()
- TODO: add a modifier to each generations prompt based on critique feedback on overall quality of generation (or perhaps all history, could even show it the ranks, etc.) -- Oracle prompt. Remove worst idea and replace with new one generated from oracle prompt each generation. Track impact on evolution.

## Bugs
- manual rating without saving
- current evolution gets wipe from viewer display after going to rate and returning
