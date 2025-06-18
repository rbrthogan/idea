# TODO

## Features
- Allow users to provide more guidance on particular flavours of ideas or add more contraints e.g. not general AI research but specifically on mechanistic interpretability.
- Add an Idea type that focuses on field cross-pollination, e.g. "Idea from field X that could be applied to problem Y from field Z".
- For slower models with high latency it seems like nothing is happening for a while, and a some more verbosity to the UI e.g. One line of grey text under the progress bar that reports the current log activity. e.g. "Generating initial context for idea x of y ... ", "rated limited, backing off for X seconds", etc. This will whizz by quickly for fast models, but be more visible for slow models.
- Add download button for to charts (diversity and ELO ratings) to allow easy use later.
- Add another diversity metric (inter-generation diversity; compute centroids of each generation and compute distance between them)
- LLM controls: consolidate temparature and top_p controls for various agents into one control for temperature and a new control for top_p. Don't set top_k. Formatter will remain hardcoded.

## Research Ideas
- Have an evolution variant to take human ratings as input during evolution
- Use diversity metrics to cull duplicates
- TODO: add a modifier to each generations prompt based on critique feedback on overall quality of generation (or perhaps all history, could even show it the ranks, etc.) -- Oracle prompt. Remove worst idea and replace with new one generated from oracle prompt each generation. Track impact on evolution.
- Add a meta prompt to get the model to improve the new idea generation prompts based on the history of the evolution.
- Split evolution population into multiple "islands" of ideas, which are evolved independently with only rare opportunity for cross-island communication.
- Relook at genotype breeding. More explicit crossover and mutation. Idea creation should be similar to initial population.

## Bugs
- When opening on "current evolution" the diversity chart initially appears with data but is then refreshed to empty. If saved it can be restored by selecting the saved evolution, but current evolution view should also work.