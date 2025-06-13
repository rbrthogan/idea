# TODO

## Features
- Allow users to provide more guidance on particular flavours of ideas or add more contraints e.g. not general AI research but specifically on mechanistic interpretability.
- Add an Idea type that focuses on field cross-pollination, e.g. "Idea from field X that could be applied to problem Y from field Z".
- For slower models with high latency it seems like nothing is happening for a while, and a some more verbosity to the UI e.g. One line of grey text under the progress bar that reports the current log activity. e.g. "Generating initial context for idea x of y ... ", "rated limited, backing off for X seconds", etc. This will whizz by quickly for fast models, but be more visible for slow models.



## Research Ideas
- Update to evolution to use the genotype/phenotype model
- Have an evolution variant to take human ratings as input during evolution
- Use embeddings to monitor similarity/diversity of ideas, track in UI, and maybe cull duplicates ()
- TODO: add a modifier to each generations prompt based on critique feedback on overall quality of generation (or perhaps all history, could even show it the ranks, etc.) -- Oracle prompt. Remove worst idea and replace with new one generated from oracle prompt each generation. Track impact on evolution.
- Add a meta prompt to get the model to improve the new idea generation prompts based on the history of the evolution.
- Split evolution population into multiple "islands" of ideas, which are evolved independently with only rare opportunity for cross-island communication.

## Bugs
- Fix oracle replacement mode (idea doesn't get replaced, oracle idea is lost)
- Oracle not appearing in cost estimate - check if it's getting counted properly
