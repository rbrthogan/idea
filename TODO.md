# TODO

## Features
- Allow users to provide more guidance on particular flavours of ideas or add more contraints e.g. not general AI research but specifically on mechanistic interpretability.
- Add an Idea type that focuses on field cross-pollination, e.g. "Idea from field X that could be applied to problem Y from field Z".
- For slower models with high latency it seems like nothing is happening for a while, and a some more verbosity to the UI e.g. One line of grey text under the progress bar that reports the current log activity. e.g. "Generating initial context for idea x of y ... ", "rated limited, backing off for X seconds", etc. This will whizz by quickly for fast models, but be more visible for slow models.
- Add download button for to charts (diversity and ELO ratings) to allow easy use later.
- Add another diversity metric (inter-generation diversity; compute centroids of each generation and compute distance between them)
- Make it easier to add new templates by using an LLM to generate the template from a simple prompt (using existing templates as few shot examples)

## Research Ideas
- Have an evolution variant to take human ratings as input during evolution
- Add a meta prompt to get the model to improve the new idea generation prompts based on the history of the evolution.
- Split evolution population into multiple "islands" of ideas, which are evolved independently with only rare opportunity for cross-island communication.
- introduce a concept of mutation to the genotype
- let the most different idea survive to the next generation (perhaps with a new mutation)
- improve the tournament selection by using niching - group similar ideas to keep with each other -> maintain multiple directions of exploration -> similar to the "island" idea but more dynamic vs frozen in from the start.

## Bugs
- When opening on "current evolution" the diversity chart initially appears with data but is then refreshed to empty. If saved it can be restored by selecting the saved evolution, but current evolution view should also work.
