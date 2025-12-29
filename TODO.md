# TODO

## Features
- Allow users to provide more guidance on particular flavours of ideas or add more contraints e.g. not general AI research but specifically on mechanistic interpretability.
- Add an Idea type that focuses on field cross-pollination, e.g. "Idea from field X that could be applied to problem Y from field Z".
- For slower models with high latency it seems like nothing is happening for a while, and a some more verbosity to the UI e.g. One line of grey text under the progress bar that reports the current log activity. e.g. "Generating initial context for idea x of y ... ", "rated limited, backing off for X seconds", etc. This will whizz by quickly for fast models, but be more visible for slow models.
- Add download button for to charts (diversity and ELO ratings) to allow easy use later.

## Research Ideas
- Have an evolution variant to take human ratings as input during evolution
- Add a meta prompt to get the model to improve the new idea generation prompts based on the history of the evolution.
- Split evolution population into multiple "islands" of ideas, which are evolved independently with only rare opportunity for cross-island communication.
- introduce a concept of mutation to the genotype
- improve the tournament selection by using niching - group similar ideas to keep with each other -> maintain multiple directions of exploration -> similar to the "island" idea but more dynamic vs frozen in from the start.
- There is still too much dependency on the intital context prompt - similar themes are often generated --> inject more true randomness e.g return to original idea of random words but now just augment the prompt with these random words. Or ideally something better.
- Explore combining cheaper models with expensive ones. Keep models can go wide and explore much larger space of ideas perhaps providing a sampling space for the more expensive models to get better quality ideas from.
- Consider not replacing entire population each generation but instead only a subset (selected by either weakness, age or both) -- probably need to try larger population sizes to see if this works well.


## Bugs
- When you attached to a running evolution the progress bar is there but the timer state resets to 0 so you don't know how long it has been running.
- Diversity chart is not updating

## UX Improvements
- Rater page should reuse the same history sidebar as the main viewer page instead of having a separate dropdown