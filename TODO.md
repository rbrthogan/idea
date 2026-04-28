# TODO

## Features
- Allow users to provide more guidance on particular flavours of ideas or add more constraints e.g. not general AI research but specifically on mechanistic interpretability.
- Add an Idea type that focuses on field cross-pollination, e.g. "Idea from field X that could be applied to problem Y from field Z".
- For slower models with high latency it seems like nothing is happening for a while. Add more verbosity to the UI, e.g. one line of grey text under the progress bar that reports the current log activity: "Generating initial context for idea x of y ... ", "rated limited, backing off for X seconds", etc. This will whizz by quickly for fast models, but be more visible for slow models.
- Add download button for charts (diversity and ELO ratings) to allow easy use later.
- Add file upload for e.g. pdf to provide context to idea template generation 
- update default template generator to 3.1

## Research Ideas
- Have an evolution variant to take human ratings as input during evolution
- Add a meta prompt to get the model to improve the new idea generation and breeding prompts based on the history of the evolution.
- Improve LLM-as-a-judge quality using debate.
- Improve tournament selection using niching: group similar ideas with each other to maintain multiple exploration directions, similar to islands but more dynamic.
- Reduce remaining dependency on the initial context prompt. Seed novelty checks and genotype mutation help, but similar themes can still recur; consider stronger randomness or richer novelty sources.
- Explore combining cheaper models with expensive ones. Cheap models can go wide and explore a much larger space of ideas, perhaps providing a sampling space for the more expensive models to improve.
- Explore a fuller population-level approach. Current partial replacement and hybrid fitness help, but the algorithm is still generation based.
- Genotype might be too aggressive a compression. Use the captured genotype history to inspect whether the essence of the idea is getting lost.
- Reject bred or Oracle-generated ideas that are too similar to previous ones.
- Is critique and refine really worth it? Maybe the idea and its critique should be kept together without mandatory refinement. Another parent and its critique could breed with it, especially if genotype compression loses the refinement anyway.
- Investigate cheaper evals (e.g. each new idea is rated just vs a ~ median ELO idea)


## Bugs

## UX Improvements
- Rater page should reuse the same history sidebar as the main viewer page instead of having a separate dropdown
