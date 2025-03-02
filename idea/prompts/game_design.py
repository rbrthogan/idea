"""
Prompt configurations for Game Design idea type
"""

# Metadata
ITEM_TYPE = "game designs"

# Context generation prompts
RANDOM_WORDS_PROMPT = """Generate a list of 50 randomly chosen English words.
When generating each new word, review what has come before
and create a word that is as different as possible from the preceding set.
Return the list of words as a single string, separated by spaces with no other text."""

KEY_IDEAS_ELEMENTS_PROMPT = """You are a uniquely creative game designer.
Please give 5 simple game designs that demonstrated crucial innovation in the field of {field}.
Write a very short description of each game design, no more than 20 words.
Finally, extract some of the key elements or concepts that are important to the game design without
giving away the game design itself. Keep them more general and abstract.
When finished, collect all the concepts/elements and return them as a comma separated list in the format:
CONCEPTS:<concept1>, <concept2>, <concept3>, <concept4>, <concept5>"""

# Idea generation prompt
IDEA_PROMPT = """You are a uniquely creative game designer.
The above is some context to get you inspired.
Using some or none of it for inspiration,
suggest a new game design that is both interesting and fun to play.
With key game mechanics and controls.
The game should be simple enough that it can be implemented as a browser game without relying on lots of external assets.
The game complexity should similar in level to classic games like Breakout, Snake, Tetris, Pong, Pac-Man, Space Invaders, Frogger, Minesweeper, Tic Tac Toe, 2048, Wordle etc.
Avoid being derivative of these but limit the ambition to the level of complexity of these games.
You should format your idea with enough specificity that a developer can implement it.
The body of the proposal should be written in markdown syntax with headings, paragraphs, bullet lists as appropriate"""

NEW_IDEA_PROMPT = """You are a uniquely creative game designer.
You are given the preceeding list of game designs.
Considering the above ideas please propose a new idea, that could be completely new
and different from the ideas above or could combine ideas to create a new idea.
Please avoid minor refinements of the ideas above, and instead propose a new idea
that is a significant departure.
The body of the proposal should be written in markdown syntax with headings, paragraphs, bullet lists as appropriate"""

# Formatting prompt
FORMAT_PROMPT = """Take the following idea and rewrite it in a clear,
structured format. The body of the proposal should be written in markdown syntax with headings, paragraphs, bullet lists as appropriate: {input_text}"""

# Critic prompts
CRITIQUE_PROMPT = """You are a helpful AI that refines ideas.
Current Idea:
{idea}

Please consider the above proposal. Offer critical feedback.
Pointing out potential pitfalls as well as strengths. If the idea doesn't have a clear structure,
or sufficient detail, please suggest how to improve it and ask for elaboration of specific points of interest.
No additional text, just the critique."""

REFINE_PROMPT = """Current Idea:
{idea}

Critique: {critique}

Please review both, consider your own opinion and create your own proposal.
This could be a refinement of the original proposal or a fresh take on it.
No additional text, just the refined idea on its own.
Try have a sensible structure to the idea with markdown headings.
It should be sufficient detailed to convey main idea to someone in the field."""

# Comparison criteria for rating
COMPARISON_CRITERIA = [
    "originality and creativity",
    "simplicity",
    "fun factor",
    "feasibility for simple standard browser implementation"
]

REMOVE_WORST_IDEA_PROMPT = """You are an experienced game designer and you are given a list of game designs.
Please review the designs and give a once sentence pro and con for each.
If a design is unsufficiently detailed or lacks a clear structure this should count against it.
After this, please give the design that you think is the worst considering {criteria}.
The designs are:
{ideas}
Please return the design that you think is the worst in the following format (with no other text following):
Worst Entry: <design number>"""