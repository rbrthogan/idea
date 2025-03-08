"""
Prompt configurations for Game Design idea type
"""

# Metadata
ITEM_TYPE = "game designs"

# Design requirements
DESIGN_REQUIREMENTS = """
The game should be simple enough that it can be implemented as a browser game without relying on lots of external assets.
The game complexity should similar in level to classic games like Breakout, Snake, Tetris, Pong, Pac-Man, Space Invaders, Frogger, Minesweeper, Tic Tac Toe, 2048, Wordle etc.
Avoid being derivative of these but limit the ambition to the level of complexity of these games.
You should format your idea with enough specificity that a developer can implement it.
The body of the proposal should be written in markdown syntax with headings, paragraphs, bullet lists as appropriate
"""

# Context generation prompts

CONTEXT_PROMPT = """Generate a list of 50 game design concepts. These concepts should be one of the following:
a genre, a game style, a game mechanic, a game object, a game art concept, a gameplay loop, a goal/objective, a narrative/theme, a game world
Return the list of concepts as a single string, separated by commas with no other text in format:
CONCEPTS:<concept1>, <concept2>, <concept3>, <concept4>, <concept5>, ...."""

# Idea generation prompt
IDEA_PROMPT = """You are a uniquely creative game designer.
The above is some context to get you inspired.
Using some or none of it for inspiration,
suggest a new game design that is both interesting and fun to play.
With key game mechanics and controls.""" + DESIGN_REQUIREMENTS

NEW_IDEA_PROMPT = """You are a uniquely creative game designer.
You are given the preceeding list of game designs.
Considering the above ideas please propose a new idea, that could be completely new
and different from the ideas above or could combine ideas to create a new idea.
Please avoid minor refinements of the ideas above, and instead propose a new idea
that is a significant departure.""" + DESIGN_REQUIREMENTS

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
""" + DESIGN_REQUIREMENTS

# Comparison criteria for rating
COMPARISON_CRITERIA = [
    "originality and creativity",
    "simplicity",
    "fun factor",
    "feasibility for simple standard browser implementation"
]

BREED_PROMPT = """
{ideas}
You are an experienced game designer and you are given the above game designs.
These designs received good feedback but the studio leadership passed on selecting them for development.
Your task it to create a new design that is better, that will be more likely to be selected for development.
This can be a combination of the best elements of the existing designs, a refinement of the existing designs or a completely new one that you were inspired to create.

Importantly you will be judged on originality of the new design so need to make sure you bring something new to the table.
Think outside the box and be creative.""" + DESIGN_REQUIREMENTS