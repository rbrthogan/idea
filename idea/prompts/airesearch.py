"""
Prompt configurations for AI Research idea type
"""

# Metadata
ITEM_TYPE = "AI research proposals"

# Context generation prompts
RANDOM_WORDS_PROMPT = """Generate a list of 50 randomly chosen English words.
When generating each new word, review what has come before
and create a word that is as different as possible from the preceding set.
Return the list of words as a single string, separated by spaces with no other text."""

KEY_IDEAS_ELEMENTS_PROMPT = """You are a historian of innovation.
Please give 5 ideas that demonstrated crucial innovation in the field of {field}.
Write a very short description of each idea, no more than 20 words.
Finally, extract some of the key elements or concepts that are important to the idea without
giving away the idea itself. Keep them more general and abstract.
When finished, collect all the concepts/elements and return them as a comma separated list in the format:
CONCEPTS:<concept1>, <concept2>, <concept3>, <concept4>, <concept5>"""

# Idea generation prompt
IDEA_PROMPT = """The above is some context to get you inspired.
Using some or none of it for inspiration,
suggest a promising area for novel research in AI (it can be a big idea, or a small idea, in any subdomain).
keep it concise and to the point but with enough detail for a researcher to critique it.
The body of the proposal should be written in markdown syntax with headings, paragraphs, bullet lists as appropriate"""

NEW_IDEA_PROMPT = """You are an experienced researcher and you are given the preceeding list of proposals.
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
    "originality and novelty",
    "potential impact and significance",
    "feasibility and practicality",
    "clarity and coherence"
]

REMOVE_WORST_IDEA_PROMPT = """You are an experienced researcher and you are given a list of proposals.
Please review the proposals and give a once sentence pro and con for each.
If a proposal is unsufficiently detailed or lacks a clear structure this should count against it.
After this, please give the proposal that you think is the worst considering {criteria}.
The proposals are:
{ideas}
Please return the proposal that you think is the worst in the following format (with no other text following):
Worst Entry: <proposal number>"""