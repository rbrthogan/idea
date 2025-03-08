"""
Prompt configurations for AI Research idea type
"""

# Metadata
ITEM_TYPE = "AI research proposals"

# Context generation prompts
CONTEXT_PROMPT = """Generate a list of 50 AI research concepts. These concepts should be one of the following:
a technique/method, an algorithm, an architecture, a dataset, a modality, a task, a problem/challenge, a solution, an application/use case, a future direction.
Return the list of concepts as a single string, separated by commas with no other text in format:
CONCEPTS:<concept1>, <concept2>, <concept3>, <concept4>, <concept5>, ...."""

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


BREED_PROMPT = """
{ideas}
You are an experienced ai researcher and you are given the above research proposals.
These proposals received good feedback but were unable to be selected for funding. Your task it to create a new proposal that is better.
This can be a combination of the best elements of the existing proposals, a refinement of the existing proposals or a completely new one that you were inspired to create.

Importantly you will be judged on originality of the new propos so need to make sure you bring something new to the table.
Think outside the box and be creative.

Format your response as:
Title: [Your creative title]
Proposal: [Your 100-word story]"""