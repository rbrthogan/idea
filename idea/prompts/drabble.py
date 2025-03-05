"""
Prompt configurations for Drabble (100-word story) idea type
"""

# Metadata
ITEM_TYPE = "stories"

# Context generation prompts
# RANDOM_WORDS_PROMPT = """Generate a list of 50 randomly chosen English words.
# When generating each new word, review what has come before
# and create a word that is as different as possible from the preceding set.
# Return the list of words as a single string, separated by commas with no other text."""

DRABBLE_FORMAT_PROMPT = """A drabble is a short work of fiction exactly 100 words in length (not including the title).
Despite its brevity, a drabble should tell a complete story with a beginning, middle, and end.

Your story will be judge on creativity, originality, and the ability to tell a complete story in 100 words.

Format your response as:
Title: [Your creative title]
Proposal: [Your 100-word story]"""

RANDOM_WORDS_PROMPT = """Generate a list of 50 English words. These words should be one of the following:
a story telling genre, theme, character, setting, plot, technique, moral, message, style, tone.
Return the list of words as a single string, separated by commas with no other text."""

KEY_IDEAS_ELEMENTS_PROMPT = """You are a uniquely creative writer.
Please list 5 impactful short stories that had a lasting impact on the field of {field}.
Write a very short description of each story, no more than 20 words.
Finally, extract some of the key elements or concepts that are important to the story without
giving away the story itself. Keep them more general and abstract.
When finished, collect all the concepts/elements and return them as a comma separated list in the format:
CONCEPTS:<concept1>, <concept2>, <concept3>, <concept4>, <concept5>, ...."""

# Idea generation prompt
IDEA_PROMPT = """
The above is some context that you can choose to use or ignore. It's just there a potential source of inspiration.

Write a drabble - a complete story in exactly 100 words.
""" + DRABBLE_FORMAT_PROMPT

NEW_IDEA_PROMPT = """You are a uniquely creative writer and you are given the preceeding list of drabbles.
With the above stories as inspiration, please propose a new drabble that could be completely new
and different from those above or could combine ideas from those above to create a new story.
Please avoid minor refinements and instead write something that is a significant departure.
Your story will be judge on creativity, originality, and the ability to tell a complete story in 100 words.
""" + DRABBLE_FORMAT_PROMPT

# Formatting prompt
FORMAT_PROMPT = """Take the following drabble and rewrite it in a clear,
structured format. With a title denoted by [Title:], and the story proposal denoted by [Proposal:]: {input_text}"""

# Critic prompts
CRITIQUE_PROMPT = """You are a professor of creative writing that reviewing drabble assignments.
Drabble from a promising student:
{idea}

Please consider the above story and offer critical feedback.
Pointing out weaknesses as well as strengths. If the story doesn't fit the drabble format or have a clear narrative arc,
or is too derivative or banal, please suggest how to improve it.
No additional text, just the critique."""

REFINE_PROMPT = """You are a professional creative writer. You have taken over from a previous write who has written a drabble.
The drabble had received a critique from a top creative writing professor and you are now tasked with refining it.

Original Drabble:
{idea}

Critique: {critique}

Please review both, consider your own opinion, style and skill and create your own take on the drabble.
This could be a refinement of the original or a fresh take on it.
Very importantly, the drabble should be 100 words long and a complete story. Do not describe a story or review a story.
Write the drabble itself.
No additional text, just the refined drabble on its own.
""" + DRABBLE_FORMAT_PROMPT

# Comparison criteria for rating
COMPARISON_CRITERIA = [
    "creativity",
    "originality",
    "completeness",
    "impact",
    "adherence to the drabble format (an actual 100 word story - not a description of a story or a review of a story)"
]

BREED_PROMPT = """
{ideas}
You are a professional creative writer setting yourself writing challenges.
You are presented with the above 100 word short stories (drabbles) and tasked with creating a new one.
This can be a combination of the existing drabbles or a completely new one that you were inspired to create.

Importantly you will be judged on originality of the new story so need to make sure you bring something new to the table.
Think outside the box and be creative.

""" + DRABBLE_FORMAT_PROMPT