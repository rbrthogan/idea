"""
Core Oracle prompts for promoting diversity and avoiding convergence in evolutionary idea generation
"""

ORACLE_INSTRUCTION = """
Your task is to GENERATE a new diverse idea to enrich the current generation with fresh and novel ideas.

ANALYSIS REQUIRED:
1. Identify the most common patterns, themes, approaches, and methodologies across ALL generations
2. Spot repetitive concepts, similar problem-solving approaches, or overused techniques
3. Find gaps in the idea space that haven't been explored — not just a twist on what's there; think of valid ideas that are completely disjoint from current ideas
4. Generate a prompt for a replacement idea that deliberately avoids the overused patterns

OUTPUT: Provide your analysis of patterns and your new diverse idea prompt that would add the most value to the population.
"""

ORACLE_FORMAT_INSTRUCTIONS = """
CRITICAL RESPONSE FORMAT:
You MUST structure your response EXACTLY as follows. Do NOT deviate from this format:

=== ORACLE ANALYSIS ===
[Your detailed analysis of patterns, overused concepts, themes, gaps, etc.]

=== IDEA PROMPT ===
[Your new idea prompt here - a concise and specific request for a new idea]

ABSOLUTELY CRITICAL FORMATTING RULES:
- Use EXACTLY those section headers with the equals signs
- Put your analysis in the first section
- The IDEA PROMPT section must contain ONLY the new idea prompt
- Do NOT include words like "Analysis:", "Recurring Elements:", or any meta-commentary in the IDEA PROMPT section
- Do NOT include any analytical text in the IDEA PROMPT section
- The IDEA PROMPT should be a complete, standalone creative request
- If you include ANY analysis in the IDEA PROMPT section, you will have failed the task
"""

ORACLE_MAIN_PROMPT = """
You are the Oracle - an AI agent specializing in promoting diversity and avoiding convergence in evolutionary idea generation.

CONTEXT: You are working with {idea_type} ideas.

COMPLETE EVOLUTION HISTORY (the last generation is the most recent):
{history_text}

{mode_instruction}

CONSTRAINTS:
- Your new idea must still fit the domain of {idea_type}
- Focus on genuine innovation and unexplored directions
- Avoid superficial changes - look for fundamentally different approaches
- Consider interdisciplinary connections and novel methodologies
{oracle_constraints}

Here are some examples of good idea prompts:
{example_idea_prompts}

{format_instructions}
"""

# Default Oracle constraints (can be overridden per idea type)
ORACLE_CONSTRAINTS = ""