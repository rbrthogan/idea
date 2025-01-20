from tqdm import tqdm

from tenacity import retry, stop_after_attempt, wait_exponential

from typing import List
from idea.models import Idea
from idea.llm import LLMWrapper

@retry(
    stop=stop_after_attempt(5),  # Maximum 5 attempts
    wait=wait_exponential(multiplier=1, min=30, max=300),  # Wait between 30 and 300 seconds, doubling each time
    reraise=True
)
def _call_llm_with_retry(llm: LLMWrapper, prompt: str, temperature: float = 1.0) -> str:
    """Wrapper function to call LLM with retry logic"""
    return llm.generate_text(prompt, temperature=temperature)

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=30, max=300),
    reraise=True
)
def _generate_idea_with_retry(llm: LLMWrapper, prompt: str, temperature: float = 1.0) -> Idea:
    """Wrapper function to generate idea with retry logic"""
    return llm.generate_idea(prompt, temperature=temperature)

def initial_idea_context(method: str, llm: LLMWrapper) -> str:
    """
    Given a prompt, generate an initial idea context.
    """
    if method == "random_words":
        prompt = (
            "generate a list of 50 randomly choses english words. "
            "When generating each new reword, review what has come before "
            "and create a word that is as different as possible from the preceding set. "
            "Return the list of words as a single string, separated by spaces with no other text."
        )
    else:
        raise ValueError(f"Invalid method: {method}")

    return _call_llm_with_retry(llm, prompt, temperature=1)

def get_idea_prompt(idea_type: str) -> str:
    """
    Given a idea type, generate an idea prompt.
    """
    if idea_type == "airesearch":
        idea_context = (
            "The above is some context to get you inspired. "
            "Using some or none of it for inspiration,  "
            "suggest a promising area for novel research in AI (it can be a big idea, or a small idea, in any subdomain)."
            "keep it concise and to the point but with enough detail for a researcher to critique it."
        )
    else:
        raise ValueError(f"Invalid idea type: {idea_type}")

    return idea_context

def seed_ideas(n: int, context_type: str, idea_type: str, llm: LLMWrapper) -> List[Idea]:
    """
    Given a context, generate n ideas.
    """
    idea_prompt = get_idea_prompt(idea_type)
    ideas = []
    for i in tqdm(range(n), desc="Generating ideas"):
        context = initial_idea_context(context_type, llm)
        prompt = (
            f"{context}\n"
            f"Instruction: {idea_prompt}"
        )
        ideas.append(_generate_idea_with_retry(llm, prompt, temperature=1))
    return ideas


def critique_idea(idea: Idea, llm: LLMWrapper) -> Idea:
    """
    Take an idea and let an LLM 'critique' it.
    """
    prompt = (
        "You are a helpful AI that refines research ideas.\n"
        f"Current Idea:\nTitle: {idea.title}\n"
        f"Proposal: {idea.proposal}\n\n"
        "Please consider the above proposal. Offer critical feedback. Pointing out potential pitfalls as well as strengths."
        "No additional text, just the critique."
    )
    return _call_llm_with_retry(llm, prompt)


def refine_idea(idea: Idea, llm: LLMWrapper) -> Idea:
    """
    Take an idea and let an LLM refine it.
    """
    critique = critique_idea(idea, llm)
    prompt = (
        f"Current Idea:\nTitle: {idea.title}\nProposal: {idea.proposal}\n\n"
        f"Critique: {critique}\n\n"
        "Please review both, consider your own opinion and create your own proposal. This could be a refinement of the original proposal or a fresh take on it."
    )
    return _generate_idea_with_retry(llm, prompt)

def remove_worst_idea(ideas: List[Idea], llm: LLMWrapper) -> List[Idea]:
    """
    Remove the worst idea from the list.
    """
    idea_str = "\n".join([f"{i+1}. Title: {idea.title}\nProposal: {idea.proposal}" for i, idea in enumerate(ideas)])
    prompt = (
        "You are an experienced researcher and you are given a list of proposals. \n"
        "Please review the ideas and give a once sentence pro and con for each.\n"
        "After this, please give the idea that you think is the worst considering value, novelty, and feasibility.\n"
        "The ideas are:\n"
        f"{idea_str}\n"
        "Please return the idea that you think is the worst in the following format (with no other text following):\n"
        "Worst Idea: <idea number>"
    )
    result = _call_llm_with_retry(llm, prompt)

    parsed_result = result.split("Worst Idea:")[1].strip()
    remaining_ideas = [ideas[i] for i in range(len(ideas)) if i != int(parsed_result) - 1]
    return remaining_ideas

def generate_new_idea(ideas: List[Idea], llm: LLMWrapper) -> Idea:
    """
    Generate a new idea to add to the population.
    """
    idea_str = "\n".join([f"{i+1}. Title: {idea.title}\nProposal: {idea.proposal}" for i, idea in enumerate(ideas)])
    prompt = (
        "You are an experienced researcher and you are given a list of proposals. \n"
        f"{idea_str}\n"
        "Considering the above ideas please propose a new idea, that could be completely new and different from the ideas above or could combine ideas to create a new idea.\n"
        "Please avoid minor refinements of the ideas above, and instead propose a new idea that is a significant departure.\n"
    )
    return _generate_idea_with_retry(llm, prompt)


if __name__ == "__main__":
    llm = LLMWrapper()
    ideas = seed_ideas(2, context_type="random_words", idea_type="airesearch", llm=llm)

    for idea in ideas:
        print(idea.title)
        print(idea.proposal)
        print("\n\n")
