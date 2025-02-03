from pydantic import BaseModel
from datetime import datetime


class Idea(BaseModel):
    """
    Represents a single proposal.
    """
    title: str  # optional
    proposal: str

    # You can include more fields if needed, e.g. metrics, iteration history, etc.

class FlattenedIdea(BaseModel):
    """
    Internal data model for an Idea, including ELO rating
    and references to generation/index if desired.
    """
    id: str
    generation_index: int
    idea_index: int
    title: str
    proposal: str
    elo: float

class RatingResult(BaseModel):
    """
    Data model for a user's rating result POST.
    We specify the IDs of the two ideas and which outcome occurred.
    outcome can be "A", "B", or "tie".
    """
    idea_a_id: str
    idea_b_id: str
    outcome: str  # "A", "B", or "tie"
