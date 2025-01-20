from pydantic import BaseModel


class Idea(BaseModel):
    """
    Represents a single proposal.
    """
    title: str  # optional
    proposal: str

    # You can include more fields if needed, e.g. metrics, iteration history, etc.