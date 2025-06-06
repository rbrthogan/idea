from pydantic import BaseModel
from datetime import datetime


class Idea(BaseModel):
    """
    Represents a single idea.
    """
    title: str  # optional
    content: str

    # You can include more fields if needed, e.g. metrics, iteration history, etc.

