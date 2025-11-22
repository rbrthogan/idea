from pydantic import BaseModel


from typing import Optional

class Idea(BaseModel):
    """
    Represents a single idea.
    """
    title: Optional[str] = None
    content: str

    # You can include more fields if needed, e.g. metrics, iteration history, etc.
