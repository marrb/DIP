# Autor: Martin Bublavý [xbubla02]

from typing import TypedDict, List
from .Answer import Answer

class UserAnswer(TypedDict):
    id: str
    data: List[Answer]