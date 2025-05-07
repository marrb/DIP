# Autor: Martin Bublavý [xbubla02]

from typing import TypedDict, Optional
from .EModel import EModel

class Video(TypedDict):
    id: str
    originalName: str
    url: str
    model: EModel
    originalVideoName: Optional[str]
    