# Autor: Martin Bublavý [xbubla02]

from typing import List, TypedDict
from .Question import Question
from .Video import Video
from .UserAnswer import UserAnswer

class QuestionJsonData(TypedDict):
    questions: List[Question]
    videos: List[Video]
    answers: List[UserAnswer]
    