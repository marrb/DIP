from .EQuestionType import EQuestionType
from .EGeneralAnswerType import EGeneralAnswerType
from .EVideoAnswerType import EVideoAnswerType
from typing import TypedDict, Optional, Union, List
from .Video import Video
from dataclasses import dataclass

class Question(TypedDict):
    id: str
    prompt: Optional[str]
    promptSk: Optional[str]
    title: str
    titleSk: str
    questionType: EQuestionType
    answerType: Union[EGeneralAnswerType, EVideoAnswerType]
    videos: List[Video]
    hint: Optional[str]
    hintSk: Optional[str]
    videoSortOrder: Optional[List[int]]
    