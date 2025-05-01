from typing import TypedDict, Union, List, Optional

class Answer(TypedDict):
    id: str
    questionId: str
    answer: Union[List[str], int]
    videoId: Optional[str]
    createdAt: str
    