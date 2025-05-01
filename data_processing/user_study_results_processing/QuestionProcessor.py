import json
import numpy as np
from json_types.QuestionJsonData import QuestionJsonData
from json_types.EQuestionType import EQuestionType
from json_types.Question import Question
from json_types.Answer import Answer
from typing import List, Tuple

class QuestionProcessor:
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.data: QuestionJsonData = self._load_json()
        
    def _load_json(self):
        with open(self.json_path, 'r', encoding="utf-8") as file:
            data: QuestionJsonData = json.load(file)
            
        return data
    
    def get_general_questions_and_answers(self) -> List[Tuple[Question, List[Answer]]]:    
        general_questions = [q for q in self.data['questions'] if q['questionType'] == EQuestionType.GENERAL.value]
        print(f"General Questions: {len(general_questions)}")
        
        general_answers = [
            a
            for user_answer in self.data['answers']
            for a in user_answer['data']
            if a['questionId'] in [q['id'] for q in general_questions]
        ]
        print(f"General Answers: {len(general_answers)}")   
        
        question_answer_pairs = []
        for question in general_questions:
            answers = [a for a in general_answers if a['questionId'] == question['id']]
            question_answer_pairs.append((question, answers))
            
        return question_answer_pairs
        
        