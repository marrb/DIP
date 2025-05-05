import json
from json_types.QuestionJsonData import QuestionJsonData
from json_types.EQuestionType import EQuestionType
from json_types.Question import Question
from json_types.Answer import Answer
from json_types.Video import Video
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
    
    def get_question_by_id_and_answers(self, id: str) -> Tuple[Question, List[Answer]]:
        question = next((q for q in self.data['questions'] if q['id'] == id), None)
        
        if question is None:
            raise ValueError(f"Question with id {id} not found.")
        
        answers = [
            a 
            for user_answer in self.data['answers']
            for a in user_answer['data']
            if a['questionId'] == id
        ]
        
        return question, answers
    
    def get_questions_by_ids_and_answers(self, ids: List[str]) -> List[Tuple[Question, List[Answer]]]:
        result = []
        
        for id in ids:
            result.append(self.get_question_by_id_and_answers(id))
            
        return result
        
    def get_videos_by_question_id(self, question_id: str) -> List[Video]:
        question = next((q for q in self.data['questions'] if q['id'] == question_id), None)
        
        if question is None:
            raise ValueError(f"Question with id {question_id} not found.")
        
        videos = [
            v 
            for v in self.data['videos'] 
            if v['id'] in [qv['id'] for qv in question['videos']]
        ]
        
        return videos
    
    def get_videos_by_question_ids(self, question_ids: List[str]) -> List[Video]:
        result = []
        
        for question_id in question_ids:
            videos = self.get_videos_by_question_id(question_id)
            result.extend(videos)
            
        return result
        