import type { EAnswerType } from "./enums/EAnswerType";
import type { EQuestionType } from "./enums/EQuestionType";

declare interface IQuestion {
	id: string;
	prompt: string;
	promptSk: string;
	title: string;
	titleSk: string;
	questionType: EQuestionType;
	answerType: EAnswerType;
	videos: IVideo[];
}