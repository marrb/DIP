import type { EGeneralAnswerType } from "./enums/EGeneralAnswerType";
import type { EQuestionType } from "./enums/EQuestionType";
import type { EVideoAnswerType } from "./enums/EVideoAnswerType";
import type { IVideo } from "./IVideo";

declare interface IQuestion {
	id: string;
	prompt: string;
	promptSk: string;
	title: string;
	titleSk: string;
	questionType: EQuestionType;
	answerType: EGeneralAnswerType | EVideoAnswerType;
	videos: IVideo[];
	hint?: string;
	hintSk?: string;
	videoSortOrder?: number[];
}