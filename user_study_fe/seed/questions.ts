import { EAnswerType } from "~/types/enums/EAnswerType";
import { EQuestionType } from "~/types/enums/EQuestionType";
import type { IQuestion } from "~/types/IQuestion";

export const questionSeedData = (getRandomId: () => string): IQuestion[] => [
	{
		id: getRandomId(),
		title: "How familiar are you with AI-generated media?",
		titleSk: "Ako dobre poznáš AI generovaný obsah?",
		prompt: "",
		promptSk: "",
		questionType: EQuestionType.GENERAL,
		answerType: EAnswerType.SCALE_1_5,
		videos: [],
	},
	{
		id: getRandomId(),
		title: "How often do you use AI image/video generation tools?",
		titleSk: "Ako často používaš nástroje na generovanie AI obrázkov/videí?",
		prompt: "",
		promptSk: "",
		questionType: EQuestionType.GENERAL,
		answerType: EAnswerType.SCALE_1_5,
		videos: [],
	},
	{
		id: getRandomId(),
		title: "What is your experience with video editing or VFX?",
		titleSk: "Aké máš skúsenosti s úpravou videa alebo vizuálnymi efektmi?",
		prompt: "",
		promptSk: "",
		questionType: EQuestionType.GENERAL,
		answerType: EAnswerType.SCALE_1_5,
		videos: [],
	},
	{
		id: getRandomId(),
		title: "How often do you watch short-form video content (e.g. TikTok, Instagram Reels)?",
		titleSk: "Ako často sleduješ krátke videoformáty (napr. TikTok, Instagram Reels)?",
		prompt: "",
		promptSk: "",
		questionType: EQuestionType.GENERAL,
		answerType: EAnswerType.SCALE_1_5,
		videos: [],
	},
	{
		id: getRandomId(),
		title: "How confident are you in identifying AI-generated visuals in videos?",
		titleSk: "Ako sebavedomo vieš rozpoznať AI generovaný obraz vo videách?",
		prompt: "",
		promptSk: "",
		questionType: EQuestionType.GENERAL,
		answerType: EAnswerType.SCALE_1_5,
		videos: [],
	},
	{
		id: getRandomId(),
		title: "How would you rate your attention to visual detail in media?",
		titleSk: "Ako by si ohodnotil/a svoju pozornosť na vizuálne detaily v médiách?",
		prompt: "",
		promptSk: "",
		questionType: EQuestionType.GENERAL,
		answerType: EAnswerType.SCALE_1_5,
		videos: [],
	},
];
