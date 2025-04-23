import { EGeneralAnswerType } from "~/types/enums/EGeneralAnswerType";
import { EQuestionType } from "~/types/enums/EQuestionType";
import { EVideoAnswerType } from "~/types/enums/EVideoAnswerType";
import type { IQuestion } from "~/types/IQuestion";
import type { IVideo } from "~/types/IVideo";

export const questionSeedData = (getRandomId: () => string, videos: IVideo[]): IQuestion[] => {
	const questions: IQuestion[] = [
		{
			id: getRandomId(),
			title: "How familiar are you with AI-generated media?",
			titleSk: "Ako dobre poznáš AI generovaný obsah?",
			prompt: "",
			promptSk: "",
			questionType: EQuestionType.GENERAL,
			answerType: EGeneralAnswerType.SCALE_1_5,
			videos: [],
		},
		{
			id: getRandomId(),
			title: "How often do you use AI image/video generation tools?",
			titleSk: "Ako často používaš nástroje na generovanie AI obrázkov/videí?",
			prompt: "",
			promptSk: "",
			questionType: EQuestionType.GENERAL,
			answerType: EGeneralAnswerType.SCALE_1_5,
			videos: [],
		},
		{
			id: getRandomId(),
			title: "What is your experience with video editing or VFX?",
			titleSk: "Aké máš skúsenosti s úpravou videa alebo vizuálnymi efektmi?",
			prompt: "",
			promptSk: "",
			questionType: EQuestionType.GENERAL,
			answerType: EGeneralAnswerType.SCALE_1_5,
			videos: [],
		},
		{
			id: getRandomId(),
			title: "How often do you watch short-form video content (e.g. TikTok, Instagram Reels)?",
			titleSk: "Ako často sleduješ krátke videoformáty (napr. TikTok, Instagram Reels)?",
			prompt: "",
			promptSk: "",
			questionType: EQuestionType.GENERAL,
			answerType: EGeneralAnswerType.SCALE_1_5,
			videos: [],
		},
		{
			id: getRandomId(),
			title: "How confident are you in identifying AI-generated visuals in videos?",
			titleSk: "Ako sebavedomo vieš rozpoznať AI generovaný obraz vo videách?",
			prompt: "",
			promptSk: "",
			questionType: EQuestionType.GENERAL,
			answerType: EGeneralAnswerType.SCALE_1_5,
			videos: [],
		},
		{
			id: getRandomId(),
			title: "How would you rate your attention to visual detail in media?",
			titleSk: "Ako by si ohodnotil/a svoju pozornosť na vizuálne detaily v médiách?",
			prompt: "",
			promptSk: "",
			questionType: EQuestionType.GENERAL,
			answerType: EGeneralAnswerType.SCALE_1_5,
			videos: [],
		},
	]

	// Add video-based questions
	const groupedVideosByOriginal = videos.reduce((acc: Record<string, IVideo[]>, video) => {
		const originalVideoName = video.originalVideoName;
		if(!originalVideoName) return acc;

		if (!acc[originalVideoName]) {
			acc[originalVideoName] = [];
		}

		acc[originalVideoName].push(video);
		return acc;
	}, {} as Record<string, IVideo[]>);

	for (const videoGroup of Object.values(groupedVideosByOriginal)) {
		const groupVideos = Array.from(videoGroup.values());
		const originalVideo = videos.find((video) => video.originalName === videoGroup[0].originalVideoName);

		let prompt;
		let promptSk;

		switch (originalVideo.originalName) {
			case "car":
				prompt = "a futuristic car is driving on the road";
				promptSk = "futuristické auto jazdí po ceste";
				break;

			case "cat":
				prompt = "a dog with a sombrero is sitting in front of a tree";
				promptSk = "pes so sombrérom sedí pred stromom";
				break;

			case "girl_dance":
				prompt = "a man is dancing";
				promptSk = "muž tancuje";
				break;

			case "motorbike":
				prompt = "a Spider-Man is driving a bicycle in the forest";
				promptSk = "Spider-Man jazdí na bicykli v lese";
				break;

			case "rabbit_jump":
				prompt = "a origami rabbit is jumping on the grass";
				promptSk = "origami zajac skáče na tráve";
				break;

			case "squirrel_carrot":
				prompt = "a rabbit with sunglasses is eating a carrot";
				promptSk = "zajac v slnečných okuliaroch žerie mrkvu";
				break;

			default: 
				throw new Error(`Unknown original video name: ${originalVideo.originalName}`);
		}

		questions.push({
			id: getRandomId(),
			questionType: EQuestionType.VIDEO,
			answerType: EVideoAnswerType.RANKING,
			title: "Rank the videos by how well the main character or object in the video has been edited based on the text used for the edit.",
			titleSk: "Zoraď videá podľa toho, ako dobre bola upravená hlavná postava alebo objekt vo videu na základe textu použitého na úpravu.",
			prompt: prompt,
			promptSk: promptSk,
			videos: [...groupVideos, originalVideo],
			hint: "Click on the videos in order, starting with the best (1), to rank them from best to worst.",
			hintSk: "Klikni na videá podľa poradia, začni tým najlepším (1 = najlepšie) a pokračuj po najhoršie.",
		})

		questions.push({
			id: getRandomId(),
			questionType: EQuestionType.VIDEO,
			answerType: EVideoAnswerType.RANKING,
			title: "Rank videos by whether the edit looks the same from start to finish (e.g. no flickering or style changes).",
			titleSk: "Zoraď videá podľa toho, či úprava pôsobí rovnakým spôsobom od začiatku až do konca (napr. bez blikania alebo zmeny štýlu).",
			prompt: prompt,
			promptSk: promptSk,
			videos: [...groupVideos, originalVideo],
			hint: "Click on the videos in order, starting with the best (1), to rank them from best to worst.",
			hintSk: "Klikni na videá podľa poradia, začni tým najlepším (1 = najlepšie) a pokračuj po najhoršie.",
		})

		questions.push({
			id: getRandomId(),
			questionType: EQuestionType.VIDEO,
			answerType: EVideoAnswerType.CHOICE,
			title: "Which video kept most of the original scene intact?",
			titleSk: "Ktoré video zachovalo väčšinu pôvodnej scény?",
			prompt: prompt,
			promptSk: promptSk,
			videos: [...groupVideos, originalVideo],
			hint: "Click on the video that kept most of the original scene intact.",
			hintSk: "Klikni na video, ktoré zachovalo väčšinu pôvodnej scény.",
		})
	}

	return questions;
}
