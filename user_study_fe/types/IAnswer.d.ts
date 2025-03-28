declare interface IAnswer {
	id: string;
	questionId: string;
	answer?: string | number;
	videoId?: string;
	createdAt: string;
}

declare interface IAnswerCreate {
	questionId: string;
	answer?: string;
	videoId?: string;
	createdAt: string;
	timeTakenMinutes: number;
}
