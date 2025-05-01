import { ECollection } from "~/types/enums/ECollection";
import type { IQuestion } from "~/types/IQuestion";
import { useCollection } from "./useCollection";
import type { IVideo } from "~/types/IVideo";

export function useFirebaseExport() {
	const exportFirebaseToJson = async () => {
		const questions = (await useCollection<IQuestion>(ECollection.QUESTIONS, "questions"))?.documents[0]?.data;
		const answerDocuments = (await useCollection<IAnswer>(ECollection.ANSWERS, "answers"))?.documents;
		const videos = (await useCollection<IVideo>(ECollection.VIDEOS, "videos"))?.documents[0]?.data;

		const filteredAnswerDocuments = (answerDocuments).filter((answerDocument) => {
			// Filter all answers that have less than 3 minutes between the first and last answer and are not fully answered
			if (answerDocument.data.length < 24) {
				return false;
			}

			const firstAnswerTime = new Date(answerDocument.data[0].createdAt).getTime();
			const lastAnswerTime = new Date(answerDocument.data[answerDocument.data.length - 1].createdAt).getTime();

			const timeDifference = (lastAnswerTime - firstAnswerTime) / 1000 / 60; // in minutes

			if (timeDifference < 4) {
				return false;
			}

			return true;
		})

		const jsonData = {
			questions: questions,
			videos: videos,
			answers: filteredAnswerDocuments,
		}

		const jsonString = JSON.stringify(jsonData, null, 2); // Pretty print with 2 spaces
		const blob = new Blob([jsonString], { type: "application/json" });
		const url = URL.createObjectURL(blob);
		const a = document.createElement("a");
		a.href = url;
		a.download = "export.json";
		document.body.appendChild(a);
		a.click();
		document.body.removeChild(a);
		URL.revokeObjectURL(url); // Clean up the URL object
	};

	return {
		exportFirebaseToJson,
	};
}
