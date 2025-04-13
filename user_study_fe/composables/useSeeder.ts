import { doc, setDoc } from "firebase/firestore";
import { questionSeedData } from "~/seed/questions";
import type { IQuestion } from "~/types/IQuestion";

export async function useSeeder() {
	const { $db } = useNuxtApp();
	const { getRandomId } = useIdGenerator();
	const { seedVideos } = useVideoSeeder();

	// Seed videos first to ensure they are available for questions
	const videos = await seedVideos();

	const questionData: IQuestion[] = questionSeedData(getRandomId, videos);
	const questionsDocRef = doc($db, "questions", "questions");
	await setDoc(questionsDocRef, {
		questions: questionData,
	});
}
