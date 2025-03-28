import { collection, doc, setDoc } from "firebase/firestore";
import { modelSeedData } from "~/seed/models";
import { questionSeedData } from "~/seed/questions";
import type { IQuestion } from "~/types/IQuestion";

export async function useSeeder() {
	const { $db } = useNuxtApp();
	const { getRandomId } = useIdGenerator();

	const modelsData: IModel[] = modelSeedData(getRandomId);
	const modelsDocRef = doc($db, "models", "models_1");
	await setDoc(modelsDocRef, {
		models: modelsData,
	});

	const questionData: IQuestion[] = questionSeedData(getRandomId);
	const questionsDocRef = doc($db, "questions", "questions_1");
	await setDoc(questionsDocRef, {
		questions: questionData,
	});
}
