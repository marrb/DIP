import { doc, setDoc, getDoc } from "firebase/firestore";
import { ECollection } from "~/types/enums/ECollection";

export function useAnswers() {
	const getAnswers = async (): Promise<IAnswer[]> => {
		const { $db, $auth } = useNuxtApp();
		const docRef = doc($db, ECollection.ANSWERS, $auth.currentUser.uid);

		return (await getDoc(docRef)).data()?.answers as IAnswer[];
	};

	const updateAnswers = async (answers: IAnswer[]) => {
		const { $db, $auth } = useNuxtApp();
		const docRef = doc($db, ECollection.ANSWERS, $auth.currentUser.uid);

		await setDoc(docRef, {
			answers,
		});
	};

	return {
		getAnswers,
		updateAnswers,
	};
}
