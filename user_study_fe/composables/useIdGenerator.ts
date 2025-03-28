import { collection, doc } from "firebase/firestore";

export function useIdGenerator() {
	const getRandomId = () => {
		const { $db } = useNuxtApp();

		return doc(collection($db, "_")).id;
	};

	return {
		getRandomId,
	};
}
