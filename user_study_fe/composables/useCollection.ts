import { collection, getDocs } from "firebase/firestore";

export const useCollection = async <TReturn>(
	collectionName: string,
	attribute?: string
): Promise<ICollection<TReturn>> => {
	const { $db } = useNuxtApp();

	const colRef = collection($db, collectionName);
	const snapshot = await getDocs(colRef);

	return {
		documents: snapshot.docs.map((doc) => ({
			id: doc.id,
			data: attribute ? doc.data()[attribute] : (doc.data() as TReturn[]),
		})),
	};
};
