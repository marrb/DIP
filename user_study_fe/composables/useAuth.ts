// composables/useAuthUser.ts
import { onAuthStateChanged, type User } from "firebase/auth";

export const useAuth = async (): Promise<User | null> => {
	const { $auth } = useNuxtApp();

	return new Promise((resolve) => {
		const unsub = onAuthStateChanged($auth, (user) => {
			unsub();
			resolve(user);
		});
	});
};
