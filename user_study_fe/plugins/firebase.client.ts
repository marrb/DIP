import { getFirestore } from "firebase/firestore";
import { getAuth, signInAnonymously } from "firebase/auth";

export default defineNuxtPlugin((nuxtApp) => {
	const app = nuxtApp.$firebaseApp;
	const db = getFirestore(app as any);
	const auth = getAuth(app as any);

	signInAnonymously(auth).catch((error) => {
		console.error("Anonymous sign-in failed:", error);
	});

	return {
		provide: {
			db,
			auth,
		},
	};
});
