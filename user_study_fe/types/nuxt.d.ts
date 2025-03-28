import type { Firestore, CollectionReference } from "firebase/firestore";
import type { Auth } from "firebase/auth";

declare module "#app" {
	interface NuxtApp {
		$db: Firestore;
		$modelsRef: CollectionReference;
		$auth: Auth;
	}
}
