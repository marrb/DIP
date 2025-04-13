import { EModel } from "~/types/enums/EModel";
import { getStorage, ref, uploadBytes, getDownloadURL, listAll, deleteObject } from "firebase/storage";
import type { IVideo } from "~/types/IVideo";
import { doc, setDoc } from "firebase/firestore";

export function useVideoSeeder() {
	const { $db } = useNuxtApp();
	const { getRandomId } = useIdGenerator();
	const storage = getStorage();

	const removeExistingVideos = async () => {
		const videosRef = ref(storage, "/");

		const list = await listAll(videosRef);
		for (const item of list.items) {
			console.log("Deleting video:", item.name);
			await deleteObject(item);
		}
	};

	const uploadVideo = async (videoPath: string, id: string): Promise<string> => {
		console.info(`Uplading video: ${videoPath}`);

		const response = await fetch(videoPath);
		const blob = await response.blob();
		const ext = videoPath.split(".").pop() as string;

		const newFileName = `${id}.${ext}`;
		const fileRef = ref(storage, newFileName);
		await uploadBytes(fileRef, blob);
		const downloadUrl = await getDownloadURL(fileRef);

		return downloadUrl;
	};

	const seedVideos = async () => {
		await removeExistingVideos();

		const data = await (await fetch("/api/get-video-paths")).json();
		const paths = data.videoPaths;
		const videos: IVideo[] = [];
		const originalVideoPaths: string[] = paths.filter((path: string) => path.includes(EModel.NONE));

		for (const originalVideoPath of originalVideoPaths) {
			const originalVideoId = getRandomId();
			const originalVideoName = originalVideoPath.split("/").pop().split(".")[0];

			const originalVideoDownloadUrl = await uploadVideo(originalVideoPath, originalVideoId);
			videos.push({
				id: originalVideoId,
				originalName: originalVideoName,
				url: originalVideoDownloadUrl,
				model: EModel.NONE,
			});

			const modelVideoPaths = paths.filter((path: string) => {
				const fileName = path.split("/").pop().split(".")[0];
				return fileName.includes(`[${originalVideoName}]`);
			});

			for (const modelVideoPath of modelVideoPaths) {
				const id = getRandomId();
				const downloadUrl = await uploadVideo(modelVideoPath, id);
				const model = modelVideoPath.split("/")[2] as EModel;
				const modelVideoName = modelVideoPath.split("/").pop();

				videos.push({
					id: id,
					originalName: modelVideoName,
					url: downloadUrl,
					model: model,
					originalVideoName: originalVideoName,
				});
			}
		}

		const videosDocRef = doc($db, "videos", "videos");
		await setDoc(videosDocRef, {
			videos: videos,
		});

		return videos;
	};

	return {
		seedVideos,
	};
}
