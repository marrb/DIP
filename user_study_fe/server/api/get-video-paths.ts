import fs from "fs";

const readDirectory = (dir: string): string[] => {
	const items = fs.readdirSync(dir);
	const paths: string[] = [];

	for(const item of items) {
		const itemPath = `${dir}/${item}`;
		const stats = fs.statSync(itemPath);

		if(stats.isDirectory()) {
			paths.push(...readDirectory(itemPath));
		} else if(stats.isFile()) {
			paths.push(itemPath.replace("./public", ""));
		}
	}

	return paths;
};

export default defineEventHandler(() => {
	const videosDir = "./public/videos";
	const videoPaths = readDirectory(videosDir);

	return {
		videoPaths,
	};
});
