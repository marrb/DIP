export const modelSeedData = (getRandomId: () => string): IModel[] => [
	{
		id: getRandomId(),
		name: "Video-P2P",
		description: "Baseline video-p2p",
	},
	{
		id: getRandomId(),
		name: "EI Integration",
		description:
			"Enhanced model with STAM and FFAM modules integrated into Video-P2P for better temporal and spatial attention.",
	},
];
