<script setup lang="ts">
	import type { IVideo } from "~/types/IVideo";
	import Video from "../Basic/Video.vue";
	import P from "../Basic/P.vue";

	// Props
	defineProps<{
		videos: IVideo[];
	}>();

	// Model
	const model = defineModel<string[]>();

	// Refs
	const ranking = ref<{ [key: string]: number }>({});

	const addToRanking = (video: IVideo) => {
		// Check if video is already ranked, if so, remove it from ranking
		if (isRanked.value(video)) {
			delete ranking.value[video.id];
			model.value = Object.keys(ranking.value).sort((a, b) => ranking.value[a] - ranking.value[b]);
			return;
		}

		// Get first available rank
		let rank = 1;
		while (Object.values(ranking.value).includes(rank)) {
			rank++;
		}

		ranking.value[video.id] = rank;
		model.value = Object.keys(ranking.value).sort((a, b) => ranking.value[a] - ranking.value[b]);
	};

	const isRanked = computed(() => (video: IVideo) => {
		if (!ranking.value) {
			return false;
		}

		return ranking.value[video.id] !== undefined;
	});

	const getRank = computed(() => (video: IVideo) => {
		if (!ranking.value) {
			return -1;
		}

		return ranking.value[video.id];
	});

	watch(
		model,
		(newModel) => {
			if (newModel && newModel.length !== 0) {
				let rank = 1;

				for (const id of newModel) {
					ranking.value[id] = rank;
					rank++;
				}
			}
			else {
				ranking.value = {};
			}
		},
	);
</script>

<template>
	<div class="flex gap-4">
		<div
			v-for="video of videos"
			class="flex flex-col gap-2 justify-end"
		>
			<P v-if="isRanked(video)">{{ getRank(video) }}</P>
			<Video
				:class="isRanked(video) ? 'border-2 border-blue-500' : ''"
				:video="video"
				class="cursor-pointer max-w-52 min-w-52 min-h-52"
				@click="addToRanking(video)"
			/>
		</div>
	</div>
</template>
