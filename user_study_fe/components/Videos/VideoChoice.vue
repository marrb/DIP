<script setup lang="ts">
	import type { IVideo } from "~/types/IVideo";
	import Video from "../Basic/Video.vue";
	import { warn } from "vue";

	// Props
	defineProps<{
		videos: IVideo[];
	}>();

	// Model
	const model = defineModel<string[]>();

	// Refs
	const choice = ref<string>(null);

	// Computed
	const isSelected = computed(() => (video: IVideo) => {
		if (!choice.value) {
			return false;
		}

		return choice.value === video.id;
	});

	// Methods
	const selectChoice = (video: IVideo) => {
		choice.value = video.id;
		model.value = [video.id];
	};

	watch(
		model,
		(newModel) => {
			if (newModel && newModel.length !== 0) {
				choice.value = newModel[0];
			}
			else {
				choice.value = null;
			}
		},
	);
</script>

<template>
	<div class="flex gap-4">
		<Video
			v-for="video of videos"
			:class="isSelected(video) ? 'border-2 border-blue-500' : ''"
			:video="video"
			class="cursor-pointer max-w-52 min-w-52 min-h-52"
			@click="selectChoice(video)"
		/>
	</div>
</template>
