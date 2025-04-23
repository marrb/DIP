<script setup lang="ts">
	import type { IVideo } from "~/types/IVideo";
	import Video from "../Basic/Video.vue";

	// Props
	const props = defineProps<{
		videos: IVideo[];
		sortOrder: number[];
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

	const sortedVideos = computed(() => {
		return props.sortOrder.map(index => props.videos[index]);
	});

	// Methods
	const selectChoice = (video: IVideo) => {
		choice.value = video.id;
		model.value = [video.id];
	};

	if (model.value && model.value.length > 0) {
		choice.value = model.value[0];
	}

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
	<div class="flex gap-4 flex-wrap justify-center">
		<Video
			v-for="video of sortedVideos"
			:class="isSelected(video) ? 'border-4 border-blue-500' : ''"
			:video="video"
			class="cursor-pointer max-w-52 min-w-52 min-h-52"
			@click="selectChoice(video)"
		/>
	</div>
</template>
