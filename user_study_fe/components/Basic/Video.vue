<script setup lang="ts">
	import type { IVideo } from "~/types/IVideo";
	import LoadingSpinner from "./LoadingSpinner.vue";

	defineProps<{
		video: IVideo;
	}>();

	const showLoadingSpinner = ref(true);

	const onLoad = () => {
		showLoadingSpinner.value = false;
	};
</script>

<template>
	<div>
		<LoadingSpinner v-if="showLoadingSpinner" class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
		<NuxtImg :src="video.url" :onLoad="onLoad" />
	</div>
</template>
