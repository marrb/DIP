<script setup lang="ts">
	import type { IVideo } from "~/types/IVideo";
	import LoadingSpinner from "./LoadingSpinner.vue";
	import { faMaximize } from "@fortawesome/free-solid-svg-icons";
	import Modal from "./Modal.vue";

	defineProps<{
		video: IVideo;
	}>();

	// Refs
	const modalRef = ref<InstanceType<typeof Modal> | null>(null);
	const showLoadingSpinner = ref(true);

	const onLoad = () => {
		showLoadingSpinner.value = false;
	};
</script>

<template>
	<div class="relative">
		<LoadingSpinner
			v-if="showLoadingSpinner"
			class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2"
		/>

		<div class="relative w-full">
			<font-awesome
				:icon="faMaximize"
				class="absolute top-2 right-2 hover:text-blue-400 cursor-pointer"
				@click="modalRef?.show()"
			/>
			<NuxtImg
				:src="video.url"
				:onLoad="onLoad"
				class="w-full h-auto"
			/>
		</div>

		<Modal
			ref="modalRef"
			size="md"
		>
			<NuxtImg
				:src="video.url"
				class="w-full h-auto"
			/>
		</Modal>
	</div>
</template>
