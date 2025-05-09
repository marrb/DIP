<script setup lang="ts">
	import type { IQuestion } from "~/types/IQuestion";
	import P from "../Basic/P.vue";
	import Video from "../Basic/Video.vue";
	import VideoRanking from "../Videos/VideoRanking.vue";
	import VideoChoice from "../Videos/VideoChoice.vue";
	import { EVideoAnswerType } from "~/types/enums/EVideoAnswerType";

	// Composables
	const { locale } = useI18n();

	// Props
	const props = defineProps<{
		question: IQuestion;
	}>();

	// Model
	const model = defineModel<string[]>();

	// Computed
	const originalVideo = computed(() => {
		if (!props.question.videos || !props.question.videos.length) {
			return null;
		}

		return props.question.videos.find((video) => !video.originalVideoName);
	});

	const videos = computed(() => {
		if (!props.question.videos || !props.question.videos.length) {
			return null;
		}

		return props.question.videos.filter((video) => video.originalVideoName);
	});

	const canProceed = computed(() => {
		if (!model.value) {
			return false;
		}

		if (props.question.answerType === EVideoAnswerType.RANKING) {
			return model.value.length === videos.value.length;
		}

		if (props.question.answerType === EVideoAnswerType.CHOICE) {
			return model.value.length === 1;
		}
	});

	// Expose
	defineExpose({
		canProceed,
	});
</script>

<template>
	<div
		v-if="question"
		class="text-center"
	>
		<div>
			<P class="mb-4 font-extrabold underline text-lg">{{ $t("OriginalVideo") }}</P>
			<Video
				v-if="originalVideo"
				class="max-w-52 mx-auto mb-4"
				:video="originalVideo"
			/>
		</div>
		<div class="border-t-2 border-gray-300 pt-4 mt-4">
			<P class="mb-4 font-extrabold underline text-lg">{{ $t("EditedVideos") }}</P>
			<P
				v-if="question?.prompt"
				class="mt-2 mb-4"
			>
				{{ $t("PromptText") }}
				<span class="font-bold">{{ locale === "en" ? question?.prompt : question?.promptSk }}</span>
			</P>
			<VideoRanking
				v-if="question.answerType == EVideoAnswerType.RANKING"
				:videos="videos"
				:sort-order="question.videoSortOrder"
				v-model="model"
			/>
			<VideoChoice
				v-else-if="question.answerType == EVideoAnswerType.CHOICE"
				:videos="videos"
				:sort-order="question.videoSortOrder"
				v-model="model"
			/>
		</div>
		<P
			v-if="question?.hint"
			class="mt-2"
		>
			{{ $t("HintText") }}
			<span class="font-bold">{{ locale === "en" ? question?.hint : question?.hintSk }}</span>
		</P>
	</div>
</template>
