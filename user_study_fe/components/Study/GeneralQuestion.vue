<script setup lang="ts">
	import type { IQuestion } from "~/types/IQuestion";
	import ScaleInput from "../Form/ScaleInput.vue";
	import { EGeneralAnswerType } from "~/types/enums/EGeneralAnswerType";

	// Props
	const props = defineProps<{
		question: IQuestion;
	}>();

	// Model
	const model = defineModel<number>();

	// Computed
	const canProceed = computed(() => {
		if (props.question.answerType === EGeneralAnswerType.SCALE_1_5) {
			return model.value >= 1 && model.value <= 5;
		}

		return false;
	});

	// Expose
	defineExpose({
		canProceed,
	});
</script>

<template>
	<ScaleInput
		v-if="question?.answerType === EGeneralAnswerType.SCALE_1_5"
		:range="5"
		v-model="model"
	/>
</template>
