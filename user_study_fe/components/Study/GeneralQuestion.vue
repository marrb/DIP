<script setup lang="ts">
	import type { IQuestion } from "~/types/IQuestion";
	import ScaleInput from "../Form/ScaleInput.vue";
	import { EAnswerType } from "~/types/enums/EAnswerType";

	defineProps<{
		question: IQuestion;
	}>();

	const model = defineModel<number>();
</script>

<template>
	<ScaleInput
		v-if="question?.answerType === EAnswerType.SCALE_1_5"
		:range="5"
		v-model="model"
	/>
</template>
