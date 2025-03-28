<script setup lang="ts">
	import type { IQuestion } from "~/types/IQuestion";
	import P from "../Basic/P.vue";

	const { locale } = useI18n();

	defineProps<{
		question: IQuestion;
	}>();

	const model = defineModel<string>();
</script>

<template>
	<div class="text-center">
		<div>VIDEOS</div>
		<P
			v-if="question.prompt"
			class="mt-2"
		>
			{{ locale === "en" ? question?.prompt : question?.promptSk }}
		</P>
	</div>
</template>
