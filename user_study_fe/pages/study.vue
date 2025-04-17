<script setup lang="ts">
	import type { User } from "firebase/auth";
	import { useCollection } from "~/composables/useCollection";
	import { ECollection } from "~/types/enums/ECollection";
	import type { IQuestion } from "~/types/IQuestion";
	import Section from "~/components/Basic/Section.vue";
	import ProgressBar from "~/components/ProgressBar.vue";
	import H3 from "~/components/Basic/H3.vue";
	import { EQuestionType } from "~/types/enums/EQuestionType";
	import Button from "~/components/Basic/Button.vue";
	import LoadingSpinner from "~/components/Basic/LoadingSpinner.vue";
	import GeneralQuestion from "~/components/Study/GeneralQuestion.vue";
	import VideoQuestion from "~/components/Study/VideoQuestion.vue";
	import ErrorMessage from "~/components/Basic/ErrorMessage.vue";

	// Composables
	const { getAnswers, updateAnswers } = useAnswers();
	const { getRandomId } = useIdGenerator();
	const { locale } = useI18n();
	const { deadline } = useRuntimeConfig().public;

	// Data
	const questions = ref<IQuestion[]>(null);
	const user = ref<User>(null);
	const answers = ref<IAnswer[]>([]);
	const currentQuestion = ref<IQuestion>(null);
	const isLoading = ref(true);
	const isUpdating = ref(false);
	const showError = ref(false);
	const today = new Date();

	// Refs
	const questionModel = ref<number | string[]>();
	const videoQuestionRef = ref<typeof VideoQuestion>(null);
	const generalQuestionRef = ref<typeof GeneralQuestion>(null);

	// Computed
	const remainingQuestions = computed(() => {
		if (!answers.value || !answers.value.length) {
			return questions.value;
		}

		if (!questions.value || !questions.value.length) {
			return [];
		}

		return questions.value?.filter(
			(question) => !answers.value.find((answer) => answer.questionId === question.id)
		);
	});

	const currentQuestionIdx = computed(() => {
		if (!questions.value || !currentQuestion.value) {
			return -1;
		}

		return questions.value.findIndex((question) => question.id === currentQuestion.value.id);
	});

	const prevButtonDisabled = computed(() => {
		if (!currentQuestion.value) {
			return true;
		}

		return currentQuestionIdx.value === 0 || currentQuestionIdx.value === -1;
	});

	const nextButtonDisabled = computed(() => {
		if (currentQuestion.value.questionType === EQuestionType.VIDEO) {
			return !videoQuestionRef.value?.canProceed;
		}

		if (currentQuestion.value.questionType === EQuestionType.GENERAL) {
			return !generalQuestionRef.value?.canProceed;
		}

		return false;
	});

	const isDeadlinePassed = computed(() => {
		if (!deadline) {
			return false;
		}

		return new Date(deadline).getTime() < today.getTime();
	});

	// Methods
	const nextQuestion = async () => {
		if (!currentQuestion.value) {
			return;
		}

		const exisitingAnswer = answers.value.find((answer) => answer.questionId === currentQuestion.value.id);
		if (exisitingAnswer) {
			if (exisitingAnswer.answer === questionModel.value) {
				currentQuestion.value = questions.value[currentQuestionIdx.value + 1];
				questionModel.value = null;
				return;
			}

			answers.value = answers.value.filter((answer) => answer.questionId !== currentQuestion.value.id);
		}

		answers.value.push({
			id: getRandomId(),
			questionId: currentQuestion.value.id,
			createdAt: new Date().toISOString(),
			answer: questionModel.value,
		});

		isUpdating.value = true;
		await updateAnswers(answers.value);
		isUpdating.value = false;

		if (!remainingQuestions?.value || remainingQuestions.value.length === 0) {
			return;
		}

		currentQuestion.value = questions.value[currentQuestionIdx.value + 1];
		questionModel.value = null;

		const answer = answers.value.find((answer) => answer.questionId === currentQuestion.value.id);
		if (!answer) {
			return;
		}

		questionModel.value = answer.answer;
	};

	const prevQuestion = () => {
		if (currentQuestionIdx.value === 0) {
			return;
		}

		currentQuestion.value = questions.value[currentQuestionIdx.value - 1];
		const answer = answers.value.find((answer) => answer.questionId === currentQuestion.value.id);
		if (!answer) {
			return;
		}

		questionModel.value = answer.answer;
	};

	// Lifecycle
	onMounted(async () => {
		questions.value = (await useCollection<IQuestion>(ECollection.QUESTIONS, "questions"))?.documents[0]?.data;
		user.value = await useAuth();
		answers.value = await getAnswers();

		if (!answers.value) {
			answers.value = [];
		}

		if (remainingQuestions?.value?.length > 0) {
			currentQuestion.value = remainingQuestions.value[0];
			const answer = answers.value.find((answer) => answer.questionId === currentQuestion.value.id);

			if (answer) {
				questionModel.value = answer.answer;
			}
		}

		isLoading.value = false;
	});
</script>

<template>
	<Section class="mb-10">
		<H3
			v-if="isDeadlinePassed"
			class="text-center mt-12"
		>
			{{ $t("DeadlinePassed") }}
		</H3>
		<template v-else>
			<LoadingSpinner
				v-if="isLoading"
				class="mx-auto mt-10 w-10 h-10"
			/>
			<template v-else-if="remainingQuestions?.length > 0">
				<ProgressBar
					:total="questions?.length"
					:current="currentQuestionIdx"
					class="mb-10"
				/>

				<div class="flex flex-col gap-12 items-center">
					<H3 class="text-center">
						{{ locale === "en" ? currentQuestion?.title : currentQuestion?.titleSk }}
					</H3>

					<div>
						<VideoQuestion
							ref="videoQuestionRef"
							v-if="currentQuestion?.questionType === EQuestionType.VIDEO"
							:question="currentQuestion"
							v-model="questionModel as string[]"
						/>
						<GeneralQuestion
							ref="generalQuestionRef"
							v-else-if="currentQuestion?.questionType === EQuestionType.GENERAL"
							:question="currentQuestion"
							v-model="questionModel as number"
						/>
						<ErrorMessage
							v-if="showError"
							class="text-center mt-2"
						>
							{{ $t("NoAnswer") }}
						</ErrorMessage>
					</div>

					<div class="flex gap-4">
						<Button
							v-if="currentQuestionIdx !== 0"
							variant="filled"
							@click="prevQuestion"
							:disabled="prevButtonDisabled"
						>
							{{ $t("Back") }}
						</Button>

						<Button
							variant="filled"
							@click="nextQuestion"
							:disabled="nextButtonDisabled"
						>
							<template v-if="isUpdating">
								<LoadingSpinner class="mx-auto w-5 h-5" />
							</template>
							<template v-else>
								{{ $t("Next") }}
							</template>
						</Button>
					</div>
				</div>
			</template>
			<H3
				v-else
				class="text-center mt-10"
			>
				{{ $t("AllQuestionsAnswered") }}
			</H3>
		</template>
	</Section>
</template>
