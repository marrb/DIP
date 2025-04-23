<script setup lang="ts">
	import Section from "~/components/Basic/Section.vue";
	import P from "~/components/Basic/P.vue";
	import A from "~/components/Basic/A.vue";
	import { ROUTES } from "#imports";
	import H4 from "~/components/Basic/H4.vue";
	import H2 from "~/components/Basic/H2.vue";
	import Ul from "~/components/Basic/Ul.vue";

	const runtimeConfig = useRuntimeConfig().public;
	const { formatDate } = useDateFormatter();
	const { locale } = useI18n();

	const deadline = formatDate(runtimeConfig.deadline, locale.value)
</script>

<template>
	<Section class="mb-10 flex flex-col gap-4">
		<H2 class="mb-2 mt-10 text-center">{{ $t("BeforeYouStart") }}! </H2>

		<P class="text-center">
			{{ $t("Information1") }}
		</P>

		<H4 class="mt-8">
			{{ $t("PurposeOfTheStudy") }}
		</H4>

		<P>
			{{ $t("Information2") }}
		</P>

		<Ul>
			<li>{{ $t("Background") }}</li>
			<li>{{ $t("MotionAndTiming") }}</li>
			<li>{{ $t("OverallVisualAppearance") }}</li>
		</Ul>

		<P>
			{{ $t("Information3") }}
		</P>

		<H4>
			{{ $t("WhatToDo") }}
		</H4>

		<Ul>
			<li>{{ $t("WhatToDo1") }}</li>
			<li>{{ $t("WhatToDo2") }}</li>
			<li>{{ $t("WhatToDo3") }}</li>
		</Ul>

		<H4>
			{{ $t("SavingProgress") }}
		</H4>

		<P>
			{{ $t("SavingProgress1") }}
		</P>

		<H4>
			{{ $t("Tips") }}
		</H4>

		<Ul>
			<li>{{ $t("Tips1") }}</li>
		</Ul>

		<P class="mt-4">
			{{ $t("Runtime", { deadline: deadline }) }}
		</P>

		<A
			:to="ROUTES.STUDY"
			class="mx-auto mt-4"
		>
			{{ $t("UserStudyStart") }}
		</A>
	</Section>
</template>
