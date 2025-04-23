<script setup lang="ts">
	import Section from "~/components/Basic/Section.vue";
	import P from "~/components/Basic/P.vue";
	import H3 from "~/components/Basic/H3.vue";
	import { faEye, faClock, faLock } from "@fortawesome/free-solid-svg-icons";
	import A from "~/components/Basic/A.vue";
</script>

<template>
	<Section class="md:mt-40 text-gray-200 h-full">
		<div class="my-auto text-center flex flex-col gap-2">
			<H3 class="mb-2">{{ $t("Welcome") }}! </H3>
			<P>
				<font-awesome
					:icon="faEye"
					class="text-blue-400"
				/>
				{{ $t("UserStudy") }}
			</P>
			<P>
				<font-awesome
					:icon="faClock"
					class="text-yellow-400"
				/>
				{{ $t("UserStudyDuration") }}
			</P>
			<P>
				<font-awesome
					:icon="faLock"
					class="text-green-400"
				/>
				{{ $t("UserStudyConfidential") }}
			</P>
			<P class="font-semibold text-2xl mt-4">
				{{ $t("UserStudyThanks") }}
			</P>
			<A
				:to="ROUTES.INFORMATION"
				class="mt-4 mx-auto"
			>
				{{ $t("UserStudyStart") }}
			</A>
		</div>
	</Section>
</template>
