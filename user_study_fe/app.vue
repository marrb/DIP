<script setup lang="ts">
	import { useFirebaseExport } from "./composables/useFirebaseExport";

	const { t } = useI18n();
	const { exportFirebaseToJson } = useFirebaseExport();

	useHead({
		title: t("Header"),
	});

	if (import.meta.client) {
		//await useSeeder();
		//await exportFirebaseToJson();
	}
</script>

<template>
	<NuxtLayout>
		<NuxtPage />
	</NuxtLayout>
</template>
