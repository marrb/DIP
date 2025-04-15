<script setup lang="ts">
	const { t } = useI18n();

	useHead({
		title: t("Header"),
	});

	if (import.meta.client) {
		//await useSeeder();
	}
</script>

<template>
	<NuxtLayout>
		<NuxtPage />
	</NuxtLayout>
</template>
