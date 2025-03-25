<script setup lang="ts">
	import type { RouteLocationAsPathGeneric, RouteLocationAsRelativeGeneric } from "vue-router";

	withDefaults(
		defineProps<{
			to: RouteLocationAsPathGeneric | RouteLocationAsRelativeGeneric;
			newTab?: boolean;
		}>(),
		{
			newTab: false,
		}
	);
</script>

<template>
	<NuxtLink
		:to="to"
		class="px-6 py-2 rounded-2xl bg-blue-600 hover:bg-blue-700 text-white font-semibold transition-all duration-200 shadow-md hover:shadow-lg w-max"
		:target="newTab ? '_blank' : undefined"
		:rel="newTab ? 'noopener noreferrer' : undefined"
	>
		<slot></slot
	></NuxtLink>
</template>
