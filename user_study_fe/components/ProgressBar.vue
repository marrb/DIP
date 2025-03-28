<script setup lang="ts">
	const props = defineProps<{
		total: number;
		current: number;
	}>();

	console.log(props.total, props.current);
</script>

<template>
	<div
		v-if="total != null && current != null"
		class="h-[20px] bg-gray-200 rounded-full w-full relative"
	>
		<div
			class="h-[20px] bg-blue-400 rounded-full transition-all duration-300"
			:style="{ width: `${(current / total) * 100}%` }"
		></div>

		<div class="absolute -translate-x-1/2 -translate-y-1/2 top-1/2 left-1/2 text-primary font-bold">
			{{ total === 0 ? 100 : ((current / total) * 100).toFixed(0) }}%
		</div>
	</div>
</template>
