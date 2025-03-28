<script setup lang="ts">
	withDefaults(
		defineProps<{
			variant?: "filled" | "transparent";
		}>(),
		{
			variant: "transparent",
		}
	);
</script>

<template>
	<button
		class="cursor-pointer"
		:class="{
			'px-6 py-2 rounded-2xl bg-blue-600 hover:bg-blue-700 text-white font-semibold transition-all duration-200 shadow-md hover:shadow-lg w-max':
				variant === 'filled',
			'bg-transparent text-primary': variant === 'transparent',
			'!bg-blue-300': $attrs.disabled
		}"
	>
		<slot></slot>
	</button>
</template>
