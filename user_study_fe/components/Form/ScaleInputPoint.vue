<script setup lang="ts">
	defineProps<{
		active: boolean;
	}>();
</script>

<template>
	<div
		class="border-blue-400 border-2 w-2 h-2 rounded-2xl hover:bg-blue-400 cursor-pointer transition-all duration-200 p-3"
		:class="{ 'bg-blue-400': active }"
	></div>
</template>
