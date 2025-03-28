<script setup lang="ts">
	import ScaleInputPoint from "./ScaleInputPoint.vue";
	import P from "../Basic/P.vue";

	defineProps<{
		range: number;
	}>();

	const model = defineModel<number>();
</script>

<template>
	<div class="flex gap-4 items-center justify-center font-bold text-xl">
		<div
			v-for="i in range"
			:key="i"
			class="flex flex-col items-center justify-center gap-2"
		>
			<P>
				{{ i }}
			</P>
			<ScaleInputPoint
				:active="model === i"
				@click="model = i"
			/>
		</div>
	</div>
</template>
