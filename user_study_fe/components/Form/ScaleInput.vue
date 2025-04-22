<script setup lang="ts">
	import ScaleInputPoint from "./ScaleInputPoint.vue";
	import P from "../Basic/P.vue";

	defineProps<{
		range: number;
	}>();

	const model = defineModel<number>();
</script>

<template>
	<div class="flex gap-4 items-end justify-center font-bold text-xl flex-wrap">
		<div
			v-for="i in range"
			:key="i"
			class="flex flex-col items-center justify-center gap-2 flex-1"
		>
			<div class="flex flex-col items-center justify-center gap-2">
				<P>
					{{ i }}
				</P>
				<P class="text-sm md:text-lg text-center">
					<template v-if="i === 1">
						{{ $t("ScaleInput.1") }}
					</template>
					<template v-else-if="i === 2">
						{{ $t("ScaleInput.2") }}
					</template>
					<template v-else-if="i === 3">
						{{ $t("ScaleInput.3") }}
					</template>
					<template v-else-if="i === 4">
						{{ $t("ScaleInput.4") }}
					</template>
					<template v-else-if="i === 5">
						{{ $t("ScaleInput.5") }}
					</template>
				</P>
			</div>
			<ScaleInputPoint
				:active="model === i"
				@click="model = i"
			/>
		</div>
	</div>
</template>
