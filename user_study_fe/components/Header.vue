<script setup lang="ts">
	import Section from "./Basic/Section.vue";
</script>

<template>
	<header class="p-5 border-b-2 border-white">
		<Section class="text-center flex-col md:flex-row md:text-3xl flex gap-4 items-center uppercase">
			<NuxtImg
				src="images/VUT-logo.png"
				loading="lazy"
				class="max-w-[100px] h-auto"
				width="100"
				height="100"
			/>
			Editing and Modifying Videos Created Using Diffusion Models Based on Text Input and User Interaction
		</Section>
	</header>
</template>
