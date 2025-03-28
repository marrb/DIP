<script setup lang="ts">
	import Section from "./Basic/Section.vue";
	import P from "./Basic/P.vue";
</script>

<template>
	<footer class="border-t-2 border-white p-5 absolute bottom-0 w-full">
		<Section>
			<P class="text-gray-300 text-center">
				© {{ new Date().getFullYear() }} {{ $t("Footer") }}
			</P>
		</Section>
	</footer>
</template>
