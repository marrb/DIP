<script setup lang="ts">
	import Header from "~/components/Header.vue";
	import Footer from "~/components/Footer.vue";
</script>

<template>
	<div class="min-h-screen">
		<Header />
		<div class="p-5">
			<slot></slot>
		</div>
		<Footer />
	</div>
</template>
