<script setup lang="ts">
	import Header from "~/components/Header.vue";
	import Footer from "~/components/Footer.vue";
</script>

<template>
	<div class="min-h-screen">
		<Header />
		<div class="p-5 min-h-[calc(100vh-210px)]">
			<slot></slot>
		</div>
		<Footer />
	</div>
</template>
