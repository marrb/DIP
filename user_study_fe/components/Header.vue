<script setup lang="ts">
	import Section from "./Basic/Section.vue";
	import { faChevronDown } from "@fortawesome/free-solid-svg-icons";
	import Button from "./Basic/Button.vue";

	// Composables
	const { locale, locales, setLocale } = useI18n();

	// Refs
	const langDropdownActive = ref(false);
	const langDropdown = ref(null);

	// Methods
	onClickOutside(langDropdown, () => (langDropdownActive.value = false));
</script>

<template>
	<header class="p-5 border-b-2 border-white flex items-center gap-4">
		<div
			class="flex gap-1 items-center relative cursor-pointer"
			ref="langDropdown"
			@click.self="langDropdownActive = !langDropdownActive"
		>
			{{ locale.toUpperCase() }}
			<font-awesome
				:icon="faChevronDown"
				class="text-white text-sm"
				@click.self="langDropdownActive = !langDropdownActive"
			/>

			<div
				class="absolute top-7 left-0 bg-white text-primary p-2 flex flex-col gap-2 rounded-lg cursor-default"
				:class="langDropdownActive ? 'block' : 'hidden'"
			>
				<Button
					v-for="localeOption of locales"
					@click="
						setLocale(localeOption.code);
						langDropdownActive = false;
					"
					class="mr-auto hover:underline"
					:class="{ 'underline font-bold': localeOption.code === locale }"
				>
					{{ localeOption.name }}
				</Button>
			</div>
		</div>
		<Section class="text-center flex-col md:flex-row md:text-3xl flex gap-4 items-center uppercase">
			<NuxtImg
				src="images/VUT-logo.png"
				loading="lazy"
				class="max-w-[50px] md:max-w-[100px] h-auto"
				width="100"
				height="100"
			/>
			<span class="text-xs md:text-xl">{{ $t("Header") }}</span>
		</Section>
	</header>
</template>
