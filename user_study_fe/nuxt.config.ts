import tailwindcss from "@tailwindcss/vite";

// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
	compatibilityDate: "2024-11-01",
	css: ["~/assets/css/main.css"],
	plugins: ["~/plugins/firebase.client.ts"],
	vite: {
		plugins: [tailwindcss()],
		build: {
			minify: true,
		},
	},
	nitro: {
		preset: "firebase",
		firebase: {
			gen: 2,
			nodeVersion: "20",
		},
		compressPublicAssets: true,
	},
	devtools: { enabled: true },
	ssr: true,
	modules: ["nuxt-vuefire", "@nuxt/fonts", "@vesp/nuxt-fontawesome", "@nuxtjs/i18n", "@vueuse/nuxt"],
	vuefire: {
		config: {
			apiKey: process.env.FB_API_KEY,
			authDomain: process.env.FB_AUTH_DOMAIN,
			projectId: process.env.FB_PROJECT_ID,
			storageBucket: process.env.FB_STORAGE_BUCKET,
			messagingSenderId: process.env.FB_MESSAGING_SENDER_ID,
			appId: process.env.FB_APP_ID,
			measurementId: process.env.FB_MEASUREMENT_ID,
		},
	},
	runtimeConfig: {
		public: {
			deadline: "2025-04-30T23:59Z",
		},
	},
	i18n: {
		locales: [
			{ code: "en", language: "en-US", file: "en.json", name: "English" },
			{ code: "sk", language: "sk-SK", file: "sk.json", name: "Slovenčina" },
		],
		defaultLocale: "en",
		strategy: "no_prefix",
	},
});
