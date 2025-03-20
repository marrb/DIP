import tailwindcss from "@tailwindcss/vite";

// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
	compatibilityDate: "2024-11-01",
	css: ["~/assets/css/main.css"],
	vite: {
		plugins: [tailwindcss()],
		build: {
			minify: true,
		},
	},
	nitro: {
		preset: "node-server",
		compressPublicAssets: true,
	},
	devtools: { enabled: true },
	ssr: true,
});
