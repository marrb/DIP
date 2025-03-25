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
								preset: "node-server",
								compressPublicAssets: true,
				},
				devtools: { enabled: true },
				ssr: true,
				modules: [
				 "nuxt-vuefire",
				 "@nuxt/fonts",
				 "@nuxt/image",
				 "@vesp/nuxt-fontawesome",
				],
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
				image: {
								provider: "ipx",
								quality: 80,
								formats: ["webp"],
								ipx: {
												maxAge: 60 * 60 * 24 * 365,
								},
				},
});