export function useDateFormatter() {
	const formatDate = (timestamp: string | number, locale: string = "en-EN") => {
		const date = new Date(timestamp);

		return new Intl.DateTimeFormat(locale, {
			year: "numeric",
			month: "2-digit",
			day: "2-digit",
			hour: "2-digit",
			minute: "2-digit",
			timeZone: "UTC",
		}).format(date);
	}

	return { formatDate };
}