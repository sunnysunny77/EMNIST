import { events } from "./utillites.js";

export const fallback = () => {

	const go_back = document.querySelector("#go-back");

	if ( !go_back ) {

		return;
	}

	events(go_back, "click", () => {

		history.back();
	});
};