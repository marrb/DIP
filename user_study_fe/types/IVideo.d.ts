import type { EModel } from "./enums/EModel";
import fs from "fs";
import path from "path";

declare interface IVideo {
	id: string;
	originalName: string;
	url: string;
	model: EModel;
	originalVideoName?: string;
}
