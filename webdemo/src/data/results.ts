import type { ResultsData } from "../types";
import { fetchJson } from "./manifest";

export function resolveResultsUrl(indexUrl: URL): URL {
  return new URL("results/index.json", indexUrl);
}

export async function loadResultsData(indexUrl: URL, signal?: AbortSignal): Promise<ResultsData> {
  return assertResultsData(await fetchJson<ResultsData>(resolveResultsUrl(indexUrl), signal));
}

export function assertResultsData(data: ResultsData): ResultsData {
  if (data.schema_version !== 1) {
    throw new Error(`Unsupported results schema ${String(data.schema_version)}; expected 1`);
  }
  if (!Array.isArray(data.loss_decompositions)) {
    throw new Error("Results data is missing loss decomposition summaries");
  }
  if (
    !data.indicators ||
    !Array.isArray(data.indicators.temporal) ||
    !Array.isArray(data.indicators.directionality)
  ) {
    throw new Error("Results data is missing indicator summaries");
  }
  return data;
}
