import { expect, test, type Page } from "@playwright/test";

const FIXTURE_URL = "/?data=fixtures/tiny-data/index.json";

async function canvasIsNonblank(page: Page, selector: string): Promise<boolean> {
  return page.locator(selector).evaluate((canvas) => {
    const element = canvas as HTMLCanvasElement;
    const ctx = element.getContext("2d");
    if (!ctx || element.width === 0 || element.height === 0) return false;
    const sample = ctx.getImageData(0, 0, element.width, element.height).data;
    for (let index = 0; index < sample.length; index += 4) {
      if (sample[index] !== 251 || sample[index + 1] !== 252 || sample[index + 2] !== 253) {
        return true;
      }
    }
    return false;
  });
}

async function expectNonblankCanvas(page: Page, selector: string): Promise<void> {
  await expect.poll(() => canvasIsNonblank(page, selector), { timeout: 10_000 }).toBe(true);
}

test("desktop renders important plots and continues background prefetching", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "desktop-only viewport assertions");
  const requests: string[] = [];
  page.on("request", (request) => requests.push(request.url()));

  await page.goto(FIXTURE_URL);

  await expect(page.locator("#mainTitle")).toHaveText(/Prediction/);
  await expectNonblankCanvas(page, "#mainCanvas");
  await expectNonblankCanvas(page, "#influenceCanvas");
  await expectNonblankCanvas(page, "#globalCanvas");

  const mainBox = await page.locator(".plot-panel.primary").boundingBox();
  const localBox = await page.locator("#localPanel").boundingBox();
  const globalBox = await page.locator("#globalPanel").boundingBox();
  const viewport = page.viewportSize();
  expect(mainBox?.y).toBeGreaterThanOrEqual(0);
  expect(localBox?.y).toBeGreaterThanOrEqual(0);
  expect(globalBox?.y).toBeGreaterThanOrEqual(0);
  expect((mainBox?.y ?? 0) + (mainBox?.height ?? 0)).toBeLessThanOrEqual(viewport!.height + 2);
  expect((globalBox?.y ?? 0) + (globalBox?.height ?? 0)).toBeLessThanOrEqual(viewport!.height + 2);

  const predIndex = requests.findIndex((url) => url.includes("pred_output_0_raster.u16"));
  const lossIndex = requests.findIndex((url) => url.includes("loss_total_raster.u16"));
  const absChunk0Index = requests.findIndex((url) => url.includes("abs/chunks/0_indices.u16"));
  const absChunk1Index = requests.findIndex((url) => url.includes("abs/chunks/1_indices.u16"));
  expect(predIndex).toBeGreaterThanOrEqual(0);
  expect(absChunk0Index).toBeGreaterThanOrEqual(0);
  if (lossIndex >= 0) expect(predIndex).toBeLessThan(lossIndex);
  if (absChunk1Index >= 0) expect(absChunk0Index).toBeLessThan(absChunk1Index);

  await expect.poll(() => requests.some((url) => url.includes("loss_total_raster.u16"))).toBe(true);
  await expect.poll(() => requests.some((url) => url.includes("abs/chunks/1_indices.u16"))).toBe(true);

  await page.locator("button[data-kind='loss']").click();
  await expect(page.locator("#mainTitle")).toHaveText(/Total loss/);

  await page.locator("button[data-sign='pos']").click();
  await expect.poll(() => requests.some((url) => url.includes("pos/chunks/0_indices.u16"))).toBe(true);
  await expect.poll(() => requests.some((url) => url.includes("pos/chunks/1_indices.u16"))).toBe(true);
});

test("mobile keeps main and selected secondary plot in the first viewport", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "mobile", "mobile-only viewport assertions");
  await page.goto(FIXTURE_URL);

  await expectNonblankCanvas(page, "#mainCanvas");
  await expectNonblankCanvas(page, "#influenceCanvas");
  await expect(page.locator("#globalPanel")).toBeHidden();

  const mainBox = await page.locator(".plot-panel.primary").boundingBox();
  const localBox = await page.locator("#localPanel").boundingBox();
  const viewport = page.viewportSize();
  expect((mainBox?.y ?? 0) + (mainBox?.height ?? 0)).toBeLessThanOrEqual(viewport!.height + 2);
  expect((localBox?.y ?? 0) + (localBox?.height ?? 0)).toBeLessThanOrEqual(viewport!.height + 2);

  await page.locator("button[data-tab='global']").click();
  await expect(page.locator("#localPanel")).toBeHidden();
  await expect(page.locator("#globalPanel")).toBeVisible();
  await expectNonblankCanvas(page, "#globalCanvas");
});
