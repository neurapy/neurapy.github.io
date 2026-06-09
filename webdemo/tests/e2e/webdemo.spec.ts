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

async function dragMainRegion(page: Page): Promise<void> {
  const box = await page.locator("#mainCanvas").boundingBox();
  if (!box) throw new Error("Missing main canvas bounds");
  await page.mouse.move(box.x + box.width * 0.28, box.y + box.height * 0.28);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.82, box.y + box.height * 0.82, { steps: 8 });
  await page.mouse.up();
}

async function clickMainPoint(page: Page): Promise<void> {
  const box = await page.locator("#mainCanvas").boundingBox();
  if (!box) throw new Error("Missing main canvas bounds");
  await page.mouse.click(box.x + box.width * 0.35, box.y + box.height * 0.65);
}

test("desktop renders two plots and continues background prefetching", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "desktop-only viewport assertions");
  const requests: string[] = [];
  page.on("request", (request) => requests.push(request.url()));

  await page.goto(FIXTURE_URL);

  await expect(page.locator(".plot-panel")).toHaveCount(2);
  await expect(page.locator("#globalPanel")).toHaveCount(0);
  await expect(page.locator("#mainTitle")).toHaveText("Model");
  await expect(page.locator("#mainRange")).toHaveText(/Prediction output/);
  await expect(page.locator("#trainTitle")).toHaveText("Train");
  await expect(page.locator("button[data-train-mode='local']")).toHaveClass(/active/);
  await expectNonblankCanvas(page, "#mainCanvas");
  await expectNonblankCanvas(page, "#trainCanvas");

  const mainBox = await page.locator(".model-panel").boundingBox();
  const trainBox = await page.locator("#trainPanel").boundingBox();
  const viewport = page.viewportSize();
  expect(mainBox?.y).toBeGreaterThanOrEqual(0);
  expect(trainBox?.y).toBeGreaterThanOrEqual(0);
  expect((mainBox?.y ?? 0) + (mainBox?.height ?? 0)).toBeLessThanOrEqual(viewport!.height + 2);
  expect((trainBox?.y ?? 0) + (trainBox?.height ?? 0)).toBeLessThanOrEqual(viewport!.height + 2);

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
  await expect(page.locator("#mainTitle")).toHaveText("Model");
  await expect(page.locator("#mainRange")).toHaveText(/Total loss/);

  await page.locator("button[data-sign='pos']").click();
  await expect.poll(() => requests.some((url) => url.includes("pos/chunks/0_indices.u16"))).toBe(true);
  await expect.poll(() => requests.some((url) => url.includes("pos/chunks/1_indices.u16"))).toBe(true);

  await page.locator("button[data-train-mode='global']").click();
  await expect(page.locator("button[data-train-mode='global']")).toHaveClass(/active/);
  await expect(page.locator("#trainRange")).toHaveText(/Global ·/);
  await expectNonblankCanvas(page, "#trainCanvas");
  await expect(page.locator("#globalCanvas")).toHaveCount(0);

  await page.locator("button[data-train-mode='local']").click();
  await expect(page.locator("button[data-train-mode='local']")).toHaveClass(/active/);
  await page.locator("button[data-mode='region']").click();
  await dragMainRegion(page);
  await expect(page.locator("#selectedPoint")).toHaveText(/x .* y /);
  await expect(page.locator("#candidateCount")).toHaveText(/[1-4] \/ 4/);
  await expect(page.locator("#trainRange")).toHaveText(/Local region · sum over [1-4] candidates/);
  await expectNonblankCanvas(page, "#trainCanvas");

  await page.locator("button[data-mode='point']").click();
  await clickMainPoint(page);
  await expect(page.locator("#selectedPoint")).toHaveText(/\(.+, .+\)/);
  await expect(page.locator("#trainRange")).toHaveText(/Local/);
  await expectNonblankCanvas(page, "#trainCanvas");
});

test("mobile keeps Model and Train visible in the first viewport", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "mobile", "mobile-only viewport assertions");
  await page.goto(FIXTURE_URL);

  await expect(page.locator(".plot-panel")).toHaveCount(2);
  await expect(page.locator("#globalPanel")).toHaveCount(0);
  await expectNonblankCanvas(page, "#mainCanvas");
  await expectNonblankCanvas(page, "#trainCanvas");

  const mainBox = await page.locator(".model-panel").boundingBox();
  const trainBox = await page.locator("#trainPanel").boundingBox();
  const viewport = page.viewportSize();
  expect((mainBox?.y ?? 0) + (mainBox?.height ?? 0)).toBeLessThanOrEqual(viewport!.height + 2);
  expect((trainBox?.y ?? 0) + (trainBox?.height ?? 0)).toBeLessThanOrEqual(viewport!.height + 2);

  await page.locator("button[data-train-mode='global']").click();
  await expect(page.locator("#trainPanel")).toBeVisible();
  await expect(page.locator("#trainRange")).toHaveText(/Global ·/);
  await expectNonblankCanvas(page, "#trainCanvas");
});
