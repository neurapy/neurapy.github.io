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

async function expectContourPaths(page: Page, panel: "model" | "train"): Promise<void> {
  await expect
    .poll(() => page.locator(`.${panel}-panel .contours path`).count(), { timeout: 10_000 })
    .toBeGreaterThan(0);
}

async function canvasSignature(page: Page, selector: string): Promise<number> {
  return page.locator(selector).evaluate((canvas) => {
    const element = canvas as HTMLCanvasElement;
    const ctx = element.getContext("2d");
    if (!ctx || element.width === 0 || element.height === 0) return 0;
    const data = ctx.getImageData(0, 0, element.width, element.height).data;
    let hash = 2166136261;
    const stride = Math.max(4, Math.floor(data.length / 4096) * 4);
    for (let index = 0; index < data.length; index += stride) {
      hash ^= data[index] + (data[index + 1] << 8) + (data[index + 2] << 16) + (data[index + 3] << 24);
      hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
  });
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

async function openControlsIfMenu(page: Page, panel: "model" | "train"): Promise<void> {
  const actions = page.locator(`.${panel}-actions`);
  const layout = await actions.getAttribute("data-control-layout");
  const open = await actions.getAttribute("data-open");
  if (layout === "menu" && open !== "true") {
    await page.locator(`#${panel}MenuButton`).click();
  }
}

async function expectVisibleControlsInsidePanels(page: Page): Promise<void> {
  const leaks = await page.locator(".plot-panel").evaluateAll((panels) =>
    panels.flatMap((panel) => {
      const panelRect = panel.getBoundingClientRect();
      const elements = panel.querySelectorAll<HTMLElement>(
        ".plot-actions, .plot-menu, .plot-control, select, .segmented, .menu-button",
      );
      return Array.from(elements)
        .filter((element) => {
          const style = getComputedStyle(element);
          const rect = element.getBoundingClientRect();
          return style.display !== "none" && rect.width > 0 && rect.height > 0;
        })
        .filter((element) => {
          const rect = element.getBoundingClientRect();
          return rect.left < panelRect.left - 1 || rect.right > panelRect.right + 1;
        })
        .map((element) => `${panel.id || panel.className}:${element.id || element.className}`);
    }),
  );
  expect(leaks).toEqual([]);
}

async function tapMainPoint(page: Page): Promise<void> {
  const box = await page.locator("#mainCanvas").boundingBox();
  if (!box) throw new Error("Missing main canvas bounds");
  await page.touchscreen.tap(box.x + box.width * 0.68, box.y + box.height * 0.36);
}

async function touchDoubleTapThenDragRegion(page: Page): Promise<void> {
  await page.locator("#mainCanvas").evaluate((canvas) => {
    const element = canvas as HTMLCanvasElement;
    const box = element.getBoundingClientRect();
    const point = (xRatio: number, yRatio: number): [number, number] => [
      box.left + box.width * xRatio,
      box.top + box.height * yRatio,
    ];
    const fire = (type: string, pointValue: [number, number], pointerId: number) => {
      element.dispatchEvent(
        new PointerEvent(type, {
          bubbles: true,
          cancelable: true,
          composed: true,
          pointerId,
          pointerType: "touch",
          isPrimary: true,
          clientX: pointValue[0],
          clientY: pointValue[1],
          button: 0,
          buttons: type === "pointerup" || type === "pointercancel" ? 0 : 1,
        }),
      );
    };
    const tap = (pointValue: [number, number], pointerId: number) => {
      fire("pointerdown", pointValue, pointerId);
      fire("pointerup", pointValue, pointerId);
    };

    const tapPoint = point(0.44, 0.44);
    tap(tapPoint, 21);
    tap(tapPoint, 22);

    const start = point(0.22, 0.24);
    const mid = point(0.52, 0.56);
    const end = point(0.82, 0.82);
    fire("pointerdown", start, 23);
    fire("pointermove", mid, 23);
    fire("pointermove", end, 23);
    fire("pointerup", end, 23);
  });
}

test("desktop renders two plots and continues background prefetching", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "desktop-only viewport assertions");
  const requests: string[] = [];
  page.on("request", (request) => requests.push(request.url()));

  await page.goto(FIXTURE_URL);

  await expect(page.locator(".control-panel")).toHaveCount(0);
  await expect(page.locator(".control-group")).toHaveCount(0);
  await expect(page.locator(".plot-panel")).toHaveCount(2);
  await expect(page.locator("#globalPanel")).toHaveCount(0);
  await expect(page.locator("#mainTitle")).toHaveText("Model");
  await expect(page.locator("#mainRange")).toHaveText(/Prediction output/);
  await expect(page.locator("#trainTitle")).toHaveText("Train");
  await expect(page.locator(".model-panel #fieldSelect")).toBeVisible();
  await expect(page.locator(".model-panel #fieldKindButtons")).toBeVisible();
  await expect(page.locator(".train-panel #trainModeButtons")).toBeVisible();
  await expect(page.locator(".train-panel #matrixSelect")).toHaveCount(1);
  await expect(page.locator(".train-panel #signButtons")).toHaveCount(1);
  await expect(page.locator(".train-panel #kSlider")).toHaveCount(1);
  await expectVisibleControlsInsidePanels(page);
  await expect(page.locator("button[data-train-mode='local']")).toHaveClass(/active/);
  await expectNonblankCanvas(page, "#mainCanvas");
  await expectNonblankCanvas(page, "#trainCanvas");
  await expectContourPaths(page, "model");
  await expectContourPaths(page, "train");
  await openControlsIfMenu(page, "train");
  await expect(page.locator("#mapControl")).toBeVisible();
  await expect(page.locator("#methodControl")).toBeHidden();
  await expect(page.locator("#influenceMapToggle")).not.toBeChecked();
  const pointSignature = await canvasSignature(page, "#trainCanvas");
  await page.locator("#influenceMapToggle").check();
  await expect(page.locator("#influenceMapToggle")).toBeChecked();
  await expect(page.locator("#methodControl")).toBeVisible();
  await expect(page.locator("#influenceMapMethodSelect")).toHaveValue("linear");
  await expect(page.locator("#influenceMapMethodSelect option")).toHaveText([
    "Linear",
    "Cells",
  ]);
  await expect(page.locator("#kControl")).toBeHidden();
  await expect(page.locator("#trainRange")).toHaveText(/Local map · Linear · all exported influences \(\d+\)/);
  await expectNonblankCanvas(page, "#trainCanvas");
  await expect.poll(() => canvasSignature(page, "#trainCanvas")).not.toBe(pointSignature);

  const methodSignatures: number[] = [await canvasSignature(page, "#trainCanvas")];
  for (const [method, label] of [
    ["cells", "Cells"],
  ] as const) {
    await page.locator("#influenceMapMethodSelect").selectOption(method);
    await expect(page.locator("#trainRange")).toHaveText(
      new RegExp(`Local map · ${label} · all exported influences \\(\\d+\\)`),
    );
    await expectNonblankCanvas(page, "#trainCanvas");
    await expect.poll(() => canvasSignature(page, "#trainCanvas")).not.toBe(methodSignatures.at(-1));
    methodSignatures.push(await canvasSignature(page, "#trainCanvas"));
  }
  expect(new Set(methodSignatures).size).toBe(methodSignatures.length);

  const absMapSignature = methodSignatures.at(-1)!;
  await page.locator("button[data-sign='pos']").click();
  await expect.poll(() => canvasSignature(page, "#trainCanvas")).not.toBe(absMapSignature);
  const posMapSignature = await canvasSignature(page, "#trainCanvas");
  await page.locator("button[data-sign='neg']").click();
  await expect.poll(() => canvasSignature(page, "#trainCanvas")).not.toBe(posMapSignature);
  await expectVisibleControlsInsidePanels(page);
  await page.locator("#influenceMapToggle").uncheck();
  await expect(page.locator("#kControl")).toBeVisible();
  await expect(page.locator("#methodControl")).toBeHidden();
  await expect(page.locator("#trainRange")).toHaveText(/^Local/);

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

  await openControlsIfMenu(page, "train");
  await page.locator("button[data-sign='pos']").click();
  await expect.poll(() => requests.some((url) => url.includes("pos/chunks/0_indices.u16"))).toBe(true);
  await expect.poll(() => requests.some((url) => url.includes("pos/chunks/1_indices.u16"))).toBe(true);

  await expect(page.locator("#summaryControl")).toBeHidden();
  await page.locator("button[data-train-mode='global']").click();
  await expect(page.locator("button[data-train-mode='global']")).toHaveClass(/active/);
  await openControlsIfMenu(page, "train");
  await expect(page.locator("#summaryControl")).toBeVisible();
  await expect(page.locator("#methodControl")).toBeHidden();
  await expectVisibleControlsInsidePanels(page);
  await expect(page.locator("#trainRange")).toHaveText(/Global ·/);
  await expectNonblankCanvas(page, "#trainCanvas");
  await expectContourPaths(page, "train");
  await expect(page.locator("#globalCanvas")).toHaveCount(0);

  await page.locator("button[data-train-mode='local']").click();
  await expect(page.locator("button[data-train-mode='local']")).toHaveClass(/active/);
  await openControlsIfMenu(page, "train");
  await expect(page.locator("#summaryControl")).toBeHidden();
  await expect(page.locator("#methodControl")).toBeHidden();
  await dragMainRegion(page);
  await expect(page.locator("#selectedPoint")).toHaveText(/x .* y /);
  await expect(page.locator("#trainRange")).toHaveText(/Local region · sum over [1-4] candidates/);
  await expectNonblankCanvas(page, "#trainCanvas");

  await clickMainPoint(page);
  await expect(page.locator("#selectedPoint")).toHaveText(/\(.+, .+\)/);
  await expect(page.locator("#trainRange")).toHaveText(/Local/);
  await expectNonblankCanvas(page, "#trainCanvas");
});

test("mobile keeps Model and Train visible in the first viewport", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "mobile", "mobile-only viewport assertions");
  await page.goto(FIXTURE_URL);

  await expect(page.locator(".control-panel")).toHaveCount(0);
  await expect(page.locator(".control-group")).toHaveCount(0);
  await expect(page.locator(".plot-panel")).toHaveCount(2);
  await expect(page.locator("#globalPanel")).toHaveCount(0);
  await openControlsIfMenu(page, "train");
  await expect(page.locator("#mapControl")).toBeVisible();
  await expect(page.locator("#methodControl")).toBeHidden();
  await page.locator("#influenceMapToggle").check();
  await expect(page.locator("#methodControl")).toBeVisible();
  await expect(page.locator("#trainRange")).toHaveText(/Local map · Linear · all exported influences \(\d+\)/);
  await expect(page.locator("#influenceMapMethodSelect option")).toHaveText([
    "Linear",
    "Cells",
  ]);
  for (const method of ["cells", "linear"]) {
    await page.locator("#influenceMapMethodSelect").selectOption(method);
    await expectNonblankCanvas(page, "#trainCanvas");
  }
  await expectNonblankCanvas(page, "#trainCanvas");
  await expectVisibleControlsInsidePanels(page);
  await page.locator("#influenceMapToggle").uncheck();
  await expect(page.locator("#methodControl")).toBeHidden();
  await expectVisibleControlsInsidePanels(page);
  await expectNonblankCanvas(page, "#mainCanvas");
  await expectNonblankCanvas(page, "#trainCanvas");

  const mainBox = await page.locator(".model-panel").boundingBox();
  const trainBox = await page.locator("#trainPanel").boundingBox();
  const viewport = page.viewportSize();
  expect((mainBox?.y ?? 0) + (mainBox?.height ?? 0)).toBeLessThanOrEqual(viewport!.height + 2);
  expect((trainBox?.y ?? 0) + (trainBox?.height ?? 0)).toBeLessThanOrEqual(viewport!.height + 2);

  await tapMainPoint(page);
  await expect(page.locator("#selectedPoint")).toHaveText(/\(.+, .+\)/);

  await touchDoubleTapThenDragRegion(page);
  await expect(page.locator("#selectedPoint")).toHaveText(/x .* y /);
  await expect(page.locator("#trainRange")).toHaveText(/Local region · sum over [1-4] candidates/);
  await expectNonblankCanvas(page, "#trainCanvas");

  await page.locator("button[data-train-mode='global']").click();
  await openControlsIfMenu(page, "train");
  await expect(page.locator("#summaryControl")).toBeVisible();
  await expect(page.locator("#methodControl")).toBeHidden();
  await expectVisibleControlsInsidePanels(page);
  await expect(page.locator("#trainPanel")).toBeVisible();
  await expect(page.locator("#trainRange")).toHaveText(/Global ·/);
  await expectNonblankCanvas(page, "#trainCanvas");
});

test("settings use wrapped in-panel bars when there is tile space", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "desktop-only viewport assertions");
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.goto(FIXTURE_URL);

  await expect(page.locator(".model-actions")).toHaveAttribute("data-control-layout", /bar|inline/);
  await expect(page.locator(".train-actions")).toHaveAttribute("data-control-layout", /bar|inline/);
  await expectVisibleControlsInsidePanels(page);

  await page.setViewportSize({ width: 390, height: 900 });
  await openControlsIfMenu(page, "model");
  await openControlsIfMenu(page, "train");
  await expectVisibleControlsInsidePanels(page);
});
