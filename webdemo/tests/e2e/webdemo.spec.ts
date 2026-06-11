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

async function axisFrameRatio(page: Page, selector: string): Promise<number> {
  return page.locator(`${selector} .axis-frame`).evaluate((frame) => {
    const width = Number(frame.getAttribute("width"));
    const height = Number(frame.getAttribute("height"));
    return width / height;
  });
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

async function selectedMarkerMetrics(page: Page): Promise<{
  width: number;
  height: number;
  count: number;
  dpr: number;
  scale: number;
  cssWidth: number;
  backingWidth: number;
  axisStrokeWidth: number;
} | null> {
  return page.locator("#mainCanvas").evaluate((canvas) => {
    const element = canvas as HTMLCanvasElement;
    const ctx = element.getContext("2d");
    const svg = document.querySelector<SVGSVGElement>("#mainSvg");
    if (!ctx || !svg || element.width === 0 || element.height === 0) return null;

    const data = ctx.getImageData(0, 0, element.width, element.height).data;
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let count = 0;
    for (let index = 0; index < data.length; index += 4) {
      const red = data[index];
      const green = data[index + 1];
      const blue = data[index + 2];
      const alpha = data[index + 3];
      const isSelectedMarker =
        red >= 232 &&
        red <= 248 &&
        green >= 170 &&
        green <= 200 &&
        blue >= 55 &&
        blue <= 90 &&
        alpha > 220;
      if (!isSelectedMarker) continue;
      const pixel = index / 4;
      const x = pixel % element.width;
      const y = Math.floor(pixel / element.width);
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
      count += 1;
    }
    if (!count) return null;

    const rect = element.getBoundingClientRect();
    const axisFrame = svg.querySelector<SVGRectElement>(".axis-frame");
    const axisStrokeWidth = axisFrame
      ? Number.parseFloat(getComputedStyle(axisFrame).strokeWidth)
      : Number.NaN;
    return {
      width: maxX - minX + 1,
      height: maxY - minY + 1,
      count,
      dpr: window.devicePixelRatio || 1,
      scale: Number(svg.style.getPropertyValue("--plot-visual-scale")) || 0,
      cssWidth: rect.width,
      backingWidth: element.width,
      axisStrokeWidth,
    };
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

async function clickMainPlotRatio(page: Page, xRatio: number, yRatio: number): Promise<void> {
  const point = await page.locator("#mainCanvas").evaluate(
    (canvas, ratios) => {
      const element = canvas as HTMLCanvasElement;
      const box = element.getBoundingClientRect();
      const axisFrame = document.querySelector<SVGRectElement>("#mainSvg .axis-frame");
      const frame = axisFrame
        ? {
            x: Number(axisFrame.getAttribute("x")),
            y: Number(axisFrame.getAttribute("y")),
            width: Number(axisFrame.getAttribute("width")),
            height: Number(axisFrame.getAttribute("height")),
          }
        : { x: 0, y: 0, width: box.width, height: box.height };
      return {
        x: box.left + frame.x + frame.width * ratios.xRatio,
        y: box.top + frame.y + frame.height * (1 - ratios.yRatio),
      };
    },
    { xRatio, yRatio },
  );
  await page.mouse.click(point.x, point.y);
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

async function expectTopbarControlsFit(page: Page): Promise<void> {
  const leaks = await page.locator(".topbar").evaluate((topbar) => {
    const topbarRect = topbar.getBoundingClientRect();
    const viewportRight = document.documentElement.clientWidth;
    const elements = topbar.querySelectorAll<HTMLElement>(
      ".topbar-controls, .topbar-control, select, .segmented, .icon-button",
    );
    return Array.from(elements)
      .filter((element) => {
        const style = getComputedStyle(element);
        const rect = element.getBoundingClientRect();
        return style.display !== "none" && rect.width > 0 && rect.height > 0;
      })
      .filter((element) => {
        const rect = element.getBoundingClientRect();
        return (
          rect.left < topbarRect.left - 1 ||
          rect.right > Math.min(topbarRect.right, viewportRight) + 1
        );
      })
      .map((element) => element.id || element.className);
  });
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
    const axisFrame = document.querySelector<SVGRectElement>("#mainSvg .axis-frame");
    const frame = axisFrame
      ? {
          x: Number(axisFrame.getAttribute("x")),
          y: Number(axisFrame.getAttribute("y")),
          width: Number(axisFrame.getAttribute("width")),
          height: Number(axisFrame.getAttribute("height")),
        }
      : { x: 0, y: 0, width: box.width, height: box.height };
    const point = (xRatio: number, yRatio: number): [number, number] => [
      box.left + frame.x + frame.width * xRatio,
      box.top + frame.y + frame.height * yRatio,
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

  await expect(page.locator("#problemSelect option")).toHaveText([
    "Fixture",
    "Shifted Fixture",
    "Drift Diffusion",
  ]);
  await expect(page.locator("button[data-model-quality='good']")).toHaveClass(/active/);
  await expect(page.locator("#runMeta")).toHaveText(/Fixture · Good · 4 candidate · 5 train/);
  await expectTopbarControlsFit(page);
  await expect(page.locator(".control-panel")).toHaveCount(0);
  await expect(page.locator(".control-group")).toHaveCount(0);
  await expect(page.locator(".plot-panel")).toHaveCount(2);
  await expect(page.locator("#globalPanel")).toHaveCount(0);
  await expect(page.locator("#mainTitle")).toHaveText("Model");
  await expect(page.locator("#mainRange")).toHaveText("");
  await expect(page.locator("#trainTitle")).toHaveText("Train");
  await expect(page.locator(".model-panel #fieldSelect")).toBeVisible();
  await expect(page.locator(".model-panel #fieldKindButtons")).toHaveCount(0);
  await expect(page.locator(".train-panel #trainModeButtons")).toHaveCount(0);
  await expect(page.locator("#summaryControl")).toHaveCount(0);
  await expect(page.locator(".train-panel #matrixSelect")).toHaveCount(1);
  await expect(page.locator(".train-panel #matrixSelect option")).toHaveText(["loss -> loss"]);
  await expect(page.locator(".train-panel #signButtons")).toHaveCount(1);
  await expect(page.locator(".train-panel #kSlider")).toHaveCount(1);
  await expectVisibleControlsInsidePanels(page);
  await expectNonblankCanvas(page, "#mainCanvas");
  await expectNonblankCanvas(page, "#trainCanvas");
  const goodMainSignature = await canvasSignature(page, "#mainCanvas");
  await page.locator("button[data-model-quality='bad']").click();
  await expect(page.locator("button[data-model-quality='bad']")).toHaveClass(/active/);
  await expect(page.locator("#runMeta")).toHaveText(/Fixture · Bad · 4 candidate · 5 train/);
  await expectNonblankCanvas(page, "#mainCanvas");
  await expectNonblankCanvas(page, "#trainCanvas");
  await expect.poll(() => canvasSignature(page, "#mainCanvas")).not.toBe(goodMainSignature);
  await expectTopbarControlsFit(page);
  await expectContourPaths(page, "model");
  await expectContourPaths(page, "train");
  await expect(page.locator(".model-panel .axis-label-x")).toHaveText("x");
  await expect(page.locator(".model-panel .axis-label-y")).toHaveText("y");
  await expect(page.locator(".train-panel .axis-label-x")).toHaveText("x");
  await expect(page.locator(".train-panel .axis-label-y")).toHaveText("y");
  await expect(page.locator(".model-panel .colorbar-frame")).toHaveCount(1);
  await expect(page.locator(".train-panel .colorbar-frame")).toHaveCount(1);
  await openControlsIfMenu(page, "train");
  await expect(page.locator("#mapControl")).toHaveCount(0);
  await expect(page.locator("#methodControl")).toHaveCount(0);
  await expect(page.locator("#influenceMapToggle")).toHaveCount(0);
  await expect(page.locator("#backgroundControl")).toBeVisible();
  await expect(page.locator("#backgroundButtons button")).toHaveText([
    "Points",
    "Linear",
    "Cell",
  ]);
  await expect(page.locator("button[data-background-mode='points']")).toHaveClass(/active/);
  await expect(page.locator("#kControl")).toBeVisible();
  await expect(page.locator("#kSlider")).toHaveAttribute("min", "0");
  await expect(page.locator("#kSlider")).toHaveAttribute("max", "256");
  await expect(page.locator("#kOutput")).toHaveText("25");
  await expect(page.locator("#trainRange")).toHaveText(/Local · Points( · max \|I\| .*)?/);
  await expect(page.locator("#trainRange")).not.toHaveText(/all exported influences/);
  await expectNonblankCanvas(page, "#trainCanvas");

  const backgroundSignatures: number[] = [await canvasSignature(page, "#trainCanvas")];
  for (const [mode, label] of [
    ["linear", "Linear"],
    ["cell", "Cell"],
  ] as const) {
    await page.locator(`button[data-background-mode='${mode}']`).click();
    await expect(page.locator(`button[data-background-mode='${mode}']`)).toHaveClass(/active/);
    await expect(page.locator("#kControl")).toBeVisible();
    await expect(page.locator("#trainRange")).toHaveText(
      new RegExp(`Local · ${label}( · max \\|I\\| .*)?`),
    );
    await expect(page.locator("#trainRange")).not.toHaveText(/all exported influences/);
    await expectNonblankCanvas(page, "#trainCanvas");
    await expect.poll(() => canvasSignature(page, "#trainCanvas")).not.toBe(backgroundSignatures.at(-1));
    backgroundSignatures.push(await canvasSignature(page, "#trainCanvas"));
  }
  expect(new Set(backgroundSignatures).size).toBe(backgroundSignatures.length);

  const absBackgroundSignature = backgroundSignatures.at(-1)!;
  await page.locator("button[data-sign='pos']").click();
  await expect.poll(() => canvasSignature(page, "#trainCanvas")).not.toBe(absBackgroundSignature);
  const posBackgroundSignature = await canvasSignature(page, "#trainCanvas");
  await page.locator("button[data-sign='neg']").click();
  await expect.poll(() => canvasSignature(page, "#trainCanvas")).not.toBe(posBackgroundSignature);
  const withTopKSignature = await canvasSignature(page, "#trainCanvas");
  await page.locator("#kSlider").fill("0");
  await expect(page.locator("#kOutput")).toHaveText("0");
  await expect.poll(() => canvasSignature(page, "#trainCanvas")).not.toBe(withTopKSignature);
  await expectVisibleControlsInsidePanels(page);

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

  await openControlsIfMenu(page, "model");
  await page.locator("#fieldSelect").selectOption("loss_total");
  await expect(page.locator("#mainTitle")).toHaveText("Model");
  await expect(page.locator("#mainRange")).toHaveText("");

  await openControlsIfMenu(page, "train");
  await page.locator("button[data-sign='pos']").click();
  await expect.poll(() => requests.some((url) => url.includes("pos/chunks/0_indices.u16"))).toBe(true);
  await expect.poll(() => requests.some((url) => url.includes("pos/chunks/1_indices.u16"))).toBe(true);

  await openControlsIfMenu(page, "train");
  await expect(page.locator("#summaryControl")).toHaveCount(0);
  await expect(page.locator("button[data-train-mode='global']")).toHaveCount(0);
  await expect(page.locator("#backgroundControl")).toBeVisible();
  await expect(page.locator("#methodControl")).toHaveCount(0);
  await expectVisibleControlsInsidePanels(page);
  await expectNonblankCanvas(page, "#trainCanvas");
  await expectContourPaths(page, "train");
  await expect(page.locator("#globalCanvas")).toHaveCount(0);

  await dragMainRegion(page);
  await expect(page.locator("#selectedPoint")).toHaveText(/x .* y /);
  await expect(page.locator("#trainRange")).toHaveText(/Local region · Cell · sum over [1-4] candidates/);
  await expectNonblankCanvas(page, "#trainCanvas");

  await clickMainPoint(page);
  await expect(page.locator("#selectedPoint")).toHaveText(/\(.+, .+\)/);
  await expect(page.locator("#trainRange")).toHaveText(/Local/);
  await expectNonblankCanvas(page, "#trainCanvas");
});

test("switching models and problems preserves comparison state", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "desktop-only state persistence assertions");
  await page.goto(FIXTURE_URL);

  await openControlsIfMenu(page, "model");
  await page.locator("#fieldSelect").selectOption("loss_total");
  await expect(page.locator("#fieldSelect")).toHaveValue("loss_total");
  await expect(page.locator("#mainRange")).toHaveText("");

  await openControlsIfMenu(page, "train");
  await page.locator("button[data-background-mode='cell']").click();
  await page.locator("button[data-sign='neg']").click();
  await page.locator("#kSlider").fill("7");
  await expect(page.locator("button[data-background-mode='cell']")).toHaveClass(/active/);
  await expect(page.locator("button[data-sign='neg']")).toHaveClass(/active/);
  await expect(page.locator("#kOutput")).toHaveText("7");

  await clickMainPlotRatio(page, 0.22, 0.78);
  await expect(page.locator("#selectedPoint")).toHaveText(/\(0.125, 0.875\)/);
  const selectedBeforeModelSwitch = await page.locator("#selectedPoint").textContent();

  await page.locator("button[data-model-quality='bad']").click();
  await expect(page.locator("#runMeta")).toHaveText(/Fixture · Bad · 4 candidate · 5 train/);
  await expect(page.locator("#fieldSelect")).toHaveValue("loss_total");
  await expect(page.locator("#mainRange")).toHaveText("");
  await expect(page.locator("button[data-background-mode='cell']")).toHaveClass(/active/);
  await expect(page.locator("button[data-sign='neg']")).toHaveClass(/active/);
  await expect(page.locator("#kOutput")).toHaveText("7");
  await expect(page.locator("#selectedPoint")).toHaveText(selectedBeforeModelSwitch ?? "");
  await expectNonblankCanvas(page, "#mainCanvas");
  await expectNonblankCanvas(page, "#trainCanvas");

  await page.locator("#problemSelect").selectOption("shifted_fixture");
  await expect(page.locator("#runMeta")).toHaveText(/Shifted Fixture · Bad · 4 candidate · 5 train/);
  await expect(page.locator("#fieldSelect")).toHaveValue("loss_residual");
  await expect(page.locator("#matrixSelect")).toHaveValue("m_shifted");
  await expect(page.locator("#mainRange")).toHaveText("");
  await expect(page.locator("button[data-background-mode='cell']")).toHaveClass(/active/);
  await expect(page.locator("button[data-sign='neg']")).toHaveClass(/active/);
  await expect(page.locator("#kOutput")).toHaveText("7");
  await expect(page.locator("#selectedPoint")).toHaveText(/\(11.25, 3.75\)/);
  await expectNonblankCanvas(page, "#mainCanvas");
  await expectNonblankCanvas(page, "#trainCanvas");
});

test("drift diffusion uses a compressed physical pi axis", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "desktop-only visual axis assertions");
  await page.goto(FIXTURE_URL);

  await page.locator("#problemSelect").selectOption("drift_diffusion");
  await expect(page.locator("#runMeta")).toHaveText(/Drift Diffusion · Good · 4 candidate · 5 train/);
  await expectNonblankCanvas(page, "#mainCanvas");
  await expectNonblankCanvas(page, "#trainCanvas");

  await expect.poll(() => axisFrameRatio(page, "#mainSvg")).toBeGreaterThan(1.9);
  await expect.poll(() => axisFrameRatio(page, "#mainSvg")).toBeLessThan(2.1);
  await expect(page.locator(".model-panel .axis-label-x")).toHaveText("x");
  await expect(page.locator(".model-panel .axis-label-y")).toHaveText("t");
  await expect(page.locator(".train-panel .axis-label-y")).toHaveText("t");
  await expect(page.locator(".model-panel .colorbar-frame")).toHaveCount(1);
  await expect(page.locator(".train-panel .colorbar-frame")).toHaveCount(1);

  const xTickLabels = await page.locator("#mainSvg .axis-x .tick text").allTextContents();
  expect(xTickLabels).toContain("0");
  expect(xTickLabels).toContain("π");
  expect(xTickLabels).toContain("2π");

  await clickMainPlotRatio(page, 0.96, 0.52);
  const selected = (await page.locator("#selectedPoint").textContent()) ?? "";
  expect(selected).toMatch(/\((5|6)/);
});

test("mobile keeps Model and Train visible in the first viewport", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "mobile", "mobile-only viewport assertions");
  await page.goto(FIXTURE_URL);

  await expect(page.locator("#problemSelect option")).toHaveText([
    "Fixture",
    "Shifted Fixture",
    "Drift Diffusion",
  ]);
  await expect(page.locator("button[data-model-quality='good']")).toHaveClass(/active/);
  await expectTopbarControlsFit(page);
  await expect(page.locator(".control-panel")).toHaveCount(0);
  await expect(page.locator(".control-group")).toHaveCount(0);
  await expect(page.locator(".plot-panel")).toHaveCount(2);
  await expect(page.locator("#globalPanel")).toHaveCount(0);
  await expect(page.locator(".train-panel #trainModeButtons")).toHaveCount(0);
  await expect(page.locator("#summaryControl")).toHaveCount(0);
  await openControlsIfMenu(page, "train");
  await expect(page.locator("#mapControl")).toHaveCount(0);
  await expect(page.locator("#methodControl")).toHaveCount(0);
  await expect(page.locator("#backgroundControl")).toBeVisible();
  await expect(page.locator("#kControl")).toBeVisible();
  await expect(page.locator("#backgroundButtons button")).toHaveText([
    "Points",
    "Linear",
    "Cell",
  ]);
  for (const mode of ["cell", "linear", "points"]) {
    await page.locator(`button[data-background-mode='${mode}']`).click();
    await expectNonblankCanvas(page, "#trainCanvas");
  }
  await expectNonblankCanvas(page, "#trainCanvas");
  await expectVisibleControlsInsidePanels(page);
  await expectTopbarControlsFit(page);
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
  await expect(page.locator("#trainRange")).toHaveText(/Local region · Points · sum over [1-4] candidates/);
  await expectNonblankCanvas(page, "#trainCanvas");
});

test("settings use wrapped in-panel bars when there is tile space", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "desktop-only viewport assertions");
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.goto(FIXTURE_URL);

  await expectTopbarControlsFit(page);
  await expect(page.locator(".model-actions")).toHaveAttribute("data-control-layout", /bar|inline/);
  await expect(page.locator(".train-actions")).toHaveAttribute("data-control-layout", /bar|inline/);
  await expectVisibleControlsInsidePanels(page);

  await page.setViewportSize({ width: 390, height: 900 });
  await openControlsIfMenu(page, "model");
  await openControlsIfMenu(page, "train");
  await expectTopbarControlsFit(page);
  await expectVisibleControlsInsidePanels(page);
});

test("high-DPI rendering keeps point and line overlays proportional to the plot", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-hidpi", "high-DPI-only viewport assertions");
  await page.goto(FIXTURE_URL);
  await expectNonblankCanvas(page, "#mainCanvas");
  await expect.poll(async () => (await selectedMarkerMetrics(page))?.count ?? 0).toBeGreaterThan(0);

  const metrics = await selectedMarkerMetrics(page);
  expect(metrics).not.toBeNull();
  expect(metrics!.dpr).toBe(2);
  expect(metrics!.backingWidth).toBe(Math.round(metrics!.cssWidth * metrics!.dpr));
  expect(metrics!.scale).toBeGreaterThan(0);
  expect(metrics!.scale).toBeLessThan(0.75);
  expect(metrics!.axisStrokeWidth).toBeCloseTo(metrics!.scale, 1);

  const expectedYellowInterior = 12 * metrics!.scale * metrics!.dpr;
  expect(metrics!.width).toBeGreaterThan(expectedYellowInterior * 0.65);
  expect(metrics!.width).toBeLessThan(expectedYellowInterior * 1.5);
  expect(metrics!.height).toBeGreaterThan(expectedYellowInterior * 0.65);
  expect(metrics!.height).toBeLessThan(expectedYellowInterior * 1.5);
});
