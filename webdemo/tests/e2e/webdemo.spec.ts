import { expect, test, type Page } from "@playwright/test";

const FIXTURE_URL = "/?data=fixtures/tiny-data/index.json";

async function canvasIsNonblank(page: Page, selector: string): Promise<boolean> {
  return page.locator(selector).evaluate((canvas) => {
    const element = canvas as HTMLCanvasElement;
    const ctx = element.getContext("2d");
    if (!ctx || element.width === 0 || element.height === 0) return false;
    const sample = ctx.getImageData(0, 0, element.width, element.height).data;
    for (let index = 0; index < sample.length; index += 4) {
      if (sample[index] !== 246 || sample[index + 1] !== 249 || sample[index + 2] !== 252) {
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

async function waitForIndicatorsCharts(page: Page): Promise<void> {
  await expect
    .poll(() => page.locator(".results-loss-chart .loss-fraction-area").count(), { timeout: 10_000 })
    .toBeGreaterThan(0);
  await expect
    .poll(() => page.locator('[data-results-chart="ic"] .results-ic-line').count(), { timeout: 10_000 })
    .toBeGreaterThan(0);
}

async function expectNoDocumentHorizontalOverflow(page: Page): Promise<void> {
  const overflow = await page.evaluate(() => {
    const bodyWidth = document.body?.scrollWidth ?? 0;
    return Math.max(document.documentElement.scrollWidth, bodyWidth) - document.documentElement.clientWidth;
  });
  expect(overflow).toBeLessThanOrEqual(1);
}

async function openIndicatorsDashboard(page: Page, problem = "burgers"): Promise<void> {
  await page.goto(FIXTURE_URL);
  await page.locator("button[data-app-view='indicators']").click();
  await page.locator("#problemSelect").selectOption(problem);
  await expect(page.locator("#resultsWorkspace")).toBeVisible();
  await waitForIndicatorsCharts(page);
}

async function resultsDashboardLayoutMetrics(page: Page): Promise<{
  workspace: { clientHeight: number; scrollHeight: number };
  grid: { top: number; right: number; bottom: number; left: number; width: number; height: number };
  loss: { top: number; right: number; bottom: number; left: number; width: number; height: number };
  ic: { top: number; right: number; bottom: number; left: number; width: number; height: number };
  indicator: { top: number; right: number; bottom: number; left: number; width: number; height: number };
}> {
  return page.evaluate(() => {
    const workspace = document.querySelector<HTMLElement>("#resultsWorkspace");
    const grid = document.querySelector<HTMLElement>("#resultsWorkspace .results-grid");
    const loss = document.querySelector<SVGSVGElement>(".results-loss-chart")?.closest<HTMLElement>(".results-panel");
    const ic = document.querySelector<SVGSVGElement>(".results-ic-chart")?.closest<HTMLElement>(".results-panel");
    const indicator = document
      .querySelector<HTMLElement>(".results-indicator-table")
      ?.closest<HTMLElement>(".results-panel");
    if (!workspace || !grid || !loss || !ic || !indicator) throw new Error("Missing results dashboard layout nodes");

    const rect = (element: Element) => {
      const box = element.getBoundingClientRect();
      return {
        top: box.top,
        right: box.right,
        bottom: box.bottom,
        left: box.left,
        width: box.width,
        height: box.height,
      };
    };

    return {
      workspace: {
        clientHeight: workspace.clientHeight,
        scrollHeight: workspace.scrollHeight,
      },
      grid: rect(grid),
      loss: rect(loss),
      ic: rect(ic),
      indicator: rect(indicator),
    };
  });
}

async function axisFrameRatio(page: Page, selector: string): Promise<number> {
  return page.locator(`${selector} .axis-frame`).evaluate((frame) => {
    const width = Number(frame.getAttribute("width"));
    const height = Number(frame.getAttribute("height"));
    return width / height;
  });
}

async function plotFrameMetrics(
  page: Page,
  panelSelector: string,
  svgSelector: string,
): Promise<{
  bodyWidth: number;
  frameWidth: number;
  marginLeft: number;
  marginRight: number;
  colorbars: number;
}> {
  return page.locator(panelSelector).evaluate((panel, selector) => {
    const body = panel.querySelector<HTMLElement>(".plot-body");
    const svg = document.querySelector<SVGSVGElement>(selector);
    const frame = svg?.querySelector<SVGRectElement>(".axis-frame");
    if (!body || !svg || !frame) throw new Error("Missing plot frame");
    const bodyBox = body.getBoundingClientRect();
    const svgBox = svg.getBoundingClientRect();
    const x = Number(frame.getAttribute("x"));
    const width = Number(frame.getAttribute("width"));
    return {
      bodyWidth: bodyBox.width,
      frameWidth: width,
      marginLeft: x,
      marginRight: svgBox.width - x - width,
      colorbars: svg.querySelectorAll(".colorbar-frame").length,
    };
  }, svgSelector);
}

async function expectModelTrainFramesHorizontallyAligned(page: Page): Promise<void> {
  const metrics = await page.evaluate(() => {
    const frameMetrics = (selector: string) => {
      const frame = document.querySelector<SVGRectElement>(`${selector} .axis-frame`);
      if (!frame) throw new Error(`Missing axis frame for ${selector}`);
      const x = Number(frame.getAttribute("x"));
      const width = Number(frame.getAttribute("width"));
      return { x, width, right: x + width };
    };
    return {
      model: frameMetrics("#mainSvg"),
      train: frameMetrics("#trainSvg"),
    };
  });

  expect(Math.abs(metrics.train.x - metrics.model.x)).toBeLessThanOrEqual(1);
  expect(Math.abs(metrics.train.width - metrics.model.width)).toBeLessThanOrEqual(1);
  expect(Math.abs(metrics.train.right - metrics.model.right)).toBeLessThanOrEqual(1);
}

async function expectCompactOutsideDecorations(page: Page, selector: string): Promise<void> {
  const metrics = await page.locator(selector).evaluate((svg) => {
    const root = svg as SVGSVGElement;
    const box = root.getBoundingClientRect();
    const frame = root.querySelector<SVGRectElement>(".axis-frame");
    if (!frame) throw new Error("Missing axis frame");
    const x = Number(frame.getAttribute("x"));
    const y = Number(frame.getAttribute("y"));
    const width = Number(frame.getAttribute("width"));
    const height = Number(frame.getAttribute("height"));
    const frameBox = { x, y, right: x + width, bottom: y + height };
    const rect = (element: Element | null) => {
      if (!element) return null;
      const elementBox = element.getBoundingClientRect();
      return {
        x: elementBox.left - box.left,
        y: elementBox.top - box.top,
        right: elementBox.right - box.left,
        bottom: elementBox.bottom - box.top,
      };
    };
    return {
      box: { width: box.width, height: box.height },
      frame: frameBox,
      margins: {
        left: x,
        right: box.width - x - width,
        top: y,
        bottom: box.height - y - height,
      },
      colorbar: rect(root.querySelector(".colorbar-frame")),
      xLabel: rect(root.querySelector(".axis-label-x")),
      yLabel: rect(root.querySelector(".axis-label-y")),
    };
  });

  expect(metrics.margins.left).toBeGreaterThan(12);
  expect(metrics.margins.right).toBeGreaterThan(6);
  expect(metrics.margins.bottom).toBeGreaterThan(20);
  expect(metrics.colorbar).not.toBeNull();
  const colorbarIntersectsFrame =
    metrics.colorbar!.x < metrics.frame.right - 1 &&
    metrics.colorbar!.right > metrics.frame.x + 1 &&
    metrics.colorbar!.y < metrics.frame.bottom - 1 &&
    metrics.colorbar!.bottom > metrics.frame.y + 1;
  const colorbarInRightGutter = metrics.colorbar!.x >= metrics.frame.right + 1;
  const colorbarInBottomGutter = metrics.colorbar!.y >= metrics.frame.bottom + 1;
  expect(colorbarIntersectsFrame).toBe(false);
  expect(colorbarInRightGutter || colorbarInBottomGutter).toBe(true);
  expect(metrics.colorbar!.right).toBeLessThanOrEqual(metrics.box.width + 1);
  expect(metrics.colorbar!.bottom).toBeLessThanOrEqual(metrics.box.height + 1);
  expect(metrics.xLabel).not.toBeNull();
  expect(metrics.xLabel!.y).toBeGreaterThanOrEqual(metrics.frame.bottom - 1);
  expect(metrics.xLabel!.bottom).toBeLessThanOrEqual(metrics.box.height + 1);
  expect(metrics.yLabel).not.toBeNull();
  expect(metrics.yLabel!.right).toBeLessThanOrEqual(metrics.frame.x + 1);
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

async function modelPointDarkeningAtRatio(page: Page, xRatio: number, yRatio: number): Promise<number> {
  return page.locator("#mainCanvas").evaluate(
    (canvas, ratios) => {
      const element = canvas as HTMLCanvasElement;
      const ctx = element.getContext("2d");
      const axisFrame = document.querySelector<SVGRectElement>("#mainSvg .axis-frame");
      if (!ctx || !axisFrame || element.width === 0 || element.height === 0) return Number.POSITIVE_INFINITY;

      const box = element.getBoundingClientRect();
      const frame = {
        x: Number(axisFrame.getAttribute("x")),
        y: Number(axisFrame.getAttribute("y")),
        width: Number(axisFrame.getAttribute("width")),
        height: Number(axisFrame.getAttribute("height")),
      };
      const dprX = element.width / box.width;
      const dprY = element.height / box.height;
      const centerX = Math.round((frame.x + frame.width * ratios.xRatio) * dprX);
      const centerY = Math.round((frame.y + frame.height * (1 - ratios.yRatio)) * dprY);
      const patchRadius = 8;
      const patchSize = patchRadius * 2 + 1;
      const left = Math.max(0, Math.min(element.width - patchSize, centerX - patchRadius));
      const top = Math.max(0, Math.min(element.height - patchSize, centerY - patchRadius));
      const data = ctx.getImageData(left, top, patchSize, patchSize).data;
      let centerLuma = 0;
      let centerCount = 0;
      let ringLuma = 0;
      let ringCount = 0;
      for (let y = 0; y < patchSize; y += 1) {
        for (let x = 0; x < patchSize; x += 1) {
          const distance = Math.hypot(x - patchRadius, y - patchRadius);
          const offset = (y * patchSize + x) * 4;
          const luma = 0.2126 * data[offset] + 0.7152 * data[offset + 1] + 0.0722 * data[offset + 2];
          if (distance <= 2) {
            centerLuma += luma;
            centerCount += 1;
          } else if (distance >= 5 && distance <= 8) {
            ringLuma += luma;
            ringCount += 1;
          }
        }
      }
      return ringLuma / ringCount - centerLuma / centerCount;
    },
    { xRatio, yRatio },
  );
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
        red <= 235 &&
        green >= 166 &&
        green <= 170 &&
        blue >= 45 &&
        blue <= 50 &&
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

async function clickTrainPlotRatio(page: Page, xRatio: number, yRatio: number): Promise<void> {
  const point = await page.locator("#trainCanvas").evaluate(
    (canvas, ratios) => {
      const element = canvas as HTMLCanvasElement;
      const box = element.getBoundingClientRect();
      const axisFrame = document.querySelector<SVGRectElement>("#trainSvg .axis-frame");
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

async function expectVisibleControlsInsidePanels(page: Page): Promise<void> {
  const leaks = await page.locator(".plot-panel").evaluateAll((panels) =>
    panels.flatMap((panel) => {
      const panelRect = panel.getBoundingClientRect();
      const elements = panel.querySelectorAll<HTMLElement>(
        ".plot-actions, .plot-controls, .plot-control, select, .segmented",
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

async function expectPlotControlsStayInline(page: Page): Promise<void> {
  const issues = await page.locator(".plot-controls").evaluateAll((containers) =>
    containers.flatMap((container) => {
      const containerElement = container as HTMLElement;
      const visibleControls = Array.from(
        containerElement.querySelectorAll<HTMLElement>(":scope > .plot-control"),
      ).filter((element) => {
        const style = getComputedStyle(element);
        const rect = element.getBoundingClientRect();
        return style.display !== "none" && rect.width > 0 && rect.height > 0;
      });
      if (!visibleControls.length) return [];

      const firstTop = visibleControls[0].getBoundingClientRect().top;
      const rowIssues = visibleControls
        .filter((element) => Math.abs(element.getBoundingClientRect().top - firstTop) > 2)
        .map((element) => `${containerElement.className}:${element.id || element.className}:wrapped`);
      const clippedButtonIssues = Array.from(
        containerElement.querySelectorAll<HTMLButtonElement>(".segmented button"),
      )
        .filter((button) => {
          const style = getComputedStyle(button);
          const rect = button.getBoundingClientRect();
          return style.display !== "none" && rect.width > 0 && button.scrollWidth > button.clientWidth + 1;
        })
        .map((button) => `${containerElement.className}:${button.textContent ?? ""}:clipped`);
      const clippedSelectIssues = Array.from(
        containerElement.querySelectorAll<HTMLSelectElement>("select"),
      )
        .filter((select) => {
          const style = getComputedStyle(select);
          const rect = select.getBoundingClientRect();
          if (style.display === "none" || rect.width <= 0) return false;
          const selectedText = select.selectedOptions[0]?.textContent ?? "";
          const canvas = document.createElement("canvas");
          const context = canvas.getContext("2d");
          if (!context) return false;
          context.font = `${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
          const textWidth = context.measureText(selectedText).width;
          const usableWidth =
            select.clientWidth -
            Number.parseFloat(style.paddingLeft) -
            Number.parseFloat(style.paddingRight);
          return textWidth > usableWidth + 1;
        })
        .map((select) => `${containerElement.className}:${select.id}:clipped`);
      const overflowIssues =
        containerElement.scrollWidth > containerElement.clientWidth + 2 ||
        containerElement.scrollHeight > containerElement.clientHeight + 2
          ? [`${containerElement.className}:overflow`]
          : [];
      return [...rowIssues, ...clippedButtonIssues, ...clippedSelectIssues, ...overflowIssues];
    }),
  );
  expect(issues).toEqual([]);
}

async function expectReadoutFits(page: Page): Promise<void> {
  const issues = await page.locator(".selection-readout").evaluate((readout) => {
    const elements = Array.from(readout.querySelectorAll<HTMLElement>(".readout-item, .readout-value"));
    return elements
      .filter((element) => {
        const style = getComputedStyle(element);
        const rect = element.getBoundingClientRect();
        return style.display !== "none" && rect.width > 0 && element.scrollWidth > element.clientWidth + 1;
      })
      .map((element) => `${element.id || element.className}:clipped`);
  });
  expect(issues).toEqual([]);
}

async function expectTrainSummaryBadgeFits(page: Page): Promise<void> {
  const issues = await page.locator("#trainPanel").evaluate((panel) => {
    const badge = panel.querySelector<HTMLElement>("#trainRange");
    const body = panel.querySelector<HTMLElement>(".plot-body");
    const frame = panel.querySelector<SVGRectElement>("#trainSvg .axis-frame");
    const colorbar = panel.querySelector<SVGRectElement>("#trainSvg .colorbar-frame");
    if (!badge || !body || !frame) return ["missing"];

    const badgeStyle = getComputedStyle(badge);
    const svg = panel.querySelector<SVGSVGElement>("#trainSvg");
    const svgZIndex = Number.parseInt(getComputedStyle(svg!).zIndex, 10) || 0;
    const badgeZIndex = Number.parseInt(badgeStyle.zIndex, 10) || 0;
    const badgeRect = badge.getBoundingClientRect();
    const bodyRect = body.getBoundingClientRect();
    const frameRect = frame.getBoundingClientRect();
    const colorbarRect = colorbar?.getBoundingClientRect() ?? null;
    const svgDecorationRects = Array.from(
      panel.querySelectorAll<SVGGraphicsElement>(
        "#trainSvg .axis text, #trainSvg .axis-label, #trainSvg .colorbar-ticks text",
      ),
      (element) => element.getBoundingClientRect(),
    );
    const intersectsFrame =
      badgeRect.left < frameRect.right - 1 &&
      badgeRect.right > frameRect.left + 1 &&
      badgeRect.top < frameRect.bottom - 1 &&
      badgeRect.bottom > frameRect.top + 1;
    const intersectsColorbar =
      colorbarRect !== null &&
      badgeRect.left < colorbarRect.right - 1 &&
      badgeRect.right > colorbarRect.left + 1 &&
      badgeRect.top < colorbarRect.bottom - 1 &&
      badgeRect.bottom > colorbarRect.top + 1;
    const intersectsSvgDecoration = svgDecorationRects.some(
      (rect) =>
        badgeRect.left < rect.right - 1 &&
        badgeRect.right > rect.left + 1 &&
        badgeRect.top < rect.bottom - 1 &&
        badgeRect.bottom > rect.top + 1,
    );
    const outsideBody =
      badgeRect.left < bodyRect.left - 1 ||
      badgeRect.right > bodyRect.right + 1 ||
      badgeRect.top < bodyRect.top - 1 ||
      badgeRect.bottom > bodyRect.bottom + 1;
    const clipped =
      badge.scrollWidth > badge.clientWidth + 1 ||
      badge.scrollHeight > badge.clientHeight + 1;

    return [
      badgeStyle.display === "none" || badgeStyle.visibility === "hidden" || badgeRect.width <= 0
        ? "hidden"
        : "",
      clipped ? "clipped" : "",
      intersectsFrame ? "intersects-frame" : "",
      intersectsColorbar ? "intersects-colorbar" : "",
      intersectsSvgDecoration && badgeZIndex >= svgZIndex ? "intersects-svg-decoration" : "",
      outsideBody ? "outside-body" : "",
    ].filter(Boolean);
  });
  expect(issues).toEqual([]);
}

async function expectTopbarControlsFit(page: Page): Promise<void> {
  const leaks = await page.locator(".topbar").evaluate((topbar) => {
    const topbarRect = topbar.getBoundingClientRect();
    const viewportRight = document.documentElement.clientWidth;
    const elements = topbar.querySelectorAll<HTMLElement>(
      [
        ".brand-lockup",
        ".hhi-logo",
        ".topbar-copy",
        ".topbar h1",
        "#runMeta",
        ".topbar-actions",
        ".topbar-links",
        ".topbar-link",
        ".topbar-controls",
        ".topbar-control",
        "select",
        ".segmented",
        ".icon-button",
      ].join(", "),
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
          rect.right > Math.min(topbarRect.right, viewportRight) + 1 ||
          rect.top < topbarRect.top - 1 ||
          rect.bottom > topbarRect.bottom + 1
        );
      })
      .map((element) => element.id || element.className);
  });
  expect(leaks).toEqual([]);

  const clippedButtons = await page.locator(".topbar").evaluate((topbar) => {
    const buttons = topbar.querySelectorAll<HTMLElement>(".segmented button");
    return Array.from(buttons)
      .filter((button) => {
        const style = getComputedStyle(button);
        const rect = button.getBoundingClientRect();
        return style.display !== "none" && rect.width > 0 && rect.height > 0;
      })
      .filter((button) => {
        const rect = button.getBoundingClientRect();
        const range = document.createRange();
        range.selectNodeContents(button);
        const textRects = Array.from(range.getClientRects()).filter(
          (textRect) => textRect.width > 0 && textRect.height > 0,
        );
        range.detach();
        if (!textRects.length) return false;
        const textLeft = Math.min(...textRects.map((textRect) => textRect.left));
        const textRight = Math.max(...textRects.map((textRect) => textRect.right));
        const textTop = Math.min(...textRects.map((textRect) => textRect.top));
        const textBottom = Math.max(...textRects.map((textRect) => textRect.bottom));
        return (
          button.scrollWidth > button.clientWidth + 1 ||
          button.scrollHeight > button.clientHeight + 1 ||
          textLeft < rect.left - 1 ||
          textRight > rect.right + 1 ||
          textTop < rect.top - 1 ||
          textBottom > rect.bottom + 1
        );
      })
      .map((button) => button.textContent?.trim() || button.getAttribute("aria-label") || "button");
  });
  expect(clippedButtons).toEqual([]);

  const topbarBottom = await page
    .locator(".topbar")
    .evaluate((topbar) => topbar.getBoundingClientRect().bottom);
  expect(topbarBottom).toBeLessThanOrEqual(page.viewportSize()!.height);
}

async function expectTopbarResponsiveFit(page: Page, maxHeight: number): Promise<void> {
  await expectTopbarControlsFit(page);
  await expectTopbarProjectLinks(page);
  await expectLegalFooter(page);
  await expectLegalFooterFits(page);

  const metrics = await page.locator(".topbar").evaluate((topbar) => {
    const title = topbar.querySelector<HTMLElement>("h1");
    const logo = topbar.querySelector<HTMLElement>(".hhi-logo");
    const linkStack = topbar.querySelector<HTMLElement>(".topbar-links");
    const actions = topbar.querySelector<HTMLElement>(".topbar-actions");
    const viewSwitch = topbar.querySelector<HTMLElement>(".view-switch");
    const problemControl = topbar.querySelector<HTMLElement>(".problem-control");
    const qualityControl = topbar.querySelector<HTMLElement>(".quality-control");
    const links = Array.from(topbar.querySelectorAll<HTMLElement>(".topbar-link"));
    const topbarBox = topbar.getBoundingClientRect();
    const titleRange = document.createRange();
    if (title) titleRange.selectNodeContents(title);
    const titleRects = title
      ? Array.from(titleRange.getClientRects()).filter(
          (rect) => rect.width > 1 && rect.height > 1,
        )
      : [];
    const titleBox = title?.getBoundingClientRect() ?? null;
    const linkStackBox = linkStack?.getBoundingClientRect() ?? null;
    const actionsBox = actions?.getBoundingClientRect() ?? null;
    const viewSwitchBox = viewSwitch?.getBoundingClientRect() ?? null;
    const problemControlBox = problemControl?.getBoundingClientRect() ?? null;
    const qualityControlBox = qualityControl?.getBoundingClientRect() ?? null;
    const linkRects = links
      .map((link) => link.getBoundingClientRect())
      .filter((rect) => rect.width > 1 && rect.height > 1);
    const lineCount = (rects: DOMRect[]): number =>
      new Set(rects.map((rect) => Math.round(rect.top))).size;
    const titleFits =
      titleBox !== null &&
      titleRects.length > 0 &&
      Math.min(...titleRects.map((rect) => rect.left)) >= titleBox.left - 1 &&
      Math.max(...titleRects.map((rect) => rect.right)) <= titleBox.right + 1;
    const linksShareControlsRow =
      linkStackBox !== null &&
      actionsBox !== null &&
      linkStackBox.bottom > actionsBox.top + 1 &&
      actionsBox.bottom > linkStackBox.top + 1;
    const controlsBetweenTitleAndLinks =
      !linksShareControlsRow ||
      (titleBox !== null &&
        linkStackBox !== null &&
        actionsBox !== null &&
        actionsBox.left > titleBox.right + 4 &&
        actionsBox.right < linkStackBox.left - 4);
    const viewKeepsUsableWidth =
      topbarBox.width < 1200 || (viewSwitchBox !== null && viewSwitchBox.width >= 220);
    const dataControlsStayRight =
      !linksShareControlsRow ||
      (qualityControlBox !== null &&
        linkStackBox !== null &&
        linkStackBox.left - qualityControlBox.right <= 36);
    const viewCenteredBetweenTitleAndProblem =
      !linksShareControlsRow ||
      topbarBox.width < 1200 ||
      (titleBox !== null &&
        viewSwitchBox !== null &&
        problemControlBox !== null &&
        Math.abs(
          (viewSwitchBox.left + viewSwitchBox.right) / 2 -
            (titleBox.right + problemControlBox.left) / 2,
        ) <= 16);

    return {
      dataControlsStayRight,
      height: topbarBox.height,
      linkLineCount: lineCount(linkRects),
      linkStackHeight: linkStack?.getBoundingClientRect().height ?? 0,
      controlsBetweenTitleAndLinks,
      linksVisible: linkRects.length === links.length,
      logoHeight: logo?.getBoundingClientRect().height ?? 0,
      overflowX: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      titleFits,
      titleLineCount: lineCount(titleRects),
      viewCenteredBetweenTitleAndProblem,
      viewKeepsUsableWidth,
    };
  });

  expect(metrics.height).toBeLessThanOrEqual(maxHeight);
  expect(metrics.titleLineCount).toBe(1);
  expect(metrics.titleFits).toBe(true);
  expect(metrics.linksVisible).toBe(true);
  expect(metrics.linkLineCount).toBe(2);
  expect(metrics.controlsBetweenTitleAndLinks).toBe(true);
  expect(metrics.dataControlsStayRight).toBe(true);
  expect(Math.abs(metrics.linkStackHeight - metrics.logoHeight)).toBeLessThanOrEqual(1);
  expect(metrics.viewCenteredBetweenTitleAndProblem).toBe(true);
  expect(metrics.viewKeepsUsableWidth).toBe(true);
  expect(metrics.overflowX).toBeLessThanOrEqual(1);
}

async function expectTopbarProjectLinks(page: Page): Promise<void> {
  const paperLink = page.locator(".topbar").getByRole("link", { name: "arXiv" });
  const githubLink = page.locator(".topbar").getByRole("link", { name: "GitHub" });

  await expect(paperLink).toBeVisible();
  await expect(paperLink).toHaveAttribute("href", "https://arxiv.org/abs/2409.08958");
  await expect(paperLink).toHaveAttribute("target", "_blank");
  await expect(paperLink).toHaveAttribute("rel", /noopener/);
  await expect(paperLink).toHaveAttribute("rel", /noreferrer/);

  await expect(githubLink).toBeVisible();
  await expect(githubLink).toHaveAttribute("href", "https://github.com/aleks-krasowski/PINNfluence/");
  await expect(githubLink).toHaveAttribute("target", "_blank");
  await expect(githubLink).toHaveAttribute("rel", /noopener/);
  await expect(githubLink).toHaveAttribute("rel", /noreferrer/);
}

async function expectLegalFooter(page: Page): Promise<void> {
  const footer = page.locator(".legal-footer");
  const privacyLink = footer.getByRole("link", { name: "Privacy" });
  const legalNoticeLink = footer.getByRole("link", { name: "Legal Notice" });
  const copyright = footer.locator(".legal-footer-copyright");

  await expect(footer).toBeVisible();
  await expect(privacyLink).toBeVisible();
  await expect(privacyLink).toHaveAttribute("href", "./legal/privacy.html");
  await expect(legalNoticeLink).toBeVisible();
  await expect(legalNoticeLink).toHaveAttribute("href", "./legal/impressum.html");
  await expect(copyright).toHaveText("© Fraunhofer HHI");
  await expect(copyright).toBeVisible();
}

async function expectLegalFooterFits(page: Page): Promise<void> {
  const leaks = await page.locator(".legal-footer").evaluate((footer) => {
    const footerRect = footer.getBoundingClientRect();
    const viewportRight = document.documentElement.clientWidth;
    const elements = footer.querySelectorAll<HTMLElement>(
      [".legal-footer-links", ".legal-footer-link", ".legal-footer-copyright"].join(", "),
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
          rect.left < footerRect.left - 1 ||
          rect.right > Math.min(footerRect.right, viewportRight) + 1 ||
          rect.top < footerRect.top - 1 ||
          rect.bottom > footerRect.bottom + 1
        );
      })
      .map((element) => element.className);
  });
  expect(leaks).toEqual([]);
}

async function tapMainPoint(page: Page): Promise<void> {
  const box = await page.locator("#mainCanvas").boundingBox();
  if (!box) throw new Error("Missing main canvas bounds");
  await page.touchscreen.tap(box.x + box.width * 0.68, box.y + box.height * 0.36);
}

async function touchDragMainRegion(page: Page): Promise<void> {
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
    const start = point(0.22, 0.24);
    const mid = point(0.52, 0.56);
    const end = point(0.82, 0.82);
    fire("pointerdown", start, 21);
    fire("pointermove", mid, 21);
    fire("pointermove", end, 21);
    fire("pointerup", end, 21);
  });
}

test("topbar stays compact with controls before project links", async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "desktop-only responsive topbar coverage");
  await page.goto(FIXTURE_URL);

  for (const { width, height, maxHeight } of [
    { width: 1440, height: 900, maxHeight: 72 },
    { width: 1280, height: 900, maxHeight: 72 },
    { width: 1120, height: 900, maxHeight: 70 },
    { width: 920, height: 900, maxHeight: 70 },
    { width: 901, height: 900, maxHeight: 70 },
    { width: 900, height: 900, maxHeight: 116 },
    { width: 390, height: 844, maxHeight: 138 },
    { width: 320, height: 844, maxHeight: 138 },
  ]) {
    await page.setViewportSize({ width, height });
    await expectTopbarResponsiveFit(page, maxHeight);
  }
});

test("model plot shows an initial interaction hint", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "desktop-only timing coverage is enough");
  await page.goto(FIXTURE_URL);
  await expectNonblankCanvas(page, "#mainCanvas");

  const hint = page.locator("#modelInteractionHint");
  await expect(hint).toHaveText("Click or Drag to see Influences.");
  await expect(hint).toHaveAttribute("data-state", "visible");
  await expect(hint).toBeVisible();
  await expect(hint).toHaveCSS("pointer-events", "none");

  const hintBox = await hint.boundingBox();
  const bodyBox = await page.locator("#modelPanel .plot-body").boundingBox();
  expect(hintBox).not.toBeNull();
  expect(bodyBox).not.toBeNull();
  expect(hintBox!.x).toBeGreaterThanOrEqual(bodyBox!.x - 1);
  expect(hintBox!.x + hintBox!.width).toBeLessThanOrEqual(bodyBox!.x + bodyBox!.width + 1);
  expect(hintBox!.y).toBeGreaterThanOrEqual(bodyBox!.y - 1);
  expect(hintBox!.y + hintBox!.height).toBeLessThanOrEqual(bodyBox!.y + bodyBox!.height + 1);

  await expect(hint).toBeHidden({ timeout: 5_000 });

  await page.goto(FIXTURE_URL);
  await expectNonblankCanvas(page, "#mainCanvas");
  await expect(hint).toHaveAttribute("data-state", "visible");
  await clickMainPoint(page);
  await expect(hint).toBeHidden({ timeout: 2_000 });
});

test("desktop renders two plots and continues background prefetching", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "desktop-only viewport assertions");
  const requests: string[] = [];
  page.on("request", (request) => requests.push(request.url()));

  await page.goto(FIXTURE_URL);

  await expectTopbarProjectLinks(page);
  await expect(page.getByRole("heading", { name: "PINNteractive PINNfluence Demo", level: 1 })).toBeVisible();
  await expect(page.locator("#problemSelect option")).toHaveText([
    "Fixture",
    "Shifted Fixture",
    "Burgers",
    "Drift Diffusion",
    "Navier Stokes",
  ]);
  await expect(page.locator("button[data-model-quality='good']")).toHaveClass(/active/);
  await expect(page.locator("#runMeta")).toHaveText(/Fixture · Well-Trained · 4 candidate · 5 train/);
  await expect(page.locator("#modelMeta")).toBeHidden();
  await expectTopbarControlsFit(page);
  await expect(page.locator(".control-panel")).toHaveCount(0);
  await expect(page.locator(".control-group")).toHaveCount(0);
  await expect(page.locator(".plot-panel")).toHaveCount(2);
  await expect(page.locator("#globalPanel")).toHaveCount(0);
  await expect(page.locator("#mainTitle")).toHaveText("Model");
  await expect(page.locator("#trainTitle")).toHaveText("Training");
  await expect(page.locator(".model-panel #fieldSelect")).toBeVisible();
  await expect(page.locator(".model-panel #fieldKindButtons")).toHaveCount(0);
  await expect(page.locator(".train-panel #trainModeButtons")).toHaveCount(0);
  await expect(page.locator("#summaryControl")).toHaveCount(0);
  await expect(page.locator(".train-panel #matrixSelect")).toHaveCount(1);
  await expect(page.locator(".train-panel #matrixSelect option")).toHaveText(["ℒ → ℒ"]);
  await expect(page.locator(".train-panel #signButtons")).toHaveCount(1);
  await expect(page.locator(".train-panel #kSlider")).toHaveCount(1);
  await expectVisibleControlsInsidePanels(page);
  await expectNonblankCanvas(page, "#mainCanvas");
  await expectNonblankCanvas(page, "#trainCanvas");
  await expectModelTrainFramesHorizontallyAligned(page);
  await expect.poll(() => modelPointDarkeningAtRatio(page, 0.58, 0.5)).toBeLessThan(8);
  const goodMainSignature = await canvasSignature(page, "#mainCanvas");
  await page.locator("button[data-model-quality='bad']").click();
  await expect(page.locator("button[data-model-quality='bad']")).toHaveClass(/active/);
  await expect(page.locator("#runMeta")).toHaveText(/Fixture · Poorly-Trained · 4 candidate · 5 train/);
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
  await expectCompactOutsideDecorations(page, "#mainSvg");
  await expectCompactOutsideDecorations(page, "#trainSvg");
  await expect(page.locator("#mapControl")).toHaveCount(0);
  await expect(page.locator("#methodControl")).toHaveCount(0);
  await expect(page.locator("#influenceMapToggle")).toHaveCount(0);
  await expect(page.locator("#backgroundControl")).toBeVisible();
  await expect(page.locator("#backgroundButtons button")).toHaveText([
    "Pts",
    "KDE",
    "Cell",
  ]);
  await expect(page.locator("button[data-background-mode='points']")).toHaveClass(/active/);
  await expect(page.locator("#kControl")).toBeVisible();
  await expect(page.locator("#kSlider")).toHaveAttribute("min", "0");
  await expect(page.locator("#kSlider")).toHaveAttribute("max", "256");
  await expect(page.locator("#kOutput")).toHaveText("25");
  await expect(page.locator("#trainRange")).toHaveText(/Local · Points( · max \|I\| .*)?/);
  await expect(page.locator("#trainRange")).not.toHaveText(/all exported influences/);
  await expectTrainSummaryBadgeFits(page);
  await expectNonblankCanvas(page, "#trainCanvas");

  const backgroundSignatures: number[] = [await canvasSignature(page, "#trainCanvas")];
  for (const [mode, label] of [
    ["smooth", "Smooth"],
    ["cell", "Cells"],
  ] as const) {
    await page.locator(`button[data-background-mode='${mode}']`).click();
    await expect(page.locator(`button[data-background-mode='${mode}']`)).toHaveClass(/active/);
    await expect(page.locator("#kControl")).toBeVisible();
    await expect(page.locator("#trainRange")).toHaveText(
      new RegExp(`Local · ${label}( · max \\|I\\| .*)?`),
    );
    await expect(page.locator("#trainRange")).not.toHaveText(/all exported influences/);
    await expectTrainSummaryBadgeFits(page);
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
  const scoreIndex = requests.findIndex((url) => url.includes("/influence/") && url.includes("/scores.f32"));
  expect(predIndex).toBeGreaterThanOrEqual(0);
  expect(scoreIndex).toBeGreaterThanOrEqual(0);
  if (lossIndex >= 0) expect(predIndex).toBeLessThan(lossIndex);

  await expect.poll(() => requests.some((url) => url.includes("loss_total_raster.u16"))).toBe(true);
  await expect.poll(() => requests.some((url) => url.includes("/influence/") && url.includes("/scores.f32"))).toBe(true);

  await page.locator("#fieldSelect").selectOption("loss_total");
  await expect(page.locator("#mainTitle")).toHaveText("Model");

  await page.locator("button[data-sign='pos']").click();
  await expectNonblankCanvas(page, "#trainCanvas");

  await expect(page.locator("#summaryControl")).toHaveCount(0);
  await expect(page.locator("button[data-train-mode='global']")).toHaveCount(0);
  await expect(page.locator("#backgroundControl")).toBeVisible();
  await expect(page.locator("#methodControl")).toHaveCount(0);
  await expectVisibleControlsInsidePanels(page);
  await page.locator("button[data-background-mode='cell']").click();
  await expect(page.locator("button[data-background-mode='cell']")).toHaveClass(/active/);
  await expectNonblankCanvas(page, "#trainCanvas");
  await expectContourPaths(page, "train");
  await expect(page.locator("#globalCanvas")).toHaveCount(0);

  await dragMainRegion(page);
  await expect(page.locator("#selectedPoint")).toHaveText(/x\[.+,.+\] y\[.+,.+\]/);
  await expectReadoutFits(page);
  await expect(page.locator("#trainRange")).toHaveText(/Local region · Cells · average over [1-4] candidates · mean I -?(?:\d|\.)/);
  await expectTrainSummaryBadgeFits(page);
  await expectNonblankCanvas(page, "#trainCanvas");

  await clickMainPoint(page);
  await expect(page.locator("#selectedPoint")).toHaveText(/\(.+,.+\)/);
  await expectReadoutFits(page);
  await expect(page.locator("#trainRange")).toHaveText(/Local/);
  await expectTrainSummaryBadgeFits(page);
  await expectNonblankCanvas(page, "#trainCanvas");

  await clickTrainPlotRatio(page, 0.82, 0.75);
  await expect(page.locator("#selectedPoint")).toHaveText(/\(0.875,0.625\)/);
  await expectReadoutFits(page);
  await expect(page.locator("#trainRange")).toHaveText(/Local/);
  await expectTrainSummaryBadgeFits(page);
  await expectNonblankCanvas(page, "#trainCanvas");
});

test("explainer tab renders an interactive formula guide", async ({ page }) => {
  await page.goto(FIXTURE_URL);
  await page.locator("button[data-app-view='explainer']").click();

  await expect(page.locator("#explainerWorkspace")).toBeVisible();
  await expect(page.locator("#playgroundWorkspace")).toBeHidden();
  await expect(page.locator("#resultsWorkspace")).toBeHidden();
  await expect(page.getByRole("heading", { name: "How one training point shapes a PINN prediction" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Influence as local sensitivity" })).toBeVisible();
  await expect(page.locator("[data-explainer-token]")).toHaveCount(12);
  await expect(page.locator("[data-explainer-token='score']")).toHaveText("Inf");
  await expect(page.locator(".explainer-transpose")).toHaveText("⊤");
  const formulaOverflow = await page.locator(".explainer-formula-scroll").evaluate((container) => {
    const formula = container.querySelector<HTMLElement>(".explainer-formula .katex");
    if (!formula) return ["missing-formula"];
    const containerBox = container.getBoundingClientRect();
    const formulaBox = formula.getBoundingClientRect();
    return [
      formulaBox.left < containerBox.left - 1 ? "left" : "",
      formulaBox.right > containerBox.right + 1 ? "right" : "",
      formulaBox.top < containerBox.top - 1 ? "top" : "",
      formulaBox.bottom > containerBox.bottom + 1 ? "bottom" : "",
      container.scrollWidth > container.clientWidth + 1 ? "scroll-width" : "",
    ].filter(Boolean);
  });
  expect(formulaOverflow).toEqual([]);
  await expect(page.locator(".explainer-domain-point-pde").first()).toHaveCSS("background-color", "rgba(197, 14, 31, 0.82)");
  await expect(page.locator(".explainer-loss-fill-pde")).toHaveCSS("background-color", "rgba(197, 14, 31, 0.82)");
  await expect(page.locator(".explainer-domain-point-ic").first()).toHaveCSS("background-color", "rgba(31, 119, 180, 0.82)");
  await expect(page.locator(".explainer-loss-fill-ic")).toHaveCSS("background-color", "rgba(31, 119, 180, 0.82)");
  await expect(page.locator(".explainer-domain-point-bc").first()).toHaveCSS("background-color", "rgba(247, 144, 17, 0.82)");
  await expect(page.locator(".explainer-loss-fill-bc")).toHaveCSS("background-color", "rgba(247, 144, 17, 0.82)");
  const bcEdgeOffsets = await page.locator(".explainer-domain-stage").evaluate((stage) => {
    const stageBox = stage.getBoundingClientRect();
    const points = Array.from(stage.querySelectorAll<HTMLElement>(".explainer-domain-point-bc"));
    return points.map((point) => {
      const pointBox = point.getBoundingClientRect();
      const centerX = pointBox.left + pointBox.width / 2;
      const leftEdge = stageBox.left + stageBox.width * 0.09;
      const rightEdge = stageBox.left + stageBox.width * 0.91;
      return Math.min(Math.abs(centerX - leftEdge), Math.abs(centerX - rightEdge));
    });
  });
  expect(Math.max(...bcEdgeOffsets)).toBeLessThanOrEqual(1);
  await expect(page.locator("#explainer-token-title")).toHaveText("Influence Score");
  await expect(page.locator("#explainer-token-body")).toContainText("locally sensitive");
  await expectNoDocumentHorizontalOverflow(page);

  await page.locator("[data-explainer-token='hessian']").click();
  await expect(page.locator("#explainer-token-title")).toHaveText("Local Training Geometry");
  await expect(page.locator("#explainer-token-body")).toContainText("reverse map");
  await expect(page.locator("[data-explainer-token='hessian']")).toHaveAttribute("aria-pressed", "true");

  await page.locator("[data-explainer-token='train-point']").focus();
  await expect(page.locator("#explainer-token-title")).toHaveText("Training Point x");
  await expect(page.locator("#explainer-token-body")).toContainText("participated in training");

  await page.locator("[data-explainer-token='loss-fraction']").click();
  await expect(page.locator("#explainer-token-title")).toHaveText("Loss-Term Fraction");
  await expect(page.locator("#explainer-token-body")).toContainText("one loss component");
  await expect(page.locator("[data-explainer-token='loss-fraction']")).toHaveAttribute("aria-pressed", "true");
  await expectNoDocumentHorizontalOverflow(page);
});

test("indicators tab renders dashboard charts and preserves the Playground", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "desktop-only tab coverage is enough");
  await page.goto(FIXTURE_URL);
  await expectNonblankCanvas(page, "#mainCanvas");

  await page.locator("button[data-app-view='explainer']").click();
  await expect(page.locator("#explainerWorkspace")).toBeVisible();
  await expect(page.locator("#playgroundWorkspace")).toBeHidden();
  await expect(page.locator("#resultsWorkspace")).toBeHidden();
  await expect(page.locator("#runMeta")).toHaveText(/Fixture · Well-Trained · 4 candidate · 5 train/);
  await expect(page.locator("#problemSelect")).toBeDisabled();
  await expect(page.locator("button[data-model-quality='good']")).toBeDisabled();
  await expect(page.locator("button[data-model-quality='bad']")).toBeDisabled();

  await page.locator("button[data-app-view='indicators']").click();
  await expect(page.locator("#problemSelect")).toBeEnabled();
  await expect(page.locator("button[data-model-quality='good']")).toBeDisabled();
  await expect(page.locator("button[data-model-quality='bad']")).toBeDisabled();
  await expect(page.locator(".quality-control")).toHaveAttribute("data-disabled", "true");
  await page.locator("#problemSelect").selectOption("burgers");

  await expect(page.locator("#resultsWorkspace")).toBeVisible();
  await expect(page.locator("#playgroundWorkspace")).toBeHidden();
  await expect(page.locator("#explainerWorkspace")).toBeHidden();
  await expect(page.locator("#resetButton")).toHaveCount(0);
  await expect(page.locator("#runMeta")).toHaveText(/Burgers · Well-Trained · 4 candidate · 5 train/);
  await expect(page.locator("#resultsWorkspace .results-panel")).toHaveCount(3);
  await expect(page.locator("#resultsWorkspace [data-results-chart]")).toHaveCount(2);
  await expect(page.getByRole("heading", { name: "Loss Component Decomposition" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "IC Fraction Across Problems" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Influence Indicator" })).toBeVisible();
  await expect(page.locator("#resultsWorkspace .results-panel-header .results-hint-button")).toHaveCount(3);
  await waitForIndicatorsCharts(page);
  const dashboardLayout = await resultsDashboardLayoutMetrics(page);
  expect(dashboardLayout.workspace.scrollHeight - dashboardLayout.workspace.clientHeight).toBeLessThanOrEqual(2);
  expect(Math.abs(dashboardLayout.loss.left - dashboardLayout.grid.left)).toBeLessThanOrEqual(1);
  expect(Math.abs(dashboardLayout.loss.right - dashboardLayout.grid.right)).toBeLessThanOrEqual(1);
  expect(dashboardLayout.ic.top - dashboardLayout.loss.bottom).toBeGreaterThanOrEqual(10);
  expect(dashboardLayout.ic.top - dashboardLayout.loss.bottom).toBeLessThanOrEqual(14);
  expect(Math.abs(dashboardLayout.ic.top - dashboardLayout.indicator.top)).toBeLessThanOrEqual(1);
  expect(Math.abs(dashboardLayout.ic.bottom - dashboardLayout.indicator.bottom)).toBeLessThanOrEqual(1);
  expect(Math.abs(dashboardLayout.ic.left - dashboardLayout.grid.left)).toBeLessThanOrEqual(1);
  expect(Math.abs(dashboardLayout.indicator.right - dashboardLayout.grid.right)).toBeLessThanOrEqual(1);
  const lossLegend = page.locator(".loss-decomposition-legend");
  await expect(lossLegend).toContainText("PDE");
  await expect(lossLegend).toContainText("IC");
  await expect(lossLegend).toContainText("BC");
  await expect(lossLegend).not.toContainText(/PDE Loss|BC 0|BC 1|BC: boundary|PDE lossBC 0BC 1/i);
  const icLegend = page.locator(".results-ic-legend");
  await expect(icLegend).toContainText("Diffusion");
  await expect(icLegend).not.toContainText("Heat");
  const hintChecks = [
    {
      buttonName: "Explain Loss Component Decomposition",
      popover: "#results-hint-loss",
      text: /absolute influence|Cancellation κ/,
    },
    {
      buttonName: "Explain IC Fraction Across Problems",
      popover: "#results-hint-ic",
      text: /initial-condition terms|late peaks/,
    },
    {
      buttonName: "Explain Influence Indicator",
      popover: "#results-hint-indicator",
      text: /U\(z\)|training-point layout/,
    },
  ];
  for (const hint of hintChecks) {
    const button = page.getByRole("button", { name: hint.buttonName });
    await expect(button).toHaveText("?");
    await button.click();
    await expect(button).toHaveAttribute("aria-expanded", "true");
    await expect(page.locator(hint.popover)).toBeVisible();
    await expect(page.locator(hint.popover)).toContainText(hint.text);
    await page.keyboard.press("Escape");
    await expect(button).toHaveAttribute("aria-expanded", "false");
    await expect(page.locator(hint.popover)).toBeHidden();
  }
  await expect(page.locator('[data-results-table="indicator"] table')).toBeVisible();
  await expect(page.locator('[data-results-chart="dominance"]')).toHaveCount(0);

  const resultsBeforeQualitySwitch = await page.locator("#resultsWorkspace").innerHTML();
  await expect(page.locator("button[data-model-quality='bad']")).toBeDisabled();
  await expect(page.locator("#runMeta")).toHaveText(/Burgers · Well-Trained · 4 candidate · 5 train/);
  expect(await page.locator("#resultsWorkspace").innerHTML()).toBe(resultsBeforeQualitySwitch);

  await page.locator("button[data-app-view='playground']").click();
  await expect(page.locator("#playgroundWorkspace")).toBeVisible();
  await expect(page.locator("#explainerWorkspace")).toBeHidden();
  await expect(page.locator("#resultsWorkspace")).toBeHidden();
  await expect(page.locator("#problemSelect")).toBeEnabled();
  await expect(page.locator("button[data-model-quality='good']")).toBeEnabled();
  await expect(page.locator("button[data-model-quality='bad']")).toBeEnabled();
  await expectNonblankCanvas(page, "#mainCanvas");
  await expectNonblankCanvas(page, "#trainCanvas");
});

test("indicators tab stays reachable on a short desktop viewport", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "desktop-only short viewport coverage");
  await page.setViewportSize({ width: 1440, height: 620 });
  await openIndicatorsDashboard(page);

  const scrollMetrics = await page.locator("#resultsWorkspace").evaluate((workspace) => {
    const element = workspace as HTMLElement;
    element.scrollTop = 0;
    element.scrollTop = element.scrollHeight;
    return {
      clientHeight: element.clientHeight,
      scrollHeight: element.scrollHeight,
      scrollTop: element.scrollTop,
    };
  });
  expect(scrollMetrics.scrollHeight - scrollMetrics.clientHeight).toBeGreaterThan(50);
  expect(scrollMetrics.scrollTop).toBeGreaterThan(0);

  const panels = page.locator("#resultsWorkspace .results-panel");
  await expect(panels).toHaveCount(3);
  for (let index = 0; index < 3; index += 1) {
    const panel = panels.nth(index);
    await panel.scrollIntoViewIfNeeded();
    const visibleHeight = await panel.evaluate((element) => {
      const workspace = document.querySelector<HTMLElement>("#resultsWorkspace");
      if (!workspace) throw new Error("Missing results workspace");
      const panelBox = element.getBoundingClientRect();
      const workspaceBox = workspace.getBoundingClientRect();
      return Math.min(panelBox.bottom, workspaceBox.bottom) - Math.max(panelBox.top, workspaceBox.top);
    });
    expect(visibleHeight).toBeGreaterThan(40);
  }
});

test("mobile Indicators gives the stacked IC comparison enough height", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "mobile", "mobile-only Indicators chart sizing");
  await page.goto(FIXTURE_URL);

  await page.locator("button[data-app-view='indicators']").click();
  await page.locator("#problemSelect").selectOption("burgers");
  await expect
    .poll(() => page.locator('[data-results-chart="ic"] .results-ic-line').count(), { timeout: 10_000 })
    .toBeGreaterThan(0);

  const metrics = await page.locator(".results-ic-chart").evaluate((svg) => {
    const root = svg as SVGSVGElement;
    const frames = Array.from(root.querySelectorAll<SVGRectElement>(".results-ic-panel .loss-decomposition-frame"));
    return {
      chartHeight: root.getBoundingClientRect().height,
      frameHeights: frames.map((frame) => Number(frame.getAttribute("height"))),
    };
  });

  expect(metrics.chartHeight).toBeGreaterThanOrEqual(450);
  expect(metrics.frameHeights).toHaveLength(2);
  expect(Math.min(...metrics.frameHeights)).toBeGreaterThanOrEqual(170);
});

test("wide desktop uses the viewport width and keeps plot tiles side by side", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "desktop-only viewport assertions");
  await page.setViewportSize({ width: 2400, height: 1600 });
  await page.goto(FIXTURE_URL);

  await expectNonblankCanvas(page, "#mainCanvas");
  await expectNonblankCanvas(page, "#trainCanvas");
  await expect(page.locator("#plotGrid")).toHaveAttribute("data-layout", "row");

  const metrics = await page.evaluate(() => {
    const shell = document.querySelector<HTMLElement>(".app-shell");
    const grid = document.querySelector<HTMLElement>("#plotGrid");
    const model = document.querySelector<HTMLElement>("#modelPanel");
    const train = document.querySelector<HTMLElement>("#trainPanel");
    if (!shell || !grid || !model || !train) throw new Error("Missing layout elements");
    const shellRect = shell.getBoundingClientRect();
    const gridRect = grid.getBoundingClientRect();
    const modelRect = model.getBoundingClientRect();
    const trainRect = train.getBoundingClientRect();
    return {
      viewportWidth: document.documentElement.clientWidth,
      shellWidth: shellRect.width,
      gridWidth: gridRect.width,
      modelRight: modelRect.right,
      trainLeft: trainRect.left,
      modelTop: modelRect.top,
      trainTop: trainRect.top,
    };
  });

  expect(metrics.shellWidth).toBeGreaterThanOrEqual(metrics.viewportWidth - 1);
  expect(metrics.gridWidth).toBeGreaterThan(metrics.viewportWidth - 50);
  expect(metrics.modelRight).toBeLessThan(metrics.trainLeft);
  expect(Math.abs(metrics.modelTop - metrics.trainTop)).toBeLessThan(1);
});

test("legal pages load through the dev server", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "static page smoke coverage is enough on desktop");

  await page.goto("/legal/privacy.html");
  await expect(page).toHaveTitle(/Privacy Notice \| PINNfluence/);
  await expect(page.getByRole("heading", { level: 1, name: "Privacy Notice" })).toBeVisible();
  await expect(page.getByText("PINNfluence").first()).toBeVisible();
  await expect(page.getByText("Fraunhofer-Gesellschaft").first()).toBeVisible();
  await expect(
    page.getByText("The storage of the IP address is done anonymously by removing the last block of characters."),
  ).toBeVisible();
  await expect(page.getByText("Information about your right to object under Article 21 of the GDPR")).toBeVisible();

  await page.goto("/legal/impressum.html");
  await expect(page).toHaveTitle(/Impressum \| PINNfluence/);
  await expect(page.getByRole("heading", { level: 1, name: "Impressum" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Editorial Notes" })).toBeVisible();
  await expect(page.getByText("The Fraunhofer Heinrich-Hertz Institut HHI")).toBeVisible();
  await expect(
    page.getByText("Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V"),
  ).toBeVisible();
  await expect(page.getByRole("heading", { name: "Disclaimer" })).toBeVisible();
});

test("switching models and problems preserves comparison state", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "desktop-only state persistence assertions");
  await page.goto(FIXTURE_URL);

  await page.locator("#fieldSelect").selectOption("loss_total");
  await expect(page.locator("#fieldSelect")).toHaveValue("loss_total");

  await page.locator("button[data-background-mode='cell']").click();
  await page.locator("button[data-sign='neg']").click();
  await page.locator("#kSlider").fill("7");
  await expect(page.locator("button[data-background-mode='cell']")).toHaveClass(/active/);
  await expect(page.locator("button[data-sign='neg']")).toHaveClass(/active/);
  await expect(page.locator("#kOutput")).toHaveText("7");

  await clickMainPlotRatio(page, 0.22, 0.78);
  await expect(page.locator("#selectedPoint")).toHaveText(/\(0.125,0.875\)/);
  await expectReadoutFits(page);
  const selectedBeforeModelSwitch = await page.locator("#selectedPoint").textContent();

  await page.locator("button[data-model-quality='bad']").click();
  await expect(page.locator("#runMeta")).toHaveText(/Fixture · Poorly-Trained · 4 candidate · 5 train/);
  await expect(page.locator("#fieldSelect")).toHaveValue("loss_total");
  await expect(page.locator("button[data-background-mode='cell']")).toHaveClass(/active/);
  await expect(page.locator("button[data-sign='neg']")).toHaveClass(/active/);
  await expect(page.locator("#kOutput")).toHaveText("7");
  await expect(page.locator("#selectedPoint")).toHaveText(selectedBeforeModelSwitch ?? "");
  await expectReadoutFits(page);
  await expectNonblankCanvas(page, "#mainCanvas");
  await expectNonblankCanvas(page, "#trainCanvas");

  await page.locator("#problemSelect").selectOption("shifted_fixture");
  await expect(page.locator("#runMeta")).toHaveText(/Shifted Fixture · Poorly-Trained · 4 candidate · 5 train/);
  await expect(page.locator("#fieldSelect")).toHaveValue("loss_residual");
  await expect(page.locator("#matrixSelect")).toHaveValue("m_shifted");
  await expect(page.locator("button[data-background-mode='cell']")).toHaveClass(/active/);
  await expect(page.locator("button[data-sign='neg']")).toHaveClass(/active/);
  await expect(page.locator("#kOutput")).toHaveText("7");
  await expect(page.locator("#selectedPoint")).toHaveText(/\(11.25,3.75\)/);
  await expectReadoutFits(page);
  await expectNonblankCanvas(page, "#mainCanvas");
  await expectNonblankCanvas(page, "#trainCanvas");
});

test("drift diffusion uses a square physical pi axis", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "desktop-only visual axis assertions");
  await page.goto(FIXTURE_URL);

  await page.locator("#problemSelect").selectOption("drift_diffusion");
  await expect(page.locator("#runMeta")).toHaveText(/Drift Diffusion · Well-Trained · 4 candidate · 5 train/);
  await expectNonblankCanvas(page, "#mainCanvas");
  await expectNonblankCanvas(page, "#trainCanvas");

  await expect.poll(() => axisFrameRatio(page, "#mainSvg")).toBeGreaterThan(0.95);
  await expect.poll(() => axisFrameRatio(page, "#mainSvg")).toBeLessThan(1.05);
  await expect(page.locator(".model-panel .axis-label-x")).toHaveText("x");
  await expect(page.locator(".model-panel .axis-label-y")).toHaveText("t");
  await expect(page.locator(".train-panel .axis-label-y")).toHaveText("t");
  await expect(page.locator(".model-panel .colorbar-frame")).toHaveCount(1);
  await expect(page.locator(".train-panel .colorbar-frame")).toHaveCount(1);
  await expectCompactOutsideDecorations(page, "#mainSvg");
  await expectCompactOutsideDecorations(page, "#trainSvg");

  const xTickLabels = await page.locator("#mainSvg .axis-x .tick text").allTextContents();
  expect(xTickLabels).toContain("0");
  expect(xTickLabels).toContain("π");
  expect(xTickLabels).toContain("2π");

  await clickMainPlotRatio(page, 0.96, 0.52);
  const selected = (await page.locator("#selectedPoint").textContent()) ?? "";
  expect(selected).toMatch(/\((5|6)/);
  await expectReadoutFits(page);
});

test("mobile keeps Model and Train visible in the first viewport", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "mobile", "mobile-only viewport assertions");
  await page.goto(FIXTURE_URL);

  await expectTopbarProjectLinks(page);
  await expect(page.locator("#problemSelect option")).toHaveText([
    "Fixture",
    "Shifted Fixture",
    "Burgers",
    "Drift Diffusion",
    "Navier Stokes",
  ]);
  await expect(page.locator("button[data-model-quality='good']")).toHaveClass(/active/);
  await expectTopbarControlsFit(page);
  await expect(page.locator(".control-panel")).toHaveCount(0);
  await expect(page.locator(".control-group")).toHaveCount(0);
  await expect(page.locator(".plot-panel")).toHaveCount(2);
  await expect(page.locator("#globalPanel")).toHaveCount(0);
  await expect(page.locator(".train-panel #trainModeButtons")).toHaveCount(0);
  await expect(page.locator("#summaryControl")).toHaveCount(0);
  await expect(page.locator("#mapControl")).toHaveCount(0);
  await expect(page.locator("#methodControl")).toHaveCount(0);
  await expect(page.locator("#backgroundControl")).toBeVisible();
  await expect(page.locator("#kControl")).toBeVisible();
  await expect(page.locator("#backgroundButtons button")).toHaveText([
    "Pts",
    "KDE",
    "Cell",
  ]);
  for (const mode of ["cell", "smooth", "points"]) {
    await page.locator(`button[data-background-mode='${mode}']`).click();
    await expectNonblankCanvas(page, "#trainCanvas");
  }
  await expectNonblankCanvas(page, "#trainCanvas");
  await expectVisibleControlsInsidePanels(page);
  await expectTopbarControlsFit(page);
  await expectVisibleControlsInsidePanels(page);
  await expectNonblankCanvas(page, "#mainCanvas");
  await expectNonblankCanvas(page, "#trainCanvas");
  await expectModelTrainFramesHorizontallyAligned(page);

  const mainBox = await page.locator(".model-panel").boundingBox();
  const trainBox = await page.locator("#trainPanel").boundingBox();
  const viewport = page.viewportSize();
  expect((mainBox?.y ?? 0) + (mainBox?.height ?? 0)).toBeLessThanOrEqual(viewport!.height + 2);
  expect((trainBox?.y ?? 0) + (trainBox?.height ?? 0)).toBeLessThanOrEqual(viewport!.height + 2);

  await page.locator("#problemSelect").selectOption("burgers");
  await expect(page.locator("#runMeta")).toHaveText(/Burgers · Well-Trained · 4 candidate · 5 train/);
  await expectNonblankCanvas(page, "#mainCanvas");
  const burgersFrame = await plotFrameMetrics(page, "#modelPanel", "#mainSvg");
  expect(burgersFrame.frameWidth).toBeGreaterThan(310);
  expect(burgersFrame.frameWidth / burgersFrame.bodyWidth).toBeGreaterThan(0.84);
  expect(burgersFrame.colorbars).toBe(1);

  await page.locator("#problemSelect").selectOption("navier_stokes_nd");
  await expect(page.locator("#runMeta")).toHaveText(/Navier Stokes · Well-Trained · 4 candidate · 5 train/);
  await expectNonblankCanvas(page, "#mainCanvas");
  const navierFrame = await plotFrameMetrics(page, "#modelPanel", "#mainSvg");
  expect(navierFrame.frameWidth).toBeGreaterThan(310);
  expect(navierFrame.frameWidth / navierFrame.bodyWidth).toBeGreaterThan(0.84);
  expect(navierFrame.marginLeft + navierFrame.marginRight).toBeLessThan(60);
  expect(navierFrame.colorbars).toBe(1);

  await page.locator("#problemSelect").selectOption("fixture");
  await expect(page.locator("#runMeta")).toHaveText(/Fixture · Well-Trained · 4 candidate · 5 train/);
  await expectNonblankCanvas(page, "#mainCanvas");
  await expectNonblankCanvas(page, "#trainCanvas");

  await tapMainPoint(page);
  await expect(page.locator("#selectedPoint")).toHaveText(/\(.+,.+\)/);
  await expectReadoutFits(page);
  const contextMenuCancelled = await page
    .locator("#mainCanvas")
    .evaluate(
      (canvas) =>
        !canvas.dispatchEvent(new MouseEvent("contextmenu", { bubbles: true, cancelable: true })),
    );
  expect(contextMenuCancelled).toBe(true);

  await touchDragMainRegion(page);
  await expect(page.locator("#selectedPoint")).toHaveText(/x\[.+,.+\] y\[.+,.+\]/);
  await expectReadoutFits(page);
  await expect(page.locator("#trainRange")).toHaveText(/Local region · Points · average over [1-4] candidates · mean I -?(?:\d|\.)/);
  await expectTrainSummaryBadgeFits(page);
  await expectNonblankCanvas(page, "#trainCanvas");
});

test("polish states clear loading and update range progress", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "desktop-only state coverage is enough");
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 390, height: 900 });
  await page.goto(FIXTURE_URL);

  await expectNonblankCanvas(page, "#mainCanvas");
  await expectNonblankCanvas(page, "#trainCanvas");
  await expect(page.locator("#modelPanel")).toHaveAttribute("aria-busy", "false");
  await expect(page.locator("#trainPanel")).toHaveAttribute("aria-busy", "false");
  await expect(page.locator("#modelPanel")).toHaveAttribute("data-loading", "false");
  await expect(page.locator("#trainPanel")).toHaveAttribute("data-loading", "false");
  await expect(page.locator(".menu-button")).toHaveCount(0);
  await expect(page.locator(".plot-menu")).toHaveCount(0);
  await expectPlotControlsStayInline(page);
  await expectReadoutFits(page);

  await page.locator("#kSlider").fill("128");
  await expect(page.locator("#kOutput")).toHaveText("128");
  await expect
    .poll(() =>
      page.locator("#kSlider").evaluate((slider) =>
        (slider as HTMLElement).style.getPropertyValue("--range-progress"),
      ),
    )
    .toBe("50%");
});

test("settings stay inline and label-free in plot tiles", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "desktop-only viewport assertions");
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.goto(FIXTURE_URL);

  await expectTopbarControlsFit(page);
  await expect(page.locator(".menu-button")).toHaveCount(0);
  await expect(page.locator(".plot-menu")).toHaveCount(0);
  await expect(page.locator(".model-actions")).not.toContainText("Field");
  await expect(page.locator(".train-actions")).not.toContainText(/Influence|Sign|Top k|Background/);
  await expect(page.locator("#fieldSelect")).toHaveAttribute("aria-label", "Field");
  await expect(page.locator("#matrixSelect")).toHaveAttribute("aria-label", "Influence");
  await expect(page.locator("#signButtons")).toHaveAttribute("aria-label", "Influence sign");
  await expect(page.locator("#kSlider")).toHaveAttribute("aria-label", "Top k");
  await expect(page.locator("#backgroundButtons")).toHaveAttribute("aria-label", "Influence background");
  await expectVisibleControlsInsidePanels(page);
  await expectPlotControlsStayInline(page);
  await expectReadoutFits(page);

  await page.setViewportSize({ width: 390, height: 900 });
  await expectTopbarControlsFit(page);
  await expectVisibleControlsInsidePanels(page);
  await expectPlotControlsStayInline(page);
  await expectReadoutFits(page);
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
