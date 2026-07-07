import { renderToString } from "katex";

const KATEX_OPTIONS = {
  displayMode: false,
  output: "html",
  strict: "ignore",
  throwOnError: false,
  trust: true,
} as const;

const CORE_FORMULA_LATEX = String.raw`
  \htmlData{explainer-token=score}{\operatorname{Inf}}_{\theta_0}^{\htmlData{explainer-token=loss}{L}\to\htmlData{explainer-token=quantity}{f}}
  \left(\htmlData{explainer-token=train-point}{x},\htmlData{explainer-token=test-point}{z}\right)
  =
  \htmlData{explainer-token=sign}{-}
  \htmlData{explainer-token=grad-f}{\nabla_{\theta} f(z;\theta_0)}^{\htmlClass{explainer-transpose}{\top}}
  \htmlData{explainer-token=hessian}{H_{\theta_0}^{-1}}
  \htmlData{explainer-token=grad-loss}{\nabla_{\theta} L(x;\theta_0)}
`;

const MOBILE_FORMULA_LHS_LATEX = String.raw`
  \htmlData{explainer-token=score}{\operatorname{Inf}}_{\theta_0}^{\htmlData{explainer-token=loss}{L}\to\htmlData{explainer-token=quantity}{f}}
  \left(\htmlData{explainer-token=train-point}{x},\htmlData{explainer-token=test-point}{z}\right)
  =
`;

const MOBILE_FORMULA_RHS_LATEX = String.raw`
  \htmlData{explainer-token=sign}{-}
  \htmlData{explainer-token=grad-f}{\nabla_{\theta} f(z;\theta_0)}^{\htmlClass{explainer-transpose}{\top}}
  \htmlData{explainer-token=hessian}{H_{\theta_0}^{-1}}
  \htmlData{explainer-token=grad-loss}{\nabla_{\theta} L(x;\theta_0)}
`;

const MATH_X = inlineMath("x", "<mi>x</mi>");
const MATH_Z = inlineMath("z", "<mi>z</mi>");
const MATH_F = inlineMath("f", "<mi>f</mi>");
const MATH_L = inlineMath("L", "<mi>L</mi>");
const MATH_H_INV = inlineMath(
  "H inverse",
  "<msup><mi>H</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup>",
);
const MATH_GRAD_LOSS = inlineMath(
  "gradient theta L of x at theta zero",
  `<mrow>
    <msub><mo>∇</mo><mi>θ</mi></msub>
    <mi>L</mi><mo>(</mo><mi>x</mi><mo>;</mo><msub><mi>θ</mi><mn>0</mn></msub><mo>)</mo>
  </mrow>`,
);
const MATH_ABS_INF_I = inlineLatex("absolute influence Inf i", "\\left|\\operatorname{Inf}_i\\right|");
const MATH_SUM_ABS_INF_J = inlineLatex(
  "sum over j of absolute influence Inf j",
  "\\sum_j \\left|\\operatorname{Inf}_j\\right|",
);
const MATH_KAPPA = inlineMath("kappa", "<mi>κ</mi>");

export const EXPLAINER_TOKENS = [
  {
    id: "score",
    title: "Influence Score",
    body:
      "The score estimates how much a training point x changes the quantity f at location z when that point is removed. Large absolute values mean the trained PINN is locally sensitive to that training point.",
    bodyHtml: `The score estimates how much a training point ${MATH_X} changes the quantity ${MATH_F} at location ${MATH_Z} when that point is removed. Large absolute values mean the trained PINN is locally sensitive to that training point.`,
  },
  {
    id: "loss",
    title: "Training Loss L",
    body:
      "L is the training objective. In a PINN it is a composite loss made from the PDE residual, initial-condition terms, boundary-condition terms, and any extra measurement terms.",
    bodyHtml: `${MATH_L} is the training objective. In a PINN it is a composite loss made from the PDE residual, initial-condition terms, boundary-condition terms, and any extra measurement terms.`,
  },
  {
    id: "quantity",
    title: "Quantity f",
    body:
      "f is what we inspect after training. In the demo this is usually a prediction field or a loss value at a candidate location.",
    bodyHtml: `${MATH_F} is what we inspect after training. In the demo this is usually a prediction field or a loss value at a candidate location.`,
  },
  {
    id: "train-point",
    title: "Training Point x",
    body:
      "x is a collocation, initial-condition, or boundary-condition point that participated in training. PINNfluence asks how the model would respond if this point were removed.",
    bodyHtml: `${MATH_X} is a collocation, initial-condition, or boundary-condition point that participated in training. PINNfluence asks how the model would respond if this point were removed.`,
  },
  {
    id: "test-point",
    title: "Evaluation Point z",
    body:
      "z is the candidate points at which we observe the model (which we can sample at will). PINNfluence approximates how f(z) changes upon removal of x. Fixing z produces an attribution map over how training points influence f(z).",
    bodyHtml: `${MATH_Z} is the candidate points at which we observe the model (which we can sample at will). PINNfluence approximates how ${MATH_F}(${MATH_Z}) changes upon removal of ${MATH_X}. Fixing ${MATH_Z} produces an attribution map over how training points influence ${MATH_F}(${MATH_Z}).`,
  },
  {
    id: "sign",
    title: "Sign Convention",
    body:
      "Enforces sign convention: A positive score means this training point pushes f(z) upward — removing it would decrease f(z).",
    bodyHtml: `Enforces sign convention: A positive score means this training point pushes ${MATH_F}(${MATH_Z}) upward — removing it would decrease ${MATH_F}(${MATH_Z}).`,
  },
  {
    id: "grad-f",
    title: "Sensitivity of f",
    body:
      "After H⁻¹ turns the loss push into a parameter shift, this left term projects that shift into the final change of f at z.",
    bodyHtml: `After ${MATH_H_INV} turns the loss push into a parameter shift, this left term projects that shift into the final change of ${MATH_F} at ${MATH_Z}.`,
  },
  {
    id: "hessian",
    title: "Local Training Geometry",
    body:
      "The Hessian maps parameter-space movement to loss-gradient change. PINNfluence needs the reverse: given the tiny loss push from x, what parameter shift would the trained landscape allow? H⁻¹ applies that reverse map; steep directions shrink, flat directions carry more.",
    bodyHtml: `The Hessian maps parameter-space movement to loss-gradient change. PINNfluence needs the reverse: given the tiny loss push from ${MATH_X}, what parameter shift would the trained landscape allow? ${MATH_H_INV} applies that reverse map; steep directions shrink, flat directions carry more.`,
  },
  {
    id: "grad-loss",
    title: "Training-Point Gradient",
    body:
      "This gradient describes how the selected training point pulls on the model parameters through its loss contribution.",
    bodyHtml: `The gradient ${MATH_GRAD_LOSS} describes how the selected training point pulls on the model parameters through its loss contribution.`,
  },
  {
    id: "loss-fraction",
    title: "Loss-Term Fraction",
    body:
      "A term fraction measures how much of the absolute influence comes from one loss component, such as the PDE, IC, or BC term.",
    bodyHtml: `A term fraction uses ${MATH_ABS_INF_I} to measure how much of the absolute influence comes from one loss component, such as the PDE, IC, or BC term.`,
  },
  {
    id: "loss-normalizer",
    title: "All Loss Terms",
    body:
      "The denominator sums absolute influence over all loss components. This makes the fractions comparable and keeps them between 0 and 1.",
    bodyHtml: `The denominator ${MATH_SUM_ABS_INF_J} sums absolute influence over all loss components. This makes the fractions comparable and keeps them between 0 and 1.`,
  },
  {
    id: "cancellation",
    title: "Cancellation κ",
    body:
      "Cancellation rises when signed loss-term influences oppose each other. High cancellation means the fractions should be read together with the signed effects.",
    bodyHtml: `Cancellation ${MATH_KAPPA} rises when signed loss-term influences oppose each other. High cancellation means the fractions should be read together with the signed effects.`,
  },
] as const;

export type ExplainerToken = (typeof EXPLAINER_TOKENS)[number];
export type ExplainerTokenId = ExplainerToken["id"];

const DEFAULT_TOKEN_ID: ExplainerTokenId = "score";
const TOKEN_BY_ID: ReadonlyMap<string, ExplainerToken> = new Map(
  EXPLAINER_TOKENS.map((token) => [token.id, token]),
);
const READ_DIRECTION_HINT_VISIBLE_MS = 2500;
const READ_DIRECTION_HINT_HIDE_MS = 440;

export class ExplainerView {
  private activeTokenId: ExplainerTokenId = DEFAULT_TOKEN_ID;
  private readDirectionHintShown = false;
  private readDirectionHintAutoTimer = 0;
  private readDirectionHintHideTimer = 0;
  private readDirectionHintShowFrame = 0;

  constructor(private readonly root: HTMLElement) {
    this.render();
    this.root.addEventListener("click", this.handleTokenEvent);
    this.root.addEventListener("focusin", this.handleTokenEvent);
    this.root.addEventListener("keydown", this.handleTokenKeydown);
    this.root.addEventListener("pointerover", this.handleTokenEvent);
  }

  activate(): void {
    this.setActiveToken(this.activeTokenId);
  }

  private readonly handleTokenEvent = (event: Event): void => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    const tokenElement = target.closest<HTMLElement>("[data-explainer-token]");
    if (!tokenElement || !this.root.contains(tokenElement)) return;
    if (
      (event.type === "pointerover" || event.type === "focusin") &&
      tokenElement.closest(".explainer-formula")
    ) {
      this.maybeShowReadDirectionHint();
    }
    const id = tokenElement.dataset.explainerToken;
    if (id) this.setActiveToken(id);
  };

  private readonly handleTokenKeydown = (event: KeyboardEvent): void => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    const tokenElement = target.closest<HTMLElement>("[data-explainer-token]");
    if (
      !tokenElement ||
      !this.root.contains(tokenElement) ||
      tokenElement instanceof HTMLButtonElement ||
      (event.key !== "Enter" && event.key !== " ")
    ) {
      return;
    }
    event.preventDefault();
    const id = tokenElement.dataset.explainerToken;
    if (id) this.setActiveToken(id);
  };

  private render(): void {
    this.root.innerHTML = `
      <div class="explainer-shell">
        <section class="explainer-panel explainer-hero" aria-labelledby="explainer-title">
          <div class="explainer-hero-copy">
            <p class="explainer-eyebrow">PINNfluence in one pass</p>
            <h2 id="explainer-title">How one training point shapes a PINN prediction</h2>
            <p>
              A physics-informed neural network learns a solution by minimizing physics, initial-condition,
              and boundary-condition losses. PINNfluence estimates how a trained model would react if one
              training point were removed, without running a full retraining experiment.
            </p>
          </div>
          <div class="explainer-domain" role="img" aria-label="PINN domain with PDE, initial-condition, and boundary-condition training points">
            <div class="explainer-domain-stage">
              <span class="explainer-domain-band explainer-domain-band-ic">IC</span>
              <span class="explainer-domain-band explainer-domain-band-bc-left">BC</span>
              <span class="explainer-domain-band explainer-domain-band-bc-right">BC</span>
              ${domainPoints()}
              <span class="explainer-domain-query">z</span>
            </div>
            <div class="explainer-domain-legend">
              <span><i class="explainer-dot explainer-dot-pde"></i>PDE residual</span>
              <span><i class="explainer-dot explainer-dot-ic"></i>Initial condition</span>
              <span><i class="explainer-dot explainer-dot-bc"></i>Boundary condition</span>
            </div>
          </div>
        </section>

        <section class="explainer-flow-panel" aria-label="PINNfluence workflow">
          <ol class="explainer-flow">
            ${flowStep("1", "Train", "Fit the PINN on PDE, IC, and BC losses.")}
            ${flowStep("2", "Approximate Removal Effect", "Estimate how the trained model's output would change if one training point were removed — without retraining.")}
            ${flowStep("3", "Interpret", "Read the trained PINN through its sensitivity to individual training points: attribution maps, loss-term fractions, and aggregate regional/temporal indicators.")}
          </ol>
        </section>

        <section class="explainer-panel explainer-formula-panel" aria-labelledby="explainer-formula-title">
          <div class="explainer-panel-header">
            <div>
              <p class="explainer-eyebrow">Core estimate</p>
              <h2 id="explainer-formula-title">Influence as local sensitivity</h2>
            </div>
            <div class="explainer-read-direction" aria-label="Formula reading order">
              <span class="explainer-read-direction-chain" aria-hidden="true">
                <span class="explainer-read-direction-chip explainer-read-direction-chip-result">change in f</span>
                <span class="explainer-read-direction-pair">
                  <i class="explainer-read-direction-arrow">←</i>
                  <span class="explainer-read-direction-chip">landscape response</span>
                </span>
                <span class="explainer-read-direction-pair">
                  <i class="explainer-read-direction-arrow">←</i>
                  <span class="explainer-read-direction-chip">training-point push</span>
                </span>
              </span>
              <span class="sr-only">change in f from landscape response from training-point push</span>
            </div>
          </div>
          <div class="explainer-formula-workbench">
            <div class="explainer-formula-scroll" aria-label="PINNfluence influence function">
              <div
                id="explainerReadDirectionHint"
                class="model-interaction-hint explainer-read-direction-tooltip"
                data-state="hidden"
                aria-hidden="true"
                hidden
              >Read the product from right to left.</div>
              <div class="explainer-formula explainer-formula-desktop" aria-describedby="explainer-token-body">
                ${latexMarkup(CORE_FORMULA_LATEX)}
              </div>
              <div class="explainer-formula explainer-formula-mobile" aria-describedby="explainer-token-body">
                <span class="explainer-formula-mobile-line explainer-formula-mobile-lhs">
                  ${latexMarkup(MOBILE_FORMULA_LHS_LATEX)}
                </span>
                <span class="explainer-formula-mobile-line explainer-formula-mobile-rhs">
                  ${latexMarkup(MOBILE_FORMULA_RHS_LATEX)}
                </span>
              </div>
            </div>
            <aside class="explainer-token-detail" aria-live="polite">
              <p class="explainer-detail-kicker">Formula component</p>
              <h3 id="explainer-token-title"></h3>
              <p id="explainer-token-body"></p>
            </aside>
          </div>
        </section>

        <section class="explainer-panel explainer-split-panel" aria-labelledby="explainer-split-title">
          <div class="explainer-panel-header">
            <div>
              <p class="explainer-eyebrow">Composite PINN losses</p>
              <h2 id="explainer-split-title">Split influence by PDE, IC, and BC terms</h2>
            </div>
            <p>
              Because influence is linear in the loss term, the same score can be decomposed into
              the constraints that shaped it.
            </p>
          </div>
          <div class="explainer-split-content">
            <div class="explainer-mini-formula">
              <span>r<sub>Li</sub></span>
              <span>=</span>
              <span class="explainer-fraction">
                <span>${latexToken("loss-fraction", "\\left|\\operatorname{Inf}_i\\right|", "", "|Inf_i|")}</span>
                <span>${latexToken("loss-normalizer", "\\sum_j \\left|\\operatorname{Inf}_j\\right|", "", "sum_j |Inf_j|")}</span>
              </span>
            </div>
            <div class="explainer-loss-bars" aria-label="Example loss-term influence fractions">
              ${lossBar("PDE", 58, "pde")}
              ${lossBar("IC", 24, "ic")}
              ${lossBar("BC", 18, "bc")}
              <button type="button" class="explainer-cancellation" data-explainer-token="cancellation">
                κ cancellation
              </button>
            </div>
          </div>
        </section>

        <section class="explainer-diagnostics" aria-label="PINNfluence diagnostics">
          ${diagnosticTile("Point Map", "Fix z and color training points by |I(x,z)| to locate influential samples.")}
          ${diagnosticTile("Loss Split", "Compare PDE, IC, and BC shares to see which constraints dominate.")}
          ${diagnosticTile("Region Aggregate", "Sum over sets of points to compare how one domain region shapes another.")}
          ${diagnosticTile("Temporal or Directional Influence Indicator", "Compress temporal or spatial directionality into a comparable diagnostic number.")}
        </section>

        <section class="explainer-caveat" aria-label="Interpretation caveats">
          <strong>Read as sensitivity.</strong>
          <span>
            PINNfluence is a first-order, local approximation around the trained parameters. Large influence
            identifies sensitivity to a training point or constraint, not a formal causal guarantee. It also
            applies only to the equilibrium the model found. Retraining without the training point in question
            could push the model into an entirely different equilibrium.
          </span>
        </section>
      </div>
    `;
    this.initializeFormulaTokens();
    this.setActiveToken(this.activeTokenId);
  }

  private initializeFormulaTokens(): void {
    this.root.querySelectorAll<HTMLElement>("[data-explainer-token]").forEach((element) => {
      const token = TOKEN_BY_ID.get(element.dataset.explainerToken ?? "");
      if (!(element instanceof HTMLButtonElement)) {
        element.setAttribute("role", "button");
        element.tabIndex = 0;
      }
      if (token) element.setAttribute("aria-label", token.title);
      element.setAttribute("aria-pressed", "false");
    });
  }

  private setActiveToken(id: string): void {
    const token = TOKEN_BY_ID.get(id);
    if (!token) return;
    this.activeTokenId = token.id;
    this.root.querySelectorAll<HTMLElement>("[data-explainer-token]").forEach((element) => {
      const active = element.dataset.explainerToken === token.id;
      element.classList.toggle("active", active);
      element.setAttribute("aria-pressed", String(active));
    });
    const title = this.root.querySelector<HTMLElement>("#explainer-token-title");
    const body = this.root.querySelector<HTMLElement>("#explainer-token-body");
    if (title) title.textContent = token.title;
    if (body) body.innerHTML = token.bodyHtml;
  }

  private maybeShowReadDirectionHint(): void {
    if (this.readDirectionHintShown) return;
    this.readDirectionHintShown = true;
    this.showReadDirectionHint();
  }

  private showReadDirectionHint(): void {
    const hint = this.root.querySelector<HTMLElement>("#explainerReadDirectionHint");
    if (!hint) return;
    this.clearReadDirectionHintTimers();
    hint.hidden = false;
    hint.setAttribute("aria-hidden", "false");
    hint.dataset.state = "hidden";

    this.readDirectionHintShowFrame = requestAnimationFrame(() => {
      this.readDirectionHintShowFrame = 0;
      hint.dataset.state = "visible";
      this.readDirectionHintAutoTimer = window.setTimeout(
        () => this.dismissReadDirectionHint(),
        READ_DIRECTION_HINT_VISIBLE_MS,
      );
    });
  }

  private dismissReadDirectionHint(): void {
    const hint = this.root.querySelector<HTMLElement>("#explainerReadDirectionHint");
    if (!hint || (hint.hidden && !this.readDirectionHintShowFrame)) return;
    this.clearReadDirectionHintTimers();
    hint.dataset.state = "hidden";
    hint.setAttribute("aria-hidden", "true");
    this.readDirectionHintHideTimer = window.setTimeout(() => {
      this.readDirectionHintHideTimer = 0;
      hint.hidden = true;
    }, READ_DIRECTION_HINT_HIDE_MS);
  }

  private clearReadDirectionHintTimers(): void {
    if (this.readDirectionHintShowFrame) {
      cancelAnimationFrame(this.readDirectionHintShowFrame);
      this.readDirectionHintShowFrame = 0;
    }
    if (this.readDirectionHintAutoTimer) {
      window.clearTimeout(this.readDirectionHintAutoTimer);
      this.readDirectionHintAutoTimer = 0;
    }
    if (this.readDirectionHintHideTimer) {
      window.clearTimeout(this.readDirectionHintHideTimer);
      this.readDirectionHintHideTimer = 0;
    }
  }
}

function inlineMath(label: string, markup: string): string {
  return `<math class="explainer-inline-math" aria-label="${escapeHtml(label)}">${markup}</math>`;
}

function inlineLatex(label: string, latex: string): string {
  return `<span class="explainer-inline-math" aria-label="${escapeHtml(label)}">${latexMarkup(latex)}</span>`;
}

function latexToken(
  id: ExplainerTokenId,
  latex: string,
  variant = "",
  fallbackLabel = latex,
): string {
  return formulaTokenMarkup(id, latexMarkup(latex), variant, fallbackLabel);
}

function latexMarkup(latex: string): string {
  return renderToString(latex, KATEX_OPTIONS);
}

function formulaTokenMarkup(id: ExplainerTokenId, html: string, variant = "", fallbackLabel = ""): string {
  const token = TOKEN_BY_ID.get(id);
  const classes = [
    "explainer-formula-token",
    ...variant
      .split(/\s+/)
      .filter(Boolean)
      .map((name) => `explainer-formula-token-${name}`),
  ].join(" ");
  return `
    <button
      type="button"
      class="${classes}"
      data-explainer-token="${escapeHtml(id)}"
      aria-label="${escapeHtml(token?.title ?? fallbackLabel)}"
      aria-pressed="false"
    >${html}</button>
  `;
}

function flowStep(index: string, title: string, body: string): string {
  return `
    <li>
      <span class="explainer-flow-index">${escapeHtml(index)}</span>
      <h3>${escapeHtml(title)}</h3>
      <p>${escapeHtml(body)}</p>
    </li>
  `;
}

function diagnosticTile(title: string, body: string): string {
  return `
    <article class="explainer-diagnostic">
      <h3>${escapeHtml(title)}</h3>
      <p>${escapeHtml(body)}</p>
    </article>
  `;
}

function lossBar(label: string, value: number, kind: string): string {
  return `
    <div class="explainer-loss-row">
      <span>${escapeHtml(label)}</span>
      <span class="explainer-loss-track">
        <span class="explainer-loss-fill explainer-loss-fill-${escapeHtml(kind)}" style="--value: ${value}%"></span>
      </span>
      <span>${value}%</span>
    </div>
  `;
}

function domainPoints(): string {
  const points = [
    ["pde", 20, 28],
    ["pde", 38, 38],
    ["pde", 58, 26],
    ["pde", 74, 48],
    ["pde", 30, 62],
    ["pde", 52, 70],
    ["pde", 68, 64],
    ["ic", 18, 86],
    ["ic", 38, 86],
    ["ic", 58, 86],
    ["ic", 78, 86],
    ["bc", 9, 30],
    ["bc", 9, 58],
    ["bc", 91, 34],
    ["bc", 91, 66],
  ] as const;
  return points
    .map(
      ([kind, x, y]) =>
        `<span class="explainer-domain-point explainer-domain-point-${kind}" style="--x: ${x}%; --y: ${y}%"></span>`,
    )
    .join("");
}

function escapeHtml(value: unknown): string {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
